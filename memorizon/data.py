"""Long-span training samples.

A sample covers a random span of an episode, ``1 + (m + 1) * chunk`` latents with
``m`` drawn per sample (10 s to 100 s at the default settings). Only the last
``n_query_chunks`` chunks are trained on; everything earlier reaches the model
through the memory bank, so every sample packs into the same short sequence::

    [ A | bank (retrieved from the span's history) | recent chunk | 10 target chunks ]

Shard layout under ``dataset_path`` (latents are Wan2.2-VAE latents, normalised
with the VAE's ``latents_mean``/``latents_std``; poses are camera-to-world with
translations in metres / ``traj_scale``, one pose per latent frame)::

    index.json                {"episode_ids": [...], "frame_count": [...]}
    latents/<ep>.pt           {"latents": [48, T, h, w], "c2w_latent": [T, 4, 4],
                               "intrinsics": [4]  (fx, fy, cx, cy)}
    cond/<ep>.pt              {"cond_latents": [48, n, h, w], "cond_starts": [n],
                               "cond_stride": s}   single-frame encodes of frame
                               ``cond_starts[i]``, used as the first frame A
    prompts_flat/<ep>.emb.npy  umT5 caption embeddings (bf16 bits as uint16), all
                  <ep>.off.npy   rows concatenated; caption i is rows off[i]:off[i+1]
                  <ep>.starts.npy and belongs to latent ``starts[i]`` (= cond_starts)
"""

from __future__ import annotations

import json
import logging
import os
import random

import numpy as np
import torch

from .camera import invert_se3
from .layout import Layout, attention_mask, rope_index
from .retrieval import Frustum, select_bank

logger = logging.getLogger(__name__)


class LongSpanDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, train_episodes: str | list[str] | None = None,
                 chunk: int = 4, n_query_chunks: int = 10, m_min: int = 9, m_max: int = 99,
                 topk_per_chunk: int = 6, max_bank: int = 60, traj_scale: float = 4.0,
                 prompt_dropout: float = 0.0, null_prompt_path: str | None = None,
                 text_max_len: int = 512, dtype: torch.dtype | str | None = torch.bfloat16):
        self.root = dataset_path
        with open(os.path.join(dataset_path, "index.json")) as f:
            index = json.load(f)
        episodes, counts = index["episode_ids"], index.get("frame_count")
        if train_episodes is not None:
            if isinstance(train_episodes, str):
                with open(train_episodes) as f:
                    train_episodes = json.load(f)["episodes"]
            missing = set(train_episodes) - set(episodes)
            if missing:
                raise ValueError(f"{len(missing)} listed episodes are not in {dataset_path}")
            keep = set(train_episodes)
            episodes = [e for e in episodes if e in keep]
        need = 1 + (max(m_min, n_query_chunks - 1) + 1) * chunk
        if counts:
            length = dict(zip(index["episode_ids"], counts))
            episodes = [e for e in episodes if length.get(e, need) >= need]
        self.episode_ids = episodes

        self.chunk, self.n_query_chunks = chunk, n_query_chunks
        self.m_min, self.m_max = m_min, m_max
        self.topk, self.max_bank = topk_per_chunk, max_bank
        self.traj_scale = traj_scale
        self.prompt_dropout, self.text_max_len = prompt_dropout, text_max_len
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.null_prompt = None
        if prompt_dropout > 0:
            if null_prompt_path is None:
                raise ValueError("prompt_dropout > 0 needs null_prompt_path (the empty-prompt "
                                 "embedding, see `python -m memorizon.text`)")
            self.null_prompt = torch.load(null_prompt_path, map_location="cpu", weights_only=True)[0]
        self._frustums: dict[tuple, Frustum] = {}
        logger.info("LongSpanDataset: %d episodes, spans of %d..%d latents", len(self.episode_ids),
                    1 + (max(m_min, n_query_chunks - 1) + 1) * chunk, 1 + (m_max + 1) * chunk)

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, index: int) -> dict:
        try:
            return self.getitem(index)
        except Exception:
            logger.exception("failed to load %s", self.episode_ids[index])
            return self.getitem(random.randrange(len(self.episode_ids)))

    def _frustum(self, intr: torch.Tensor) -> Frustum:
        key = tuple(float(x) for x in intr)
        if key not in self._frustums:
            self._frustums[key] = Frustum.from_intrinsics(key, self.traj_scale)
        return self._frustums[key]

    def _draw_span(self, T: int, stride: int, start_max: int, rng: random.Random):
        """Span start ``s`` (on the cond grid), then history / recent / target frame indices."""
        C, nq = self.chunk, self.n_query_chunks
        lo, hi = max(self.m_min, nq - 1), min(self.m_max, (T - 1) // C - 1)
        if hi < lo:
            raise ValueError(f"episode of {T} latents is too short")
        m = rng.randint(lo, hi)
        s = rng.randrange(0, min(T - (1 + (m + 1) * C), start_max) // stride + 1) * stride
        chunk_idx = lambda j: np.arange(s + 1 + j * C, s + 1 + (j + 1) * C)  # noqa: E731
        target = np.concatenate([chunk_idx(j) for j in range(m - nq + 1, m + 1)])
        recent = chunk_idx(m - nq) if m >= nq else np.zeros(0, dtype=int)
        end = int(recent[0]) if len(recent) else int(target[0])
        history = np.arange(s + 1, end) if end > s + 1 else np.zeros(0, dtype=int)
        return s, history, recent, target

    def getitem(self, index: int) -> dict:
        ep = self.episode_ids[index]
        rng = random.Random(random.getrandbits(63))
        d = torch.load(os.path.join(self.root, "latents", f"{ep}.pt"), weights_only=False,
                       map_location="cpu")
        full = d["latents"]
        c2w_ep = torch.as_tensor(d["c2w_latent"], dtype=torch.float32)
        intr = torch.as_tensor(d["intrinsics"], dtype=torch.float32)
        frustum = self._frustum(intr)
        cond = torch.load(os.path.join(self.root, "cond", f"{ep}.pt"), weights_only=False,
                          map_location="cpu", mmap=True)
        stride = int(cond["cond_stride"])

        a, history, recent, target = self._draw_span(full.shape[1], stride,
                                                     int(cond["cond_starts"][-1]), rng)
        C, q0 = self.chunk, int(target[0])
        c2w_np = c2w_ep.numpy().astype(np.float64)
        bank = select_bank(c2w_np, history, c2w_np[target[C - 1::C]], frustum, k=self.topk,
                           q0=q0, chunk=C, cap=self.max_bank)
        lay = Layout(n_bank=len(bank), n_recent=len(recent), chunk=C,
                     n_query_chunks=self.n_query_chunks)
        idx = np.concatenate([[a], bank, recent, target]).astype(int)

        # A is the single-frame encode of frame a, as at inference from one image
        latents = torch.as_tensor(full[:, idx], dtype=self.dtype)
        ci = a // stride
        assert int(cond["cond_starts"][ci]) == a
        latents[:, 0] = torch.as_tensor(cond["cond_latents"][:, ci], dtype=self.dtype)
        # cameras relative to the first target frame
        c2w = c2w_ep[idx].clone()
        c2w = invert_se3(c2w[lay.n_cond:lay.n_cond + 1]) @ c2w
        c2w_np = c2w.numpy().astype(np.float64)

        return {
            "latents": latents,
            "c2w": c2w,
            "intrinsics": intr,
            "text_embedding": self._caption(ep, ci, rng),
            "attention_mask": torch.from_numpy(attention_mask(
                lay, bank, q0, query_c2w=c2w_np[lay.target], bank_c2w=c2w_np[lay.bank],
                recent_c2w=c2w_np[lay.recent], frustum=frustum, k=self.topk)),
            "rope_index": torch.from_numpy(rope_index(lay)),
            "n_cond": torch.tensor(lay.n_cond, dtype=torch.long),
            "n_bank": torch.tensor(lay.n_bank, dtype=torch.long),
            "n_recent": torch.tensor(lay.n_recent, dtype=torch.long),
        }

    def _caption(self, ep: str, ci: int, rng: random.Random) -> torch.Tensor:
        """Caption embedding of the cond position ``ci``, zero-padded to ``text_max_len``."""
        prefix = os.path.join(self.root, "prompts_flat", ep)
        if rng.random() < self.prompt_dropout:
            return self.null_prompt.to(torch.float32)
        off = np.load(prefix + ".off.npy", mmap_mode="r")
        rows = np.load(prefix + ".emb.npy", mmap_mode="r")[int(off[ci]):int(off[ci + 1])]
        e = torch.from_numpy(np.ascontiguousarray(rows).view(np.int16)).view(torch.bfloat16)
        emb = torch.zeros((self.text_max_len, e.shape[-1]), dtype=torch.float32)
        emb[:min(len(e), self.text_max_len)] = e[:self.text_max_len].float()
        return emb


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
