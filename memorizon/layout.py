"""Sequence layout, temporal RoPE indices and attention mask.

One forward pass sees a fixed-size sequence, however long the video is::

    [ A | memory bank | recent | target chunk(s) ]

``A`` is the first frame, the bank holds retrieved past frames, ``recent`` is the
chunk right before the targets, and each target chunk has ``chunk`` latents.
Training packs ``n_query_chunks`` target chunks into one sequence; inference
generates one chunk at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .retrieval import Frustum, topk

BANK_ROPE_INDEX = 1   # every bank slot shares one temporal index: the bank is a set


@dataclass(frozen=True)
class Layout:
    n_bank: int
    n_recent: int = 4
    chunk: int = 4
    n_query_chunks: int = 1

    @property
    def n_cond(self) -> int:
        """Clean context slots (A + bank + recent); no loss is computed on them."""
        return 1 + self.n_bank + self.n_recent

    @property
    def total(self) -> int:
        return self.n_cond + self.chunk * self.n_query_chunks

    @property
    def bank(self) -> slice:
        return slice(1, 1 + self.n_bank)

    @property
    def recent(self) -> slice:
        return slice(1 + self.n_bank, self.n_cond)

    @property
    def target(self) -> slice:
        return slice(self.n_cond, self.total)

    def chunk_slice(self, j: int) -> slice:
        s = self.n_cond + j * self.chunk
        return slice(s, s + self.chunk)


def rope_index(layout: Layout) -> np.ndarray:
    """``A=0 | bank=1 ... 1 | recent, targets = 2, 3, ...`` (plain ``0..F-1`` without a bank)."""
    if layout.n_bank == 0:
        return np.arange(layout.total, dtype=np.int64)
    idx = np.empty(layout.total, dtype=np.int64)
    idx[0] = 0
    idx[layout.bank] = BANK_ROPE_INDEX
    idx[layout.recent.start:] = np.arange(BANK_ROPE_INDEX + 1,
                                          BANK_ROPE_INDEX + 1 + layout.total - layout.recent.start)
    return idx


def attention_mask(layout: Layout, bank_idx: np.ndarray | None = None, q0: int | None = None,
                   query_c2w: np.ndarray | None = None, bank_c2w: np.ndarray | None = None,
                   recent_c2w: np.ndarray | None = None, frustum: Frustum | None = None,
                   k: int = 6) -> np.ndarray:
    """``[F, F]`` bool mask, row = query frame.

    Context: the bank sees A and the bank, recent sees A, the bank and itself.
    Target chunk ``j`` sees A, itself, its predecessor (``recent`` for ``j = 0``) and
    its own top-``k`` memory, chosen by camera pose from one pool: bank frames taken
    before chunk ``j - 1`` (``bank_idx < q0 + (j - 1) * chunk``), the recent frames
    (for ``j >= 1``, when ``recent_c2w`` is given) and target chunks ``0 .. j - 2``.
    ``q0`` is the frame index of the first target frame; ``query_c2w`` holds the
    target cameras. Without ``bank_idx`` every chunk sees the whole bank.
    """
    F, C = layout.total, layout.chunk
    A, bank, rec, t0 = slice(0, 1), layout.bank, layout.recent, layout.target.start
    m = np.zeros((F, F), dtype=bool)
    m[A, A] = True
    m[bank, A] = m[bank, bank] = True
    m[rec, A] = m[rec, bank] = m[rec, rec] = True

    for j in range(layout.n_query_chunks):
        rows = layout.chunk_slice(j)
        m[rows, A] = m[rows, rows] = True
        if j == 0:
            m[rows, rec] = True
        else:
            m[rows, layout.chunk_slice(j - 1)] = True
        if bank_idx is None:
            m[rows, bank] = True
            continue

        pool, where = [], []
        if layout.n_bank:
            ok = np.flatnonzero(np.asarray(bank_idx) < q0 + max(j - 1, 0) * C)
            if len(ok):
                pool.append(bank_c2w[ok])
                where += [bank.start + int(i) for i in ok]
        if recent_c2w is not None and j >= 1 and layout.n_recent:
            pool.append(np.asarray(recent_c2w)[:layout.n_recent])
            where += [rec.start + i for i in range(layout.n_recent)]
        n_in = max(j - 1, 0) * C
        if n_in:
            pool.append(query_c2w[:n_in])
            where += [t0 + i for i in range(n_in)]
        if not pool:
            continue
        pool = np.concatenate(pool, axis=0)
        for t in topk(query_c2w[rows.stop - t0 - 1], pool, frustum, min(k, len(pool))):
            m[rows, where[t]] = True
    return m
