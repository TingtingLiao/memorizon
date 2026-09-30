"""Image + camera trajectory -> long video, one chunk at a time with retrieved memory."""

from __future__ import annotations

import os
import subprocess

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .camera import camera_transforms
from .caption import caption
from .layout import Layout, attention_mask, rope_index
from .model import MemorizonModel
from .retrieval import Frustum, select_bank
from .text import TextEncoder
from .wan.transformer import frame_mask_to_block_mask

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards")

# Intrinsics (fx, fy, cx, cy) of the 864x480 training frames (90 degree horizontal FOV).
DEFAULT_INTRINSICS = (432.0, 426.66666, 432.0, 240.0)


class MemorizonPipeline:
    def __init__(self, model: MemorizonModel, device: str = "cuda"):
        from diffusers import AutoencoderKLWan

        self.device = torch.device(device)
        self.model = model.to(self.device, torch.bfloat16).eval()
        self.dtype = self.model.dtype
        base = model.config.base_model
        self.vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae").to(self.device, torch.float32).eval()
        z = self.vae.config.z_dim
        self.lat_mean = torch.tensor(self.vae.config.latents_mean).view(1, z, 1, 1, 1).to(self.device)
        self.lat_std = torch.tensor(self.vae.config.latents_std).view(1, z, 1, 1, 1).to(self.device)
        self._text_encoder = None

    @classmethod
    def from_pretrained(cls, checkpoint: str, device: str = "cuda") -> "MemorizonPipeline":
        return cls(MemorizonModel.from_pretrained(checkpoint), device)

    # ------------------------------------------------------------------ encoders
    @torch.inference_mode()
    def encode_image(self, image: str | Image.Image) -> torch.Tensor:
        """Centre-crop to the training aspect ratio, resize, VAE-encode. ``[1, 48, 1, h, w]``."""
        w, h = self.model.config.image_width, self.model.config.image_height
        im = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        iw, ih = im.size
        aspect = w / h
        if iw / ih > aspect:
            nw = int(ih * aspect)
            box = ((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih)
        else:
            nh = int(iw / aspect)
            box = (0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh)
        im = im.crop(box).resize((w, h), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0)
        x = x.permute(2, 0, 1)[None, :, None].to(self.device, torch.float32)
        z = self.vae.encode(x).latent_dist.mode()
        return ((z - self.lat_mean) / self.lat_std).to(self.dtype)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """umT5 embedding of ``prompt`` ``[1, 512, 4096]`` (the encoder stays on the CPU)."""
        if self._text_encoder is None:
            self._text_encoder = TextEncoder(self.model.config.base_model,
                                             self.model.config.text_max_sequence_length)
        return self._text_encoder(prompt)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> np.ndarray:
        """Latents ``[1, 48, T, h, w]`` -> uint8 frames ``[4T - 3, H, W, 3]``, one latent at a time."""
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

        vae = self.vae
        z = vae.post_quant_conv(latents.to(self.device, torch.float32) * self.lat_std + self.lat_mean)
        patch = getattr(vae.config, "patch_size", None)
        vae.clear_cache()
        frames = []
        for i in range(z.shape[2]):
            vae._conv_idx = [0]
            out = vae.decoder(z[:, :, i:i + 1], feat_cache=vae._feat_map, feat_idx=vae._conv_idx,
                              first_chunk=(i == 0)).to("cpu")
            if patch is not None:
                out = unpatchify(out, patch_size=patch)
            video = out[0].permute(1, 2, 3, 0).clamp(-1, 1)
            frames.append(((video + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy())
        vae.clear_cache()
        return np.concatenate(frames, axis=0)

    # ------------------------------------------------------------------ rollout
    @torch.inference_mode()
    def rollout(self, first_latent: torch.Tensor, c2w: np.ndarray, intrinsics, num_chunks: int,
                text: torch.Tensor, uncond: torch.Tensor | None = None, guidance_scale: float = 4.0,
                num_steps: int = 20, memory_size: int = 6, seed: int = 0,
                progress: bool = True) -> torch.Tensor:
        """Generate ``num_chunks`` chunks after ``first_latent``. Returns ``[1, 48, 1 + C*num_chunks, h, w]``.

        ``c2w``: ``[>= 1 + C*num_chunks, 4, 4]`` camera per latent frame (4 per second),
        in the first frame's coordinates, translations in metres / 4.
        """
        dit, C = self.model.model, self.model.config.chunk_size
        c2w = np.asarray(c2w, dtype=np.float64)
        intr = torch.as_tensor(np.asarray(intrinsics, dtype=np.float32))
        frustum = Frustum.from_intrinsics(intr.tolist())
        _, ch, _, h, w = first_latent.shape
        _, p_h, p_w = dit.config.patch_size
        tpf = (h // p_h) * (w // p_w)
        end = 1 + num_chunks * C
        assert len(c2w) >= end, f"trajectory has {len(c2w)} poses, {num_chunks} chunks need {end}"

        sched = self.model.noise_scheduler
        sched.set_timesteps(num_steps)
        timesteps = sched.timesteps.to(self.device)

        B = 2 if guidance_scale > 0 else 1
        cond_t = text.to(self.device, self.dtype)
        if B == 2:
            text = torch.cat([uncond.to(self.device, self.dtype).expand(1, -1, -1), cond_t.expand(1, -1, -1)])
        else:
            text = cond_t
        intr_b = intr[None].to(self.device).repeat(B, 1)
        gen = torch.zeros((1, ch, end, h, w), device=self.device, dtype=self.dtype)
        gen[:, :, 0] = first_latent[:, :, 0].to(self.device, self.dtype)
        rng = torch.Generator(device="cpu").manual_seed(seed)

        for ci in tqdm(range(num_chunks), disable=not progress):
            q_lo, q_hi = 1 + ci * C, 1 + (ci + 1) * C
            n_rec = min(C, q_lo)
            recent = np.arange(q_lo - n_rec, q_lo)
            # memory: the top-k past frames (never A or the recent chunk) for this chunk's camera
            bank = select_bank(c2w, np.arange(1, max(1, q_lo - n_rec)), c2w[q_hi - 1][None],
                               frustum, k=memory_size, q0=q_lo, chunk=C, cap=memory_size)
            lay = Layout(n_bank=len(bank), n_recent=n_rec, chunk=C, n_query_chunks=1)
            F, n_cond = lay.total, lay.n_cond
            rope = torch.as_tensor(rope_index(lay), dtype=torch.long, device=self.device)
            mask = (attention_mask(lay, bank, q_lo, c2w[q_lo:q_hi], c2w[bank], frustum=frustum,
                                   k=memory_size) if len(bank) else attention_mask(lay))
            mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            ctx_mask = frame_mask_to_block_mask(mask[:n_cond, :n_cond], tpf)

            idx = np.concatenate([[0], bank, recent]).astype(int)
            cond = gen[:, :, idx]
            cams = torch.as_tensor(np.concatenate([c2w[idx], c2w[q_lo:q_hi]]), dtype=torch.float32,
                                   device=self.device)[None].repeat(B, 1, 1, 1)
            x = torch.randn((1, ch, F, h, w), generator=rng).to(self.device, self.dtype)
            x, cond = x.repeat(B, 1, 1, 1, 1), cond.repeat(B, 1, 1, 1, 1)
            x[:, :, :n_cond] = cond

            # camera projections in float32, outside autocast
            cam_ctx = camera_transforms(dit, cams[:, :n_cond], intr_b)
            cam_q = camera_transforms(dit, cams[:, n_cond:], intr_b)
            kv = dit.init_kv_cache(B, F, h, w)
            with torch.autocast("cuda", torch.bfloat16):
                # the clean context is forwarded once; its keys/values stay in the cache
                dit(cond, torch.zeros((B, n_cond), device=self.device), text, rope_index=rope[:n_cond],
                    attention_mask=ctx_mask, camera=cam_ctx, kv_cache=kv, kv_cache_range=(0, n_cond))
                for layer in kv:
                    layer.pop("cross_attn_key", None)
                    layer.pop("cross_attn_value", None)
                for t in timesteps:
                    ts = t.repeat(B, C).clone()
                    pred = dit(x[:, :, n_cond:], ts, text, rope_index=rope, camera=cam_q,
                               kv_cache=kv, kv_cache_range=(0, F))
                    if B == 2:
                        pred = pred[0:1] + guidance_scale * (pred[1:2] - pred[0:1])
                        pred = torch.cat([pred, pred])
                    x[:, :, n_cond:] = sched.step_diff_noise_level(pred, ts, x[:, :, n_cond:]).to(x)
            gen[:, :, q_lo:q_hi] = x[0:1, :, n_cond:]
        return gen

    def __call__(self, image, c2w: np.ndarray, prompt: str | None = None, intrinsics=DEFAULT_INTRINSICS,
                 num_chunks: int | None = None, negative_prompt: str = NEGATIVE_PROMPT,
                 guidance_scale: float | None = None, num_steps: int | None = None,
                 memory_size: int = 6, seed: int = 0, progress: bool = True) -> np.ndarray:
        """Generate a video. Returns uint8 frames ``[T, H, W, 3]`` at 16 fps (4 per camera pose).

        ``prompt`` should follow the training caption format; if None, the image is
        captioned with the training captioner (``memorizon.caption``).
        ``num_steps`` / ``guidance_scale`` default to the checkpoint's own settings
        (20 steps with CFG 4 for the base model, 4 steps without CFG when distilled).
        """
        if prompt is None:
            prompt = caption(image, device=str(self.device))
            if progress:
                print(f"prompt: {prompt}")
        cfg = self.model.config
        num_steps = num_steps or getattr(cfg, "default_num_steps", 20)
        if guidance_scale is None:
            guidance_scale = getattr(cfg, "default_guidance_scale", 4.0)
        C = cfg.chunk_size
        if num_chunks is None:
            num_chunks = (len(c2w) - 1) // C
        uncond = self.encode_prompt(negative_prompt) if guidance_scale > 0 else None
        latents = self.rollout(self.encode_image(image), c2w, intrinsics, num_chunks,
                               self.encode_prompt(prompt), uncond, guidance_scale, num_steps,
                               memory_size, seed, progress)
        return self.decode(latents)


def save_video(frames: np.ndarray, path: str, fps: int = 16, crf: int = 14) -> None:
    """Write uint8 frames ``[T, H, W, 3]`` to an H.264 mp4 with ffmpeg."""
    t, h, w, _ = frames.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(crf), "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")
