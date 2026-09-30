"""Camera conditioning with PRoPE ("Cameras as Relative Positional Encoding",
arXiv:2507.10496).

Every transformer block gets a narrow side attention whose only positional
signal is the relative projective transform between the cameras of two frames,
added as a residual after the block. Its output projection is zero-initialised,
so a freshly attached branch leaves the pretrained backbone unchanged.

Conventions: camera-to-world poses (``c2w``, 4x4, OpenCV axes: x right, y down,
z forward), translations divided by a fixed scene scale, intrinsics
``(fx, fy, cx, cy)`` in pixels of the ``image_width x image_height`` frame.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan.transformer import flex_attention

VAE_SPATIAL_RATIO = 16


def invert_se3(mat: torch.Tensor) -> torch.Tensor:
    """Invert rigid transforms ``[..., 4, 4]`` in closed form."""
    rot_t = mat[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(mat)
    out[..., :3, :3] = rot_t
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", rot_t, mat[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _intrinsics_matrix(intr: torch.Tensor) -> torch.Tensor:
    """``[..., 4]`` (fx, fy, cx, cy) -> ``[..., 3, 3]``."""
    K = intr.new_zeros(intr.shape[:-1] + (3, 3))
    K[..., 0, 0], K[..., 1, 1] = intr[..., 0], intr[..., 1]
    K[..., 0, 2], K[..., 1, 2] = intr[..., 2], intr[..., 3]
    K[..., 2, 2] = 1.0
    return K


def _normalize_K(K: torch.Tensor, width: int, height: int) -> torch.Tensor:
    out = torch.zeros_like(K)
    out[..., 0, 0] = K[..., 0, 0] / width
    out[..., 1, 1] = K[..., 1, 1] / height
    out[..., 0, 2] = K[..., 0, 2] / width - 0.5
    out[..., 1, 2] = K[..., 1, 2] / height - 0.5
    out[..., 2, 2] = 1.0
    return out


def _invert_K(K: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(K)
    out[..., 0, 0] = 1.0 / K[..., 0, 0]
    out[..., 1, 1] = 1.0 / K[..., 1, 1]
    out[..., 0, 2] = -K[..., 0, 2] / K[..., 0, 0]
    out[..., 1, 2] = -K[..., 1, 2] / K[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def _lift(K: torch.Tensor) -> torch.Tensor:
    out = K.new_zeros(K.shape[:-2] + (4, 4))
    out[..., :3, :3] = K
    out[..., 3, 3] = 1.0
    return out


def _apply_projmat(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a per-camera 4x4 matrix to ``feats`` ``[B, heads, frames*tokens, D]``
    viewed as ``D/4`` homogeneous 4-vectors."""
    b, n, s, d = feats.shape
    cams = matrix.shape[1]
    return torch.einsum("bcij,bncpkj->bncpki", matrix,
                        feats.reshape(b, n, cams, -1, d // 4, 4)).reshape(feats.shape)


def _rope_coeffs(positions, freq_base, freq_scale, dim, dtype):
    n = dim // 2
    freqs = freq_scale * (freq_base ** (-torch.arange(n, device=positions.device)[None, None, None, :] / n))
    angles = positions[None, None, :, None] * freqs
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _apply_rope(feats, coeffs, inverse=False):
    cos, sin = coeffs
    if cos.shape[2] != feats.shape[2]:
        reps = feats.shape[2] // cos.shape[2]
        cos, sin = cos.repeat(1, 1, reps, 1), sin.repeat(1, 1, reps, 1)
    cos, sin = cos.to(feats.dtype), sin.to(feats.dtype)
    half = feats.shape[-1] // 2
    x, y = feats[..., :half], feats[..., half:]
    if inverse:
        return torch.cat((cos * x - sin * y, sin * x + cos * y), dim=-1)
    return torch.cat((cos * x + sin * y, -sin * x + cos * y), dim=-1)


def _block_diag(feats, funcs_and_sizes):
    funcs, sizes = zip(*funcs_and_sizes)
    return torch.cat([f(x) for f, x in zip(funcs, torch.split(feats, list(sizes), dim=-1))], dim=-1)


class CameraTransforms:
    """The q / kv / output transforms of one forward, shared by every block."""

    def __init__(self, c2w: torch.Tensor, intrinsics: torch.Tensor, head_dim: int,
                 patches_x: int, patches_y: int, image_width: int, image_height: int,
                 freq_base: float = 100.0, freq_scale: float = 1.0,
                 dtype: torch.dtype = torch.float32):
        """``c2w`` ``[B, N, 4, 4]`` for the N frames being forwarded, ``intrinsics``
        ``[B, 4]`` (one camera model per clip)."""
        K = _intrinsics_matrix(intrinsics)[:, None].expand(-1, c2w.shape[1], -1, -1)
        K = _normalize_K(K, image_width, image_height)
        P = (_lift(K) @ invert_se3(c2w)).to(dtype)                 # image <- world
        P_inv = (c2w @ _lift(_invert_K(K))).to(dtype)              # world <- image
        pos_x = torch.tile(torch.arange(patches_x, device=c2w.device), (patches_y,)).float()
        pos_y = torch.repeat_interleave(torch.arange(patches_y, device=c2w.device), patches_x).float()
        rx = _rope_coeffs(pos_x, freq_base, freq_scale, head_dim // 4, dtype)
        ry = _rope_coeffs(pos_y, freq_base, freq_scale, head_dim // 4, dtype)
        half, quarter = head_dim // 2, head_dim // 4

        def make(matrix, inverse):
            return partial(_block_diag, funcs_and_sizes=[
                (partial(_apply_projmat, matrix=matrix), half),
                (partial(_apply_rope, coeffs=rx, inverse=inverse), quarter),
                (partial(_apply_rope, coeffs=ry, inverse=inverse), quarter)])

        self.apply_q = make(P.transpose(-1, -2), False)
        self.apply_kv = make(P_inv, False)
        self.apply_o = make(P, True)


class CameraBranch(nn.Module):
    """PRoPE side attention attached to one transformer block (``block.prope_attn``)."""

    def __init__(self, dim: int, num_heads: int, compress: int = 8, eps: float = 1e-6):
        super().__init__()
        self.attn_dim = dim // compress
        self.num_heads = max(1, num_heads // compress)
        self.head_dim = self.attn_dim // self.num_heads
        assert self.head_dim % 8 == 0
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.q_proj = nn.Linear(dim, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.attn_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, camera: CameraTransforms, attn_mask=None,
                kv_cache: dict | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.norm(x)
        split = lambda proj: proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # noqa: E731
        q = camera.apply_q(split(self.q_proj))
        k = camera.apply_kv(split(self.k_proj))
        v = camera.apply_kv(split(self.v_proj))

        if kv_cache is not None:
            # Chunked inference: the first call (the clean context) is stored; later
            # calls (the chunk being denoised) attend over context + chunk.
            if "prope_key" in kv_cache:
                k = torch.cat((kv_cache["prope_key"], k), dim=2)
                v = torch.cat((kv_cache["prope_value"], v), dim=2)
            else:
                kv_cache["prope_key"], kv_cache["prope_value"] = k, v
            out = F.scaled_dot_product_attention(q, k, v)
        elif attn_mask is not None:
            out = flex_attention(q, k, v, block_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = camera.apply_o(out).transpose(1, 2).reshape(B, T, self.attn_dim)
        return self.out_proj(out)


def attach_camera_branch(transformer: nn.Module, image_width: int, image_height: int,
                         compress: int = 8, freq_base: float = 100.0, freq_scale: float = 1.0):
    """Give every block of ``transformer`` a zero-initialised ``prope_attn`` branch."""
    cfg = transformer.config
    dim = cfg.num_attention_heads * cfg.attention_head_dim
    _, p_h, p_w = cfg.patch_size
    for block in transformer.blocks:
        if block.prope_attn is None:
            block.prope_attn = CameraBranch(dim, cfg.num_attention_heads, compress)
    transformer.camera_config = dict(
        head_dim=transformer.blocks[0].prope_attn.head_dim,
        patches_x=image_width // (VAE_SPATIAL_RATIO * p_w),
        patches_y=image_height // (VAE_SPATIAL_RATIO * p_h),
        image_width=image_width, image_height=image_height,
        freq_base=freq_base, freq_scale=freq_scale)


def camera_transforms(transformer: nn.Module, c2w: torch.Tensor,
                      intrinsics: torch.Tensor) -> CameraTransforms:
    """Transforms for ``c2w`` ``[B, N, 4, 4]`` and ``intrinsics`` ``[B, 4]``."""
    return CameraTransforms(c2w, intrinsics, **transformer.camera_config)
