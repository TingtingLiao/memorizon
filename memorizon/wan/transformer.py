"""Wan2.2 DiT with slot-indexed RoPE, a KV cache, and a camera (PRoPE) branch per block.

Changes from ``diffusers.WanTransformer3DModel``:

* **Slot-indexed temporal RoPE.** The caller passes one temporal index per latent
  frame (``rope_index``). A retrieved memory frame from minutes ago can therefore
  sit next to the frame being generated, and several memory frames can share an
  index.
* **Frame-level attention masks** (``[F, F]`` bool) are converted to a
  ``flex_attention`` block mask, so each chunk attends only to its own context.
* **KV cache** for chunk-by-chunk inference: the clean context is forwarded once
  per chunk, then each denoising step forwards only the chunk being generated.
* **Camera branch.** Each block may carry ``prope_attn`` (see ``memorizon.camera``),
  added as a residual after the block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import (
    WanTimeTextImageEmbedding,
    WanTransformer3DModel,
)
from einops import rearrange, repeat
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.nn.attention.flex_attention import flex_attention as _flex_attention

flex_attention = torch.compile(_flex_attention)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = cos[..., 0::2], sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(x)


def attention(q, k, v, mask=None):
    """``[B, S, H, D]`` in and out. ``mask`` is a flex ``BlockMask`` or None."""
    q, k, v = (rearrange(t, "b s h d -> b h s d") for t in (q, k, v))
    if mask is None:
        out = F.scaled_dot_product_attention(q, k, v)
    else:
        out = flex_attention(q, k, v, block_mask=mask)
    return rearrange(out, "b h s d -> b s h d")


def frame_mask_to_block_mask(mask: torch.Tensor, tokens_per_frame: int) -> BlockMask:
    """``[F_q, F_k]`` bool frame mask -> token-level ``BlockMask``."""
    mask = mask.to(torch.bool)
    return _frame_mask_to_block_mask(mask, tokens_per_frame)


@torch.compile
def _frame_mask_to_block_mask(mask, tpf):
    def mask_mod(b, h, q_idx, kv_idx):
        return mask[q_idx // tpf, kv_idx // tpf]

    q_frames, kv_frames = mask.shape
    return create_block_mask(mask_mod, None, None, q_frames * tpf, kv_frames * tpf,
                             device=mask.device)


def _modulate(x, scale, shift):
    b, f, _, d = scale.shape
    return (x.view(b, f, -1, d) * (1 + scale) + shift).view(b, -1, d)


def _gated_residual(x, y, gate):
    b, f, _, d = gate.shape
    return (x.view(b, f, -1, d) + y.view(b, f, -1, d) * gate).view(b, -1, d)


class SelfAttnProcessor:
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None, rotary_emb_kv=None, kv_cache=None,
                 kv_cache_range=None):
        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        head_dim = attn.out_dim // attn.heads
        q, k, v = (rearrange(t, "b s (n e) -> b s n e", e=head_dim) for t in (q, k, v))

        q = apply_rotary_emb(q, *rotary_emb)
        if kv_cache is None:
            k = apply_rotary_emb(k, *rotary_emb)
        else:
            # Keys are cached un-rotated; the whole window is rotated to its slot
            # indices on every call. ``kv_cache_range`` is in tokens here.
            start, end = kv_cache_range
            k = _write_cache(kv_cache, "self_attn_key", k, start, end)
            v = _write_cache(kv_cache, "self_attn_value", v, start, end)
            k = apply_rotary_emb(k, *rotary_emb_kv)

        out = attention(q, k, v, attention_mask)
        return attn.to_out[1](attn.to_out[0](rearrange(out, "b s n e -> b s (n e)")))


def _write_cache(cache: dict, key: str, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Write ``x`` at the end of ``cache[key][:, :end]`` and return ``cache[key][:, start:end]``."""
    buf = cache[key]
    buf[:, end - x.shape[1]:end] = x
    return buf[:, start:end]


class CrossAttnProcessor:
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, num_frames=1, kv_cache=None):
        q = attn.to_q(hidden_states)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        head_dim = attn.out_dim // attn.heads
        b = q.size(0)
        q = rearrange(q, "b (f hw) (h e) -> (b f) h hw e", f=num_frames, e=head_dim)

        if kv_cache is not None and "cross_attn_key" in kv_cache:
            k, v = kv_cache["cross_attn_key"], kv_cache["cross_attn_value"]
        else:
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            if attn.norm_k is not None:
                k = attn.norm_k(k)
            k = rearrange(repeat(k, "b s he -> b f s he", f=num_frames),
                          "b f s (h e) -> (b f) h s e", e=head_dim)
            v = rearrange(repeat(v, "b s he -> b f s he", f=num_frames),
                          "b f s (h e) -> (b f) h s e", e=head_dim)
            if kv_cache is not None:
                kv_cache["cross_attn_key"], kv_cache["cross_attn_value"] = k, v

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "(b f) h hw e -> b (f hw) (h e)", b=b).type_as(q)
        return attn.to_out[1](attn.to_out[0](out))


class MemorizonBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False, eps: float = 1e-6):
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(query_dim=dim, heads=num_heads, kv_heads=num_heads,
                               dim_head=dim // num_heads, qk_norm=qk_norm, eps=eps, bias=True,
                               cross_attention_dim=None, out_bias=True,
                               processor=SelfAttnProcessor())
        self.attn2 = Attention(query_dim=dim, heads=num_heads, kv_heads=num_heads,
                               dim_head=dim // num_heads, qk_norm=qk_norm, eps=eps, bias=True,
                               cross_attention_dim=None, out_bias=True,
                               processor=CrossAttnProcessor())
        self.norm2 = (FP32LayerNorm(dim, eps, elementwise_affine=True)
                      if cross_attn_norm else nn.Identity())
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        self.prope_attn = None      # set by memorizon.camera.attach_camera_branch

    def forward(self, hidden_states, encoder_hidden_states, attention_mask, temb, rotary_emb,
                rotary_emb_kv, num_frames, camera=None, kv_cache=None, kv_cache_range=None):
        shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = (
            self.scale_shift_table.unsqueeze(1) + temb.float()).chunk(6, dim=2)

        h = _modulate(self.norm1(hidden_states.float()), scale_msa, shift_msa).type_as(hidden_states)
        h = self.attn1(hidden_states=h, rotary_emb=rotary_emb, rotary_emb_kv=rotary_emb_kv,
                       attention_mask=attention_mask, kv_cache=kv_cache,
                       kv_cache_range=kv_cache_range)
        hidden_states = _gated_residual(hidden_states.float(), h, gate_msa).type_as(hidden_states)

        h = self.norm2(hidden_states.float()).type_as(hidden_states)
        hidden_states = hidden_states + self.attn2(
            hidden_states=h, encoder_hidden_states=encoder_hidden_states,
            num_frames=num_frames, kv_cache=kv_cache)

        h = _modulate(self.norm3(hidden_states.float()), c_scale, c_shift).type_as(hidden_states)
        h = self.ffn(h)
        hidden_states = _gated_residual(hidden_states.float(), h.float(), c_gate).type_as(hidden_states)

        if self.prope_attn is not None:
            hidden_states = hidden_states + self.prope_attn(
                hidden_states, camera, attention_mask, kv_cache).type_as(hidden_states)
        return hidden_states


class TimeTextEmbedding(WanTimeTextImageEmbedding):
    def forward(self, timestep, encoder_hidden_states):
        timestep = self.timesteps_proj(timestep)
        dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != dtype and dtype != torch.int8:
            timestep = timestep.to(dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        return temb, self.time_proj(self.act_fn(temb)), self.text_embedder(encoder_hidden_states)


class SlotRotaryEmbedding(nn.Module):
    """3-D RoPE whose temporal axis is indexed by an arbitrary per-frame index."""

    def __init__(self, attention_head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        f64 = torch.float64
        cos, sin = [], []
        for dim in (t_dim, h_dim, w_dim):
            inv_freq = 1.0 / ((theta * torch.ones(1, dtype=f64))[:, None]
                              ** (torch.arange(0, dim, 2, dtype=f64)[: dim // 2] / dim))
            angles = torch.arange(max_seq_len, dtype=f64)[:, None, None] * inv_freq[None]
            cos.append(angles.cos().repeat_interleave(2, dim=-1).float())
            sin.append(angles.sin().repeat_interleave(2, dim=-1).float())
        self.register_buffer("freqs_cos", torch.cat(cos, dim=-1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(sin, dim=-1), persistent=False)

    def forward(self, t_index: torch.Tensor, pph: int, ppw: int):
        d = self.attention_head_dim
        split = [d - 2 * (d // 3), d // 3, d // 3]
        t_index = t_index.to(self.freqs_cos.device).long().reshape(-1)
        if int(t_index.max()) >= self.freqs_cos.shape[0]:
            raise ValueError(f"temporal index {int(t_index.max())} exceeds the RoPE table")
        f = t_index.numel()
        full = (f, pph, ppw, 1, -1)
        out = []
        for table in (self.freqs_cos, self.freqs_sin):
            t, h, w = table.split(split, dim=-1)
            out.append(torch.cat([
                t.index_select(0, t_index).view(f, 1, 1, 1, -1).expand(*full),
                h[:pph].view(1, pph, 1, 1, -1).expand(*full),
                w[:ppw].view(1, 1, ppw, 1, -1).expand(*full),
            ], dim=-1).reshape(1, f * pph * ppw, 1, -1))
        return tuple(out)


class MemorizonTransformer(WanTransformer3DModel):
    _no_split_modules = ["MemorizonBlock"]

    @register_to_config
    def __init__(self, patch_size=(1, 2, 2), num_attention_heads=40, attention_head_dim=128,
                 in_channels=16, out_channels=16, text_dim=4096, freq_dim=256, ffn_dim=13824,
                 num_layers=40, cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6,
                 image_dim=None, added_kv_proj_dim=None, rope_max_seq_len=1024,
                 rope_theta: float = 10000.0):
        super().__init__.__wrapped__(
            self, patch_size=patch_size, num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim, in_channels=in_channels,
            out_channels=out_channels, text_dim=text_dim, freq_dim=freq_dim, ffn_dim=ffn_dim,
            num_layers=num_layers, cross_attn_norm=cross_attn_norm, qk_norm=qk_norm, eps=eps,
            image_dim=image_dim, added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len)
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = SlotRotaryEmbedding(attention_head_dim, rope_max_seq_len, theta=rope_theta)
        self.condition_embedder = TimeTextEmbedding(
            dim=inner_dim, time_freq_dim=freq_dim, time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim, image_embed_dim=image_dim)
        self.blocks = nn.ModuleList([
            MemorizonBlock(inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)])

    @torch.compile
    def _patch_embedding(self, hidden_states: torch.Tensor):
        return self.patch_embedding(hidden_states).flatten(2).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor, rope_index: torch.Tensor,
                attention_mask: torch.Tensor | BlockMask | None = None, camera=None,
                kv_cache: list[dict] | None = None,
                kv_cache_range: tuple[int, int] | None = None) -> torch.Tensor:
        """Predict the flow for every frame of ``hidden_states`` ``[B, C, F, H, W]``.

        ``timestep``: ``[B, F]`` noise level per frame. ``rope_index``: temporal
        index per frame -- or, with a KV cache, per frame of the whole cache window
        ``kv_cache_range = (start, end)``, the frames being forwarded last.
        """
        b, _, f, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        pph, ppw = height // p_h, width // p_w
        assert timestep.shape == (b, f)

        if kv_cache is None:
            rotary_emb, rotary_emb_kv = self.rope(rope_index, pph, ppw), None
        else:
            start, end = kv_cache_range
            assert rope_index.numel() == end - start >= f
            rotary_emb = self.rope(rope_index[-f:], pph, ppw)
            rotary_emb_kv = self.rope(rope_index, pph, ppw)

        x = self._patch_embedding(hidden_states)
        temb, timestep_proj, text = self.condition_embedder(timestep.flatten(), encoder_hidden_states)
        temb = temb.view(b, f, 1, -1)
        timestep_proj = timestep_proj.view(b, f, 6, -1)

        if attention_mask is not None and not isinstance(attention_mask, BlockMask):
            attention_mask = frame_mask_to_block_mask(attention_mask, pph * ppw)

        token_range = None
        if kv_cache is not None:
            token_range = (kv_cache_range[0] * pph * ppw, kv_cache_range[1] * pph * ppw)
        for i, block in enumerate(self.blocks):
            x = block(x, text, attention_mask, timestep_proj, rotary_emb, rotary_emb_kv, f,
                      camera, None if kv_cache is None else kv_cache[i], token_range)

        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2, dim=2)
        x = _modulate(self.norm_out(x.float()), scale, shift).type_as(x)
        x = self.proj_out(x)
        return rearrange(x, "b (t h w) (ph pw c) -> b c t (h ph) (w pw)",
                         t=f, h=pph, w=ppw, ph=p_h, pw=p_w)

    def init_kv_cache(self, batch: int, frames: int, height: int, width: int) -> list[dict]:
        _, p_h, p_w = self.config.patch_size
        shape = (batch, frames * (height // p_h) * (width // p_w),
                 self.config.num_attention_heads, self.config.attention_head_dim)
        make = lambda: torch.zeros(shape, dtype=self.dtype, device=self.device)  # noqa: E731
        return [{"self_attn_key": make(), "self_attn_value": make()} for _ in self.blocks]
