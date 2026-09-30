"""umT5 prompt embeddings, as Wan2.2 computes them (no token padding; the embedding
is zero-padded to ``max_len``).

    python -m memorizon.text "" null_prompt_emb.pt       # empty-prompt embedding for training
"""

from __future__ import annotations

import sys

import torch


class TextEncoder:
    def __init__(self, base_model: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers", max_len: int = 512):
        from transformers import AutoTokenizer, UMT5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        self.encoder = UMT5EncoderModel.from_pretrained(base_model, subfolder="text_encoder").float().eval()
        self.max_len = max_len

    @torch.no_grad()
    def __call__(self, prompt: str) -> torch.Tensor:
        """``[1, max_len, 4096]`` float32."""
        tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_len, return_tensors="pt")
        emb = self.encoder(**tokens).last_hidden_state
        return torch.cat([emb, emb.new_zeros(1, self.max_len - emb.shape[1], emb.shape[2])], dim=1)


if __name__ == "__main__":
    prompt, out = sys.argv[1], sys.argv[2]
    base = sys.argv[3] if len(sys.argv) > 3 else "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    torch.save(TextEncoder(base)(prompt), out)
    print(f"wrote {out}")
