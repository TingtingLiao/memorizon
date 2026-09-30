"""Caption an image the way the training captions were made (Qwen3-VL-8B, same
instruction, 448x252 input, greedy decoding). Memorizon expects prompts in this
format; a short free-form prompt is out of distribution.

    python -m memorizon.caption photo.jpg
"""

from __future__ import annotations

import sys

import torch
from PIL import Image

CAPTION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CAPTION_INSTRUCTION = """You are a game scene analyst. Analyze the provided IMAGE and produce an environment description.

IMPORTANT:
- Base everything strictly on what is visible in the IMAGE. Do not invent or assume anything not shown.
- Ignore any UI/HUD overlays; do not mention them.
- Keep the caption correct and detailed but concise.
- Output ONLY the final caption paragraph, max 100 words, single flowing paragraph.

PERSPECTIVE DECISION AND REQUIRED STARTING TEXT:
- If a person/character/avatar is visible: start with "Third-person perspective — character visible. "
- If NO person/character/avatar is visible: start with "First-person perspective — character not visible. "

Describe: perspective, setting/environment, lighting, colors, terrain, key objects and spatial layout."""


class Captioner:
    def __init__(self, model: str = CAPTION_MODEL, device: str = "cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, dtype=torch.bfloat16, device_map=device).eval()

    @torch.no_grad()
    def __call__(self, image: str | Image.Image) -> str:
        im = Image.open(image) if isinstance(image, str) else image
        im = im.convert("RGB").resize((448, 252), Image.LANCZOS)
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": CAPTION_INSTRUCTION}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[im], return_tensors="pt").to(self.device)
        ids = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)
        return self.processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0].strip().replace("\n", " ")


def caption(image: str | Image.Image, model: str = CAPTION_MODEL, device: str = "cuda") -> str:
    """Caption one image, then free the captioning model."""
    captioner = Captioner(model, device)
    out = captioner(image)
    del captioner
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    print(caption(sys.argv[1], *(sys.argv[2:3] or [CAPTION_MODEL])))
