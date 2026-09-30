"""Memorizon: Wan2.2 TI2V-5B with camera conditioning and a retrieved memory bank."""

from __future__ import annotations

import torch
from transformers import PretrainedConfig, PreTrainedModel

from .camera import attach_camera_branch, camera_transforms
from .wan.scheduler import FlowMatchScheduler
from .wan.transformer import MemorizonTransformer


class MemorizonConfig(PretrainedConfig):
    model_type = "memorizon"

    def __init__(self, base_model: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                 image_width: int = 864, image_height: int = 480, chunk_size: int = 4,
                 scheduler_shift: float = 3.0, num_train_timesteps: int = 1000,
                 text_max_sequence_length: int = 512, camera_compress: int = 8,
                 camera_freq_base: float = 100.0, camera_freq_scale: float = 1.0,
                 train_backbone: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.image_width = image_width
        self.image_height = image_height
        self.chunk_size = chunk_size
        self.scheduler_shift = scheduler_shift
        self.num_train_timesteps = num_train_timesteps
        self.text_max_sequence_length = text_max_sequence_length
        self.camera_compress = camera_compress
        self.camera_freq_base = camera_freq_base
        self.camera_freq_scale = camera_freq_scale
        self.train_backbone = train_backbone


class MemorizonModel(PreTrainedModel):
    """Wraps the DiT (``self.model``) and the flow-matching training objective.

    ``MemorizonModel(config)`` starts from the Wan2.2 weights with a fresh camera
    branch; ``MemorizonModel.from_pretrained(path)`` loads a Memorizon checkpoint.
    """

    config_class = MemorizonConfig
    main_input_name = "latents"
    _no_split_modules = ["MemorizonBlock"]

    def __init__(self, config: MemorizonConfig, _from_pretrained: bool = False):
        super().__init__(config)
        self.noise_scheduler = FlowMatchScheduler(shift=config.scheduler_shift, sigma_min=0.0,
                                                  extra_one_step=True)
        self.noise_scheduler.set_timesteps(config.num_train_timesteps, training=True)
        # index ``num_train_timesteps`` means "clean" (timestep 0)
        self.register_buffer("timesteps", torch.cat((self.noise_scheduler.timesteps,
                                                     torch.zeros(1))), persistent=False)
        if _from_pretrained:        # weights are loaded afterwards by from_pretrained
            dit_config = MemorizonTransformer.load_config(config.base_model, subfolder="transformer")
            self.model = MemorizonTransformer.from_config(dit_config)
        else:
            self.model = MemorizonTransformer.from_pretrained(config.base_model, subfolder="transformer")
        attach_camera_branch(self.model, config.image_width, config.image_height,
                             config.camera_compress, config.camera_freq_base,
                             config.camera_freq_scale)
        for name, p in self.named_parameters():
            p.requires_grad_(config.train_backbone or "prope_attn" in name)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("_from_pretrained", True)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def forward(self, latents: torch.Tensor, c2w: torch.Tensor, intrinsics: torch.Tensor,
                text_embedding: torch.Tensor, attention_mask: torch.Tensor,
                rope_index: torch.Tensor, n_cond: torch.Tensor, **_) -> dict:
        """Diffusion-forcing loss on one packed sample (batch size 1).

        The first ``n_cond`` frames are clean context. Every target chunk gets its
        own random noise level; the loss covers target frames only.
        """
        B, _, T, _, _ = latents.shape
        assert B == 1, "samples have different lengths; use batch size 1"
        device, cs = latents.device, self.config.chunk_size
        n = int(n_cond[0])
        n_chunks = (T - n) // cs
        clean = self.config.num_train_timesteps

        ids = torch.randint(clean, size=(B, n_chunks)).repeat_interleave(cs, dim=-1)
        ids = torch.cat([torch.full((B, n), clean), ids], dim=1).to(device)
        noise = torch.randn_like(latents)
        timesteps = self.timesteps[ids]
        is_target = (ids < clean).view(B, 1, T, 1, 1)
        noisy = self.noise_scheduler.batch_add_frame_noise(latents, noise, timesteps)
        noisy = torch.where(is_target, noisy, latents).to(self.dtype)
        target = self.noise_scheduler.training_target(latents, noise, timesteps)
        weights = self.noise_scheduler.batch_frame_training_weight(timesteps).to(device)

        pred = self.model(noisy, timesteps, text_embedding.to(device, self.dtype),
                          rope_index=rope_index[0].to(device),
                          attention_mask=attention_mask[0].to(device),
                          camera=camera_transforms(self.model, c2w.to(device), intrinsics.to(device)))
        loss = (pred.float() - target.float()) ** 2
        loss = (weights.view(B, 1, T, 1, 1) * loss)[is_target.broadcast_to(loss.shape)]
        return {"loss": loss.mean()}
