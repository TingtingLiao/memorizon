"""4-step distillation with Self-Forcing + DMD.

The generator is trained on its own rollouts: each of a sample's ten target
chunks is generated from the chunks it generated before (same layout, memory
retrieval and camera conditioning as ``MemorizonPipeline``), then the generated
chunks are scored by a critic (``fake_score``) and by the teacher (``real_score``,
with classifier-free guidance and the negative prompt). The difference of their
denoised predictions is the DMD gradient. Generator, critic and teacher all start
from one Memorizon checkpoint; the saved checkpoints hold the generator only and
load with ``MemorizonPipeline`` (``num_steps=4, guidance_scale=0``).

    torchrun ... -m memorizon.self_forcing configs/self_forcing_4step.yaml
"""

from __future__ import annotations

import functools
import inspect
import os
from dataclasses import dataclass

import numpy as np
import torch
from accelerate.utils.dataclasses import get_module_class_from_name
from omegaconf import OmegaConf
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import PretrainedConfig, PreTrainedModel, Trainer, TrainerCallback, TrainingArguments

from .camera import camera_transforms
from .data import LongSpanDataset, collate
from .layout import Layout, attention_mask, rope_index
from .model import MemorizonModel
from .retrieval import Frustum, topk
from .train import OffsetSeedCallback, PreemptionCallback, latest_checkpoint, run
from .wan.scheduler import FlowMatchScheduler


class SelfForcingConfig(PretrainedConfig):
    model_type = "memorizon_self_forcing"

    def __init__(self, generator_path: str | None = None, fake_score_path: str | None = None,
                 real_score_path: str | None = None, negative_embedding_path: str | None = None,
                 generator_timesteps: int = 4, scheduler_shift: float = 3.0,
                 teacher_min_timestep: float = 20.0, teacher_max_timestep: float = 980.0,
                 guidance_scale: float = 4.0, generator_update_frequency: int = 5,
                 fake_score_loss_weighting: bool = True, dmd_normalization_eps: float = 1e-6,
                 chunk: int = 4, topk_per_chunk: int = 6, traj_scale: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        self.generator_path = generator_path
        self.fake_score_path = fake_score_path or generator_path
        self.real_score_path = real_score_path or generator_path
        self.negative_embedding_path = negative_embedding_path
        self.generator_timesteps = generator_timesteps
        self.scheduler_shift = scheduler_shift
        self.teacher_min_timestep = teacher_min_timestep
        self.teacher_max_timestep = teacher_max_timestep
        self.guidance_scale = guidance_scale
        self.generator_update_frequency = generator_update_frequency
        self.fake_score_loss_weighting = fake_score_loss_weighting
        self.dmd_normalization_eps = dmd_normalization_eps
        self.chunk = chunk
        self.topk_per_chunk = topk_per_chunk
        self.traj_scale = traj_scale


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy().astype(np.float64)


class SelfForcingModel(PreTrainedModel):
    config_class = SelfForcingConfig
    main_input_name = "latents"
    _no_split_modules = ["MemorizonBlock"]

    def __init__(self, config: SelfForcingConfig):
        super().__init__(config)
        load = lambda p: MemorizonModel.from_pretrained(p).to(torch.bfloat16)  # noqa: E731
        self.generator = load(config.generator_path)
        self.fake_score = load(config.fake_score_path)
        self.real_score = load(config.real_score_path)
        self.real_score.requires_grad_(False)

        self.gen_sched = FlowMatchScheduler(shift=config.scheduler_shift, sigma_min=0.0, extra_one_step=True)
        self.gen_sched.set_timesteps(config.generator_timesteps, training=True)
        self.tea_sched = FlowMatchScheduler(shift=config.scheduler_shift, sigma_min=0.0, extra_one_step=True)
        self.tea_sched.set_timesteps(self.generator.config.num_train_timesteps, training=True)
        neg = torch.load(config.negative_embedding_path, map_location="cpu", weights_only=True)
        self.register_buffer("negative_embedding", neg.reshape(1, *neg.shape[-2:]).to(torch.bfloat16),
                             persistent=False)

    def _flow(self, net, x, timesteps, text, mask, rope, c2w, intr):
        """One packed forward of ``net``'s DiT; returns the predicted flow."""
        return net.model(x, timesteps, text, rope_index=rope, attention_mask=mask,
                         camera=camera_transforms(net.model, c2w, intr))

    @torch.no_grad()
    def _rollout(self, lat, c2w, intr, text, n_cond, n_bank, exit_steps):
        """Generate every target chunk from the chunks generated before it (no grad).

        Chunk ``j`` stops after ``exit_steps[j] + 1`` of the few-step schedule. Returns
        the generated chunks and, per chunk, the inputs of its last step.
        """
        C, dev, dt = self.config.chunk, lat.device, lat.dtype
        ch, h, w = lat.shape[1], lat.shape[3], lat.shape[4]
        cams = c2w[0]
        bank_lat, bank_c2w = lat[:, :, 1:1 + n_bank], cams[1:1 + n_bank]
        rec_lat, rec_c2w = lat[:, :, 1 + n_bank:n_cond], cams[1 + n_bank:n_cond]
        frustum = Frustum.from_intrinsics(intr[0].tolist(), self.config.traj_scale)
        gen, steps = [], []
        for j in range((lat.shape[2] - n_cond) // C):
            q_c2w = cams[n_cond + j * C:n_cond + (j + 1) * C]
            # memory pool: the sample's bank, the real recent chunk (from j = 1), and
            # the chunks generated before j - 1 (chunk j - 1 is the new recent chunk)
            pool_lat, pool_c2w = [bank_lat], [bank_c2w]
            if j >= 1 and rec_lat.shape[2]:
                pool_lat.append(rec_lat)
                pool_c2w.append(rec_c2w)
            for t in range(j - 1):
                pool_lat.append(gen[t])
                pool_c2w.append(cams[n_cond + t * C:n_cond + (t + 1) * C])
            pool_lat, pool_c2w = torch.cat(pool_lat, dim=2), torch.cat(pool_c2w)
            if len(pool_c2w):
                take = np.sort(topk(_np(q_c2w[-1]), _np(pool_c2w), frustum,
                                    min(self.config.topk_per_chunk, len(pool_c2w))))
                take = torch.as_tensor(take, device=dev, dtype=torch.long)
                mem_lat, mem_c2w = pool_lat[:, :, take], pool_c2w[take]
            else:
                mem_lat, mem_c2w = pool_lat[:, :, :0], pool_c2w[:0]
            prev_lat = rec_lat if j == 0 else gen[j - 1]
            prev_c2w = rec_c2w if j == 0 else cams[n_cond + (j - 1) * C:n_cond + j * C]

            cond = torch.cat([lat[:, :, :1], mem_lat, prev_lat], dim=2)
            lay = Layout(n_bank=mem_lat.shape[2], n_recent=prev_lat.shape[2], chunk=C)
            F, n_c = lay.total, lay.n_cond
            seq_c2w = torch.cat([cams[:1], mem_c2w, prev_c2w, q_c2w])[None]
            rope = torch.as_tensor(rope_index(lay), dtype=torch.long, device=dev)
            mask = torch.as_tensor(attention_mask(lay), dtype=torch.bool, device=dev)

            x = torch.randn((1, ch, C, h, w), device=dev, dtype=dt)
            for s in range(exit_steps[j] + 1):
                ts = torch.zeros((1, F), device=dev)
                ts[:, n_c:] = float(self.gen_sched.timesteps[s])
                flow = self._flow(self.generator, torch.cat([cond, x], dim=2), ts, text, mask, rope,
                                  seq_c2w, intr[:1])[:, :, n_c:]
                if s == exit_steps[j]:
                    steps.append(dict(cond=cond, x=x.clone(), t=ts[0, -1].item(), c2w=seq_c2w,
                                      rope=rope, mask=mask, F=F, n_c=n_c))
                    x = self.gen_sched.step_diff_noise_level(flow, ts[:, n_c:], x, to_final=True).to(dt)
                else:
                    x = self.gen_sched.step_diff_noise_level(flow, ts[:, n_c:], x).to(dt)
            gen.append(x)
        return torch.cat(gen, dim=2), steps

    def _generator_x0(self, steps, text, intr):
        """The last step of every chunk, recomputed with grad as one block-diagonal forward."""
        dev, C = steps[0]["x"].device, self.config.chunk
        total = sum(s["F"] for s in steps)
        mask = torch.zeros((total, total), dtype=torch.bool, device=dev)
        xs, ts, cams, ropes, rows, off = [], [], [], [], [], 0
        for s in steps:
            xs.append(torch.cat([s["cond"], s["x"]], dim=2))
            t = torch.zeros(s["F"], device=dev)
            t[s["n_c"]:] = s["t"]
            ts.append(t)
            cams.append(s["c2w"][0])
            ropes.append(s["rope"])
            mask[off:off + s["F"], off:off + s["F"]] = s["mask"]
            rows.append(slice(off + s["n_c"], off + s["F"]))
            off += s["F"]
        flow = self._flow(self.generator, torch.cat(xs, dim=2), torch.cat(ts)[None], text, mask,
                          torch.cat(ropes), torch.cat(cams)[None], intr[:1])
        flow = torch.cat([flow[:, :, r] for r in rows], dim=2)
        x = torch.cat([s["x"] for s in steps], dim=2)
        t = torch.cat([torch.full((1, C), s["t"]) for s in steps], dim=1)
        return self.gen_sched.step_diff_noise_level(flow, t, x, to_final=True)

    def _noise_level_per_chunk(self, n_chunks, dev):
        tab = self.tea_sched.timesteps
        valid = tab[(tab >= self.config.teacher_min_timestep) & (tab <= self.config.teacher_max_timestep)]
        return valid[torch.randint(valid.numel(), (n_chunks,))].repeat_interleave(self.config.chunk)[None].to(dev)

    def _score(self, net, noisy, ctx, t, text):
        """``net``'s flow on the sample's packed layout, target frames only."""
        n = ctx["n_cond"]
        x = torch.cat([ctx["cond"], noisy], dim=2)
        ts = torch.cat([torch.zeros((1, n), device=x.device), t.float()], dim=1)
        return self._flow(net, x, ts, text, ctx["mask"], ctx["rope"], ctx["c2w"], ctx["intr"])[:, :, n:]

    def forward(self, latents, c2w, intrinsics, text_embedding, attention_mask, rope_index,
                n_cond, n_bank, should_update_generator: bool = True, **_):
        cfg, dev, dt = self.config, latents.device, torch.bfloat16
        lat, c2w, intr = latents.to(dt), c2w.to(dev, torch.float32), intrinsics.to(dev, torch.float32)
        text = text_embedding.to(dev, dt)
        n = int(n_cond[0])
        n_chunks = (lat.shape[2] - n) // cfg.chunk
        ctx = dict(cond=lat[:, :, :n], n_cond=n, c2w=c2w, intr=intr,
                   mask=attention_mask[0].to(dev), rope=rope_index[0].to(dev))

        # every rank must run the same number of forwards
        exit_steps = torch.randint(cfg.generator_timesteps, (n_chunks,), device=dev)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(exit_steps, src=0)
        x0, steps = self._rollout(lat, c2w, intr, text, n, int(n_bank[0]), exit_steps.tolist())

        with torch.no_grad():
            t = self._noise_level_per_chunk(n_chunks, dev)
            noise = torch.randn_like(x0)
            noisy = self.tea_sched.batch_add_frame_noise(x0, noise, t).to(dt)

        if not should_update_generator:            # critic: denoising loss on generated chunks
            target = self.tea_sched.training_target(x0, noise, t).float()
            loss = (self._score(self.fake_score, noisy, ctx, t, text).float() - target) ** 2
            if cfg.fake_score_loss_weighting:
                loss = loss * self.tea_sched.batch_frame_training_weight(t).to(dev).view(1, 1, -1, 1, 1)
            return {"loss": loss.mean()}

        with torch.no_grad():                      # generator: DMD gradient
            fake = self.tea_sched.step_diff_noise_level(
                self._score(self.fake_score, noisy, ctx, t, text), t, noisy, to_final=True)
            cond = self._score(self.real_score, noisy, ctx, t, text)
            uncond = self._score(self.real_score, noisy, ctx, t, self.negative_embedding.to(dev, dt))
            real = self.tea_sched.step_diff_noise_level(
                uncond + cfg.guidance_scale * (cond - uncond), t, noisy, to_final=True)
            ch, h, w = x0.shape[1], x0.shape[3], x0.shape[4]
            err = (x0.float() - real.float()).abs().view(1, ch, n_chunks, cfg.chunk, h, w)
            norm = err.mean(dim=(1, 3, 4, 5), keepdim=True) + cfg.dmd_normalization_eps
            grad = torch.nan_to_num(((fake.float() - real.float()).view_as(err) / norm).view_as(x0))
        x0_grad = self._generator_x0(steps, text, intr).float()
        return {"loss": 0.5 * ((x0_grad - (x0_grad - grad).detach()) ** 2).mean()}

    def save_pretrained(self, save_directory, is_main_process=True, state_dict=None, **kwargs):
        """Save the generator as a plain Memorizon checkpoint."""
        if not is_main_process:
            return
        state_dict = self.state_dict() if state_dict is None else state_dict
        gen = {k[len("generator."):]: v for k, v in state_dict.items() if k.startswith("generator.")}
        os.makedirs(save_directory, exist_ok=True)
        self.generator.save_pretrained(save_directory, state_dict=gen, safe_serialization=True,
                                       max_shard_size="5GB")


@dataclass
class SelfForcingArguments(TrainingArguments):
    fake_score_learning_rate: float = 4e-7


class SelfForcingTrainer(Trainer, TrainerCallback):
    """Alternates critic and generator updates (the generator every
    ``generator_update_frequency`` steps), each with its own learning rate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(self)
        self.n_generator_groups = None

    def _update_generator(self, offset=0):
        return (self.state.global_step + offset) % self.model.config.generator_update_frequency == 0

    def _groups(self, module, lr):
        decay = self.get_decay_parameter_names(module)
        return [{"params": [p for n, p in module.named_parameters() if n in decay and p.requires_grad],
                 "weight_decay": self.args.weight_decay, "lr": lr},
                {"params": [p for n, p in module.named_parameters() if n not in decay and p.requires_grad],
                 "weight_decay": 0.0, "lr": lr}]

    def get_optimizer_cls_and_kwargs(self, args, model=None):
        cls, kwargs = super().get_optimizer_cls_and_kwargs(args, model)
        groups = self._groups(model.generator, args.learning_rate)
        self.n_generator_groups = len(groups)
        kwargs["params"] = groups + self._groups(model.fake_score, args.fake_score_learning_rate)
        return cls, kwargs

    def _get_learning_rate(self):
        lrs = self.lr_scheduler.get_last_lr()
        lr = lrs[0] if self._update_generator() else lrs[self.n_generator_groups]
        return lr.item() if torch.is_tensor(lr) else lr

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        inputs["should_update_generator"] = self._update_generator()
        return inputs

    def on_step_begin(self, args, state, control, **kwargs):
        update_generator = self._update_generator()
        self.model_wrapped.generator.requires_grad_(update_generator)
        self.model_wrapped.fake_score.requires_grad_(not update_generator)

    def log(self, logs, start_time=None):
        prefix = "generator:" if self._update_generator(offset=-1) else "fake_score:"
        logs.update({prefix + k: v for k, v in logs.items()})
        return super().log(logs, start_time)


class CheckpointWrapPolicy:
    """Let FSDP wrap one set of module classes while activation checkpointing wraps
    another (accelerate uses a single auto-wrap policy for both)."""

    def __init__(self, model, fsdp_policy, checkpoint_classes: list[str]):
        self.fsdp_policy = fsdp_policy
        classes = {get_module_class_from_name(model, n) for n in checkpoint_classes}
        self.checkpoint_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=classes)

    def __call__(self, module, recurse, nonwrapped_numel):
        frame = inspect.currentframe().f_back
        while frame is not None and frame.f_code.co_filename.endswith(os.path.join("fsdp", "wrap.py")):
            frame = frame.f_back
        if frame is not None and frame.f_code.co_name == "apply_activation_checkpointing":
            return self.checkpoint_policy(module, recurse, nonwrapped_numel)
        return None if self.fsdp_policy is None else self.fsdp_policy(module, recurse, nonwrapped_numel)


def main(cfg):
    dtype = getattr(torch, cfg.get("train_dtype", "bfloat16"))
    sf_config = SelfForcingConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    trainer_args = OmegaConf.to_container(cfg.trainer, resolve=True)
    checkpoint_classes = (trainer_args.get("fsdp_config") or {}).pop("transformer_layer_cls_to_checkpoint", None)
    args = SelfForcingArguments(**trainer_args)
    assert args.per_device_train_batch_size == 1 and not args.remove_unused_columns
    dataset = LongSpanDataset(**OmegaConf.to_container(cfg.data, resolve=True), dtype=dtype)

    trainer = SelfForcingTrainer(model_init=lambda: SelfForcingModel(sf_config).type(dtype), args=args,
                                 train_dataset=dataset, data_collator=collate)
    trainer.add_callback(OffsetSeedCallback())
    trainer.add_callback(PreemptionCallback())
    plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
    if plugin is not None and checkpoint_classes:
        set_policy = plugin.set_auto_wrap_policy

        def set_auto_wrap_policy(model):
            set_policy(model)
            plugin.auto_wrap_policy = CheckpointWrapPolicy(model, plugin.auto_wrap_policy, checkpoint_classes)

        plugin.set_auto_wrap_policy = set_auto_wrap_policy
    resume = latest_checkpoint(args.output_dir) if cfg.get("resume", True) else None
    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    run(main)
