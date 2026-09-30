"""Train Memorizon.

    torchrun --nnodes N --nproc_per_node 8 ... -m memorizon.train configs/train_100s.yaml [key=value ...]
"""

from __future__ import annotations

import os
import pathlib
import shutil
import signal
import sys
from datetime import timedelta

import torch
from omegaconf import OmegaConf
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import set_seed

try:
    from transformers.trainer_pt_utils import ALL_LAYERNORM_LAYERS
except ImportError:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from .data import LongSpanDataset, collate
from .model import MemorizonConfig, MemorizonModel


class MemorizonTrainer(Trainer):
    """AdamW with a separate learning rate for the camera branch."""

    camera_learning_rate: float | None = None

    def create_optimizer(self):
        if self.optimizer is not None or self.camera_learning_rate is None:
            return super().create_optimizer()
        decay = {n for n in get_parameter_names(self.model, ALL_LAYERNORM_LAYERS) if "bias" not in n}
        groups = []
        for is_cam in (False, True):
            for is_decay in (True, False):
                params = [p for n, p in self.model.named_parameters()
                          if p.requires_grad and ("prope_attn" in n) == is_cam and (n in decay) == is_decay]
                if params:
                    groups.append({"params": params,
                                   "weight_decay": self.args.weight_decay if is_decay else 0.0,
                                   "lr": self.camera_learning_rate if is_cam else self.args.learning_rate})
        cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        kwargs.pop("lr", None)
        self.optimizer = cls(groups, **kwargs)
        return self.optimizer


class OffsetSeedCallback(TrainerCallback):
    """Different random spans and noise on every rank."""

    def on_train_begin(self, args, state, control, **kwargs):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        set_seed(args.seed + rank)


class PreemptionCallback(TrainerCallback):
    """Save and stop on SIGTERM / SIGUSR1 (e.g. SLURM preemption)."""

    def __init__(self):
        self.stop = False
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, lambda *_: setattr(self, "stop", True))

    def on_step_end(self, args, state, control, **kwargs):
        flag = torch.tensor(self.stop, device=args.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, torch.distributed.ReduceOp.MAX)
        if flag:
            control.should_save = control.should_training_stop = True


def latest_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(pathlib.Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    ckpts = [c for c in ckpts if (c / "trainer_state.json").exists()]
    return str(ckpts[-1]) if ckpts else None


def main(cfg):
    dtype = getattr(torch, cfg.get("train_dtype", "bfloat16"))
    model_args = OmegaConf.to_container(cfg.model, resolve=True)
    init_from = model_args.pop("from_pretrained", None)

    def model_init():
        if init_from:
            model = MemorizonModel.from_pretrained(init_from)
        else:
            model = MemorizonModel(MemorizonConfig(**model_args))
        return model.type(dtype)

    args = TrainingArguments(**OmegaConf.to_container(cfg.trainer, resolve=True))
    assert args.per_device_train_batch_size == 1, "samples have different lengths"
    dataset = LongSpanDataset(**OmegaConf.to_container(cfg.data, resolve=True), dtype=dtype)
    print(f"[data] {len(dataset)} episodes from {dataset.root}")

    trainer = MemorizonTrainer(model_init=model_init, args=args, train_dataset=dataset,
                               data_collator=collate)
    trainer.camera_learning_rate = cfg.get("camera_learning_rate")
    trainer.add_callback(OffsetSeedCallback())
    trainer.add_callback(PreemptionCallback())
    resume = latest_checkpoint(args.output_dir) if cfg.get("resume", True) else None
    print(f"resuming from {resume}" if resume else "starting fresh")
    trainer.train(resume_from_checkpoint=resume)


def load_config(argv: list[str]):
    cfg = OmegaConf.merge(OmegaConf.load(argv[0]), OmegaConf.from_cli(argv[1:]))
    if os.environ.get("RANK", "0") == "0":
        os.makedirs(cfg.trainer.output_dir, exist_ok=True)
        shutil.copy2(argv[0], cfg.trainer.output_dir)
    return cfg


def run(main_fn):
    cfg = load_config(sys.argv[1:])
    if "RANK" in os.environ:
        torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=cfg.get("dist_timeout", 3600)))
    main_fn(cfg)
    if "RANK" in os.environ:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run(main)
