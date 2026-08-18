#!/usr/bin/env python3
"""Single-910B Flash3D pre-training without a CUDA/Lightning dependency.

Depth Anything V2 Base is loaded as a frozen depth teacher.  This script
updates only Flash3D's lightweight ResNet and Gaussian heads, using the
repository's standard-PyTorch differentiable Gaussian renderer on NPU.
It intentionally covers one card; distributed HCCL launch is out of scope.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.util import create_datasets
from models.model import to_device
from trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True, help="RealEstate10K root")
    parser.add_argument("--output", type=Path, required=True, help="Experiment directory; checkpoints are written below it")
    parser.add_argument("--depth-path", type=Path, default=None, help="Optional frozen Depth Anything cache")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs all epochs; otherwise stop after this many optimizer steps")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint file or checkpoints directory")
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override, repeatable")
    return parser.parse_args()


def make_cfg(args: argparse.Namespace):
    overrides = [
        "+experiment=layered_re10k_npu",
        f"dataset.data_path={args.data_path}",
        f"data_loader.batch_size={args.batch_size}",
        f"data_loader.num_workers={args.workers}",
        f"optimiser.learning_rate={args.learning_rate}",
        "model.depth.freeze=true",
        # This is a dedicated non-Lightning loop. Avoid external logging and
        # metric-model downloads while bringing up an NPU environment.
        "train.logging=false",
        "loss.lpips.weight=0.0",
    ]
    if args.depth_path is not None:
        overrides.extend(("dataset.preload_depths=true", f"dataset.depth_path={args.depth_path}"))
    overrides.extend(args.override)
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("--epochs and --batch-size must be positive; --workers must be non-negative")
    if args.max_steps < 0 or args.save_every < 1 or args.log_every < 1:
        raise ValueError("--max-steps must be non-negative; --save-every/--log-every must be positive")
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise SystemExit("torch_npu is required. Source CANN and install requirements_npu.txt first.") from error
    if not torch.npu.is_available():
        raise SystemExit("No Ascend NPU is available. Check CANN and ASCEND_RT_VISIBLE_DEVICES.")

    # Resolve user paths before changing into the experiment directory, so a
    # relative dataset/cache/resume path has the same meaning as the shell
    # command that launched pre-training.
    args.data_path = args.data_path.resolve()
    args.output = args.output.resolve()
    if args.depth_path is not None:
        args.depth_path = args.depth_path.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output)
    cfg = make_cfg(args)
    cfg.run.num_keep_ckpts = max(1, int(cfg.run.num_keep_ckpts))
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.manual_seed(int(cfg.run.random_seed))
    trainer = Trainer(cfg).to(device)
    depth_module = trainer.model.models["depth_anything_extended"].depth_model
    if any(parameter.requires_grad for parameter in depth_module.parameters()):
        raise RuntimeError("Refusing to pre-train: model.depth.freeze must be true")
    trainer.model.set_train()
    optimizer = torch.optim.Adam(trainer.model.parameters_to_train, args.learning_rate)
    if args.resume is not None:
        trainer.model.load_model(args.resume, optimiser=optimizer, device="cpu")
    _, loader = create_datasets(cfg, split="train")
    if len(loader) == 0:
        raise RuntimeError("The training loader is empty. Check --data-path and dataset filters.")

    stats_path = Path("train_metrics.jsonl")
    started = time.perf_counter()
    stopped = False
    for epoch in range(args.epochs):
        trainer.epoch = epoch
        for batch_index, inputs in enumerate(loader):
            inputs = to_device(inputs, device)
            inputs["target_frame_ids"] = cfg.model.gauss_novel_frames
            optimizer.zero_grad(set_to_none=True)
            torch.npu.synchronize()
            step_started = time.perf_counter()
            with torch.autocast(device_type="npu", dtype=torch.float16, enabled=args.amp):
                losses, _ = trainer(inputs)
            losses["loss/total"].backward()
            optimizer.step()
            torch.npu.synchronize()
            duration = time.perf_counter() - step_started
            trainer.step += 1
            if trainer.step % args.log_every == 0 or trainer.step == 1:
                item = {
                    "step": trainer.step,
                    "epoch": epoch,
                    "batch": batch_index,
                    "loss": float(losses["loss/total"].detach().float().cpu()),
                    "seconds": duration,
                    "steps_per_second": 1.0 / max(duration, 1e-9),
                }
                with stats_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item) + "\n")
                print(json.dumps(item))
            if trainer.step % args.save_every == 0:
                trainer.model.save_model(optimizer, trainer.step)
            if args.max_steps and trainer.step >= args.max_steps:
                stopped = True
                break
        if stopped:
            break

    trainer.model.save_model(optimizer, trainer.step)
    torch.npu.synchronize()
    report = {
        "device": str(device),
        "steps": trainer.step,
        "epochs_completed": epoch + 1,
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_depth_anything_v2": True,
        "depth_cache": None if args.depth_path is None else str(args.depth_path.resolve()),
        "checkpoint_dir": str((Path.cwd() / "checkpoints").resolve()),
    }
    Path("pretrain_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
