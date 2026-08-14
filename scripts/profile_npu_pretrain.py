#!/usr/bin/env python3
"""Profile real single-card Flash3D NPU pre-training steps on RE10K.

This runs forward, reconstruction loss, backward and optimizer update on real
batches. It reports NPU-synchronised step time and peak allocated memory, which
are the two values needed to tune ``npu_renderer_max_gaussians`` and batch size.
"""

from __future__ import annotations

import argparse
import json
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--depth-path", type=Path, default=None, help="Optional precomputed Depth Anything cache")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/npu_profile.json"))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra override, repeatable")
    return parser.parse_args()


def next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def one_step(trainer: Trainer, optimizer: torch.optim.Optimizer, inputs: dict, amp: bool) -> float:
    inputs["target_frame_ids"] = trainer.cfg.model.gauss_novel_frames
    optimizer.zero_grad(set_to_none=True)
    context = torch.autocast(device_type="npu", dtype=torch.float16, enabled=amp)
    with context:
        losses, _ = trainer(inputs)
    losses["loss/total"].backward()
    optimizer.step()
    return float(losses["loss/total"].detach().cpu())


def main() -> None:
    args = arguments()
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise SystemExit("torch_npu is required for NPU profiling.") from error
    if not torch.npu.is_available():
        raise SystemExit("No Ascend NPU is available.")
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("--iterations must be positive and --warmup must be non-negative")

    overrides = [
        "+experiment=layered_re10k_npu",
        f"dataset.data_path={args.data_path}",
        "data_loader.batch_size=1",
        "data_loader.num_workers=0",
    ]
    if args.depth_path is not None:
        overrides.extend(("dataset.preload_depths=true", f"dataset.depth_path={args.depth_path}"))
    overrides.extend(args.override)
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="config", overrides=overrides)

    _, loader = create_datasets(cfg, split="train")
    trainer = Trainer(cfg).to("npu:0")
    trainer.model.set_train()
    optimizer = torch.optim.Adam(trainer.model.parameters_to_train, cfg.optimiser.learning_rate)
    iterator = iter(loader)
    torch.npu.empty_cache()
    if hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats()

    for _ in range(args.warmup):
        inputs, iterator = next_batch(loader, iterator)
        one_step(trainer, optimizer, to_device(inputs, torch.device("npu:0")), args.amp)
    torch.npu.synchronize()
    durations, losses = [], []
    for _ in range(args.iterations):
        inputs, iterator = next_batch(loader, iterator)
        torch.npu.synchronize()
        started = time.perf_counter()
        loss = one_step(trainer, optimizer, to_device(inputs, torch.device("npu:0")), args.amp)
        torch.npu.synchronize()
        durations.append(time.perf_counter() - started)
        losses.append(loss)

    report = {
        "iterations": args.iterations,
        "mean_step_seconds": sum(durations) / len(durations),
        "median_step_seconds": sorted(durations)[len(durations) // 2],
        "steps_per_second": len(durations) / sum(durations),
        "mean_loss": sum(losses) / len(losses),
        "amp": args.amp,
        "renderer_max_gaussians": cfg.model.npu_renderer_max_gaussians,
        "renderer_max_gaussians_per_tile": cfg.model.npu_renderer_max_gaussians_per_tile,
    }
    if hasattr(torch.npu, "max_memory_allocated"):
        report["peak_allocated_mib"] = round(torch.npu.max_memory_allocated() / 2**20, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved profile to {args.output.resolve()}")


if __name__ == "__main__":
    main()
