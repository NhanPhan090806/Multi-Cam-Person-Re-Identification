"""Torchreid training entry point for the Market-1501 baseline."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from multicam_reid.data.market1501 import inspect_market1501
from multicam_reid.training.config import TrainingConfig


def _torchreid_data_root(dataset_root: Path) -> Path:
    if dataset_root.parent.name.casefold() != "market1501":
        raise ValueError(
            "Torchreid expects Market-1501 at "
            "<data_root>/market1501/Market-1501-v15.09.15. "
            f"Found: {dataset_root}"
        )
    return dataset_root.parent.parent


def _resolve_device(config: TrainingConfig, torch_module: Any) -> bool:
    cuda_available = bool(torch_module.cuda.is_available())
    if config.device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    return cuda_available and config.device != "cpu"


def _set_seed(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def describe_run(config: TrainingConfig) -> dict[str, Any]:
    """Validate data and return a serializable run description."""
    summary = inspect_market1501(config.data_root)
    torchreid_root = _torchreid_data_root(summary.dataset_root)
    return {
        "data_root": str(torchreid_root),
        "dataset_root": str(summary.dataset_root),
        "output_dir": str(config.output_dir),
        "model_name": config.model_name,
        "batch_size": config.batch_size,
        "max_epoch": config.max_epoch,
        "train_images": summary.train.num_images,
        "train_identities": summary.train.num_identities,
        "query_images": summary.query.num_images,
        "gallery_images": summary.gallery.num_images,
    }


def run_training(config: TrainingConfig) -> None:
    """Train and evaluate OSNet using Torchreid."""
    import torch
    import torchreid

    description = describe_run(config)
    use_gpu = _resolve_device(config, torch)
    _set_seed(config.seed, torch)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    datamanager = torchreid.data.ImageDataManager(
        root=description["data_root"],
        sources="market1501",
        targets="market1501",
        height=config.height,
        width=config.width,
        transforms=["random_flip", "random_crop"],
        use_gpu=use_gpu,
        batch_size_train=config.batch_size,
        batch_size_test=config.test_batch_size,
        workers=config.workers,
        num_instances=config.num_instances,
        train_sampler="RandomIdentitySampler",
    )
    model = torchreid.models.build_model(
        name=config.model_name,
        num_classes=datamanager.num_train_pids,
        loss="triplet",
        pretrained=config.pretrained,
        use_gpu=use_gpu,
    )
    if use_gpu:
        model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=config.stepsize,
        gamma=config.gamma,
        max_epoch=config.max_epoch,
    )
    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer,
        margin=config.margin,
        weight_t=1,
        weight_x=1,
        scheduler=scheduler,
        use_gpu=use_gpu,
        label_smooth=True,
    )
    engine.run(
        save_dir=str(config.output_dir),
        max_epoch=config.max_epoch,
        print_freq=config.print_freq,
        eval_freq=config.eval_freq,
        dist_metric="euclidean",
        normalize_feature=True,
        ranks=[1, 5, 10],
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the training command-line parser."""
    parser = argparse.ArgumentParser(description="Train OSNet-x0.25 on Market-1501.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_market1501.yaml"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one epoch, a smaller batch, and no worker subprocesses.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print the resolved run without training.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Market-1501 training CLI."""
    args = build_parser().parse_args(argv)
    config = TrainingConfig.from_yaml(args.config)
    if args.smoke:
        config = config.for_smoke_test()
    if args.dry_run:
        print(json.dumps(describe_run(config), indent=2))
        return 0
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
