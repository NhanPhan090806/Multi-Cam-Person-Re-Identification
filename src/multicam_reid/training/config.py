"""Typed configuration for Market-1501 Re-ID training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Validated settings for a Torchreid training run."""

    data_root: Path
    output_dir: Path
    model_name: str = "osnet_x0_25"
    height: int = 256
    width: int = 128
    batch_size: int = 32
    test_batch_size: int = 64
    workers: int = 4
    num_instances: int = 4
    max_epoch: int = 60
    learning_rate: float = 0.0003
    weight_decay: float = 0.0005
    margin: float = 0.3
    stepsize: int = 40
    gamma: float = 0.1
    eval_freq: int = 5
    print_freq: int = 20
    seed: int = 42
    embedding_samples: int = 8
    device: str = "auto"
    pretrained: bool = True

    def __post_init__(self) -> None:
        positive_fields = (
            "height",
            "width",
            "batch_size",
            "test_batch_size",
            "num_instances",
            "max_epoch",
            "learning_rate",
            "stepsize",
            "eval_freq",
            "print_freq",
            "embedding_samples",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in the interval (0, 1]")
        if self.batch_size % self.num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: Path,
    ) -> TrainingConfig:
        """Construct a config and resolve relative paths against a known base."""
        known_fields = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known_fields)
        if unknown:
            raise ValueError(f"Unknown training configuration fields: {', '.join(unknown)}")

        parsed = dict(values)
        for key in ("data_root", "output_dir"):
            raw_path = Path(parsed[key])
            parsed[key] = (
                raw_path if raw_path.is_absolute() else base_dir / raw_path
            ).expanduser().resolve()
        return cls(**parsed)

    @classmethod
    def from_yaml(cls, path: Path) -> TrainingConfig:
        """Load a training config from YAML."""
        config_path = path.expanduser().resolve()
        with config_path.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError(f"Training config must be a YAML mapping: {config_path}")
        return cls.from_mapping(values, base_dir=config_path.parent.parent)

    def for_smoke_test(self) -> TrainingConfig:
        """Return a cheap one-epoch variant without mutating the original."""
        smoke_batch_size = min(self.batch_size, 16)
        smoke_batch_size -= smoke_batch_size % self.num_instances
        return replace(
            self,
            max_epoch=1,
            batch_size=max(smoke_batch_size, self.num_instances),
            workers=0,
            eval_freq=1,
            print_freq=5,
        )
