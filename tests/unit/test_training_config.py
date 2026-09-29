from pathlib import Path

import pytest

from multicam_reid.training.config import TrainingConfig


def test_training_config_from_mapping_resolves_paths(tmp_path: Path) -> None:
    config = TrainingConfig.from_mapping(
        {
            "data_root": "data/reid",
            "output_dir": "outputs/train",
            "model_name": "osnet_x0_25",
            "batch_size": 16,
            "max_epoch": 1,
            "workers": 0,
        },
        base_dir=tmp_path,
    )

    assert config.data_root == (tmp_path / "data/reid").resolve()
    assert config.output_dir == (tmp_path / "outputs/train").resolve()
    assert config.model_name == "osnet_x0_25"
    assert config.batch_size == 16
    assert config.max_epoch == 1
    assert config.workers == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("max_epoch", 0),
        ("workers", -1),
        ("learning_rate", 0),
        ("embedding_samples", 0),
    ],
)
def test_training_config_rejects_invalid_numeric_values(field: str, value: int) -> None:
    values = {
        "data_root": "data/reid",
        "output_dir": "outputs/train",
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        TrainingConfig.from_mapping(values, base_dir=Path.cwd())


def test_training_config_smoke_copy_is_small_and_non_destructive(tmp_path: Path) -> None:
    config = TrainingConfig.from_mapping(
        {
            "data_root": "data/reid",
            "output_dir": "outputs/train",
            "batch_size": 32,
            "max_epoch": 60,
            "workers": 4,
        },
        base_dir=tmp_path,
    )

    smoke = config.for_smoke_test()

    assert smoke.max_epoch == 1
    assert smoke.batch_size <= 16
    assert smoke.workers == 0
    assert config.max_epoch == 60
    assert config.batch_size == 32


def test_training_config_from_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "train.yaml"
    config_path.write_text(
        "data_root: data/reid\n"
        "output_dir: outputs/train\n"
        "batch_size: 16\n"
        "num_instances: 4\n",
        encoding="utf-8",
    )

    config = TrainingConfig.from_yaml(config_path)

    assert config.data_root == (tmp_path / "data/reid").resolve()
    assert config.output_dir == (tmp_path / "outputs/train").resolve()


def test_training_config_rejects_unknown_field(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        TrainingConfig.from_mapping(
            {
                "data_root": "data/reid",
                "output_dir": "outputs/train",
                "mystery": True,
            },
            base_dir=tmp_path,
        )
