import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from multicam_reid.training import market1501
from multicam_reid.training.config import TrainingConfig


def _make_market1501(root: Path) -> Path:
    dataset_root = root / "market1501" / "Market-1501-v15.09.15"
    filenames = {
        "bounding_box_train": ["0001_c1s1_000001_00.jpg"],
        "query": ["0002_c1s1_000002_00.jpg"],
        "bounding_box_test": ["0002_c2s1_000003_00.jpg"],
    }
    for directory, names in filenames.items():
        split_dir = dataset_root / directory
        split_dir.mkdir(parents=True)
        for name in names:
            (split_dir / name).touch()
    return dataset_root


def _config(tmp_path: Path, *, device: str = "auto") -> TrainingConfig:
    return TrainingConfig.from_mapping(
        {
            "data_root": "data/reid",
            "output_dir": "outputs/train",
            "batch_size": 16,
            "max_epoch": 1,
            "workers": 0,
            "device": device,
        },
        base_dir=tmp_path,
    )


def test_describe_run_validates_layout_and_counts(tmp_path: Path) -> None:
    _make_market1501(tmp_path / "data/reid")

    description = market1501.describe_run(_config(tmp_path))

    assert description["data_root"] == str((tmp_path / "data/reid").resolve())
    assert description["train_images"] == 1
    assert description["query_images"] == 1
    assert description["gallery_images"] == 1


def test_torchreid_data_root_rejects_nonstandard_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Torchreid expects"):
        market1501._torchreid_data_root(tmp_path / "Market-1501-v15.09.15")


def test_resolve_device_rejects_missing_requested_cuda(tmp_path: Path) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    with pytest.raises(RuntimeError, match="CUDA"):
        market1501._resolve_device(_config(tmp_path, device="cuda"), fake_torch)


@pytest.mark.parametrize(
    ("configured", "available", "expected"),
    [
        ("auto", True, True),
        ("auto", False, False),
        ("cpu", True, False),
        ("cuda", True, True),
    ],
)
def test_resolve_device(
    tmp_path: Path, configured: str, available: bool, expected: bool
) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))

    assert market1501._resolve_device(_config(tmp_path, device=configured), fake_torch) is expected


def test_run_training_wires_torchreid_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    model = Mock()
    optimizer = object()
    scheduler = object()
    engine = Mock()
    datamanager = SimpleNamespace(num_train_pids=751)

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            manual_seed_all=Mock(),
        ),
        manual_seed=Mock(),
    )
    fake_torchreid = SimpleNamespace(
        data=SimpleNamespace(ImageDataManager=Mock(return_value=datamanager)),
        models=SimpleNamespace(build_model=Mock(return_value=model)),
        optim=SimpleNamespace(
            build_optimizer=Mock(return_value=optimizer),
            build_lr_scheduler=Mock(return_value=scheduler),
        ),
        engine=SimpleNamespace(ImageTripletEngine=Mock(return_value=engine)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchreid", fake_torchreid)
    monkeypatch.setattr(
        market1501,
        "describe_run",
        lambda _: {
            "data_root": str(tmp_path / "data/reid"),
            "output_dir": str(config.output_dir),
        },
    )

    market1501.run_training(config)

    fake_torchreid.data.ImageDataManager.assert_called_once()
    manager_kwargs = fake_torchreid.data.ImageDataManager.call_args.kwargs
    assert manager_kwargs["train_sampler"] == "RandomIdentitySampler"
    assert manager_kwargs["num_instances"] == 4
    fake_torchreid.models.build_model.assert_called_once_with(
        name="osnet_x0_25",
        num_classes=751,
        loss="triplet",
        pretrained=True,
        use_gpu=False,
    )
    fake_torchreid.engine.ImageTripletEngine.assert_called_once()
    engine.run.assert_called_once()
    assert engine.run.call_args.kwargs["ranks"] == [1, 5, 10]
    assert engine.run.call_args.kwargs["normalize_feature"] is True


def test_training_cli_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "configs" / "train.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "data_root: data/reid\n"
        "output_dir: outputs/train\n"
        "batch_size: 16\n"
        "num_instances: 4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        market1501,
        "describe_run",
        lambda config: {"max_epoch": config.max_epoch, "batch_size": config.batch_size},
    )

    result = market1501.main(
        ["--config", str(config_path), "--smoke", "--dry-run"]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert '"max_epoch": 1' in output
    assert '"batch_size": 16' in output
