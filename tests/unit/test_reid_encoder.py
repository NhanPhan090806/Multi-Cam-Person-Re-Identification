from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from multicam_reid.reid import encoder as encoder_module
from multicam_reid.reid.encoder import (
    ReIDEncoder,
    ReIDEncoderConfig,
    extract_state_dict,
    load_checkpoint,
    preprocess_person_crop,
    read_bgr_image,
)


class MeanEmbeddingModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channel_means = inputs.mean(dim=(2, 3))
        return torch.cat((channel_means, channel_means.sum(dim=1, keepdim=True)), dim=1)


class RecordingModel:
    def __init__(self) -> None:
        self.loaded_state: dict[str, torch.Tensor] | None = None
        self.strict: bool | None = None
        self.selected_device: torch.device | None = None
        self.evaluation_mode = False

    def load_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool) -> None:
        self.loaded_state = state
        self.strict = strict

    def to(self, device: torch.device) -> RecordingModel:
        self.selected_device = device
        return self

    def eval(self) -> RecordingModel:
        self.evaluation_mode = True
        return self


def test_preprocess_person_crop_converts_bgr_to_normalized_rgb() -> None:
    red_bgr_crop = np.array([[[0, 0, 255]]], dtype=np.uint8)

    tensor = preprocess_person_crop(red_bgr_crop, height=2, width=1)

    expected = np.array(
        [
            (1.0 - 0.485) / 0.229,
            (0.0 - 0.456) / 0.224,
            (0.0 - 0.406) / 0.225,
        ],
        dtype=np.float32,
    )
    assert tensor.shape == (3, 2, 1)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[:, 0, 0], expected, rtol=1e-6)


@pytest.mark.parametrize(
    "crop",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((0, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
    ],
)
def test_preprocess_person_crop_rejects_invalid_crops(crop: np.ndarray) -> None:
    with pytest.raises(ValueError):
        preprocess_person_crop(crop)


def test_read_bgr_image_supports_unicode_path(tmp_path: Path) -> None:
    image_path = tmp_path / "Bài Tập" / "person.png"
    image_path.parent.mkdir()
    expected = np.full((8, 4, 3), (10, 20, 30), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".png", expected)
    assert encoded is True
    image_path.write_bytes(buffer.tobytes())

    image = read_bgr_image(image_path)

    np.testing.assert_array_equal(image, expected)


def test_config_rejects_unsupported_device() -> None:
    with pytest.raises(ValueError, match="device"):
        ReIDEncoderConfig(device="tpu")


@pytest.mark.parametrize(
    "values",
    [
        {"model_name": ""},
        {"num_classes": 0},
        {"height": 0},
        {"width": -1},
    ],
)
def test_config_rejects_invalid_model_dimensions(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ReIDEncoderConfig(**values)


def test_load_checkpoint_accepts_torchreid_numpy_metadata(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pth.tar-60"
    torch.save(
        {
            "epoch": 60,
            "rank1": np.float32(0.8402613),
            "state_dict": {"weight": torch.ones(2)},
        },
        checkpoint_path,
    )

    checkpoint = load_checkpoint(checkpoint_path, torch_module=torch)

    assert checkpoint["epoch"] == 60
    assert float(checkpoint["rank1"]) == pytest.approx(0.8402613)
    assert torch.equal(checkpoint["state_dict"]["weight"], torch.ones(2))


def test_load_checkpoint_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_checkpoint(tmp_path / "missing.pth", torch_module=torch)


def test_extract_state_dict_accepts_wrapped_and_parallel_weights() -> None:
    checkpoint = {
        "state_dict": {
            "module.conv.weight": torch.ones(1),
            "module.conv.bias": torch.zeros(1),
        }
    }

    state_dict = extract_state_dict(checkpoint)

    assert set(state_dict) == {"conv.weight", "conv.bias"}


def test_extract_state_dict_accepts_direct_weights() -> None:
    state_dict = extract_state_dict({"conv.weight": torch.ones(1)})

    assert torch.equal(state_dict["conv.weight"], torch.ones(1))


def test_extract_state_dict_rejects_invalid_checkpoint() -> None:
    with pytest.raises(ValueError, match="state_dict"):
        extract_state_dict({"epoch": 60})


def test_encoder_returns_unit_normalized_batch_embeddings() -> None:
    encoder = ReIDEncoder(
        MeanEmbeddingModel(),
        torch_module=torch,
        device=torch.device("cpu"),
        height=4,
        width=2,
    )
    crops = [
        np.full((8, 4, 3), (0, 0, 255), dtype=np.uint8),
        np.full((8, 4, 3), (0, 255, 0), dtype=np.uint8),
    ]

    embeddings = encoder.encode(crops)

    assert embeddings.shape == (2, 4)
    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), np.ones(2), atol=1e-6)
    assert not np.allclose(embeddings[0], embeddings[1])


def test_from_checkpoint_reconstructs_model_and_metadata(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pth.tar-60"
    torch.save(
        {
            "epoch": 60,
            "rank1": np.float32(0.84),
            "state_dict": {"weight": torch.ones(2)},
        },
        checkpoint_path,
    )
    model = RecordingModel()
    builder_arguments: dict[str, object] = {}

    def model_builder(**kwargs: object) -> RecordingModel:
        builder_arguments.update(kwargs)
        return model

    encoder = ReIDEncoder.from_checkpoint(
        checkpoint_path,
        ReIDEncoderConfig(device="cpu"),
        torch_module=torch,
        model_builder=model_builder,
    )

    assert builder_arguments == {
        "name": "osnet_x0_25",
        "num_classes": 751,
        "loss": "triplet",
        "pretrained": False,
        "use_gpu": False,
    }
    assert model.strict is True
    assert model.loaded_state is not None
    assert torch.equal(model.loaded_state["weight"], torch.ones(2))
    assert model.evaluation_mode is True
    assert encoder.device == "cpu"
    assert encoder.checkpoint_epoch == 60
    assert encoder.checkpoint_rank1 == pytest.approx(0.84)


def test_from_checkpoint_rejects_unavailable_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        ReIDEncoder.from_checkpoint(
            tmp_path / "unused.pth",
            ReIDEncoderConfig(device="cuda"),
            torch_module=torch,
            model_builder=lambda **kwargs: RecordingModel(),
        )


@pytest.mark.parametrize("output", [torch.ones(4), torch.zeros((1, 4))])
def test_encoder_rejects_invalid_model_embedding(output: torch.Tensor) -> None:
    class FixedModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return output

    encoder = ReIDEncoder(
        FixedModel(),
        torch_module=torch,
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError):
        encoder.encode_one(np.zeros((8, 4, 3), dtype=np.uint8))


def test_encoder_rejects_empty_batch() -> None:
    encoder = ReIDEncoder(
        MeanEmbeddingModel(),
        torch_module=torch,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="at least one"):
        encoder.encode([])


def test_checkpoint_verification_cli_prints_embedding_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint_path = tmp_path / "model.pth.tar-60"
    checkpoint_path.touch()

    class FakeEncoder:
        checkpoint_epoch = 60
        checkpoint_rank1 = 0.84
        device = "cpu"

        def encode_one(self, crop: np.ndarray) -> np.ndarray:
            assert crop.shape == (256, 128, 3)
            return np.array([0.6, 0.8], dtype=np.float32)

    monkeypatch.setattr(
        encoder_module.ReIDEncoder,
        "from_checkpoint",
        lambda checkpoint, config: FakeEncoder(),
    )

    exit_code = encoder_module.main(
        ["--checkpoint", str(checkpoint_path), "--device", "cpu"]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"checkpoint_epoch": 60' in output
    assert '"embedding_dimension": 2' in output
    assert '"embedding_norm": 1.0' in output
