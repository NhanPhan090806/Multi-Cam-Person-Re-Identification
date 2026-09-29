"""Load a trained OSNet checkpoint and encode OpenCV person crops."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
DEFAULT_CHECKPOINT = Path(
    "outputs/train/osnet_x0_25_market1501_results/model/model.pth.tar-60"
)


@dataclass(frozen=True, slots=True)
class ReIDEncoderConfig:
    """Runtime settings required to reconstruct the trained OSNet model."""

    model_name: str = "osnet_x0_25"
    num_classes: int = 751
    height: int = 256
    width: int = 128
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")


def read_bgr_image(image_path: Path) -> np.ndarray:
    """Read a BGR image without OpenCV's Windows Unicode-path limitation."""
    path = image_path.expanduser().resolve()
    try:
        encoded = path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Person crop does not exist: {path}") from error
    if not encoded:
        raise ValueError(f"Person crop is empty: {path}")

    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode person crop: {path}")
    return image


def preprocess_person_crop(
    crop: np.ndarray,
    *,
    height: int = 256,
    width: int = 128,
) -> np.ndarray:
    """Convert one uint8 OpenCV BGR crop to normalized CHW RGB floats."""
    if not isinstance(crop, np.ndarray):
        raise TypeError("crop must be a NumPy array")
    if crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("crop shape must be HxWx3")
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        raise ValueError("crop must be non-empty")
    if crop.dtype != np.uint8:
        raise ValueError("crop dtype must be uint8")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    pixels = resized.astype(np.float32) / 255.0
    normalized = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


def _numpy_safe_globals() -> list[Any]:
    """Return the narrow NumPy allowlist needed by Torchreid checkpoints."""
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core

    safe_globals: list[Any] = [numpy_core.multiarray.scalar, np.dtype]
    numpy_dtypes = getattr(np, "dtypes", None)
    if numpy_dtypes is not None:
        for name in ("Float32DType", "Float64DType"):
            dtype_class = getattr(numpy_dtypes, name, None)
            if dtype_class is not None:
                safe_globals.append(dtype_class)
    return safe_globals


def load_checkpoint(
    checkpoint_path: Path,
    *,
    torch_module: Any | None = None,
) -> Mapping[str, Any]:
    """Load a trusted project checkpoint using PyTorch's restricted loader."""
    path = checkpoint_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Re-ID checkpoint does not exist: {path}")

    if torch_module is None:
        import torch as torch_module

    safe_globals = getattr(torch_module.serialization, "safe_globals", None)
    if safe_globals is None:
        raise RuntimeError("This project requires PyTorch with weights-only checkpoint loading")

    with safe_globals(_numpy_safe_globals()):
        checkpoint = torch_module.load(path, map_location="cpu", weights_only=True)

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping")
    return checkpoint


def extract_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Extract model weights and remove an optional DataParallel prefix."""
    wrapped_state = checkpoint.get("state_dict")
    if wrapped_state is not None:
        if not isinstance(wrapped_state, Mapping):
            raise ValueError("Checkpoint state_dict must be a mapping")
        state_dict = dict(wrapped_state)
    elif checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
        state_dict = dict(checkpoint)
    else:
        raise ValueError("Checkpoint does not contain a valid state_dict")

    if not state_dict or not all(isinstance(key, str) for key in state_dict):
        raise ValueError("Checkpoint state_dict must contain string keys")
    if all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _resolve_device(device: str, torch_module: Any) -> Any:
    cuda_available = bool(torch_module.cuda.is_available())
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    selected = "cuda" if cuda_available and device != "cpu" else "cpu"
    return torch_module.device(selected)


class ReIDEncoder:
    """Produce L2-normalized appearance embeddings from person crops."""

    def __init__(
        self,
        model: Any,
        *,
        torch_module: Any,
        device: Any,
        height: int = 256,
        width: int = 128,
        checkpoint_epoch: int | None = None,
        checkpoint_rank1: float | None = None,
    ) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        self._torch = torch_module
        self._device = device
        self._height = height
        self._width = width
        self._model = model.to(device)
        self._model.eval()
        self.checkpoint_epoch = checkpoint_epoch
        self.checkpoint_rank1 = checkpoint_rank1

    @property
    def device(self) -> str:
        """Return the selected inference device."""
        return str(self._device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        config: ReIDEncoderConfig | None = None,
        *,
        torch_module: Any | None = None,
        model_builder: Any | None = None,
    ) -> ReIDEncoder:
        """Reconstruct OSNet and strictly load a project checkpoint."""
        settings = config or ReIDEncoderConfig()
        if torch_module is None:
            import torch as torch_module
        if model_builder is None:
            import torchreid

            model_builder = torchreid.models.build_model

        device = _resolve_device(settings.device, torch_module)
        checkpoint = load_checkpoint(checkpoint_path, torch_module=torch_module)
        model = model_builder(
            name=settings.model_name,
            num_classes=settings.num_classes,
            loss="triplet",
            pretrained=False,
            use_gpu=device.type == "cuda",
        )
        model.load_state_dict(extract_state_dict(checkpoint), strict=True)

        epoch_value = checkpoint.get("epoch")
        rank1_value = checkpoint.get("rank1")
        return cls(
            model,
            torch_module=torch_module,
            device=device,
            height=settings.height,
            width=settings.width,
            checkpoint_epoch=int(epoch_value) if epoch_value is not None else None,
            checkpoint_rank1=float(rank1_value) if rank1_value is not None else None,
        )

    def encode(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Encode a non-empty batch of BGR person crops into unit vectors."""
        if not crops:
            raise ValueError("encode requires at least one person crop")
        batch = np.stack(
            [
                preprocess_person_crop(crop, height=self._height, width=self._width)
                for crop in crops
            ]
        )
        inputs = self._torch.from_numpy(batch).to(self._device)

        with self._torch.inference_mode():
            embeddings = self._model(inputs)
            if not hasattr(embeddings, "ndim") or embeddings.ndim != 2:
                raise RuntimeError("OSNet must return a 2D embedding tensor")
            norms = self._torch.linalg.vector_norm(embeddings, dim=1, keepdim=True)
            if bool(self._torch.any(norms <= 1e-12).item()):
                raise RuntimeError("OSNet produced a zero-length embedding")
            embeddings = embeddings / norms

        return np.ascontiguousarray(embeddings.detach().cpu().numpy(), dtype=np.float32)

    def encode_one(self, crop: np.ndarray) -> np.ndarray:
        """Encode one BGR person crop into a unit vector."""
        return self.encode([crop])[0]


def build_parser() -> argparse.ArgumentParser:
    """Create the checkpoint verification command-line parser."""
    parser = argparse.ArgumentParser(
        description="Load the trained OSNet checkpoint and verify embedding inference."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", type=Path, help="Optional BGR person crop to encode.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a checkpoint and embedding smoke test."""
    args = build_parser().parse_args(argv)
    encoder = ReIDEncoder.from_checkpoint(
        args.checkpoint,
        ReIDEncoderConfig(device=args.device),
    )
    if args.image is None:
        crop = np.zeros((256, 128, 3), dtype=np.uint8)
        input_description = "synthetic black crop"
    else:
        crop = read_bgr_image(args.image)
        input_description = str(args.image.resolve())

    embedding = encoder.encode_one(crop)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": encoder.checkpoint_epoch,
        "checkpoint_rank1": encoder.checkpoint_rank1,
        "device": encoder.device,
        "input": input_description,
        "embedding_dimension": int(embedding.shape[0]),
        "embedding_norm": float(np.linalg.norm(embedding)),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
