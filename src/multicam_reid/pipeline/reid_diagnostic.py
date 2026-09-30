"""Standalone Stage 4 diagnostic over actual detector/tracker person crops."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from multicam_reid.detection.yolo import YoloConfig, YoloPersonDetector
from multicam_reid.inputs.video import VideoFileSource
from multicam_reid.pipeline.reid_camera import ReIDCameraPipeline
from multicam_reid.pipeline.single_camera import CropConfig, SingleCameraPipeline
from multicam_reid.reid.encoder import DEFAULT_CHECKPOINT, ReIDEncoder, ReIDEncoderConfig
from multicam_reid.reid.tracklets import (
    TrackletEmbeddingConfig,
    TrackletEmbeddingStore,
    TrackletSeparationDiagnostic,
    evaluate_tracklet_separation,
)
from multicam_reid.tracking.bytetrack import ByteTrackConfig, ByteTrackLocalTracker
from multicam_reid.visualization.tracks import draw_reid_status

DEFAULT_OUTPUT_DIR = Path("outputs/stage4/reid_diagnostic")


@dataclass(frozen=True, slots=True)
class ReIDDiagnosticConfig:
    """Input and inference settings for the Stage 4 integration check."""

    source: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    checkpoint: Path = DEFAULT_CHECKPOINT
    camera_id: str = "C1"
    yolo_model: Path = Path("yolo26n.pt")
    device: str = "auto"
    confidence: float = 0.1
    image_size: int = 640
    history_size: int = 10
    embedding_interval_frames: int = 5
    stale_after_frames: int = 150
    max_frames: int | None = None
    display: bool = False

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        TrackletEmbeddingConfig(
            history_size=self.history_size,
            embedding_interval_frames=self.embedding_interval_frames,
            stale_after_frames=self.stale_after_frames,
        )


@dataclass(frozen=True, slots=True)
class ReIDDiagnosticSummary:
    """Saved runtime and separation evidence for the Stage 4 exit gate."""

    source: str
    checkpoint: str
    camera_id: str
    frames_processed: int
    tracklets: int
    embedding_samples: int
    embedding_dimension: int
    elapsed_seconds: float
    processing_fps: float
    diagnostic: TrackletSeparationDiagnostic
    embedding_artifact: str

    def to_dict(self) -> dict[str, object]:
        """Return a nested JSON-ready representation."""
        values = asdict(self)
        values["diagnostic"] = self.diagnostic.to_dict()
        return values


def build_reid_pipeline(config: ReIDDiagnosticConfig) -> ReIDCameraPipeline:
    """Construct the production Stage 3 + Stage 4 pipeline."""
    detector = YoloPersonDetector(
        YoloConfig(
            model=config.yolo_model,
            confidence=config.confidence,
            image_size=config.image_size,
            device=config.device,
        )
    )
    tracker = ByteTrackLocalTracker(config.camera_id, ByteTrackConfig())
    stage3 = SingleCameraPipeline(
        detector=detector,
        tracker=tracker,
        crop_config=CropConfig(),
    )
    encoder = ReIDEncoder.from_checkpoint(
        config.checkpoint,
        ReIDEncoderConfig(device=config.device),
    )
    tracklets = TrackletEmbeddingStore(
        encoder,
        TrackletEmbeddingConfig(
            history_size=config.history_size,
            embedding_interval_frames=config.embedding_interval_frames,
            stale_after_frames=config.stale_after_frames,
        ),
    )
    return ReIDCameraPipeline(stage3=stage3, tracklets=tracklets)


def _save_embedding_artifact(path: Path, store: TrackletEmbeddingStore) -> None:
    appearances = store.all()
    histories = store.histories()
    if not appearances:
        raise ValueError("No tracklet embeddings are available to save")

    sample_keys = [
        key
        for key in sorted(histories)
        for _ in range(histories[key].shape[0])
    ]
    sample_embeddings = np.concatenate([histories[key] for key in sorted(histories)])
    np.savez_compressed(
        path,
        tracklet_embeddings=np.stack([item.embedding for item in appearances]),
        tracklet_camera_ids=np.asarray([item.key.camera_id for item in appearances]),
        tracklet_local_ids=np.asarray([item.key.local_id for item in appearances], dtype=np.int64),
        tracklet_stored_samples=np.asarray(
            [item.stored_samples for item in appearances], dtype=np.int64
        ),
        sample_embeddings=sample_embeddings,
        sample_camera_ids=np.asarray([key.camera_id for key in sample_keys]),
        sample_local_ids=np.asarray([key.local_id for key in sample_keys], dtype=np.int64),
    )


def run_reid_diagnostic(
    config: ReIDDiagnosticConfig,
    *,
    pipeline: ReIDCameraPipeline | None = None,
) -> ReIDDiagnosticSummary:
    """Run Stage 3 and Stage 4, then test temporal appearance separation."""
    active_pipeline = pipeline or build_reid_pipeline(config)
    frames_processed = 0
    started = time.perf_counter()

    with VideoFileSource(config.source, config.camera_id) as source:
        try:
            for packet in source:
                if config.max_frames is not None and frames_processed >= config.max_frames:
                    break
                result = active_pipeline.process(packet)
                frames_processed += 1
                if config.display:
                    displayed = draw_reid_status(
                        result.stage3.annotated_frame,
                        result.appearances,
                    )
                    cv2.imshow(f"Stage 4 Re-ID diagnostic - {config.camera_id}", displayed)
                    if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                        break
        finally:
            if config.display:
                cv2.destroyAllWindows()

    if frames_processed == 0:
        raise ValueError("Video source did not yield any frames")
    histories = active_pipeline.tracklets.histories()
    diagnostic = evaluate_tracklet_separation(histories)
    elapsed = time.perf_counter() - started

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "tracklet_embeddings.npz"
    _save_embedding_artifact(artifact_path, active_pipeline.tracklets)
    appearances = active_pipeline.tracklets.all()
    summary = ReIDDiagnosticSummary(
        source=str(config.source.expanduser().resolve()),
        checkpoint=str(config.checkpoint.expanduser().resolve()),
        camera_id=config.camera_id,
        frames_processed=frames_processed,
        tracklets=len(appearances),
        embedding_samples=sum(item.total_samples for item in appearances),
        embedding_dimension=active_pipeline.tracklets.embedding_dimension or 0,
        elapsed_seconds=elapsed,
        processing_fps=frames_processed / elapsed,
        diagnostic=diagnostic,
        embedding_artifact=str(artifact_path),
    )
    (output_dir / "diagnostic.json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone Stage 4 diagnostic parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO, ByteTrack, and OSNet on a video containing at least two "
            "stable people, then compare same-track and different-track similarities."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--camera-id", default="C1")
    parser.add_argument("--model", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--embedding-interval", type=int, default=5)
    parser.add_argument("--stale-after-frames", type=int, default=150)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--display", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone diagnostic and return nonzero when its exit gate fails."""
    args = build_parser().parse_args(argv)
    summary = run_reid_diagnostic(
        ReIDDiagnosticConfig(
            source=args.source,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            camera_id=args.camera_id,
            yolo_model=args.model,
            device=args.device,
            confidence=args.confidence,
            image_size=args.image_size,
            history_size=args.history_size,
            embedding_interval_frames=args.embedding_interval,
            stale_after_frames=args.stale_after_frames,
            max_frames=args.max_frames,
            display=args.display,
        )
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0 if summary.diagnostic.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
