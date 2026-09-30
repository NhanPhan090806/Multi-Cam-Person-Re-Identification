from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from multicam_reid.detection.yolo import DetectionBatch
from multicam_reid.pipeline.reid_camera import ReIDCameraPipeline, ReIDProcessedFrame
from multicam_reid.pipeline.reid_diagnostic import (
    ReIDDiagnosticConfig,
    run_reid_diagnostic,
)
from multicam_reid.pipeline.single_camera import PersonCrop, ProcessedFrame
from multicam_reid.reid.tracklets import (
    TrackKey,
    TrackletEmbeddingConfig,
    TrackletEmbeddingStore,
    evaluate_tracklet_separation,
)
from multicam_reid.tracking.bytetrack import LocalTrack
from multicam_reid.types import FramePacket


class PixelEncoder:
    """Map a crop's first two pixel values to a small deterministic embedding."""

    def __init__(self) -> None:
        self.calls: list[list[np.ndarray]] = []

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        self.calls.append(crops)
        return np.asarray(
            [[float(crop[0, 0, 0]), float(crop[0, 0, 1])] for crop in crops],
            dtype=np.float32,
        )


def make_crop(
    local_id: int,
    frame_id: int,
    vector: tuple[int, int],
    *,
    camera_id: str = "C1",
) -> PersonCrop:
    image = np.zeros((32, 16, 3), dtype=np.uint8)
    image[0, 0, :2] = vector
    track = LocalTrack(
        camera_id=camera_id,
        local_id=local_id,
        xyxy=(0.0, 0.0, 16.0, 32.0),
        confidence=0.9,
        class_id=0,
        detection_index=0,
    )
    return PersonCrop(
        track=track,
        frame_id=frame_id,
        timestamp=frame_id / 10.0,
        xyxy=(0, 0, 16, 32),
        image=image,
    )


def test_store_batches_due_crops_and_normalizes_tracklet_average() -> None:
    encoder = PixelEncoder()
    store = TrackletEmbeddingStore(
        encoder,
        TrackletEmbeddingConfig(history_size=3, embedding_interval_frames=1),
    )

    first = store.update((make_crop(1, 0, (10, 0)), make_crop(2, 0, (0, 10))))
    second = store.update((make_crop(1, 1, (6, 8)),))

    assert len(encoder.calls) == 2
    assert len(encoder.calls[0]) == 2
    assert {item.key for item in first} == {TrackKey("C1", 1), TrackKey("C1", 2)}
    track_one = second[0]
    expected = np.asarray([0.8, 0.4], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(track_one.embedding, expected, atol=1e-6)
    assert track_one.stored_samples == 2
    assert track_one.total_samples == 2


def test_store_samples_periodically_and_bounds_history() -> None:
    encoder = PixelEncoder()
    store = TrackletEmbeddingStore(
        encoder,
        TrackletEmbeddingConfig(history_size=2, embedding_interval_frames=2),
    )

    for frame_id, vector in enumerate(((10, 0), (0, 10), (6, 8), (8, 6), (0, 10))):
        store.update((make_crop(1, frame_id, vector),))

    history = store.history(TrackKey("C1", 1))
    assert len(encoder.calls) == 3
    assert history.shape == (2, 2)
    np.testing.assert_allclose(history[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(history[1], [0.0, 1.0], atol=1e-6)
    appearance = store.get(TrackKey("C1", 1))
    assert appearance is not None
    assert appearance.stored_samples == 2
    assert appearance.total_samples == 3
    assert appearance.last_seen_frame_id == 4


def test_store_keeps_same_local_id_separate_between_cameras_and_prunes_stale() -> None:
    store = TrackletEmbeddingStore(
        PixelEncoder(),
        TrackletEmbeddingConfig(stale_after_frames=2),
    )
    store.update(
        (
            make_crop(1, 0, (10, 0), camera_id="C1"),
            make_crop(1, 0, (0, 10), camera_id="C2"),
        )
    )

    assert len(store) == 2
    removed = store.prune(current_frame_id=3)

    assert set(removed) == {TrackKey("C1", 1), TrackKey("C2", 1)}
    assert len(store) == 0


@pytest.mark.parametrize(
    "bad_embeddings",
    [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([[np.nan, 0.0]], dtype=np.float32),
    ],
)
def test_store_rejects_invalid_encoder_output(bad_embeddings: np.ndarray) -> None:
    class BadEncoder:
        def encode(self, _crops: list[np.ndarray]) -> np.ndarray:
            return bad_embeddings

    store = TrackletEmbeddingStore(BadEncoder())

    with pytest.raises((ValueError, RuntimeError), match="embedding|Embedding"):
        store.update((make_crop(1, 0, (10, 0)),))


def test_tracklet_diagnostic_compares_same_and_different_tracks() -> None:
    histories = {
        TrackKey("C1", 1): np.asarray([[1.0, 0.0], [0.99, 0.1]], dtype=np.float32),
        TrackKey("C1", 2): np.asarray([[0.0, 1.0], [0.1, 0.99]], dtype=np.float32),
    }

    diagnostic = evaluate_tracklet_separation(histories)

    assert diagnostic.same_track_pairs == 2
    assert diagnostic.different_track_pairs == 4
    assert diagnostic.same_track_mean > diagnostic.different_track_mean
    assert diagnostic.passed is True


class FixedStage3Pipeline:
    def __init__(self, result: ProcessedFrame) -> None:
        self.result = result
        self.packets: list[FramePacket] = []

    def process(self, packet: FramePacket) -> ProcessedFrame:
        self.packets.append(packet)
        return replace(self.result, packet=packet)


def test_reid_camera_pipeline_consumes_stage3_crops() -> None:
    packet = FramePacket(
        camera_id="C1",
        frame_id=0,
        timestamp=0.0,
        frame=np.zeros((40, 20, 3), dtype=np.uint8),
    )
    crop = make_crop(1, 0, (10, 0))
    stage3_result = ProcessedFrame(
        packet=packet,
        detections=DetectionBatch.empty(),
        tracks=(crop.track,),
        crops=(crop,),
        annotated_frame=packet.frame.copy(),
    )
    stage3 = FixedStage3Pipeline(stage3_result)
    store = TrackletEmbeddingStore(PixelEncoder())
    pipeline = ReIDCameraPipeline(stage3=stage3, tracklets=store)

    result = pipeline.process(packet)

    assert isinstance(result, ReIDProcessedFrame)
    assert result.stage3.packet is packet
    assert result.appearances[0].key == TrackKey("C1", 1)
    assert store.get(TrackKey("C1", 1)) is not None


class TwoPersonStage3Pipeline:
    def process(self, packet: FramePacket) -> ProcessedFrame:
        vectors = (
            ((10, 0), (0, 10)),
            ((10, 1), (1, 10)),
            ((9, 1), (1, 9)),
        )
        crops = tuple(
            make_crop(local_id, packet.frame_id, vector)
            for local_id, vector in enumerate(vectors[packet.frame_id], start=1)
        )
        return ProcessedFrame(
            packet=packet,
            detections=DetectionBatch.empty(),
            tracks=tuple(crop.track for crop in crops),
            crops=crops,
            annotated_frame=packet.frame.copy(),
        )


def test_reid_diagnostic_runner_writes_pickle_free_artifacts(tmp_path: Path) -> None:
    source_path = tmp_path / "two_people.avi"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 32),
    )
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    pipeline = ReIDCameraPipeline(
        stage3=TwoPersonStage3Pipeline(),
        tracklets=TrackletEmbeddingStore(
            PixelEncoder(),
            TrackletEmbeddingConfig(embedding_interval_frames=1),
        ),
    )
    output_dir = tmp_path / "diagnostic"

    summary = run_reid_diagnostic(
        ReIDDiagnosticConfig(
            source=source_path,
            output_dir=output_dir,
            checkpoint=Path("unused-in-injected-test.pt"),
            max_frames=3,
        ),
        pipeline=pipeline,
    )

    assert summary.frames_processed == 3
    assert summary.tracklets == 2
    assert summary.embedding_dimension == 2
    assert summary.diagnostic.passed is True
    assert json.loads((output_dir / "diagnostic.json").read_text())["diagnostic"]["passed"]
    with np.load(output_dir / "tracklet_embeddings.npz", allow_pickle=False) as artifact:
        assert artifact["tracklet_embeddings"].shape == (2, 2)
        assert artifact["sample_embeddings"].shape == (6, 2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"history_size": 0}, "history_size"),
        ({"embedding_interval_frames": 0}, "embedding_interval_frames"),
        ({"stale_after_frames": -1}, "stale_after_frames"),
    ],
)
def test_tracklet_config_rejects_invalid_values(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TrackletEmbeddingConfig(**kwargs)
