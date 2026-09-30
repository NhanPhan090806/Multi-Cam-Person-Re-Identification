from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from multicam_reid.detection.yolo import DetectionBatch, YoloConfig, YoloPersonDetector
from multicam_reid.pipeline.single_camera import (
    CropConfig,
    ProcessedFrame,
    SingleCameraPipeline,
    extract_person_crops,
)
from multicam_reid.pipeline.video_runner import VideoRunConfig, run_video
from multicam_reid.tracking.bytetrack import ByteTrackConfig, ByteTrackLocalTracker, LocalTrack
from multicam_reid.types import FramePacket
from multicam_reid.visualization.tracks import draw_local_tracks


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class RecordingYoloModel:
    def __init__(self, boxes: object) -> None:
        self.boxes = boxes
        self.calls: list[dict[str, object]] = []

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[object]:
        self.calls.append({"frame": frame, **kwargs})
        return [SimpleNamespace(boxes=self.boxes)]


class RecordingByteTracker:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.inputs: list[tuple[DetectionBatch, np.ndarray]] = []
        self.was_reset = False

    def update(self, detections: DetectionBatch, img: np.ndarray) -> np.ndarray:
        self.inputs.append((detections, img))
        return self.output

    def reset(self) -> None:
        self.was_reset = True


def make_track(
    local_id: int,
    box: tuple[float, float, float, float],
    *,
    confidence: float = 0.9,
) -> LocalTrack:
    return LocalTrack(
        camera_id="C1",
        local_id=local_id,
        xyxy=box,
        confidence=confidence,
        class_id=0,
        detection_index=0,
    )


def test_detection_batch_calculates_xywh_and_supports_boolean_indexing() -> None:
    detections = DetectionBatch(
        xyxy=np.asarray([[10, 20, 30, 60], [2, 4, 8, 12]], dtype=np.float32),
        conf=np.asarray([0.9, 0.4], dtype=np.float32),
        cls=np.asarray([0, 0], dtype=np.float32),
    )

    np.testing.assert_allclose(detections.xywh[0], [20, 40, 20, 40])
    filtered = detections[np.asarray([True, False])]

    assert len(filtered) == 1
    np.testing.assert_array_equal(filtered.xyxy[0], [10, 20, 30, 60])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xyxy": np.zeros((2, 5), dtype=np.float32)},
        {"conf": np.zeros(3, dtype=np.float32)},
        {"cls": np.zeros(3, dtype=np.float32)},
    ],
)
def test_detection_batch_rejects_invalid_shapes(kwargs: dict[str, np.ndarray]) -> None:
    values = {
        "xyxy": np.zeros((2, 4), dtype=np.float32),
        "conf": np.zeros(2, dtype=np.float32),
        "cls": np.zeros(2, dtype=np.float32),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match="shape"):
        DetectionBatch(**values)


def test_yolo_person_detector_returns_numpy_person_detections() -> None:
    boxes = SimpleNamespace(
        xyxy=FakeTensor(np.asarray([[1, 2, 30, 70]], dtype=np.float32)),
        conf=FakeTensor(np.asarray([0.88], dtype=np.float32)),
        cls=FakeTensor(np.asarray([0], dtype=np.float32)),
    )
    model = RecordingYoloModel(boxes)
    detector = YoloPersonDetector(
        YoloConfig(device="cpu", confidence=0.3, image_size=320),
        model=model,
    )
    frame = np.zeros((80, 40, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 1
    np.testing.assert_array_equal(detections.xyxy[0], [1, 2, 30, 70])
    assert model.calls[0]["classes"] == [0]
    assert model.calls[0]["device"] == "cpu"
    assert model.calls[0]["verbose"] is False


def test_yolo_person_detector_handles_no_boxes() -> None:
    detector = YoloPersonDetector(YoloConfig(), model=RecordingYoloModel(None))

    detections = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(detections) == 0
    assert detections.xyxy.shape == (0, 4)


def test_bytetrack_wrapper_converts_output_to_local_tracks() -> None:
    raw_tracks = np.asarray([[1, 2, 31, 72, 7, 0.88, 0, 0]], dtype=np.float32)
    backend = RecordingByteTracker(raw_tracks)
    tracker = ByteTrackLocalTracker(
        camera_id="C1",
        config=ByteTrackConfig(track_buffer=20),
        tracker=backend,
    )
    frame = np.zeros((80, 40, 3), dtype=np.uint8)
    detections = DetectionBatch(
        xyxy=np.asarray([[1, 2, 31, 72]], dtype=np.float32),
        conf=np.asarray([0.88], dtype=np.float32),
        cls=np.asarray([0], dtype=np.float32),
    )

    tracks = tracker.update(detections, frame)

    assert len(tracks) == 1
    assert tracks[0].camera_id == "C1"
    assert tracks[0].local_id == 7
    assert tracks[0].xyxy == (1.0, 2.0, 31.0, 72.0)
    assert tracks[0].confidence == pytest.approx(0.88)
    assert backend.inputs[0][0] is detections
    assert backend.inputs[0][1] is frame
    tracker.reset()
    assert backend.was_reset is True


def test_extract_person_crops_clips_boxes_and_rejects_tiny_tracks() -> None:
    frame = np.arange(60 * 40 * 3, dtype=np.uint8).reshape(60, 40, 3)
    packet = FramePacket(camera_id="C1", frame_id=3, timestamp=0.1, frame=frame)
    tracks = (
        make_track(1, (-4.2, 5.1, 25.2, 58.7)),
        make_track(2, (10.0, 10.0, 13.0, 16.0)),
        make_track(3, (50.0, 10.0, 60.0, 30.0)),
    )

    crops = extract_person_crops(
        packet,
        tracks,
        CropConfig(min_width=8, min_height=16, min_area=128),
    )

    assert len(crops) == 1
    assert crops[0].track.local_id == 1
    assert crops[0].xyxy == (0, 5, 26, 59)
    np.testing.assert_array_equal(crops[0].image, frame[5:59, 0:26])
    assert not np.shares_memory(crops[0].image, frame)


def test_draw_local_tracks_returns_annotated_copy() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    annotated = draw_local_tracks(frame, (make_track(3, (10, 10, 50, 70)),), "C1")

    assert annotated is not frame
    assert np.any(annotated != frame)
    assert not np.any(frame)


class FixedDetector:
    def __init__(self, detections: DetectionBatch) -> None:
        self.detections = detections
        self.frames: list[np.ndarray] = []

    def detect(self, frame: np.ndarray) -> DetectionBatch:
        self.frames.append(frame)
        return self.detections


class FixedTracker:
    camera_id = "C1"

    def __init__(self, tracks: tuple[LocalTrack, ...]) -> None:
        self.tracks = tracks
        self.calls: list[tuple[DetectionBatch, np.ndarray]] = []

    def update(
        self,
        detections: DetectionBatch,
        frame: np.ndarray,
    ) -> tuple[LocalTrack, ...]:
        self.calls.append((detections, frame))
        return self.tracks


def test_single_camera_pipeline_connects_detection_tracking_crops_and_rendering() -> None:
    detections = DetectionBatch(
        xyxy=np.asarray([[10, 5, 40, 75]], dtype=np.float32),
        conf=np.asarray([0.9], dtype=np.float32),
        cls=np.asarray([0], dtype=np.float32),
    )
    tracks = (make_track(4, (10, 5, 40, 75)),)
    detector = FixedDetector(detections)
    tracker = FixedTracker(tracks)
    pipeline = SingleCameraPipeline(detector=detector, tracker=tracker)
    packet = FramePacket(
        camera_id="C1",
        frame_id=1,
        timestamp=0.04,
        frame=np.zeros((80, 60, 3), dtype=np.uint8),
    )

    result = pipeline.process(packet)

    assert isinstance(result, ProcessedFrame)
    assert result.packet is packet
    assert result.detections is detections
    assert result.tracks == tracks
    assert len(result.crops) == 1
    assert result.crops[0].image.shape == (70, 30, 3)
    assert np.any(result.annotated_frame != packet.frame)


def test_pipeline_rejects_packet_from_another_camera() -> None:
    pipeline = SingleCameraPipeline(
        detector=FixedDetector(DetectionBatch.empty()),
        tracker=FixedTracker(()),
    )
    packet = FramePacket(
        camera_id="C2",
        frame_id=0,
        timestamp=0.0,
        frame=np.zeros((10, 10, 3), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="C1"):
        pipeline.process(packet)


class PassthroughPipeline:
    def __init__(self) -> None:
        self.frame_ids: list[int] = []

    def process(self, packet: FramePacket) -> ProcessedFrame:
        self.frame_ids.append(packet.frame_id)
        detections = DetectionBatch.empty()
        return ProcessedFrame(
            packet=packet,
            detections=detections,
            tracks=(),
            crops=(),
            annotated_frame=packet.frame.copy(),
        )


def test_video_runner_processes_frames_and_writes_summary(tmp_path: Path) -> None:
    source_path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for value in (20, 60, 100):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    pipeline = PassthroughPipeline()
    output_path = tmp_path / "annotated.avi"

    summary = run_video(
        VideoRunConfig(
            source=source_path,
            output=output_path,
            camera_id="C1",
            codec="MJPG",
        ),
        pipeline,
    )

    assert pipeline.frame_ids == [0, 1, 2]
    assert summary.frames_processed == 3
    assert summary.total_detections == 0
    assert summary.total_track_observations == 0
    assert summary.total_valid_crops == 0
    assert summary.processing_fps > 0
    assert output_path.is_file()
    assert output_path.with_suffix(".json").is_file()


def test_video_runner_honors_max_frames(tmp_path: Path) -> None:
    source_path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (32, 24),
    )
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    pipeline = PassthroughPipeline()

    summary = run_video(
        VideoRunConfig(
            source=source_path,
            output=tmp_path / "two_frames.avi",
            max_frames=2,
            codec="MJPG",
        ),
        pipeline,
    )

    assert summary.frames_processed == 2
    assert pipeline.frame_ids == [0, 1]


@pytest.mark.parametrize(
    ("config_type", "kwargs", "message"),
    [
        (YoloConfig, {"confidence": 0.0}, "confidence"),
        (YoloConfig, {"device": "tpu"}, "device"),
        (ByteTrackConfig, {"track_low_thresh": 0.5, "track_high_thresh": 0.2}, "threshold"),
        (CropConfig, {"min_area": 0}, "min_area"),
        (VideoRunConfig, {"source": Path("x"), "max_frames": 0}, "max_frames"),
    ],
)
def test_stage3_configs_reject_invalid_values(
    config_type: type[object],
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config_type(**kwargs)
