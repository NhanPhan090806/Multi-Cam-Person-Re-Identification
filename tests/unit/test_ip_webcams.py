from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from multicam_reid.detection.yolo import DetectionBatch
from multicam_reid.inputs.ip_camera import (
    IpCameraConfig,
    IpCameraStatus,
    IpCameraWorker,
    load_ip_camera_configs,
)
from multicam_reid.pipeline.ip_webcams import (
    IpWebcamRunConfig,
    build_camera_pipelines,
    run_ip_webcams,
)
from multicam_reid.pipeline.single_camera import ProcessedFrame
from multicam_reid.types import FramePacket


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, opened: bool = True) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def set(self, *_args: object) -> bool:
        return True

    def release(self) -> None:
        self.released = True


class FakeDetector:
    pass


class FakeTracker:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id


class FakePipeline:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.packets: list[FramePacket] = []

    def process(self, packet: FramePacket) -> ProcessedFrame:
        self.packets.append(packet)
        return ProcessedFrame(
            packet=packet,
            detections=DetectionBatch.empty(),
            tracks=(),
            crops=(),
            annotated_frame=packet.frame.copy(),
        )


class FakeWorker:
    def __init__(self, config: IpCameraConfig, packets: list[FramePacket]) -> None:
        self.config = config
        self._packets = list(packets)
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def snapshot(self, after_frame_id: int = -1) -> FramePacket | None:
        if self._packets and self._packets[0].frame_id > after_frame_id:
            return self._packets.pop(0)
        return None

    def status(self) -> IpCameraStatus:
        return IpCameraStatus(
            camera_id=self.config.camera_id,
            connected=True,
            frames_received=1,
            last_error=None,
        )

    def stop(self) -> None:
        self.stopped = True


def test_ip_camera_config_validates_url_and_window_geometry() -> None:
    config = IpCameraConfig(
        camera_id="C1",
        url="http://192.168.1.20:8080/video",
        window_width=800,
        window_height=450,
        window_x=100,
        window_y=50,
    )

    assert config.window_size == (800, 450)
    assert config.window_position == (100, 50)

    with pytest.raises(ValueError, match="scheme"):
        IpCameraConfig(camera_id="C1", url="192.168.1.20:8080/video")
    with pytest.raises(ValueError, match="window_width"):
        IpCameraConfig(camera_id="C1", url="http://example/video", window_width=0)


def test_load_ip_camera_configs_filters_disabled_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "cameras.yaml"
    config_path.write_text(
        """
cameras:
  - camera_id: C1
    url: http://192.168.1.20:8080/video
    enabled: true
    window_width: 640
    window_height: 360
  - camera_id: C2
    url: http://192.168.1.21:8080/video
    enabled: false
""".strip(),
        encoding="utf-8",
    )

    configs = load_ip_camera_configs(config_path)

    assert [config.camera_id for config in configs] == ["C1"]
    assert configs[0].window_size == (640, 360)


def test_load_ip_camera_configs_rejects_duplicate_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text(
        """
cameras:
  - {camera_id: C1, url: http://one/video}
  - {camera_id: C1, url: http://two/video}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_ip_camera_configs(config_path)


def test_ip_camera_worker_publishes_latest_frame_and_stops() -> None:
    frame = np.full((24, 32, 3), 80, dtype=np.uint8)
    captures: list[FakeCapture] = []

    def capture_factory(_url: str) -> FakeCapture:
        capture = FakeCapture([frame.copy()])
        captures.append(capture)
        return capture

    worker = IpCameraWorker(
        IpCameraConfig(
            camera_id="C1",
            url="http://camera/video",
            reconnect_delay_seconds=0.01,
        ),
        capture_factory=capture_factory,
    )

    worker.start()
    deadline = time.monotonic() + 1.0
    packet = None
    while packet is None and time.monotonic() < deadline:
        packet = worker.snapshot()
        time.sleep(0.005)
    worker.stop()

    assert packet is not None
    assert packet.camera_id == "C1"
    assert packet.frame_id >= 0
    np.testing.assert_array_equal(packet.frame, frame)
    assert captures
    assert all(capture.released for capture in captures)
    assert worker.status().frames_received >= 1


def test_build_camera_pipelines_shares_detector_but_separates_trackers() -> None:
    configs = (
        IpCameraConfig(camera_id="C1", url="http://one/video"),
        IpCameraConfig(camera_id="C2", url="http://two/video"),
    )
    detector = FakeDetector()
    trackers: list[FakeTracker] = []

    def tracker_factory(camera_id: str) -> FakeTracker:
        tracker = FakeTracker(camera_id)
        trackers.append(tracker)
        return tracker

    pipelines = build_camera_pipelines(
        configs,
        detector,
        tracker_factory=tracker_factory,
    )

    assert set(pipelines) == {"C1", "C2"}
    assert pipelines["C1"].detector is detector
    assert pipelines["C2"].detector is detector
    assert pipelines["C1"].tracker is trackers[0]
    assert pipelines["C2"].tracker is trackers[1]
    assert trackers[0] is not trackers[1]


def test_run_ip_webcams_processes_each_camera_and_stops_workers() -> None:
    configs = (
        IpCameraConfig(camera_id="C1", url="http://one/video"),
        IpCameraConfig(camera_id="C2", url="http://two/video"),
    )
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    pipelines = {camera.camera_id: FakePipeline(camera.camera_id) for camera in configs}
    workers = {
        camera.camera_id: FakeWorker(
            camera,
            [
                FramePacket(
                    camera_id=camera.camera_id,
                    frame_id=0,
                    timestamp=0.0,
                    frame=frame.copy(),
                )
            ],
        )
        for camera in configs
    }

    summary = run_ip_webcams(
        configs,
        pipelines=pipelines,
        workers=workers,
        run_config=IpWebcamRunConfig(
            display=False,
            max_frames_per_camera=1,
            startup_timeout_seconds=0.5,
            poll_interval_seconds=0.001,
        ),
    )

    assert summary.frames_by_camera == {"C1": 1, "C2": 1}
    assert len(pipelines["C1"].packets) == 1
    assert len(pipelines["C2"].packets) == 1
    assert all(worker.started and worker.stopped for worker in workers.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"startup_timeout_seconds": 0.0},
        {"poll_interval_seconds": -1.0},
        {"max_frames_per_camera": 0},
    ],
)
def test_ip_webcam_run_config_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        IpWebcamRunConfig(**kwargs)
