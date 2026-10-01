from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from multicam_reid.association import AssociationConfig, GlobalIdentityRegistry
from multicam_reid.detection.yolo import DetectionBatch
from multicam_reid.inputs.ip_camera import IpCameraConfig, IpCameraStatus
from multicam_reid.inputs.webcam import WebcamConfig, WebcamWorker
from multicam_reid.pipeline.camera_demo import (
    CameraDemoConfig,
    DemoDisplayConfig,
    DemoModelConfig,
    DemoRuntimeConfig,
    IdentityDisplay,
    ScreenshotController,
    TransientNotice,
    apply_source_overrides,
    build_live_pipelines,
    build_live_workers,
    load_camera_demo_config,
    main,
    render_dashboard,
    run_camera_demo,
    save_dashboard_screenshot,
)
from multicam_reid.pipeline.reid_camera import ReIDProcessedFrame
from multicam_reid.pipeline.single_camera import ProcessedFrame
from multicam_reid.reid.tracklets import TrackKey, TrackletAppearance
from multicam_reid.tracking.bytetrack import LocalTrack
from multicam_reid.types import FramePacket


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, opened: bool = True) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.released = False
        self.settings: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def set(self, prop: int, value: float) -> bool:
        self.settings.append((prop, value))
        return True

    def release(self) -> None:
        self.released = True


class FixedPipeline:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id

    def process(self, packet: FramePacket) -> ReIDProcessedFrame:
        track = LocalTrack(
            camera_id=self.camera_id,
            local_id=1,
            xyxy=(4.0, 3.0, 28.0, 45.0),
            confidence=0.9,
            class_id=0,
            detection_index=0,
        )
        stage3 = ProcessedFrame(
            packet=packet,
            detections=DetectionBatch.empty(),
            tracks=(track,),
            crops=(),
            annotated_frame=packet.frame.copy(),
        )
        appearance = TrackletAppearance(
            key=TrackKey(self.camera_id, 1),
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            stored_samples=2,
            total_samples=2,
            first_seen_frame_id=0,
            last_seen_frame_id=packet.frame_id,
            first_seen_timestamp=0.0,
            last_seen_timestamp=packet.timestamp,
        )
        return ReIDProcessedFrame(stage3=stage3, appearances=(appearance,))


class FakeWorker:
    def __init__(self, camera_id: str, packet: FramePacket) -> None:
        self.camera_id = camera_id
        self.packet = packet
        self.consumed = False
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def snapshot(self, after_frame_id: int = -1) -> FramePacket | None:
        if not self.consumed and self.packet.frame_id > after_frame_id:
            self.consumed = True
            return self.packet
        return None

    def status(self) -> IpCameraStatus:
        return IpCameraStatus(self.camera_id, True, 1, None)

    def stop(self) -> None:
        self.stopped = True


class EmptyWorker(FakeWorker):
    def snapshot(self, after_frame_id: int = -1) -> FramePacket | None:
        return None


def _write_demo_config(path: Path) -> None:
    path.write_text(
        """
active_mode: solo
models:
  detector_model: yolo26n.pt
  checkpoint: outputs/model.pth.tar-60
  device: cpu
runtime:
  output_dir: outputs/stage8/test
  minimum_active_cameras: 2
display:
  enabled: false
  tile_width: 320
  tile_height: 180
modes:
  solo:
    cameras:
      - kind: ip
        camera_id: PHONE
        url: http://192.168.1.10:8080/video
      - kind: webcam
        camera_id: LAPTOP
        device_index: 0
  multi_ip:
    cameras:
      - kind: ip
        camera_id: PHONE_A
        url: http://192.168.1.10:8080/video
      - kind: ip
        camera_id: PHONE_B
        url: http://192.168.1.11:8080/video
""".strip(),
        encoding="utf-8",
    )


def test_webcam_worker_publishes_latest_frame_and_stops() -> None:
    frame = np.full((24, 32, 3), 90, dtype=np.uint8)
    captures: list[FakeCapture] = []

    def capture_factory(_index: int, _backend: int | None) -> FakeCapture:
        capture = FakeCapture([frame.copy()])
        captures.append(capture)
        return capture

    worker = WebcamWorker(
        WebcamConfig(
            camera_id="LAPTOP",
            device_index=0,
            reconnect_delay_seconds=0.01,
            capture_width=640,
            capture_height=360,
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
    assert packet.camera_id == "LAPTOP"
    np.testing.assert_array_equal(packet.frame, frame)
    assert captures and all(capture.released for capture in captures)
    assert worker.status().frames_received >= 1


def test_load_camera_demo_config_supports_solo_and_multi_ip(tmp_path: Path) -> None:
    path = tmp_path / "camera_demo.yaml"
    _write_demo_config(path)

    solo = load_camera_demo_config(path)
    multi = load_camera_demo_config(path, mode="multi_ip")

    assert solo.mode == "solo"
    assert isinstance(solo.cameras[0], IpCameraConfig)
    assert isinstance(solo.cameras[1], WebcamConfig)
    assert multi.mode == "multi_ip"
    assert len(multi.cameras) == 2
    assert all(isinstance(camera, IpCameraConfig) for camera in multi.cameras)


def test_source_overrides_do_not_mutate_original_config(tmp_path: Path) -> None:
    path = tmp_path / "camera_demo.yaml"
    _write_demo_config(path)
    config = load_camera_demo_config(path)

    updated = apply_source_overrides(
        config,
        camera_urls=("PHONE=http://10.0.0.8:8080/video",),
        webcam_index=2,
    )

    assert isinstance(updated.cameras[0], IpCameraConfig)
    assert updated.cameras[0].url == "http://10.0.0.8:8080/video"
    assert isinstance(updated.cameras[1], WebcamConfig)
    assert updated.cameras[1].device_index == 2
    assert isinstance(config.cameras[0], IpCameraConfig)
    assert config.cameras[0].url == "http://192.168.1.10:8080/video"


def test_mode_contract_rejects_wrong_camera_mix() -> None:
    with pytest.raises(ValueError, match="one IP camera and one local webcam"):
        CameraDemoConfig(
            mode="solo",
            cameras=(
                IpCameraConfig("A", "http://one/video"),
                IpCameraConfig("B", "http://two/video"),
            ),
        )


def test_dashboard_has_clickable_screenshot_button_and_saves_unicode_path(
    tmp_path: Path,
) -> None:
    controller = ScreenshotController()
    frames = {
        "PHONE": np.full((90, 160, 3), 30, dtype=np.uint8),
        "LAPTOP": np.full((90, 160, 3), 90, dtype=np.uint8),
    }
    dashboard = render_dashboard(
        ("PHONE", "LAPTOP"),
        frames,
        {"PHONE": "connected", "LAPTOP": "connected"},
        {
            "PHONE": (IdentityDisplay(global_id=7, local_id=3, confidence=0.91),),
            "LAPTOP": (IdentityDisplay(global_id=7, local_id=8, confidence=0.88),),
        },
        DemoDisplayConfig(
            enabled=True,
            tile_width=160,
            tile_height=90,
            info_panel_height=90,
            columns=2,
        ),
        controller=controller,
    )
    x1, y1, x2, y2 = controller.button_bounds

    controller.handle_mouse(cv2.EVENT_LBUTTONUP, (x1 + x2) // 2, (y1 + y2) // 2, 0)
    screenshot = save_dashboard_screenshot(dashboard, tmp_path / "ảnh chụp")

    assert controller.consume_request() is True
    assert screenshot.is_file()
    decoded = cv2.imdecode(np.frombuffer(screenshot.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    np.testing.assert_array_equal(dashboard[:90, :160], frames["PHONE"])
    assert np.any(dashboard[90:180, :160] != 0)


def test_transient_screenshot_notice_lasts_one_second_and_clean_frame_is_saved(
    tmp_path: Path,
) -> None:
    controller = ScreenshotController()
    display = DemoDisplayConfig(
        tile_width=160,
        tile_height=90,
        info_panel_height=90,
    )
    frame = np.full((90, 160, 3), 55, dtype=np.uint8)
    clean_dashboard = render_dashboard(
        ("PHONE",),
        {"PHONE": frame},
        {"PHONE": "connected"},
        {"PHONE": (IdentityDisplay(global_id=2, local_id=4, confidence=0.9),)},
        display,
        controller=controller,
        notification=None,
    )
    expected_ok, expected_jpeg = cv2.imencode(".jpg", clean_dashboard)
    assert expected_ok

    screenshot = save_dashboard_screenshot(clean_dashboard, tmp_path)
    notice = TransientNotice()
    notice.show(f"Screenshot saved: {screenshot.name}", now=10.0, duration_seconds=1.0)

    assert screenshot.read_bytes() == expected_jpeg.tobytes()
    assert notice.message(10.999) == f"Screenshot saved: {screenshot.name}"
    assert notice.message(11.001) is None


def test_run_camera_demo_associates_sources_logs_evidence_and_screenshot(
    tmp_path: Path,
) -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    cameras = (
        IpCameraConfig("PHONE", "http://phone/video"),
        WebcamConfig("LAPTOP", 0),
    )
    config = CameraDemoConfig(
        mode="solo",
        cameras=cameras,
        models=DemoModelConfig(),
        runtime=DemoRuntimeConfig(
            output_dir=tmp_path / "stage8",
            max_frames_per_camera=1,
            startup_timeout_seconds=0.5,
            poll_interval_seconds=0.0,
            minimum_active_cameras=2,
        ),
        display=DemoDisplayConfig(enabled=False, tile_width=160, tile_height=90),
    )
    workers = {
        camera.camera_id: FakeWorker(
            camera.camera_id,
            FramePacket(camera.camera_id, 0, 0.0, frame.copy()),
        )
        for camera in cameras
    }
    controller = ScreenshotController()
    controller.request()

    summary = run_camera_demo(
        config,
        pipelines={camera.camera_id: FixedPipeline(camera.camera_id) for camera in cameras},
        workers=workers,
        registry=GlobalIdentityRegistry(
            AssociationConfig(max_cosine_distance=0.35, min_stored_samples=2)
        ),
        screenshot_controller=controller,
    )

    assert summary.frames_by_camera == {"PHONE": 1, "LAPTOP": 1}
    assert summary.global_identities_final == 1
    assert summary.merge_events == 1
    assert len(summary.screenshots) == 1
    assert Path(summary.screenshots[0]).is_file()
    assert all(worker.started and worker.stopped for worker in workers.values())

    assert json.loads((tmp_path / "stage8" / "summary.json").read_text())["mode"] == "solo"
    assert len((tmp_path / "stage8" / "assignments.jsonl").read_text().splitlines()) == 2


def test_live_default_handoff_policy_waits_then_recovers_id_and_records_raw_session(
    tmp_path: Path,
) -> None:
    cameras = (IpCameraConfig("PHONE", "http://phone/video"), WebcamConfig("LAPTOP", 0))
    config = CameraDemoConfig(
        mode="solo",
        cameras=cameras,
        models=DemoModelConfig(exit_grace_seconds=0.0),
        runtime=DemoRuntimeConfig(
            output_dir=tmp_path / "live",
            record_dir=tmp_path / "recordings",
            max_frames_per_camera=5,
            poll_interval_seconds=0.0,
            association_window_seconds=10.0,
        ),
        display=DemoDisplayConfig(enabled=False),
    )

    class ScheduleWorker(FakeWorker):
        def __init__(self, camera: str, markers: list[int]):
            super().__init__(camera, FramePacket(camera, 0, 0.0, np.zeros((48, 64, 3), np.uint8)))
            self.markers = markers
            self.index = 0

        def snapshot(self, after_frame_id=-1):
            if self.index >= len(self.markers):
                return None
            image = np.full((48, 64, 3), self.markers[self.index], np.uint8)
            packet = FramePacket(self.camera_id, self.index, self.index / 10.0, image)
            self.index += 1
            return packet

    class SchedulePipeline(FixedPipeline):
        def process(self, packet):
            marker = int(packet.frame[0, 0, 0])
            if not marker:
                return ReIDProcessedFrame(
                    ProcessedFrame(
                        packet,
                        DetectionBatch.empty(),
                        (),
                        (),
                        packet.frame.copy(),
                    ),
                    (),
                )
            result = super().process(packet)
            item = result.appearances[0]
            from dataclasses import replace

            return ReIDProcessedFrame(
                result.stage3,
                (
                    replace(
                        item,
                        stored_samples=marker,
                        total_samples=marker,
                    ),
                ),
            )

    workers = {
        "PHONE": ScheduleWorker("PHONE", [2, 2, 0, 0, 0]),
        "LAPTOP": ScheduleWorker("LAPTOP", [0, 0, 0, 1, 2]),
    }
    summary = run_camera_demo(
        config,
        pipelines={camera.camera_id: SchedulePipeline(camera.camera_id) for camera in cameras},
        workers=workers,
    )
    assert summary.association_mode == "handoff"
    assert summary.global_ids_issued == 1
    assert summary.handoff_events == 1
    event = json.loads((tmp_path / "live" / "handoff_events.jsonl").read_text())
    assert event["from_camera"] == "PHONE" and event["to_camera"] == "LAPTOP"
    assert summary.recording_dir is not None
    assert len((Path(summary.recording_dir) / "frames.jsonl").read_text().splitlines()) == 10


def test_default_live_policy_keeps_simultaneous_lookalikes_separate(tmp_path: Path) -> None:
    cameras = (IpCameraConfig("PHONE", "http://phone/video"), WebcamConfig("LAPTOP", 0))
    config = CameraDemoConfig(
        mode="solo",
        cameras=cameras,
        runtime=DemoRuntimeConfig(output_dir=tmp_path, max_frames_per_camera=1),
        display=DemoDisplayConfig(enabled=False),
    )
    frame = np.zeros((48, 64, 3), np.uint8)
    summary = run_camera_demo(
        config,
        pipelines={camera.camera_id: FixedPipeline(camera.camera_id) for camera in cameras},
        workers={
            camera.camera_id: FakeWorker(
                camera.camera_id,
                FramePacket(camera.camera_id, 0, 0.0, frame),
            )
            for camera in cameras
        },
    )
    assert summary.global_ids_issued == 2
    assert summary.handoff_events == 0


def test_live_builders_share_models_but_isolate_camera_state() -> None:
    cameras = (
        IpCameraConfig("PHONE", "http://phone/video"),
        WebcamConfig("LAPTOP", 0),
    )

    class FakeTracker:
        def __init__(self, camera_id: str) -> None:
            self.camera_id = camera_id

    detector = object()
    encoder = object()
    pipelines = build_live_pipelines(
        cameras,
        detector,
        encoder,
        tracker_factory=FakeTracker,
    )
    workers = build_live_workers(cameras)

    assert pipelines["PHONE"].stage3.detector is detector
    assert pipelines["LAPTOP"].stage3.detector is detector
    assert pipelines["PHONE"].stage3.tracker is not pipelines["LAPTOP"].stage3.tracker
    assert pipelines["PHONE"].tracklets is not pipelines["LAPTOP"].tracklets
    assert set(workers) == {"PHONE", "LAPTOP"}


def test_surface_main_dry_run_applies_overrides_without_exposing_url(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "camera_demo.yaml"
    _write_demo_config(path)

    exit_code = main(
        [
            "--config",
            str(path),
            "--mode",
            "solo",
            "--camera-url",
            "PHONE=http://10.0.0.9:8080/video",
            "--webcam-index",
            "3",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "output"),
            "--max-runtime-seconds",
            "5",
            "--max-frames-per-camera",
            "10",
            "--no-display",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    public_config = json.loads(output)

    assert exit_code == 0
    assert public_config["mode"] == "solo"
    assert public_config["models"]["device"] == "cpu"
    assert public_config["display"]["enabled"] is False
    assert public_config["runtime"]["max_frames_per_camera"] == 10
    assert "10.0.0.9" not in output


def test_demo_times_out_cleanly_when_not_enough_cameras_start(tmp_path: Path) -> None:
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    cameras = (
        IpCameraConfig("PHONE", "http://phone/video"),
        WebcamConfig("LAPTOP", 0),
    )
    config = CameraDemoConfig(
        mode="solo",
        cameras=cameras,
        runtime=DemoRuntimeConfig(
            output_dir=tmp_path / "timeout",
            startup_timeout_seconds=0.01,
            poll_interval_seconds=0.001,
        ),
        display=DemoDisplayConfig(enabled=False),
    )
    workers = {
        camera.camera_id: EmptyWorker(
            camera.camera_id,
            FramePacket(camera.camera_id, 0, 0.0, frame.copy()),
        )
        for camera in cameras
    }

    with pytest.raises(TimeoutError, match="PHONE, LAPTOP"):
        run_camera_demo(
            config,
            pipelines={camera.camera_id: FixedPipeline(camera.camera_id) for camera in cameras},
            workers=workers,
        )

    assert all(worker.started and worker.stopped for worker in workers.values())
