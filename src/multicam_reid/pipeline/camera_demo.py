"""Stage 8 live two-mode camera demonstration with global person identities."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import yaml

from multicam_reid.association import AssociationConfig, GlobalIdentityRegistry
from multicam_reid.detection.yolo import YoloConfig, YoloPersonDetector
from multicam_reid.inputs.ip_camera import IpCameraConfig, IpCameraWorker
from multicam_reid.inputs.webcam import WebcamConfig, WebcamWorker
from multicam_reid.pipeline.reid_camera import ReIDCameraPipeline, ReIDProcessedFrame
from multicam_reid.pipeline.single_camera import SingleCameraPipeline
from multicam_reid.reid.encoder import DEFAULT_CHECKPOINT, ReIDEncoder, ReIDEncoderConfig
from multicam_reid.reid.tracklets import (
    TrackKey,
    TrackletEmbeddingConfig,
    TrackletEmbeddingStore,
)
from multicam_reid.tracking.bytetrack import ByteTrackLocalTracker
from multicam_reid.types import FramePacket
from multicam_reid.visualization.tracks import draw_global_boxes, global_id_color

DEFAULT_CONFIG_PATH = Path("configs/camera_demo.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/stage8/live_demo")
SUPPORTED_MODES = {"solo", "multi_ip"}
CameraSourceConfig = IpCameraConfig | WebcamConfig


class LivePipeline(Protocol):
    def process(self, packet: FramePacket) -> ReIDProcessedFrame: ...


class LiveWorker(Protocol):
    def start(self) -> None: ...

    def snapshot(self, after_frame_id: int = -1) -> FramePacket | None: ...

    def status(self) -> Any: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DemoModelConfig:
    """Frozen detector, Re-ID, tracklet, and association settings."""

    detector_model: Path = Path("yolo26n.pt")
    checkpoint: Path = DEFAULT_CHECKPOINT
    device: str = "auto"
    confidence: float = 0.1
    image_size: int = 640
    history_size: int = 10
    embedding_interval: int = 5
    max_cosine_distance: float = 0.35
    min_stored_samples: int = 2
    max_idle_updates: int = 250

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        TrackletEmbeddingConfig(
            history_size=self.history_size,
            embedding_interval_frames=self.embedding_interval,
            stale_after_frames=self.max_idle_updates,
        )
        AssociationConfig(
            max_cosine_distance=self.max_cosine_distance,
            min_stored_samples=self.min_stored_samples,
            max_idle_frames=self.max_idle_updates,
        )


@dataclass(frozen=True, slots=True)
class DemoRuntimeConfig:
    """Live scheduling, output, and graceful-degradation settings."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    startup_timeout_seconds: float = 20.0
    poll_interval_seconds: float = 0.005
    association_window_seconds: float = 0.75
    minimum_active_cameras: int = 2
    max_runtime_seconds: float | None = None
    max_frames_per_camera: int | None = None

    def __post_init__(self) -> None:
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must not be negative")
        if self.association_window_seconds <= 0:
            raise ValueError("association_window_seconds must be positive")
        if self.minimum_active_cameras < 2:
            raise ValueError("minimum_active_cameras must be at least two")
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        if self.max_frames_per_camera is not None and self.max_frames_per_camera <= 0:
            raise ValueError("max_frames_per_camera must be positive")


@dataclass(frozen=True, slots=True)
class DemoDisplayConfig:
    """One dashboard layout shared by both Stage 8 source modes."""

    enabled: bool = True
    window_name: str = "Multi-Camera Person Re-Identification"
    tile_width: int = 640
    tile_height: int = 360
    info_panel_height: int = 136
    columns: int = 2
    control_bar_height: int = 54

    def __post_init__(self) -> None:
        if not self.window_name.strip():
            raise ValueError("window_name must not be empty")
        if self.tile_width <= 0 or self.tile_height <= 0:
            raise ValueError("tile dimensions must be positive")
        if self.info_panel_height < 80:
            raise ValueError("info_panel_height must be at least 80")
        if self.columns <= 0:
            raise ValueError("columns must be positive")
        if self.control_bar_height < 40:
            raise ValueError("control_bar_height must be at least 40")


@dataclass(frozen=True, slots=True)
class CameraDemoConfig:
    """Complete Stage 8 session after selecting one configured source mode."""

    mode: str
    cameras: tuple[CameraSourceConfig, ...]
    models: DemoModelConfig = DemoModelConfig()
    runtime: DemoRuntimeConfig = DemoRuntimeConfig()
    display: DemoDisplayConfig = DemoDisplayConfig()

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of: {', '.join(sorted(SUPPORTED_MODES))}")
        camera_ids = [camera.camera_id for camera in self.cameras]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("camera_id values must be unique")
        if self.runtime.minimum_active_cameras > len(self.cameras):
            raise ValueError("minimum_active_cameras exceeds configured cameras")
        ip_count = sum(isinstance(camera, IpCameraConfig) for camera in self.cameras)
        webcam_count = sum(isinstance(camera, WebcamConfig) for camera in self.cameras)
        if self.mode == "solo" and (
            len(self.cameras) != 2 or ip_count != 1 or webcam_count != 1
        ):
            raise ValueError("solo mode requires exactly one IP camera and one local webcam")
        if self.mode == "multi_ip" and (
            len(self.cameras) < 2 or ip_count != len(self.cameras)
        ):
            raise ValueError("multi_ip mode requires at least two IP cameras")

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(camera.camera_id for camera in self.cameras)

    def public_dict(self) -> dict[str, object]:
        """Return reproducible settings without camera URLs or credentials."""
        return {
            "mode": self.mode,
            "cameras": [
                {
                    "camera_id": camera.camera_id,
                    "kind": "ip" if isinstance(camera, IpCameraConfig) else "webcam",
                }
                for camera in self.cameras
            ],
            "models": {
                **asdict(self.models),
                "detector_model": str(self.models.detector_model),
                "checkpoint": str(self.models.checkpoint),
            },
            "runtime": {**asdict(self.runtime), "output_dir": str(self.runtime.output_dir)},
            "display": asdict(self.display),
        }


@dataclass(frozen=True, slots=True)
class CameraDemoSummary:
    """Non-sensitive evidence emitted when a live session ends."""

    mode: str
    camera_ids: tuple[str, ...]
    elapsed_seconds: float
    total_processing_fps: float
    frames_by_camera: dict[str, int]
    association_updates: int
    global_identities_final: int
    global_ids_issued: int
    merge_events: int
    screenshots: tuple[str, ...]
    final_status: dict[str, dict[str, object]]
    output_dir: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IdentityDisplay:
    """One visible track's explicit identity row in a camera's black panel."""

    global_id: int | None
    local_id: int
    confidence: float

    @property
    def text(self) -> str:
        global_text = str(self.global_id) if self.global_id is not None else "pending"
        return f"Global {global_text} | Local {self.local_id} | conf {self.confidence:.2f}"

    @property
    def color(self) -> tuple[int, int, int]:
        if self.global_id is None:
            return 160, 160, 160
        return global_id_color(self.global_id)


class ScreenshotController:
    """Turn mouse clicks or the ``S`` key into one-shot screenshot requests."""

    def __init__(self) -> None:
        self._requested = False
        self.button_bounds = (0, 0, 0, 0)

    @property
    def requested(self) -> bool:
        return self._requested

    def request(self) -> None:
        self._requested = True

    def consume_request(self) -> bool:
        requested = self._requested
        self._requested = False
        return requested

    def set_button_bounds(self, bounds: tuple[int, int, int, int]) -> None:
        self.button_bounds = bounds

    def handle_mouse(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _parameter: object | None = None,
    ) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        x1, y1, x2, y2 = self.button_bounds
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.request()


class TransientNotice:
    """Hold a dashboard message for a bounded monotonic-time interval."""

    def __init__(self) -> None:
        self._text: str | None = None
        self._expires_at = 0.0

    def show(self, text: str, *, now: float, duration_seconds: float = 1.0) -> None:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        self._text = text
        self._expires_at = now + duration_seconds

    def message(self, now: float) -> str | None:
        if self._text is None or now >= self._expires_at:
            return None
        return self._text


def _mapping(raw: object, *, name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(raw)


def _load_camera(item: object, *, index: int) -> CameraSourceConfig | None:
    if not isinstance(item, dict):
        raise ValueError(f"Camera entry {index} must be a mapping")
    values = dict(item)
    kind = values.pop("kind", None)
    if values.get("enabled", True) is False:
        return None
    try:
        if kind == "ip":
            return IpCameraConfig(**values)
        if kind == "webcam":
            return WebcamConfig(**values)
    except TypeError as error:
        raise ValueError(f"Invalid camera entry {index}: {error}") from error
    raise ValueError(f"Camera entry {index} kind must be 'ip' or 'webcam'")


def load_camera_demo_config(path: Path, *, mode: str | None = None) -> CameraDemoConfig:
    """Load one Stage 8 mode from YAML and validate its complete serving contract."""
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Camera demo configuration does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, name="camera demo configuration")
    selected_mode = mode or root.get("active_mode")
    if selected_mode not in SUPPORTED_MODES:
        raise ValueError(f"active_mode must be one of: {', '.join(sorted(SUPPORTED_MODES))}")
    modes = _mapping(root.get("modes"), name="modes")
    selected = _mapping(modes.get(selected_mode), name=f"mode {selected_mode}")
    camera_rows = selected.get("cameras")
    if not isinstance(camera_rows, list):
        raise ValueError(f"mode {selected_mode} must contain a cameras list")
    cameras = tuple(
        camera
        for index, item in enumerate(camera_rows)
        if (camera := _load_camera(item, index=index)) is not None
    )

    model_values = _mapping(root.get("models"), name="models")
    if "detector_model" in model_values:
        model_values["detector_model"] = Path(model_values["detector_model"])
    if "checkpoint" in model_values:
        model_values["checkpoint"] = Path(model_values["checkpoint"])
    runtime_values = _mapping(root.get("runtime"), name="runtime")
    if "output_dir" in runtime_values:
        runtime_values["output_dir"] = Path(runtime_values["output_dir"])
    display_values = _mapping(root.get("display"), name="display")
    try:
        return CameraDemoConfig(
            mode=str(selected_mode),
            cameras=cameras,
            models=DemoModelConfig(**model_values),
            runtime=DemoRuntimeConfig(**runtime_values),
            display=DemoDisplayConfig(**display_values),
        )
    except TypeError as error:
        raise ValueError(f"Invalid camera demo setting: {error}") from error


def apply_source_overrides(
    config: CameraDemoConfig,
    *,
    camera_urls: Sequence[str] = (),
    webcam_index: int | None = None,
) -> CameraDemoConfig:
    """Apply explicit CLI source overrides without mutating the YAML-derived config."""
    url_by_id: dict[str, str] = {}
    for value in camera_urls:
        camera_id, separator, url = value.partition("=")
        if not separator or not camera_id or not url:
            raise ValueError("--camera-url must use CAMERA_ID=URL")
        if camera_id in url_by_id:
            raise ValueError(f"Duplicate --camera-url override: {camera_id}")
        url_by_id[camera_id] = url

    known_ids = set(config.camera_ids)
    unknown_ids = sorted(set(url_by_id) - known_ids)
    if unknown_ids:
        raise ValueError(f"Unknown camera URL override IDs: {', '.join(unknown_ids)}")
    webcam_count = sum(isinstance(camera, WebcamConfig) for camera in config.cameras)
    if webcam_index is not None and webcam_count != 1:
        raise ValueError("--webcam-index requires exactly one configured webcam")

    cameras: list[CameraSourceConfig] = []
    for camera in config.cameras:
        if camera.camera_id in url_by_id:
            if not isinstance(camera, IpCameraConfig):
                raise ValueError(f"Camera {camera.camera_id} is not an IP camera")
            camera = replace(camera, url=url_by_id[camera.camera_id])
        if webcam_index is not None and isinstance(camera, WebcamConfig):
            camera = replace(camera, device_index=webcam_index)
        cameras.append(camera)
    return replace(config, cameras=tuple(cameras))


def build_live_workers(
    cameras: Sequence[CameraSourceConfig],
) -> dict[str, LiveWorker]:
    """Construct the correct latest-frame worker for each configured source."""
    workers: dict[str, LiveWorker] = {}
    for camera in cameras:
        if isinstance(camera, IpCameraConfig):
            workers[camera.camera_id] = IpCameraWorker(camera)
        else:
            workers[camera.camera_id] = WebcamWorker(camera)
    return workers


def build_live_pipelines(
    cameras: Sequence[CameraSourceConfig],
    detector: Any,
    encoder: Any,
    *,
    tracker_factory: Callable[[str], Any] = ByteTrackLocalTracker,
    tracklet_config: TrackletEmbeddingConfig | None = None,
) -> dict[str, ReIDCameraPipeline]:
    """Share heavy models while isolating tracker and tracklet state per camera."""
    return {
        camera.camera_id: ReIDCameraPipeline(
            stage3=SingleCameraPipeline(
                detector=detector,
                tracker=tracker_factory(camera.camera_id),
            ),
            tracklets=TrackletEmbeddingStore(encoder, tracklet_config),
        )
        for camera in cameras
    }


def _status_text(status: Any) -> str:
    if bool(status.connected):
        return f"connected | received {status.frames_received}"
    return f"disconnected | {status.last_error or 'waiting for frames'}"


def _placeholder(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def render_dashboard(
    camera_ids: Sequence[str],
    frames: Mapping[str, np.ndarray],
    statuses: Mapping[str, str],
    identities: Mapping[str, Sequence[IdentityDisplay]],
    config: DemoDisplayConfig,
    *,
    controller: ScreenshotController,
    notification: str | None = None,
) -> np.ndarray:
    """Render unobstructed camera pixels above separate black information panels."""
    ids = tuple(camera_ids)
    rows = math.ceil(len(ids) / config.columns)
    tiles: list[np.ndarray] = []
    for camera_id in ids:
        source = frames.get(camera_id)
        if source is None:
            video = _placeholder(config.tile_width, config.tile_height)
        else:
            video = cv2.resize(
                source,
                (config.tile_width, config.tile_height),
                interpolation=cv2.INTER_AREA,
            )
        panel = np.zeros(
            (config.info_panel_height, config.tile_width, 3),
            dtype=np.uint8,
        )
        cv2.putText(
            panel,
            camera_id,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            statuses.get(camera_id, "waiting"),
            (10, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (190, 220, 190),
            1,
            cv2.LINE_AA,
        )
        rows_for_camera = tuple(identities.get(camera_id, ()))
        max_rows = max(1, (config.info_panel_height - 64) // 22)
        if not rows_for_camera:
            cv2.putText(
                panel,
                "No active person tracks",
                (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )
        for row_index, identity in enumerate(rows_for_camera[:max_rows]):
            text_y = 72 + row_index * 22
            cv2.rectangle(
                panel,
                (10, text_y - 12),
                (24, text_y + 2),
                identity.color,
                -1,
            )
            cv2.putText(
                panel,
                identity.text,
                (32, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                identity.color,
                1,
                cv2.LINE_AA,
            )
        tiles.append(np.vstack((video, panel)))

    blank = np.zeros(
        (config.tile_height + config.info_panel_height, config.tile_width, 3),
        dtype=np.uint8,
    )
    while len(tiles) < rows * config.columns:
        tiles.append(blank.copy())
    grid_rows = [
        np.hstack(tiles[index : index + config.columns])
        for index in range(0, len(tiles), config.columns)
    ]
    grid = np.vstack(grid_rows)
    bar = np.full(
        (config.control_bar_height, grid.shape[1], 3),
        (32, 32, 32),
        dtype=np.uint8,
    )
    button_width = min(260, max(180, grid.shape[1] // 3))
    button_bounds = (12, 8, 12 + button_width, config.control_bar_height - 8)
    controller.set_button_bounds(
        (
            button_bounds[0],
            grid.shape[0] + button_bounds[1],
            button_bounds[2],
            grid.shape[0] + button_bounds[3],
        )
    )
    cv2.rectangle(
        bar,
        (button_bounds[0], button_bounds[1]),
        (button_bounds[2], button_bounds[3]),
        (52, 130, 220),
        -1,
    )
    cv2.putText(
        bar,
        "Save screenshot [S]",
        (button_bounds[0] + 14, button_bounds[3] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    message = notification or "Quit: Q or Esc"
    message_color = (90, 240, 120) if notification else (220, 220, 220)
    cv2.putText(
        bar,
        message,
        (button_bounds[2] + 24, button_bounds[3] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        message_color,
        2 if notification else 1,
        cv2.LINE_AA,
    )
    return np.vstack((grid, bar))


def save_dashboard_screenshot(frame: np.ndarray, directory: Path) -> Path:
    """Save one annotated dashboard using Unicode-safe OpenCV encoding."""
    output_dir = directory.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output = output_dir / f"camera_demo_{stamp}.jpg"
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("OpenCV could not encode the dashboard screenshot")
    output.write_bytes(encoded.tobytes())
    return output


def _final_status(workers: Mapping[str, LiveWorker]) -> dict[str, dict[str, object]]:
    return {camera_id: asdict(worker.status()) for camera_id, worker in workers.items()}


def run_camera_demo(
    config: CameraDemoConfig,
    *,
    pipelines: Mapping[str, LivePipeline],
    workers: Mapping[str, LiveWorker] | None = None,
    registry: GlobalIdentityRegistry | None = None,
    screenshot_controller: ScreenshotController | None = None,
) -> CameraDemoSummary:
    """Run live detection through global association for either Stage 8 source mode."""
    camera_ids = config.camera_ids
    if set(pipelines) != set(camera_ids):
        raise ValueError("pipelines must contain exactly one entry per camera")
    active_workers = workers or build_live_workers(config.cameras)
    if set(active_workers) != set(camera_ids):
        raise ValueError("workers must contain exactly one entry per camera")
    association = registry or GlobalIdentityRegistry(
        AssociationConfig(
            max_cosine_distance=config.models.max_cosine_distance,
            min_stored_samples=config.models.min_stored_samples,
            max_idle_frames=config.models.max_idle_updates,
        )
    )
    controller = screenshot_controller or ScreenshotController()
    output_dir = config.runtime.output_dir.expanduser().resolve()
    screenshot_dir = output_dir / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "assignments.jsonl"
    events_path = output_dir / "merge_events.jsonl"
    counts = {camera_id: 0 for camera_id in camera_ids}
    last_frame_ids = {camera_id: -1 for camera_id in camera_ids}
    latest_results: dict[str, ReIDProcessedFrame] = {}
    latest_received: dict[str, float] = {}
    rendered_frames: dict[str, np.ndarray] = {}
    identity_rows: dict[str, tuple[IdentityDisplay, ...]] = {}
    screenshots: list[str] = []
    notice = TransientNotice()
    association_updates = 0
    merge_count = 0
    started = time.monotonic()

    if config.display.enabled:
        cv2.namedWindow(config.display.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(config.display.window_name, controller.handle_mouse)
    for worker in active_workers.values():
        worker.start()

    try:
        with assignments_path.open("w", encoding="utf-8") as assignments_file, events_path.open(
            "w", encoding="utf-8"
        ) as events_file:
            while True:
                received = False
                for camera_id in camera_ids:
                    limit = config.runtime.max_frames_per_camera
                    if limit is not None and counts[camera_id] >= limit:
                        continue
                    packet = active_workers[camera_id].snapshot(last_frame_ids[camera_id])
                    if packet is None:
                        continue
                    result = pipelines[camera_id].process(packet)
                    now = time.monotonic()
                    last_frame_ids[camera_id] = packet.frame_id
                    latest_results[camera_id] = result
                    latest_received[camera_id] = now
                    counts[camera_id] += 1
                    received = True

                if received:
                    now = time.monotonic()
                    elapsed = now - started
                    current_results = {
                        camera_id: result
                        for camera_id, result in latest_results.items()
                        if now - latest_received[camera_id]
                        <= config.runtime.association_window_seconds
                    }
                    appearances = tuple(
                        appearance
                        for camera_id in camera_ids
                        if camera_id in current_results
                        for appearance in current_results[camera_id].appearances
                    )
                    association_result = association.update(
                        appearances,
                        frame_id=association_updates,
                        timestamp=elapsed,
                    )
                    for event in association_result.merge_events:
                        events_file.write(json.dumps(event.to_dict()) + "\n")
                        merge_count += 1
                    for appearance in appearances:
                        assignments_file.write(
                            json.dumps(
                                {
                                    "association_update": association_updates,
                                    "timestamp": elapsed,
                                    "camera_id": appearance.key.camera_id,
                                    "source_frame_id": current_results[
                                        appearance.key.camera_id
                                    ].stage3.packet.frame_id,
                                    "local_id": appearance.key.local_id,
                                    "global_id": association_result.assignments[appearance.key],
                                    "stored_samples": appearance.stored_samples,
                                    "total_samples": appearance.total_samples,
                                }
                            )
                            + "\n"
                        )
                    for camera_id, result in current_results.items():
                        rendered_frames[camera_id] = draw_global_boxes(
                            result.stage3.packet.frame,
                            result.stage3.tracks,
                            association_result.assignments,
                            camera_id,
                        )
                        identity_rows[camera_id] = tuple(
                            IdentityDisplay(
                                global_id=association_result.assignments.get(
                                    TrackKey(camera_id, track.local_id)
                                ),
                                local_id=track.local_id,
                                confidence=track.confidence,
                            )
                            for track in result.stage3.tracks
                        )
                    association_updates += 1

                dashboard: np.ndarray | None = None
                if config.display.enabled or controller.requested:
                    statuses = {
                        camera_id: _status_text(active_workers[camera_id].status())
                        for camera_id in camera_ids
                    }
                    dashboard = render_dashboard(
                        camera_ids,
                        rendered_frames,
                        statuses,
                        identity_rows,
                        config.display,
                        controller=controller,
                        notification=notice.message(time.monotonic()),
                    )
                should_quit = False
                if config.display.enabled and dashboard is not None:
                    cv2.imshow(config.display.window_name, dashboard)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {27, ord("q")}:
                        should_quit = True
                    elif key == ord("s"):
                        controller.request()
                if controller.consume_request() and dashboard is not None:
                    clean_dashboard = render_dashboard(
                        camera_ids,
                        rendered_frames,
                        {
                            camera_id: _status_text(active_workers[camera_id].status())
                            for camera_id in camera_ids
                        },
                        identity_rows,
                        config.display,
                        controller=controller,
                        notification=None,
                    )
                    screenshot = save_dashboard_screenshot(clean_dashboard, screenshot_dir)
                    screenshots.append(str(screenshot))
                    notice.show(
                        f"Screenshot saved: {screenshot.name}",
                        now=time.monotonic(),
                        duration_seconds=1.0,
                    )
                    if config.display.enabled:
                        confirmation = render_dashboard(
                            camera_ids,
                            rendered_frames,
                            {
                                camera_id: _status_text(
                                    active_workers[camera_id].status()
                                )
                                for camera_id in camera_ids
                            },
                            identity_rows,
                            config.display,
                            controller=controller,
                            notification=notice.message(time.monotonic()),
                        )
                        cv2.imshow(config.display.window_name, confirmation)

                elapsed = time.monotonic() - started
                active_camera_count = sum(count > 0 for count in counts.values())
                if (
                    elapsed >= config.runtime.startup_timeout_seconds
                    and active_camera_count < config.runtime.minimum_active_cameras
                ):
                    missing = [camera_id for camera_id, count in counts.items() if count == 0]
                    raise TimeoutError(
                        "Not enough cameras produced frames; missing: " + ", ".join(missing)
                    )
                if should_quit:
                    break
                if (
                    config.runtime.max_runtime_seconds is not None
                    and elapsed >= config.runtime.max_runtime_seconds
                ):
                    break
                if config.runtime.max_frames_per_camera is not None:
                    completed = sum(
                        count >= config.runtime.max_frames_per_camera
                        for count in counts.values()
                    )
                    if completed >= config.runtime.minimum_active_cameras:
                        break
                if not received and config.runtime.poll_interval_seconds:
                    time.sleep(config.runtime.poll_interval_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        for worker in active_workers.values():
            worker.stop()
        if config.display.enabled:
            cv2.destroyWindow(config.display.window_name)

    elapsed = time.monotonic() - started
    summary = CameraDemoSummary(
        mode=config.mode,
        camera_ids=camera_ids,
        elapsed_seconds=elapsed,
        total_processing_fps=sum(counts.values()) / elapsed if elapsed > 0 else 0.0,
        frames_by_camera=counts,
        association_updates=association_updates,
        global_identities_final=len(association),
        global_ids_issued=association.total_ids_issued,
        merge_events=merge_count,
        screenshots=tuple(screenshots),
        final_status=_final_status(active_workers),
        output_dir=str(output_dir),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the surface-level Stage 8 command interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Run live multi-camera person Re-ID in solo phone+laptop or multi-IP mode."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES))
    parser.add_argument(
        "--camera-url",
        action="append",
        default=[],
        metavar="CAMERA_ID=URL",
        help="Override one configured IP stream without editing YAML.",
    )
    parser.add_argument("--webcam-index", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--detector-model", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--max-frames-per-camera", type=int)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the selected mode without opening models or cameras.",
    )
    return parser


def _apply_cli_overrides(config: CameraDemoConfig, args: argparse.Namespace) -> CameraDemoConfig:
    config = apply_source_overrides(
        config,
        camera_urls=args.camera_url,
        webcam_index=args.webcam_index,
    )
    model_updates = {
        name: value
        for name, value in {
            "device": args.device,
            "detector_model": args.detector_model,
            "checkpoint": args.checkpoint,
        }.items()
        if value is not None
    }
    runtime_updates = {
        name: value
        for name, value in {
            "output_dir": args.output_dir,
            "max_runtime_seconds": args.max_runtime_seconds,
            "max_frames_per_camera": args.max_frames_per_camera,
        }.items()
        if value is not None
    }
    models = replace(config.models, **model_updates)
    runtime = replace(config.runtime, **runtime_updates)
    display = replace(config.display, enabled=False) if args.no_display else config.display
    return replace(config, models=models, runtime=runtime, display=display)


def main(argv: Sequence[str] | None = None) -> int:
    """Load one configured source mode and run the complete Stage 8 pipeline."""
    args = build_parser().parse_args(argv)
    config = load_camera_demo_config(args.config, mode=args.mode)
    config = _apply_cli_overrides(config, args)
    if args.dry_run:
        print(json.dumps(config.public_dict(), indent=2))
        return 0

    detector = YoloPersonDetector(
        YoloConfig(
            model=config.models.detector_model,
            confidence=config.models.confidence,
            image_size=config.models.image_size,
            device=config.models.device,
        )
    )
    encoder = ReIDEncoder.from_checkpoint(
        config.models.checkpoint,
        ReIDEncoderConfig(device=config.models.device),
    )
    pipelines = build_live_pipelines(
        config.cameras,
        detector,
        encoder,
        tracklet_config=TrackletEmbeddingConfig(
            history_size=config.models.history_size,
            embedding_interval_frames=config.models.embedding_interval,
            stale_after_frames=config.models.max_idle_updates,
        ),
    )
    summary = run_camera_demo(config, pipelines=pipelines)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
