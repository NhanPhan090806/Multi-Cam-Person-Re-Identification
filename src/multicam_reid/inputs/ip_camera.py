"""Threaded IP-camera input adapter with reconnect and latest-frame semantics."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import yaml

from multicam_reid.types import FramePacket

CaptureFactory = Callable[[str], Any]
ALLOWED_SCHEMES = {"http", "https", "rtsp"}


@dataclass(frozen=True, slots=True)
class IpCameraConfig:
    """Connection and display settings for one independent IP camera."""

    camera_id: str
    url: str
    enabled: bool = True
    window_width: int = 640
    window_height: int = 360
    window_x: int = 0
    window_y: int = 0
    reconnect_delay_seconds: float = 2.0
    open_timeout_ms: int = 5_000
    read_timeout_ms: int = 2_000

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        parsed = urlparse(self.url)
        if parsed.scheme.lower() not in ALLOWED_SCHEMES or not parsed.netloc:
            raise ValueError("url must include an http, https, or rtsp scheme and host")
        if self.window_width <= 0:
            raise ValueError("window_width must be positive")
        if self.window_height <= 0:
            raise ValueError("window_height must be positive")
        if self.window_x < 0 or self.window_y < 0:
            raise ValueError("window positions must not be negative")
        if self.reconnect_delay_seconds <= 0:
            raise ValueError("reconnect_delay_seconds must be positive")
        if self.open_timeout_ms <= 0 or self.read_timeout_ms <= 0:
            raise ValueError("camera timeouts must be positive")

    @property
    def window_size(self) -> tuple[int, int]:
        return self.window_width, self.window_height

    @property
    def window_position(self) -> tuple[int, int]:
        return self.window_x, self.window_y


@dataclass(frozen=True, slots=True)
class IpCameraStatus:
    """Non-sensitive health snapshot for one camera worker."""

    camera_id: str
    connected: bool
    frames_received: int
    last_error: str | None


def load_ip_camera_configs(path: Path) -> tuple[IpCameraConfig, ...]:
    """Load enabled camera entries from a YAML file and reject ambiguous IDs."""
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"IP webcam configuration does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cameras"), list):
        raise ValueError("IP webcam configuration must contain a cameras list")

    configs: list[IpCameraConfig] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw["cameras"]):
        if not isinstance(item, dict):
            raise ValueError(f"Camera entry {index} must be a mapping")
        try:
            config = IpCameraConfig(**item)
        except TypeError as error:
            raise ValueError(f"Invalid camera entry {index}: {error}") from error
        if config.camera_id in seen_ids:
            raise ValueError(f"Duplicate camera_id: {config.camera_id}")
        seen_ids.add(config.camera_id)
        if config.enabled:
            configs.append(config)
    if not configs:
        raise ValueError("No enabled IP webcams were configured")
    return tuple(configs)


class IpCameraWorker:
    """Continuously read one stream without allowing it to block other cameras."""

    def __init__(
        self,
        config: IpCameraConfig,
        *,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self.config = config
        self._capture_factory = capture_factory
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: Any | None = None
        self._latest_packet: FramePacket | None = None
        self._connected = False
        self._frames_received = 0
        self._last_error: str | None = None

    def _open_capture(self) -> Any:
        if self._capture_factory is not None:
            capture = self._capture_factory(self.config.url)
        else:
            params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                self.config.open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                self.config.read_timeout_ms,
            ]
            capture = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG, params)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def start(self) -> None:
        """Start the daemon reader once."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"ip-camera-{self.config.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def _set_health(self, *, connected: bool, error: str | None) -> None:
        with self._lock:
            self._connected = connected
            self._last_error = error

    def _run(self) -> None:
        frame_id = 0
        started = time.monotonic()
        while not self._stop_event.is_set():
            capture = None
            try:
                capture = self._open_capture()
                with self._lock:
                    self._capture = capture
                if not capture.isOpened():
                    raise RuntimeError("stream could not be opened")
                self._set_health(connected=True, error=None)
                while not self._stop_event.is_set():
                    success, frame = capture.read()
                    if not success or frame is None:
                        raise RuntimeError("stream read failed")
                    packet = FramePacket(
                        camera_id=self.config.camera_id,
                        frame_id=frame_id,
                        timestamp=time.monotonic() - started,
                        frame=frame,
                    )
                    with self._lock:
                        self._latest_packet = packet
                        self._frames_received += 1
                    frame_id += 1
            except Exception as error:
                self._set_health(connected=False, error=str(error))
            finally:
                if capture is not None:
                    capture.release()
                with self._lock:
                    if self._capture is capture:
                        self._capture = None
                    self._connected = False
            self._stop_event.wait(self.config.reconnect_delay_seconds)

    def snapshot(self, after_frame_id: int = -1) -> FramePacket | None:
        """Return the newest unseen frame without blocking."""
        with self._lock:
            if self._latest_packet is None or self._latest_packet.frame_id <= after_frame_id:
                return None
            return self._latest_packet

    def status(self) -> IpCameraStatus:
        """Return connection health without exposing the configured URL."""
        with self._lock:
            return IpCameraStatus(
                camera_id=self.config.camera_id,
                connected=self._connected,
                frames_received=self._frames_received,
                last_error=self._last_error,
            )

    def stop(self) -> None:
        """Request shutdown, interrupt the capture, and wait briefly for cleanup."""
        self._stop_event.set()
        with self._lock:
            capture = self._capture
        if capture is not None:
            capture.release()
        if self._thread is not None:
            self._thread.join(timeout=max(3.0, self.config.read_timeout_ms / 1000.0 + 1.0))
