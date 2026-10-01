"""Threaded local-webcam input with reconnect and latest-frame semantics."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2

from multicam_reid.types import FramePacket

CaptureFactory = Callable[[int, int | None], Any]
BACKEND_CODES = {
    "auto": None,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


@dataclass(frozen=True, slots=True)
class WebcamConfig:
    """Connection settings for one physical webcam attached to this computer."""

    camera_id: str
    device_index: int
    enabled: bool = True
    capture_width: int | None = 640
    capture_height: int | None = 360
    capture_fps: float | None = None
    backend: str = "auto"
    reconnect_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if self.device_index < 0:
            raise ValueError("device_index must not be negative")
        if self.capture_width is not None and self.capture_width <= 0:
            raise ValueError("capture_width must be positive")
        if self.capture_height is not None and self.capture_height <= 0:
            raise ValueError("capture_height must be positive")
        if self.capture_fps is not None and self.capture_fps <= 0:
            raise ValueError("capture_fps must be positive")
        if self.backend not in BACKEND_CODES:
            raise ValueError(f"backend must be one of: {', '.join(BACKEND_CODES)}")
        if self.reconnect_delay_seconds <= 0:
            raise ValueError("reconnect_delay_seconds must be positive")


@dataclass(frozen=True, slots=True)
class WebcamStatus:
    """Non-sensitive health snapshot for a local webcam worker."""

    camera_id: str
    connected: bool
    frames_received: int
    last_error: str | None


class WebcamWorker:
    """Continuously read a local webcam without blocking the inference loop."""

    def __init__(
        self,
        config: WebcamConfig,
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
        backend = BACKEND_CODES[self.config.backend]
        if self._capture_factory is not None:
            capture = self._capture_factory(self.config.device_index, backend)
        elif backend is None:
            capture = cv2.VideoCapture(self.config.device_index)
        else:
            capture = cv2.VideoCapture(self.config.device_index, backend)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.config.capture_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.capture_width)
        if self.config.capture_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.capture_height)
        if self.config.capture_fps is not None:
            capture.set(cv2.CAP_PROP_FPS, self.config.capture_fps)
        return capture

    def start(self) -> None:
        """Start the daemon reader once."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"webcam-{self.config.camera_id}",
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
                    raise RuntimeError(
                        f"webcam device {self.config.device_index} could not be opened"
                    )
                self._set_health(connected=True, error=None)
                while not self._stop_event.is_set():
                    success, frame = capture.read()
                    if not success or frame is None:
                        raise RuntimeError("webcam read failed")
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

    def status(self) -> WebcamStatus:
        """Return health without exposing hardware or other sensitive settings."""
        with self._lock:
            return WebcamStatus(
                camera_id=self.config.camera_id,
                connected=self._connected,
                frames_received=self._frames_received,
                last_error=self._last_error,
            )

    def stop(self) -> None:
        """Request shutdown, release the capture, and wait briefly for cleanup."""
        self._stop_event.set()
        with self._lock:
            capture = self._capture
        if capture is not None:
            capture.release()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
