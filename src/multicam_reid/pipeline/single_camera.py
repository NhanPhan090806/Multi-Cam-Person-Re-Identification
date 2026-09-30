"""Detector-to-local-tracker single-camera processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from multicam_reid.detection.yolo import DetectionBatch
from multicam_reid.tracking.bytetrack import LocalTrack
from multicam_reid.types import FramePacket
from multicam_reid.visualization.tracks import draw_local_tracks


class PersonDetector(Protocol):
    def detect(self, frame: np.ndarray) -> DetectionBatch: ...


class LocalTracker(Protocol):
    camera_id: str

    def update(
        self,
        detections: DetectionBatch,
        frame: np.ndarray,
    ) -> tuple[LocalTrack, ...]: ...


@dataclass(frozen=True, slots=True)
class CropConfig:
    """Geometry guardrails for crops passed to the Re-ID stage."""

    min_width: int = 16
    min_height: int = 32
    min_area: int = 512

    def __post_init__(self) -> None:
        if self.min_width <= 0:
            raise ValueError("min_width must be positive")
        if self.min_height <= 0:
            raise ValueError("min_height must be positive")
        if self.min_area <= 0:
            raise ValueError("min_area must be positive")


@dataclass(frozen=True, slots=True, eq=False)
class PersonCrop:
    """A validated person crop tied to one local track and source frame."""

    track: LocalTrack
    frame_id: int
    timestamp: float
    xyxy: tuple[int, int, int, int]
    image: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class ProcessedFrame:
    """All Stage 3 outputs for one frame."""

    packet: FramePacket
    detections: DetectionBatch
    tracks: tuple[LocalTrack, ...]
    crops: tuple[PersonCrop, ...]
    annotated_frame: np.ndarray


def extract_person_crops(
    packet: FramePacket,
    tracks: tuple[LocalTrack, ...],
    config: CropConfig | None = None,
) -> tuple[PersonCrop, ...]:
    """Clip track boxes and copy only valid, sufficiently large person crops."""
    settings = config or CropConfig()
    height, width = packet.frame.shape[:2]
    crops: list[PersonCrop] = []
    for track in tracks:
        x1 = int(np.clip(np.floor(track.xyxy[0]), 0, width))
        y1 = int(np.clip(np.floor(track.xyxy[1]), 0, height))
        x2 = int(np.clip(np.ceil(track.xyxy[2]), 0, width))
        y2 = int(np.clip(np.ceil(track.xyxy[3]), 0, height))
        crop_width = x2 - x1
        crop_height = y2 - y1
        if (
            crop_width < settings.min_width
            or crop_height < settings.min_height
            or crop_width * crop_height < settings.min_area
        ):
            continue
        crop = packet.frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        crops.append(
            PersonCrop(
                track=track,
                frame_id=packet.frame_id,
                timestamp=packet.timestamp,
                xyxy=(x1, y1, x2, y2),
                image=crop,
            )
        )
    return tuple(crops)


class SingleCameraPipeline:
    """Connect person detection, one camera's ByteTrack, crops, and rendering."""

    def __init__(
        self,
        *,
        detector: PersonDetector,
        tracker: LocalTracker,
        crop_config: CropConfig | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.crop_config = crop_config or CropConfig()

    def process(self, packet: FramePacket) -> ProcessedFrame:
        """Process exactly one frame while preserving camera-local tracker state."""
        if packet.camera_id != self.tracker.camera_id:
            raise ValueError(
                f"Tracker belongs to camera {self.tracker.camera_id}, got {packet.camera_id}"
            )
        detections = self.detector.detect(packet.frame)
        tracks = self.tracker.update(detections, packet.frame)
        crops = extract_person_crops(packet, tracks, self.crop_config)
        annotated = draw_local_tracks(packet.frame, tracks, packet.camera_id)
        return ProcessedFrame(
            packet=packet,
            detections=detections,
            tracks=tracks,
            crops=crops,
            annotated_frame=annotated,
        )
