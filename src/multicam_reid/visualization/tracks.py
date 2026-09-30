"""Draw local person tracks without mutating the source frame."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np

from multicam_reid.tracking.bytetrack import LocalTrack

if TYPE_CHECKING:
    from multicam_reid.reid.tracklets import TrackletAppearance


def _track_color(local_id: int) -> tuple[int, int, int]:
    return (
        64 + (local_id * 53) % 192,
        64 + (local_id * 97) % 192,
        64 + (local_id * 151) % 192,
    )


def draw_local_tracks(
    frame: np.ndarray,
    tracks: Sequence[LocalTrack],
    camera_id: str,
) -> np.ndarray:
    """Return a copy labeled with camera-local ByteTrack identities."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    cv2.putText(
        annotated,
        f"Camera {camera_id} | active tracks: {len(tracks)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for track in tracks:
        x1 = int(np.clip(np.floor(track.xyxy[0]), 0, width - 1))
        y1 = int(np.clip(np.floor(track.xyxy[1]), 0, height - 1))
        x2 = int(np.clip(np.ceil(track.xyxy[2]), 0, width - 1))
        y2 = int(np.clip(np.ceil(track.xyxy[3]), 0, height - 1))
        color = _track_color(track.local_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"person | {camera_id} local {track.local_id} | {track.confidence:.2f}"
        text_y = max(20, y1 - 7)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_reid_status(
    frame: np.ndarray,
    appearances: Sequence[TrackletAppearance],
) -> np.ndarray:
    """Add honest Stage 4 status without pretending local tracks are global IDs."""
    annotated = frame.copy()
    dimension = appearances[0].embedding.shape[0] if appearances else 0
    samples = sum(item.stored_samples for item in appearances)
    cv2.putText(
        annotated,
        f"Re-ID tracklets: {len(appearances)} | samples: {samples} | dim: {dimension}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated
