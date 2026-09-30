"""Draw local person tracks without mutating the source frame."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np

from multicam_reid.tracking.bytetrack import LocalTrack

if TYPE_CHECKING:
    from multicam_reid.reid.tracklets import TrackKey, TrackletAppearance


def _track_color(local_id: int) -> tuple[int, int, int]:
    return (
        64 + (local_id * 53) % 192,
        64 + (local_id * 97) % 192,
        64 + (local_id * 151) % 192,
    )


def _global_color(global_id: int) -> tuple[int, int, int]:
    """Return the same high-contrast BGR color for an ID in every camera."""
    return (
        48 + (global_id * 67) % 208,
        48 + (global_id * 109) % 208,
        48 + (global_id * 163) % 208,
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


def draw_global_tracks(
    frame: np.ndarray,
    tracks: Sequence[LocalTrack],
    assignments: Mapping[TrackKey, int],
    camera_id: str,
) -> np.ndarray:
    """Draw global and local IDs, coloring a global ID identically across views."""
    from multicam_reid.reid.tracklets import TrackKey

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for row_index, track in enumerate(tracks):
        key = TrackKey(camera_id, track.local_id)
        global_id = assignments.get(key)
        color = _global_color(global_id) if global_id is not None else (160, 160, 160)
        x1 = int(np.clip(np.floor(track.xyxy[0]), 0, width - 1))
        y1 = int(np.clip(np.floor(track.xyxy[1]), 0, height - 1))
        x2 = int(np.clip(np.ceil(track.xyxy[2]), 0, width - 1))
        y2 = int(np.clip(np.ceil(track.xyxy[3]), 0, height - 1))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        global_label = f"Global {global_id}" if global_id is not None else "Global ?"
        label = f"{global_label} | Local {track.local_id}"
        compact_label = f"G{global_id}" if global_id is not None else "G?"
        compact_y = min(y2 - 4, y1 + 15)
        compact_size, compact_baseline = cv2.getTextSize(
            compact_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        )
        cv2.rectangle(
            annotated,
            (x1, max(0, compact_y - compact_size[1] - 3)),
            (
                min(width - 1, x1 + compact_size[0] + 4),
                min(height - 1, compact_y + compact_baseline),
            ),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            compact_label,
            (x1 + 2, compact_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        text_y = 42 + row_index * 18
        text_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        )
        legend_x = 5
        background_right = min(width - 1, legend_x + text_size[0] + 5)
        cv2.rectangle(
            annotated,
            (legend_x, max(0, text_y - text_size[1] - 3)),
            (background_right, min(height - 1, text_y + baseline)),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (legend_x + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated
