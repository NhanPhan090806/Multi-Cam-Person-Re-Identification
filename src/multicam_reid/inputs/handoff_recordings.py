"""Raw full-scene frames and actual timestamps for reproducible handoff replays."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from multicam_reid.types import FramePacket


class SessionRecorder:
    """Write a new session of JPEGs plus JSONL; preserve every previous session."""

    def __init__(self, root: Path, camera_ids: Sequence[str]) -> None:
        if len(set(camera_ids)) != len(camera_ids) or len(camera_ids) < 2:
            raise ValueError("recording requires at least two unique camera IDs")
        self.camera_ids = tuple(camera_ids)
        name = f"session_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        self.directory = root.expanduser().resolve() / name
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "frames").mkdir()
        self._timeline = (self.directory / "frames.jsonl").open("w", encoding="utf-8")
        self._count = 0
        self._last_elapsed = -1.0
        (self.directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "camera_ids": self.camera_ids,
                    "description": "Raw full-scene frames; timestamps are session elapsed seconds.",
                    "ground_truth": (
                        "None; add human identity/entry/exit annotations for accuracy claims."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def write(self, packet: FramePacket, *, elapsed: float) -> None:
        if packet.camera_id not in self.camera_ids:
            raise ValueError("unknown recording camera")
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed < self._last_elapsed:
            raise ValueError("recording timestamps must be finite, nonnegative and monotonic")
        success, encoded = cv2.imencode(".jpg", packet.frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise RuntimeError("could not encode recording frame")
        relative = Path("frames") / f"{self._count:08d}.jpg"
        (self.directory / relative).write_bytes(encoded.tobytes())
        self._timeline.write(
            json.dumps(
                {
                    "camera_id": packet.camera_id,
                    "frame_id": packet.frame_id,
                    "source_timestamp": packet.timestamp,
                    "elapsed_seconds": elapsed,
                    "path": relative.as_posix(),
                }
            )
            + "\n"
        )
        self._timeline.flush()
        self._last_elapsed = elapsed
        self._count += 1

    def close(self) -> None:
        self._timeline.close()

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def recording_camera_ids(directory: Path) -> tuple[str, ...]:
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    camera_ids = tuple(metadata["camera_ids"])
    if (
        metadata.get("schema_version") != 1
        or len(camera_ids) < 2
        or len(set(camera_ids)) != len(camera_ids)
        or not all(isinstance(key, str) and key.strip() for key in camera_ids)
    ):
        raise ValueError("invalid handoff recording manifest")
    return camera_ids


def iter_recorded_frames(directory: Path) -> Iterator[FramePacket]:
    """Replay in source event time; unequal camera rates and blind gaps survive."""
    root = directory.expanduser().resolve()
    cameras = recording_camera_ids(root)
    last_time = -1.0
    last_frames: dict[str, int] = {}
    with (root / "frames.jsonl").open(encoding="utf-8") as timeline:
        for line in timeline:
            row = json.loads(line)
            path = (root / row["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("frame path points outside recording directory")
            camera_id, frame_id = row["camera_id"], int(row["frame_id"])
            timestamp = float(row["elapsed_seconds"])
            if (
                camera_id not in cameras
                or frame_id <= last_frames.get(camera_id, -1)
                or not math.isfinite(timestamp)
                or timestamp < 0
                or timestamp < last_time
            ):
                raise ValueError("invalid camera/frame/timestamp in recording timeline")
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"recorded image is unreadable: {path.name}")
            last_time, last_frames[camera_id] = timestamp, frame_id
            yield FramePacket(camera_id, frame_id, timestamp, image)
