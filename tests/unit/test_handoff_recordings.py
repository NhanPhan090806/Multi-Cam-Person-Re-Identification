from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from multicam_reid.inputs.handoff_recordings import SessionRecorder, iter_recorded_frames
from multicam_reid.types import FramePacket


def test_recording_round_trip_preserves_camera_ids_and_real_elapsed_timestamps(tmp_path: Path):
    frame = np.full((24, 32, 3), 90, dtype=np.uint8)
    with SessionRecorder(tmp_path, ("PHONE", "LAPTOP")) as recorder:
        recorder.write(FramePacket("PHONE", 7, 12.0, frame), elapsed=0.2)
        recorder.write(FramePacket("LAPTOP", 3, 50.0, frame), elapsed=5.0)
        session = recorder.directory
    rows = list(iter_recorded_frames(session))
    assert [packet.camera_id for packet in rows] == ["PHONE", "LAPTOP"]
    assert [packet.timestamp for packet in rows] == [0.2, 5.0]
    assert [packet.frame_id for packet in rows] == [7, 3]
    assert rows[0].frame.shape == frame.shape
    assert json.loads((session / "manifest.json").read_text())["camera_ids"] == ["PHONE", "LAPTOP"]


def test_new_recording_never_overwrites_previous_session(tmp_path: Path):
    with SessionRecorder(tmp_path, ("A", "B")) as first:
        previous = first.directory
    with SessionRecorder(tmp_path, ("A", "B")) as second:
        assert second.directory != previous
    assert previous.is_dir()


def test_recording_rejects_unknown_camera_and_backward_timestamp(tmp_path: Path):
    frame = np.zeros((8, 8, 3), np.uint8)
    with SessionRecorder(tmp_path, ("A", "B")) as recorder:
        recorder.write(FramePacket("A", 0, 0.0, frame), elapsed=2.0)
        with pytest.raises(ValueError):
            recorder.write(FramePacket("C", 0, 0.0, frame), elapsed=3.0)
        with pytest.raises(ValueError):
            recorder.write(FramePacket("A", 1, 0.0, frame), elapsed=1.0)


def test_replay_rejects_paths_outside_recording(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_ids": ["A", "B"],
            }
        )
    )
    (tmp_path / "frames.jsonl").write_text(
        json.dumps(
            {
                "camera_id": "A",
                "frame_id": 0,
                "elapsed_seconds": 0.0,
                "path": "../outside.jpg",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="outside"):
        list(iter_recorded_frames(tmp_path))
