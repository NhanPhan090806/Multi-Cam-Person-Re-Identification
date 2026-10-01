from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from multicam_reid.detection.yolo import DetectionBatch
from multicam_reid.inputs.handoff_recordings import SessionRecorder
from multicam_reid.pipeline.camera_demo import DemoDisplayConfig, DemoModelConfig
from multicam_reid.pipeline.handoff_replay import run_handoff_replay
from multicam_reid.pipeline.reid_camera import ReIDProcessedFrame
from multicam_reid.pipeline.single_camera import ProcessedFrame
from multicam_reid.reid.tracklets import TrackKey, TrackletAppearance
from multicam_reid.tracking.bytetrack import LocalTrack
from multicam_reid.types import FramePacket


class ScheduledPipeline:
    def process(self, packet):
        is_person = packet.frame[0, 0, 0] > 100
        tracks = ()
        appearances = ()
        if is_person:
            tracks = (LocalTrack(packet.camera_id, 1, (1, 1, 20, 30), 0.9, 0, 0),)
            samples = 1 if packet.frame_id == 0 else 2
            appearances = (
                TrackletAppearance(
                    TrackKey(packet.camera_id, 1),
                    np.array([1.0, 0.0]),
                    samples,
                    samples,
                    0,
                    packet.frame_id,
                    0,
                    packet.timestamp,
                ),
            )
        return ReIDProcessedFrame(
            ProcessedFrame(
                packet,
                DetectionBatch.empty(),
                tracks,
                (),
                packet.frame.copy(),
            ),
            appearances,
        )


def test_raw_full_scene_replay_recovers_id_after_blind_gap(tmp_path: Path):
    with SessionRecorder(tmp_path / "recordings", ("A", "B")) as recorder:
        for camera, frame_id, timestamp, person in [
            ("A", 0, 0.0, True),
            ("A", 1, 1.0, True),
            ("A", 2, 2.0, False),
            ("B", 0, 8.0, True),
            ("B", 1, 9.0, True),
        ]:
            frame = np.full((32, 24, 3), 200 if person else 0, dtype=np.uint8)
            recorder.write(FramePacket(camera, frame_id, timestamp, frame), elapsed=timestamp)
        recording = recorder.directory
    output = tmp_path / "results"
    result = run_handoff_replay(
        recording,
        pipelines={"A": ScheduledPipeline(), "B": ScheduledPipeline()},
        models=DemoModelConfig(),
        display=DemoDisplayConfig(enabled=False),
        output_dir=output,
    )
    assert result["handoff_events"] == 1
    assert result["global_ids_issued"] == 1
    assert result["frames_by_camera"] == {"A": 3, "B": 2}
    event = json.loads((output / "handoff_events.jsonl").read_text())
    assert event["global_id"] == 1 and event["to_camera"] == "B"
    assert event["timestamp"] == 9.0
    assert list((output / "screenshots").glob("*.jpg"))
