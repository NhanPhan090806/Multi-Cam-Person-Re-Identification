from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

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


def test_external_video_packets_use_common_pipeline_and_write_dashboard(tmp_path):
    packets = [
        FramePacket(camera, frame, timestamp, np.full((32, 24, 3), value, dtype=np.uint8))
        for camera, frame, timestamp, value in [
            ("A", 0, 0, 200),
            ("B", 0, 0, 0),
            ("A", 1, 1, 200),
            ("B", 1, 1, 0),
            ("A", 2, 2, 0),
            ("B", 2, 2, 0),
            ("A", 3, 8, 0),
            ("B", 3, 8, 200),
        ]
    ]
    output = tmp_path / "output"
    result = run_handoff_replay(
        tmp_path / "source",
        frame_packets=packets,
        camera_ids=("A", "B"),
        pipelines={"A": ScheduledPipeline(), "B": ScheduledPipeline()},
        models=DemoModelConfig(),
        output_dir=output,
        video_fps=1,
    )
    assert result["source_duration_seconds"] == 8
    assert result["handoff_events"] == 1
    assert (output / "dashboard.mp4").stat().st_size > 0


def test_hybrid_replay_logs_simultaneous_matches_separately(tmp_path):
    packets = [
        FramePacket(camera, 1, 0.0, np.full((32, 24, 3), 200, dtype=np.uint8))
        for camera in ("A", "B")
    ]
    output = tmp_path / "output"
    result = run_handoff_replay(
        tmp_path / "source",
        frame_packets=packets,
        camera_ids=("A", "B"),
        pipelines={"A": ScheduledPipeline(), "B": ScheduledPipeline()},
        models=DemoModelConfig(
            association_mode="hybrid",
            overlap_pairs=(("A", "B"),),
            overlap_confirmations=1,
        ),
        output_dir=output,
    )
    assert result["association_mode"] == "hybrid"
    assert result["global_ids_issued"] == 1
    assert result["overlap_events"] == 1 and result["handoff_events"] == 0
    assert json.loads((output / "overlap_events.jsonl").read_text())["event"] == "overlap"
    assert json.loads((output / "settings.json").read_text())["association"]["overlap_pairs"] == [
        ["A", "B"],
    ]
    assert list((output / "screenshots").glob("*.jpg"))


def test_replay_reconciles_existing_ids_and_writes_decisions_and_frame_manifest(tmp_path):
    class ImprovingPipeline(ScheduledPipeline):
        def process(self, packet):
            result = super().process(packet)
            vector = (0, 1) if packet.camera_id == "B" and packet.frame_id == 0 else (1, 0)
            appearance = replace(
                result.appearances[0],
                embedding=np.asarray(vector, np.float32),
                stored_samples=2,
                total_samples=packet.frame_id + 2,
            )
            return replace(result, appearances=(appearance,))

    packets = [
        FramePacket(camera, frame, float(frame), np.full((32, 24, 3), 200, np.uint8))
        for frame in range(6)
        for camera in ("A", "B")
    ]
    output = tmp_path / "output"
    result = run_handoff_replay(
        tmp_path / "source",
        frame_packets=packets,
        camera_ids=("A", "B"),
        pipelines={"A": ImprovingPipeline(), "B": ImprovingPipeline()},
        models=DemoModelConfig(association_mode="hybrid", overlap_pairs=(("A", "B"),)),
        output_dir=output,
        freshness_seconds=2.0,
    )
    assert result["merge_events"] == 1 and result["completed"] is True
    assert result["retained_identities"] == 1 and result["identity_aliases"] == {2: 1}
    assert len((output / "processed_frames.jsonl").read_text().splitlines()) == 12
    decisions = [
        json.loads(line)
        for line in (output / "association_decisions.jsonl").read_text().splitlines()
    ]
    assert any(row["reason"] == "reconciliation_accepted" for row in decisions)
    assert json.loads((output / "merge_events.jsonl").read_text())["merged_global_id"] == 2
    assert list((output / "screenshots").glob("*.jpg"))


def test_interrupted_preview_records_only_processed_packets(tmp_path, monkeypatch):
    for name in ("namedWindow", "imshow", "destroyWindow"):
        monkeypatch.setattr("multicam_reid.pipeline.handoff_replay.cv2." + name, lambda *args: None)
    monkeypatch.setattr(
        "multicam_reid.pipeline.handoff_replay.cv2.waitKey", lambda _delay: ord("q")
    )
    packets = [
        FramePacket(camera, 1, 0.0, np.full((32, 24, 3), 200, np.uint8)) for camera in ("A", "B")
    ]
    output = tmp_path / "output"
    result = run_handoff_replay(
        tmp_path / "source",
        frame_packets=packets,
        camera_ids=("A", "B"),
        pipelines={"A": ScheduledPipeline(), "B": ScheduledPipeline()},
        models=DemoModelConfig(),
        output_dir=output,
        display=DemoDisplayConfig(enabled=True),
    )
    assert result["completed"] is False
    assert result["frames_by_camera"] == {"A": 1, "B": 0}
    assert len((output / "processed_frames.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("fps", [0, -1, float("nan")])
def test_invalid_dashboard_fps_is_rejected(tmp_path, fps):
    with pytest.raises(ValueError, match="video_fps"):
        run_handoff_replay(
            tmp_path,
            pipelines={},
            models=DemoModelConfig(),
            output_dir=tmp_path / "out",
            video_fps=fps,
        )


def test_external_packets_require_distinct_camera_ids(tmp_path):
    with pytest.raises(ValueError, match="distinct camera IDs"):
        run_handoff_replay(
            tmp_path,
            pipelines={},
            models=DemoModelConfig(),
            output_dir=tmp_path / "out",
            camera_ids=("A", "A"),
            frame_packets=[],
        )
