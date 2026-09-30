from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from multicam_reid.detection.yolo import DetectionBatch
from multicam_reid.pipeline.global_epfl_runner import (
    GlobalEpflRunConfig,
    build_global_epfl_pipelines,
    run_global_epfl,
)
from multicam_reid.pipeline.reid_camera import ReIDProcessedFrame
from multicam_reid.pipeline.single_camera import ProcessedFrame
from multicam_reid.reid.tracklets import TrackKey, TrackletAppearance
from multicam_reid.tracking.bytetrack import LocalTrack
from multicam_reid.types import FramePacket
from multicam_reid.visualization.tracks import draw_global_tracks

CALIBRATION = """\
# Camera 0
# Ground plane homography
1 0 0
0 1 0
0 0 1
# Head plane height in camera view
0

# Camera 1
# Ground plane homography
1 0 0
0 1 0
0 0 1
# Head plane height in camera view
0
"""

GROUND_TRUTH = """\
1
2 1 56 56 1 0 1
10
10
"""


def _write_video(path: Path, value: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48)
    )
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()


def make_epfl_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "epfl_lab"
    root.mkdir()
    _write_video(root / "6p-c0.avi", 20)
    _write_video(root / "6p-c1.avi", 80)
    (root / "calibration-6p.txt").write_text(CALIBRATION, encoding="utf-8")
    (root / "gt_lab_6p.txt").write_text(GROUND_TRUTH, encoding="utf-8")
    return root


class FixedReIDPipeline:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id

    def process(self, packet: FramePacket) -> ReIDProcessedFrame:
        track = LocalTrack(
            camera_id=self.camera_id,
            local_id=1,
            xyxy=(4.0, 3.0, 28.0, 45.0),
            confidence=0.9,
            class_id=0,
            detection_index=0,
        )
        stage3 = ProcessedFrame(
            packet=packet,
            detections=DetectionBatch.empty(),
            tracks=(track,),
            crops=(),
            annotated_frame=packet.frame.copy(),
        )
        appearance = TrackletAppearance(
            key=TrackKey(self.camera_id, 1),
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            stored_samples=2,
            total_samples=2,
            first_seen_frame_id=0,
            last_seen_frame_id=packet.frame_id,
            first_seen_timestamp=0.0,
            last_seen_timestamp=packet.timestamp,
        )
        return ReIDProcessedFrame(stage3=stage3, appearances=(appearance,))


def test_global_renderer_uses_shared_id_color_and_labels() -> None:
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    tracks = (
        LocalTrack("C0", 1, (5.0, 20.0, 35.0, 60.0), 0.9, 0, 0),
        LocalTrack("C0", 2, (50.0, 20.0, 85.0, 60.0), 0.8, 0, 1),
    )

    rendered = draw_global_tracks(
        frame,
        tracks,
        {TrackKey("C0", 1): 7},
        "C0",
    )

    assert np.any(rendered != frame)
    assert tuple(rendered[20, 5]) != (0, 0, 0)
    assert tuple(rendered[20, 50]) != (0, 0, 0)


def test_global_runner_writes_assignment_and_merge_evidence(tmp_path: Path) -> None:
    root = make_epfl_fixture(tmp_path)
    output = tmp_path / "global"

    summary = run_global_epfl(
        GlobalEpflRunConfig(
            dataset_root=root,
            output_dir=output,
            max_frames=2,
            save_video=False,
        ),
        pipelines={camera: FixedReIDPipeline(camera) for camera in ("C0", "C1")},
    )

    assert summary.frames_processed == 2
    assert summary.global_identities_final == 1
    assert summary.global_ids_issued == 2
    assert summary.merge_events == 1
    assert summary.camera_stats["C0"].appearance_observations == 2
    assert summary.camera_stats["C1"].unique_global_ids == 1
    assert json.loads((output / "summary.json").read_text())["merge_events"] == 1
    assignment_lines = (output / "assignments.jsonl").read_text().splitlines()
    assignments = [json.loads(line) for line in assignment_lines]
    assert len(assignments) == 4
    assert {row["global_id"] for row in assignments} == {1}
    merges = [json.loads(line) for line in (output / "merge_events.jsonl").read_text().splitlines()]
    assert len(merges) == 1
    assert merges[0]["survivor_global_id"] == 1


def test_global_pipeline_builder_shares_models_but_separates_state() -> None:
    detector = object()
    encoder = object()

    class FakeTracker:
        def __init__(self, camera_id: str) -> None:
            self.camera_id = camera_id

    pipelines = build_global_epfl_pipelines(
        ("C0", "C1"),
        detector,
        encoder,
        tracker_factory=FakeTracker,
    )

    assert pipelines["C0"].stage3.detector is detector
    assert pipelines["C1"].stage3.detector is detector
    assert pipelines["C0"].stage3.tracker is not pipelines["C1"].stage3.tracker
    assert pipelines["C0"].tracklets is not pipelines["C1"].tracklets
    assert pipelines["C0"].tracklets.encoder is encoder
    assert pipelines["C1"].tracklets.encoder is encoder
