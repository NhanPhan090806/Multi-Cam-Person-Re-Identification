from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from multicam_reid.data.epfl_lab import EpflCameraCalibration, EpflPersonPosition
from multicam_reid.evaluation.epfl_global import (
    AssignmentObservation,
    AttributedObservation,
    EpflGlobalEvaluationConfig,
    TopViewGeometry,
    attribute_frame,
    calculate_epfl_metrics,
    image_point_to_top_view,
    run_epfl_global_evaluation,
    top_view_point_to_grid,
)


def observation(
    camera_id: str,
    local_id: int,
    global_id: int,
    xyxy: tuple[float, float, float, float],
    *,
    frame_id: int = 0,
) -> AssignmentObservation:
    return AssignmentObservation(
        frame_id=frame_id,
        timestamp=frame_id / 25.0,
        camera_id=camera_id,
        local_id=local_id,
        global_id=global_id,
        xyxy=xyxy,
        confidence=0.9,
        stored_samples=3,
        total_samples=3,
    )


def attributed(
    camera_id: str,
    person_id: int,
    local_id: int,
    global_id: int,
    *,
    frame_id: int = 0,
) -> AttributedObservation:
    return AttributedObservation(
        frame_id=frame_id,
        timestamp=frame_id / 25.0,
        camera_id=camera_id,
        person_id=person_id,
        local_id=local_id,
        global_id=global_id,
        confidence=0.9,
        ground_distance_cells=0.25,
    )


def test_projection_converts_image_point_to_grid_coordinates() -> None:
    top_view = image_point_to_top_view((2.5, 3.5), np.eye(3))
    grid = top_view_point_to_grid(
        top_view,
        grid_size=(10, 10),
        geometry=TopViewGeometry(origin=(0.0, 0.0), size=(10.0, 10.0)),
    )

    assert top_view == pytest.approx((2.5, 3.5))
    assert grid == pytest.approx((2.0, 3.0))


def test_attribution_is_hungarian_one_to_one_and_distance_gated() -> None:
    predictions = (
        observation("C0", 1, 10, (0.0, 0.0, 1.0, 0.5)),
        observation("C0", 2, 20, (8.0, 0.0, 9.0, 9.5)),
    )
    positions = (
        EpflPersonPosition(0, 0, "active"),
        EpflPersonPosition(1, 99, "active"),
    )
    calibration = EpflCameraCalibration("C0", np.eye(3), None, 0.0)

    result = attribute_frame(
        predictions,
        positions,
        calibration,
        grid_size=(10, 10),
        geometry=TopViewGeometry(origin=(0.0, 0.0), size=(10.0, 10.0)),
        max_ground_distance_cells=1.0,
    )

    assert [(item.local_id, item.person_id) for item in result.matches] == [
        (1, 0),
        (2, 1),
    ]
    assert result.unmatched_predictions == ()
    assert result.unobserved_person_ids == ()


def test_cross_camera_metrics_count_correct_merge_false_merge_and_missed_merge() -> None:
    rows = (
        attributed("C0", 0, 1, 10),
        attributed("C0", 1, 2, 20),
        attributed("C1", 0, 3, 10),
        attributed("C1", 1, 4, 10),
    )

    metrics = calculate_epfl_metrics(
        rows,
        eligible_predictions=4,
        unmatched_predictions=0,
        scene_person_opportunities={(0, 0), (0, 1)},
    )

    association = metrics["cross_camera_association"]
    assert association["true_positive_pairs"] == 1
    assert association["false_positive_pairs"] == 1
    assert association["false_negative_pairs"] == 1
    assert association["true_negative_pairs"] == 1
    assert association["precision"] == pytest.approx(0.5)
    assert association["recall"] == pytest.approx(0.5)
    assert association["f1"] == pytest.approx(0.5)
    assert metrics["baselines"]["separate_camera"]["recall"] == 0.0
    assert metrics["baselines"]["single_global_identity"]["recall"] == 1.0


def test_metrics_report_local_and_global_switches_and_fragmentation() -> None:
    rows = (
        attributed("C0", 0, 1, 10, frame_id=0),
        attributed("C1", 0, 3, 10, frame_id=0),
        attributed("C0", 0, 2, 11, frame_id=25),
        attributed("C1", 0, 3, 11, frame_id=25),
    )

    metrics = calculate_epfl_metrics(
        rows,
        eligible_predictions=4,
        unmatched_predictions=0,
        scene_person_opportunities={(0, 0), (25, 0)},
    )

    assert metrics["local_tracking"]["id_switches"] == 1
    assert metrics["local_tracking"]["fragmentation"] == 1
    assert metrics["global_tracking"]["id_switches"] == 2
    assert metrics["global_tracking"]["fragmentation"] == 1
    assert metrics["scene_person_coverage"] == 1.0


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


def test_evaluation_runner_writes_reproducible_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "epfl"
    dataset.mkdir()
    (dataset / "calibration-6p.txt").write_text(CALIBRATION, encoding="utf-8")
    (dataset / "gt_lab_6p.txt").write_text(
        "1\n1 1 10 10 1 0 0\n0\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "stage6"
    run_dir.mkdir()
    rows = [
        {
            "frame_id": 0,
            "timestamp": 0.0,
            "camera_id": camera,
            "local_id": index,
            "global_id": 7,
            "stored_samples": 2,
            "total_samples": 2,
            "xyxy": [0.0, 0.0, 1.0, 0.5],
            "confidence": 0.9,
        }
        for index, camera in enumerate(("C0", "C1"), start=1)
    ]
    (run_dir / "assignments.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "cameras": ["C0", "C1"],
                "start_frame": 0,
                "frame_stride": 1,
                "frames_processed": 1,
                "threshold": 0.35,
            }
        ),
        encoding="utf-8",
    )
    market_metrics = tmp_path / "market1501_metrics.json"
    market_metrics.write_text(
        json.dumps(
            {
                "mAP": 0.64,
                "cmc": {"rank_1": 0.84, "rank_5": 0.94, "rank_10": 0.96},
                "checkpoint_sha256": "abc123",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation"

    metrics = run_epfl_global_evaluation(
        EpflGlobalEvaluationConfig(
            dataset_root=dataset,
            stage6_run_dir=run_dir,
            output_dir=output,
            top_view=TopViewGeometry(origin=(0.0, 0.0), size=(10.0, 10.0)),
            max_ground_distance_cells=1.0,
            market1501_metrics_path=market_metrics,
        )
    )

    assert metrics["cross_camera_association"]["f1"] == 1.0
    assert metrics["evaluated_annotated_frames"] == 1
    assert metrics["provenance"]["assignments_sha256"]
    assert (output / "metrics.json").is_file()
    assert (output / "attributions.jsonl").is_file()
    assert (output / "contingency.csv").is_file()
    assert (output / "report.md").is_file()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Market-1501 Rank-1 | 0.8400" in report


def test_evaluation_rejects_old_assignment_schema_without_boxes(tmp_path: Path) -> None:
    assignments = tmp_path / "assignments.jsonl"
    assignments.write_text(
        json.dumps(
            {
                "frame_id": 0,
                "timestamp": 0.0,
                "camera_id": "C0",
                "local_id": 1,
                "global_id": 1,
                "stored_samples": 2,
                "total_samples": 2,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rerun Stage 6"):
        AssignmentObservation.load_jsonl(assignments)
