from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from multicam_reid.evaluation.wisenet import evaluate_wisenet, match_boxes
from multicam_reid.inputs.wisenet import WiseNETSequence


def test_wisenet_cli_hybrid_dry_run_and_invalid_pairs(dataset, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    runner = import_module("main_wisenet_test")
    assert (
        runner.main(
            [
                "--data-root",
                str(dataset),
                "--cameras",
                "3",
                "4",
                "--overlap",
                "3,4",
                "--dry-run",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["association_mode"] == "hybrid"
    assert result["overlap_pairs"] == [["C3", "C4"]]
    with pytest.raises(SystemExit) as error:
        runner.main(
            [
                "--data-root",
                str(dataset),
                "--cameras",
                "3",
                "4",
                "--overlap",
                "3,5",
                "--dry-run",
            ]
        )
    assert error.value.code == 2
    assert "selected camera IDs" in capsys.readouterr().err


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "wise"
    (root / "video_set2").mkdir(parents=True)
    annotations = root / "annotations_set2/people_detection"
    annotations.mkdir(parents=True)
    for camera in (3, 4):
        writer = cv2.VideoWriter(
            str(root / f"video_set2/video2_{camera}.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10,
            (32, 32),
        )
        for _ in range(30):
            writer.write(np.full((32, 32, 3), 150, dtype=np.uint8))
        writer.release()
        active = range(0, 5) if camera == 3 else range(20, 25)
        frames = [
            {"frameNumber": frame, "detections": [{"id": 1, "xywh": [-2, 0, 22, 30]}]}
            for frame in active
        ]
        (annotations / f"video2_{camera}.json").write_text(
            json.dumps({"frames": frames, "resolution": [{"width": 32, "height": 32}]}),
            encoding="utf-8",
        )
    return root


def test_video_adapter_preserves_time_and_empty_frames(dataset):
    sequence = WiseNETSequence(dataset, (3, 4), stride=5)
    packets = list(sequence.frames())
    assert len(packets) == 12
    assert [packet.timestamp for packet in packets] == sorted(p.timestamp for p in packets)
    assert packets[2].frame_id == 5 and packets[2].timestamp == 0.5
    assert sequence.camera_ids == ("C3", "C4")
    assert sequence.ground_truth("C3")[0] == [(1, (0, 0, 20, 30))]
    assert sequence.ground_truth("C3").get(10, []) == []


@pytest.mark.parametrize("cameras,stride", [((3,), 1), ((3, 3), 1), ((0, 4), 1), ((3, 4), 0)])
def test_invalid_sequence_options(dataset, cameras, stride):
    with pytest.raises(ValueError):
        WiseNETSequence(dataset, cameras, stride=stride)


def test_missing_video_is_clear(dataset):
    with pytest.raises(FileNotFoundError):
        WiseNETSequence(dataset, (3, 5))


def test_time_slice_does_not_reset_timestamps(dataset):
    packets = list(WiseNETSequence(dataset, stride=5, start=1, end=2).frames())
    assert [p.timestamp for p in packets] == [1, 1, 1.5, 1.5]


def test_matching_is_one_to_one_and_thresholded():
    box = (0, 0, 20, 30)
    assert match_boxes([box], [box, box], 0.5) == [(0, 0)]
    assert match_boxes([box], [(25, 25, 30, 30)], 0.5) == []
    assert match_boxes([], [box], 0.5) == []


def write_predictions(path, *, source=7, arrival=7):
    rows = [
        {
            "camera_id": camera,
            "source_frame_id": frame,
            "local_id": 1,
            "global_id": global_id,
            "xyxy": [0, 0, 20, 30],
        }
        for camera, frames, global_id in [("C3", range(5), source), ("C4", range(20, 25), arrival)]
        for frame in frames
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_evaluation_recognizes_handoff_and_deduplicates(dataset, tmp_path):
    predictions = tmp_path / "assignments.jsonl"
    write_predictions(predictions)
    with predictions.open("a", encoding="utf-8") as stream:
        stream.write("\n" + predictions.read_text(encoding="utf-8"))
    report = evaluate_wisenet(WiseNETSequence(dataset, stride=1), predictions)
    assert report["expected_cross_camera_transitions"] == 1
    assert report["correct_transitions"] == 1
    assert report["transitions"][0]["blind_gap_seconds"] == pytest.approx(1.6)
    assert report["matched_gt_boxes"] == 10
    assert report["global_ids_by_person"] == {"1": [7]}


def test_interrupted_run_scores_only_processed_frames_including_empty_views(dataset, tmp_path):
    predictions = tmp_path / "assignments.jsonl"
    write_predictions(predictions)
    (tmp_path / "processed_frames.jsonl").write_text(
        "\n".join(
            json.dumps({"camera_id": camera, "source_frame_id": frame})
            for camera in ("C3", "C4")
            for frame in range(10)
        )
    )
    report = evaluate_wisenet(WiseNETSequence(dataset, stride=1), predictions)
    assert report["ground_truth_boxes"] == 5
    assert report["expected_cross_camera_transitions"] == 0
    assert report["evaluated_frames_by_camera"] == {"C3": 10, "C4": 10}
    assert report["evaluation_coverage_source"] == "processed_frames.jsonl"


def test_legacy_summary_counts_limit_scoring_without_guessing_from_detection_rows(
    dataset, tmp_path
):
    predictions = tmp_path / "assignments.jsonl"
    write_predictions(predictions)
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "frames_by_camera": {"C3": 10, "C4": 9},
                "sampling": {"stride": 1, "fps": 10, "start": 0, "end": None},
            }
        )
    )
    report = evaluate_wisenet(WiseNETSequence(dataset, stride=1), predictions)
    assert report["ground_truth_boxes"] == 5
    assert report["evaluated_frames_by_camera"] == {"C3": 10, "C4": 9}
    assert report["evaluation_coverage_source"] == "legacy_summary_counts"


def test_final_alias_report_does_not_rewrite_earlier_identity_predictions(dataset, tmp_path):
    predictions = tmp_path / "assignments.jsonl"
    write_predictions(predictions, source=7, arrival=8)
    (tmp_path / "merge_events.jsonl").write_text(
        json.dumps(
            {
                "survivor_global_id": 7,
                "merged_global_id": 8,
                "timestamp": 2.5,
            }
        )
    )
    report = evaluate_wisenet(WiseNETSequence(dataset, stride=1), predictions)
    assert report["global_ids_by_person"] == {"1": [7, 8]}
    assert report["resolved_global_ids_by_person"] == {"1": [7]}
    assert report["correct_transitions"] == 0


@pytest.mark.parametrize(
    "camera,frame,stride,start,end",
    [
        ("BAD", 0, 1, 0, None),
        ("C3", -1, 1, 0, None),
        ("C3", 0.5, 1, 0, None),
        ("C3", 30, 1, 0, None),
        ("C3", 1, 5, 0, None),
        ("C3", 0, 1, 1, None),
        ("C3", 10, 1, 0, 1),
    ],
)
def test_processing_manifest_rejects_wrong_selection(
    dataset, tmp_path, camera, frame, stride, start, end
):
    predictions = tmp_path / "assignments.jsonl"
    predictions.write_text("")
    (tmp_path / "processed_frames.jsonl").write_text(
        json.dumps(
            {
                "camera_id": camera,
                "source_frame_id": frame,
            }
        )
    )
    with pytest.raises(ValueError, match="manifest"):
        evaluate_wisenet(WiseNETSequence(dataset, stride=stride, start=start, end=end), predictions)


@pytest.mark.parametrize(
    "count,stride,end,message",
    [
        (-1, 1, None, "nonnegative"),
        (1.5, 1, None, "nonnegative"),
        (31, 1, None, "exceed"),
        (11, 1, 1, "exceed"),
        (1, 2, None, "repeat saved"),
    ],
)
def test_bad_legacy_summary_is_rejected(dataset, tmp_path, count, stride, end, message):
    predictions = tmp_path / "assignments.jsonl"
    predictions.write_text("")
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "frames_by_camera": {"C3": count, "C4": 0},
                "sampling": {"stride": stride, "fps": 10, "start": 0, "end": end},
            }
        )
    )
    with pytest.raises(ValueError, match=message):
        evaluate_wisenet(WiseNETSequence(dataset, stride=1, end=end), predictions)


def test_empty_processed_views_and_blank_lines_are_handled(dataset, tmp_path):
    predictions = tmp_path / "assignments.jsonl"
    predictions.write_text("")
    (tmp_path / "processed_frames.jsonl").write_text("\n\n")
    (tmp_path / "merge_events.jsonl").write_text("\n")
    report = evaluate_wisenet(WiseNETSequence(dataset, stride=1), predictions)
    assert report["ground_truth_boxes"] == 0
    assert report["evaluated_source_times_by_camera"] == {"C3": None, "C4": None}


def test_alias_chains_are_resolved_separately(dataset, tmp_path):
    predictions = tmp_path / "assignments.jsonl"
    write_predictions(predictions, source=7, arrival=9)
    (tmp_path / "merge_events.jsonl").write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"merged_global_id": 9, "survivor_global_id": 8},
                {"merged_global_id": 8, "survivor_global_id": 7},
            ]
        )
    )
    report = evaluate_wisenet(WiseNETSequence(dataset, stride=1), predictions)
    assert report["final_identity_aliases"] == {"9": 7, "8": 7}


def test_alias_log_rejects_cycles_or_nonpositive_ids(dataset, tmp_path):
    predictions = tmp_path / "assignments.jsonl"
    predictions.write_text("")
    (tmp_path / "merge_events.jsonl").write_text(
        json.dumps(
            {
                "merged_global_id": 7,
                "survivor_global_id": 8,
            }
        )
    )
    with pytest.raises(ValueError, match="smaller survivor"):
        evaluate_wisenet(WiseNETSequence(dataset), predictions)


def test_final_alias_audit_exposes_a_wrong_merge_of_two_different_people(dataset, tmp_path):
    labels = dataset / "annotations_set2/people_detection/video2_4.json"
    data = json.loads(labels.read_text())
    for frame in data["frames"]:
        frame["detections"][0]["id"] = 2
    labels.write_text(json.dumps(data))
    predictions = tmp_path / "assignments.jsonl"
    write_predictions(predictions, source=7, arrival=8)
    (tmp_path / "merge_events.jsonl").write_text(
        json.dumps(
            {
                "merged_global_id": 8,
                "survivor_global_id": 7,
            }
        )
    )
    report = evaluate_wisenet(WiseNETSequence(dataset, stride=1), predictions)
    assert report["shared_global_ids"] == {}
    assert report["resolved_shared_global_ids"] == {"7": [1, 2]}


@pytest.mark.parametrize("arrival,outcome", [(8, "changed_id"), (None, "unidentified")])
def test_evaluation_reports_failed_handoff(dataset, tmp_path, arrival, outcome):
    predictions = tmp_path / "assignments.jsonl"
    write_predictions(predictions, arrival=arrival)
    report = evaluate_wisenet(WiseNETSequence(dataset), predictions)
    assert report["correct_transitions"] == 0
    assert report["transitions"][0]["outcome"] == outcome


def test_no_predictions_report_detector_misses(dataset, tmp_path):
    predictions = tmp_path / "assignments.jsonl"
    predictions.write_text("\n", encoding="utf-8")
    report = evaluate_wisenet(WiseNETSequence(dataset), predictions)
    assert report["matched_gt_boxes"] == 0
    assert report["gt_box_recall"] == 0
    assert report["transitions"][0]["outcome"] == "unidentified"


def test_invalid_iou_is_rejected(dataset, tmp_path):
    with pytest.raises(ValueError, match="IoU"):
        evaluate_wisenet(WiseNETSequence(dataset), tmp_path / "unused", iou_threshold=0)
    assert match_boxes([(0, 0, 0, 0)], [(0, 0, 0, 0)], 0.5) == []


def test_same_camera_occlusion_is_not_counted_as_camera_handoff(dataset, tmp_path):
    annotation = dataset / "annotations_set2/people_detection/video2_3.json"
    data = json.loads(annotation.read_text())
    data["frames"] += [
        {"frameNumber": frame, "detections": [{"id": 1, "xywh": [0, 0, 20, 30]}]}
        for frame in range(10, 15)
    ]
    annotation.write_text(json.dumps(data))
    predictions = tmp_path / "assignments.jsonl"
    write_predictions(predictions)
    report = evaluate_wisenet(WiseNETSequence(dataset, stride=1), predictions)
    assert report["expected_cross_camera_transitions"] == 1
    assert report["transitions"][0]["blind_gap_seconds"] == pytest.approx(0.6)


def test_overlapping_visibility_is_not_a_blind_handoff(dataset, tmp_path):
    annotation = dataset / "annotations_set2/people_detection/video2_4.json"
    data = json.loads(annotation.read_text())
    data["frames"] = [
        {"frameNumber": frame, "detections": [{"id": 1, "xywh": [0, 0, 20, 30]}]}
        for frame in range(5)
    ]
    annotation.write_text(json.dumps(data))
    predictions = tmp_path / "assignments.jsonl"
    predictions.touch()
    report = evaluate_wisenet(WiseNETSequence(dataset), predictions)
    assert report["expected_cross_camera_transitions"] == 0


def test_evaluation_does_not_score_frames_outside_requested_range(dataset, tmp_path):
    predictions = tmp_path / "assignments.jsonl"
    write_predictions(predictions)
    report = evaluate_wisenet(WiseNETSequence(dataset, start=2, end=2.5, stride=1), predictions)
    assert report["ground_truth_boxes"] == 5


def test_global_id_shared_by_different_people_is_not_success(dataset, tmp_path):
    annotation = dataset / "annotations_set2/people_detection/video2_4.json"
    data = json.loads(annotation.read_text())
    data["frames"].append({"frameNumber": 28, "detections": [{"id": 2, "xywh": [0, 0, 20, 30]}]})
    annotation.write_text(json.dumps(data))
    predictions = tmp_path / "assignments.jsonl"
    write_predictions(predictions)
    with predictions.open("a") as stream:
        stream.write(
            "\n"
            + json.dumps(
                {
                    "camera_id": "C4",
                    "source_frame_id": 28,
                    "local_id": 2,
                    "global_id": 7,
                    "xyxy": [0, 0, 20, 30],
                }
            )
        )
    report = evaluate_wisenet(WiseNETSequence(dataset, stride=1), predictions)
    assert report["shared_global_ids"] == {"7": [1, 2]}
    assert report["transitions"][0]["outcome"] == "shared_id"
    assert report["eventually_recovered_transitions"] == 0


@pytest.mark.parametrize("start,end", [(-1, None), (float("nan"), None), (0, 0), (0, float("inf"))])
def test_invalid_time_range_is_rejected(dataset, start, end):
    with pytest.raises(ValueError, match="time range"):
        WiseNETSequence(dataset, start=start, end=end)


def test_unknown_ground_truth_camera(dataset):
    with pytest.raises(ValueError, match="unknown camera"):
        WiseNETSequence(dataset).ground_truth("C5")


def test_delayed_recovery_is_distinct_from_prompt_success(dataset, tmp_path):
    video = dataset / "video_set2/video2_4.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    for _ in range(50):
        writer.write(np.full((32, 32, 3), 150, dtype=np.uint8))
    writer.release()
    annotation = dataset / "annotations_set2/people_detection/video2_4.json"
    data = json.loads(annotation.read_text())
    data["frames"] = [
        {"frameNumber": frame, "detections": [{"id": 1, "xywh": [0, 0, 20, 30]}]}
        for frame in range(20, 50)
    ]
    annotation.write_text(json.dumps(data))
    predictions = tmp_path / "assignments.jsonl"
    write_predictions(predictions, arrival=None)
    with predictions.open("a") as stream:
        stream.write(
            "\n"
            + json.dumps(
                {
                    "camera_id": "C4",
                    "source_frame_id": 44,
                    "local_id": 2,
                    "global_id": 7,
                    "xyxy": [0, 0, 20, 30],
                }
            )
        )
    sequence = WiseNETSequence(dataset, stride=1)
    assert len(list(sequence.frames())) == 80
    report = evaluate_wisenet(sequence, predictions)
    assert report["correct_transitions"] == 0
    assert report["eventually_recovered_transitions"] == 1
    assert report["transitions"][0]["arrival_assignment_delay_seconds"] == pytest.approx(2.4)


@pytest.mark.parametrize("opened,fps,count", [(False, 10, 30), (True, 0, 30), (True, 10, 0)])
def test_bad_video_metadata_fails_and_releases(dataset, monkeypatch, opened, fps, count):
    capture = MagicMock()
    capture.isOpened.return_value = opened
    capture.get.side_effect = lambda property_id: fps if property_id == cv2.CAP_PROP_FPS else count
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)
    with pytest.raises(ValueError):
        WiseNETSequence(dataset)
    capture.release.assert_called()


@pytest.mark.parametrize(
    "grab,retrieve,message", [(False, True, "truncated"), (True, False, "unreadable")]
)
def test_frame_read_errors_release_captures(dataset, monkeypatch, grab, retrieve, message):
    sequence = WiseNETSequence(dataset)
    capture = MagicMock()
    capture.grab.return_value = grab
    capture.retrieve.return_value = (retrieve, None)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)
    with pytest.raises(ValueError, match=message):
        next(sequence.frames())
    assert capture.release.call_count == 2


def test_fully_outside_annotation_box_is_skipped(dataset):
    annotation = dataset / "annotations_set2/people_detection/video2_3.json"
    data = json.loads(annotation.read_text())
    data["frames"][0]["detections"][0]["xywh"] = [100, 100, 20, 30]
    annotation.write_text(json.dumps(data))
    assert WiseNETSequence(dataset).ground_truth("C3")[0] == []
