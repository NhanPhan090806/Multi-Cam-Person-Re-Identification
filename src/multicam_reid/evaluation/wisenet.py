"""Post-inference, annotation-based diagnostics for a small WiseNET camera pair."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from multicam_reid.inputs.wisenet import WiseNETSequence


def processing_coverage(
    sequence: WiseNETSequence,
    output: Path,
) -> tuple[dict[str, set[int]] | None, str]:
    """Include empty processed views; never infer coverage from detection rows."""
    manifest = output / "processed_frames.jsonl"
    if manifest.exists():
        coverage = {camera: set() for camera in sequence.camera_ids}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            camera, frame = row["camera_id"], row["source_frame_id"]
            if (
                camera not in coverage
                or not isinstance(frame, int)
                or frame < 0
                or frame >= sequence.frame_counts[camera]
                or frame % sequence.stride
                or frame / sequence.fps < sequence.start
                or sequence.end is not None
                and frame / sequence.fps >= sequence.end
            ):
                raise ValueError("processed frame manifest does not match selected sequence")
            coverage[camera].add(frame)
        return coverage, "processed_frames.jsonl"
    summary_path = output / "summary.json"
    if not summary_path.exists():
        return None, "requested_sequence_range"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sampling = summary.get("sampling", {})
    expected = {
        "stride": sequence.stride,
        "fps": sequence.fps,
        "start": sequence.start,
        "end": sequence.end,
    }
    counts = summary.get("frames_by_camera", {})
    if sampling != expected or set(counts) != set(sequence.camera_ids):
        raise ValueError("repeat saved --cameras, --stride, --start and --end when re-scoring")
    if any(not isinstance(count, int) or count < 0 for count in counts.values()):
        raise ValueError("saved processed-frame counts must be nonnegative integers")
    first = math.ceil(sequence.start * sequence.fps / sequence.stride) * sequence.stride
    coverage = {
        camera: set(range(first, first + count * sequence.stride, sequence.stride))
        for camera, count in counts.items()
    }
    for camera, frames in coverage.items():
        if any(
            frame >= sequence.frame_counts[camera]
            or sequence.end is not None
            and frame / sequence.fps >= sequence.end
            for frame in frames
        ):
            raise ValueError("saved processed-frame counts exceed the selected sequence")
    return coverage, "legacy_summary_counts"


def final_identity_aliases(output: Path) -> dict[int, int]:
    """Audit final merged IDs separately; do not retroactively improve continuity scores."""
    path = output / "merge_events.jsonl"
    aliases = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            old, survivor = event["merged_global_id"], event["survivor_global_id"]
            if not isinstance(old, int) or not isinstance(survivor, int) or not 0 < survivor < old:
                raise ValueError("merge log must use positive IDs with the smaller survivor")
            aliases[old] = survivor
    for old, survivor in aliases.items():
        while survivor in aliases:
            survivor = aliases[survivor]
        aliases[old] = survivor
    return aliases


def match_boxes(truth: list, predicted: list, threshold: float) -> list[tuple[int, int]]:
    """Maximum-weight, one-to-one IoU matching; below-threshold edges are forbidden."""
    if not truth or not predicted:
        return []
    ious = np.zeros((len(truth), len(predicted)))
    for i, a in enumerate(truth):
        for j, b in enumerate(predicted):
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
            area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            union = area_a + area_b - intersection
            ious[i, j] = intersection / union if union else 0
    rows, columns = linear_sum_assignment(np.where(ious >= threshold, ious, 0), maximize=True)
    return [(int(i), int(j)) for i, j in zip(rows, columns, strict=True) if ious[i, j] >= threshold]


def evaluate_wisenet(
    sequence: WiseNETSequence,
    assignments: Path,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Score tracked boxes and cross-camera ID continuity, not model classification.

    Visibility comes from manual detection labels, not the coarse space graph.
    The replay log repeats cached views; deduplicate by source frame/local track.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    coverage, coverage_source = processing_coverage(sequence, assignments.parent)
    aliases = final_identity_aliases(assignments.parent)
    predictions = defaultdict(dict)
    for line in assignments.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            predictions[(row["camera_id"], row["source_frame_id"])][row["local_id"]] = row
    observations = defaultdict(list)
    visibility = defaultdict(list)
    global_people = defaultdict(set)
    person_counts = defaultdict(lambda: {"gt_boxes": 0, "matched_boxes": 0, "identified_boxes": 0})
    gt_count = predicted_count = matched_count = identified_count = 0
    evaluated = {camera: [] for camera in sequence.camera_ids}
    for camera in sequence.camera_ids:
        labels = sequence.ground_truth(camera)
        for frame_id in range(0, sequence.frame_counts[camera], sequence.stride):
            timestamp = frame_id / sequence.fps
            if timestamp < sequence.start or sequence.end is not None and timestamp >= sequence.end:
                continue
            if coverage is not None and frame_id not in coverage[camera]:
                continue
            evaluated[camera].append(timestamp)
            truth = labels.get(frame_id, [])
            rows = list(predictions[(camera, frame_id)].values())
            gt_count += len(truth)
            predicted_count += len(rows)
            for person, _box in truth:
                visibility[(person, camera)].append(timestamp)
                person_counts[str(person)]["gt_boxes"] += 1
            matches = match_boxes(
                [box for _, box in truth], [r["xyxy"] for r in rows], iou_threshold
            )
            matched_count += len(matches)
            for i, j in matches:
                person = truth[i][0]
                person_counts[str(person)]["matched_boxes"] += 1
                global_id = rows[j]["global_id"]
                if global_id is not None:
                    identified_count += 1
                    person_counts[str(person)]["identified_boxes"] += 1
                    observations[(person, camera)].append((timestamp, global_id))
                    global_people[global_id].add(person)
    visits = defaultdict(list)
    for (person, camera), times in visibility.items():
        first = previous = times[0]
        for timestamp in times[1:]:
            if timestamp - previous > sequence.stride / sequence.fps * 1.5:
                visits[person].append({"camera": camera, "start": first, "end": previous})
                first = timestamp
            previous = timestamp
        visits[person].append({"camera": camera, "start": first, "end": previous})
    transitions = []
    for person, entries in visits.items():
        ordered = sorted(entries, key=lambda v: v["start"])
        previous = ordered[0]
        for arrival in ordered[1:]:
            if arrival["camera"] == previous["camera"]:
                previous = {**previous, "end": arrival["end"]}
                continue
            if arrival["start"] <= previous["end"]:
                previous = arrival
                continue  # Simultaneous visibility is not a blind handoff test.
            source_ids = [
                gid
                for t, gid in observations[(person, previous["camera"])]
                if previous["start"] <= t <= previous["end"]
            ]
            arrival_ids = [
                gid
                for t, gid in observations[(person, arrival["camera"])]
                if arrival["start"] <= t <= min(arrival["end"], arrival["start"] + 2)
            ]
            source_id = source_ids[-1] if source_ids else None
            arrival_id = arrival_ids[0] if arrival_ids else None
            eventual = [
                (t, gid)
                for t, gid in observations[(person, arrival["camera"])]
                if arrival["start"] <= t <= arrival["end"]
            ]
            eventual_id = eventual[0][1] if eventual else None
            recovered = (
                source_id is not None
                and source_id == eventual_id
                and len(global_people[eventual_id]) == 1
            )
            outcome = (
                "unidentified"
                if source_id is None or arrival_id is None
                else (
                    "changed_id"
                    if source_id != arrival_id
                    else ("shared_id" if len(global_people[arrival_id]) > 1 else "correct")
                )
            )
            transitions.append(
                {
                    "person_id": person,
                    "from_camera": previous["camera"],
                    "to_camera": arrival["camera"],
                    "departure_seconds": previous["end"],
                    "arrival_seconds": arrival["start"],
                    "blind_gap_seconds": arrival["start"] - previous["end"],
                    "source_global_id": source_id,
                    "arrival_global_id": arrival_id,
                    "outcome": outcome,
                    "eventual_arrival_global_id": eventual_id,
                    "eventually_recovered": recovered,
                    "arrival_assignment_delay_seconds": (
                        eventual[0][0] - arrival["start"] if eventual else None
                    ),
                }
            )
            previous = arrival
    person_ids = defaultdict(set)
    resolved_people = defaultdict(set)
    for global_id, people in global_people.items():
        resolved_people[aliases.get(global_id, global_id)].update(people)
        for person in people:
            person_ids[str(person)].add(global_id)
    return {
        "scope": "WiseNET Set 2 selected cameras; small diagnostic, not a general accuracy claim",
        "cameras": sequence.camera_ids,
        "source_fps": sequence.fps,
        "stride": sequence.stride,
        "iou_threshold": iou_threshold,
        "evaluation_coverage_source": coverage_source,
        "evaluated_frames_by_camera": {camera: len(times) for camera, times in evaluated.items()},
        "evaluated_source_times_by_camera": {
            camera: {"first": times[0], "last": times[-1]} if times else None
            for camera, times in evaluated.items()
        },
        "ground_truth_boxes": gt_count,
        "predicted_track_boxes": predicted_count,
        "matched_gt_boxes": matched_count,
        "identified_gt_boxes": identified_count,
        "gt_box_recall": matched_count / gt_count if gt_count else 0,
        "track_box_precision": matched_count / predicted_count if predicted_count else 0,
        "global_ids_by_person": {p: sorted(ids) for p, ids in person_ids.items()},
        "resolved_global_ids_by_person": {
            person: sorted({aliases.get(gid, gid) for gid in ids})
            for person, ids in person_ids.items()
        },
        "final_identity_aliases": {str(old): survivor for old, survivor in aliases.items()},
        "per_person": dict(person_counts),
        "shared_global_ids": {str(g): sorted(p) for g, p in global_people.items() if len(p) > 1},
        "resolved_shared_global_ids": {
            str(gid): sorted(people) for gid, people in resolved_people.items() if len(people) > 1
        },
        "expected_cross_camera_transitions": len(transitions),
        "correct_transitions": sum(t["outcome"] == "correct" for t in transitions),
        "eventually_recovered_transitions": sum(t["eventually_recovered"] for t in transitions),
        "arrival_identification_window_seconds": 2,
        "transitions": transitions,
        "notes": [
            "Labels are used after inference; tracked boxes are matched one-to-one by IoU.",
            "Manual labels repeat every-fifth-frame annotations; timing is approximate.",
            "Arrival ID must be assigned within two seconds of annotated first visibility.",
            "Delayed recovery is reported separately, not silently counted as prompt success.",
            "Recall/precision above measure tracked boxes, not ReID classification accuracy.",
            "Scoring uses processed-frame coverage when available, including empty frames.",
            "Resolved IDs are audit-only; earlier predictions and continuity scores "
            "stay unchanged.",
        ],
    }
