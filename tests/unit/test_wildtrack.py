from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from multicam_reid.data.wildtrack import (
    WildtrackLayoutError,
    download_wildtrack,
    find_wildtrack_root,
    inspect_wildtrack,
)
from multicam_reid.detection.yolo import DetectionBatch
from multicam_reid.inputs.wildtrack import (
    WildtrackSource,
    load_wildtrack_annotations,
)
from multicam_reid.pipeline.single_camera import ProcessedFrame
from multicam_reid.pipeline.wildtrack_runner import (
    WildtrackRunConfig,
    build_wildtrack_pipelines,
    run_wildtrack,
)
from multicam_reid.tracking.bytetrack import LocalTrack
from multicam_reid.types import FramePacket


def _write_png(path: Path, value: int) -> None:
    frame = np.full((48, 64, 3), value, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".png", frame)
    assert encoded
    path.write_bytes(buffer.tobytes())


def _person(
    person_id: int,
    *,
    c1_visible: bool = True,
    c2_visible: bool = True,
) -> dict[str, object]:
    c1 = (
        {"viewNum": 0, "xmin": 1, "ymin": 2, "xmax": 21, "ymax": 42}
        if c1_visible
        else {"viewNum": 0, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1}
    )
    c2 = (
        {"viewNum": 1, "xmin": 5, "ymin": 4, "xmax": 25, "ymax": 44}
        if c2_visible
        else {"viewNum": 1, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1}
    )
    return {
        "personID": person_id,
        "positionID": 100 + person_id,
        "views": [
            c1,
            c2,
            *[
                {
                    "viewNum": view_num,
                    "xmin": -1,
                    "ymin": -1,
                    "xmax": -1,
                    "ymax": -1,
                }
                for view_num in range(2, 7)
            ],
        ],
    }


def make_wildtrack_fixture(tmp_path: Path) -> Path:
    dataset_root = (
        tmp_path / "Wildtrack_dataset_full" / "Wildtrack_dataset"
    )
    image_root = dataset_root / "Image_subsets"
    for camera in ("C1", "C2"):
        (image_root / camera).mkdir(parents=True)
    annotation_root = dataset_root / "annotations_positions"
    annotation_root.mkdir()

    for index, frame_key in enumerate(("00000000", "00000005")):
        _write_png(image_root / "C1" / f"{frame_key}.png", 20 + index)
        _write_png(image_root / "C2" / f"{frame_key}.png", 80 + index)
        people = [_person(7), _person(9, c1_visible=False)]
        (annotation_root / f"{frame_key}.json").write_text(
            json.dumps(people),
            encoding="utf-8",
        )
    return dataset_root


def test_find_and_inspect_nested_wildtrack_layout(tmp_path: Path) -> None:
    expected_root = make_wildtrack_fixture(tmp_path)

    found = find_wildtrack_root(tmp_path)
    summary = inspect_wildtrack(tmp_path, cameras=("C1", "C2"))

    assert found == expected_root.resolve()
    assert summary.dataset_root == expected_root.resolve()
    assert summary.cameras == ("C1", "C2")
    assert summary.synchronized_frames == 2
    assert summary.images_per_camera == {"C1": 2, "C2": 2}
    assert summary.image_size == (64, 48)
    assert summary.person_annotations == 4
    assert summary.visible_boxes_by_camera == {"C1": 2, "C2": 4}


def test_inspector_rejects_missing_synchronized_camera_frame(tmp_path: Path) -> None:
    root = make_wildtrack_fixture(tmp_path)
    (root / "Image_subsets" / "C2" / "00000005.png").unlink()

    with pytest.raises(WildtrackLayoutError, match="C2.*00000005"):
        inspect_wildtrack(tmp_path, cameras=("C1", "C2"))


def test_download_wildtrack_limits_mirror_to_selected_cameras(tmp_path: Path) -> None:
    root = make_wildtrack_fixture(tmp_path)
    calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(root.parents[1])

    output_dir = tmp_path / "download"
    summary = download_wildtrack(
        output_dir,
        cameras=("C1", "C2"),
        downloader=downloader,
    )

    assert summary.synchronized_frames == 2
    assert calls[0]["repo_id"] == "disl/my_dataset"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["local_dir"] == str(output_dir.resolve())
    patterns = calls[0]["allow_patterns"]
    assert isinstance(patterns, list)
    assert any("Image_subsets/C1" in pattern for pattern in patterns)
    assert any("Image_subsets/C2" in pattern for pattern in patterns)
    assert not any("Image_subsets/C3" in pattern for pattern in patterns)
    assert any("annotations_positions" in pattern for pattern in patterns)


def test_annotations_preserve_person_ids_and_visibility(tmp_path: Path) -> None:
    root = make_wildtrack_fixture(tmp_path)

    annotations = load_wildtrack_annotations(
        root / "annotations_positions" / "00000000.json",
        cameras=("C1", "C2"),
    )

    assert len(annotations) == 2
    assert annotations[0].person_id == 7
    assert annotations[0].position_id == 107
    assert annotations[0].box_for("C1") is not None
    assert annotations[0].box_for("C2") is not None
    assert annotations[1].box_for("C1") is None
    assert annotations[1].box_for("C2") is not None


def test_annotation_with_any_minus_one_coordinate_is_invisible(tmp_path: Path) -> None:
    root = make_wildtrack_fixture(tmp_path)
    annotation_path = root / "annotations_positions" / "00000000.json"
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    raw[0]["views"][0]["xmin"] = -1
    annotation_path.write_text(json.dumps(raw), encoding="utf-8")

    annotations = load_wildtrack_annotations(
        annotation_path,
        cameras=("C1", "C2"),
    )

    assert annotations[0].box_for("C1") is None
    assert annotations[0].box_for("C2") is not None


def test_wildtrack_source_yields_synchronized_packets_and_annotations(tmp_path: Path) -> None:
    root = make_wildtrack_fixture(tmp_path)

    frames = list(WildtrackSource(root, cameras=("C1", "C2"), fps=2.0))

    assert len(frames) == 2
    first = frames[0]
    assert first.frame_key == "00000000"
    assert [packet.camera_id for packet in first.packets] == ["C1", "C2"]
    assert {packet.frame_id for packet in first.packets} == {0}
    assert {packet.timestamp for packet in first.packets} == {0.0}
    assert int(first.packet("C1").frame[0, 0, 0]) == 20
    assert int(first.packet("C2").frame[0, 0, 0]) == 80
    assert frames[1].packets[0].frame_id == 1
    assert frames[1].packets[0].timestamp == pytest.approx(0.5)
    assert {person.person_id for person in first.people} == {7, 9}


class RecordingPipeline:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.packets: list[FramePacket] = []

    def process(self, packet: FramePacket) -> ProcessedFrame:
        assert packet.camera_id == self.camera_id
        self.packets.append(packet)
        track = LocalTrack(
            camera_id=self.camera_id,
            local_id=1,
            xyxy=(1.0, 2.0, 21.0, 42.0),
            confidence=0.9,
            class_id=0,
            detection_index=0,
        )
        return ProcessedFrame(
            packet=packet,
            detections=DetectionBatch(
                xyxy=np.asarray([[1, 2, 21, 42]], dtype=np.float32),
                conf=np.asarray([0.9], dtype=np.float32),
                cls=np.asarray([0], dtype=np.float32),
            ),
            tracks=(track,),
            crops=(),
            annotated_frame=packet.frame.copy(),
        )


def test_wildtrack_runner_replays_both_views_through_separate_pipelines(
    tmp_path: Path,
) -> None:
    root = make_wildtrack_fixture(tmp_path)
    for annotation_path in (root / "annotations_positions").glob("*.json"):
        people = json.loads(annotation_path.read_text(encoding="utf-8"))
        people.append(_person(11, c1_visible=False, c2_visible=False))
        annotation_path.write_text(json.dumps(people), encoding="utf-8")
    pipelines = {camera: RecordingPipeline(camera) for camera in ("C1", "C2")}

    summary = run_wildtrack(
        WildtrackRunConfig(
            dataset_root=root,
            output_dir=tmp_path / "output",
            cameras=("C1", "C2"),
            max_frames=2,
            save_video=False,
        ),
        pipelines=pipelines,
    )

    assert summary.frames_processed == 2
    assert summary.camera_frames_processed == 4
    assert summary.unique_ground_truth_ids == 2
    assert summary.camera_stats["C1"].track_observations == 2
    assert summary.camera_stats["C2"].track_observations == 2
    assert pipelines["C1"] is not pipelines["C2"]
    assert [packet.frame_id for packet in pipelines["C1"].packets] == [0, 1]
    saved = json.loads((tmp_path / "output" / "summary.json").read_text())
    assert saved["cameras"] == ["C1", "C2"]


def test_pipeline_builder_shares_detector_but_creates_one_tracker_per_camera() -> None:
    detector = object()
    trackers: list[object] = []

    class FakeTracker:
        def __init__(self, camera_id: str) -> None:
            self.camera_id = camera_id
            trackers.append(self)

    pipelines = build_wildtrack_pipelines(
        ("C1", "C2"),
        detector,
        tracker_factory=FakeTracker,
    )

    assert pipelines["C1"].detector is detector
    assert pipelines["C2"].detector is detector
    assert pipelines["C1"].tracker is not pipelines["C2"].tracker
    assert len(trackers) == 2


@pytest.mark.parametrize(
    ("cameras", "message"),
    [
        ((), "camera"),
        (("C1", "C1"), "unique"),
        (("C0",), "C1 through C7"),
    ],
)
def test_source_rejects_invalid_camera_selection(
    tmp_path: Path,
    cameras: tuple[str, ...],
    message: str,
) -> None:
    make_wildtrack_fixture(tmp_path)

    with pytest.raises(ValueError, match=message):
        WildtrackSource(tmp_path, cameras=cameras)
