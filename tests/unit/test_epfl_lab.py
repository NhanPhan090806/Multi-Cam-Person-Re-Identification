from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from multicam_reid.data.epfl_lab import (
    EpflLabLayoutError,
    download_epfl_lab,
    inspect_epfl_lab,
    load_epfl_calibration,
    load_epfl_ground_truth,
)
from multicam_reid.detection.yolo import DetectionBatch
from multicam_reid.inputs.epfl_lab import EpflLabSource
from multicam_reid.pipeline.epfl_lab_runner import (
    EpflLabRunConfig,
    build_epfl_lab_pipelines,
    run_epfl_lab,
)
from multicam_reid.pipeline.single_camera import ProcessedFrame
from multicam_reid.tracking.bytetrack import LocalTrack
from multicam_reid.types import FramePacket

CALIBRATION = """\
# Camera 0
# Ground plane homography
1 0 0
0 1 0
0 0 1
# Head plane homography
1 0 0
0 1 1
0 0 1

# Camera 1
# Ground plane homography
2 0 0
0 2 0
0 0 1
# Head plane height in camera view
0
"""

GROUND_TRUTH = """\
1
3 2 56 56 1 0 2
10 -2
11 100
-1 101
"""


def _write_video(path: Path, values: tuple[int, ...]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (64, 48),
    )
    assert writer.isOpened()
    for value in values:
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()


def make_epfl_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "epfl_lab"
    root.mkdir()
    _write_video(root / "6p-c0.avi", (20, 21, 22))
    _write_video(root / "6p-c1.avi", (80, 81, 82))
    (root / "calibration-6p.txt").write_text(CALIBRATION, encoding="utf-8")
    (root / "gt_lab_6p.txt").write_text(GROUND_TRUTH, encoding="utf-8")
    return root


def test_ground_truth_preserves_identity_columns_and_frame_numbers(tmp_path: Path) -> None:
    root = make_epfl_fixture(tmp_path)

    ground_truth = load_epfl_ground_truth(root / "gt_lab_6p.txt")

    assert ground_truth.number_of_frames == 3
    assert ground_truth.person_ids == (0, 1)
    assert ground_truth.grid_size == (56, 56)
    assert ground_truth.annotation_step == 1
    assert ground_truth.positions_at(0)[0].position_id == 10
    assert ground_truth.positions_at(0)[1].state == "out_of_scene"
    assert ground_truth.positions_at(1)[1].position_id == 100
    assert ground_truth.positions_at(2)[0].state == "undefined"


def test_ground_truth_keeps_only_scheduled_rows_from_dense_official_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gt_lab_6p.txt"
    path.write_text(
        "1\n5 1 56 56 2 0 4\n10\n-1\n11\n-1\n12\n",
        encoding="utf-8",
    )

    ground_truth = load_epfl_ground_truth(path)

    assert tuple(ground_truth.annotations) == (0, 2, 4)
    assert ground_truth.positions_at(1) == ()
    assert [ground_truth.positions_at(frame)[0].position_id for frame in (0, 2, 4)] == [
        10,
        11,
        12,
    ]


def test_calibration_parser_reads_ground_and_optional_head_planes(tmp_path: Path) -> None:
    root = make_epfl_fixture(tmp_path)

    calibrations = load_epfl_calibration(root / "calibration-6p.txt")

    assert set(calibrations) == {"C0", "C1"}
    np.testing.assert_allclose(calibrations["C0"].ground_homography, np.eye(3))
    assert calibrations["C0"].head_homography is not None
    assert calibrations["C0"].head_plane_height is None
    assert calibrations["C1"].head_homography is None
    assert calibrations["C1"].head_plane_height == 0.0


def test_inspector_validates_synchronized_video_contract(tmp_path: Path) -> None:
    root = make_epfl_fixture(tmp_path)

    summary = inspect_epfl_lab(root, cameras=("C0", "C1"))

    assert summary.dataset_root == root.resolve()
    assert summary.cameras == ("C0", "C1")
    assert summary.synchronized_frames == 3
    assert summary.video_fps == pytest.approx(25.0)
    assert summary.image_size == (64, 48)
    assert summary.annotated_frames == 3
    assert summary.persistent_person_ids == 2
    assert summary.active_position_annotations == 4
    assert summary.maximum_people_in_scene == 2
    assert summary.to_dict()["dataset_root"] == str(root.resolve())


def test_inspector_rejects_unequal_camera_lengths(tmp_path: Path) -> None:
    root = make_epfl_fixture(tmp_path)
    _write_video(root / "6p-c1.avi", (80, 81))

    with pytest.raises(EpflLabLayoutError, match="frame counts"):
        inspect_epfl_lab(root, cameras=("C0", "C1"))


def test_downloader_fetches_selected_videos_and_shared_metadata(tmp_path: Path) -> None:
    source = make_epfl_fixture(tmp_path)
    calls: list[tuple[str, str]] = []

    def downloader(url: str, destination: Path) -> None:
        calls.append((url, destination.name))
        shutil.copyfile(source / destination.name, destination)

    destination = tmp_path / "downloaded"
    summary = download_epfl_lab(
        destination,
        cameras=("C0", "C1"),
        downloader=downloader,
    )

    assert summary.synchronized_frames == 3
    assert [name for _, name in calls] == [
        "6p-c0.avi",
        "6p-c1.avi",
        "calibration-6p.txt",
        "gt_lab_6p.txt",
    ]
    assert all(url.startswith("https://") for url, _ in calls)


def test_source_yields_raw_frame_ids_shared_timestamps_and_ground_truth(
    tmp_path: Path,
) -> None:
    root = make_epfl_fixture(tmp_path)

    frames = list(EpflLabSource(root, cameras=("C0", "C1"), frame_stride=1))

    assert len(frames) == 3
    assert frames[0].frame_key == "000000"
    assert [packet.camera_id for packet in frames[0].packets] == ["C0", "C1"]
    assert {packet.frame_id for packet in frames[1].packets} == {1}
    assert [packet.timestamp for packet in frames[1].packets] == pytest.approx([0.04, 0.04])
    assert np.mean(frames[0].packet("C0").frame) == pytest.approx(20, abs=3)
    assert np.mean(frames[0].packet("C1").frame) == pytest.approx(80, abs=3)
    assert {position.person_id for position in frames[1].positions} == {0, 1}


def test_source_stride_keeps_source_time_and_annotation_alignment(tmp_path: Path) -> None:
    root = make_epfl_fixture(tmp_path)

    frames = list(EpflLabSource(root, cameras=("C0", "C1"), frame_stride=2))

    assert [frame.raw_frame_id for frame in frames] == [0, 2]
    assert [frame.packets[0].timestamp for frame in frames] == pytest.approx([0.0, 0.08])
    assert frames[1].positions[0].state == "undefined"


def test_source_can_seek_to_a_debugging_window_without_resetting_time(tmp_path: Path) -> None:
    root = make_epfl_fixture(tmp_path)

    frames = list(
        EpflLabSource(
            root,
            cameras=("C0", "C1"),
            start_frame=1,
            frame_stride=1,
        )
    )

    assert [frame.raw_frame_id for frame in frames] == [1, 2]
    assert frames[0].packets[0].timestamp == pytest.approx(0.04)


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


def test_runner_replays_selected_views_through_independent_pipelines(tmp_path: Path) -> None:
    root = make_epfl_fixture(tmp_path)
    pipelines = {camera: RecordingPipeline(camera) for camera in ("C0", "C1")}

    summary = run_epfl_lab(
        EpflLabRunConfig(
            dataset_root=root,
            output_dir=tmp_path / "output",
            cameras=("C0", "C1"),
            max_frames=3,
            save_video=False,
        ),
        pipelines=pipelines,
    )

    assert summary.frames_processed == 3
    assert summary.camera_frames_processed == 6
    assert summary.annotated_frames_seen == 3
    assert summary.unique_ground_truth_ids_seen == 2
    assert summary.source_fps == pytest.approx(25.0)
    assert summary.output_fps == pytest.approx(25.0)
    assert summary.camera_stats["C0"].track_observations == 3
    assert summary.camera_stats["C1"].unique_local_ids == 1
    assert pipelines["C0"] is not pipelines["C1"]
    assert [packet.frame_id for packet in pipelines["C0"].packets] == [0, 1, 2]
    assert (tmp_path / "output" / "summary.json").is_file()


def test_runner_writes_separate_and_synchronized_videos(tmp_path: Path) -> None:
    root = make_epfl_fixture(tmp_path)
    output = tmp_path / "rendered"
    pipelines = {camera: RecordingPipeline(camera) for camera in ("C0", "C1")}

    summary = run_epfl_lab(
        EpflLabRunConfig(
            dataset_root=root,
            output_dir=output,
            cameras=("C0", "C1"),
            max_frames=1,
            save_video=True,
            save_composite=True,
        ),
        pipelines=pipelines,
    )

    assert summary.composite_video == str((output / "synchronized_annotated.mp4").resolve())
    for name in ("C0_annotated.mp4", "C1_annotated.mp4", "synchronized_annotated.mp4"):
        assert (output / name).stat().st_size > 0


def test_runner_rejects_missing_or_shared_camera_pipelines(tmp_path: Path) -> None:
    root = make_epfl_fixture(tmp_path)
    config = EpflLabRunConfig(
        dataset_root=root,
        output_dir=tmp_path / "output",
        cameras=("C0", "C1"),
        save_video=False,
    )

    with pytest.raises(ValueError, match="exactly one"):
        run_epfl_lab(config, pipelines={"C0": RecordingPipeline("C0")})

    shared = RecordingPipeline("C0")
    with pytest.raises(ValueError, match="separate pipeline"):
        run_epfl_lab(config, pipelines={"C0": shared, "C1": shared})


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_frame": -1},
        {"frame_stride": 0},
        {"max_frames": 0},
        {"codec": "bad"},
    ],
)
def test_run_config_rejects_invalid_values(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    kwargs: dict[str, object] = {
        "dataset_root": tmp_path,
        "output_dir": tmp_path / "output",
        **overrides,
    }

    with pytest.raises(ValueError):
        EpflLabRunConfig(**kwargs)


def test_pipeline_builder_shares_detector_and_separates_trackers() -> None:
    detector = object()
    trackers: list[object] = []

    class FakeTracker:
        def __init__(self, camera_id: str) -> None:
            self.camera_id = camera_id
            trackers.append(self)

    pipelines = build_epfl_lab_pipelines(
        ("C0", "C1"),
        detector,
        tracker_factory=FakeTracker,
    )

    assert pipelines["C0"].detector is detector
    assert pipelines["C1"].detector is detector
    assert pipelines["C0"].tracker is not pipelines["C1"].tracker
    assert len(trackers) == 2


@pytest.mark.parametrize(
    ("cameras", "message"),
    [
        ((), "camera"),
        (("C0", "C0"), "unique"),
        (("C4",), "C0 through C3"),
    ],
)
def test_source_rejects_invalid_camera_selection(
    tmp_path: Path,
    cameras: tuple[str, ...],
    message: str,
) -> None:
    root = make_epfl_fixture(tmp_path)

    with pytest.raises(ValueError, match=message):
        EpflLabSource(root, cameras=cameras)
