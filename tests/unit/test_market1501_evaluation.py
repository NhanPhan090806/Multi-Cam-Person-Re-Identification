from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from multicam_reid.data.market1501 import Market1501Sample
from multicam_reid.evaluation import market1501 as evaluation_module
from multicam_reid.evaluation.market1501 import (
    EmbeddingSet,
    EvaluationConfig,
    evaluate_market1501,
    extract_embeddings,
    load_embedding_set,
    main,
    run_evaluation,
    save_embedding_set,
    select_retrieval_examples,
    write_retrieval_sheet,
)


class RecordingEncoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        self.batch_sizes.append(len(crops))
        rows = []
        for crop in crops:
            value = float(crop[0, 0, 0])
            row = np.array([value, 1.0], dtype=np.float32)
            rows.append(row / np.linalg.norm(row))
        return np.stack(rows)


class RuntimeEncoder(RecordingEncoder):
    device = "cpu"
    checkpoint_epoch = 60
    checkpoint_rank1 = 0.84

    @classmethod
    def from_checkpoint(cls, *_args: object, **_kwargs: object) -> RuntimeEncoder:
        return cls()


def make_embedding_set(
    embeddings: list[list[float]],
    person_ids: list[int],
    camera_ids: list[int],
    *,
    prefix: str,
) -> EmbeddingSet:
    return EmbeddingSet(
        embeddings=np.asarray(embeddings, dtype=np.float32),
        person_ids=np.asarray(person_ids, dtype=np.int32),
        camera_ids=np.asarray(camera_ids, dtype=np.int16),
        paths=tuple(Path(f"{prefix}_{index}.jpg") for index in range(len(person_ids))),
    )


def test_market1501_metrics_remove_same_identity_same_camera() -> None:
    query = make_embedding_set(
        [[1.0, 0.0], [0.0, 1.0]],
        [1, 2],
        [1, 1],
        prefix="query",
    )
    gallery = make_embedding_set(
        [
            [1.0, 0.0],
            [0.9, np.sqrt(0.19)],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
        [1, 3, 1, 2],
        [1, 2, 2, 2],
        prefix="gallery",
    )

    result = evaluate_market1501(query, gallery, max_rank=4, retrieval_topk=3)

    assert result.num_queries == 2
    assert result.num_gallery == 4
    assert result.num_valid_queries == 2
    assert result.mean_ap == pytest.approx(0.75)
    assert result.cmc[0] == pytest.approx(0.5)
    assert result.cmc[1] == pytest.approx(1.0)
    assert result.rank_at(1) == pytest.approx(0.5)
    assert result.rank_at(10) == pytest.approx(1.0)
    assert result.rankings[0].first_match_rank == 2
    assert result.rankings[1].first_match_rank == 1


def test_market1501_metrics_reject_no_valid_query() -> None:
    query = make_embedding_set([[1.0, 0.0]], [1], [1], prefix="query")
    gallery = make_embedding_set([[1.0, 0.0]], [1], [1], prefix="gallery")

    with pytest.raises(ValueError, match="No query has a valid cross-camera match"):
        evaluate_market1501(query, gallery)


def test_embedding_set_rejects_mismatched_metadata() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        make_embedding_set([[1.0, 0.0]], [1, 2], [1], prefix="bad")


def test_embedding_artifact_round_trip_without_pickle(tmp_path: Path) -> None:
    expected = make_embedding_set(
        [[1.0, 0.0], [0.0, 1.0]],
        [1, 2],
        [3, 4],
        prefix="sample",
    )
    artifact = tmp_path / "embeddings.npz"

    save_embedding_set(expected, artifact)
    actual = load_embedding_set(artifact)

    np.testing.assert_array_equal(actual.embeddings, expected.embeddings)
    np.testing.assert_array_equal(actual.person_ids, expected.person_ids)
    np.testing.assert_array_equal(actual.camera_ids, expected.camera_ids)
    assert actual.paths == expected.paths


def test_extract_embeddings_uses_batches_and_skips_no_samples(tmp_path: Path) -> None:
    samples = []
    for index in range(5):
        image_path = tmp_path / f"000{index + 1}_c1s1_000001_00.jpg"
        image = np.full((8, 4, 3), index + 1, dtype=np.uint8)
        encoded, buffer = cv2.imencode(".jpg", image)
        assert encoded is True
        image_path.write_bytes(buffer.tobytes())
        samples.append(
            Market1501Sample(path=image_path, person_id=index + 1, camera_id=1)
        )
    encoder = RecordingEncoder()

    artifact = extract_embeddings(samples, encoder, batch_size=2)

    assert encoder.batch_sizes == [2, 2, 1]
    assert artifact.embeddings.shape == (5, 2)
    np.testing.assert_array_equal(artifact.person_ids, np.arange(1, 6))
    assert np.allclose(np.linalg.norm(artifact.embeddings, axis=1), 1.0)


def test_select_retrieval_examples_returns_good_and_bad_queries() -> None:
    query = make_embedding_set(
        [[1.0, 0.0], [0.0, 1.0]],
        [1, 2],
        [1, 1],
        prefix="query",
    )
    gallery = make_embedding_set(
        [[0.0, 1.0], [0.8, 0.6], [1.0, 0.0]],
        [2, 1, 3],
        [2, 2, 2],
        prefix="gallery",
    )
    result = evaluate_market1501(query, gallery, retrieval_topk=3)

    selected = select_retrieval_examples(result.rankings, count=2)

    assert selected["good"][0].query_index == 1
    assert selected["bad"][0].query_index == 0


def test_write_retrieval_sheet_creates_decodable_image(tmp_path: Path) -> None:
    query_path = tmp_path / "query.jpg"
    gallery_paths = [tmp_path / f"gallery_{index}.jpg" for index in range(2)]
    for index, path in enumerate([query_path, *gallery_paths]):
        image = np.full((32, 16, 3), 40 + index * 30, dtype=np.uint8)
        encoded, buffer = cv2.imencode(".jpg", image)
        assert encoded is True
        path.write_bytes(buffer.tobytes())

    query = EmbeddingSet(
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        person_ids=np.asarray([1]),
        camera_ids=np.asarray([1]),
        paths=(query_path,),
    )
    gallery = EmbeddingSet(
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        person_ids=np.asarray([1, 2]),
        camera_ids=np.asarray([2, 2]),
        paths=tuple(gallery_paths),
    )
    ranking = evaluate_market1501(query, gallery, retrieval_topk=2).rankings
    output_path = tmp_path / "sheet.jpg"

    write_retrieval_sheet(query, gallery, ranking, output_path, title="Good retrievals")

    assert output_path.is_file()
    decoded = cv2.imdecode(
        np.frombuffer(output_path.read_bytes(), np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None


def _write_market_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 8, 3), value, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded is True
    path.write_bytes(buffer.tobytes())


def test_run_evaluation_writes_reproducible_artifacts_and_reuses_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "Market-1501-v15.09.15"
    (dataset_root / "bounding_box_train").mkdir(parents=True)
    _write_market_image(dataset_root / "query" / "0001_c1s1_000001_00.jpg", 10)
    _write_market_image(dataset_root / "bounding_box_test" / "0001_c2s1_000001_00.jpg", 10)
    _write_market_image(dataset_root / "bounding_box_test" / "0002_c2s1_000001_00.jpg", 50)
    _write_market_image(dataset_root / "bounding_box_test" / "-1_c3s1_000001_00.jpg", 80)
    checkpoint = tmp_path / "model.pth.tar-60"
    checkpoint.write_bytes(b"safe test checkpoint")
    output_dir = tmp_path / "evaluation"
    monkeypatch.setattr(evaluation_module, "ReIDEncoder", RuntimeEncoder)
    config = EvaluationConfig(
        dataset_root=dataset_root,
        checkpoint=checkpoint,
        output_dir=output_dir,
        batch_size=1,
        retrieval_topk=2,
        example_count=1,
    )

    metrics = run_evaluation(config)

    assert metrics["mAP"] == pytest.approx(1.0)
    assert metrics["cmc"]["rank_1"] == pytest.approx(1.0)
    assert metrics["num_gallery"] == 2
    assert (output_dir / "query_embeddings.npz").is_file()
    assert (output_dir / "gallery_embeddings.npz").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "retrieval_good.jpg").is_file()
    assert (output_dir / "retrieval_examples.json").is_file()

    reused = run_evaluation(
        EvaluationConfig(
            dataset_root=dataset_root,
            checkpoint=checkpoint,
            output_dir=output_dir,
            reuse_embeddings=True,
            retrieval_topk=2,
            example_count=0,
        )
    )

    assert reused["mAP"] == pytest.approx(1.0)
    assert reused["reused_embeddings"] is True

    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="do not match"):
        run_evaluation(
            EvaluationConfig(
                dataset_root=dataset_root,
                checkpoint=checkpoint,
                output_dir=output_dir,
                reuse_embeddings=True,
            )
        )


def test_reuse_requires_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Market-1501-v15.09.15"
    for directory in ("bounding_box_train", "query", "bounding_box_test"):
        (dataset_root / directory).mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(FileNotFoundError, match="embedding_manifest"):
        run_evaluation(
            EvaluationConfig(
                dataset_root=dataset_root,
                checkpoint=checkpoint,
                output_dir=tmp_path / "output",
                reuse_embeddings=True,
            )
        )


def test_cli_prints_metrics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {"mAP": 0.5, "cmc": {"rank_1": 0.75}}
    monkeypatch.setattr(evaluation_module, "run_evaluation", lambda _config: expected)

    exit_code = main(["--device", "cpu", "--example-count", "0"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [("batch_size", 0), ("max_rank", 0), ("retrieval_topk", 0), ("example_count", -1)],
)
def test_evaluation_config_rejects_invalid_numbers(field: str, value: int) -> None:
    values = {field: value}

    with pytest.raises(ValueError, match=field):
        EvaluationConfig(**values)
