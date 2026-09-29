"""Standalone Market-1501 embedding extraction, ranking, and visualization."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from multicam_reid.data.market1501 import (
    REQUIRED_SPLITS,
    Market1501Sample,
    find_market1501_root,
    parse_market1501_filename,
)
from multicam_reid.reid.encoder import (
    DEFAULT_CHECKPOINT,
    ReIDEncoder,
    ReIDEncoderConfig,
    read_bgr_image,
)

DEFAULT_DATASET_ROOT = Path("data/reid/market1501")
DEFAULT_OUTPUT_DIR = Path("outputs/evaluation/market1501")
ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Inputs and resource limits for standalone Market-1501 evaluation."""

    dataset_root: Path = DEFAULT_DATASET_ROOT
    checkpoint: Path = DEFAULT_CHECKPOINT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    batch_size: int = 64
    max_rank: int = 50
    retrieval_topk: int = 5
    example_count: int = 5
    query_chunk_size: int = 128
    reuse_embeddings: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "batch_size",
            "max_rank",
            "retrieval_topk",
            "query_chunk_size",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.example_count < 0:
            raise ValueError("example_count must not be negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")


@dataclass(frozen=True, slots=True)
class EmbeddingSet:
    """Embeddings and official identity metadata for one dataset split."""

    embeddings: np.ndarray
    person_ids: np.ndarray
    camera_ids: np.ndarray
    paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty 2D array")
        sample_count = self.embeddings.shape[0]
        if (
            self.person_ids.shape != (sample_count,)
            or self.camera_ids.shape != (sample_count,)
            or len(self.paths) != sample_count
        ):
            raise ValueError("embeddings and metadata must contain the same number of samples")
        if not np.issubdtype(self.embeddings.dtype, np.floating):
            raise ValueError("embeddings must use a floating-point dtype")
        if not np.all(np.isfinite(self.embeddings)):
            raise ValueError("embeddings must contain only finite values")
        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("embeddings must be L2-normalized")


@dataclass(frozen=True, slots=True)
class QueryRanking:
    """Filtered ranked-gallery evidence for one query."""

    query_index: int
    average_precision: float
    first_match_rank: int | None
    gallery_indices: tuple[int, ...]
    distances: tuple[float, ...]
    matches: tuple[bool, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready ranking evidence."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Market1501Evaluation:
    """Official Market-1501 CMC and mean average precision results."""

    mean_ap: float
    cmc: tuple[float, ...]
    num_queries: int
    num_gallery: int
    num_valid_queries: int
    rankings: tuple[QueryRanking, ...]

    def rank_at(self, rank: int) -> float:
        """Return CMC at a one-based rank, clipping to the computed curve."""
        if rank <= 0:
            raise ValueError("rank must be positive")
        return self.cmc[min(rank, len(self.cmc)) - 1]

    def to_dict(self) -> dict[str, Any]:
        """Return stable, JSON-ready metrics without the per-query rankings."""
        return {
            "protocol": "Market-1501 single-query",
            "distance": "cosine",
            "num_queries": self.num_queries,
            "num_gallery": self.num_gallery,
            "num_valid_queries": self.num_valid_queries,
            "mAP": self.mean_ap,
            "cmc": {
                "rank_1": self.rank_at(1),
                "rank_5": self.rank_at(5),
                "rank_10": self.rank_at(10),
            },
        }


def collect_split_samples(dataset_root: Path, split: str) -> tuple[Market1501Sample, ...]:
    """Collect a split using Torchreid's PID -1 junk filtering."""
    if split not in REQUIRED_SPLITS:
        raise ValueError(f"Unknown Market-1501 split: {split}")
    root = find_market1501_root(dataset_root)
    split_dir = root / REQUIRED_SPLITS[split]
    samples = tuple(
        sample
        for path in sorted(split_dir.glob("*.jpg"))
        if not (sample := parse_market1501_filename(path)).is_junk
    )
    if not samples:
        raise ValueError(f"No usable images found in Market-1501 {split} split")
    return samples


def extract_embeddings(
    samples: Sequence[Market1501Sample],
    encoder: Any,
    *,
    batch_size: int = 64,
) -> EmbeddingSet:
    """Encode all samples in deterministic path order using bounded batches."""
    if not samples:
        raise ValueError("samples must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batches: list[np.ndarray] = []
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        crops = [read_bgr_image(sample.path) for sample in batch_samples]
        encoded = np.asarray(encoder.encode(crops), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != len(batch_samples):
            raise RuntimeError("Encoder returned an invalid embedding batch shape")
        batches.append(encoded)

    return EmbeddingSet(
        embeddings=np.ascontiguousarray(np.concatenate(batches, axis=0)),
        person_ids=np.asarray([sample.person_id for sample in samples], dtype=np.int32),
        camera_ids=np.asarray([sample.camera_id for sample in samples], dtype=np.int16),
        paths=tuple(sample.path.resolve() for sample in samples),
    )


def save_embedding_set(embedding_set: EmbeddingSet, output_path: Path) -> None:
    """Save an embedding artifact that can be loaded with pickle disabled."""
    path = output_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embedding_set.embeddings.astype(np.float32, copy=False),
        person_ids=embedding_set.person_ids.astype(np.int32, copy=False),
        camera_ids=embedding_set.camera_ids.astype(np.int16, copy=False),
        paths=np.asarray([str(item) for item in embedding_set.paths], dtype=np.str_),
    )


def load_embedding_set(input_path: Path) -> EmbeddingSet:
    """Load and validate an embedding artifact without enabling pickle."""
    path = input_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Embedding artifact does not exist: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        required = {"embeddings", "person_ids", "camera_ids", "paths"}
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(f"Embedding artifact is missing fields: {sorted(missing)}")
        return EmbeddingSet(
            embeddings=np.asarray(artifact["embeddings"], dtype=np.float32),
            person_ids=np.asarray(artifact["person_ids"], dtype=np.int32),
            camera_ids=np.asarray(artifact["camera_ids"], dtype=np.int16),
            paths=tuple(Path(value) for value in artifact["paths"].tolist()),
        )


def _padded_cmc(matches: np.ndarray, max_rank: int) -> np.ndarray:
    cmc = np.minimum(matches.cumsum(), 1).astype(np.float64)
    if len(cmc) >= max_rank:
        return cmc[:max_rank]
    return np.pad(cmc, (0, max_rank - len(cmc)), constant_values=float(cmc[-1]))


def evaluate_market1501(
    query: EmbeddingSet,
    gallery: EmbeddingSet,
    *,
    max_rank: int = 50,
    retrieval_topk: int = 5,
    query_chunk_size: int = 128,
) -> Market1501Evaluation:
    """Evaluate cosine retrieval with the official Market-1501 filtering rule."""
    if query.embeddings.shape[1] != gallery.embeddings.shape[1]:
        raise ValueError("query and gallery embedding dimensions must match")
    for name, value in {
        "max_rank": max_rank,
        "retrieval_topk": retrieval_topk,
        "query_chunk_size": query_chunk_size,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    effective_max_rank = min(max_rank, len(gallery.paths))
    cmc_rows: list[np.ndarray] = []
    average_precisions: list[float] = []
    rankings: list[QueryRanking] = []

    gallery_matrix = np.ascontiguousarray(gallery.embeddings.T)
    for chunk_start in range(0, len(query.paths), query_chunk_size):
        chunk = query.embeddings[chunk_start : chunk_start + query_chunk_size]
        distances = 1.0 - chunk @ gallery_matrix
        for offset, distance_row in enumerate(distances):
            query_index = chunk_start + offset
            query_pid = query.person_ids[query_index]
            query_camid = query.camera_ids[query_index]
            order = np.argsort(distance_row, kind="stable")
            same_identity_camera = (
                (gallery.person_ids[order] == query_pid)
                & (gallery.camera_ids[order] == query_camid)
            )
            filtered_order = order[~same_identity_camera]
            matches = gallery.person_ids[filtered_order] == query_pid
            if not np.any(matches):
                continue

            cmc_rows.append(_padded_cmc(matches, effective_max_rank))
            relevant_count = int(matches.sum())
            precision = matches.cumsum() / np.arange(1, len(matches) + 1)
            average_precision = float((precision * matches).sum() / relevant_count)
            average_precisions.append(average_precision)
            first_match_rank = int(np.flatnonzero(matches)[0]) + 1
            evidence_count = min(retrieval_topk, len(filtered_order))
            evidence = filtered_order[:evidence_count]
            rankings.append(
                QueryRanking(
                    query_index=query_index,
                    average_precision=average_precision,
                    first_match_rank=first_match_rank,
                    gallery_indices=tuple(int(value) for value in evidence),
                    distances=tuple(float(distance_row[value]) for value in evidence),
                    matches=tuple(bool(value) for value in matches[:evidence_count]),
                )
            )

    if not cmc_rows:
        raise ValueError("No query has a valid cross-camera match in the gallery")
    mean_cmc = np.mean(np.stack(cmc_rows), axis=0)
    return Market1501Evaluation(
        mean_ap=float(np.mean(average_precisions)),
        cmc=tuple(float(value) for value in mean_cmc),
        num_queries=len(query.paths),
        num_gallery=len(gallery.paths),
        num_valid_queries=len(cmc_rows),
        rankings=tuple(rankings),
    )


def select_retrieval_examples(
    rankings: Sequence[QueryRanking],
    *,
    count: int,
) -> dict[str, tuple[QueryRanking, ...]]:
    """Choose strong rank-1 successes and informative rank-1 failures."""
    if count < 0:
        raise ValueError("count must not be negative")
    good = sorted(
        (ranking for ranking in rankings if ranking.matches and ranking.matches[0]),
        key=lambda ranking: (-ranking.average_precision, ranking.query_index),
    )
    bad = sorted(
        (ranking for ranking in rankings if ranking.matches and not ranking.matches[0]),
        key=lambda ranking: (
            -(ranking.first_match_rank or 0),
            ranking.average_precision,
            ranking.query_index,
        ),
    )
    return {"good": tuple(good[:count]), "bad": tuple(bad[:count])}


def _image_tile(path: Path, label: str, color: tuple[int, int, int]) -> np.ndarray:
    image = read_bgr_image(path)
    resized = cv2.resize(image, (128, 256), interpolation=cv2.INTER_AREA)
    bordered = cv2.copyMakeBorder(resized, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
    tile = cv2.copyMakeBorder(
        bordered,
        48,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(245, 245, 245),
    )
    for line_number, line in enumerate(label.splitlines()[:2]):
        cv2.putText(
            tile,
            line,
            (5, 18 + line_number * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
        )
    return tile


def write_retrieval_sheet(
    query: EmbeddingSet,
    gallery: EmbeddingSet,
    rankings: Sequence[QueryRanking],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Write query-plus-top-gallery rows with green/red correctness borders."""
    if not rankings:
        raise ValueError("rankings must not be empty")
    rows: list[np.ndarray] = []
    for ranking in rankings:
        query_pid = int(query.person_ids[ranking.query_index])
        tiles = [
            _image_tile(
                query.paths[ranking.query_index],
                f"QUERY\npid={query_pid} cam={int(query.camera_ids[ranking.query_index])}",
                (255, 120, 0),
            )
        ]
        for rank, (gallery_index, distance, is_match) in enumerate(
            zip(ranking.gallery_indices, ranking.distances, ranking.matches, strict=True),
            start=1,
        ):
            gallery_pid = int(gallery.person_ids[gallery_index])
            correctness = "MATCH" if is_match else "WRONG"
            label = f"R{rank} pid={gallery_pid}\nd={distance:.3f} {correctness}"
            tiles.append(
                _image_tile(
                    gallery.paths[gallery_index],
                    label,
                    (40, 180, 40) if is_match else (40, 40, 220),
                )
            )
        rows.append(np.concatenate(tiles, axis=1))

    sheet = np.concatenate(rows, axis=0)
    header = np.full((48, sheet.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(header, title, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    sheet = np.concatenate((header, sheet), axis=0)
    encoded, buffer = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not encoded:
        raise RuntimeError(f"Could not encode retrieval sheet: {output_path}")
    path = output_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _artifact_identity(config: EvaluationConfig, dataset_root: Path) -> dict[str, Any]:
    checkpoint = config.checkpoint.expanduser().resolve()
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "model_name": "osnet_x0_25",
        "embedding_dimension": 512,
        "input_size": [256, 128],
    }


def _ranking_example_dict(
    ranking: QueryRanking,
    query: EmbeddingSet,
    gallery: EmbeddingSet,
) -> dict[str, Any]:
    result = ranking.to_dict()
    result["query_path"] = str(query.paths[ranking.query_index])
    result["query_person_id"] = int(query.person_ids[ranking.query_index])
    result["gallery_paths"] = [str(gallery.paths[index]) for index in ranking.gallery_indices]
    result["gallery_person_ids"] = [
        int(gallery.person_ids[index]) for index in ranking.gallery_indices
    ]
    return result


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Extract artifacts, evaluate official retrieval, and write visual evidence."""
    started = time.perf_counter()
    dataset_root = find_market1501_root(config.dataset_root)
    checkpoint = config.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Re-ID checkpoint does not exist: {checkpoint}")
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = output_dir / "query_embeddings.npz"
    gallery_path = output_dir / "gallery_embeddings.npz"
    manifest_path = output_dir / "embedding_manifest.json"
    identity = _artifact_identity(config, dataset_root)

    encoder_metadata: dict[str, Any]
    if config.reuse_embeddings:
        if not manifest_path.is_file():
            raise FileNotFoundError("Cannot reuse embeddings without embedding_manifest.json")
        previous_identity = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous_identity.get(key) != value for key, value in identity.items()):
            raise ValueError("Saved embeddings do not match the requested dataset and checkpoint")
        query = load_embedding_set(query_path)
        gallery = load_embedding_set(gallery_path)
        encoder_metadata = {
            "device": previous_identity.get("device"),
            "checkpoint_epoch": previous_identity.get("checkpoint_epoch"),
            "checkpoint_rank1": previous_identity.get("checkpoint_rank1"),
        }
    else:
        encoder = ReIDEncoder.from_checkpoint(
            checkpoint,
            ReIDEncoderConfig(device=config.device),
        )
        query = extract_embeddings(
            collect_split_samples(dataset_root, "query"),
            encoder,
            batch_size=config.batch_size,
        )
        gallery = extract_embeddings(
            collect_split_samples(dataset_root, "gallery"),
            encoder,
            batch_size=config.batch_size,
        )
        save_embedding_set(query, query_path)
        save_embedding_set(gallery, gallery_path)
        encoder_metadata = {
            "device": encoder.device,
            "checkpoint_epoch": encoder.checkpoint_epoch,
            "checkpoint_rank1": encoder.checkpoint_rank1,
        }
        _write_json(manifest_path, {**identity, **encoder_metadata})

    result = evaluate_market1501(
        query,
        gallery,
        max_rank=config.max_rank,
        retrieval_topk=config.retrieval_topk,
        query_chunk_size=config.query_chunk_size,
    )
    selected = select_retrieval_examples(result.rankings, count=config.example_count)
    sheets: dict[str, str] = {}
    for kind, examples in selected.items():
        if examples:
            sheet_path = output_dir / f"retrieval_{kind}.jpg"
            write_retrieval_sheet(
                query,
                gallery,
                examples,
                sheet_path,
                title=f"Market-1501 {kind} retrievals",
            )
            sheets[kind] = str(sheet_path)

    example_manifest = {
        kind: [_ranking_example_dict(item, query, gallery) for item in examples]
        for kind, examples in selected.items()
    }
    _write_json(output_dir / "retrieval_examples.json", example_manifest)
    metrics = {
        **result.to_dict(),
        **identity,
        **encoder_metadata,
        "query_embeddings": str(query_path),
        "gallery_embeddings": str(gallery_path),
        "reused_embeddings": config.reuse_embeddings,
        "retrieval_sheets": sheets,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone Market-1501 evaluation parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate the trained OSNet checkpoint on Market-1501."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument("--example-count", type=int, default=5)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse matching saved embedding artifacts after provenance validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone evaluator CLI."""
    args = build_parser().parse_args(argv)
    metrics = run_evaluation(
        EvaluationConfig(
            dataset_root=args.dataset_root,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            max_rank=args.max_rank,
            retrieval_topk=args.retrieval_topk,
            example_count=args.example_count,
            query_chunk_size=args.query_chunk_size,
            reuse_embeddings=args.reuse_embeddings,
        )
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
