"""Reproducible model and system evaluation utilities."""

from multicam_reid.evaluation.market1501 import (
    EmbeddingSet,
    EvaluationConfig,
    Market1501Evaluation,
    evaluate_market1501,
    run_evaluation,
)

__all__ = [
    "EmbeddingSet",
    "EvaluationConfig",
    "Market1501Evaluation",
    "evaluate_market1501",
    "run_evaluation",
]
