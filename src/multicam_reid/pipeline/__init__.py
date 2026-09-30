"""Composable camera processing pipelines."""

from multicam_reid.pipeline.reid_camera import ReIDCameraPipeline, ReIDProcessedFrame
from multicam_reid.pipeline.single_camera import (
    CropConfig,
    PersonCrop,
    ProcessedFrame,
    SingleCameraPipeline,
    extract_person_crops,
)

__all__ = [
    "CropConfig",
    "PersonCrop",
    "ProcessedFrame",
    "ReIDCameraPipeline",
    "ReIDProcessedFrame",
    "SingleCameraPipeline",
    "extract_person_crops",
]
