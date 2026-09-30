"""Composable camera processing pipelines."""

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
    "SingleCameraPipeline",
    "extract_person_crops",
]
