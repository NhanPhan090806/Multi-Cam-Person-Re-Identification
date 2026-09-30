"""Frame sources that normalize videos and cameras into FramePacket objects."""

from multicam_reid.inputs.epfl_lab import EpflLabSource, EpflLabSynchronizedFrame
from multicam_reid.inputs.ip_camera import IpCameraConfig, IpCameraWorker
from multicam_reid.inputs.video import VideoFileSource
from multicam_reid.inputs.wildtrack import WildtrackSource, WildtrackSynchronizedFrame

__all__ = [
    "EpflLabSource",
    "EpflLabSynchronizedFrame",
    "IpCameraConfig",
    "IpCameraWorker",
    "VideoFileSource",
    "WildtrackSource",
    "WildtrackSynchronizedFrame",
]
