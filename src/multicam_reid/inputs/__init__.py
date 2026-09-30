"""Frame sources that normalize videos and cameras into FramePacket objects."""

from multicam_reid.inputs.ip_camera import IpCameraConfig, IpCameraWorker
from multicam_reid.inputs.video import VideoFileSource

__all__ = ["IpCameraConfig", "IpCameraWorker", "VideoFileSource"]
