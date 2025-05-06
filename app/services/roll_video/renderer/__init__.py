"""滚动视频渲染器模块"""

# 导入和导出主要类，使它们可以从外部直接导入
from .text_renderer import TextRenderer
from .video_renderer import VideoRenderer
from .performance import PerformanceMonitor
from .memory_management import FrameMemoryPool
from .frame_processors import _process_frame, _process_frame_optimized_shm
from .utils import limit_resources

# 导出所有公共API
__all__ = [
    "TextRenderer",
    "VideoRenderer",
    "PerformanceMonitor",
    "FrameMemoryPool",
    "_process_frame",
    "_process_frame_optimized",
    "limit_resources",
]
