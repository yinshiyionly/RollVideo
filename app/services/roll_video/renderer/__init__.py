"""滚动视频渲染器模块"""

# 导入和导出主要类，使它们可以从外部直接导入
from .text_renderer import TextRenderer
from .video_renderer import VideoRenderer
from .performance import PerformanceMonitor, log_system_info, FrameProcessingTracker
from .memory_management import FrameMemoryPool, SharedMemoryFramePool, FrameBuffer
from .frame_processors import (
    init_shared_memory,
    cleanup_shared_memory,
    init_worker,
    _process_frame_optimized_shm,
)
from .utils import limit_resources

# 导出所有公共API
__all__ = [
    "TextRenderer",
    "VideoRenderer",
    "PerformanceMonitor",
    "FrameMemoryPool",
    "SharedMemoryFramePool",
    "FrameBuffer",
    "init_shared_memory",
    "cleanup_shared_memory",
    "init_worker",
    "_process_frame_optimized_shm",
    "log_system_info",
    "FrameProcessingTracker",
    "limit_resources",
]
