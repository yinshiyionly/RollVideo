"""
滚动视频渲染模块

此模块负责将文本渲染成滚动视频，主要包含:
1. TextRenderer - 将文本渲染为图像
2. VideoRenderer - 创建滚动效果的视频

优化版本包含更多高性能组件:
- 共享内存处理
- 异步帧处理
- 高效内存管理
"""

# 直接从子模块导入所有组件
from .text_renderer import TextRenderer
from .video_renderer import VideoRenderer
from .performance import PerformanceMonitor, log_system_info, FrameProcessingTracker
from .memory_management import FrameMemoryPool, SharedMemoryFramePool, FrameBuffer
from .frame_processors import (
    _process_frame, 
    _process_frame_optimized,
    _process_frame_optimized_shm,
    init_shared_memory,
    cleanup_shared_memory,
    init_worker
)
from .utils import limit_resources

__all__ = [
    "TextRenderer",
    "VideoRenderer",
    "PerformanceMonitor",
    "FrameMemoryPool",
    "SharedMemoryFramePool",
    "FrameBuffer",
    "_process_frame",
    "_process_frame_optimized",
    "_process_frame_optimized_shm",
    "init_shared_memory",
    "cleanup_shared_memory",
    "init_worker",
    "limit_resources",
    "log_system_info",
    "FrameProcessingTracker",
]
