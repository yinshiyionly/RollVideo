"""
滚动视频渲染模块

此模块负责将文本渲染成滚动视频，提供了:
- 文本渲染到图像
- 视频渲染和动画生成
- 高性能图像处理
- 共享内存和异步处理优化

同时包含了性能监控和资源管理工具
"""

# 从renderer.py导入所有导出的组件
from .renderer import (
    TextRenderer,
    VideoRenderer,
    PerformanceMonitor,
    FrameMemoryPool,
    SharedMemoryFramePool,
    FrameBuffer,
    _process_frame,
    _process_frame_optimized,
    _process_frame_optimized_shm,
    init_shared_memory,
    cleanup_shared_memory,
    init_worker,
    test_worker_shared_memory,
    limit_resources,
    log_system_info,
    FrameProcessingTracker,
    emergency_cleanup,
    get_memory_usage,
    optimize_memory,
    time_tracker
)

# 导出所有组件
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
    "test_worker_shared_memory",
    "limit_resources",
    "log_system_info",
    "FrameProcessingTracker",
    "emergency_cleanup",
    "get_memory_usage",
    "optimize_memory",
    "time_tracker"
]
