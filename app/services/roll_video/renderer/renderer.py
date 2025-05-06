"""
滚动视频渲染模块

此模块负责将文本渲染成滚动视频，主要包含:
1. TextRenderer - 将文本渲染为图像
2. VideoRenderer - 创建滚动效果的视频
"""

# 直接从子模块导入所有组件
from .text_renderer import TextRenderer
from .video_renderer import VideoRenderer
from .performance import PerformanceMonitor
from .memory_management import FrameMemoryPool
from .frame_processors import _process_frame_optimized
from .utils import limit_resources

__all__ = [
    "TextRenderer",
    "VideoRenderer",
    "PerformanceMonitor",
    "FrameMemoryPool",
    "_process_frame_optimized",
    "limit_resources",
]
