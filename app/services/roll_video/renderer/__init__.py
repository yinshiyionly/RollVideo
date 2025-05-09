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
    PerformanceMonitor
)

# 导出所有组件
__all__ = [
    "TextRenderer",
    "VideoRenderer",
    "PerformanceMonitor"
]
