"""滚动视频服务主包"""

# 继续导出RollVideoService作为主要API
from app.services.roll_video.roll_video_service import RollVideoService

# 也导出渲染组件以便直接访问
from app.services.roll_video.renderer import TextRenderer, VideoRenderer

__all__ = [
    "RollVideoService",
    "TextRenderer",
    "VideoRenderer"
] 