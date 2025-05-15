import os
import logging
import platform
from typing import Dict, Tuple, List, Optional, Union

from .renderer.text_renderer import TextRenderer
from .renderer.video_renderer import VideoRenderer

# 配置日志
logger = logging.getLogger(__name__)


class RollVideoService:
    """滚动视频制作服务"""

    def __init__(self):
        """初始化滚动视频服务"""
        # 获取fonts目录
        self.fonts_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fonts"
        )

        # 检查字体文件
        self.available_fonts = self._get_available_fonts()

        # 默认使用自定义字体，如果没有则使用系统默认字体
        if self.available_fonts:
            self.default_font_path = self.available_fonts[0]
            logger.info(f"使用自定义字体: {os.path.basename(self.default_font_path)}")
        else:
            # 系统默认字体路径，根据实际情况调整
            self.default_font_path = self.get_system_default_font()
            logger.warning(
                f"未找到自定义字体，将使用系统默认字体: {self.default_font_path}"
            )

    def _get_available_fonts(self, font_name: Optional[str] = None) -> List[str]:
        """
        获取可用的字体文件

        Args:
            font_name: 可选的指定字体名称（可以不包含扩展名）

        Returns:
            可用字体文件路径列表
        """
        fonts = []

        # 如果指定了字体名称，首先尝试查找匹配的字体
        if font_name:
            # 如果是完整路径且文件存在
            if os.path.isfile(font_name):
                return [font_name]

            # 获取不带扩展名的字体名称（用于匹配）
            base_name = os.path.basename(font_name)
            name_without_ext = os.path.splitext(base_name)[0]

            # 在fonts目录中查找匹配的字体
            if os.path.isdir(self.fonts_dir):
                # 精确匹配（包括扩展名）
                for file in os.listdir(self.fonts_dir):
                    if file.lower() == base_name.lower():
                        found_font = os.path.join(self.fonts_dir, file)
                        return [found_font]

                # 精确匹配（不包括扩展名）
                for file in os.listdir(self.fonts_dir):
                    file_without_ext = os.path.splitext(file)[0]
                    if file_without_ext.lower() == name_without_ext.lower():
                        found_font = os.path.join(self.fonts_dir, file)
                        return [found_font]

            logger.warning(f"找不到指定字体: {font_name}")

        # 如果未指定字体或未找到指定字体，扫描字体目录
        if os.path.isdir(self.fonts_dir):
            # 优先选择方正黑体简体
            fz_font = os.path.join(self.fonts_dir, "FangZhengHeiTiJianTi.ttf")
            if os.path.isfile(fz_font):
                fonts.append(fz_font)

            # 添加目录中的其他ttf/otf字体
            for file in os.listdir(self.fonts_dir):
                if file.lower().endswith((".ttf", ".otf", ".ttc")) and file not in [
                    "FangZhengHeiTiJianTi.ttf",
                ]:
                    fonts.append(os.path.join(self.fonts_dir, file))

        return fonts

    def get_system_default_font(self) -> str:
        """根据不同操作系统获取默认字体路径"""
        system = platform.system()

        if system == "Darwin":  # macOS
            return "/System/Library/Fonts/PingFang.ttc"
        elif system == "Windows":
            return "C:\\Windows\\Fonts\\msyh.ttc"  # 微软雅黑
        elif system == "Linux":
            # 尝试常见的Linux字体路径
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            ]
            for path in font_paths:
                if os.path.exists(path):
                    return path
            return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # 默认返回
        else:
            logger.warning(f"未知操作系统: {system}，使用备用字体")
            return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    def get_font_path(self, font_path: Optional[str] = None) -> str:
        """
        获取字体文件路径

        Args:
            font_path: 指定的字体路径或字体名称

        Returns:
            有效的字体文件路径
        """
        # 如果未指定字体路径，使用默认字体
        if not font_path:
            return self.default_font_path

        # 尝试使用_get_available_fonts方法查找匹配的字体
        available_fonts = self._get_available_fonts(font_path)

        # 如果找到匹配的字体，使用第一个
        if available_fonts:
            return available_fonts[0]

        # 如果找不到指定字体，记录警告并使用系统字体
        logger.warning(f"找不到指定字体，将使用系统默认字体")
        return self.get_system_default_font()

    def create_roll_video_crop(
        self,
        text: str,
        output_path: str,
        width: int = 1080,
        height: int = 1920,
        font_path: Optional[str] = None,
        font_size: int = 40,
        font_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Union[
            Tuple[int, int, int], Tuple[int, int, int, Union[int, float]]
        ] = (
            0,  # 默认黑色背景，不透明
            0,
            0,
            255,
        ),
        background_url: str = None,
        line_spacing: int = 20,
        char_spacing: int = 0,
        fps: int = 60,
        roll_px: float = 1.6,  # 修改为每秒滚动的像素px
        audio_path: Optional[str] = None,
        top_margin: int = 10,      # 默认上边距10px
        bottom_margin: int = 10,   # 默认下边距10px
        left_margin: int = 10,     # 默认左边距10px
        right_margin: int = 10,    # 默认右边距10px
        top_blank: int = 0,        # 默认顶部留白0px
    ) -> Dict[str, Union[str, bool]]:
        """
        使用FFmpeg滤镜创建滚动视频，自动根据透明度选择格式
        
        优势：
        1. 更高效 - 不需要逐帧渲染，直接使用FFmpeg滤镜实现滚动效果
        2. 更平滑 - 滚动效果由FFmpeg实时计算，支持亚像素精度的滚动
        3. 更低内存 - 不需要在内存中处理大量帧
        
        """
        try:
            
            # 标准的背景色处理逻辑
            normalized_bg_color = list(bg_color)
            bg_color_final = tuple(normalized_bg_color)
            
            output_dir = os.path.dirname(os.path.abspath(output_path))
            base_name = os.path.splitext(os.path.basename(output_path))[0]

            preferred_codec = "h264_nvenc"  # 优先尝试 GPU H.264 编码
            actual_output_path = os.path.join(output_dir, f"{base_name}.mp4")

            logger.info(f"开始创建滚动视频 (FFmpeg crop滤镜方式)，实际输出路径: {actual_output_path}")

            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 获取有效的字体路径
            font_path = self.get_font_path(font_path)

            # 创建文字渲染器
            text_renderer = TextRenderer(
                width=width,
                font_path=font_path,
                font_size=font_size,
                font_color=(
                    tuple(font_color) if isinstance(font_color, list) else font_color
                ),
                bg_color=bg_color_final,
                line_spacing=line_spacing,
                char_spacing=char_spacing,
                top_margin=top_margin,       # 传递上边距
                bottom_margin=bottom_margin, # 传递下边距
                left_margin=left_margin,     # 传递左边距
                right_margin=right_margin    # 传递右边距
            )

            # 将文本渲染为图片，并获取文本实际高度
            # 如果指定了背景URL，使用完全透明背景渲染文字
            if background_url:
                text_image, text_actual_height = text_renderer.render_text_to_transparent_image(
                    text, min_height=height
                )
                logger.info("使用完全透明背景渲染文本")
            else:
                # 否则使用普通渲染方法
                text_image, text_actual_height = text_renderer.render_text_to_image(
                    text, min_height=height
                )
            
            logger.info(
                f"文本渲染完成，文本实际高度: {text_actual_height}px, 渲染图像总高度: {text_image.height}px"
            )

            # 估算行高 (字体大小 + 行间距)
            estimated_line_height = font_size + line_spacing

            # roll_px 设定为每秒滚动的像素
            logger.info(
                f"滚动速度设置: {roll_px}像素/帧 (行高约{estimated_line_height}像素)"
            )

            # 创建视频渲染器
            video_renderer = VideoRenderer(
                width=width, height=height, fps=fps, roll_px=roll_px
            )

            # 使用FFmpeg crop滤镜方式创建滚动视频 - 直接传递PIL图像，不转换为numpy数组
            logger.info("开始创建滚动视频 (FFmpeg滤镜方式)...")
            final_output_path = video_renderer.create_scrolling_video_crop(
                image=text_image,  # 直接传递PIL图像对象
                output_path=actual_output_path,  # 使用自动调整后的路径
                text_actual_height=text_actual_height,
                preferred_codec=preferred_codec,  # 传递首选编码器
                audio_path=audio_path,
                bg_color=bg_color_final,  # 传递最终的bg_color供非透明路径使用
                background_url=background_url  # 传递背景图URL
            )

            logger.info(f"滚动视频创建完成 (FFmpeg crop滤镜方式): {final_output_path}")

            return {
                "status": "success",
                "message": "滚动视频创建成功 (FFmpeg crop滤镜方式)",
                "output_path": final_output_path,
            }

        except Exception as e:
            logger.error(f"创建滚动视频失败 (FFmpeg crop滤镜方式): {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"创建滚动视频失败 (FFmpeg crop滤镜方式): {str(e)}",
                "output_path": None,
            }

    def create_roll_video_overlay_cuda(
        self,
        text: str,
        output_path: str,
        width: int = 1080,
        height: int = 1920,
        font_path: Optional[str] = None,
        font_size: int = 40,
        font_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Union[
            Tuple[int, int, int], Tuple[int, int, int, Union[int, float]]
        ] = (
            0,  # 默认黑色背景，不透明
            0,
            0,
            255,
        ),
        background_url: str = None,
        line_spacing: int = 20,
        char_spacing: int = 0,
        fps: int = 60,
        roll_px: float = 1.6,  # 每秒滚动的像素px
        audio_path: Optional[str] = None,
        top_margin: int = 10,      # 默认上边距10px
        bottom_margin: int = 10,   # 默认下边距10px
        left_margin: int = 10,     # 默认左边距10px
        right_margin: int = 10,    # 默认右边距10px
        top_blank: int = 0,        # 默认顶部留白0px
    ) -> Dict[str, Union[str, bool]]:
        """
        使用overlay_cuda GPU滤镜创建滚动视频 - 只支持基础匀速滚动效果和从下到上滚动方向
        
        参数:
            text: 要滚动的文本内容
            output_path: 输出视频文件路径
            width: 视频宽度
            height: 视频高度
            font_path: 字体文件路径
            font_size: 字体大小
            font_color: 字体颜色(R,G,B)
            bg_color: 背景颜色(R,G,B)或(R,G,B,A)
            background_url: 视频容器的背景图片URL，文字将叠加在此背景上
            line_spacing: 行间距
            char_spacing: 字符间距
            fps: 视频帧率
            roll_px: 每秒滚动的像素px
            audio_path: 音频文件路径
            top_margin: 上边距（像素）
            bottom_margin: 下边距（像素）
            left_margin: 左边距（像素）
            right_margin: 右边距（像素）
        
        Returns:
            包含状态、消息和输出路径的字典
        """
        try:
            # 标准的背景色处理逻辑
            normalized_bg_color = list(bg_color)
            normalized_bg_color.append(255)  # RGB 转 RGBA，默认不透明
            normalized_bg_color[3] = max(0, min(255, normalized_bg_color[3]))
            bg_color_final = tuple(normalized_bg_color)
            

            output_dir = os.path.dirname(os.path.abspath(output_path))
            base_name = os.path.splitext(os.path.basename(output_path))[0]

            preferred_codec = "h264_nvenc"  # 优先尝试 GPU H.264 编码
            actual_output_path = os.path.join(output_dir, f"{base_name}.mp4")

            logger.info(f"开始创建滚动视频 (overlay_cuda GPU加速方式)，实际输出路径: {actual_output_path}")

            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 获取有效的字体路径
            font_path = self.get_font_path(font_path)

            # 创建文字渲染器
            text_renderer = TextRenderer(
                width=width,
                font_path=font_path,
                font_size=font_size,
                font_color=(
                    tuple(font_color) if isinstance(font_color, list) else font_color
                ),
                bg_color=bg_color_final,
                line_spacing=line_spacing,
                char_spacing=char_spacing,
                top_margin=top_margin,       # 传递上边距
                bottom_margin=bottom_margin, # 传递下边距
                left_margin=left_margin,     # 传递左边距
                right_margin=right_margin    # 传递右边距
            )

            # 将文本渲染为图片，并获取文本实际高度
            # 如果指定了背景URL，使用完全透明背景渲染文字
            if background_url:
                text_image, text_actual_height = text_renderer.render_text_to_transparent_image(
                    text, min_height=height
                )
                logger.info("使用完全透明背景渲染文本")
            else:
                # 否则使用普通渲染方法
                text_image, text_actual_height = text_renderer.render_text_to_image(
                    text, min_height=height
                )
            
            logger.info(
                f"文本渲染完成，文本实际高度: {text_actual_height}px, 渲染图像总高度: {text_image.height}px"
            )

            # 估算行高 (字体大小 + 行间距)
            estimated_line_height = font_size + line_spacing

            # roll_px 设定为每秒滚动的像素
            logger.info(
                f"滚动速度设置: {roll_px}像素/帧 (行高约{estimated_line_height}像素)"
            )

            # 创建视频渲染器
            video_renderer = VideoRenderer(
                width=width, height=height, fps=fps, roll_px=roll_px
            )

            # 使用FFmpeg CUDA overlay滤镜方式创建滚动视频
            logger.info("开始创建滚动视频 (overlay_cuda GPU加速方式)...")
            final_output_path = video_renderer.create_scrolling_video_overlay_cuda(
                image=text_image,
                output_path=actual_output_path,
                text_actual_height=text_actual_height,
                preferred_codec=preferred_codec,
                audio_path=audio_path,
                bg_color=bg_color_final,
                background_url=background_url
            )

            logger.info(f"滚动视频创建完成 (overlay_cuda GPU加速方式): {final_output_path}")

            return {
                "status": "success",
                "message": "滚动视频创建成功 (overlay_cuda GPU加速方式)",
                "output_path": final_output_path,
            }

        except Exception as e:
            logger.error(f"创建滚动视频失败 (overlay_cuda GPU加速方式): {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"创建滚动视频失败 (overlay_cuda GPU加速方式): {str(e)}",
                "output_path": None,
            }
