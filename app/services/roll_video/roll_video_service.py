import os
import logging
import platform
from typing import Dict, Tuple, List, Optional, Union, Callable
from PIL import Image
import io

from app.services.roll_video.renderer import TextRenderer, VideoRenderer

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
            logger.info(f"将使用自定义字体: {os.path.basename(self.default_font_path)}")
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
                logger.info(f"使用指定字体路径: {font_name}")
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
                        logger.info(f"找到精确匹配字体: {found_font}")
                        return [found_font]
                
                # 精确匹配（不包括扩展名）
                for file in os.listdir(self.fonts_dir):
                    file_without_ext = os.path.splitext(file)[0]
                    if file_without_ext.lower() == name_without_ext.lower():
                        found_font = os.path.join(self.fonts_dir, file)
                        logger.info(f"找到字体名匹配: {found_font}")
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

    def list_available_fonts(self) -> List[str]:
        """列出所有可用的字体文件名"""
        return [os.path.basename(font) for font in self.available_fonts]

    def create_roll_video(
        self,
        text: str,
        output_path: str,
        bg_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
        fps: int = 30,
        scale_factor: float = 0.75,
        frame_skip: int = 1,  # 默认不跳帧
        scroll_speed: int = 2,
        with_audio: bool = False,
        width: int = 720,
        height: int = 1280,
        font_size: int = 40,
        font_color: Tuple[int, int, int] = (255, 255, 255),
        transparent: bool = False,
        font_path: Optional[str] = None,
        line_spacing: int = 10,
        char_spacing: int = 0,
        audio_path: Optional[str] = None,
        override_temp_working_dir: Optional[str] = None,
        error_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Union[str, bool]]:
        """
        创建滚动视频，自动根据透明度选择适合的格式。
        使用优化后的多线程渲染引擎以获得最佳性能和流畅度。
        
        Args:
            text: 要展示的文本内容
            output_path: 期望的输出视频路径（扩展名会被自动调整）
            bg_color: 背景颜色，可以是RGB(A)元组或列表(r,g,b)或(r,g,b,a)，a为透明度(0-255或0.0-1.0)
            fps: 视频帧率
            scale_factor: 缩放因子(0.5-1.0)，降低处理分辨率以提高速度
            frame_skip: 跳帧率，默认为1不跳帧，提高值会减少平滑度但提高渲染速度
            scroll_speed: 滚动速度(像素/帧)
            with_audio: 是否需要音频
            width: 视频宽度
            height: 视频高度
            font_size: 字体大小
            font_color: 字体颜色 (R,G,B)，可以是列表或元组
            transparent: 是否需要透明背景
            font_path: 字体文件路径，不指定则使用默认字体
            line_spacing: 行间距
            char_spacing: 字符间距
            audio_path: 可选的音频文件路径
            override_temp_working_dir: 可选的临时工作目录
            error_callback: 错误回调函数
            
        Returns:
            包含处理结果的字典
        """
        try:
            # --- 参数类型转换和预处理 ---
            # 确保颜色是元组
            font_color_tuple = tuple(font_color) if isinstance(font_color, list) else font_color
            bg_color_tuple = tuple(bg_color) if isinstance(bg_color, list) else bg_color
            
            # 自适应帧率 - 根据滚动速度调整
            adjusted_fps = fps
            if scroll_speed > 4:
                # 对于快速滚动，自动提高帧率以保持视觉流畅度
                recommended_fps = min(30, int(scroll_speed * 6))  # 基于滚动速度计算推荐帧率
                adjusted_fps = max(fps, recommended_fps)
                if adjusted_fps != fps:
                    logger.info(f"滚动速度较快({scroll_speed}像素/帧)，自动调整帧率: {fps} → {adjusted_fps}fps，以保持视觉流畅")
                    fps = adjusted_fps
            
            # 计算实际行间距（像素）
            actual_line_spacing_pixels = int(font_size * (line_spacing / 100)) if line_spacing > 0 else line_spacing

            # --- 决定透明度需求和编码策略 --- 
            normalized_bg_color = list(bg_color_tuple)
            if len(normalized_bg_color) == 3:
                normalized_bg_color.append(255) # RGB 转 RGBA，默认不透明
            elif len(normalized_bg_color) == 4 and isinstance(normalized_bg_color[3], float):
                 # 将 float alpha (0.0-1.0) 转为 int (0-255)
                if 0 <= normalized_bg_color[3] <= 1:
                    normalized_bg_color[3] = int(normalized_bg_color[3] * 255)
                else:
                    normalized_bg_color[3] = int(normalized_bg_color[3]) # 超出范围直接取整
            # 确保 alpha 值在 0-255 范围内
            normalized_bg_color[3] = max(0, min(255, normalized_bg_color[3])) 
            bg_color_final = tuple(normalized_bg_color)

            # 检测是否需要透明
            should_be_transparent = transparent or bg_color_final[3] < 255
            
            # 限制缩放因子范围
            if scale_factor < 0.5:
                scale_factor = 0.5
                logger.warning(f"缩放因子太小，已调整为最小值0.5")
            elif scale_factor > 1.0:
                scale_factor = 1.0
                logger.warning(f"缩放因子不应大于1.0，已调整为1.0")
                
            # 限制跳帧率范围
            if frame_skip < 1:
                frame_skip = 1
                logger.warning(f"跳帧率不能小于1，已调整为1")
                
            # 应用缩放因子（如果不是1.0）
            render_width = width
            render_height = height
            render_font_size = font_size
            render_line_spacing = actual_line_spacing_pixels
            render_scroll_speed = scroll_speed
            
            if scale_factor < 1.0 and not should_be_transparent:
                logger.info(f"使用缩放因子: {scale_factor}，降低处理分辨率以提高速度")
                render_width = int(width * scale_factor)
                render_height = int(height * scale_factor)
                render_font_size = int(font_size * scale_factor)
                render_line_spacing = int(actual_line_spacing_pixels * scale_factor)
                render_scroll_speed = max(1, int(scroll_speed * scale_factor))
                logger.info(f"原始分辨率: {width}x{height}, 渲染分辨率: {render_width}x{render_height}")
                logger.info(f"原始字体大小: {font_size}, 渲染字体大小: {render_font_size}")
                logger.info(f"原始滚动速度: {scroll_speed}, 渲染滚动速度: {render_scroll_speed}")

            # 确保输出目录存在
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取有效的字体路径
            actual_font_path = self.get_font_path(font_path)
            
            # 创建文字渲染器 (使用最终确定的bg_color和计算后的行距)
            text_renderer = TextRenderer(
                width=render_width,
                font_path=actual_font_path,
                font_size=render_font_size,
                font_color=font_color_tuple, 
                bg_color=bg_color_final, 
                line_spacing=render_line_spacing, 
                char_spacing=char_spacing,
            )

            # 将文本渲染为图片，并获取文本实际高度
            logger.info("将文本渲染为图片...")
            text_image, text_height = text_renderer.render_text_to_image(text, min_height=render_height)
            logger.info(f"文本实际高度: {text_height}px, 渲染图像总高度: {text_image.height}px")

            # 创建视频渲染器 (直接使用新的优化版VideoRenderer)
            video_renderer = VideoRenderer(
                width=render_width, 
                height=render_height, 
                fps=fps, 
                output_path=output_path,
                frame_skip=frame_skip,
                scale_factor=1.0,  # 已经在输入尺寸上应用过缩放，这里不需要再缩放
                with_audio=with_audio,
                audio_path=audio_path,
                transparent=should_be_transparent,
                override_temp_working_dir=override_temp_working_dir,
                error_callback=error_callback
            )
            
            # 计算总帧数
            total_frames = video_renderer.calculate_total_frames(text_height, render_scroll_speed)
            
            # 创建帧生成函数
            def frame_generator(frame_index):
                # 计算当前滚动位置
                scroll_pos = frame_index * render_scroll_speed
                
                # 创建视频帧
                if should_be_transparent:
                    # 透明背景
                    frame = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))
                else:
                    # 使用指定背景色
                    frame = Image.new("RGB", (render_width, render_height), bg_color_final[:3])
                
                # 计算文本位置 (从底部向上滚动)
                text_y = render_height - scroll_pos
                
                # 将文本图像粘贴到当前帧
                frame.paste(text_image, (0, text_y), text_image if text_image.mode == "RGBA" else None)
                
                # 转换为字节流
                buffer = io.BytesIO()
                frame_format = "raw"
                if should_be_transparent:
                    frame.save(buffer, format="PNG")
                    frame_bytes = buffer.getvalue()
                else:
                    frame_bytes = frame.tobytes()
                
                return frame_bytes
            
            # 开始渲染视频
            logger.info(f"开始渲染滚动视频，总帧数: {total_frames}...")
            success = video_renderer.render_frames(total_frames, frame_generator)
            
            if success:
                logger.info(f"滚动视频创建成功: {output_path}")
                return {
                    "status": "success",
                    "message": "滚动视频创建成功",
                    "output_path": output_path,
                }
            else:
                error_msg = "视频渲染失败，请检查日志获取详细信息"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "output_path": None,
                }

        except Exception as e:
            logger.error(f"创建滚动视频失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"创建滚动视频失败: {str(e)}",
                "output_path": None,
            }