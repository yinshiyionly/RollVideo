import os
import logging
import platform
from typing import Dict, Tuple, List, Optional, Union
from PIL import Image

from renderer import TextRenderer, VideoRenderer

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
        
        # 检测系统的CPU核心数，用于决定默认线程数
        self.cpu_count = os.cpu_count() or 4
        logger.info(f"检测到系统CPU核心数: {self.cpu_count}")

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
        width: int = 1080,
        height: int = 1920,
        font_path: Optional[str] = None,
        font_size: int = 40,
        font_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Union[Tuple[int, int, int], Tuple[int, int, int, Union[int, float]]] = (
            0, # 默认黑色背景，不透明
            0,
            0,
            255,
        ),
        line_spacing: int = 20,
        char_spacing: int = 0,
        fps: int = 30,
        scroll_speed: int = 2,
        audio_path: Optional[str] = None,
        worker_threads: Optional[int] = None,
        frame_buffer_size: Optional[int] = None,
    ) -> Dict[str, Union[str, bool]]:
        """
        创建滚动视频，自动根据透明度选择CPU/GPU和格式。
        透明: CPU + prores_ks + mov
        不透明: 尝试 GPU (h264_nvenc) 回退 CPU (libx264) + mp4
        
        Args:
            text: 要展示的文本内容
            output_path: 期望的输出视频路径（扩展名会被自动调整）
            width: 视频宽度
            height: 视频高度
            font_path: 字体文件路径，不指定则使用默认字体
            font_size: 字体大小
            font_color: 字体颜色 (R,G,B)
            bg_color: 背景颜色，可以是RGB元组(r,g,b)或RGBA元组(r,g,b,a)，a为透明度，0完全透明，255完全不透明，支持float类型的alpha值
            line_spacing: 行间距
            char_spacing: 字符间距
            fps: 视频帧率
            scroll_speed: 滚动速度(像素/帧)
            audio_path: 可选的音频文件路径
            worker_threads: 用于帧处理的工作线程数 (默认为CPU核心数或4)
            frame_buffer_size: 帧缓冲区大小 (默认为fps的80%)
            
        Returns:
            包含处理结果的字典
        """
        try:
            # --- 决定透明度需求和编码策略 --- 
            normalized_bg_color = list(bg_color)
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

            transparency_required = bg_color_final[3] < 255
            
            # 根据透明度需求确定输出格式和编码器
            output_dir = os.path.dirname(os.path.abspath(output_path))
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            
            if transparency_required:
                preferred_codec = "prores_ks" # 高质量透明编码（CPU）
                actual_output_path = os.path.join(output_dir, f"{base_name}.mov")
                logger.info("检测到透明背景需求，将使用 CPU (prores_ks) 输出 .mov 文件。")
            else:
                preferred_codec = "h264_nvenc" # 优先尝试 GPU H.264 编码
                actual_output_path = os.path.join(output_dir, f"{base_name}.mp4")
                logger.info("无透明背景需求，将优先尝试 GPU (h264_nvenc) 输出 .mp4 文件。")
            # --- 结束决策 --- 

            logger.info(f"开始创建滚动视频，实际输出路径: {actual_output_path}")
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取有效的字体路径
            font_path = self.get_font_path(font_path)
            
            # 创建文字渲染器 (使用最终确定的bg_color)
            text_renderer = TextRenderer(
                width=width,
                font_path=font_path,
                font_size=font_size,
                font_color=font_color,
                bg_color=bg_color_final, 
                line_spacing=line_spacing,
                char_spacing=char_spacing,
            )

            # 将文本渲染为图片，并获取文本实际高度
            logger.info("将文本渲染为图片...")
            text_image, text_actual_height = text_renderer.render_text_to_image(text, min_height=height)
            logger.info(f"文本实际高度: {text_actual_height}px, 渲染图像总高度: {text_image.height}px")

            # 设置线程数和缓冲区大小
            if worker_threads is None:
                worker_threads = min(self.cpu_count, 8)  # 限制最大线程数为8
            
            if frame_buffer_size is None:
                frame_buffer_size = min(int(fps * 0.8), 24)  # 默认为fps的80%，但不超过24
            
            logger.info(f"使用线程数: {worker_threads}, 帧缓冲区大小: {frame_buffer_size}")

            # 创建视频渲染器，传递优化参数
            video_renderer = VideoRenderer(
                width=width, 
                height=height, 
                fps=fps, 
                scroll_speed=scroll_speed,
                worker_threads=worker_threads,
                frame_buffer_size=frame_buffer_size
            )

            # 创建滚动视频，传递决策结果
            logger.info("开始创建滚动视频...")
            final_output_path = video_renderer.create_scrolling_video(
                image=text_image,
                output_path=actual_output_path, # 使用自动调整后的路径
                text_actual_height=text_actual_height,
                transparency_required=transparency_required, # 传递透明度需求
                preferred_codec=preferred_codec, # 传递首选编码器
                audio_path=audio_path,
                bg_color=bg_color_final # 传递最终的bg_color供非透明路径使用
            )

            logger.info(f"滚动视频创建完成: {final_output_path}")

            return {
                "status": "success",
                "message": "滚动视频创建成功",
                "output_path": final_output_path,
            }

        except Exception as e:
            logger.error(f"创建滚动视频失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"创建滚动视频失败: {str(e)}",
                "output_path": None,
            }
