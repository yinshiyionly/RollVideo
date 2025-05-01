import os
import logging
import platform
from typing import Dict, Tuple, List, Optional, Union, Callable
from PIL import Image, ImageFont
import io

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
        """获取系统默认字体"""
        # 尝试寻找内置字体
        font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
        if os.path.exists(font_dir):
            # 首先尝试查找中文字体
            for chinese_font in ['方正黑体简体.ttf', 'MicroSoftYaHei.ttf', 'SimHei.ttf', 'NotoSansSC-Regular.otf']:
                font_path = os.path.join(font_dir, chinese_font)
                if os.path.exists(font_path):
                    logger.info(f"将使用自定义字体: {chinese_font}")
                    return font_path
                    
            # 如果没有中文字体，使用任何可用字体
            font_files = [f for f in os.listdir(font_dir) if f.endswith(('.ttf', '.otf'))]
            if font_files:
                font_path = os.path.join(font_dir, font_files[0])
                logger.info(f"将使用自定义字体: {font_files[0]}")
                return font_path
                
        # 如果内置字体目录不存在或没有找到字体，尝试系统字体
        try:
            system = platform.system()
            if system == 'Windows':
                return os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'Fonts', 'arial.ttf')
            elif system == 'Darwin':  # macOS
                return '/System/Library/Fonts/STHeiti Light.ttc'
            else:  # Linux
                # 尝试常见的Linux字体路径
                common_fonts = [
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    '/usr/share/fonts/TTF/DejaVuSans.ttf',
                    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
                ]
                for font in common_fonts:
                    if os.path.exists(font):
                        return font
        except Exception as e:
            logger.warning(f"查找系统字体时出错: {str(e)}")
            
        # 最终回退：使用PIL默认字体
        logger.warning("未找到合适的字体，将使用Pillow默认字体")
        return ImageFont.load_default().path

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
        width: int = 720,
        height: int = 1280,
        font_path: Optional[str] = None,
        font_size: int = 24,
        font_color: Tuple[int, int, int] = (0, 0, 0),
        bg_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        fps: int = 30,
        scroll_speed: int = 1,
        with_audio: bool = False,
        audio_path: Optional[str] = None,
        scale_factor: float = 1.0,
        frame_skip: int = 1,
        transparent: bool = False,
        line_spacing: int = 10,  # 添加行间距参数
        char_spacing: int = 0,   # 添加字符间距参数
        respect_original_newlines: bool = True,  # 添加是否尊重原始换行参数
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        创建滚动视频，将文本从下向上滚动
        
        Args:
            text: 要显示的文本内容
            output_path: 输出视频路径
            width: 视频宽度，默认720
            height: 视频高度，默认1280
            font_path: 字体路径，默认使用系统默认字体
            font_size: 字体大小，默认24
            font_color: 字体颜色RGB元组，默认黑色
            bg_color: 背景颜色RGBA元组，默认白色不透明
            fps: 视频帧率，默认30fps
            scroll_speed: 滚动速度，单位像素/帧，默认1
            with_audio: 是否添加音频，默认False
            audio_path: 音频文件路径，仅当with_audio=True时有效
            scale_factor: 缩放因子，用于减少处理分辨率以提高性能，默认1.0
            frame_skip: 帧间隔，跳过的帧数，默认1（不跳过）
            transparent: 是否透明背景，默认False
            line_spacing: 行间距(像素)，默认10
            char_spacing: 字符间距(像素)，默认0
            respect_original_newlines: 是否尊重原始文本中的换行符，默认True
            error_callback: 错误回调函数，默认None
        
        Returns:
            输出视频路径
        """
        logger.info(f"使用缩放因子: {scale_factor}，降低处理分辨率以提高速度")
        # 应用缩放因子调整渲染分辨率
        scaled_width = int(width * scale_factor)
        scaled_height = int(height * scale_factor)
        scaled_font_size = int(font_size * scale_factor)
        scaled_scroll_speed = max(1, int(scroll_speed * scale_factor))
        
        logger.info(f"原始分辨率: {width}x{height}, 渲染分辨率: {scaled_width}x{scaled_height}")
        logger.info(f"原始字体大小: {font_size}, 渲染字体大小: {scaled_font_size}")
        logger.info(f"原始滚动速度: {scroll_speed}, 渲染滚动速度: {scaled_scroll_speed}")
        
        # 设置文本样式参数
        scaled_line_spacing = int(line_spacing * scale_factor)
        scaled_char_spacing = int(char_spacing * scale_factor)
        logger.info(f"原始行间距: {line_spacing}px, 渲染行间距: {scaled_line_spacing}px")
        logger.info(f"原始字符间距: {char_spacing}px, 渲染字符间距: {scaled_char_spacing}px")
        
        # 处理文本的换行符
        if respect_original_newlines:
            # 保留原始文本中的换行符，转换为内部换行标记
            text = text.replace('\n', '\\n')
            logger.info("尊重原始文本换行")
        else:
            # 移除所有原始换行，依赖排版系统自动换行
            text = text.replace('\n', ' ')
            logger.info("忽略原始文本换行，使用自动换行")
        
        # 查找字体文件
        if font_path:
            # 如果提供了具体的字体路径，直接使用
            if not os.path.exists(font_path):
                # 尝试在字体目录中查找
                font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
                font_path_in_dir = os.path.join(font_dir, os.path.basename(font_path))
                if os.path.exists(font_path_in_dir):
                    font_path = font_path_in_dir
                    logger.info(f"找到精确匹配字体: {font_path}")
                else:
                    logger.warning(f"找不到字体文件: {font_path}，尝试使用默认字体")
                    font_path = self.get_system_default_font()
        else:
            # 如果没有提供字体路径，使用默认字体
            font_path = self.get_system_default_font()
            
        # 初始化渲染器
        text_renderer = TextRenderer(
            width=scaled_width,
            font_path=font_path,
            font_size=scaled_font_size,
            font_color=font_color,
            bg_color=bg_color,
            line_spacing=scaled_line_spacing,  # 使用缩放后的行间距
            char_spacing=scaled_char_spacing   # 使用缩放后的字符间距
        )
        
        # 渲染文本为图片
        logger.info("将文本渲染为图片...")
        img, text_height = text_renderer.render_text_to_image(text, scaled_height)
        
        # 创建视频渲染器
        video_renderer = VideoRenderer(
            width=scaled_width,
            height=scaled_height,
            fps=fps,
            output_path=output_path,
            frame_skip=frame_skip,
            scale_factor=1.0,  # 这里已经应用了外部缩放，所以内部不再缩放
            with_audio=with_audio,
            audio_path=audio_path,
            transparent=transparent,
            error_callback=error_callback
        )
        
        # 计算总帧数
        total_frames = video_renderer.calculate_total_frames(text_height, scaled_scroll_speed)
        
        # 开始渲染视频
        logger.info(f"开始渲染滚动视频，总帧数: {total_frames}...")
        
        # 创建帧生成器
        frame_generator = self._create_frame_generator(img, video_renderer, scaled_scroll_speed)
        
        # 渲染视频
        success = video_renderer.render_frames(total_frames, frame_generator)
        
        if success:
            logger.info(f"滚动视频创建成功: {output_path}")
            return output_path
        else:
            error_message = f"创建滚动视频失败: {output_path}"
            logger.error(error_message)
            if error_callback:
                error_callback(error_message)
            return ""

    def _create_frame_generator(self, img, video_renderer, scroll_speed):
        """
        创建帧生成器函数
        
        Args:
            img: 渲染的文本图像
            video_renderer: 视频渲染器实例
            scroll_speed: 滚动速度
            
        Returns:
            帧生成器函数
        """
        render_width = video_renderer.width
        render_height = video_renderer.height
        should_be_transparent = video_renderer.transparent
        
        # 获取背景色，防止透明视频的背景也是透明的
        if hasattr(img, 'info') and 'background' in img.info:
            bg_color = img.info['background'][:3]
        else:
            # 默认白色背景
            bg_color = (255, 255, 255)
        
        # 返回帧生成函数
        def frame_generator(frame_index):
            try:
                # 修改: 从第一帧开始就在屏幕底部显示文字，而不是在屏幕外
                # 原始代码: scroll_pos = (frame_index + 1) * scroll_speed
                scroll_pos = frame_index * scroll_speed  # 移除 +1，让第一帧文字就出现在底部
                
                # 创建视频帧
                if should_be_transparent:
                    frame = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))
                else:
                    frame = Image.new("RGB", (render_width, render_height), bg_color)
                
                # 计算文本图像顶部应该在当前帧的哪个 Y 坐标
                # 当 scroll_pos = 0, text_y = render_height (文本在屏幕正下方)
                # 当 scroll_pos = render_height, text_y = 0 (文本顶部到达屏幕顶部)
                # 当 scroll_pos = render_height + img.height, text_y = -img.height (文本底部离开屏幕顶部)
                text_y = render_height - scroll_pos
                
                # --- Corrected Cropping and Pasting Logic --- 
                # 确定需要从源文本图像 (img) 上裁剪的 Y 范围
                src_y_start = max(0, -text_y) # 源图像裁剪起始 Y
                src_y_end = min(img.height, render_height - text_y) # 源图像裁剪结束 Y
                
                # 计算实际裁剪高度
                crop_height = src_y_end - src_y_start

                # 确定在目标帧 (frame) 上粘贴的起始 Y 坐标
                paste_y = max(0, text_y) # 目标帧粘贴起始 Y

                # 只有当实际裁剪高度大于0时才进行裁剪和粘贴
                if crop_height > 0:
                    try:
                        # 从源图像裁剪
                        visible_crop = img.crop((
                            0,             # left
                            src_y_start,   # top
                            render_width,  # right
                            src_y_end      # bottom
                        ))
                        
                        # 粘贴到目标帧的正确位置
                        # 如果源图像是 RGBA，使用其 alpha 通道作为 mask
                        mask = visible_crop if visible_crop.mode == 'RGBA' else None
                        frame.paste(visible_crop, (0, paste_y), mask=mask)
                        
                        # 显式删除裁剪对象释放内存
                        del visible_crop
                        
                    except Exception as e:
                        logger.error(f"裁剪或粘贴帧 {frame_index} 时出错: {e}, text_y={text_y}, src_y=({src_y_start},{src_y_end}), paste_y={paste_y}", exc_info=True)
                # else: # crop_height <= 0，表示文本完全在屏幕外，不需要粘贴
                #    logger.debug(f"Frame {frame_index}: Text out of view (text_y={text_y}, crop_height={crop_height})")
                # --- End of Corrected Logic --- 

                # 转换为字节流
                buffer = io.BytesIO()
                if should_be_transparent:
                    frame.save(buffer, format="PNG")
                    frame_bytes = buffer.getvalue()
                else:
                    # 对于非透明视频，直接使用更高效的raw格式
                    frame_bytes = frame.tobytes()
                
                # 清理以释放内存
                buffer.close()
                del frame
                
                return frame_bytes
            
            except Exception as e:
                import traceback
                logger.error(f"生成帧 {frame_index} 时出错: {str(e)}\n{traceback.format_exc()}")
                # 遇到错误时，返回一个空帧或默认帧而不是None，确保流程能继续
                if should_be_transparent:
                    default_frame = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))
                    buffer = io.BytesIO()
                    default_frame.save(buffer, format="PNG")
                    return buffer.getvalue()
                else:
                    default_frame = Image.new("RGB", (render_width, render_height), bg_color)
                    return default_frame.tobytes()
                    
        return frame_generator