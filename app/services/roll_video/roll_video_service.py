import os
import logging
import platform
from typing import Dict, Tuple, List, Optional, Union, Callable
from PIL import Image, ImageFont
import io
import numpy as np

from renderer import TextRenderer, VideoRenderer, blend_alpha_fast

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
        
        # 最终回退：使用Pillow默认字体，并返回空字符串让上层回退
        logger.warning("未找到合适的字体，将使用Pillow默认字体")
        try:
            ImageFont.load_default()
            logger.info("已加载Pillow内置默认字体")
        except Exception:
            logger.warning("加载Pillow默认字体失败")
        # 返回空字符串，上层TextRenderer会在加载失败时回退到内置字体
        return ""

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
        else:
            # 移除所有原始换行，依赖排版系统自动换行
            text = text.replace('\n', ' ')
        
        # 查找并确认最终字体路径
        final_font_path = self.get_font_path(font_path)

        # 记录传递给 TextRenderer 的参数
        logger.info(f"TextRenderer 参数: width={scaled_width}, font_path={final_font_path}, font_size={scaled_font_size}, "
                    f"font_color={font_color}, bg_color={bg_color}, line_spacing={scaled_line_spacing}, "
                    f"char_spacing={scaled_char_spacing}")

        # 初始化 TextRenderer
        try:
            text_renderer = TextRenderer(
                width=scaled_width,
                font_path=final_font_path,  # 使用确认后的路径
                font_size=scaled_font_size,
                font_color=font_color,
                bg_color=bg_color,
                line_spacing=scaled_line_spacing,
                char_spacing=scaled_char_spacing,
            )
        except Exception as e:
            logger.error(f"初始化 TextRenderer 失败，字体: {final_font_path}, 错误: {e}", exc_info=True)
            if error_callback:
                error_callback(f"字体加载或渲染器初始化失败: {os.path.basename(final_font_path)} - {e}")
            # 可以根据需要决定是抛出异常还是返回错误信息
            raise  # 重新抛出异常，以便上层捕获

        # 使用 TextRenderer 渲染文本为图片
        try:
            text_img, text_height = text_renderer.render_text_to_image(
                text,
                min_height=scaled_height # 传递缩放后的高度作为最小高度参考
            )
        except Exception as e:
            logger.error(f"文本渲染为图片时出错: {e}", exc_info=True)
            if error_callback:
                error_callback(f"文本渲染失败: {e}")
            raise # 重新抛出异常
            
        logger.info(f"文本图片已生成，尺寸: {text_img.size}, 文本高度: {text_height}")

        # --- 计算滚动结束帧 --- 
        # 注意：render_text_to_image已经创建了一个高度为(text_height + screen_height)的图像
        # 所以图像本身已经包含了所需的空白区域，不需要再加屏幕高度
        # 只需要滚动整个图像的高度即可
        img_height = text_img.size[1]  # 获取图像实际高度
        scroll_frames_needed = int(np.ceil(img_height / scaled_scroll_speed))
        logger.info(f"计算得到文本滚出所需帧数: {scroll_frames_needed} (图像总高度={img_height}px, 文本实际高度={text_height}px)")
        # ---------------------

        # 初始化 VideoRenderer
        video_renderer = VideoRenderer(
            width=width, # 使用原始宽度
            height=height, # 使用原始高度
            fps=fps,
            output_path=output_path,
            frame_skip=frame_skip,
            scale_factor=1.0, # VideoRenderer 内部不再处理缩放，TextRenderer已处理
            with_audio=with_audio,
            audio_path=audio_path,
            transparent=transparent, # 传递透明背景选项
            error_callback=error_callback,
        )

        # 计算总帧数
        # 注意：这里使用原始的滚动速度和缩放后的文本高度
        # 因为滚动是在最终分辨率下进行的，但文本内容的高度是在渲染分辨率下确定的
        total_frames = video_renderer.calculate_total_frames(
            text_height=text_height, # 使用渲染后的文本高度
            scroll_speed=scaled_scroll_speed # 使用渲染分辨率下的滚动速度
        )
        logger.info(f"计算得到的总帧数: {total_frames}")

        # 创建帧生成器，传递用户指定的背景色
        frame_generator = self._create_frame_generator(
            img=text_img,                # 渲染好的文本图片
            video_renderer=video_renderer,# VideoRenderer实例
            scroll_speed=scaled_scroll_speed,# 渲染分辨率下的滚动速度
            scroll_frames_needed=scroll_frames_needed,# 滚动结束帧
            bg_color=bg_color           # 用户指定的背景色RGBA
        )

        # 开始渲染视频帧
        try:
            video_renderer.render_frames(
                total_frames=total_frames,
                frame_generator=frame_generator
            )
        except Exception as e:
            logger.error(f"视频帧渲染过程中出错: {e}", exc_info=True)
            if error_callback:
                error_callback(f"视频渲染失败: {e}")
            raise # 重新抛出异常

        logger.info(f"滚动视频已成功创建: {output_path}")
        return output_path

    def _create_frame_generator(self,
                                 img: Image.Image,
                                 video_renderer: VideoRenderer,
                                 scroll_speed: int,
                                 scroll_frames_needed: int,
                                 bg_color: Tuple[int, int, int, int]
    ) -> Callable[[int], Optional[np.ndarray]]:
        """
        创建用于生成视频帧的函数 (闭包)
        
        Args:
            img: 包含渲染文本的PIL Image对象 (RGBA格式)
            video_renderer: VideoRenderer实例，用于获取参数
            scroll_speed: 每帧滚动的像素数 (在渲染分辨率下)
            scroll_frames_needed: 滚动结束帧数
            bg_color: 用户指定的背景色RGBA
            
        Returns:
            一个函数，接收帧索引，返回该帧的Numpy数组 (H, W, C) 或 None
        """
        img_width, img_height = img.size
        target_width = video_renderer.original_width # 目标视频宽度
        target_height = video_renderer.original_height # 目标视频高度
        scale_factor = video_renderer.scale_factor # 获取原始缩放因子
        transparent_bg = video_renderer.transparent # 是否需要透明背景
        
        # 将Pillow图像转换为Numpy数组以便快速切片 (预转换为浮点数用于潜在的混合操作)
        # 确保图像是RGBA格式
        if img.mode != 'RGBA':
            logger.warning(f"文本图像模式为 {img.mode}, 正在转换为 RGBA")
            img = img.convert('RGBA')
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # 提取Alpha通道
        alpha_channel = img_np[:, :, 3:4] # 保持维度 (H, W, 1)
        # 提取RGB通道
        rgb_channel = img_np[:, :, :3]

        # 缓存帧数据，避免重复创建
        frame_cache = {}

        # 确认目标缓冲区的数据类型
        # 如果背景不透明，可以直接使用uint8
        # 如果背景透明，可能需要保持float32进行混合，或根据FFmpeg要求调整
        target_dtype = np.uint8 if not transparent_bg else np.float32 

        logger.info(f"帧生成器设置: img_size=({img_width},{img_height}), target_size=({target_width},{target_height}), scroll_speed={scroll_speed}, transparent={transparent_bg}, scroll_end_frame={scroll_frames_needed}")
        # 使用用户指定的背景色创建背景帧模板
        bg_color_arr = np.array(bg_color, dtype=np.uint8)
        if transparent_bg:
            # RGBA背景模板
            background_frame = np.ones((target_height, target_width, 4), dtype=np.uint8) * bg_color_arr
            # 预转换为浮点用于混合，仅RGB通道
            background_float = background_frame[..., :3].astype(np.float32) / 255.0
        else:
            # RGB背景模板
            background_frame = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color_arr[:3]
            background_float = background_frame.astype(np.float32) / 255.0

        # 记录最大有效帧索引以防止循环 - 移除额外停留帧
        # 确保最大帧索引至少与滚动帧数一致
        if hasattr(video_renderer, 'total_frames') and video_renderer.total_frames > 0:
            max_valid_frame_index = min(scroll_frames_needed, video_renderer.total_frames - 1)
            logger.info(f"最大有效帧索引: {max_valid_frame_index}, 总帧数: {video_renderer.total_frames}, 滚动结束帧: {scroll_frames_needed}")
        else:
            max_valid_frame_index = scroll_frames_needed
            logger.info(f"未设置总帧数，使用计算的最大有效帧索引: {max_valid_frame_index}, 滚动结束帧: {scroll_frames_needed}")

        def frame_generator(frame_index: int) -> Optional[np.ndarray]:
            """生成指定索引的视频帧"""
            nonlocal frame_cache
            
            # --- 预检查帧索引是否超出预期范围 ---
            # 只有在设置了total_frames且为正值时才进行检查
            if hasattr(video_renderer, 'total_frames') and video_renderer.total_frames > 0:
                if frame_index < 0 or frame_index >= video_renderer.total_frames:
                    logger.warning(f"帧索引 {frame_index} 超出有效范围 [0, {video_renderer.total_frames-1}]，返回背景帧")
                    return background_frame.copy()
            
            # --- 重点修复：强制帧索引上限 ---
            # 定义严格滚动界限：文本完全滚出屏幕所需的帧数
            scroll_limit = scroll_frames_needed
            
            # 如果帧索引超过了滚动所需帧数，直接返回纯背景帧以结束滚动
            if frame_index >= scroll_limit:
                logger.debug(f"帧 {frame_index} 超过滚动限制 {scroll_limit}，返回纯背景帧")
                # 返回纯背景帧
                return background_frame.copy()
            
            # 如果启用了帧缓存且已缓存，直接返回 (滚动阶段)
            if video_renderer.use_frame_cache and frame_index in frame_cache:
                return frame_cache[frame_index]
            
            # --- 滚动阶段计算逻辑 --- 
            # 计算当前帧文本图像的起始y坐标
            y_start = frame_index * scroll_speed
            
            # 如果当前位置已经超出图像高度，直接返回背景帧
            if y_start >= img_height:
                logger.debug(f"帧 {frame_index}: 滚动位置 {y_start}px 已超出图像高度 {img_height}px，返回背景帧")
                return background_frame.copy()
            
            # 计算需要从文本图像中截取的区域的结束y坐标
            y_end = y_start + target_height
            
            # 边界检查 (确保截取范围在图像内)
            slice_y_start = max(0, y_start)
            slice_y_end = min(img_height, y_end)
            
            # 确保截取范围有效
            if slice_y_end <= slice_y_start:
                # 无效区域，返回背景帧
                logger.debug(f"帧 {frame_index}: 截取范围无效 ({slice_y_start}:{slice_y_end})，返回背景帧")
                return background_frame.copy()
                
            # 检查是否已经完全滚到图像底部
            remaining_height = img_height - slice_y_start
            if remaining_height <= 0:
                # 已完全滚出，返回背景帧
                logger.debug(f"帧 {frame_index}: 已完全滚出图像 (剩余高度: {remaining_height}px)，返回背景帧")
                return background_frame.copy()
            
            # 如果只剩一小部分图像，混合背景色处理
            if remaining_height < target_height:
                # 从图像中截取剩余部分
                source_rgb = rgb_channel[slice_y_start:slice_y_end, :, :]
                source_alpha = alpha_channel[slice_y_start:slice_y_end, :, :]
                
                # 创建背景画布
                if transparent_bg:
                    frame_canvas = background_frame.copy()
                    # 复制到画布顶部
                    frame_canvas[0:remaining_height, :, :3] = (source_rgb * 255).astype(np.uint8)
                    frame_canvas[0:remaining_height, :, 3:4] = (source_alpha * 255).astype(np.uint8)
                else:
                    # RGB画布，以背景模板为基础
                    frame_canvas = background_float.copy()
                    
                    # 获取目标区域并应用Alpha混合
                    target_section = frame_canvas[0:remaining_height, :, :]
                    blended_section = blend_alpha_fast(source_rgb, target_section, source_alpha)
                    frame_canvas[0:remaining_height, :, :] = blended_section
                    # 转换回 uint8
                    frame_canvas = (frame_canvas * 255).astype(np.uint8)
                
                # 缓存并返回结果
                if video_renderer.use_frame_cache:
                    frame_cache[frame_index] = frame_canvas
                return frame_canvas
            
            # 正常滚动情况 - 截取完整屏幕高度
            source_rgb = rgb_channel[slice_y_start:slice_y_end, :, :]
            source_alpha = alpha_channel[slice_y_start:slice_y_end, :, :]
            
            # 创建目标帧
            if transparent_bg:
                # RGBA画布
                frame_canvas = background_frame.copy()
                frame_canvas[0:slice_y_end-slice_y_start, :, :3] = (source_rgb * 255).astype(np.uint8)
                frame_canvas[0:slice_y_end-slice_y_start, :, 3:4] = (source_alpha * 255).astype(np.uint8)
            else:
                # RGB画布
                frame_canvas = background_float.copy()
                target_section = frame_canvas[0:slice_y_end-slice_y_start, :, :]
                blended_section = blend_alpha_fast(source_rgb, target_section, source_alpha)
                frame_canvas[0:slice_y_end-slice_y_start, :, :] = blended_section
                frame_canvas = (frame_canvas * 255).astype(np.uint8)
                
            # 缓存结果
            if video_renderer.use_frame_cache:
                frame_cache[frame_index] = frame_canvas
                
            return frame_canvas

        return frame_generator