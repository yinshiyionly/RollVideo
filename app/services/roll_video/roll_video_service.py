import os
import sys
import logging
import platform
import time
import traceback
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Union
from PIL import Image, ImageFont, ImageDraw

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
        line_spacing: int = 20,
        char_spacing: int = 0,
        fps: int = 30,
        scroll_speed: float = 1,  # 修改为每秒滚动的行数
        audio_path: Optional[str] = None,
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
            scroll_speed: 滚动速度(每秒滚动的行数)，例如0.5表示每2秒滚动一行
            audio_path: 可选的音频文件路径

        Returns:
            包含处理结果的字典
        """
        try:
            # --- 决定透明度需求和编码策略 ---
            normalized_bg_color = list(bg_color)
            if len(normalized_bg_color) == 3:
                normalized_bg_color.append(255)  # RGB 转 RGBA，默认不透明
            elif len(normalized_bg_color) == 4 and isinstance(
                normalized_bg_color[3], float
            ):
                # 将 float alpha (0.0-1.0) 转为 int (0-255)
                if 0 <= normalized_bg_color[3] <= 1:
                    normalized_bg_color[3] = int(normalized_bg_color[3] * 255)
                else:
                    normalized_bg_color[3] = int(
                        normalized_bg_color[3]
                    )  # 超出范围直接取整
            # 确保 alpha 值在 0-255 范围内
            normalized_bg_color[3] = max(0, min(255, normalized_bg_color[3]))
            bg_color_final = tuple(normalized_bg_color)

            transparency_required = bg_color_final[3] < 255

            # 根据透明度需求确定输出格式和编码器
            output_dir = os.path.dirname(os.path.abspath(output_path))
            base_name = os.path.splitext(os.path.basename(output_path))[0]

            if transparency_required:
                preferred_codec = "prores_ks"  # 高质量透明编码（CPU）
                actual_output_path = os.path.join(output_dir, f"{base_name}.mov")
                logger.info(
                    "检测到透明背景需求，将使用 CPU (prores_ks) 输出 .mov 文件。"
                )
            else:
                preferred_codec = "h264_nvenc"  # 优先尝试 GPU H.264 编码
                actual_output_path = os.path.join(output_dir, f"{base_name}.mp4")
                logger.info(
                    "无透明背景需求，将优先尝试 GPU (h264_nvenc) 输出 .mp4 文件。"
                )
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
                font_color=(
                    tuple(font_color) if isinstance(font_color, list) else font_color
                ),
                bg_color=bg_color_final,
                line_spacing=line_spacing,
                char_spacing=char_spacing,
            )

            # 将文本渲染为图片，并获取文本实际高度
            logger.info("将文本渲染为图片...")
            text_image, text_actual_height = text_renderer.render_text_to_image(
                text, min_height=height
            )
            logger.info(
                f"文本实际高度: {text_actual_height}px, 渲染图像总高度: {text_image.height}px"
            )

            # 估算行高 (字体大小 + 行间距)
            estimated_line_height = font_size + line_spacing

            # 估算总行数（文本高度除以行高）
            estimated_total_lines = max(1, text_actual_height / estimated_line_height)

            # 设置最小滚动时长（单位：秒）
            min_scroll_duration = 10.0  # 至少10秒的滚动时间

            # 计算当前设置下的滚动时长（不包括头尾静止时间）
            current_scroll_duration = (
                estimated_total_lines / scroll_speed if scroll_speed > 0 else 0
            )

            # 如果滚动时间太短，则自动调整滚动速度
            adjusted_scroll_speed = scroll_speed
            if (
                current_scroll_duration < min_scroll_duration
                and current_scroll_duration > 0
            ):
                adjusted_scroll_speed = estimated_total_lines / min_scroll_duration
                logger.info(
                    f"文本较短 ({estimated_total_lines:.1f}行)，自动调整滚动速度: {scroll_speed:.2f}行/秒 → {adjusted_scroll_speed:.2f}行/秒"
                )
                logger.info(
                    f"预计滚动时长: {current_scroll_duration:.1f}秒 → {min_scroll_duration:.1f}秒 (不含头尾静止)"
                )
            else:
                logger.info(
                    f"预计滚动时长: {current_scroll_duration:.1f}秒 (不含头尾静止)"
                )

            # 将每秒滚动的行数转换为每帧滚动的像素数
            # adjusted_scroll_speed是每秒滚动的行数，乘以行高得到每秒滚动的像素数，再除以fps得到每帧滚动的像素数
            pixels_per_frame = (adjusted_scroll_speed * estimated_line_height) / fps

            # 取整，确保至少滚动1像素/帧
            pixels_per_frame = max(1, round(pixels_per_frame))

            logger.info(
                f"滚动速度设置: {adjusted_scroll_speed:.2f}行/秒 → {pixels_per_frame}像素/帧 (行高约{estimated_line_height}像素)"
            )

            # 创建视频渲染器
            video_renderer = VideoRenderer(
                width=width, height=height, fps=fps, scroll_speed=pixels_per_frame
            )

            # 创建滚动视频，传递决策结果
            logger.info("开始创建滚动视频...")
            final_output_path = video_renderer.create_scrolling_video_optimized(
                image=text_image,
                output_path=actual_output_path,  # 使用自动调整后的路径
                text_actual_height=text_actual_height,
                transparency_required=transparency_required,  # 传递透明度需求
                preferred_codec=preferred_codec,  # 传递首选编码器
                audio_path=audio_path,
                bg_color=bg_color_final,  # 传递最终的bg_color供非透明路径使用
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

    def _generate_scrolling_text_image(self, text, font_path=None, font_size=36, fg_color=(255, 255, 255), bg_color=(0, 0, 0, 0), max_width=1280, align="left", spacing=8, padding=(40, 40, 40, 40), antialias=True):
        """生成滚动文本的图像
        
        Args:
            text: 要渲染的文本
            font_path: 字体路径，如果为None则使用默认字体
            font_size: 字体大小
            fg_color: 前景色 (R, G, B) 或 (R, G, B, A)
            bg_color: 背景色 (R, G, B) 或 (R, G, B, A)
            max_width: 最大宽度
            align: 对齐方式 "left", "center", 或 "right"
            spacing: 行间距
            padding: (左, 上, 右, 下) 填充像素
            antialias: 是否使用抗锯齿
            
        Returns:
            PIL.Image: 渲染后的文本图像
        """
        # 初始化性能统计
        text_render_stats = {
            "text_length": len(text),
            "render_start_time": time.time(),
            "render_time": 0
        }
        
        logger.info(f"开始渲染文本 (长度: {text_render_stats['text_length']}字符)")
        
        try:
            # 准备字体
            if font_path:
                font = self._load_font(font_path, font_size)
            else:
                logger.warning("未指定字体，使用默认字体")
                # 使用PIL默认字体
                font = ImageFont.load_default()
            
            # 初始空白图像用于测量文本
            img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # 拆分文本行
            lines = text.split('\n')
            
            # 预处理各行文本宽度
            line_sizes = []
            for line in lines:
                # 测量每行文本的大小
                if hasattr(draw, 'textbbox'):  # PIL 9.2.0+
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_width = bbox[2] - bbox[0]
                    line_height = bbox[3] - bbox[1]
                else:  # 旧版PIL
                    line_width, line_height = draw.textsize(line, font=font)
                
                line_sizes.append((line_width, line_height))
            
            # 计算最大宽度 (考虑最大宽度限制)
            if max_width:
                # 考虑左右填充
                max_text_width = max_width - padding[0] - padding[2]
                content_width = min(max(s[0] for s in line_sizes) if line_sizes else 0, max_text_width)
            else:
                content_width = max(s[0] for s in line_sizes) if line_sizes else 0
            
            # 获取单行最大高度
            line_max_height = max(s[1] for s in line_sizes) if line_sizes else font_size
            
            # 计算总高度 (包括行间距)
            total_height = sum(line_max_height for _ in lines)
            if len(lines) > 1:
                total_height += spacing * (len(lines) - 1)
            
            # 最终图像尺寸 (包括填充)
            final_width = content_width + padding[0] + padding[2]
            final_height = total_height + padding[1] + padding[3]
            
            # 创建最终图像
            img = Image.new('RGBA', (final_width, final_height), bg_color)
            draw = ImageDraw.Draw(img)
            
            # 当前垂直位置
            y = padding[1]
            
            # 绘制每行文本
            for i, line in enumerate(lines):
                # 计算此行文本的水平位置
                if align == 'left':
                    x = padding[0]
                elif align == 'center':
                    x = padding[0] + (content_width - line_sizes[i][0]) // 2
                elif align == 'right':
                    x = padding[0] + content_width - line_sizes[i][0]
                else:
                    x = padding[0]  # 默认左对齐
                
                # 绘制文本
                if antialias:
                    # 使用内置抗锯齿
                    draw.text((x, y), line, font=font, fill=fg_color)
                else:
                    # 禁用抗锯齿
                    draw.text((x, y), line, font=font, fill=fg_color)
                
                # 移动到下一行
                y += line_sizes[i][1] + spacing
            
            # 记录实际文本高度 (不含上下填充)
            actual_text_height = total_height
            
            # 记录渲染时间
            text_render_stats["render_time"] = time.time() - text_render_stats["render_start_time"]
            logger.info(f"文本渲染完成: {text_render_stats['render_time']:.2f}秒，尺寸: {final_width}x{final_height}像素")
            
            return img, actual_text_height
        
        except Exception as e:
            end_time = time.time()
            text_render_stats["render_time"] = end_time - text_render_stats["render_start_time"]
            logger.error(f"文本渲染失败，耗时: {text_render_stats['render_time']:.2f}秒，错误: {str(e)}")
            raise

    def _draw_background_rectangle(self, img, x, y, w, h, fill, outline=None, radius=0):
        """绘制背景矩形，支持圆角"""
        # 创建绘图对象
        draw = ImageDraw.Draw(img)
        
        if radius <= 0:
            # 无圆角，直接绘制矩形
            draw.rectangle([x, y, x + w, y + h], fill=fill, outline=outline)
            return
        
        # 绘制圆角矩形
        # 四个角的圆弧
        draw.rectangle([x + radius, y, x + w - radius, y + h], fill=fill, outline=None)
        draw.rectangle([x, y + radius, x + w, y + h - radius], fill=fill, outline=None)
        
        # 左上角
        draw.pieslice([x, y, x + radius * 2, y + radius * 2], 180, 270, fill=fill, outline=outline)
        # 右上角
        draw.pieslice([x + w - radius * 2, y, x + w, y + radius * 2], 270, 0, fill=fill, outline=outline)
        # 右下角
        draw.pieslice([x + w - radius * 2, y + h - radius * 2, x + w, y + h], 0, 90, fill=fill, outline=outline)
        # 左下角
        draw.pieslice([x, y + h - radius * 2, x + radius * 2, y + h], 90, 180, fill=fill, outline=outline)
        
        # 如果有轮廓
        if outline:
            # 绘制边框
            radius = max(radius - 1, 0)  # 边框略小于填充
            
            # 上边框
            draw.line([x + radius, y, x + w - radius, y], fill=outline)
            # 右边框
            draw.line([x + w, y + radius, x + w, y + h - radius], fill=outline)  
            # 下边框
            draw.line([x + w - radius, y + h, x + radius, y + h], fill=outline)
            # 左边框
            draw.line([x, y + h - radius, x, y + radius], fill=outline)
            
            # 左上角弧线
            draw.arc([x, y, x + radius * 2, y + radius * 2], 180, 270, fill=outline)
            # 右上角弧线
            draw.arc([x + w - radius * 2, y, x + w, y + radius * 2], 270, 0, fill=outline)
            # 右下角弧线
            draw.arc([x + w - radius * 2, y + h - radius * 2, x + w, y + h], 0, 90, fill=outline)
            # 左下角弧线
            draw.arc([x, y + h - radius * 2, x + radius * 2, y + h], 90, 180, fill=outline)

    def _load_font(self, font_path, font_size):
        """加载字体
        
        Args:
            font_path: 字体文件路径
            font_size: 字体大小
            
        Returns:
            PIL.ImageFont: 字体对象
        """
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception as e:
            logger.error(f"无法加载字体 {font_path}: {str(e)}")
            # 回退到默认字体
            return ImageFont.load_default()

    def create_scrolling_video(self, text, output_path=None, font_path=None, font_size=36, 
                               fg_color=(255, 255, 255), bg_color=(0, 0, 0, 0), 
                               align="left", line_spacing=8, padding=(40, 40, 40, 40),
                               preferred_codec="libx264", audio_path=None):
        """创建滚动字幕视频
        
        Args:
            text: 要显示的文本内容
            output_path: 输出文件路径 (如果为None则自动生成)
            font_path: 字体文件路径 (如果为None则使用系统默认字体)
            font_size: 字体大小
            fg_color: 文本颜色 (R,G,B) 或 (R,G,B,A)
            bg_color: 背景颜色 (R,G,B) 或 (R,G,B,A)
            align: 对齐方式 ('left', 'center', 'right')
            line_spacing: 行间距 (像素)
            padding: 文本填充 (左,上,右,下) 像素
            preferred_codec: 首选视频编码器
            audio_path: 音频文件路径 (可选)
            
        Returns:
            输出视频文件路径
        """
        # 初始化性能统计
        self.performance_stats = {
            "total_start_time": time.time(),  # 总开始时间
            "text_render_time": 0,            # 文本渲染时间
            "video_render_time": 0,           # 视频渲染时间
            "total_time": 0                   # 总时间
        }
        
        logger.info(f"开始创建滚动字幕视频...")
        
        try:
            # 1. 生成滚动文本图像
            text_render_start = time.time()
            logger.info(f"正在渲染文本图像，长度: {len(text)} 字符")
            
            # 检查原始参数中是否有透明度设置
            has_alpha = len(bg_color) == 4 and bg_color[3] < 255
            transparency_required = has_alpha
            
            # 渲染文本到图像
            text_img, text_actual_height = self._generate_scrolling_text_image(
                text=text,
                font_path=font_path,
                font_size=font_size,
                fg_color=fg_color,
                bg_color=bg_color,
                max_width=1280,  # 标准HD宽度
                align=align,
                spacing=line_spacing,
                padding=padding
            )
            
            # 记录文本渲染时间
            text_render_end = time.time()
            self.performance_stats["text_render_time"] = text_render_end - text_render_start
            logger.info(f"文本渲染完成，耗时: {self.performance_stats['text_render_time']:.2f}秒")
            
            # 2. 创建视频渲染器
            width = text_img.width
            height = min(720, text_actual_height)  # 限制最大高度为720p
            scroll_speed = 2  # 每帧滚动像素数
            
            video_render_start = time.time()
            logger.info(f"创建视频渲染器: {width}x{height}, 滚动速度: {scroll_speed}像素/帧")
            
            renderer = VideoRenderer(
                width=width,
                height=height,
                fps=30,
                scroll_speed=scroll_speed
            )
            
            # 3. 生成输出路径 (如果未提供)
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_folder = os.path.join(
                    os.path.dirname(__file__), "output"
                )
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(
                    output_folder, f"scrolling_text_{timestamp}.mp4"
                )
            
            # 4. 渲染视频
            logger.info(f"开始渲染视频: {output_path}")
            rendered_path = renderer.create_scrolling_video_optimized(
                image=text_img,
                output_path=output_path,
                text_actual_height=text_actual_height,
                transparency_required=transparency_required,
                preferred_codec=preferred_codec,
                audio_path=audio_path,
                bg_color=bg_color[:3] if len(bg_color) > 3 else bg_color  # 取RGB部分
            )
            
            # 记录视频渲染时间
            video_render_end = time.time()
            self.performance_stats["video_render_time"] = video_render_end - video_render_start
            
            # 计算总时间
            self.performance_stats["total_time"] = video_render_end - self.performance_stats["total_start_time"]
            
            # 合并渲染器的性能统计
            if hasattr(renderer, 'performance_stats'):
                self.performance_stats.update({
                    "video_preparation_time": renderer.performance_stats.get("preparation_time", 0),
                    "video_frame_processing_time": renderer.performance_stats.get("frame_processing_time", 0),
                    "video_encoding_time": renderer.performance_stats.get("encoding_time", 0),
                    "frames_processed": renderer.performance_stats.get("frames_processed", 0),
                    "fps": renderer.performance_stats.get("fps", 0),
                })
            
            # 输出综合性能报告
            logger.info("\n" + "="*50)
            logger.info("滚动视频生成性能统计:")
            logger.info(f"1. 文本渲染阶段: {self.performance_stats['text_render_time']:.2f}秒 ({self.performance_stats['text_render_time']/self.performance_stats['total_time']*100:.1f}%)")
            logger.info(f"2. 视频生成阶段: {self.performance_stats['video_render_time']:.2f}秒 ({self.performance_stats['video_render_time']/self.performance_stats['total_time']*100:.1f}%)")
            
            if "video_preparation_time" in self.performance_stats:
                vp = self.performance_stats["video_preparation_time"]
                vf = self.performance_stats["video_frame_processing_time"]
                ve = self.performance_stats["video_encoding_time"]
                
                logger.info(f"   - 视频准备: {vp:.2f}秒 ({vp/self.performance_stats['video_render_time']*100:.1f}%)")
                logger.info(f"   - 帧处理: {vf:.2f}秒 ({vf/self.performance_stats['video_render_time']*100:.1f}%) - {self.performance_stats['fps']:.1f}帧/秒")
                logger.info(f"   - 视频编码: {ve:.2f}秒 ({ve/self.performance_stats['video_render_time']*100:.1f}%)")
            
            logger.info(f"总时间: {self.performance_stats['total_time']:.2f}秒")
            if "frames_processed" in self.performance_stats:
                logger.info(f"处理帧数: {self.performance_stats['frames_processed']}帧，平均: {self.performance_stats['fps']:.1f}帧/秒")
            logger.info("="*50 + "\n")
            
            return rendered_path
            
        except Exception as e:
            # 记录总时间（即使发生错误）
            self.performance_stats["total_time"] = time.time() - self.performance_stats["total_start_time"]
            logger.error(f"创建滚动视频失败 (总耗时: {self.performance_stats['total_time']:.2f}秒): {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_roll_video_ffmpeg(
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
        line_spacing: int = 20,
        char_spacing: int = 0,
        fps: int = 30,
        scroll_speed: float = 1,  # 修改为每秒滚动的行数
        audio_path: Optional[str] = None,
    ) -> Dict[str, Union[str, bool]]:
        """
        使用FFmpeg滤镜创建滚动视频，自动根据透明度选择格式
        
        优势：
        1. 更高效 - 不需要逐帧渲染，直接使用FFmpeg滤镜实现滚动效果
        2. 更平滑 - 滚动效果由FFmpeg实时计算，支持亚像素精度的滚动
        3. 更低内存 - 不需要在内存中处理大量帧
        
        参数与create_roll_video相同
        """
        try:
            # --- 决定透明度需求和编码策略 ---
            normalized_bg_color = list(bg_color)
            if len(normalized_bg_color) == 3:
                normalized_bg_color.append(255)  # RGB 转 RGBA，默认不透明
            elif len(normalized_bg_color) == 4 and isinstance(
                normalized_bg_color[3], float
            ):
                # 将 float alpha (0.0-1.0) 转为 int (0-255)
                if 0 <= normalized_bg_color[3] <= 1:
                    normalized_bg_color[3] = int(normalized_bg_color[3] * 255)
                else:
                    normalized_bg_color[3] = int(
                        normalized_bg_color[3]
                    )  # 超出范围直接取整
            # 确保 alpha 值在 0-255 范围内
            normalized_bg_color[3] = max(0, min(255, normalized_bg_color[3]))
            bg_color_final = tuple(normalized_bg_color)

            transparency_required = bg_color_final[3] < 255

            # 根据透明度需求确定输出格式和编码器
            output_dir = os.path.dirname(os.path.abspath(output_path))
            base_name = os.path.splitext(os.path.basename(output_path))[0]

            if transparency_required:
                preferred_codec = "prores_ks"  # 高质量透明编码（CPU）
                actual_output_path = os.path.join(output_dir, f"{base_name}.mov")
                logger.info(
                    "检测到透明背景需求，将使用 CPU (prores_ks) 输出 .mov 文件。"
                )
            else:
                preferred_codec = "h264_nvenc"  # 优先尝试 GPU H.264 编码
                actual_output_path = os.path.join(output_dir, f"{base_name}.mp4")
                logger.info(
                    "无透明背景需求，将优先尝试 GPU (h264_nvenc) 输出 .mp4 文件。"
                )
            # --- 结束决策 ---

            logger.info(f"开始创建滚动视频 (FFmpeg滤镜方式)，实际输出路径: {actual_output_path}")

            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 获取有效的字体路径
            font_path = self.get_font_path(font_path)

            # 创建文字渲染器 (使用最终确定的bg_color)
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
            )

            # 将文本渲染为图片，并获取文本实际高度
            logger.info("将文本渲染为图片...")
            text_image, text_actual_height = text_renderer.render_text_to_image(
                text, min_height=height
            )
            logger.info(
                f"文本实际高度: {text_actual_height}px, 渲染图像总高度: {text_image.height}px"
            )

            # 估算行高 (字体大小 + 行间距)
            estimated_line_height = font_size + line_spacing

            # 将每秒滚动的行数转换为每帧滚动的像素数
            # scroll_speed是每秒滚动的行数，乘以行高得到每秒滚动的像素数，再除以fps得到每帧滚动的像素数
            pixels_per_frame = (scroll_speed * estimated_line_height) / fps

            # 确保至少滚动1像素/帧
            pixels_per_frame = max(1, round(pixels_per_frame))

            logger.info(
                f"滚动速度设置: {scroll_speed:.2f}行/秒 → {pixels_per_frame}像素/帧 (行高约{estimated_line_height}像素)"
            )

            # 创建视频渲染器
            video_renderer = VideoRenderer(
                width=width, height=height, fps=fps, scroll_speed=pixels_per_frame
            )

            # 使用FFmpeg滤镜方式创建滚动视频 - 直接传递PIL图像，不转换为numpy数组
            logger.info("开始创建滚动视频 (FFmpeg滤镜方式)...")
            final_output_path = video_renderer.create_scrolling_video_ffmpeg(
                image=text_image,  # 直接传递PIL图像对象
                output_path=actual_output_path,  # 使用自动调整后的路径
                text_actual_height=text_actual_height,
                transparency_required=transparency_required,  # 传递透明度需求
                preferred_codec=preferred_codec,  # 传递首选编码器
                audio_path=audio_path,
                bg_color=bg_color_final,  # 传递最终的bg_color供非透明路径使用
            )

            logger.info(f"滚动视频创建完成 (FFmpeg滤镜方式): {final_output_path}")

            return {
                "status": "success",
                "message": "滚动视频创建成功 (FFmpeg滤镜方式)",
                "output_path": final_output_path,
            }

        except Exception as e:
            logger.error(f"创建滚动视频失败 (FFmpeg滤镜方式): {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"创建滚动视频失败 (FFmpeg滤镜方式): {str(e)}",
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
        line_spacing: int = 20,
        char_spacing: int = 0,
        fps: int = 30,
        scroll_speed: float = 1,  # 修改为每秒滚动的行数
        audio_path: Optional[str] = None,
        scroll_effect: str = "basic",  # 滚动效果类型: 'basic'=匀速, 'advanced'=加减速
    ) -> Dict[str, Union[str, bool]]:
        """
        使用FFmpeg的overlay_cuda GPU加速滤镜创建滚动视频，自动根据透明度选择格式
        
        优势：
        1. GPU加速 - 使用NVIDIA GPU硬件加速overlay操作和编码
        2. 高效平滑 - 不需要逐帧渲染，直接在GPU上实现滚动效果
        3. 支持特效 - 可以实现加速减速等特效
        
        参数:
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
            scroll_speed: 滚动速度(每秒滚动的行数)，例如0.5表示每2秒滚动一行
            audio_path: 可选的音频文件路径
            scroll_effect: 滚动效果类型，'basic'为匀速滚动，'advanced'为加速减速效果
            
        Returns:
            包含处理结果的字典
        """
        try:
            # --- 决定透明度需求和编码策略 ---
            normalized_bg_color = list(bg_color)
            if len(normalized_bg_color) == 3:
                normalized_bg_color.append(255)  # RGB 转 RGBA，默认不透明
            elif len(normalized_bg_color) == 4 and isinstance(
                normalized_bg_color[3], float
            ):
                # 将 float alpha (0.0-1.0) 转为 int (0-255)
                if 0 <= normalized_bg_color[3] <= 1:
                    normalized_bg_color[3] = int(normalized_bg_color[3] * 255)
                else:
                    normalized_bg_color[3] = int(
                        normalized_bg_color[3]
                    )  # 超出范围直接取整
            # 确保 alpha 值在 0-255 范围内
            normalized_bg_color[3] = max(0, min(255, normalized_bg_color[3]))
            bg_color_final = tuple(normalized_bg_color)

            transparency_required = bg_color_final[3] < 255

            # 根据透明度需求确定输出格式和编码器
            output_dir = os.path.dirname(os.path.abspath(output_path))
            base_name = os.path.splitext(os.path.basename(output_path))[0]

            if transparency_required:
                preferred_codec = "prores_ks"  # 高质量透明编码（CPU）
                actual_output_path = os.path.join(output_dir, f"{base_name}.mov")
                logger.info(
                    "检测到透明背景需求，将使用 CPU (prores_ks) 输出 .mov 文件。"
                )
            else:
                preferred_codec = "h264_nvenc"  # 优先尝试 GPU H.264 编码
                actual_output_path = os.path.join(output_dir, f"{base_name}.mp4")
                logger.info(
                    "无透明背景需求，将优先尝试 GPU (h264_nvenc) 输出 .mp4 文件。"
                )
            # --- 结束决策 ---

            logger.info(f"开始创建滚动视频 (overlay_cuda GPU加速方式)，实际输出路径: {actual_output_path}")
            logger.info(f"滚动效果: {scroll_effect}")

            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 获取有效的字体路径
            font_path = self.get_font_path(font_path)

            # 创建文字渲染器 (使用最终确定的bg_color)
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
            )

            # 将文本渲染为图片，并获取文本实际高度
            logger.info("将文本渲染为图片...")
            text_image, text_actual_height = text_renderer.render_text_to_image(
                text, min_height=height
            )
            logger.info(
                f"文本实际高度: {text_actual_height}px, 渲染图像总高度: {text_image.height}px"
            )

            # 估算行高 (字体大小 + 行间距)
            estimated_line_height = font_size + line_spacing

            # 将每秒滚动的行数转换为每帧滚动的像素数
            # scroll_speed是每秒滚动的行数，乘以行高得到每秒滚动的像素数，再除以fps得到每帧滚动的像素数
            pixels_per_frame = (scroll_speed * estimated_line_height) / fps

            # 确保至少滚动1像素/帧
            pixels_per_frame = max(1, round(pixels_per_frame))

            logger.info(
                f"滚动速度设置: {scroll_speed:.2f}行/秒 → {pixels_per_frame}像素/帧 (行高约{estimated_line_height}像素)"
            )

            # 创建视频渲染器
            video_renderer = VideoRenderer(
                width=width, height=height, fps=fps, scroll_speed=pixels_per_frame
            )

            # 使用overlay_cuda GPU加速滤镜方式创建滚动视频
            logger.info("开始创建滚动视频 (overlay_cuda GPU加速方式)...")
            final_output_path = video_renderer.create_scrolling_video_overlay_cuda(
                image=text_image,
                output_path=actual_output_path,
                text_actual_height=text_actual_height,
                transparency_required=transparency_required,
                preferred_codec=preferred_codec,
                audio_path=audio_path,
                bg_color=bg_color_final,
                scroll_effect=scroll_effect
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
