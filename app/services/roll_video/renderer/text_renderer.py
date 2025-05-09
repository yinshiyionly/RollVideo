"""文本渲染器模块"""

import textwrap
import requests
import io
from typing import Tuple, List, Optional, Union
from PIL import Image, ImageDraw, ImageFont, __version__ as PIL_VERSION
import logging

logger = logging.getLogger(__name__)

class TextRenderer:
    """文字渲染器，负责将文本渲染成图片"""
    
    def __init__(
        self,
        width: int,
        font_path: str,
        font_size: int,
        font_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int, int] = (255, 255, 255, 255),  # RGBA颜色，支持透明度
        background_url: Optional[str] = None,  # 背景图URL
        scale_mode: str = "stretch",  # 背景图缩放模式: 'stretch'拉伸或'tile'平铺
        line_spacing: int = 10,
        char_spacing: int = 0,
        top_margin: int = 10,      # 默认上边距10px
        bottom_margin: int = 10,   # 默认下边距10px
        left_margin: int = 10,     # 默认左边距10px
        right_margin: int = 10     # 默认右边距10px
    ):
        """
        初始化文字渲染器
        
        Args:
            width: 图片宽度
            font_path: 字体文件路径
            font_size: 字体大小
            font_color: 字体颜色 (R,G,B)
            bg_color: 背景颜色 (R,G,B,A)，A为透明度，0表示完全透明，255表示完全不透明
            background_url: 背景图片URL，如果提供，将覆盖bg_color
            scale_mode: 背景图缩放模式, 'stretch'=拉伸, 'tile'=平铺
            line_spacing: 行间距
            char_spacing: 字符间距
            top_margin: 上边距，文本与顶部的距离
            bottom_margin: 下边距，文本与底部的距离
            left_margin: 左边距，文本与左侧的距离
            right_margin: 右边距，文本与右侧的距离
        """
        self.width = width
        self.font = ImageFont.truetype(font_path, font_size)
        self.font_size = font_size
        
        # 确保字体颜色是RGB格式
        if len(font_color) == 3:
            self.font_color = font_color + (255,) # 添加alpha通道用于绘制
        elif len(font_color) == 4:
            self.font_color = font_color
        else: # 如果无效，默认使用不透明黑色
             self.font_color = (0, 0, 0, 255)
            
        # 确保背景颜色是RGBA格式
        if len(bg_color) == 4:
            self.bg_color = bg_color
        elif len(bg_color) == 3:
            self.bg_color = bg_color + (255,)  # 添加Alpha通道，默认不透明
        else:
            self.bg_color = (255, 255, 255, 0) # 默认使用透明白色
            
        # 保存背景图URL和缩放模式
        self.background_url = background_url
        self.scale_mode = scale_mode.lower() if scale_mode else "stretch"
        if self.scale_mode not in ["stretch", "tile"]:
            logger.warning(f"无效的背景图缩放模式 '{scale_mode}'，使用默认值 'stretch'")
            self.scale_mode = "stretch"

        self.line_spacing = line_spacing
        self.char_spacing = char_spacing
        
        # 添加边距属性
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin
        self.left_margin = left_margin
        self.right_margin = right_margin
        
        # 计算文本实际可用宽度（考虑左右边距）
        self.text_width = self.width - self.left_margin - self.right_margin
    
    def _calculate_text_layout(self, text: str) -> List[str]:
        """
        计算文本布局，处理换行
        
        Args:
            text: 输入文本
            
        Returns:
            分行后的文本列表
        """
        # 考虑字符间距的影响估算每行字符数
        # 如果可用，使用getlength获取更精确的宽度估计
        try:
            avg_char_width = self.font.getlength("测试") / 2 + self.char_spacing
        except AttributeError:
            avg_char_width = self.font.getbbox("测")[2] + self.char_spacing # 回退方案

        # 使用文本的实际可用宽度(考虑左右边距)来计算每行字符数
        estimated_chars_per_line = int(self.text_width / avg_char_width) if avg_char_width > 0 else 1
        if estimated_chars_per_line <= 0 : estimated_chars_per_line = 1 # 避免零或负宽度

        lines = []
        # 正确处理 \n 换行，使用正则表达式分割
        paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                # 空段落保留为空行
                lines.append("")
                continue
                
            wrapped_lines = textwrap.wrap(
                paragraph, 
                width=estimated_chars_per_line,
                replace_whitespace=False,  # 保留原始空格
                drop_whitespace=True,      # 删除开头和结尾的空白
                break_long_words=True,     # 允许在超过宽度时断开长单词
                break_on_hyphens=False     # 避免在单词中的连字符处断行
            )
            
            # 处理textwrap可能为仅包含空白的段落返回空列表的情况
            if not wrapped_lines and paragraph.strip():
                 lines.append(paragraph) # 如果包装意外失败，保留原始段落
            else:
                 lines.extend(wrapped_lines if wrapped_lines else [""]) # 确保非空段落至少有一行
        
        return lines
        
    def _load_background_image(self, width: int, height: int) -> Optional[Image.Image]:
        """
        加载背景图片并根据指定的缩放模式调整大小
        
        Args:
            width: 目标图像宽度
            height: 目标图像高度
            
        Returns:
            处理后的背景图像，如果加载失败则返回None
        """
        if not self.background_url:
            return None
            
        try:
            # 从URL加载图像
            logger.info(f"从URL加载背景图片: {self.background_url}")
            response = requests.get(self.background_url, timeout=10)
            response.raise_for_status()  # 检查是否成功获取
            
            # 从响应内容加载图像
            img = Image.open(io.BytesIO(response.content))
            
            # 根据缩放模式处理图像
            if self.scale_mode == 'stretch':
                # 拉伸模式 - 将图像拉伸到指定尺寸
                logger.info(f"使用拉伸模式调整背景图片大小: {img.size} -> ({width}, {height})")
                return img.resize((width, height), Image.LANCZOS)
            
            elif self.scale_mode == 'tile':
                # 平铺模式 - 创建新图像并平铺原图
                logger.info(f"使用平铺模式创建背景: 源图像大小={img.size}, 目标大小=({width}, {height})")
                img_width, img_height = img.size
                tiled_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                
                # 计算需要多少个瓦片
                x_tiles = (width + img_width - 1) // img_width
                y_tiles = (height + img_height - 1) // img_height
                
                # 平铺图像
                for y in range(y_tiles):
                    for x in range(x_tiles):
                        tiled_img.paste(img, (x * img_width, y * img_height))
                
                return tiled_img
                
        except Exception as e:
            logger.error(f"加载背景图片失败: {str(e)}")
            return None
            
        return None
    
    def render_text_to_image(self, text: str, min_height: Optional[int] = None) -> Tuple[Image.Image, int]:
        """
        将文本渲染到图片，并在末尾添加一个屏幕高度的空白
        
        Args:
            text: 要渲染的文本内容
            min_height: 最小图片高度，通常设置为视频高度
            
        Returns:
            元组: (包含渲染文本的PIL Image对象, 实际文本内容的高度)
        """
        lines = self._calculate_text_layout(text)
        
        # 使用getbbox计算行高以获得更准确的结果
        try:
             # 获取包含上下延伸部分的字符边界框
             bbox = self.font.getbbox("Agy!")
             line_height = bbox[3] - bbox[1] + self.line_spacing
        except AttributeError:
             # 旧版Pillow的回退方法
             line_height = self.font_size + self.line_spacing

        # 计算文本实际高度（包括上下边距）
        text_content_height = len(lines) * line_height if lines else 0  # 纯文本内容高度
        text_actual_height = text_content_height + self.top_margin + self.bottom_margin  # 加上上下边距
        
        # 确定屏幕高度（用于计算底部空白）
        screen_height = min_height if min_height else text_actual_height
        if screen_height <= 0: screen_height = 1  # 避免高度为0
        
        # 在图像末尾添加一个屏幕高度的空白区域
        total_height = text_actual_height + screen_height
        
        # 尝试加载背景图片
        bg_image = None
        if self.background_url:
            bg_image = self._load_background_image(self.width, total_height)
            
        # 创建图像
        if bg_image:
            # 如果有背景图片，使用它
            img = bg_image
            logger.info("使用背景图片作为渲染背景")
        else:
            # 否则使用指定的背景颜色
            img = Image.new('RGBA', (self.width, total_height), self.bg_color)
            logger.info(f"使用纯色背景 {self.bg_color}")
            
        draw = ImageDraw.Draw(img)
        
        # 考虑上边距，文本从图像上边距开始绘制
        y_position = self.top_margin
        
        # 使用指定的字体颜色（包括透明度）绘制文本
        for line in lines:
            # 如果需要手动添加字符间距（Pillow >= 9.2.0支持在draw.text中设置）
            if hasattr(draw, 'text') and PIL_VERSION >= '9.2.0':
                 draw.text((self.left_margin, y_position), line, font=self.font, fill=self.font_color, spacing=self.char_spacing)
            else:
                 # 对于旧版Pillow，手动添加字符间距
                 x_pos = self.left_margin
                 for char in line:
                      draw.text((x_pos, y_position), char, font=self.font, fill=self.font_color)
                      try:
                           char_width = self.font.getlength(char)
                      except AttributeError:
                           char_width = self.font.getbbox(char)[2] if char != ' ' else self.font.getbbox('a')[2] # 估算空格宽度
                      x_pos += char_width + self.char_spacing

            y_position += line_height
        
        return img, text_actual_height  # 返回图像和文本实际高度（包括上下边距） 