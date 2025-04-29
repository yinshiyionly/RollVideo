import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PIL_VERSION
from moviepy.editor import VideoClip
from typing import Dict, Tuple, List, Optional, Union
import textwrap
import platform
import logging
import subprocess
import tempfile
import shutil
import tqdm

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
        line_spacing: int = 10,
        char_spacing: int = 0
    ):
        """
        初始化文字渲染器
        
        Args:
            width: 图片宽度
            font_path: 字体文件路径
            font_size: 字体大小
            font_color: 字体颜色 (R,G,B)
            bg_color: 背景颜色 (R,G,B,A)，A为透明度，0表示完全透明，255表示完全不透明
            line_spacing: 行间距
            char_spacing: 字符间距
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

        self.line_spacing = line_spacing
        self.char_spacing = char_spacing
    
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

        estimated_chars_per_line = int(self.width / avg_char_width) if avg_char_width > 0 else 1
        if estimated_chars_per_line <= 0 : estimated_chars_per_line = 1 # 避免零或负宽度

        lines = []
        paragraphs = text.split('\\n')
        for paragraph in paragraphs:
            if not paragraph.strip():
                lines.append("")
                continue
            wrapped_lines = textwrap.wrap(
                paragraph, 
                width=estimated_chars_per_line,
                replace_whitespace=False,
                drop_whitespace=True,
                break_long_words=True, # 允许在超过宽度时断开长单词
                break_on_hyphens=False # 避免在单词中的连字符处断行
            )
            # 处理textwrap可能为仅包含空白的段落返回空列表的情况
            if not wrapped_lines and paragraph.strip():
                 lines.append(paragraph) # 如果包装意外失败，保留原始段落
            else:
                lines.extend(wrapped_lines if wrapped_lines else [""]) # 确保非空段落至少有一行
        
        return lines
    
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

        # 计算文本实际高度
        text_actual_height = len(lines) * line_height if lines else 0 # 文本内容的实际高度
        
        # 确定屏幕高度（用于计算底部空白）
        screen_height = min_height if min_height else text_actual_height
        if screen_height <= 0: screen_height = 1 # 避免高度为0
        
        # 在图像末尾添加一个屏幕高度的空白区域
        total_height = text_actual_height + screen_height
        
        # 使用指定的背景颜色（包括透明度）创建图片
        img = Image.new('RGBA', (self.width, total_height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # 文本从图像顶部开始绘制
        y_position = 0
        
        # 使用指定的字体颜色（包括透明度）绘制文本
        for line in lines:
            # 如果需要手动添加字符间距（Pillow >= 9.2.0支持在draw.text中设置）
            if hasattr(draw, 'text') and PIL_VERSION >= '9.2.0':
                 draw.text((0, y_position), line, font=self.font, fill=self.font_color, spacing=self.char_spacing)
            else:
                 # 对于旧版Pillow，手动添加字符间距
                 x_pos = 0
                 for char in line:
                      draw.text((x_pos, y_position), char, font=self.font, fill=self.font_color)
                      try:
                           char_width = self.font.getlength(char)
                      except AttributeError:
                           char_width = self.font.getbbox(char)[2] if char != ' ' else self.font.getbbox('a')[2] # 估算空格宽度
                      x_pos += char_width + self.char_spacing

            y_position += line_height
        
        return img, text_actual_height # 返回图像和文本实际高度

class VideoRenderer:
    """视频渲染器，负责创建滚动效果的视频"""
    
    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 30,
        scroll_speed: int = 2  # 每帧滚动的像素数
    ):
        """
        初始化视频渲染器
        
        Args:
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率
            scroll_speed: 滚动速度(像素/帧)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.scroll_speed = scroll_speed
    
    def create_scrolling_video(
        self,
        image: Image.Image,
        output_path: str,             # 路径已由Service确定（.mov 或 .mp4）
        text_actual_height: int,
        transparency_required: bool, # 新增：是否需要透明
        preferred_codec: str,       # 新增：首选编码器 (prores_ks 或 h264_nvenc)
        audio_path: Optional[str] = None,
        bg_color: Optional[Tuple[int, int, int, int]] = None # RGBA背景色
    ) -> str:
        """
        创建滚动效果的视频，根据透明度需求自动选择策略。
        
        Args:
            image: 要滚动的图片 (RGBA)
            output_path: 最终输出视频的路径 (.mov 或 .mp4)
            text_actual_height: 实际文本内容的高度（不包括底部空白）
            transparency_required: 是否需要透明背景
            preferred_codec: Service决定的首选编码器 (prores_ks 或 h264_nvenc)
            audio_path: 可选的音频文件路径
            bg_color: 最终的RGBA背景颜色
            
        Returns:
            输出视频的路径
        """
        # 将PIL Image转换为numpy数组
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2] # img_height 包含底部空白

        # 检查图片是否有Alpha通道
        if img_array.shape[2] != 4:
             logger.warning("输入图像意外不是 RGBA 格式，可能导致问题")
             image = image.convert("RGBA")
             img_array = np.array(image)
             if img_array.shape[2] != 4:
                  raise ValueError("无法将输入图像转换为 RGBA 格式")

        # --- 计算滚动参数 (与之前相同) --- 
        scroll_distance = text_actual_height - self.height
        scroll_distance = max(0, scroll_distance) 
        scroll_frames = int(scroll_distance / self.scroll_speed) if self.scroll_speed > 0 else 0
        padding_frames_start = int(self.fps * 0.5) 
        padding_frames_end = int(self.fps * 0.5)
        total_frames = padding_frames_start + scroll_frames + padding_frames_end
        duration = total_frames / self.fps
        # --- 结束滚动参数计算 --- 

        logger.info(f"文本高: {text_actual_height}px, 图像高: {img_height}px, 视频高: {self.height}px")
        logger.info(f"滚动距离: {scroll_distance}px, 滚动帧: {scroll_frames}, 总帧: {total_frames}, 时长: {duration:.2f}s")
        logger.info(f"输出路径: {output_path}, 透明需求: {transparency_required}")
        logger.info(f"首选编码器: {preferred_codec}")

        # ------ 根据透明度需求选择不同处理路径 ------
        if transparency_required:
            # --- 透明路径: CPU + prores_ks + .mov --- 
            logger.info("处理透明视频：使用手动帧生成和 ffmpeg (prores_ks - CPU)...")
            
            # 确保输出是 .mov
            if not output_path.lower().endswith(".mov"):
                 logger.warning(f"透明输出需要.mov格式，但指定路径为{output_path}。将强制使用.mov扩展名。")
                 output_path = os.path.splitext(output_path)[0] + ".mov"
                 logger.info(f"实际输出路径调整为: {output_path}")

            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"创建临时目录: {temp_dir}")
                png_pattern = os.path.join(temp_dir, "frame_%06d.png")

                logger.info("开始生成RGBA帧并保存为PNG序列...")
                for frame_idx in tqdm.tqdm(range(total_frames), desc="生成帧"):
                    # --- 滚动逻辑 (与之前相同) ---
                    if frame_idx < padding_frames_start:
                        current_position = 0
                    elif frame_idx < padding_frames_start + scroll_frames:
                        scroll_progress_frames = frame_idx - padding_frames_start
                        current_position = scroll_progress_frames * self.scroll_speed
                        current_position = min(current_position, scroll_distance)
                    else:
                        current_position = scroll_distance
                    # --- 结束滚动逻辑 --- 

                    # --- 帧生成 (与之前相同) ---
                    frame_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
                    img_start_y = int(current_position)
                    img_end_y = min(img_height, img_start_y + self.height)
                    frame_start_y = 0
                    frame_end_y = img_end_y - img_start_y
                    if img_start_y < img_end_y and frame_start_y < frame_end_y and frame_end_y <= self.height:
                         img_h_slice = slice(img_start_y, img_end_y)
                         img_w_slice = slice(0, min(self.width, img_width))
                         frame_h_slice = slice(frame_start_y, frame_end_y)
                         frame_w_slice = slice(0, min(self.width, img_width))
                         source_section = img_array[img_h_slice, img_w_slice]
                         target_area = frame_rgba[frame_h_slice, frame_w_slice]
                         if target_area.shape == source_section.shape:
                             np.copyto(target_area, source_section)
                         else:
                             copy_width = min(target_area.shape[1], source_section.shape[1])
                             copy_height = min(target_area.shape[0], source_section.shape[0])
                             if copy_height > 0 and copy_width > 0:
                                 target_area[:copy_height, :copy_width] = source_section[:copy_height, :copy_width]
                             else:
                                 logger.warning(f"帧 {frame_idx}: RGBA形状不匹配且无法复制。目标: {target_area.shape}, 源: {source_section.shape}")
                    elif frame_end_y > self.height:
                         logger.warning(f"帧 {frame_idx}: 计算出的 frame_end_y ({frame_end_y}) 超出视频高度 {self.height}")
                    frame_img = Image.fromarray(frame_rgba, 'RGBA')
                    frame_filename = png_pattern % frame_idx
                    try:
                         frame_img.save(frame_filename)
                    except Exception as save_err:
                         logger.error(f"无法保存帧 {frame_idx}: {save_err}")
                         raise
                    # --- 结束帧生成 --- 

                logger.info("PNG序列生成完成，开始使用ffmpeg合成视频...")
                
                # 准备ffmpeg命令 (强制使用 prores_ks CPU)
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(self.fps),
                    "-i", png_pattern,
                ]
                if audio_path and os.path.exists(audio_path):
                     ffmpeg_cmd.extend(["-i", audio_path])
                
                ffmpeg_cmd.extend([
                    "-c:v", "prores_ks",    # ProRes 4444（支持alpha） - CPU
                    "-profile:v", "4",      # 4444配置文件
                    "-pix_fmt", "yuva444p10le", # 带alpha的像素格式，10位
                    "-alpha_bits", "16",     # 显式设置alpha位
                    "-vendor", "ap10",      # Apple厂商ID
                ])
                logger.info("强制使用 prores_ks (CPU) 为透明 .mov 进行编码...")

                if audio_path and os.path.exists(audio_path):
                    ffmpeg_cmd.extend(["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest"]) 
                else:
                    ffmpeg_cmd.extend(["-map", "0:v:0"]) 
                ffmpeg_cmd.append(output_path)

                logger.info(f"执行ffmpeg命令: {' '.join(ffmpeg_cmd)}")
                # 执行ffmpeg (与之前类似，移除GPU错误提示)
                try:
                    process = subprocess.Popen(
                        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
                    )
                    for line in process.stdout: logger.info(f"ffmpeg: {line.strip()}")
                    return_code = process.wait()
                    if return_code != 0:
                        raise Exception(f"ffmpeg执行失败，返回码: {return_code}.")
                    logger.info(f"透明视频创建成功: {output_path}")
                except FileNotFoundError:
                     logger.error("ffmpeg 未找到。请确保ffmpeg已安装并添加到系统PATH。")
                     raise
                except Exception as ffmpeg_error:
                    logger.error(f"ffmpeg合成视频失败: {ffmpeg_error}", exc_info=True)
                    raise
            return output_path
            
        else:
            # --- 不透明路径: 尝试 GPU (h264_nvenc) 回退 CPU (libx264) + .mp4 --- 
            logger.info("处理不透明视频：使用 MoviePy (尝试GPU h264_nvenc，回退CPU libx264) 输出 .mp4...")
            
            # 确保输出是 .mp4
            if not output_path.lower().endswith(".mp4"):
                 logger.warning(f"不透明输出建议使用.mp4格式，但指定路径为{output_path}。将强制使用.mp4扩展名。")
                 output_path = os.path.splitext(output_path)[0] + ".mp4"
                 logger.info(f"实际输出路径调整为: {output_path}")

            # 定义 make_frame 函数 (与之前 make_frame_rgb 逻辑相同)
            def make_frame(t):
                frame_idx = int(t * self.fps)
                # --- 滚动逻辑 (与之前相同) ---
                if frame_idx < padding_frames_start:
                    current_position = 0
                elif frame_idx < padding_frames_start + scroll_frames:
                    scroll_progress_frames = frame_idx - padding_frames_start
                    current_position = scroll_progress_frames * self.scroll_speed
                    current_position = min(current_position, scroll_distance)
                else:
                    current_position = scroll_distance
                # --- 结束滚动逻辑 --- 

                # --- 帧生成 (与之前 RGB 路径相同) ---
                final_bg_color = (0, 0, 0) # 默认背景
                if bg_color and len(bg_color) >= 3:
                     final_bg_color = bg_color[:3]
                frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(final_bg_color, dtype=np.uint8)
                img_start_y = int(current_position)
                img_end_y = min(img_height, img_start_y + self.height)
                frame_start_y = 0
                frame_end_y = img_end_y - img_start_y
                if img_start_y < img_end_y and frame_start_y < frame_end_y and frame_end_y <= self.height:
                     img_h_slice = slice(img_start_y, img_end_y)
                     img_w_slice = slice(0, min(self.width, img_width))
                     frame_h_slice = slice(frame_start_y, frame_end_y)
                     frame_w_slice = slice(0, min(self.width, img_width))
                     img_section = img_array[img_h_slice, img_w_slice]
                     target_area = frame[frame_h_slice, frame_w_slice]
                     # Alpha 混合 (即使背景不透明，前景可能有透明度)
                     alpha = img_section[:, :, 3:4].astype(np.float32) / 255.0
                     blended = (img_section[:, :, :3].astype(np.float32) * alpha +
                                target_area.astype(np.float32) * (1.0 - alpha))
                     if target_area.shape[:2] == blended.shape[:2]:
                          np.copyto(target_area, blended.astype(np.uint8))
                     else:
                           logger.warning(f"帧 {frame_idx}: 不透明混合形状不匹配。目标: {target_area.shape}, 混合: {blended.shape}")
                elif frame_end_y > self.height:
                     logger.warning(f"帧 {frame_idx}: 计算出的 frame_end_y ({frame_end_y}) 超出视频高度 {self.height}")
                return frame
                # --- 结束帧生成 --- 

            # --- MoviePy 写入，带 GPU->CPU 回退逻辑 --- 
            clip = VideoClip(make_frame, duration=duration)
            if audio_path and os.path.exists(audio_path):
                try:
                    from moviepy.editor import AudioFileClip
                    audio_clip = AudioFileClip(audio_path)
                    if audio_clip.duration > duration: audio_clip = audio_clip.subclip(0, duration)
                    elif audio_clip.duration < duration: audio_clip = audio_clip.loop(duration=duration)
                    clip = clip.set_audio(audio_clip)
                except Exception as audio_err: logger.error(f"添加音频失败: {audio_err}")
            clip = clip.set_fps(self.fps)

            # 尝试 GPU 编码
            gpu_codec = preferred_codec # (h264_nvenc)
            cpu_codec = "libx264"
            try:
                logger.info(f"尝试使用 {gpu_codec} (GPU) 编码器写入视频...")
                clip.write_videofile(output_path, codec=gpu_codec, audio_codec="aac" if clip.audio else None, logger='bar')
                logger.info(f"使用 {gpu_codec} (GPU) 创建视频成功: {output_path}")
            except Exception as gpu_write_err:
                logger.warning(f"使用 {gpu_codec} (GPU) 编码失败: {gpu_write_err}")
                logger.warning("GPU编码失败提示：请确保安装了支持NVENC的ffmpeg版本，并且Nvidia驱动已正确安装。")
                logger.info(f"回退到使用 {cpu_codec} (CPU) 编码器...")
                try:
                     # 回退到 CPU 编码
                     clip.write_videofile(output_path, codec=cpu_codec, audio_codec="aac" if clip.audio else None, logger='bar')
                     logger.info(f"使用 {cpu_codec} (CPU) 创建视频成功: {output_path}")
                except Exception as cpu_write_err:
                     logger.error(f"使用 {cpu_codec} (CPU) 编码也失败: {cpu_write_err}", exc_info=True)
                     raise # 如果CPU也失败，则抛出异常
            # --- 结束 MoviePy 写入 --- 
            
            return output_path 