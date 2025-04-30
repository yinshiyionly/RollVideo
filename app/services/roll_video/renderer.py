import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PIL_VERSION
from typing import Dict, Tuple, List, Optional, Union
import textwrap
import platform
import logging
import subprocess
import tempfile
import shutil
import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import mmap  # 导入mmap模块用于共享内存
from numba import jit, pr

logger = logging.getLogger(__name__)

try:
    # 使用Numba JIT编译优化alpha混合计算
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def blend_alpha_fast(source, target, alpha):
        """
        使用Numba加速的alpha混合计算
        
        Args:
            source: 源图像数据 (RGB)
            target: 目标图像数据 (RGB)
            alpha: Alpha通道数据
            
        Returns:
            混合后的图像数据
        """
        # 确保输入是3D数组
        result = np.empty_like(target)
        h, w, c = source.shape
        
        # 并行处理每一行
        for y in prange(h):
            for x in range(w):
                for ch in range(c):
                    a = alpha[y, x, 0]
                    result[y, x, ch] = source[y, x, ch] * a + target[y, x, ch] * (1.0 - a)
        
        return result
    
    # JIT优化的图像切片复制函数
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def copy_image_section_fast(target, source, max_h, max_w):
        """使用Numba加速的图像区域复制"""
        h = min(source.shape[0], max_h)
        w = min(source.shape[1], max_w)
        
        # 并行处理每一行
        for y in prange(h):
            for x in range(w):
                for c in range(source.shape[2]):
                    target[y, x, c] = source[y, x, c]
        
        return target
    
    NUMBA_AVAILABLE = True
    logger.info("Numba JIT编译支持已启用，性能将显著提升")
except ImportError:
    # 如果Numba不可用，提供普通的回退函数
    def blend_alpha_fast(source, target, alpha):
        """普通的alpha混合计算（无Numba）"""
        return source * alpha + target * (1.0 - alpha)
    
    def copy_image_section_fast(target, source, max_h, max_w):
        """普通的图像区域复制（无Numba）"""
        h = min(source.shape[0], max_h)
        w = min(source.shape[1], max_w)
        target[:h, :w] = source[:h, :w]
        return target
    
    NUMBA_AVAILABLE = False
    logger.warning("Numba未安装，将使用标准Python函数。安装Numba可大幅提升性能：pip install numba")

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
    """视频渲染器，负责创建滚动效果的视频，使用ffmpeg管道和线程读取优化"""
    
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
        # 增加CPU线程使用数量，最大化利用8核CPU
        self.num_threads = min(7, os.cpu_count() or 1)  # 使用7个线程用于帧生成，保留1个核心给ffmpeg
        # 优化3: 设置批量处理的大小
        self.batch_size = 8  # 增加到8帧一批(原来是4帧)
        # 获取系统内存大小(GB)
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            # 如果内存超过28GB，则使用更大的批量
            if total_memory_gb > 28:
                self.batch_size = 16  # 更大内存时使用更大批量
        except ImportError:
            pass  # 如果无法获取内存大小，保持默认值
            
        logger.info(f"视频渲染器初始化: 宽={width}, 高={height}, 帧率={fps}, 滚动速度={scroll_speed}, 线程数={self.num_threads}, 批量={self.batch_size}")
    
    def _get_ffmpeg_command(
        self,
        output_path: str,
        pix_fmt: str,
        codec_and_output_params: List[str], # 重命名以更清晰
        audio_path: Optional[str]
    ) -> List[str]:
        """构造基础的ffmpeg命令"""
        command = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", pix_fmt, 
            "-r", str(self.fps),
            "-i", "-",  # 从 stdin 读取
        ]
        if audio_path and os.path.exists(audio_path):
            command.extend(["-i", audio_path])
        
        # 添加视频编码器和特定的输出参数 (如 -movflags)
        command.extend(codec_and_output_params) 
        
        if audio_path and os.path.exists(audio_path):
            command.extend(["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
        else:
            command.extend(["-map", "0:v:0"])
            
        command.append(output_path)
        return command

    def _reader_thread(self, pipe, output_queue):
        """读取管道输出并放入队列"""
        try:
            with pipe:
                for line in iter(pipe.readline, b''):
                    output_queue.put(line)
        finally:
            output_queue.put(None) # 发送结束信号

    def _prepare_frame(self, frame_idx, img_array, img_height, img_width, total_frames, 
                     padding_frames_start, scroll_frames, scroll_distance,
                     transparency_required, background_frame_rgb=None):
        """生成单帧的像素数据"""
        # 优化2: 实现帧位置缓存
        if not hasattr(self, '_position_cache'):
            self._position_cache = {}
        
        # 从缓存获取位置信息
        position_key = frame_idx
        if position_key in self._position_cache:
            img_start_y, img_end_y, frame_start_y, frame_end_y = self._position_cache[position_key]
        else:
            # 计算当前帧的滚动位置 - 使用浮点数提高精度
            if frame_idx < padding_frames_start: 
                current_position = 0.0
            elif frame_idx < padding_frames_start + scroll_frames:
                # 使用浮点数计算，避免舍入误差累积
                scroll_progress = frame_idx - padding_frames_start
                current_position = scroll_progress * self.scroll_speed
                current_position = min(current_position, float(scroll_distance))
            else: 
                current_position = float(scroll_distance)
                
            # 精确计算图像和帧的切片位置
            img_start_y = int(current_position)
            img_end_y = min(img_height, img_start_y + self.height)
            frame_start_y = 0
            frame_end_y = img_end_y - img_start_y
            
            # 将计算结果存入缓存
            self._position_cache[position_key] = (img_start_y, img_end_y, frame_start_y, frame_end_y)
        
        # 生成帧数据
        output_frame_data = None
        if img_start_y < img_end_y and frame_start_y < frame_end_y and frame_end_y <= self.height:
            img_h_slice = slice(img_start_y, img_end_y)
            img_w_slice = slice(0, min(self.width, img_width))
            frame_h_slice = slice(frame_start_y, frame_end_y)
            frame_w_slice = slice(0, min(self.width, img_width))
            source_section = img_array[img_h_slice, img_w_slice]
            
            # 优化2: 为不同透明度情况重用帧缓冲区
            if transparency_required:
                # 为RGBA模式创建/重用缓冲区
                if not hasattr(self, '_frame_buffer_rgba'):
                    self._frame_buffer_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
                frame_rgba = self._frame_buffer_rgba.copy()  # 使用copy以避免副作用
                
                target_area = frame_rgba[frame_h_slice, frame_w_slice]
                if target_area.shape[:2] == source_section.shape[:2]: 
                    np.copyto(target_area, source_section)
                else: 
                    # 优化5: 使用Numba优化的函数
                    copy_width = min(target_area.shape[1], source_section.shape[1])
                    copy_height = min(target_area.shape[0], source_section.shape[0])
                    target_area = copy_image_section_fast(
                        target_area, 
                        source_section, 
                        copy_height, 
                        copy_width
                    )
                output_frame_data = frame_rgba.tobytes()
            else:
                # 为RGB模式创建/重用缓冲区
                frame_rgb = background_frame_rgb.copy()
                target_area = frame_rgb[frame_h_slice, frame_w_slice]
                if target_area.shape[:2] == source_section.shape[:2]:
                    alpha = source_section[:, :, 3:4].astype(np.float32) / 255.0
                    # 优化5: 使用Numba优化的alpha混合
                    blended = blend_alpha_fast(
                        source_section[:, :, :3].astype(np.float32),
                        target_area.astype(np.float32),
                        alpha
                    )
                    np.copyto(target_area, blended.astype(np.uint8))
                else:
                    copy_width = min(target_area.shape[1], source_section.shape[1])
                    source_section_crop = source_section[:target_area.shape[0], :copy_width]
                    target_area_crop = target_area[:target_area.shape[0], :copy_width]
                    alpha = source_section_crop[:, :, 3:4].astype(np.float32) / 255.0
                    # 优化5: 使用Numba优化的alpha混合
                    blended = blend_alpha_fast(
                        source_section_crop[:, :, :3].astype(np.float32),
                        target_area_crop.astype(np.float32),
                        alpha
                    )
                    target_area[:target_area.shape[0], :copy_width] = blended.astype(np.uint8)
                output_frame_data = frame_rgb.tobytes()
        else:
            if transparency_required:
                if not hasattr(self, '_empty_frame_rgba'):
                    self._empty_frame_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
                output_frame_data = self._empty_frame_rgba.tobytes()
            else:
                output_frame_data = background_frame_rgb.tobytes()
                
        return output_frame_data

    def create_scrolling_video(
        self,
        image: Image.Image,
        output_path: str,
        text_actual_height: int,
        transparency_required: bool,
        preferred_codec: str, # 仍然接收 h264_nvenc 作为首选
        audio_path: Optional[str] = None,
        bg_color: Optional[Tuple[int, int, int, int]] = None,
        frame_skip: int = 1  # 新增参数：每x帧渲染1帧，然后让ffmpeg复制
    ) -> str:
        """创建滚动视频，使用ffmpeg管道优化，并应用推荐的编码参数"""
        
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]
        if img_array.shape[2] != 4:
             logger.warning("输入图像非RGBA，尝试转换")
             image = image.convert("RGBA")
             img_array = np.array(image)
             if img_array.shape[2] != 4: raise ValueError("无法转换图像为RGBA")
        
        # 修改滚动距离计算，减少视频末尾的多余空白
        # 原来: scroll_distance = text_actual_height + self.height
        # 现在只需要确保最后一行文字完全滚出屏幕即可
        # 添加约20%的视频高度作为底部缓冲区
        bottom_padding = int(self.height * 0.2)  
        scroll_distance = text_actual_height + bottom_padding
        
        # 使用浮点数精确计算滚动位置，避免累积舍入误差
        scroll_frames_float = scroll_distance / self.scroll_speed if self.scroll_speed > 0 else 0
        scroll_frames = int(scroll_frames_float + 0.5)  # 四舍五入到最接近的整数
        
        padding_frames_start = int(self.fps * 0.5)
        padding_frames_end = int(self.fps * 0.5)
        total_frames = padding_frames_start + scroll_frames + padding_frames_end
        duration = total_frames / self.fps
        logger.info(f"文本高:{text_actual_height}, 图像高:{img_height}, 视频高:{self.height}")
        logger.info(f"底部填充:{bottom_padding}, 滚动距离:{scroll_distance}, 滚动帧:{scroll_frames}, 总帧:{total_frames}, 时长:{duration:.2f}s")
        
        # 优先使用不跳帧模式，以保证滚动平滑度
        actual_frame_skip = 1  # 强制设为1，不使用跳帧
        if frame_skip > 1:
            logger.info(f"为保证滚动平滑度，不使用跳帧，改用多线程优化")
            
        # 透明视频的线程优化
        actual_threads = self.num_threads
        if transparency_required:
            # 透明视频的ProRes编码较消耗CPU，留更多核心给编码器
            actual_threads = max(3, self.num_threads - 4)  # 减少更多线程
            logger.info(f"透明视频特殊处理: 使用{actual_threads}个线程")
        
        logger.info(f"输出:{output_path}, 透明:{transparency_required}, 首选编码器:{preferred_codec}, 跳帧率:{actual_frame_skip}")
        # --- 结束滚动参数计算 ---
        
        ffmpeg_pix_fmt = ""
        video_codec_and_output_params = [] 
        cpu_fallback_codec_and_output_params = []
        final_bg_color_rgb = None
        background_frame_rgb = None
        
        # 计算实际渲染帧率
        render_fps = self.fps
        if actual_frame_skip > 1:
            render_fps = max(1, self.fps // actual_frame_skip)
            logger.info(f"启用跳帧: 输出帧率={self.fps}, 实际渲染帧率={render_fps}")
            
        if transparency_required:
            ffmpeg_pix_fmt = "rgba"
            output_path = os.path.splitext(output_path)[0] + ".mov"
            # 透明视频编码参数优化 - 降低质量换取速度
            video_codec_and_output_params = [
                "-c:v", "prores_ks", 
                "-profile:v", "0",        # 从4(高质量)改为0(代理质量)
                "-pix_fmt", "yuva444p",   # 降低位深度
                "-alpha_bits", "8",       # 降低alpha通道质量
                "-vendor", "ap10",
                # 增加缓冲区大小，提高流畅度
                "-bufsize", "20M",
                # 增强流畅度参数
                "-g", str(self.fps * 2),  # 每2秒一个关键帧
                "-bf", "2"                # 最多2个B帧，增强平滑度
            ]
            logger.info(f"设置ffmpeg(透明): 输入={ffmpeg_pix_fmt}, 输出={output_path}, 参数={' '.join(video_codec_and_output_params)}")
        else:
            ffmpeg_pix_fmt = "rgb24"
            output_path = os.path.splitext(output_path)[0] + ".mp4"
            # GPU 参数优化: 降低质量换取速度，同时优化平滑度
            video_codec_and_output_params = [
                "-c:v", preferred_codec,  # h264_nvenc
                "-preset", "p4",          # 更快速度的preset (p4比p3更快)
                "-rc:v", "vbr",
                "-cq:v", "28",            # 降低质量 (21→28)，值越大质量越低
                "-b:v", "0",
                "-pix_fmt", "yuv420p",
                # 提高流畅度的参数
                "-g", str(self.fps * 2),  # GOP大小，每2秒一个关键帧
                "-bf", "2",               # 最多2个B帧
                "-bufsize", "50M",        # 增加缓冲区大小
                "-refs", "3",             # 使用3个参考帧
                "-movflags", "+faststart"
            ]
            # 添加NVENC特别参数以提高性能
            if preferred_codec == "h264_nvenc":
                video_codec_and_output_params.extend([
                    "-gpu", "0",          # 明确指定GPU
                    "-surfaces", "64",     # 增加表面缓冲数
                    "-delay", "0",         # 降低延迟
                    "-no-scenecut", "1"    # 禁用场景切换检测以提高速度
                ])
            
            # CPU 回退参数优化: 降低质量换取速度，加强流畅度
            cpu_fallback_codec_and_output_params = [
                "-c:v", "libx264",
                "-crf", "28",              # 降低质量 (21→28)
                "-preset", "ultrafast",    # 速度最快的preset
                "-tune", "fastdecode",     # 优化解码速度
                "-pix_fmt", "yuv420p",
                # 提高流畅度的参数
                "-g", str(self.fps * 2),   # GOP大小，每2秒一个关键帧 
                "-bf", "2",                # 最多2个B帧
                "-bufsize", "50M",         # 增加缓冲区大小
                "-refs", "3",              # 使用3个参考帧
                "-movflags", "+faststart"
            ]
            logger.info(f"设置ffmpeg(不透明): 输入={ffmpeg_pix_fmt}, 输出={output_path}")
            logger.info(f"  首选GPU参数: {' '.join(video_codec_and_output_params)}")
            logger.info(f"  回退CPU参数: {' '.join(cpu_fallback_codec_and_output_params)}")
            # 准备背景色 (RGB)
            final_bg_color_rgb = (0, 0, 0)
            if bg_color and len(bg_color) >= 3:
                 final_bg_color_rgb = bg_color[:3]
            background_frame_rgb = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(final_bg_color_rgb, dtype=np.uint8)
        
        # --- 重构 run_ffmpeg_with_pipe 使用线程和多线程帧生成 --- 
        def run_ffmpeg_with_pipe(current_codec_params: List[str], is_gpu_attempt: bool) -> bool:
            ffmpeg_cmd = self._get_ffmpeg_command(output_path, ffmpeg_pix_fmt, current_codec_params, audio_path)
            # 添加帧率转换参数（如果启用跳帧）
            if actual_frame_skip > 1:
                # 修改输入帧率（之前的ffmpeg参数中的-r值）
                r_index = ffmpeg_cmd.index("-r")
                ffmpeg_cmd[r_index+1] = str(render_fps)
                # 添加输出帧率参数
                ffmpeg_cmd.insert(-1, "-r")  # 在输出文件前插入
                ffmpeg_cmd.insert(-1, str(self.fps))  # 在输出文件前插入
            
            # 为透明视频添加线程限制
            if transparency_required and "-threads" not in current_codec_params:
                threads_index = next((i for i, x in enumerate(current_codec_params) if x == "-profile:v"), -1)
                if threads_index != -1:
                    current_codec_params.insert(threads_index, "-threads")
                    current_codec_params.insert(threads_index+1, str(actual_threads))
                    logger.info(f"添加FFmpeg线程限制: -threads {actual_threads}")
            
            logger.info(f"执行ffmpeg命令: {' '.join(ffmpeg_cmd)}")
            process = None
            stdout_q = queue.Queue()
            stderr_q = queue.Queue()
            stdout_thread = None
            stderr_thread = None
            
            try:
                # 对于透明视频，设置环境变量限制线程
                env = os.environ.copy()
                if transparency_required:
                    env["OMP_NUM_THREADS"] = str(actual_threads)
                    env["MKL_NUM_THREADS"] = str(actual_threads)
                    env["OPENBLAS_NUM_THREADS"] = str(actual_threads)
                    env["VECLIB_MAXIMUM_THREADS"] = str(actual_threads)
                    logger.info(f"设置FFmpeg环境线程限制: OMP_NUM_THREADS={env['OMP_NUM_THREADS']}")
                    
                    # 尝试设置CPU亲和性 (仅在Linux系统)
                    try:
                        if hasattr(os, "sched_setaffinity") and platform.system() == "Linux":
                            import psutil
                            proc = psutil.Process()
                            # 设置只使用前8个核心
                            proc.cpu_affinity(list(range(min(8, os.cpu_count() or 1))))
                            logger.info(f"设置CPU亲和性: {proc.cpu_affinity()}")
                    except (ImportError, AttributeError, OSError) as e:
                        logger.warning(f"设置CPU亲和性失败: {e}")
                
                process = subprocess.Popen(
                    ffmpeg_cmd, stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                    env=env  # 使用修改后的环境变量
                )
                
                # --- 启动 stdout 和 stderr 读取线程 ---
                stdout_thread = threading.Thread(target=self._reader_thread, args=(process.stdout, stdout_q), daemon=True)
                stderr_thread = threading.Thread(target=self._reader_thread, args=(process.stderr, stderr_q), daemon=True)
                stdout_thread.start()
                stderr_thread.start()
                # --- 结束线程启动 ---
                
                # --- 多线程帧生成和管道写入 --- 
                codec_name = "unknown"
                try: codec_name = current_codec_params[current_codec_params.index("-c:v") + 1]
                except (ValueError, IndexError): pass
                
                # 实际渲染的帧数（考虑跳帧）
                render_frame_count = total_frames
                if actual_frame_skip > 1:
                    render_frame_count = total_frames // actual_frame_skip
                    if total_frames % actual_frame_skip > 0:
                        render_frame_count += 1  # 确保渲染所有必要帧
                
                # 创建帧生成线程池和队列 - 增大队列大小以提高性能
                # 增加队列容量以更好地平衡生产者和消费者
                queue_size = 96  # 增加到96（之前是36），充分利用内存
                if transparency_required:
                    queue_size = 64  # 透明视频也增加到64（之前是24）
                
                # 如果系统内存充足，进一步增加队列大小
                try:
                    import psutil
                    total_memory_gb = psutil.virtual_memory().total / (1024**3)
                    free_memory_gb = psutil.virtual_memory().available / (1024**3)
                    
                    # 如果有大量可用内存，则大幅增加队列大小
                    if free_memory_gb > 15 and total_memory_gb > 28:
                        if transparency_required:
                            queue_size = min(128, queue_size * 2)  # 最大增加到128
                        else:
                            queue_size = min(256, queue_size * 2)  # 最大增加到256
                        
                    logger.info(f"系统总内存:{total_memory_gb:.1f}GB，可用内存:{free_memory_gb:.1f}GB，调整队列大小:{queue_size}")
                except ImportError:
                    pass  # 如果无法获取内存信息，保持默认值
                
                frame_queue = queue.Queue(maxsize=queue_size)
                frame_pool = ThreadPoolExecutor(max_workers=actual_threads)
                
                # 计算批处理任务数量
                batch_size = min(self.batch_size, render_frame_count)
                num_batches = (render_frame_count + batch_size - 1) // batch_size  # 向上取整
                
                # 预填充队列（启动初始批处理任务） - 增加预填充比例
                prefill_factor = 0.8  # 预填充因子，填充队列的80%
                prefill_batches = min(int(queue_size * prefill_factor / batch_size), num_batches)
                
                # 优化4: 为帧数据创建共享内存 - 增加总缓冲区大小
                frame_byte_size = self.width * self.height * (4 if transparency_required else 3)
                # 增加预留系数，确保缓冲区足够大
                memory_reserve_factor = 1.2
                total_buffer_size = int(frame_byte_size * batch_size * prefill_batches * memory_reserve_factor)
                
                # 限制单次分配的最大共享内存（避免过大导致失败）
                max_buffer_size = 2 * 1024 * 1024 * 1024  # 2GB
                if total_buffer_size > max_buffer_size:
                    # 如果太大，分成多个缓冲区
                    total_buffer_size = max_buffer_size
                    logger.info(f"共享内存请求过大，限制为{max_buffer_size/1024/1024:.1f}MB")
                
                try:
                    # 尝试创建共享内存
                    shared_buffer = mmap.mmap(-1, total_buffer_size)
                    use_shared_memory = True
                    logger.info(f"创建共享内存缓冲区: {total_buffer_size/1024/1024:.1f}MB，预填充{prefill_batches}批次")
                except (ImportError, OSError, ValueError) as e:
                    logger.warning(f"无法创建共享内存: {e}，将使用标准队列")
                    use_shared_memory = False
                    shared_buffer = None
                
                # 优化3+4: 批处理帧生成 + 可选共享内存
                def generate_frame_batch(start_idx, count, buffer_offset=0):
                    """批量生成帧，减少线程切换开销，可选使用共享内存"""
                    if use_shared_memory:
                        # 使用共享内存版本
                        frames_metadata = []  # 存储每帧的元数据(偏移量和大小)
                        current_offset = buffer_offset
                        
                        for i in range(count):
                            idx = start_idx + i
                            if idx >= render_frame_count:
                                break
                            actual_frame_idx = idx * actual_frame_skip if actual_frame_skip > 1 else idx
                            actual_frame_idx = min(actual_frame_idx, total_frames - 1)
                            
                            frame_data = self._prepare_frame(
                                actual_frame_idx, img_array, img_height, img_width, 
                                total_frames, padding_frames_start, scroll_frames, 
                                scroll_distance, transparency_required, background_frame_rgb
                            )
                            
                            # 写入共享内存
                            data_size = len(frame_data)
                            shared_buffer[current_offset:current_offset+data_size] = frame_data
                            frames_metadata.append((current_offset, data_size))
                            current_offset += data_size
                        
                        frame_queue.put(frames_metadata)
                    else:
                        # 标准版本(不使用共享内存)
                        frames_batch = []
                        for i in range(count):
                            idx = start_idx + i
                            if idx >= render_frame_count:
                                break
                            actual_frame_idx = idx * actual_frame_skip if actual_frame_skip > 1 else idx
                            actual_frame_idx = min(actual_frame_idx, total_frames - 1)
                            
                            frame_data = self._prepare_frame(
                                actual_frame_idx, img_array, img_height, img_width, 
                                total_frames, padding_frames_start, scroll_frames, 
                                scroll_distance, transparency_required, background_frame_rgb
                            )
                            frames_batch.append(frame_data)
                        
                        frame_queue.put(frames_batch)
                
                logger.info(f"启动{prefill_batches}个批处理任务(每批{batch_size}帧)，使用{actual_threads}个线程")
                
                if use_shared_memory:
                    # 如果使用共享内存，计算每个批处理任务的缓冲区偏移量
                    buffer_size_per_batch = total_buffer_size // prefill_batches
                    for i in range(prefill_batches):
                        start_idx = i * batch_size
                        buffer_offset = i * buffer_size_per_batch
                        frame_pool.submit(generate_frame_batch, start_idx, batch_size, buffer_offset)
                else:
                    # 不使用共享内存，正常提交任务
                    for i in range(prefill_batches):
                        start_idx = i * batch_size
                        frame_pool.submit(generate_frame_batch, start_idx, batch_size)
                
                # 主循环：处理帧队列和提交新任务
                frame_iterator = tqdm.tqdm(range(num_batches), desc=f"编码 ({codec_name}) ")
                batch_idx = 0
                
                for _ in frame_iterator:
                    # 提交下一批帧的生成任务
                    next_batch_idx = batch_idx + prefill_batches
                    if next_batch_idx < num_batches:
                        start_idx = next_batch_idx * batch_size
                        if use_shared_memory:
                            # 计算下一批次的缓冲区偏移量 (循环使用缓冲区)
                            buffer_offset = (next_batch_idx % prefill_batches) * buffer_size_per_batch
                            frame_pool.submit(generate_frame_batch, start_idx, batch_size, buffer_offset)
                        else:
                            frame_pool.submit(generate_frame_batch, start_idx, batch_size)
                    
                    # 从队列获取当前批帧数据
                    try:
                        batch_data = frame_queue.get(timeout=60)  # 添加超时防止死锁
                        if batch_data:
                            try:
                                if use_shared_memory:
                                    # 从共享内存读取并写入管道
                                    for offset, size in batch_data:
                                        frame_bytes = shared_buffer[offset:offset+size]
                                        process.stdin.write(frame_bytes)
                                else:
                                    # 直接写入帧数据
                                    for frame_data in batch_data:
                                        process.stdin.write(frame_data)
                            except (IOError, BrokenPipeError) as e:
                                logger.error(f"写入ffmpeg管道时出错: {e}")
                                # 写入失败时，尝试获取stderr
                                stderr_lines_on_error = []
                                while True:
                                    try: line = stderr_q.get(timeout=0.1)
                                    except queue.Empty: break
                                    if line is None: break
                                    stderr_lines_on_error.append(line.decode(errors='ignore').strip())
                                stderr_content_on_error = "\n".join(stderr_lines_on_error)
                                logger.error(f"ffmpeg stderr (写入时):\n{stderr_content_on_error}")
                                raise Exception(f"ffmpeg进程意外终止: {e}") from e
                    except queue.Empty:
                        logger.error("等待帧数据超时，可能是帧生成线程卡住了")
                        raise Exception("帧生成超时")
                    
                    batch_idx += 1
                
                # 关闭线程池和清理
                frame_pool.shutdown(wait=False)
                logger.info("所有帧已生成并写入管道，关闭stdin...")
                process.stdin.close()
                
                # 清理共享内存
                if use_shared_memory and shared_buffer:
                    try:
                        shared_buffer.close()
                        logger.info("共享内存缓冲区已关闭")
                    except Exception as e:
                        logger.warning(f"关闭共享内存缓冲区时出错: {e}")
                
                # --- 等待 ffmpeg 进程结束 --- 
                logger.info("等待ffmpeg进程结束...")
                process.wait() 
                return_code = process.returncode
                # --- 结束等待 --- 

                # --- 等待读取线程结束并收集输出 --- 
                logger.info("等待输出读取线程结束...")
                stdout_thread.join()
                stderr_thread.join()
                
                stdout_lines = []
                while not stdout_q.empty():
                    line = stdout_q.get()
                    if line is not None: stdout_lines.append(line.decode(errors='ignore').strip())
                    
                stderr_lines = []
                while not stderr_q.empty():
                    line = stderr_q.get()
                    if line is not None: stderr_lines.append(line.decode(errors='ignore').strip())
                
                # 记录输出
                if stdout_lines: 
                    stdout_content = "\n".join(stdout_lines)
                    logger.info(f"ffmpeg stdout:\n{stdout_content}")
                if stderr_lines: 
                    stderr_content = "\n".join(stderr_lines)
                    logger.info(f"ffmpeg stderr:\n{stderr_content}")
                # --- 结束收集输出 --- 
                
                if return_code == 0:
                    logger.info(f"使用 {codec_name} 编码成功完成。")
                    return True
                else:
                    logger.error(f"ffmpeg ({codec_name}) 执行失败，返回码: {return_code}")
                    if is_gpu_attempt: logger.warning("GPU编码失败提示：检查ffmpeg版本/驱动/显存。")
                    return False
            
            except FileNotFoundError: logger.error("ffmpeg 未找到。请确保已安装并加入PATH。"); raise
            except Exception as e: 
                logger.error(f"执行 ffmpeg ({codec_name}) 时出错: {e}", exc_info=True)
                logger.error(f"命令: {' '.join(ffmpeg_cmd)}")
                # 尝试收集最后的 stderr
                stderr_lines_on_except = []
                while True:
                    try: line = stderr_q.get(timeout=0.1)
                    except queue.Empty: break
                    if line is None: break
                    stderr_lines_on_except.append(line.decode(errors='ignore').strip())
                # 修正：先 join 再放入 f-string
                if stderr_lines_on_except: 
                    stderr_content_on_except = "\n".join(stderr_lines_on_except)
                    logger.error(f"ffmpeg stderr (异常时):\n{stderr_content_on_except}")
                # 不返回 False，让异常传播
            finally: 
                # 清理：确保进程终止和管道关闭
                if process and process.poll() is None: 
                    logger.warning("尝试终止 ffmpeg 进程 (finally)...")
                    try: process.terminate() 
                    except ProcessLookupError: pass # 进程可能已经结束
                    try: process.wait(timeout=1) 
                    except subprocess.TimeoutExpired: 
                        logger.warning("ffmpeg 进程超时，强制终止 (kill)")
                        try: process.kill() 
                        except ProcessLookupError: pass
                    except Exception as e_wait: logger.error(f"等待终止时出错: {e_wait}")
                    logger.warning("已终止 ffmpeg 进程")
                # 关闭管道句柄（如果它们还打开着）
                for pipe in [process.stdin, process.stdout, process.stderr] if process else []:
                    if pipe and not pipe.closed:
                        try: pipe.close()
                        except: pass
                # 确保线程已结束 (即使之前出错)
                if stdout_thread and stdout_thread.is_alive(): stdout_thread.join(timeout=0.5)
                if stderr_thread and stderr_thread.is_alive(): stderr_thread.join(timeout=0.5)
        # --- 结束 run_ffmpeg_with_pipe 函数定义 ---

        # --- 执行编码 --- 
        success = run_ffmpeg_with_pipe(video_codec_and_output_params, is_gpu_attempt=(not transparency_required))
        if not success and not transparency_required and cpu_fallback_codec_and_output_params:
            logger.info(f"GPU ({preferred_codec}) 编码失败，尝试回退到 CPU ({cpu_fallback_codec_and_output_params[1]})...")
            success = run_ffmpeg_with_pipe(cpu_fallback_codec_and_output_params, is_gpu_attempt=False)
            if not success:
                 logger.error("CPU 回退编码也失败了。")
                 raise Exception("视频编码失败（GPU和CPU均失败）")
        elif not success and transparency_required:
             logger.error("透明视频 (CPU prores_ks) 编码失败。")
             raise Exception("透明视频编码失败")
             
        logger.info(f"视频渲染流程完成。输出文件: {output_path}")
        return output_path 