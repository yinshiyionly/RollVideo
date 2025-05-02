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
import gc  # 添加垃圾回收模块
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import ctypes  # 用于共享内存优化

logger = logging.getLogger(__name__)

# 设置NumPy以使用多线程加速计算
try:
    # 尝试设置NumPy使用更多线程以提高性能
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '4'  # 设置OpenMP线程数
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = '4'  # 设置MKL线程数
    logger.info(f"已设置NumPy优化: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
except Exception as e:
    logger.warning(f"设置NumPy线程优化失败: {e}")

# 尝试导入资源限制模块（如果可用）
try:
    import resource
    HAS_RESOURCE_MODULE = True
except ImportError:
    HAS_RESOURCE_MODULE = False
    logger.warning("无法导入resource模块，将不能精确限制CPU和内存使用")

def limit_resources():
    """尝试限制进程资源使用"""
    if HAS_RESOURCE_MODULE:
        try:
            # 设置内存限制 (30GB)
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            memory_limit = 30 * 1024 * 1024 * 1024  # 30GB in bytes
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))
            
            # 设置CPU时间限制 (限制单CPU使用)
            cpu_time_limit = 24 * 60 * 60  # 24小时（非常宽松的限制）
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, hard))
            
            logger.info(f"已设置资源限制: 内存={memory_limit/(1024*1024*1024):.1f}GB, CPU时间={cpu_time_limit/3600:.1f}小时")
        except Exception as e:
            logger.warning(f"设置资源限制失败: {e}")
    else:
        logger.info("由于缺少resource模块，资源限制将通过其他方式实现")

# 在导入时尝试设置资源限制
try:
    limit_resources()
except Exception as e:
    logger.warning(f"尝试限制资源时发生错误: {e}")

# 在多线程/多进程中共享的全局变量
_g_img_array = None  # 全局共享的图像数组

def _process_frame(args):
    """多进程帧处理函数"""
    global _g_img_array
    
    frame_idx, img_start_y, img_height, img_width, self_height, self_width, frame_positions, is_transparent, bg_color = args
    img_end_y = min(img_height, img_start_y + self_height)
    frame_start_y = 0
    frame_end_y = img_end_y - img_start_y
    
    # 空帧情况下直接返回
    if img_start_y >= img_end_y or frame_start_y >= frame_end_y or frame_end_y > self_height or _g_img_array is None:
        # 创建空帧
        if is_transparent:
            frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
        else:
            # 使用背景色
            frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        return frame_idx, frame
    
    # 创建帧
    if is_transparent:
        frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
    else:
        # 使用背景色
        frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
    
    # 提取图像相关区域
    img_h_slice = slice(img_start_y, img_end_y)
    img_w_slice = slice(0, min(self_width, img_width))
    frame_h_slice = slice(frame_start_y, frame_end_y)
    frame_w_slice = slice(0, min(self_width, img_width))
    
    try:
        source_section = _g_img_array[img_h_slice, img_w_slice]
        target_area = frame[frame_h_slice, frame_w_slice]
        
        if is_transparent:
            # 透明背景处理
            if target_area.shape[:2] == source_section.shape[:2]:
                np.copyto(target_area, source_section)
            else:
                copy_width = min(target_area.shape[1], source_section.shape[1])
                target_area[:target_area.shape[0], :copy_width] = source_section[:target_area.shape[0], :copy_width]
        else:
            # 不透明背景处理
            if target_area.shape[:2] == source_section.shape[:2]:
                alpha = source_section[:, :, 3:4].astype(np.float32) / 255.0
                blended = source_section[:, :, :3] * alpha + target_area * (1.0 - alpha)
                np.copyto(target_area, blended.astype(np.uint8))
            else:
                copy_width = min(target_area.shape[1], source_section.shape[1])
                source_crop = source_section[:target_area.shape[0], :copy_width]
                target_crop = target_area[:target_area.shape[0], :copy_width]
                alpha = source_crop[:, :, 3:4].astype(np.float32) / 255.0
                blended = source_crop[:, :, :3] * alpha + target_crop * (1.0 - alpha)
                target_area[:target_area.shape[0], :copy_width] = blended.astype(np.uint8)
    except Exception as e:
        logger.error(f"处理帧 {frame_idx} 时出错: {e}")
    
    return frame_idx, frame

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
        scroll_speed: int = 5  # 每帧滚动的像素数（由service层基于行高和每秒滚动行数计算而来）
    ):
        """
        初始化视频渲染器
        
        Args:
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率
            scroll_speed: 每帧滚动的像素数（由service层基于行高和每秒滚动行数计算而来）
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.scroll_speed = scroll_speed
    
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
            # 移除可能不兼容的内存参数
            # "-max_muxing_queue_size", "1024", # 减少缓冲区大小
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

    def create_scrolling_video(
        self,
        image: Image.Image,
        output_path: str,
        text_actual_height: int,
        transparency_required: bool,
        preferred_codec: str, # 仍然接收 h264_nvenc 作为首选
        audio_path: Optional[str] = None,
        bg_color: Optional[Tuple[int, int, int, int]] = None
    ) -> str:
        """创建滚动视频，使用ffmpeg管道优化，并应用推荐的编码参数"""
        
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]
        if img_array.shape[2] != 4:
             logger.warning("输入图像非RGBA，尝试转换")
             image = image.convert("RGBA")
             img_array = np.array(image)
             if img_array.shape[2] != 4: raise ValueError("无法转换图像为RGBA")
             
        # 确保滚动距离至少是文本高度，这样能保证文本从底部完全滚动到顶部
        # 注意：text_actual_height是实际文本内容高度，而不是图像高度
        scroll_distance = max(text_actual_height, img_height - self.height)
        
        # 根据滚动速度计算所需的帧数
        scroll_frames = int(scroll_distance / self.scroll_speed) if self.scroll_speed > 0 else 0
        
        # 确保短文本也有合理的滚动时间
        min_scroll_frames = self.fps * 8  # 至少8秒的纯滚动时间（不包括开头和结尾的静止帧）
        if scroll_frames < min_scroll_frames and scroll_frames > 0:
            # 计算需要的最小滚动速度
            adjusted_speed = scroll_distance / min_scroll_frames
            # 如果调整后的速度比当前速度慢，则使用调整后的速度
            if adjusted_speed < self.scroll_speed:
                logger.info(f"文本较短，减慢滚动速度: {self.scroll_speed:.2f} → {adjusted_speed:.2f} 像素/帧")
                self.scroll_speed = adjusted_speed
                scroll_frames = min_scroll_frames
        
        # 视频开头和结尾各添加足够的静止画面
        padding_frames_start = int(self.fps * 2.0)  # 增加开头的静止时间到2秒
        padding_frames_end = int(self.fps * 2.0)    # 增加结尾的静止时间到2秒
        total_frames = padding_frames_start + scroll_frames + padding_frames_end
        duration = total_frames / self.fps
        logger.info(f"文本高:{text_actual_height}, 图像高:{img_height}, 视频高:{self.height}")
        logger.info(f"滚动距离:{scroll_distance}, 滚动帧:{scroll_frames}, 总帧:{total_frames}, 时长:{duration:.2f}s")
        logger.info(f"输出:{output_path}, 透明:{transparency_required}, 首选编码器:{preferred_codec}")
        # --- 结束滚动参数计算 --- 
        
        ffmpeg_pix_fmt = ""
        video_codec_and_output_params = [] 
        cpu_fallback_codec_and_output_params = []
        final_bg_color_rgb = None
        background_frame_rgb = None
        
        if transparency_required:
            ffmpeg_pix_fmt = "rgba"
            output_path = os.path.splitext(output_path)[0] + ".mov"
            # 透明视频保持 ProRes 参数不变
            video_codec_and_output_params = [
                "-c:v", "prores_ks", "-profile:v", "4", 
                "-pix_fmt", "yuva444p10le", "-alpha_bits", "16", "-vendor", "ap10"
            ]
            logger.info(f"设置ffmpeg(透明): 输入={ffmpeg_pix_fmt}, 输出={output_path}, 参数={' '.join(video_codec_and_output_params)}")
        else:
            ffmpeg_pix_fmt = "rgb24"
            output_path = os.path.splitext(output_path)[0] + ".mp4"
            # GPU 参数: h264_nvenc, preset p3, VBR CQ 21, pix_fmt yuv420p, faststart
            video_codec_and_output_params = [
                "-c:v", preferred_codec, # h264_nvenc
                "-preset", "p3",         # Medium preset
                "-rc:v", "vbr",          # Variable Bitrate Rate Control
                "-cq:v", "21",           # Constant Quality level (good quality)
                "-b:v", "0",              # Let CQ control bitrate
                "-pix_fmt", "yuv420p",   # << 添加此行以提高兼容性
                "-movflags", "+faststart"
                # 移除不兼容的GPU限制参数
            ]
            # CPU 回退参数: libx264, preset medium(默认), CRF 21, pix_fmt yuv420p, faststart
            cpu_fallback_codec_and_output_params = [
                "-c:v", "libx264",
                "-crf", "21",            # Constant Rate Factor (good quality)
                "-preset", "medium",      # Default preset (good balance)
                "-pix_fmt", "yuv420p",   # Required by libx264 for mp4
                "-movflags", "+faststart",
                # 添加通用CPU资源限制
                "-threads", "8"          # 限制使用最多8个CPU线程
                # 移除不兼容的CPU参数
            ]
            logger.info(f"设置ffmpeg(不透明): 输入={ffmpeg_pix_fmt}, 输出={output_path}")
            logger.info(f"  首选GPU参数: {' '.join(video_codec_and_output_params)}")
            logger.info(f"  回退CPU参数: {' '.join(cpu_fallback_codec_and_output_params)}")
            # 准备背景色 (RGB)
            final_bg_color_rgb = (0, 0, 0)
            if bg_color and len(bg_color) >= 3:
                 final_bg_color_rgb = bg_color[:3]
            background_frame_rgb = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(final_bg_color_rgb, dtype=np.uint8)
        
        # --- 重构 run_ffmpeg_with_pipe 使用线程 --- 
        def run_ffmpeg_with_pipe(current_codec_params: List[str], is_gpu_attempt: bool) -> bool:
            ffmpeg_cmd = self._get_ffmpeg_command(output_path, ffmpeg_pix_fmt, current_codec_params, audio_path)
            logger.info(f"执行ffmpeg命令: {' '.join(ffmpeg_cmd)}")
            process = None
            stdout_q = queue.Queue()
            stderr_q = queue.Queue()
            stdout_thread = None
            stderr_thread = None
            
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd, stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                    # bufsize=0 might help with pipe buffering, but default is usually fine
                )
                
                # --- 启动 stdout 和 stderr 读取线程 ---
                stdout_thread = threading.Thread(target=self._reader_thread, args=(process.stdout, stdout_q), daemon=True)
                stderr_thread = threading.Thread(target=self._reader_thread, args=(process.stderr, stderr_q), daemon=True)
                stdout_thread.start()
                stderr_thread.start()
                # --- 结束线程启动 ---
                
                # --- 逐帧生成并通过管道写入 --- 
                codec_name = "unknown"
                try: codec_name = current_codec_params[current_codec_params.index("-c:v") + 1]
                except (ValueError, IndexError): pass
                
                # 预计算帧位置信息
                logger.info("预计算帧位置信息以提高性能...")
                frame_positions = []
                for frame_idx in range(total_frames):
                    if frame_idx < padding_frames_start: 
                        frame_positions.append(0)
                    elif frame_idx < padding_frames_start + scroll_frames:
                        scroll_progress = frame_idx - padding_frames_start
                        current_position = scroll_progress * self.scroll_speed
                        current_position = min(current_position, scroll_distance)
                        frame_positions.append(int(current_position))
                    else: 
                        frame_positions.append(int(scroll_distance))
                
                # 预先创建背景帧模板
                if transparency_required:
                    bg_template = np.zeros((self.height, self.width, 4), dtype=np.uint8)
                else:
                    bg_template = background_frame_rgb.copy()
                    
                # 资源管理：控制批处理大小以限制内存使用
                # 根据图像尺寸调整批处理大小，确保不会占用过多内存
                # 假设单帧RGBA图像内存 = 宽*高*4字节
                frame_memory_mb = (self.width * self.height * 4) / (1024*1024)
                # 设置最大内存使用量（提高到20GB以支持超大批处理）
                max_batch_memory_mb = 20 * 1024  # 20GB
                # 计算适合的批处理大小
                adaptive_batch_size = max(1, min(120, int(max_batch_memory_mb / frame_memory_mb)))
                logger.info(f"单帧内存估算: {frame_memory_mb:.2f}MB, 自适应批处理大小: {adaptive_batch_size}")
                
                # 使用自适应批处理大小
                batch_size = adaptive_batch_size
                num_batches = (total_frames + batch_size - 1) // batch_size
                
                # 决定使用多少个进程进行渲染
                try:
                    cpu_count = mp.cpu_count()
                    # 使用可用CPU核心数，不限制上限
                    num_processes = max(1, cpu_count - 1)  # 留出一个核心给系统和主进程
                    logger.info(f"检测到{cpu_count}个CPU核心，将使用{num_processes}个进程进行渲染")
                except:
                    # 如果无法检测CPU数量，默认使用6个进程
                    num_processes = 6
                    logger.info(f"无法检测CPU核心数，默认使用{num_processes}个进程")
                
                # 准备背景色参数（为了多进程）
                if not transparency_required and bg_color and len(bg_color) >= 3:
                    bg_color_rgb = bg_color[:3]
                else:
                    bg_color_rgb = (0, 0, 0)  # 默认黑色
                
                # 设置全局共享图像数组，用于多进程渲染
                global _g_img_array
                _g_img_array = img_array
                
                # 使用进度条显示编码进度
                frame_iterator = tqdm.tqdm(range(num_batches), desc=f"编码 ({codec_name}) ")
                
                # 周期性垃圾回收计数器
                gc_counter = 0
                
                # 提前准备批帧参数以提高性能
                frame_batch_params = []
                for batch_idx in range(num_batches):
                    start_frame = batch_idx * batch_size
                    end_frame = min(start_frame + batch_size, total_frames)
                    batch_frames = []
                    
                    for frame_idx in range(start_frame, end_frame):
                        img_start_y = frame_positions[frame_idx]
                        # 将参数保存为元组，避免在循环中重复计算
                        frame_params = (frame_idx, img_start_y, img_height, img_width, self.height, self.width, frame_positions, transparency_required, bg_color_rgb)
                        batch_frames.append(frame_params)
                    
                    frame_batch_params.append(batch_frames)
                
                # 创建线程池，用于并行写入数据到ffmpeg（增加线程数以提高I/O吞吐量）
                executor = ThreadPoolExecutor(max_workers=4)  # 增加到4个写入线程
                write_futures = []
                
                # 超大批次处理模式
                logger.info(f"启用超大批处理模式，批次大小: {adaptive_batch_size}")
                
                # 使用多进程处理帧
                with mp.Pool(processes=num_processes) as pool:
                    for batch_idx in frame_iterator:
                        batch_frames = frame_batch_params[batch_idx]
                        
                        # 并行处理一批帧
                        processed_frames = pool.map(_process_frame, batch_frames)
                        
                        # 按顺序写入处理后的帧
                        for frame_idx, frame in sorted(processed_frames):
                            if frame is not None:
                                try:
                                    # 使用线程池异步写入数据
                                    future = executor.submit(lambda d: process.stdin.write(d), frame.tobytes())
                                    write_futures.append(future)
                                    
                                    # 限制最大并行写入数量，避免队列过长（增加并行写入数）
                                    if len(write_futures) > 20:  # 从10增加到20
                                        # 等待最早的一个写入操作完成
                                        write_futures[0].result()
                                        write_futures = write_futures[1:]
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
                            
                                # 清理帧数据，及时释放内存
                                del frame
                        
                        # 周期性垃圾回收，每处理5个批次执行一次
                        gc_counter += 1
                        if gc_counter >= 5:  # 由于批处理增大，减少垃圾回收频率
                            gc_counter = 0
                            collected = gc.collect()
                            logger.debug(f"执行垃圾回收，释放对象数: {collected}")
                            
                            # 在每次垃圾回收后强制释放内存（仅适用于Linux）
                            if HAS_RESOURCE_MODULE and os.name == 'posix':
                                try:
                                    # 在Linux系统上尝试释放未使用内存回操作系统
                                    os.system('sync')  # 刷新文件系统缓冲区
                                    with open('/proc/sys/vm/drop_caches', 'w') as f:
                                        f.write('1')
                                except:
                                    pass
                
                # 等待所有写入操作完成
                for future in write_futures:
                    future.result()
                    
                # 关闭线程池
                executor.shutdown()
                
                # 清理全局图像数组
                _g_img_array = None
                
                logger.info("所有帧已写入管道，关闭stdin...")
                process.stdin.close()
                
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