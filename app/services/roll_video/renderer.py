import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PIL_VERSION
from typing import Dict, Tuple, List, Optional, Union, Callable
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
import mmap
from numba import jit, prange
import re
import time
import uuid
import ctypes
import gc
import psutil

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
    """视频渲染器，负责生成滚动效果视频"""
    
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        output_path: str,
        frame_skip: int = 1,
        scale_factor: float = 1.0,
        with_audio: bool = False,
        audio_path: Optional[str] = None,
        transparent: bool = False,
        override_temp_working_dir: Optional[str] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ):
        self.width = int(width * scale_factor)
        self.height = int(height * scale_factor)
        self.original_width = width
        self.original_height = height
        self.fps = fps
        self.output_path = output_path
        self.scale_factor = scale_factor
        self.actual_frame_skip = max(1, frame_skip)  # 确保实际跳帧至少为1
        self.total_frames = 0
        self.current_frame = 0
        self.scroll_distance = 0  # 滚动的总距离
        # 增加底部填充空间，确保最后一行文字也能完全滚出屏幕
        self.bottom_padding_ratio = 0.3  # 设置为视频高度的30%
        self.transparent = transparent
        self.with_audio = with_audio
        self.audio_path = audio_path
        self.temp_dir = override_temp_working_dir
        self.error_callback = error_callback
        
        # 优化线程数量，利用更多CPU核心加速渲染
        self.num_threads = min(16, os.cpu_count() or 8)
        logger.info(f"初始化视频渲染器: 宽={self.width}, 高={self.height}, FPS={fps}, 缩放={scale_factor}, "
                   f"跳帧={self.actual_frame_skip}, 线程数={self.num_threads}, 透明={transparent}")
        
        # 利用大内存环境，大幅增加帧队列容量
        self.frame_queue_size = 500 if not transparent else 300
        self.frame_queue = queue.Queue(maxsize=self.frame_queue_size)
        self.thread_pool = None
        self.stop_threads = False
        self.ffmpeg_process = None
        # 增加预填充数量，提高启动效率
        self.pre_fill_count = max(100, self.num_threads * 5)
        # 流控制标志
        self.producer_throttle = threading.Event()
        self.producer_throttle.set()  # 初始状态为不限制
        
        # 高内存模式标志
        self.high_memory_mode = True
        logger.info(f"已启用高内存模式: 队列容量={self.frame_queue_size}, 预填充={self.pre_fill_count}帧, 线程数={self.num_threads}")
        
    def _generate_frames_worker(self, frame_generator, task_queue):
        """工作线程函数，用于生成帧"""
        # 每个线程处理的帧批次大小
        batch_size = 3
        frames_processed = 0
        start_time = time.time()
        
        while not self.stop_threads:
            try:
                # 高内存模式下，降低检查流控制频率
                if frames_processed % 10 == 0:
                    self.producer_throttle.wait(timeout=0.05)
                
                # 高内存模式下，增加队列满检测阈值
                if self.frame_queue.qsize() > self.frame_queue_size * 0.95:
                    time.sleep(0.01)  # 短暂暂停
                    continue
                
                # 批量处理多个任务，提高效率
                frames_to_process = []
                for _ in range(batch_size):
                    try:
                        task = task_queue.get(block=False)
                        if task is None:  # 终止信号
                            # 放回终止信号给其他线程
                            task_queue.put(None)
                            return
                        frames_to_process.append(task)
                    except queue.Empty:
                        break
                
                if not frames_to_process:
                    # 如果没有取到任务，阻塞式获取一个
                    task = task_queue.get(block=True, timeout=0.2)
                    if task is None:  # 终止信号
                        # 放回终止信号给其他线程
                        task_queue.put(None)
                        return
                    frames_to_process.append(task)
                
                # 处理所有获取到的帧
                for frame_index in frames_to_process:
                    try:
                        # 生成帧并放入队列
                        frame_data = frame_generator(frame_index)
                        if frame_data:
                            # 使用非阻塞方式尝试放入队列，失败后短暂重试
                            max_retries = 5
                            for retry in range(max_retries):
                                try:
                                    self.frame_queue.put((frame_index, frame_data), block=False)
                                    frames_processed += 1
                                    break
                                except queue.Full:
                                    if retry < max_retries - 1:
                                        time.sleep(0.05)  # 短暂等待后重试
                                    else:
                                        # 最后一次尝试，使用阻塞方式
                                        try:
                                            self.frame_queue.put((frame_index, frame_data), block=True, timeout=0.5)
                                            frames_processed += 1
                                        except queue.Full:
                                            error_msg = f"帧 {frame_index} 无法加入队列: 队列已满，重新排队"
                                            logger.warning(error_msg)
                                            # 放回任务队列末尾
                                            task_queue.put(frame_index)
                            
                            # 定期报告处理速度
                            if frames_processed % 100 == 0:
                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    fps = frames_processed / elapsed
                                    logger.debug(f"线程处理速度: {fps:.2f} 帧/秒, 已处理 {frames_processed} 帧")
                                    
                    except BrokenPipeError as e:
                        error_msg = f"帧 {frame_index} 生成错误: 管道已断开 - {str(e)}"
                        logger.error(error_msg)
                        if self.error_callback:
                            self.error_callback(error_msg)
                    except MemoryError as e:
                        error_msg = f"帧 {frame_index} 生成错误: 内存不足 - {str(e)}"
                        logger.error(error_msg)
                        if self.error_callback:
                            self.error_callback(error_msg)
                    except Exception as e:
                        import traceback
                        error_msg = f"帧 {frame_index} 生成错误: {str(e)}\n{traceback.format_exc()}"
                        logger.error(error_msg)
                        if self.error_callback:
                            self.error_callback(error_msg)
                    
                    # 标记任务完成
                    task_queue.task_done()
                            
            except queue.Empty:
                # 队列为空，继续等待
                time.sleep(0.01)
                continue
            except Exception as e:
                import traceback
                logger.error(f"帧生成线程错误: {str(e)}\n{traceback.format_exc()}")
                # 继续运行，不中断渲染
                continue
    
    def _frame_writer(self):
        """从队列中获取帧并写入FFmpeg进程"""
        frame_index_expected = 0
        # 添加帧缓冲区，用于存储乱序到达的帧
        frame_buffer = {}
        # 跟踪写入性能
        frame_processed_count = 0 
        last_fps_check = time.time()
        
        # 流控制检查
        last_throttle_check = time.time()
        throttle_check_interval = 0.5  # 降低检查频率
        
        # 批量写入设置
        write_batch_size = 5  # 一次写入多少帧
        frame_batch = []
        
        while not self.stop_threads:
            try:
                # 高内存模式下降低流控制频率
                now = time.time()
                if now - last_throttle_check > throttle_check_interval:
                    queue_size = self.frame_queue.qsize()
                    
                    # 队列过满，暂停生产者 (提高阈值)
                    if queue_size > self.frame_queue_size * 0.9 and self.producer_throttle.is_set():
                        logger.debug("队列接近满，暂停生产者")
                        self.producer_throttle.clear()
                    # 队列不足，恢复生产者 (降低阈值)
                    elif queue_size < self.frame_queue_size * 0.5 and not self.producer_throttle.is_set():
                        logger.debug("队列已有空间，恢复生产者")
                        self.producer_throttle.set()
                    
                    # 计算并记录处理速度
                    if now - last_fps_check > 5.0 and frame_processed_count > 0:
                        processing_fps = frame_processed_count / (now - last_fps_check)
                        logger.info(f"写入处理速度: {processing_fps:.2f} 帧/秒，队列状态: {queue_size}/{self.frame_queue_size}")
                        frame_processed_count = 0
                        last_fps_check = now
                        
                    last_throttle_check = now
                
                # 一次尝试获取多个帧
                frames_fetched = 0
                for _ in range(write_batch_size):
                    try:
                        # 非阻塞获取帧
                        frame_index, frame_data = self.frame_queue.get(block=False)
                        
                        # 将帧存入缓冲区
                        frame_buffer[frame_index] = frame_data
                        self.frame_queue.task_done()
                        frames_fetched += 1
                    except queue.Empty:
                        break
                
                # 如果没有取到帧，则阻塞获取一个
                if frames_fetched == 0:
                    # 获取队列中的下一帧，设置较短超时以保持响应性
                    frame_index, frame_data = self.frame_queue.get(block=True, timeout=0.1)
                    
                    # 将帧存入缓冲区
                    frame_buffer[frame_index] = frame_data
                    self.frame_queue.task_done()
                
                # 处理已缓冲的帧，按顺序写入
                while frame_index_expected in frame_buffer:
                    # 收集连续帧进行批量写入
                    frame_batch = []
                    batch_indices = []
                    
                    for i in range(write_batch_size):
                        idx = frame_index_expected + i
                        if idx in frame_buffer:
                            frame_batch.append(frame_buffer[idx])
                            batch_indices.append(idx)
                        else:
                            break
                    
                    if not frame_batch:
                        break  # 没有连续帧可处理
                        
                    # 批量写入帧数据到FFmpeg
                    if self.ffmpeg_process and self.ffmpeg_process.stdin:
                        try:
                            # 一次写入所有帧数据
                            for frame_data in frame_batch:
                                self.ffmpeg_process.stdin.write(frame_data)
                            # 只刷新一次
                            self.ffmpeg_process.stdin.flush()
                            
                            frame_processed_count += len(frame_batch)
                            
                            # 更新进度
                            last_frame_in_batch = batch_indices[-1]
                            self.current_frame = last_frame_in_batch
                            
                            # 从缓冲区中删除已处理的帧
                            for idx in batch_indices:
                                del frame_buffer[idx]
                                
                            frame_index_expected = last_frame_in_batch + 1
                            
                            # 更新进度日志
                            if frame_index_expected % 50 == 0:  # 降低日志频率
                                progress = min(100, int((frame_index_expected) / self.total_frames * 100))
                                logger.info(f"渲染进度: {progress}% (帧 {frame_index_expected}/{self.total_frames})")
                                
                        except (BrokenPipeError, IOError) as e:
                            if not self.stop_threads:  # 只有在非正常停止时报错
                                logger.error(f"写入FFmpeg失败: {str(e)}")
                                if self.error_callback:
                                    self.error_callback(f"视频编码失败: {str(e)}")
                                self.stop_threads = True
                                break
                    else:
                        # FFmpeg进程不可用
                        break
                
                # 如果缓冲区太大，定期强制清理，而不是只发出警告
                buffer_size = len(frame_buffer)
                if buffer_size > 200:
                    # 缓冲区过大时，查找未处理的最小帧
                    if buffer_size > 500:  # 极端情况，强制处理最小帧
                        missing_frames = [i for i in range(frame_index_expected, frame_index_expected + 100) if i not in frame_buffer]
                        logger.warning(f"帧缓冲区过大 ({buffer_size} 帧)，跳过丢失帧: {missing_frames[:10]}...")
                        frame_index_expected = min(frame_buffer.keys())  # 跳到缓冲区中最小的帧
                    else:
                        logger.warning(f"帧缓冲区较大 ({buffer_size} 帧)，内存使用增加")
                
            except queue.Empty:
                # 检查并尝试处理缓冲区中的帧
                if frame_buffer and frame_index_expected not in frame_buffer:
                    # 如果当前等待的帧不在缓冲区中，但有其他帧，尝试跳过丢失的帧
                    possible_next = sorted([idx for idx in frame_buffer.keys() if idx > frame_index_expected])
                    if possible_next:
                        # 找到缓冲区中大于当前期望的最小帧
                        missing_count = possible_next[0] - frame_index_expected
                        if missing_count < 10:  # 只处理少量缺失
                            logger.warning(f"跳过丢失的帧 {frame_index_expected} 到 {possible_next[0]-1}")
                            frame_index_expected = possible_next[0]
                
                # 如果队列超时但渲染已完成，则退出
                if self.current_frame >= self.total_frames - 1:
                    break
                # 短暂超时，继续循环检查流控制
                continue 
            except Exception as e:
                import traceback
                logger.error(f"帧写入线程错误: {str(e)}\n{traceback.format_exc()}")
                if not self.stop_threads:
                    if self.error_callback:
                        self.error_callback(f"视频渲染错误: {str(e)}")
                    self.stop_threads = True
                break
    
    def _prepare_ffmpeg_command(self):
        """准备FFmpeg命令行，针对不同系统和需求优化参数"""
        output_ext = os.path.splitext(self.output_path)[1].lower()
        
        # 根据透明度需求和输出路径调整格式
        if self.transparent and output_ext != '.mov':
            self.output_path = os.path.splitext(self.output_path)[0] + '.mov'
            output_ext = '.mov'
        elif not self.transparent and output_ext not in ['.mp4', '.webm']:
            self.output_path = os.path.splitext(self.output_path)[0] + '.mp4'
            output_ext = '.mp4'
        
        pixel_format = 'rgba' if self.transparent else 'rgb24'
        
        # 精确控制帧率和缓冲区配置，减少抖动
        input_args = [
            '-f', 'rawvideo',
            '-pixel_format', pixel_format,
            '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(self.fps),
            '-use_wallclock_as_timestamps', '1',  # 使用精确时间戳
            '-thread_queue_size', '512',  # 增大线程队列大小
            '-i', 'pipe:'
        ]
        
        # 系统检测
        system = platform.system()
        
        # 视频大小调整配置（如果使用了缩放）
        scaling_args = []
        if self.scale_factor != 1.0 and (self.width != self.original_width or self.height != self.original_height):
            scaling_args = [
                '-vf', f'scale={self.original_width}:{self.original_height}:flags=lanczos'
            ]
        
        # 针对macOS的VideoToolbox硬件加速配置
        if system == 'Darwin' and not self.transparent:
            video_codec = [
                '-c:v', 'h264_videotoolbox',
                '-profile:v', 'high',
                '-b:v', '8M',
                '-maxrate', '12M',
                '-bufsize', '20M',
                '-pix_fmt', 'yuv420p',
                '-allow_sw', '1',
                '-vsync', 'cfr',          # 使用恒定帧率
                '-r', str(self.fps),      # 指定输出帧率
                '-g', str(self.fps * 2),  # GOP大小设置为2秒
                '-bf', '2',               # 最多使用2个B帧
                '-movflags', '+faststart'
            ]
        # 针对Windows的NVIDIA硬件加速配置
        elif system == 'Windows' and not self.transparent:
            video_codec = [
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',          # 质量优先
                '-profile:v', 'high',
                '-b:v', '8M',             # 基础比特率
                '-maxrate', '12M',        # 最大比特率
                '-bufsize', '20M',        # 缓冲区大小
                '-rc', 'vbr_hq',          # 高质量可变比特率
                '-rc-lookahead', '32',    # 前瞻帧数
                '-pix_fmt', 'yuv420p',
                '-g', str(self.fps * 2),  # GOP大小设置为2秒
                '-bf', '3',               # B帧数量
                '-vsync', 'cfr',          # 恒定帧率
                '-spatial-aq', '1',       # 空间自适应量化
                '-temporal-aq', '1',      # 时间自适应量化
                '-r', str(self.fps),      # 输出帧率
                '-movflags', '+faststart'
            ]
        # 透明视频编码配置 (使用ProRes)
        elif self.transparent:
            video_codec = [
                '-c:v', 'prores_ks',
                '-profile:v', '4444',     # 使用带Alpha通道的配置文件
                '-pix_fmt', 'yuva444p10le',  # 高质量带Alpha通道的像素格式
                '-vendor', 'ap10',        # Apple标识
                '-bits_per_mb', '8000',   # 高质量设置
                '-r', str(self.fps),      # 输出帧率
                '-vsync', 'cfr'           # 恒定帧率
            ]
        # 通用软件编码配置 (libx264)
        else:
            video_codec = [
                '-c:v', 'libx264',
                '-preset', 'fast',        # 平衡编码速度和质量
                '-tune', 'animation',     # 针对动画内容优化
                '-profile:v', 'high',     # 高配置
                '-crf', '18',             # 较高质量 (0-51, 越小越好)
                '-pix_fmt', 'yuv420p',
                '-g', str(self.fps * 2),  # GOP大小
                '-bf', '2',               # B帧数量
                '-r', str(self.fps),      # 输出帧率
                '-vsync', 'cfr',          # 恒定帧率
                '-x264opts', 'no-deblock:no-cabac:no-8x8dct:partitions=p8x8,b8x8,i8x8',  # 优化处理
                '-movflags', '+faststart'  # 优化流式播放
            ]
        
        # 针对滚动内容的去抖动和平滑处理滤镜
        smooth_filter = []
        if not self.transparent:
            # 使用仅针对抖动处理的滤镜组合，不用平滑滤镜造成拖尾
            smooth_filter = [
                '-vf', f'minterpolate=mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:fps={self.fps}'
            ]
            # 如果有缩放，合并滤镜
            if scaling_args:
                vf_scale = scaling_args[1]
                smooth_filter = ['-vf', f'{vf_scale},minterpolate=mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:fps={self.fps}']
        
        # 音频参数，如果需要
        audio_args = []
        if self.with_audio and self.audio_path and os.path.exists(self.audio_path):
            audio_args = [
                '-i', self.audio_path,
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ar', '48000',       # 采样率
                '-ac', '2',           # 声道数
                '-af', 'aresample=async=1000',  # 音频重采样，处理同步
                '-shortest'           # 使用最短的输入流长度作为输出长度
            ]
        
        # 输出文件
        output_args = [
            # 增加输出日志级别以便调试
            '-loglevel', 'info',
            # 其他全局参数
            '-threads', str(os.cpu_count()),  # 使用所有可用CPU核心
            '-y',                            # 覆盖已有文件
            self.output_path
        ]
        
        # 合并所有参数组
        # 如果应用平滑滤镜，使用滤镜配置
        if smooth_filter and not scaling_args:
            command = ['ffmpeg'] + input_args + audio_args + smooth_filter + video_codec + output_args
        # 如果只需要缩放，使用缩放滤镜
        elif scaling_args and not smooth_filter:
            command = ['ffmpeg'] + input_args + audio_args + scaling_args + video_codec + output_args
        # 如果两者都不需要，不使用滤镜
        else:
            command = ['ffmpeg'] + input_args + audio_args + video_codec + output_args
        
        # 记录完整命令以便调试
        logger.info(f"准备执行FFmpeg命令: {' '.join(command)}")
        
        return command
            
    def render_frames(self, total_frames, frame_generator):
        """使用多线程渲染所有帧"""
        self.total_frames = total_frames
        self.stop_threads = False
        # 添加开始时间记录，用于计算处理速度
        self.start_time = time.time()
        
        # 高内存模式下，降低垃圾回收频率
        if self.high_memory_mode:
            # 设置垃圾回收阈值，减少自动回收频率
            old_threshold = gc.get_threshold()
            gc.set_threshold(old_threshold[0] * 5, old_threshold[1] * 5, old_threshold[2] * 5)
            logger.info(f"调整GC阈值: {old_threshold} -> {gc.get_threshold()}")
        
        # 开始时强制进行一次完整垃圾回收
        gc.collect()
        
        # 内存监控初始设置
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        last_memory_check = time.time()
        memory_check_interval = 10.0  # 降低内存检查频率，因为我们有大量内存
        
        logger.info(f"初始内存使用: {initial_memory:.2f} MB")
        
        # 准备FFmpeg命令
        command = self._prepare_ffmpeg_command()
        
        # 启动FFmpeg进程，设置更大的缓冲区
        buffer_size = 1024 * 1024 * 50  # 增大到50MB缓冲区
        try:
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=buffer_size,
                # 设置较高的进程优先级
                creationflags=subprocess.HIGH_PRIORITY_CLASS if platform.system() == 'Windows' else 0
            )
        except Exception as e:
            error_msg = f"启动FFmpeg失败: {str(e)}"
            logger.error(error_msg)
            if self.error_callback:
                self.error_callback(error_msg)
            return False
        
        # 设置FFmpeg进程的IO优先级（如果系统支持）
        try:
            if platform.system() == 'Linux':
                # 在Linux上设置IO优先级
                subprocess.run(['ionice', '-c', '1', '-n', '0', '-p', str(self.ffmpeg_process.pid)], 
                               stderr=subprocess.DEVNULL)
            elif platform.system() == 'Darwin':
                # 在macOS上设置进程优先级
                subprocess.run(['renice', '-n', '-10', '-p', str(self.ffmpeg_process.pid)], 
                               stderr=subprocess.DEVNULL)
        except Exception as e:
            # 忽略设置进程优先级的错误
            logger.debug(f"设置FFmpeg进程优先级失败: {str(e)}")
        
        # 启动错误监控线程，实时读取FFmpeg错误输出
        def ffmpeg_error_monitor():
            while self.ffmpeg_process and not self.stop_threads:
                line = self.ffmpeg_process.stderr.readline()
                if not line:
                    break
                line = line.decode('utf-8', errors='replace').strip()
                if line and ('error' in line.lower() or 'fail' in line.lower()):
                    logger.error(f"FFmpeg错误: {line}")
                    if self.error_callback and 'non-existing' not in line.lower():
                        self.error_callback(f"编码错误: {line}")
                elif line and 'warning' in line.lower():
                    logger.warning(f"FFmpeg警告: {line}")
                elif line:
                    logger.debug(f"FFmpeg输出: {line}")
                
        # 启动错误监控线程
        error_thread = threading.Thread(target=ffmpeg_error_monitor)
        error_thread.daemon = True
        error_thread.start()
        
        # 启动帧写入线程
        writer_thread = threading.Thread(target=self._frame_writer)
        writer_thread.daemon = True
        writer_thread.start()
        
        # 创建任务队列和线程池 
        # 使用更大的队列以避免阻塞
        task_queue = queue.Queue(maxsize=self.frame_queue_size * 2)
        threads = []
        
        # 启动工作线程，使用更多线程提升性能
        for _ in range(self.num_threads):
            t = threading.Thread(target=self._generate_frames_worker, args=(frame_generator, task_queue))
            t.daemon = True
            t.start()
            threads.append(t)
            
        # 预先添加预填充任务到队列
        pre_fill_count = min(self.pre_fill_count, total_frames)
        logger.info(f"预填充 {pre_fill_count} 帧...")
        for i in range(pre_fill_count):
            task_queue.put(i)
            
        logger.info(f"已启动 {self.num_threads} 个帧生成线程, 预填充 {pre_fill_count} 帧")
        
        # 持续添加帧任务
        try:
            frame_index = pre_fill_count
            last_progress_report = time.time()
            
            # 批量任务提交设置
            # 高内存模式下增加批处理大小
            batch_size = 20
            logger.info(f"使用批次大小: {batch_size} 帧/批次")
            
            # 高内存模式可以允许更多的超前生产
            max_ahead = self.num_threads * 10
            
            while frame_index < total_frames and not self.stop_threads:
                try:
                    # 低频检查内存使用情况
                    now = time.time()
                    if now - last_memory_check > memory_check_interval:
                        current_memory = process.memory_info().rss / 1024 / 1024  # MB
                        peak_memory = max(peak_memory, current_memory)
                        memory_growth = current_memory - initial_memory
                        
                        logger.info(f"内存使用: {current_memory:.2f} MB (增长: {memory_growth:.2f} MB, 峰值: {peak_memory:.2f} MB)")
                        
                        # 高内存模式下仅在内存增长极高时执行垃圾回收
                        if memory_growth > 20000:  # 增长超过20GB
                            logger.warning(f"内存使用接近阈值，执行垃圾回收...")
                            gc.collect()
                            
                        last_memory_check = now
                    
                    # 高内存模式下允许更大的生产-消费差距
                    frames_ahead = frame_index - self.current_frame
                    if frames_ahead > max_ahead:
                        # 如果生产速度过快，给写入线程更多时间
                        time.sleep(0.05)
                        continue
                    
                    # 智能任务分配：限制队列大小
                    queue_size = task_queue.qsize()
                    max_queue_tasks = self.num_threads * 5  # 高内存模式下增大任务队列容量
                    
                    if queue_size < max_queue_tasks:
                        # 批量添加任务，一次性加入更多帧
                        current_batch_size = min(batch_size, total_frames - frame_index)
                        for batch_offset in range(current_batch_size):
                            task_queue.put(frame_index)
                            frame_index += 1
                            
                            # 降低进度报告频率
                            now = time.time()
                            if now - last_progress_report > 10.0:  # 每10秒报告一次进度
                                progress = min(100, int(self.current_frame / self.total_frames * 100))
                                frames_per_second = 0
                                if self.current_frame > 0 and now - self.start_time > 0:
                                    frames_per_second = self.current_frame / (now - self.start_time)
                                    
                                logger.info(f"渲染进度: {progress}% (帧 {self.current_frame+1}/{self.total_frames}, 速度: {frames_per_second:.2f}帧/秒)")
                                logger.info(f"任务分配: {frame_index}/{total_frames} ({int(frame_index/total_frames*100)}%)")
                                last_progress_report = now
                    else:
                        # 队列已满，短暂等待
                        time.sleep(0.01)
                except Exception as e:
                    logger.error(f"添加任务时出错: {str(e)}")
                    time.sleep(0.05)
                    
            # 等待所有任务完成
            logger.info("所有帧已分派，等待任务队列处理完成...")
            task_queue.join()
            
            # 发送终止信号给所有线程
            for _ in threads:
                task_queue.put(None)
                
            # 等待所有线程终止
            for t in threads:
                t.join(timeout=1.0)
                
            # 等待写入线程完成
            logger.info("等待写入线程完成最后的帧处理...")
            writer_thread.join(timeout=10.0)
            
            # 等待错误监控线程完成
            if error_thread.is_alive():
                error_thread.join(timeout=2.0)
                
            # 最终内存使用报告
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"最终内存使用: {final_memory:.2f} MB (增长: {final_memory-initial_memory:.2f} MB, 峰值: {peak_memory:.2f} MB)")
            
            # 恢复垃圾回收设置
            if self.high_memory_mode:
                gc.set_threshold(*old_threshold)
                
            logger.info("所有帧已渲染完成，等待FFmpeg完成编码...")
            
        except KeyboardInterrupt:
            logger.info("接收到中断信号，正在终止渲染...")
            self.stop_threads = True
            # 等待线程清理
            for t in threads:
                t.join(timeout=0.5)
        except Exception as e:
            logger.error(f"渲染过程出错: {str(e)}")
            if self.error_callback:
                self.error_callback(f"渲染失败: {str(e)}")
            self.stop_threads = True
        finally:
            # 关闭FFmpeg进程
            if self.ffmpeg_process:
                try:
                    if self.ffmpeg_process.stdin:
                        self.ffmpeg_process.stdin.close()
                    
                    # 获取FFmpeg的最终输出和错误
                    stdout, stderr = "", ""
                    try:
                        stdout, stderr = self.ffmpeg_process.communicate(timeout=10)
                        stdout = stdout.decode('utf-8', errors='replace')
                        stderr = stderr.decode('utf-8', errors='replace')
                    except subprocess.TimeoutExpired:
                        logger.warning("FFmpeg进程超时，强制终止")
                        self.ffmpeg_process.kill()
                        stdout, stderr = self.ffmpeg_process.communicate()
                    
                    # 检查是否有严重错误
                    if stderr and ('error' in stderr.lower() or 'fail' in stderr.lower()):
                        error_lines = [line for line in stderr.split('\n') 
                                      if 'error' in line.lower() or 'fail' in line.lower()]
                        if error_lines:
                            error_msg = f"FFmpeg编码错误: {error_lines[0]}"
                            logger.error(error_msg)
                            if self.error_callback:
                                self.error_callback(error_msg)
                    
                except Exception as e:
                    logger.error(f"关闭FFmpeg进程时出错: {str(e)}")
            
            # 最终垃圾回收
            gc.collect()
            logger.info(f"视频渲染完成: {self.output_path}")
            
            # 检查输出文件是否存在
            if not os.path.exists(self.output_path) or os.path.getsize(self.output_path) == 0:
                error_msg = f"视频输出失败: 无法找到或文件大小为0: {self.output_path}"
                logger.error(error_msg)
                if self.error_callback:
                    self.error_callback(error_msg)
                return False
                
            return True

    def calculate_total_frames(self, text_height, scroll_speed):
        """计算视频需要的总帧数"""
        # 需要滚动的总距离 = 文本高度 + 视频高度 + 底部填充
        bottom_padding = int(self.height * self.bottom_padding_ratio)
        self.scroll_distance = text_height + self.height + bottom_padding
        
        # 计算所需的总帧数，使用实际滚动速度
        total_frames = self.scroll_distance // scroll_speed
        if self.scroll_distance % scroll_speed != 0:
            total_frames += 1
            
        # 考虑跳帧
        if self.actual_frame_skip > 1:
            # 确保总帧数是实际跳帧的倍数，保证完整的视频
            if total_frames % self.actual_frame_skip != 0:
                total_frames += self.actual_frame_skip - (total_frames % self.actual_frame_skip)
                
        logger.info(f"文本高度: {text_height}px, 滚动距离: {self.scroll_distance}px, "
                   f"滚动速度: {scroll_speed}px/帧, 计算总帧数: {total_frames}")
        return total_frames 