import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PIL_VERSION
from typing import Dict, Tuple, List, Optional, Union, Callable
import textwrap
import platform
import logging
import subprocess
from subprocess import DEVNULL  # 添加导入DEVNULL
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
import types  # 添加types模块导入

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
        
        # 优化线程数量，利用更多CPU核心加速渲染，但留出一些核心给FFmpeg
        self.num_threads = min(18, max(8, (os.cpu_count() or 8) - 2))
        logger.info(f"初始化视频渲染器: 宽={self.width}, 高={self.height}, FPS={fps}, 缩放={scale_factor}, "
                   f"跳帧={self.actual_frame_skip}, 线程数={self.num_threads}, 透明={transparent}")
        
        # 利用大内存环境，大幅增加帧队列容量
        self.frame_queue_size = 2000 if not transparent else 1000
        self.frame_queue = queue.Queue(maxsize=self.frame_queue_size)
        self.thread_pool = None
        self.stop_threads = False
        self.ffmpeg_process = None
        # 增加预填充数量，提高启动效率
        self.pre_fill_count = max(300, self.num_threads * 10)
        # 流控制标志
        self.producer_throttle = threading.Event()
        self.producer_throttle.set()  # 初始状态为不限制
        
        # 高内存模式标志与预生成帧缓存
        self.high_memory_mode = True
        # 添加帧预缓存，直接在内存中保存生成好的帧
        self.frame_cache = {}
        self.use_frame_cache = True if width * height < 2000000 else False  # 对于小分辨率视频启用缓存
        
        logger.info(f"已启用高性能渲染模式: 队列容量={self.frame_queue_size}, 预填充={self.pre_fill_count}帧, 线程数={self.num_threads}")
        logger.info(f"帧缓存: {'启用' if self.use_frame_cache else '禁用'}")
        
    def _generate_frames_worker(self, frame_generator, task_queue):
        """工作线程函数，用于生成帧"""
        # 每个线程处理的帧批次大小
        batch_size = 5
        frames_processed = 0
        start_time = time.time()
        
        while not self.stop_threads:
            try:
                # 高内存模式下，降低检查流控制频率
                if frames_processed % 20 == 0:
                    self.producer_throttle.wait(timeout=0.02)
                
                # 只有当队列几乎满时才暂停
                if self.frame_queue.qsize() > self.frame_queue_size * 0.98:
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
                    task = task_queue.get(block=True, timeout=0.1)
                    if task is None:  # 终止信号
                        # 放回终止信号给其他线程
                        task_queue.put(None)
                        return
                    frames_to_process.append(task)
                
                # 处理所有获取到的帧
                for frame_index in frames_to_process:
                    try:
                        frame_data = None
                        
                        # 如果启用了帧缓存，先检查缓存中是否存在
                        if self.use_frame_cache and frame_index in self.frame_cache:
                            frame_data = self.frame_cache[frame_index]
                        else:
                            # 生成帧
                            frame_data = frame_generator(frame_index)
                            
                            # 如果启用缓存且内存使用率允许，则缓存帧
                            if self.use_frame_cache and frame_index % 10 == 0:  # 只缓存10%的帧节省内存
                                self.frame_cache[frame_index] = frame_data
                            
                        if frame_data:
                            # 快速放入队列
                            try:
                                self.frame_queue.put((frame_index, frame_data), block=False)
                                frames_processed += 1
                            except queue.Full:
                                # 如果队列满了，直接放回任务队列，不重试
                                if not self.stop_threads:
                                    task_queue.put(frame_index)
                            
                            # 定期报告处理速度
                            if frames_processed % 1000 == 0:
                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    fps = frames_processed / elapsed
                                    # 减少日志输出频率
                                    logger.debug(f"线程处理速度: {fps:.2f} 帧/秒, 已处理 {frames_processed} 帧")
                                    
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
                time.sleep(0.001)  # 非常短的等待，提高响应性
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
        start_time = time.time()
        bytes_written = 0
        
        # 流控制检查
        last_throttle_check = time.time()
        throttle_check_interval = 1.0  # 降低检查频率到每秒1次
        
        # 批量写入设置 - 针对高速模式增大批量
        write_batch_size = 10  # 一次写入更多帧
        frame_batch = []
        
        # 启用写入进程高优先级
        try:
            if platform.system() == 'Windows':
                import win32api, win32process, win32con
                win32process.SetPriorityClass(win32api.GetCurrentProcess(), win32process.HIGH_PRIORITY_CLASS)
            elif platform.system() in ('Darwin', 'Linux'):
                os.nice(-10)  # 提高优先级（仅适用于root权限）
        except Exception as e:
            logger.debug(f"设置写入线程优先级失败: {str(e)}")
        
        # 生成进度条
        total_frames = self.total_frames
        pbar = None
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, unit='帧', desc='渲染进度')
        except:
            pass  # 如果tqdm不可用，则跳过
        
        while not self.stop_threads:
            try:
                # 高内存模式下降低流控制频率
                now = time.time()
                if now - last_throttle_check > throttle_check_interval:
                    queue_size = self.frame_queue.qsize()
                    
                    # 修改流控制逻辑，允许更多帧进入队列
                    if queue_size > self.frame_queue_size * 0.9 and self.producer_throttle.is_set():
                        self.producer_throttle.clear()
                    elif queue_size < self.frame_queue_size * 0.7 and not self.producer_throttle.is_set():
                        self.producer_throttle.set()
                    
                    # 简化处理速度日志，仅在出现问题时警告
                    if now - last_fps_check > 10.0 and frame_processed_count > 0:
                        processing_fps = frame_processed_count / (now - last_fps_check)
                        total_fps = frame_index_expected / (now - start_time) if now > start_time else 0
                        
                        # 如果写入速度过慢，输出警告
                        if processing_fps < 10.0:
                            mb_per_sec = bytes_written / 1024 / 1024 / (now - last_fps_check)
                            logger.warning(f"写入处理速度较慢: {processing_fps:.2f}帧/秒 ({mb_per_sec:.2f} MB/s)，队列: {queue_size}/{self.frame_queue_size}")
                        else:
                            # 正常速度只在调试日志中记录
                            logger.debug(f"写入处理速度: {processing_fps:.2f}帧/秒, 平均: {total_fps:.2f}帧/秒, 队列: {queue_size}/{self.frame_queue_size}")
                        
                        frame_processed_count = 0
                        bytes_written = 0
                        last_fps_check = now
                        
                    last_throttle_check = now
                
                # 一次尝试获取多个帧
                frames_fetched = 0
                for _ in range(write_batch_size * 2):  # 尝试获取更多帧
                    try:
                        # 非阻塞获取帧
                        frame_index, frame_data = self.frame_queue.get(block=False)
                        
                        # 将帧存入缓冲区
                        frame_buffer[frame_index] = frame_data
                        self.frame_queue.task_done()
                        frames_fetched += 1
                    except queue.Empty:
                        break
                
                # 如果非阻塞方式获取不到帧，改用短超时阻塞获取
                if frames_fetched == 0:
                    try:
                        frame_index, frame_data = self.frame_queue.get(block=True, timeout=0.05)
                        frame_buffer[frame_index] = frame_data
                        self.frame_queue.task_done()
                    except queue.Empty:
                        # 如果队列确实为空，则短暂等待并继续
                        time.sleep(0.001)
                        continue
                
                # 处理已缓冲的帧，按顺序写入
                while frame_index_expected in frame_buffer:
                    # 收集连续帧进行批量写入
                    frame_batch = []
                    batch_indices = []
                    
                    # 增大批量写入大小，但确保帧序列连续性
                    for i in range(write_batch_size * 2):  # 尝试获取更多连续帧
                        idx = frame_index_expected + i
                        if idx in frame_buffer:
                            frame_batch.append(frame_buffer[idx])
                            batch_indices.append(idx)
                        else:
                            break
                    
                    if not frame_batch:
                        break  # 没有连续帧可处理
                    
                    # 批量写入帧数据到FFmpeg - 改进写入策略
                    if self.ffmpeg_process and self.ffmpeg_process.stdin:
                        try:
                            # 使用writelines批量写入所有帧数据，避免多次调用write
                            if hasattr(self.ffmpeg_process.stdin, 'writelines'):
                                self.ffmpeg_process.stdin.writelines(frame_batch)
                                self.ffmpeg_process.stdin.flush()
                            else:
                                # 回退至传统写入方式
                                for frame_data in frame_batch:
                                    self.ffmpeg_process.stdin.write(frame_data)
                                    bytes_written += len(frame_data)
                                self.ffmpeg_process.stdin.flush()
                            
                            # 记录处理帧数
                            batch_size = len(frame_batch)
                            frame_processed_count += batch_size
                            
                            # 更新进度
                            last_frame_in_batch = batch_indices[-1]
                            self.current_frame = last_frame_in_batch
                            
                            # 更新进度条
                            if pbar:
                                pbar.update(batch_size)
                            
                            # 从缓冲区中删除已处理的帧
                            for idx in batch_indices:
                                del frame_buffer[idx]
                                
                            # 更新期望的下一帧
                            frame_index_expected = last_frame_in_batch + 1
                            
                            # 降低进度日志频率
                            if frame_index_expected % 500 == 0:
                                progress = min(100, int(frame_index_expected / self.total_frames * 100))
                                elapsed = time.time() - start_time
                                estimated_total = elapsed / frame_index_expected * self.total_frames if frame_index_expected > 0 else 0
                                remaining = max(0, estimated_total - elapsed)
                                
                                mins_remaining = int(remaining / 60)
                                secs_remaining = int(remaining % 60)
                                logger.info(f"渲染进度: {progress}% (帧 {frame_index_expected}/{self.total_frames}, 剩余约 {mins_remaining}分{secs_remaining}秒)")
                                
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
                
                # 如果缓冲区太大，定期强制清理
                buffer_size = len(frame_buffer)
                if buffer_size > 200:
                    # 缓冲区过大时，查找未处理的最小帧
                    if buffer_size > 500:
                        # 找出缓冲区中最小的帧索引
                        logger.warning(f"帧缓冲区过大 ({buffer_size} 帧)，跳过部分丢失帧...")
                        frame_index_expected = min(frame_buffer.keys())  # 跳到缓冲区中最小的帧
                    else:
                        logger.debug(f"帧缓冲区较大 ({buffer_size} 帧)")
                
            except queue.Empty:
                # 短暂等待
                time.sleep(0.001)
                continue
            except Exception as e:
                import traceback
                logger.error(f"帧写入线程错误: {str(e)}\n{traceback.format_exc()}")
                if not self.stop_threads:
                    if self.error_callback:
                        self.error_callback(f"视频渲染错误: {str(e)}")
                    self.stop_threads = True
                break
        
        # 清理进度条
        if pbar:
            pbar.close()
    
    def _prepare_ffmpeg_command(self):
        """准备FFmpeg命令行，优先考虑兼容性"""
        
        # 强制使用软件编码以解决兼容性问题
        force_software_encoding = True  # 临时设置为True以解决当前问题
        
        # 确定视频编码器和硬件加速选项
        codec = None
        hwaccel = None
        extra_params = []
        
        # 获取文件扩展名，为临时文件设置正确的格式
        output_ext = os.path.splitext(self.output_path)[1].lower()
        if not output_ext or output_ext == '.tmp':
            output_ext = '.mp4'  # 默认使用MP4，避免.tmp格式
            
        # 检测系统类型
        system = platform.system()
        
        # 如果强制使用软件编码，跳过硬件加速检测
        if force_software_encoding:
            logger.info("强制使用软件编码 (libx264)，这提供最佳兼容性")
            codec = 'libx264'
            extra_params.extend(['-preset', 'ultrafast'])
            extra_params.extend(['-tune', 'fastdecode', '-crf', '28'])
        else:
            # 按系统类型选择最佳编码器
            if system == 'Darwin':  # macOS
                # 检查M1/M2芯片
                is_apple_silicon = platform.processor() == '' or 'arm' in platform.processor().lower()
                if is_apple_silicon:
                    codec = 'h264_videotoolbox'
                    hwaccel = 'videotoolbox'
                    extra_params.extend(['-b:v', '12M', '-tag:v', 'avc1'])
                    extra_params.extend(['-quality', 'speed'])
                    extra_params.extend(['-allow_sw', '1'])
                else:
                    codec = 'h264_videotoolbox'
                    hwaccel = 'videotoolbox'
                    extra_params.extend(['-b:v', '10M'])
                    extra_params.extend(['-quality', 'speed'])
            elif system == 'Windows':
                if self._is_nvidia_available():
                    codec = 'h264_nvenc'
                    hwaccel = 'cuda'
                    extra_params.extend(['-tune', 'fastdecode', '-preset', 'p1'])
                    extra_params.extend(['-rc', 'vbr', '-cq', '24', '-b:v', '0'])
                else:
                    codec = 'libx264'
                    extra_params.extend(['-preset', 'ultrafast'])
                    extra_params.extend(['-tune', 'fastdecode', '-crf', '28'])
            else:  # Linux及其他
                if self._is_nvidia_available():
                    codec = 'h264_nvenc'
                    hwaccel = 'cuda'
                    extra_params.extend(['-tune', 'fastdecode', '-preset', 'p1'])
                    extra_params.extend(['-rc', 'vbr', '-cq', '24', '-b:v', '0'])
                elif self._is_amd_available():
                    codec = 'h264_amf'
                    hwaccel = 'amf'
                    extra_params.extend(['-quality', 'speed'])
                else:
                    codec = 'libx264'
                    extra_params.extend(['-preset', 'ultrafast'])
                    extra_params.extend(['-tune', 'fastdecode', '-crf', '28'])
        
        # 如果没有选择编码器，使用libx264作为回退
        if not codec:
            codec = 'libx264'
            extra_params.extend(['-preset', 'ultrafast'])
            extra_params.extend(['-tune', 'fastdecode', '-crf', '28'])
            
        # 准备基本FFmpeg命令
        command = [
            'ffmpeg',
            '-nostdin',         # 禁用交互模式
            '-y',               # 自动覆盖输出文件
            '-framerate', str(self.fps),  # 设置帧率
        ]
        
        # 只有在非强制软件编码模式下才添加硬件加速选项
        if hwaccel and not force_software_encoding:
            command.extend(['-hwaccel', hwaccel])
        
        # 配置输入格式
        command.extend([
            '-f', 'rawvideo',   # 使用原始视频格式
            '-s', f'{self.width}x{self.height}',  # 视频尺寸
            '-pix_fmt', 'rgba' if self.transparent else 'rgb24',  # 像素格式
            '-i', 'pipe:',      # 从管道读取
        ])
        
        # 添加特殊优化选项，提高处理速度
        command.extend([
            # 设置线程数量 - 使用更多的线程
            '-threads', str(min(16, max(1, os.cpu_count()))),
            # 使用更大的线程队列，提高多线程效率
            '-thread_queue_size', '2048',
            # 使用更快的缩放过滤器
            '-sws_flags', 'fast_bilinear',
            # 禁用音频
            '-an',
            # 提高处理缓冲区大小
            '-bufsize', '100M',
        ])
        
        # 添加前面确定的额外参数
        if extra_params:
            command.extend(extra_params)
            
        # 配置视频编码 - 使用最低质量但最快的设置
        command.extend([
            '-c:v', codec,
            '-pix_fmt', 'yuv420p',  # 兼容性像素格式
            '-g', '300',            # 增大GOP大小，减少I帧数量
            '-bf', '0',             # 禁用B帧以加速编码
            '-flags', '+cgop',      # 闭合GOP，提高编码效率
        ])
        
        # 确保输出文件有正确的扩展名
        output_path = self.output_path
        if output_path.endswith('.tmp'):
            # 修改临时文件扩展名为正确的视频格式
            output_path = os.path.splitext(output_path)[0] + output_ext
            logger.info(f"修正临时文件扩展名: {self.output_path} -> {output_path}")
            self.output_path = output_path
            
        # 添加输出文件路径
        command.append(self.output_path)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"创建输出目录: {output_dir}")
            except Exception as e:
                logger.error(f"创建输出目录失败: {str(e)}")
        
        logger.info(f"FFmpeg极速命令: {' '.join(command)}")
        return command
        
    def _is_nvidia_available(self):
        """检查NVIDIA GPU是否可用"""
        try:
            # 尝试运行nvidia-smi命令
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
            
    def _is_amd_available(self):
        """检查AMD GPU是否可用"""
        system = platform.system()
        if system == 'Windows':
            # Windows下检查AMD GPU
            try:
                result = subprocess.run(['where', 'amf-enc'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                return result.returncode == 0
            except (subprocess.SubprocessError, FileNotFoundError):
                return False
        elif system == 'Linux':
            # Linux下检查AMD GPU
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    return 'amd' in f.read().lower()
            except:
                return False
        return False
        
    def _is_intel_available(self):
        """检查Intel GPU是否可用且可被FFmpeg使用"""
        try:
            # 先检查FFmpeg是否支持qsv
            result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
            output = result.stdout.decode('utf-8', errors='ignore')
            if 'h264_qsv' not in output:
                logger.info("FFmpeg不支持QSV硬件加速")
                return False
                
            # 然后检查系统是否有Intel GPU
            system = platform.system()
            if system == 'Windows':
                # 尝试通过wmic查询
                try:
                    result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                    return 'intel' in result.stdout.decode('utf-8', errors='ignore').lower()
                except:
                    return False
            elif system == 'Linux':
                # 在Linux上检查有无Intel集成显卡
                try:
                    result = subprocess.run(['lspci'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                    return 'intel' in result.stdout.decode('utf-8', errors='ignore').lower() and 'vga' in result.stdout.decode('utf-8', errors='ignore').lower()
                except:
                    # 如果lspci不可用，尝试其他检查
                    try:
                        with open('/proc/cpuinfo', 'r') as f:
                            cpu_info = f.read().lower()
                            return 'intel' in cpu_info and not self._is_nvidia_available() and not self._is_amd_available()
                    except:
                        return False
            elif system == 'Darwin':
                # macOS上检查Intel图形处理器
                try:
                    result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                    return 'intel' in result.stdout.decode('utf-8', errors='ignore').lower()
                except:
                    # 如果命令失败，尝试检查处理器类型
                    return 'intel' in platform.processor().lower()
        except Exception as e:
            logger.debug(f"检查Intel GPU时出错: {str(e)}")
        
        # 默认情况下返回False，确保安全回退到软件编码
        return False

    def render_frames(self, total_frames, frame_generator):
        """使用多线程渲染所有帧"""
        self.total_frames = total_frames
        self.stop_threads = False
        # 添加开始时间记录，用于计算处理速度
        self.start_time = time.time()
        
        # 高级内存管理设置
        memory_limit_gb = 25  # 25GB内存限制
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024  # 转换为字节
        self.memory_check_counter = 0  # 用于定期内存检查
        
        # 禁用Pillow内置缓存以减少内存使用
        try:
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许处理截断的图像
            Image.MAX_IMAGE_PIXELS = None  # 禁用图像大小限制
        except:
            pass
            
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
        
        logger.info(f"初始内存使用: {initial_memory:.2f} MB，允许最大内存使用: {memory_limit_gb}GB")
        
        # 准备FFmpeg命令
        command = self._prepare_ffmpeg_command()
        
        # 启动FFmpeg进程，设置更大的缓冲区
        buffer_size = 1024 * 1024 * 100  # 增大到100MB缓冲区
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
        
        # 修改stdin对象以支持writelines方法（如果不存在）
        if not hasattr(self.ffmpeg_process.stdin, 'writelines'):
            def writelines(self, lines):
                for line in lines:
                    self.write(line)
            self.ffmpeg_process.stdin.writelines = types.MethodType(writelines, self.ffmpeg_process.stdin)
            
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
            max_ahead = self.num_threads * 20
            
            # 定期检查内存使用，避免内存泄漏
            memory_check_freq = 1000  # 每1000帧检查一次
            
            while frame_index < total_frames and not self.stop_threads:
                try:
                    # 低频检查内存使用情况
                    now = time.time()
                    if now - last_memory_check > memory_check_interval:
                        current_memory = process.memory_info().rss / 1024 / 1024  # MB
                        peak_memory = max(peak_memory, current_memory)
                        memory_growth = current_memory - initial_memory
                        
                        # 输出当前内存使用情况
                        logger.info(f"内存使用: {current_memory:.2f} MB (增长: {memory_growth:.2f} MB, 峰值: {peak_memory:.2f} MB)")
                        
                        # 检查内存是否超过预设限制
                        if current_memory * 1024 * 1024 > self.memory_limit * 0.9:
                            logger.warning(f"内存使用接近限制 ({current_memory:.2f}MB/{memory_limit_gb}GB)，执行垃圾回收...")
                            gc.collect()
                            
                            # 清理帧缓存
                            if self.use_frame_cache and len(self.frame_cache) > 0:
                                logger.info(f"清理帧缓存以释放内存，当前缓存大小: {len(self.frame_cache)} 帧")
                                self.frame_cache.clear()
                            
                        last_memory_check = now
                    
                    # 检查是否需要进行周期性内存管理
                    self.memory_check_counter += 1
                    if self.memory_check_counter % memory_check_freq == 0:
                        # 轻量级内存检查，只获取内存占用而不记录
                        current_memory = process.memory_info().rss
                        if current_memory > self.memory_limit * 0.8:  # 如果内存使用超过限制的80%
                            # 释放帧缓存中的一半项目
                            if self.use_frame_cache and len(self.frame_cache) > 0:
                                keys_to_remove = list(self.frame_cache.keys())[::2]  # 移除一半的缓存
                                for key in keys_to_remove:
                                    del self.frame_cache[key]
                    
                    # 高内存模式下允许更大的生产-消费差距
                    frames_ahead = frame_index - self.current_frame
                    if frames_ahead > max_ahead:
                        # 如果生产速度过快，给写入线程更多时间
                        time.sleep(0.01)
                        continue
                    
                    # 智能任务分配：限制队列大小
                    queue_size = task_queue.qsize()
                    max_queue_tasks = self.num_threads * 10  # 高内存模式下增大任务队列容量
                    
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
            # 清理缓存
            if self.use_frame_cache:
                self.frame_cache.clear()
            
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