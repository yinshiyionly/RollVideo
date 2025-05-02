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
import multiprocessing

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
            line_spacing: 行间距，将乘以字体大小计算实际间距
            char_spacing: 字符间距
        """
        self.width = width
        # 尝试加载指定字体，失败时回退到Pillow默认字体
        try:
            if isinstance(font_path, str) and os.path.isfile(font_path):
                self.font = ImageFont.truetype(font_path, font_size)
            else:
                # 如果font_path不是文件路径，尝试直接用Pillow加载
                self.font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            logger.warning(f"加载字体失败 ('{font_path}'), 使用Pillow默认字体: {e}", exc_info=True)
            # 回退到Pillow内置字体
            self.font = ImageFont.load_default()

        self.font_size = font_size
        
        # 将传入的参数转换为列表，以便统一处理
        font_color_list = list(font_color) if not isinstance(font_color, list) else font_color.copy()
        bg_color_list = list(bg_color) if not isinstance(bg_color, list) else bg_color.copy()
        
        # 确保字体颜色是RGB格式，并转换为整数
        for i in range(len(font_color_list)):
            font_color_list[i] = int(font_color_list[i])
        
        # 添加alpha通道给字体颜色
        if len(font_color_list) == 3:
            font_color_list.append(255)  # 添加不透明的alpha通道
        elif len(font_color_list) != 4:
            font_color_list = [0, 0, 0, 255]  # 默认黑色不透明
        
        # 确保背景颜色是RGBA格式，并转换为整数
        for i in range(len(bg_color_list)):
            # 处理Alpha通道可能的浮点数(0.0-1.0)转整数(0-255)
            if i == 3 and 0 <= bg_color_list[i] <= 1 and isinstance(bg_color_list[i], float):
                bg_color_list[i] = int(bg_color_list[i] * 255)
            else:
                bg_color_list[i] = int(bg_color_list[i])
        
        # 补充RGBA
        if len(bg_color_list) == 3:
            bg_color_list.append(255)  # 添加不透明的alpha通道
        elif len(bg_color_list) != 4:
            bg_color_list = [255, 255, 255, 255]  # 默认白色不透明
        
        # 转换为元组，适配PIL需要
        self.font_color = tuple(font_color_list)
        self.bg_color = tuple(bg_color_list)
        
        # 行间距计算，支持比例行间距
        if line_spacing <= 0:
            self.line_spacing = 0
        elif line_spacing < 1:
            # 如果小于1，认为是字体大小的倍数
            self.line_spacing = int(line_spacing * font_size)
        else:
            # 否则直接使用像素值
            self.line_spacing = int(line_spacing)
            
        self.char_spacing = char_spacing
        
        # 记录原始参数，便于调试
        logger.info(f"文本渲染器初始化: 字体大小={font_size}, 行间距={self.line_spacing}, 字符间距={char_spacing}")
    
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
        
        # 修改：不再在图像顶部添加空白区域，只在底部添加足够的空白
        # 这样视频一开始就会显示文本，而不是空白
        total_height = text_actual_height + screen_height
        
        # 使用指定的背景颜色（包括透明度）创建图片
        img = Image.new('RGBA', (self.width, total_height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # 文本从图像顶部开始绘制
        y_position = 0
        # 手动逐字符绘制，确保字符间距生效
        for line in lines:
            x_pos = 0
            for char in line:
                draw.text((x_pos, y_position), char, font=self.font, fill=self.font_color)
                # 测量字符宽度
                try:
                    char_width = self.font.getlength(char)
                except Exception:
                    # 回退到getbbox或mask测量
                    if hasattr(self.font, 'getbbox'):
                        bbox = self.font.getbbox(char)
                        char_width = bbox[2] - bbox[0]
                    else:
                        mask_bbox = self.font.getmask(char).getbbox()
                        char_width = (mask_bbox[2] - mask_bbox[0]) if mask_bbox else self.font_size
                x_pos += char_width + self.char_spacing
            y_position += line_height
        
        # 记录渲染尺寸和背景色
        logger.info(f"Text rendered: lines={len(lines)}, text_height={text_actual_height}px, img_size={img.size}, bg_color={self.bg_color}")
        
        return img, text_actual_height # 返回图像和文本实际高度

class VideoRenderer:
    """视频渲染器，负责生成滚动效果视频"""
    
    def __init__(self, width, height, fps=30, output_path=None, frame_skip=1, scale_factor=1.0, with_audio=False, audio_path=None, transparent=False, error_callback=None, max_threads=None):
        """初始化视频渲染器"""
        # 视频参数
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.scale = float(scale_factor)
        self.transparent = bool(transparent)
        self.output_path = output_path
        
        # 跳帧设置
        self.skip_frames = max(1, int(frame_skip))
        
        # 音频设置
        self.with_audio = with_audio
        self.audio_path = audio_path
        
        # 错误回调
        self.error_callback = error_callback
        
        # 计算线程数
        cpu_count = os.cpu_count() or 4
        default_threads = max(1, cpu_count - 2)  # 保留2个核心给系统和FFmpeg
        
        # 默认线程设置 - 根据视频类型调整
        if max_threads is None:
            if transparent:
                # 透明视频处理更复杂，使用中等线程数
                self.max_threads = min(16, default_threads)
            else:
                # 不透明视频可以使用更多线程
                self.max_threads = min(24, default_threads)
        else:
            self.max_threads = max(1, int(max_threads))
            
        # 超大内存模式 - 假设有充足内存用于渲染
        self.high_memory_mode = True
        
        # 帧队列大小 - 高内存模式下使用更大队列
        if transparent:
            # 透明视频每帧更大，使用较小队列
            self.frame_queue_size = 1000 if self.high_memory_mode else 15
        else:
            # 不透明视频使用更大队列
            self.frame_queue_size = 4000 if self.high_memory_mode else 20
            
        # 预填充帧数 - 高内存模式下预填充更多帧
        if transparent:
            # 透明视频预填充较少
            self.prefill_count = 100 if self.high_memory_mode else 5
        else:
            # 不透明视频预填充更多
            self.prefill_count = 500 if self.high_memory_mode else 5
        
        # 总帧数和当前处理的帧
        self.total_frames = 0
        self.current_frame = 0
        
        # 线程同步原语
        self._event_stop = threading.Event()
        self._event_complete = threading.Event()
        self._frame_queue = queue.Queue(maxsize=self.frame_queue_size)
        self._last_frame_processed = -1
        self._last_frame_lock = threading.Lock()
        self._error = multiprocessing.Value('i', 0)
        self.stop_threads = False  # 为兼容性保留
        
        # 流控制标志
        self.producer_throttle = threading.Event()
        self.producer_throttle.set()  # 初始状态为不限制
        
        # 日志记录渲染配置
        logger.info(f"初始化视频渲染器: 宽={width}, 高={height}, FPS={fps}, 缩放={scale_factor}, 跳帧={frame_skip}, 线程数={self.max_threads}, 透明={transparent}")
        
        if self.high_memory_mode:
            logger.info(f"已启用极速渲染模式: 队列容量={self.frame_queue_size}, 预填充={self.prefill_count}帧, 线程数={self.max_threads}")
            
        # 帧缓存配置
        self.use_frame_cache = False
        total_pixels = width * height
        
        # 只为分辨率不太高的视频启用缓存
        if not transparent and total_pixels < 2000000:  # 小于200万像素
            # 对于低分辨率视频，帧缓存可以显著提高性能
            # 但这需要足够的内存
            try:
                import psutil
                mem = psutil.virtual_memory()
                # 如果可用内存大于16GB，启用帧缓存
                if mem.available > 16 * 1024 * 1024 * 1024:
                    self.use_frame_cache = True
                    self.frame_cache = {}
                    logger.info("已启用帧缓存: 所有帧将预先生成并存储在内存中")
                else:
                    self.frame_cache = None
            except ImportError:
                logger.info("无法加载psutil库，帧缓存禁用")
                self.frame_cache = None
        else:
            logger.info("帧缓存: 禁用")
            self.frame_cache = None
        
        self.original_width = width
        self.original_height = height
        self.scale_factor = scale_factor
        self.actual_frame_skip = max(1, self.skip_frames)  # 确保实际跳帧至少为1
        self.total_frames = 0
        self.current_frame = 0
        self.scroll_distance = 0  # 滚动的总距离
        # 增加底部填充空间，确保最后一行文字也能完全滚出屏幕
        self.bottom_padding_ratio = 0.3  # 设置为视频高度的30%
        self.temp_dir = None
        self.ffmpeg_process = None
        
        logger.info(f"已启用极速渲染模式: 队列容量={self.frame_queue_size}, 预填充={self.prefill_count}帧, 线程数={self.max_threads}")
        logger.info(f"帧缓存: {'启用' if self.use_frame_cache else '禁用'}")
        
    def _generate_frames_worker(self, frame_generator, thread_id, batch_size):
        """帧生成线程工作函数"""
        # 为每个工作线程分配不同的帧区域
        thread_count = self.max_threads
        frames_per_thread = self.total_frames // thread_count
        start_frame = thread_id * frames_per_thread
        end_frame = (thread_id + 1) * frames_per_thread if thread_id < thread_count - 1 else self.total_frames
        
        # 预填充阶段：所有线程协作处理前n帧
        # 多线程同时填充，但仅第一个线程处理前几帧
        if thread_id == 0:
            prefill_end = min(self.prefill_count, self.total_frames)
            # 优先处理开头部分帧
            for frame_idx in range(min(prefill_end, end_frame)):
                if self._event_stop.is_set():
                    return
                    
                try:
                    # 生成帧
                    frame = frame_generator(frame_idx)
                    # 添加到队列
                    if frame is not None:
                        self._frame_queue.put((frame_idx, frame))
                except Exception as e:
                    logger.error(f"生成帧 {frame_idx} 时出错: {str(e)}")
                    self._error.value = 1
                    self._event_stop.set()
                    return
        
        # 所有线程需要等待预填充阶段完成
        time.sleep(0.1)  # 简单等待预填充线程先行处理
        
        # 批量处理分配给当前线程的区域
        current_frame = start_frame
        # 跳过预填充阶段已处理的帧
        if thread_id == 0 and self.prefill_count > 0:
            current_frame = max(current_frame, self.prefill_count)
            
        # 帧生成主循环
        while current_frame < end_frame and not self._event_stop.is_set():
            try:
                # 批量生成多个帧
                batch_end = min(current_frame + batch_size, end_frame)
                for frame_idx in range(current_frame, batch_end):
                    if self._event_stop.is_set():
                        break
                        
                    try:
                        # 生成帧
                        frame = frame_generator(frame_idx)
                        # 添加到队列
                        if frame is not None:
                            # 阻塞式放入，需要等待队列有空间
                            self._frame_queue.put((frame_idx, frame), timeout=5)
                    except queue.Full:
                        # 队列已满，等待空间
                        time.sleep(0.1)
                        # 重试当前帧
                        frame_idx -= 1
                        continue
                    except MemoryError:
                        # 内存不足，报告错误并退出
                        logger.error(f"生成帧 {frame_idx} 时内存不足")
                        self._error.value = 1
                        self._event_stop.set()
                        return
                    except Exception as e:
                        # 其他错误
                        logger.error(f"生成帧 {frame_idx} 时出错: {str(e)}")
                        if "deque index out of range" in str(e):
                            # 跳过这一帧继续处理
                            continue
                        self._error.value = 1
                        self._event_stop.set()
                        return
                        
                # 更新当前处理的帧索引
                current_frame = batch_end
                
            except Exception as e:
                logger.error(f"帧生成线程 {thread_id} 出错: {str(e)}")
                self._error.value = 1
                self._event_stop.set()
                return
                
        logger.debug(f"帧生成线程 {thread_id} 完成工作 ({start_frame} - {end_frame})")
        return

    def _frame_writer(self):
        """将帧写入FFmpeg进程"""
        # Add log right at the beginning of the method
        logger.info("_frame_writer thread started.")
        
        frame_buffer = []
        frames_written = 0
        buffer_size = 10  # 批量写入的大小
        skipped_frames = 0
        ffmpeg_process = None
        
        try:
            # 准备FFmpeg命令
            command = self._prepare_ffmpeg_command()
            
            # 创建FFmpeg进程
            try:
                logger.info("启动FFmpeg进程...")
                ffmpeg_process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=10 * 1024 * 1024  # 10MB缓冲区
                )
                logger.info("FFmpeg进程已启动")
            except Exception as e:
                logger.error(f"启动FFmpeg进程失败: {str(e)}")
                self._error.value = 1
                self._event_stop.set()
                return
                
            # 初始化连续错误计数
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            # 记录开始时间
            start_time = time.time()
            next_log_time = start_time + 10  # 每10秒记录一次进度
            
            # 创建进度条
            progress_bar = tqdm.tqdm(
                total=self.total_frames,
                desc="帧渲染进度",
                unit="帧",
                unit_scale=False,
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )
            
            # 上次更新的帧索引
            last_updated_frame = 0
            
            # Add a counter for loop iterations
            loop_counter = 0
            
            # 添加帧计数器确保不超过总帧数
            expected_frames_to_process = self.total_frames
            
            # 强制设置最大可处理帧数，预留一些余量
            max_frames_allowed = self.total_frames * 2 if self.total_frames > 0 else 100000
            logger.info(f"设置最大允许帧数: {max_frames_allowed}, 预期处理帧数: {expected_frames_to_process}")
            
            # Log just before entering the loop
            logger.info("Entering main frame writing loop...")
            logger.debug(f"Initial state before loop: Stop event: {self._event_stop.is_set()}, Queue empty: {self._frame_queue.empty()}")

            while (not self._event_stop.is_set() or not self._frame_queue.empty()):
                # 仅在明确超出限制时中止
                if frames_written > max_frames_allowed:
                    logger.warning(f"已达到最大允许帧数 {max_frames_allowed}，强制停止处理")
                    self._event_stop.set()
                    break
                
                # Comment out verbose loop logs, revert to DEBUG or remove if not needed
                # logger.info(f"---> Loop {loop_counter + 1}: Entered loop top.") # Changed to INFO
                if loop_counter % 1000 == 0:  # 每1000次迭代记录一次日志
                    logger.debug(f"帧处理循环: {loop_counter}, 已写入: {frames_written}/{expected_frames_to_process}")
                
                loop_counter += 1
                frame_data = None # Initialize frame_data for this iteration
                try:
                    # Comment out verbose loop logs
                    # logger.info(f"Loop {loop_counter}: Attempting to get frame from queue. Stop event: {self._event_stop.is_set()}, Queue empty: {self._frame_queue.empty()}") # Changed to INFO
                    logger.debug(f"Loop {loop_counter}: Attempting to get frame from queue. Stop event: {self._event_stop.is_set()}, Queue empty: {self._frame_queue.empty()}") # Reverted to DEBUG
                    
                    # Try getting from queue with specific exception handling
                    try:
                        # 使用较长的超时时间，避免因短超时导致渲染问题
                        frame_data = self._frame_queue.get(timeout=0.5)
                        # Comment out verbose loop logs
                        # logger.info(f"Loop {loop_counter}: Got frame data from queue.") # Changed to INFO
                        logger.debug(f"Loop {loop_counter}: Got frame data from queue.") # Reverted to DEBUG
                    except queue.Empty:
                        # Use DEBUG here as it's less critical and frequent
                        if loop_counter % 100 == 0:  # 降低日志频率
                            logger.debug(f"队列暂时为空，继续等待...")
                        
                        # 只有同时满足停止信号和队列为空时才退出循环
                        if self._event_stop.is_set() and self._frame_queue.empty():
                            # Keep this INFO log as it signifies loop exit reason
                            logger.info(f"停止信号已设置且队列为空，退出循环 (已处理 {frames_written} 帧)")
                            break # Exit loop condition met
                        continue # Continue loop to check stop event again or wait for frames
                    except Exception as e:
                        logger.error(f"Loop {loop_counter}: Unexpected error getting from queue: {str(e)}", exc_info=True)
                        self._error.value = 1
                        self._event_stop.set() # Signal other threads to stop
                        break # Exit loop on unexpected error
                        
                    # Check frame_data validity (moved after potential continue/break)
                    if frame_data is None: 
                        logger.warning(f"Loop {loop_counter}: Got None frame data, skipping.")
                        continue
                        
                    # 解包帧数据
                    frame_idx, frame = frame_data
                    
                    # 检查帧索引是否超出最大值，使用更宽松的检查
                    if max_frames_allowed > 0 and frame_idx >= max_frames_allowed * 2:
                        if frame_idx % 100 == 0:  # 降低日志频率
                            logger.warning(f"帧索引 {frame_idx} 超出最大允许值 {max_frames_allowed}，但仍继续处理")
                    
                    # 更新最后处理的帧索引
                    with self._last_frame_lock:
                        self._last_frame_processed = max(self._last_frame_processed, frame_idx)
                    
                    # 添加到帧缓冲区
                    frame_buffer.append((frame_idx, frame))
                    
                    # 当缓冲区达到阈值或是最后一批帧时进行批量写入
                    if len(frame_buffer) >= buffer_size or (self._event_stop.is_set() and frame_buffer and self._frame_queue.empty()): # Check queue empty on stop
                        # 按帧索引排序
                        frame_buffer.sort(key=lambda x: x[0])
                        
                        # 检查FFmpeg进程是否还活着
                        if ffmpeg_process.poll() is not None:
                            error_output = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                            logger.error(f"FFmpeg进程已异常终止，错误信息: {error_output}")
                            
                            # 尝试重启FFmpeg进程
                            try:
                                logger.info("尝试重启FFmpeg进程...")
                                ffmpeg_process = subprocess.Popen(
                                    command,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE,
                                    bufsize=10 * 1024 * 1024
                                )
                                logger.info("FFmpeg进程已重启")
                            except Exception as e:
                                logger.error(f"重启FFmpeg进程失败: {str(e)}")
                                self._error.value = 1
                                self._event_stop.set()
                                return
                        
                        # 批量写入帧
                        for idx, frame_data in frame_buffer:
                            try:
                                if frame_data is not None:
                                    # 处理不同类型的帧数据
                                    frame_bytes = None
                                    
                                    # 如果是字节对象，直接使用
                                    if isinstance(frame_data, bytes):
                                        frame_bytes = frame_data
                                    # 如果是PIL图像或NumPy数组，转换为字节
                                    elif hasattr(frame_data, 'tobytes'):
                                        frame_bytes = frame_data.tobytes()
                                    # 如果是NumPy数组，需要确保内存连续
                                    elif hasattr(frame_data, 'copy') and hasattr(frame_data, 'tobytes'):
                                        frame_bytes = frame_data.copy(order='C').tobytes()
                                    # 其他情况下跳过这一帧
                                    else:
                                        logger.warning(f"未知帧数据类型: {type(frame_data)}")
                                        continue
                                    
                                    # 写入FFmpeg进程
                                    if ffmpeg_process.poll() is None:  # 确保进程仍在运行
                                        ffmpeg_process.stdin.write(frame_bytes)
                                        ffmpeg_process.stdin.flush()
                                        frames_written += 1
                                        consecutive_errors = 0  # 重置错误计数
                                        
                                        # 更新进度条
                                        frames_to_update = frames_written - last_updated_frame
                                        if frames_to_update > 0:
                                            if 'progress_bar' in locals() and progress_bar:
                                                try:
                                                    progress_bar.update(frames_to_update)
                                                    logger.debug(f"Loop {loop_counter}: Updated progress bar by {frames_to_update}")
                                                except Exception as pb_e:
                                                    logger.error(f"Loop {loop_counter}: Error updating progress bar: {pb_e}")
                                            last_updated_frame = frames_written
                                            # 计算进度信息
                                            elapsed = time.time() - start_time
                                            if elapsed > 0:
                                                fps = frames_written / elapsed
                                                progress = (frames_written / self.total_frames) * 100
                                                # 更新进度条附加信息
                                                progress_bar.set_postfix(
                                                    FPS=f"{fps:.1f}帧/秒",
                                                    进度=f"{progress:.1f}%"
                                                )
                                    else:
                                        # 进程已终止，结束循环
                                        logger.error("FFmpeg进程已终止，无法继续写入帧")
                                        self._error.value = 1
                                        self._event_stop.set()
                                        break
                            except BrokenPipeError as e:
                                # 管道已断开，需要重启FFmpeg
                                logger.error(f"写入FFmpeg失败: {str(e)}")
                                consecutive_errors += 1
                                
                                if consecutive_errors < max_consecutive_errors:
                                    # 尝试重启FFmpeg进程
                                    try:
                                        logger.info("检测到管道中断，尝试重启FFmpeg进程...")
                                        # 关闭旧进程
                                        try:
                                            if ffmpeg_process and ffmpeg_process.poll() is None:
                                                ffmpeg_process.stdin.close()
                                                ffmpeg_process.kill()
                                                ffmpeg_process.wait(timeout=1)
                                        except:
                                            pass
                                            
                                        # 创建新的进程
                                        ffmpeg_process = subprocess.Popen(
                                            command,
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE,
                                            bufsize=10 * 1024 * 1024
                                        )
                                        logger.info("FFmpeg进程已重启")
                                    except Exception as e:
                                        logger.error(f"重启FFmpeg进程失败: {str(e)}")
                                        self._error.value = 1
                                        self._event_stop.set()
                                        break
                                else:
                                    # 连续错误过多，放弃
                                    logger.error(f"连续{max_consecutive_errors}次重启FFmpeg失败，放弃处理")
                                    self._error.value = 1
                                    self._event_stop.set()
                                    break
                            except Exception as e:
                                # 其他写入错误
                                logger.error(f"写入帧{idx}时出错: {str(e)}")
                                skipped_frames += 1
                        
                        # 清空帧缓冲区
                        frame_buffer = []
                        logger.debug(f"Loop {loop_counter}: Processed frame buffer.")

                    # Comment out verbose loop logs
                    # logger.info(f"Loop {loop_counter}: End of iteration. Stop event: {self._event_stop.is_set()}, Queue empty: {self._frame_queue.empty()}") # Changed to INFO
                    logger.debug(f"Loop {loop_counter}: End of iteration. Stop event: {self._event_stop.is_set()}, Queue empty: {self._frame_queue.empty()}") # Reverted to DEBUG

                except Exception as loop_e:
                    # Catch any unexpected error within the main try block of the loop
                    logger.error(f"Loop {loop_counter}: Unexpected error in main loop body: {str(loop_e)}", exc_info=True)
                    if self._error.value == 0: self._error.value = 1
                    self._event_stop.set() # Signal stop
                    break # Exit loop
            
            # Keep this summary log
            logger.info(f"帧写入主循环已结束. Final loop count: {loop_counter}. Stop event: {self._event_stop.is_set()}, Queue empty: {self._frame_queue.empty()}")

            # Try closing the main progress bar here
            # ... (closing progress_bar logic remains the same) ...

            # --- Start of Post-Processing Block --- 
            logger.info("准备进入后处理 try 块...")
            # 这个 try...except...finally 块包裹整个后处理逻辑
            try: 
                # Main progress bar is already closed, so we don't close it again here.
                
                # 创建后处理进度条并立即显示
                print("\n正在完成视频编码和文件处理...")
                logger.info("准备启动视频后处理步骤...")
                post_steps = [
                    "关闭帧输入管道",
                    "等待FFmpeg完成视频编码",
                    "执行最终资源清理"
                ]
                
                # 这个 try...except 用于捕获创建后处理进度条的错误
                try:
                    post_progress = tqdm.tqdm(
                        total=len(post_steps),
                        desc="视频后处理",
                        unit="步骤",
                        ncols=100,
                        position=0, # 确保主进度条在顶层
                        leave=True
                    )
                    logger.info("后处理步骤进度条已创建")
                except Exception as e:
                    logger.error(f"创建后处理进度条失败: {str(e)}", exc_info=True)
                    self._error.value = 1
                    # 如果进度条创建失败，无法继续后处理，直接跳到finally
                    raise # 重新抛出异常，由外层 except 捕获并进入 finally
                
                # 记录总结信息
                if frames_written > 0:
                    total_time = time.time() - start_time
                    logger.info(f"帧写入完成: 已写入 {frames_written} 帧, 耗时 {total_time:.2f} 秒, 平均速度 {frames_written/total_time:.2f} 帧/秒")
                    if skipped_frames > 0:
                        logger.warning(f"跳过了 {skipped_frames} 帧由于错误")
                
                # 1. 关闭帧输入管道
                logger.info("准备关闭FFmpeg输入管道 (步骤 1/3)")
                post_progress.set_description(f"【第1步/共3步】{post_steps[0]}")
                # 这个 try...except 用于捕获关闭管道的错误
                try:
                    if ffmpeg_process and ffmpeg_process.poll() is None:
                        logger.info("尝试关闭 ffmpeg_process.stdin ...")
                        ffmpeg_process.stdin.close()
                        logger.info("ffmpeg_process.stdin 已关闭")
                    else:
                        logger.warning("FFmpeg 进程在关闭管道前已结束或不存在")
                    post_progress.update(1)
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"关闭FFmpeg输入管道时出错: {str(e)}", exc_info=True)
                    post_progress.update(1) # 即使出错也更新进度
                
                # 2. 等待FFmpeg完成视频处理
                logger.info("准备等待 FFmpeg 完成 (步骤 2/3)")
                post_progress.set_description(f"【第2步/共3步】{post_steps[1]}")
                # 这个 try...except 用于捕获等待FFmpeg或创建其进度条的错误
                try:
                    if ffmpeg_process and ffmpeg_process.poll() is None:
                        # ... (GPU 检测, 超时设置, wait_bar 创建/更新/关闭, FFmpeg 等待循环, 超时处理等逻辑不变) ...
                        # ---- Start of Wait Logic -----
                        is_high_end_gpu = False
                        gpu_info = "未知"
                        try:
                            gpu_info = subprocess.check_output('nvidia-smi --query-gpu=name --format=csv,noheader', shell=True, text=True)
                            gpu_info = gpu_info.strip().lower()
                            is_high_end_gpu = any(x in gpu_info for x in ['a10', 'a100', 'v100', 'a30', 'a40', 'a6000', 'rtx'])
                            logger.info(f"GPU类型检测: {gpu_info} - 高端GPU: {is_high_end_gpu}")
                        except Exception as e:
                            logger.warning(f"GPU检测失败: {str(e)}")
                        timeout = 600 if is_high_end_gpu else 300
                        ffmpeg_start_wait = time.time()
                        logger.info(f"等待FFmpeg完成最终编码 (最长等待{timeout}秒)...")
                        wait_bar = None
                        try:
                            logger.info("准备创建 FFmpeg 等待进度条...")
                            wait_bar = tqdm.tqdm(
                                total=timeout,
                                desc="  └─ 等待FFmpeg编码",
                                unit="秒",
                                ncols=100,
                                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}秒 [{elapsed}<{remaining}]",
                                leave=False
                            )
                            logger.info("FFmpeg 等待进度条已创建")
                        except Exception as e:
                            logger.error(f"创建 FFmpeg 等待进度条失败: {str(e)}", exc_info=True)
                        wait_complete = False
                        progress_points = max(10, timeout // 5)
                        wait_interval = timeout / progress_points
                        for i in range(progress_points):
                            if ffmpeg_process.poll() is not None:
                                wait_complete = True
                                if wait_bar:
                                    wait_bar.n = timeout
                                    wait_bar.refresh()
                                logger.info(f"FFmpeg进程在等待{i+1}/{progress_points}段后自行结束")
                                break
                            elapsed_wait = time.time() - ffmpeg_start_wait
                            percent_done = min(100, int((i+1) / progress_points * 100))
                            post_progress.set_description(f"【第2步/共3步】{post_steps[1]} - {percent_done}% [{int(elapsed_wait)}秒]")
                            if wait_bar:
                                wait_bar.update(wait_interval)
                            time.sleep(wait_interval)
                        if wait_bar:
                            wait_bar.close()
                            logger.info("FFmpeg 等待进度条已关闭")
                        if not wait_complete:
                            logger.warning(f"FFmpeg进程超时未退出，强制终止 (已等待{int(time.time()-ffmpeg_start_wait)}秒)")
                            ffmpeg_process.terminate()
                            termination_timeout = 3
                            termination_start = time.time()
                            while ffmpeg_process.poll() is None and time.time() - termination_start < termination_timeout:
                                time.sleep(0.5)
                            if ffmpeg_process.poll() is None:
                                logger.warning("FFmpeg进程未响应终止信号，强制杀死")
                                ffmpeg_process.kill()
                            if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 1000000:
                                logger.info(f"尽管超时，但输出文件已生成，大小: {os.path.getsize(self.output_path)/1024/1024:.2f}MB")
                            else:
                                logger.error("输出文件缺失或异常小，视频渲染可能失败")
                                if self._error.value == 0: self._error.value = 1
                        else:
                            return_code = ffmpeg_process.poll()
                            if return_code != 0:
                                stderr_output = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                                logger.error(f"FFmpeg进程异常退出，返回值: {return_code}, 错误信息: {stderr_output}")
                                if self._error.value == 0: self._error.value = return_code if return_code != 0 else 1
                            else:
                                logger.info("FFmpeg进程成功完成")
                        # ---- End of Wait Logic -----
                    else:
                        logger.warning("FFmpeg 进程在开始等待前已结束或不存在")
                                
                    post_progress.update(1) # 完成步骤2
                except Exception as e:
                    logger.error(f"等待FFmpeg完成步骤时出错: {str(e)}", exc_info=True)
                    post_progress.update(1) # 即使出错也标记步骤2完成
                
                # 3. 最终清理
                logger.info("准备执行最终清理 (步骤 3/3)")
                post_progress.set_description(f"【第3步/共3步】{post_steps[2]}")
                # 这个 try...except 用于捕获GC的错误
                try:
                    import gc
                    gc.collect()
                    logger.info("最终资源清理完成")
                    post_progress.update(1)
                    time.sleep(0.2)
                except Exception as e:
                    logger.error(f"最终清理时出错: {str(e)}", exc_info=True)
                    post_progress.update(1)
                
                # 完成后处理
                post_progress.set_description("视频后处理完成！")
                
                # 显示最终结果
                if self._error.value == 0:
                    print(f"\n✅ 视频渲染成功! 输出文件: {self.output_path}")
                    if os.path.exists(self.output_path):
                        print(f"   文件大小: {os.path.getsize(self.output_path)/1024/1024:.2f}MB")
                else:
                    print(f"\n❌ 视频渲染失败! 错误码: {self._error.value}")
                    
            # 这个 except 捕获整个后处理块的任何未预料错误
            except Exception as e:
                logger.error(f"在视频后处理阶段发生意外错误: {str(e)}", exc_info=True)
                if self._error.value == 0:
                    self._error.value = 1
                    
        # 这个 finally 块确保无论如何都会执行清理和事件设置
        finally:
            logger.info("进入 _frame_writer 的 finally 块")
            # 确保关闭所有可能的进度条 (progress_bar is handled outside now)
            # try:
            #     if 'progress_bar' in locals() and progress_bar and not progress_bar.disable:
            #         progress_bar.close()
            # except Exception as e: logger.debug(f"关闭 progress_bar 出错: {e}")
            # ... (closing post_progress and wait_bar remains the same) ...
            # ... (closing ffmpeg_process remains the same) ...
            
            # 设置完成事件
            logger.info("_frame_writer 线程即将设置完成事件并退出")
            self._event_complete.set()

    def _prepare_ffmpeg_command(self):
        """准备FFmpeg命令行，针对GPU/CPU选择最佳编码策略"""
        
        # 关闭强制软件编码，改用GPU优先策略
        force_software_encoding = False
        
        # 确定视频编码器和硬件加速选项
        codec = None
        hwaccel = None
        extra_params = []
        
        # 获取文件扩展名，为临时文件设置正确的格式
        output_ext = os.path.splitext(self.output_path)[1].lower()
        if not output_ext or output_ext == '.tmp':
            # 根据透明度选择合适的容器格式
            if self.transparent:
                output_ext = '.mov'  # 透明视频使用MOV容器
            else:
                output_ext = '.mp4'  # 不透明视频使用MP4容器
            
        # 检测系统类型
        system = platform.system()
        
        # 根据透明度选择不同的编码策略
        if self.transparent:
            # 透明视频 - 使用ProRes编码器和MOV容器
            logger.info("透明视频使用ProRes 4444编码 (CPU)")
            codec = 'prores_ks'
            extra_params.extend([
                '-profile:v', '4444',      # ProRes 4444支持Alpha通道
                '-vendor', 'ap10',         # Apple标识
                '-pix_fmt', 'yuva444p10le' # 支持Alpha通道的10位像素格式
            ])
            # 确保扩展名是.mov
            if not output_ext.endswith('.mov'):
                output_ext = '.mov'
        else:
            # 不透明视频 - 尝试使用GPU加速，失败时回退到CPU，追求极致速度
            try:
                # 首先尝试使用NVIDIA GPU
                if self._is_nvidia_available():
                    logger.info("检测到NVIDIA GPU，使用硬件加速编码(h264_nvenc)")
                    codec = 'h264_nvenc'
                    
                    # 检测是否为高端GPU (如A10, A100等)
                    high_end_gpu = False
                    try:
                        gpu_info = subprocess.check_output('nvidia-smi --query-gpu=name --format=csv,noheader', shell=True, text=True)
                        high_end_gpu = any(x in gpu_info.lower() for x in ['a10', 'a100', 'v100', 'a30', 'a40', 'a6000', 'rtx'])
                        if high_end_gpu:
                            logger.info(f"检测到高端NVIDIA GPU: {gpu_info.strip()}")
                    except:
                        pass
                    
                    # 根据操作系统调整NVIDIA加速参数
                    if system == 'Linux':
                        # 在Linux上使用更简单的硬件加速参数
                        hwaccel = 'cuda'
                        
                        if high_end_gpu:
                            # 高端GPU使用优化参数
                            extra_params.extend([
                                '-init_hw_device', 'cuda=0',  # 显式指定CUDA设备
                                '-preset', 'p1',              # 最快速模式
                                '-rc', 'vbr',                 # 可变比特率模式
                                '-b:v', '8M',                 # 较高比特率
                                '-maxrate', '16M',            # 较高最大比特率
                                '-bufsize', '16M',            # 更大缓冲区
                                '-g', '90',                   # GOP大小
                                '-spatial-aq', '1',           # 空间自适应量化
                                '-temporal-aq', '1',          # 时间自适应量化
                            ])
                        else:
                            # 普通GPU使用基本参数
                            extra_params.extend([
                                '-preset', 'p1',       # 最快速模式
                                '-b:v', '5M',          # 比特率
                                '-maxrate', '10M',     # 最大比特率
                                '-g', '30',            # 关键帧间隔 (修改为 30)
                            ])
                    else:
                        # Windows或macOS上使用完整硬件加速
                        hwaccel = 'cuda'
                        # 只在非Linux系统上使用hwaccel_output_format
                        if system in ['Windows']:
                            extra_params.extend([
                                '-hwaccel_output_format', 'cuda',
                                '-preset', 'p1',          # 最快速模式
                                '-tune', 'fastdecode',    # 快速解码
                                '-qp', '30',              # 质量参数
                                '-b:v', '5M',             # 比特率
                                '-g', '30',               # 关键帧间隔 (修改为 30)
                            ])
                        else:
                            # macOS或其他系统
                            extra_params.extend([
                                '-preset', 'p1',          # 最快速模式
                                '-qp', '30',              # 质量参数
                                '-b:v', '5M',             # 比特率
                                '-g', '30',               # 关键帧间隔 (修改为 30)
                            ])
                # 然后检查AMD GPU
                elif self._is_amd_available():
                    logger.info("检测到AMD GPU，使用硬件加速编码(h264_amf)")
                    codec = 'h264_amf'
                    hwaccel = 'amf'
                    extra_params.extend([
                        '-quality', 'speed',       # 速度优先
                        '-rc', 'cqp',              # 恒定质量模式
                        '-qp', '30'                # 质量参数(较低质量但超快)
                    ])
                # 最后检查Intel GPU
                elif self._is_intel_available() and self._test_qsv_support():
                    logger.info("检测到Intel GPU，使用硬件加速编码(h264_qsv)")
                    codec = 'h264_qsv'
                    hwaccel = 'qsv'
                    extra_params.extend([
                        '-preset', 'veryfast',     # 速度优先预设
                        '-global_quality', '30'    # 质量水平(较低质量但超快)
                    ])
                else:
                    # 没有检测到支持的GPU或GPU检测失败，使用CPU编码
                    raise Exception("未检测到支持的GPU或GPU检测失败")
            except Exception as e:
                # 任何GPU相关错误都回退到CPU超快速编码
                logger.warning(f"GPU编码不可用({str(e)})，回退到CPU极速编码(libx264)")
                codec = 'libx264'
                hwaccel = None
                # 追求极致速度的CPU参数
                extra_params.extend([
                    '-preset', 'ultrafast',  # 极速编码
                    '-tune', 'fastdecode',   # 快速解码
                    '-crf', '30',            # 较低质量但更快
                    '-x264opts', 'no-deblock:no-cabac:no-scenecut', # 禁用耗时选项
                    '-level', '4.0',         # 兼容性级别
                    '-g', '30',              # 关键帧间隔 (修改为 30)
                ])
        
        # 如果没有选择编码器，使用libx264作为最终回退
        if not codec:
            logger.warning("未能选择合适的编码器，使用libx264极速配置")
            codec = 'libx264'
            extra_params.extend(['-preset', 'ultrafast', '-crf', '30', '-g', '30']) # 添加 -g 30
            
        # 准备基本FFmpeg命令
        command = [
            'ffmpeg',
            '-nostdin',         # 禁用交互模式
            '-y',               # 自动覆盖输出文件
            '-framerate', str(self.fps),  # 设置帧率
        ]
        
        # 只有在有硬件加速选项并且不是透明视频时添加硬件加速参数
        if hwaccel and not self.transparent and not force_software_encoding:
            # 添加基本硬件加速参数
            command.extend(['-hwaccel', hwaccel])
            
            # 视情况添加hwaccel_output_format
            # 在Linux上不添加hwaccel_output_format，避免兼容性问题
            if hwaccel == 'cuda' and system in ['Windows'] and '-hwaccel_output_format' in extra_params:
                # hwaccel_output_format 已经在extra_params中处理
                pass
        
        # 配置输入格式
        command.extend([
            '-f', 'rawvideo',   # 使用原始视频格式
            '-s', f'{self.width}x{self.height}',  # 视频尺寸
            '-pix_fmt', 'rgba' if self.transparent else 'rgb24',  # 像素格式
            '-i', 'pipe:',      # 从管道读取
        ])
        
        # 优化FFmpeg性能参数
        command.extend([
            # 核心处理设置
            '-threads', str(os.cpu_count()),  # 使用所有可用CPU核心
            # 减小队列大小，避免某些环境中的溢出问题
            '-thread_queue_size', '1024',  # 使用适中的队列大小提高稳定性
            # 使用极速缩放算法
            '-sws_flags', 'fast_bilinear',  # 最快的缩放算法
            # 禁用音频
            '-an',
            # 使用适中的缓冲区大小
            '-bufsize', '100M',  # 使用适中的缓冲区增加稳定性
        ])
        
        # 添加前面确定的额外参数，但过滤掉hwaccel_output_format（在Linux上）
        if extra_params:
            if system == 'Linux' and hwaccel == 'cuda':
                # 在Linux上过滤掉hwaccel_output_format参数
                filtered_params = []
                skip_next = False
                for i, param in enumerate(extra_params):
                    if skip_next:
                        skip_next = False
                        continue
                    if param == '-hwaccel_output_format':
                        skip_next = True
                        continue
                    filtered_params.append(param)
                command.extend(filtered_params)
            else:
                command.extend(extra_params)
            
        # 根据透明度选择不同的编码配置
        if self.transparent:
            # 透明视频保持ProRes设置
            pass  # 已经在前面设置了ProRes参数
        else:
            # 不透明视频配置
            if codec.startswith('h264'):
                command.extend([
                    '-c:v', codec,
                    '-pix_fmt', 'yuv420p',  # 兼容性像素格式
                ])

                # 检查并添加 -g 30 (如果尚未添加)
                g_present = any(cmd == '-g' for cmd in command + extra_params)
                if not g_present:
                   logger.info(f"为编码器 {codec} 添加 -g 30")
                   command.extend(['-g', '30'])

                # 对于Linux上的NVIDIA编码，使用简化参数集
                if codec == 'h264_nvenc' and system == 'Linux':
                    # Linux下NVENC的参数已在extra_params中设置，这里不需要额外添加比特率等
                    pass
                elif codec == 'h264_nvenc' and system != 'Linux':
                    # 非Linux系统的NVENC额外参数 (如果尚未添加)
                    b_present = any(cmd == '-b:v' for cmd in command + extra_params)
                    if not b_present:
                        command.extend([
                            '-b:v', '5M',        # 设置比特率
                            '-maxrate', '10M',   # 最大比特率
                        ])
                # 为其他硬件编码器添加通用参数 (如果尚未添加)
                elif codec != 'h264_nvenc':
                    bf_present = any(cmd == '-bf' for cmd in command + extra_params)
                    if not bf_present:
                       command.extend(['-bf', '0']) # 禁用B帧加速

            else:
                # 其他编码器(如libx264)的参数
                command.extend([
                    '-c:v', codec,
                    '-pix_fmt', 'yuv420p',
                ])

                # 检查并添加 -g 30 (如果尚未添加)
                g_present = any(cmd == '-g' for cmd in command + extra_params)
                if not g_present:
                   logger.info(f"为编码器 {codec} 添加 -g 30")
                   command.extend(['-g', '30'])
                # 检查并添加 -bf 0 (如果尚未添加)
                bf_present = any(cmd == '-bf' for cmd in command + extra_params)
                if not bf_present:
                   command.extend(['-bf', '0']) # 禁用B帧加速

            # 为非透明视频（通常是MP4）添加 +faststart
            logger.info("为非透明视频添加 -movflags +faststart")
            command.extend(['-movflags', '+faststart'])
        
        # 明确限制视频时长和帧数，防止循环
        # 添加精确的帧数限制参数，确保FFmpeg不会尝试读取超过total_frames的帧
        if self.total_frames > 0:
            # 为了确保所有帧都能处理，设置一个比实际帧数略大的限制
            safe_frame_limit = int(self.total_frames * 1.1) + 60  # 增加10%并再加60帧的安全余量
            logger.info(f"指定宽松的帧数限制: {safe_frame_limit} 帧 (实际帧数: {self.total_frames})")
            command.extend(['-frames:v', str(safe_frame_limit)])
            
            # 计算并设置宽松的视频时长参数，确保视频不会过早结束
            duration_seconds = (self.total_frames / self.fps) * 1.2  # 增加20%的时长余量
            logger.info(f"指定宽松的视频时长: {duration_seconds:.2f} 秒 (实际预期: {self.total_frames / self.fps:.2f} 秒)")
            command.extend(['-t', f"{duration_seconds:.6f}"])

        # 确保输出文件有正确的扩展名
        output_path = self.output_path
        if output_path.endswith('.tmp') or os.path.splitext(output_path)[1] != output_ext:
            # 修改临时文件扩展名为正确的视频格式
            output_path = os.path.splitext(output_path)[0] + output_ext
            logger.info(f"修正文件扩展名: {self.output_path} -> {output_path}")
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
        """检测是否有可用的NVIDIA GPU"""
        try:
            system = platform.system()
            
            # Windows平台
            if system == 'Windows':
                # 使用nvidia-smi命令检测
                try:
                    subprocess.check_output('nvidia-smi', shell=True, stderr=subprocess.DEVNULL)
                    # 检查FFmpeg是否支持NVENC
                    encoders = subprocess.check_output('ffmpeg -encoders', shell=True, text=True, stderr=subprocess.DEVNULL)
                    return 'h264_nvenc' in encoders
                except:
                    # 尝试使用WMI查询
                    output = subprocess.check_output('wmic path win32_VideoController get name', shell=True, text=True)
                    if 'NVIDIA' in output:
                        # 需要进一步确认NVENC可用性
                        try:
                            encoders = subprocess.check_output('ffmpeg -encoders', shell=True, text=True, stderr=subprocess.DEVNULL)
                            return 'h264_nvenc' in encoders
                        except:
                            return False
                    return False
                    
            # Linux平台
            elif system == 'Linux':
                # 先检查nvidia-smi
                try:
                    subprocess.check_output('nvidia-smi', shell=True, stderr=subprocess.DEVNULL)
                    # 检查FFmpeg是否支持NVENC
                    encoders = subprocess.check_output('ffmpeg -encoders', shell=True, text=True, stderr=subprocess.DEVNULL)
                    return 'h264_nvenc' in encoders
                except:
                    # 尝试使用lspci
                    try:
                        output = subprocess.check_output('lspci | grep -i nvidia', shell=True, text=True)
                        if output:
                            # 需要进一步确认NVENC可用性
                            try:
                                encoders = subprocess.check_output('ffmpeg -encoders', shell=True, text=True, stderr=subprocess.DEVNULL)
                                return 'h264_nvenc' in encoders
                            except:
                                return False
                    except:
                        pass
                    return False
                    
            # macOS平台
            elif system == 'Darwin':
                # 使用system_profiler
                output = subprocess.check_output('system_profiler SPDisplaysDataType', shell=True, text=True)
                has_nvidia = 'NVIDIA' in output
                
                # 检查FFmpeg是否支持NVENC
                if has_nvidia:
                    try:
                        encoders = subprocess.check_output('ffmpeg -encoders', shell=True, text=True, stderr=subprocess.DEVNULL)
                        return 'h264_nvenc' in encoders
                    except:
                        return False
                return False
                
            return False
        except Exception as e:
            logger.warning(f"NVIDIA GPU检测失败: {str(e)}")
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
        """检测是否有可用的Intel GPU"""
        try:
            # Windows平台
            if platform.system() == 'Windows':
                # 使用wmic查询GPU
                output = subprocess.check_output('wmic path win32_VideoController get name', shell=True, text=True)
                return 'Intel' in output
            # Linux平台
            elif platform.system() == 'Linux':
                # 使用lspci查询GPU
                output = subprocess.check_output('lspci | grep -i vga', shell=True, text=True)
                return 'Intel' in output
            # macOS平台
            elif platform.system() == 'Darwin':
                # 使用system_profiler查询GPU
                output = subprocess.check_output('system_profiler SPDisplaysDataType', shell=True, text=True)
                return 'Intel' in output
            return False
        except Exception as e:
            logger.warning(f"Intel GPU检测失败: {str(e)}")
            return False
            
    def _test_qsv_support(self):
        """测试QSV硬件加速是否真正可用"""
        try:
            # 测试FFmpeg是否支持qsv
            output = subprocess.check_output(
                'ffmpeg -encoders | grep qsv', 
                shell=True, 
                text=True,
                stderr=subprocess.DEVNULL
            )
            if 'h264_qsv' not in output:
                logger.info("FFmpeg不支持QSV硬件加速")
                return False
                
            # 进一步测试QSV可用性
            system = platform.system()
            if system == 'Windows':
                # Windows测试
                try:
                    subprocess.check_call(
                        'ffmpeg -init_hw_device qsv=hw -f lavfi -i color=c=red:s=128x128 -frames:v 1 -c:v h264_qsv -y NUL',
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    return True
                except:
                    return False
            elif system == 'Linux':
                # Linux测试
                try:
                    subprocess.check_call(
                        'ffmpeg -init_hw_device qsv=hw -f lavfi -i color=c=red:s=128x128 -frames:v 1 -c:v h264_qsv -y /dev/null',
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    return True
                except:
                    return False
            elif system == 'Darwin':
                # macOS不支持QSV
                return False
                
            return False
        except Exception as e:
            logger.warning(f"QSV支持测试失败: {str(e)}")
            return False

    def render_frames(self, total_frames, frame_generator):
        """
        渲染视频帧并保存为视频文件
        
        Args:
            total_frames: 总帧数
            frame_generator: 生成帧的函数，接受帧索引参数并返回PIL图像
            
        Returns:
            bool: 渲染是否成功
        """
        ffmpeg_process = None  # 在外部定义，以便在finally块中访问
        try:
            import psutil
            # 设置内存限制为28GB，避免内存耗尽
            memory_limit = 28 * 1024 * 1024 * 1024  # 28GB
            memory_check_counter = 0
            
            # 确保输出目录存在
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # 初始化参数
            self.total_frames = total_frames
            self._event_stop.clear()
            self._event_complete.clear()
            self._error.value = 0
            self._last_frame_processed = -1
            self.stop_threads = False  # 确保兼容性
            
            # 降低GC频率，减轻内存压力
            # 在极速模式下大幅提高GC阈值
            import gc
            old_threshold = gc.get_threshold()
            new_threshold = (7000, 100, 100)  # 显著提高阈值
            logger.info(f"极速模式：大幅降低GC频率: {old_threshold} -> {new_threshold}")
            gc.set_threshold(*new_threshold)
            
            # 执行一次完整GC，确保开始时内存干净
            logger.info("执行一次性完整垃圾回收...")
            gc.collect()
            
            # 记录初始内存使用
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            peak_memory = initial_memory
            logger.info(f"初始内存使用: {initial_memory / 1024 / 1024:.2f} MB，允许最大内存使用: {memory_limit / 1024 / 1024 / 1024:.0f}GB")
            
            # --- Start Writer Thread --- 
            writer_thread = None # Initialize
            logger.info("准备创建帧写入线程 (_frame_writer)...")
            try:
                writer_thread = threading.Thread(target=self._frame_writer)
                writer_thread.daemon = True
                logger.info("帧写入线程对象已创建. Starting it...")
                writer_thread.start()
                logger.info("帧写入线程 (_frame_writer) start() 方法已调用.")
            except Exception as start_e:
                 logger.error(f"启动帧写入线程 (_frame_writer) 时出错: {start_e}", exc_info=True)
                 self._error.value = 1
                 # If writer thread fails to start, we cannot proceed
                 self._event_stop.set() # Signal generator threads to stop
                 self._event_complete.set() # Signal main loop to stop waiting
                 # Restore GC threshold before returning
                 gc.set_threshold(*old_threshold)
                 return False
                 
            # --- Start Generator Threads --- 
            logger.info(f"预填充 {self.prefill_count} 帧...")
            batch_size = 100
            logger.info(f"使用批次大小: {batch_size} 帧/批次")
            generator_threads = []
            logger.info(f"准备启动 {self.max_threads} 个帧生成线程...")
            for i in range(self.max_threads):
                thread = threading.Thread(
                    target=self._generate_frames_worker,
                    args=(frame_generator, i, batch_size),
                    name=f"Generator-{i}" # Give threads names
                )
                thread.daemon = True
                thread.start()
                generator_threads.append(thread)
            logger.info(f"已启动 {len(generator_threads)} 个帧生成线程.")

            # --- Wait for Generators to Finish --- 
            logger.info("主线程: 等待所有生成器线程完成...")
            start_join_generators = time.time()
            for i, thread in enumerate(generator_threads):
                try:
                    logger.debug(f"Joining Generator Thread-{i}...")
                    thread.join() # Wait indefinitely for this generator to finish
                    logger.debug(f"Generator Thread-{i} finished.")
                except Exception as join_e:
                     logger.error(f"等待生成器线程 {i} 时出错: {join_e}", exc_info=True)
                     # Decide if we should stop everything if one generator fails?
                     # For now, set error and stop flag
                     if self._error.value == 0: self._error.value = 1
                     self._event_stop.set() # Signal writer and other generators
            join_generators_duration = time.time() - start_join_generators
            logger.info(f"所有生成器线程已完成. 耗时: {join_generators_duration:.2f}s.")
            
            # --- Signal Writer to Stop --- 
            logger.info("主线程: 所有生成器已完成, 设置 _event_stop 信号给写入线程.")
            self._event_stop.set() 
            self.stop_threads = True # Compatibility
            
            # --- Wait for Writer Thread to Finish --- 
            # Now wait for the writer thread to process remaining queue and finish
            if writer_thread:
                logger.info(f"主线程: 准备等待写入线程 (_frame_writer) 结束 (timeout=None - wait indefinitely)...")
                writer_thread.join() # Wait indefinitely for writer thread
                logger.info(f"主线程: 写入线程 (_frame_writer) 已结束.")
                # We might not need the self._event_complete anymore if we join the writer thread? 
                # Let's keep it for now as the writer sets it in finally.
            else:
                 logger.warning("主线程: 写入线程对象未成功创建，无法 join.")

            # --- Check for Errors after threads finish --- 
            logger.info("主线程: 检查最终错误状态.")
            if self._error.value != 0:
                logger.error(f"渲染过程中发生错误，错误码: {self._error.value}")
                if self.error_callback:
                    self.error_callback(f"渲染过程中发生错误，错误码: {self._error.value}")
                # Restore GC threshold before returning False
                gc.set_threshold(*old_threshold)
                return False
                
            # 恢复原有GC设置 (moved to finally)
            # gc.set_threshold(*old_threshold)
            
            # 记录内存使用情况 (no change)
            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory
            logger.info(f"渲染完成。内存使用: {final_memory / 1024 / 1024:.2f} MB (增长: {memory_growth / 1024 / 1024:.2f} MB, 峰值: {peak_memory / 1024 / 1024:.2f} MB)")
            
            logger.info("render_frames 正常完成.")
            return True

        except Exception as e:
            logger.error(f"渲染视频时发生意外错误 (render_frames): {str(e)}", exc_info=True)
            if self.error_callback:
                self.error_callback(f"渲染视频时出错: {str(e)}")
            # Ensure flags are set to stop other threads and signal completion
            self._event_stop.set()
            self._event_complete.set()
            self.stop_threads = True 
            return False
        finally:
            # 确保停止所有线程 (This finally is for render_frames)
            logger.info("进入 render_frames 的 finally 块")
            self._event_stop.set()
            self.stop_threads = True
            
            # 确保FFmpeg进程被正确关闭
            try:
                # 尝试终止所有可能存在的ffmpeg进程
                import psutil
                current_process = psutil.Process()
                for proc in current_process.children(recursive=True):
                    try:
                        if 'ffmpeg' in proc.name().lower():
                            logger.info(f"正在终止子进程: {proc.name()} (PID: {proc.pid})")
                            proc.terminate()
                            # 给进程一些时间正常退出
                            proc.wait(timeout=2)
                            if proc.is_running():
                                logger.warning(f"进程未响应终止信号，强制结束: {proc.pid}")
                                proc.kill()
                    except Exception as proc_e:
                        logger.warning(f"终止子进程时出错: {proc_e}")
            except Exception as ps_e:
                logger.warning(f"处理子进程时出错: {ps_e}")
            
            # 强制清空帧队列避免内存泄漏
            try:
                while not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except:
                        break
                logger.info("强制清空帧队列完成")
            except Exception as q_e:
                logger.warning(f"清空帧队列时出错: {q_e}")
            
            # Restore GC threshold if not already done
            try:
                if 'old_threshold' in locals():
                   current_threshold = gc.get_threshold()
                   if current_threshold != old_threshold:
                       logger.info(f"在 render_frames finally 中恢复GC阈值: {current_threshold} -> {old_threshold}")
                       gc.set_threshold(*old_threshold)
            except NameError:
                 pass # old_threshold might not be defined if error happened early
            except Exception as gc_e:
                 logger.warning(f"恢复GC阈值时出错: {gc_e}")
            
            # 主动触发一次垃圾回收
            try:
                import gc
                gc.collect()
                logger.info("已触发额外的垃圾回收")
            except Exception as gc_e:
                logger.warning(f"触发额外垃圾回收时出错: {gc_e}")
            
            logger.info("render_frames 方法即将结束")

    def calculate_total_frames(self, text_height, scroll_speed):
        """计算视频需要的总帧数"""
        # 确保滚动速度至少为1像素/帧
        actual_scroll_speed = max(1, scroll_speed)
        logger.info(f"使用实际滚动速度: {actual_scroll_speed}px/帧 (原始: {scroll_speed})")
        
        # --- 修改计算逻辑 --- 
        # 计算文本完全滚出屏幕所需的帧数
        # text_height已经是传入的完整图像高度（包含文本实际高度和屏幕高度）
        # 因此直接使用这个高度计算总帧数
        scroll_frames_needed = int(np.ceil(text_height / actual_scroll_speed))
        logger.info(f"文本完全滚出屏幕需要 {scroll_frames_needed} 帧 (图像总高度={text_height}px)")
        
        # 添加 3 秒停留帧
        pause_frames = self.fps * 3
        logger.info(f"添加 {pause_frames} 帧 ({3} 秒) 用于滚动结束后的停留")
        
        # 总帧数 = 滚动帧数 + 停留帧数
        total_frames = scroll_frames_needed + pause_frames
        # -----------------------

        # 考虑跳帧 (frame_skip)
        # 注意：这里的跳帧计算可能需要重新审视，因为它基于总帧数。
        # 如果跳帧导致总时长变化过大，可能需要调整。
        # 目前保持原样，但标记为潜在优化点。
        if self.actual_frame_skip > 1:
            # 确保总帧数是实际跳帧的倍数，保证完整的视频
            # (这可能会略微增加总时长)
            if total_frames % self.actual_frame_skip != 0:
                original_total_frames = total_frames
                total_frames += self.actual_frame_skip - (total_frames % self.actual_frame_skip)
                logger.info(f"应用跳帧 ({self.actual_frame_skip}) 后，总帧数从 {original_total_frames} 调整为 {total_frames}")
                
        logger.info(f"文本高度: {text_height}px, 滚动速度: {actual_scroll_speed}px/帧, "
                   f"计算总帧数: {total_frames} (滚动: {scroll_frames_needed}, 停留: {pause_frames})")
                   
        # 更新 self.scroll_distance 为总滚动距离
        # 直接使用图像高度，无需再加屏幕高度
        self.scroll_distance = text_height
                   
        return total_frames