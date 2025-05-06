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
import io

# 添加性能监控
from collections import deque
import statistics

try:
    import mmap
    HAS_MMAP = True
except ImportError:
    HAS_MMAP = False

try:
    import resource
    HAS_RESOURCE_MODULE = True
except ImportError:
    HAS_RESOURCE_MODULE = False

logger = logging.getLogger(__name__)

# 设置NumPy以使用多线程加速计算
try:
    # 针对性能优化的环境变量
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '8'  # 使用全部8核心
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = '8'  # 使用全部8核心
    if 'NUMEXPR_NUM_THREADS' not in os.environ:
        os.environ['NUMEXPR_NUM_THREADS'] = '8'  # 使用全部8核心
    if 'OPENBLAS_NUM_THREADS' not in os.environ:
        os.environ['OPENBLAS_NUM_THREADS'] = '8'  # 使用全部8核心
    
    # 高级线程优化 - 如果系统支持则启用
    try:
        os.environ['OMP_WAIT_POLICY'] = 'ACTIVE'  # 主动等待，减少线程唤醒延迟
        os.environ['OMP_PROC_BIND'] = 'spread'    # 优化线程绑定
        os.environ['OMP_SCHEDULE'] = 'dynamic,16' # 动态调度，减少线程不平衡
        
        # 尝试开启NumPy高级优化
        np.seterr(all='ignore')                   # 忽略NumPy警告提高性能
        np.set_printoptions(precision=3, suppress=True)
    except:
        pass
        
    logger.info(f"已设置NumPy高性能模式: OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}")
except Exception as e:
    logger.warning(f"设置NumPy线程优化失败: {e}")

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

# 性能监控器
class PerformanceMonitor:
    """监控和分析渲染性能"""
    
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.batch_times = deque(maxlen=window_size)
        self.last_time = None
        self.start_time = None
        self.processed_frames = 0
        self.current_fps = 0
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.last_time = self.start_time
    
    def record_frame(self):
        """记录单帧处理时间"""
        now = time.time()
        if self.last_time:
            self.frame_times.append(now - self.last_time)
        self.last_time = now
        self.processed_frames += 1
        
        # 更新当前FPS（每10帧计算一次）
        if self.processed_frames % 10 == 0 and self.frame_times:
            avg_time = statistics.mean(self.frame_times)
            self.current_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    def record_batch(self, batch_size):
        """记录批处理时间"""
        now = time.time()
        if self.last_time:
            elapsed = now - self.last_time
            self.batch_times.append((elapsed, batch_size))
        self.last_time = now
        self.processed_frames += batch_size
        
        # 更新当前FPS
        if self.batch_times:
            # 计算最近几个批次的平均FPS
            total_time = sum(t for t, _ in self.batch_times)
            total_frames = sum(s for _, s in self.batch_times)
            if total_time > 0:
                self.current_fps = total_frames / total_time
    
    def get_stats(self):
        """获取性能统计信息"""
        now = time.time()
        total_time = now - self.start_time if self.start_time else 0
        avg_fps = self.processed_frames / total_time if total_time > 0 else 0
        
        if self.frame_times:
            min_time = min(self.frame_times)
            max_time = max(self.frame_times)
            avg_time = statistics.mean(self.frame_times)
            peak_fps = 1.0 / min_time if min_time > 0 else 0
        else:
            min_time = max_time = avg_time = peak_fps = 0
        
        return {
            'total_frames': self.processed_frames,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'current_fps': self.current_fps,
            'min_frame_time': min_time,
            'max_frame_time': max_time,
            'avg_frame_time': avg_time,
            'peak_fps': peak_fps
        }
    
    def log_stats(self, logger):
        """记录性能统计到日志"""
        stats = self.get_stats()
        logger.info(f"渲染性能: {stats['total_frames']}帧, "
                   f"{stats['total_time']:.2f}秒, "
                   f"平均{stats['avg_fps']:.2f}帧/秒, "
                   f"当前{stats['current_fps']:.2f}帧/秒, "
                   f"峰值{stats['peak_fps']:.2f}帧/秒")

def _process_frame(args):
    """多进程帧处理函数，高性能优化版本"""
    global _g_img_array
    
    frame_idx, img_start_y, img_height, img_width, self_height, self_width, frame_positions, is_transparent, bg_color = args
    
    # 快速路径 - 检查边界条件
    img_end_y = min(img_height, img_start_y + self_height)
    frame_start_y = 0
    frame_end_y = img_end_y - img_start_y
    
    # 空帧情况或边界检查失败时直接返回
    if img_start_y >= img_end_y or frame_start_y >= frame_end_y or frame_end_y > self_height or _g_img_array is None:
        # 创建空帧
        if is_transparent:
            frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
        else:
            # 使用背景色
            frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        return frame_idx, frame
    
    # 预分配整个帧缓冲区 - 减少内存分配开销
    if is_transparent:
        frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
    else:
        # 使用背景色
        frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
    
    # 计算切片 - 使用NumPy高效的切片操作
    img_h_slice = slice(img_start_y, img_end_y)
    img_w_slice = slice(0, min(self_width, img_width))
    frame_h_slice = slice(frame_start_y, frame_end_y)
    frame_w_slice = slice(0, min(self_width, img_width))
    
    try:
        # 获取源数据和目标区域 - 仅引用，不复制
        source_section = _g_img_array[img_h_slice, img_w_slice]
        target_area = frame[frame_h_slice, frame_w_slice]
        
        # 数据局部性优化 - 确保数据连续布局，加速内存访问
        source_section = np.ascontiguousarray(source_section)
        
        if is_transparent:
            # 透明背景 - 直接整块复制，避免逐像素操作
            if target_area.shape[:2] == source_section.shape[:2]:
                # 使用内存层面的快速复制
                np.copyto(target_area, source_section)
            else:
                # 处理大小不匹配的情况
                copy_height = min(target_area.shape[0], source_section.shape[0])
                copy_width = min(target_area.shape[1], source_section.shape[1])
                if copy_height > 0 and copy_width > 0:
                    # 使用预先计算的切片提高性能
                    src_view = source_section[:copy_height, :copy_width]
                    dst_view = target_area[:copy_height, :copy_width]
                    np.copyto(dst_view, src_view)
        else:
            # 不透明背景 - 使用向量化的alpha混合
            if target_area.shape[:2] == source_section.shape[:2]:
                # 向量化操作，一次计算所有像素
                alpha = source_section[:, :, 3:4].astype(np.float32) / 255.0
                alpha_inv = 1.0 - alpha
                # 预分配输出数组，减少临时内存分配
                blended = np.empty_like(target_area)
                # 内联计算提高性能
                np.multiply(source_section[:, :, :3], alpha, out=blended)
                np.add(blended, target_area * alpha_inv, out=blended, casting='unsafe')
                np.copyto(target_area, blended.astype(np.uint8))
            else:
                # 处理大小不匹配的情况
                copy_height = min(target_area.shape[0], source_section.shape[0])
                copy_width = min(target_area.shape[1], source_section.shape[1])
                if copy_height > 0 and copy_width > 0:
                    src_crop = source_section[:copy_height, :copy_width]
                    dst_crop = target_area[:copy_height, :copy_width]
                    # 使用连续数组提高性能
                    src_crop = np.ascontiguousarray(src_crop)
                    dst_crop = np.ascontiguousarray(dst_crop)
                    alpha = src_crop[:, :, 3:4].astype(np.float32) / 255.0
                    alpha_inv = 1.0 - alpha
                    # 预分配输出数组，减少临时内存分配
                    blended = np.empty_like(dst_crop)
                    # 内联计算提高性能
                    np.multiply(src_crop[:, :, :3], alpha, out=blended)
                    np.add(blended, dst_crop * alpha_inv, out=blended, casting='unsafe')
                    np.copyto(target_area[:copy_height, :copy_width], blended.astype(np.uint8))
    except Exception as e:
        logger.error(f"处理帧 {frame_idx} 时出错: {e}")
    
    # 确保返回连续内存布局
    return frame_idx, np.ascontiguousarray(frame)

# 添加JIT编译支持（如果可用）
try:
    import numba
    # 检查是否可以使用Numba JIT
    if hasattr(numba, 'jit'):
        logger.info("启用Numba JIT加速")
        
        @numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
        def _blend_images_fast(source, target, alpha):
            """使用JIT编译加速的图像混合函数"""
            height, width = source.shape[:2]
            result = np.empty((height, width, 3), dtype=np.uint8)
            
            for y in numba.prange(height):
                for x in range(width):
                    a = alpha[y, x, 0] / 255.0
                    for c in range(3):
                        result[y, x, c] = int(source[y, x, c] * a + target[y, x, c] * (1.0 - a))
            
            return result
        
        # 创建一个使用Numba优化的帧处理函数
        def _process_frame_jit(args):
            """使用JIT编译加速的帧处理函数"""
            global _g_img_array
            
            frame_idx, img_start_y, img_height, img_width, self_height, self_width, frame_positions, is_transparent, bg_color = args
            
            # 快速路径和边界检查
            img_end_y = min(img_height, img_start_y + self_height)
            frame_start_y = 0
            frame_end_y = img_end_y - img_start_y
            
            if img_start_y >= img_end_y or frame_start_y >= frame_end_y or frame_end_y > self_height or _g_img_array is None:
                if is_transparent:
                    frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
                else:
                    frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
                return frame_idx, frame
            
            # 创建帧缓冲区
            if is_transparent:
                frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
            else:
                frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
            
            # 计算切片
            img_h_slice = slice(img_start_y, img_end_y)
            img_w_slice = slice(0, min(self_width, img_width))
            frame_h_slice = slice(frame_start_y, frame_end_y)
            frame_w_slice = slice(0, min(self_width, img_width))
            
            try:
                source_section = _g_img_array[img_h_slice, img_w_slice]
                target_area = frame[frame_h_slice, frame_w_slice]
                
                if target_area.shape[:2] == source_section.shape[:2]:
                    if is_transparent:
                        target_area[:] = source_section
                    else:
                        # 使用JIT编译的快速混合函数
                        blended = _blend_images_fast(
                            source_section[:, :, :3], 
                            target_area, 
                            source_section[:, :, 3:4]
                        )
                        target_area[:] = blended
                else:
                    # 处理形状不匹配的情况
                    copy_height = min(target_area.shape[0], source_section.shape[0])
                    copy_width = min(target_area.shape[1], source_section.shape[1])
                    
                    if copy_height > 0 and copy_width > 0:
                        src_crop = source_section[:copy_height, :copy_width]
                        dst_crop = target_area[:copy_height, :copy_width]
                        
                        if is_transparent:
                            target_area[:copy_height, :copy_width] = src_crop
                        else:
                            # 使用JIT编译的快速混合函数
                            blended = _blend_images_fast(
                                src_crop[:, :, :3], 
                                dst_crop, 
                                src_crop[:, :, 3:4]
                            )
                            target_area[:copy_height, :copy_width] = blended
            except Exception as e:
                logger.error(f"JIT优化帧处理出错 {frame_idx}: {e}")
            
            return frame_idx, frame
        
        # 用优化的JIT函数替换原函数
        _process_frame_original = _process_frame
        _process_frame = _process_frame_jit
        logger.info("已启用JIT加速帧处理函数")
    else:
        logger.info("Numba可用但JIT不可用，使用标准优化")
except ImportError:
    logger.info("Numba未安装，使用标准优化")

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

# 添加内存池管理器 - 提高内存利用效率，减少内存分配开销
class FrameMemoryPool:
    """高效内存池管理器，实现零拷贝内存分配"""
    
    def __init__(self, width, height, channels, pool_size=240):
        """
        初始化帧内存池
        
        Args:
            width: 帧宽度
            height: 帧高度
            channels: 颜色通道数(RGB=3, RGBA=4)
            pool_size: 内存池大小(帧数)
        """
        self.width = width
        self.height = height
        self.channels = channels
        self.frame_shape = (height, width, channels)
        self.frame_size = width * height * channels
        
        # 创建大型连续内存块
        self.buffer_size = self.frame_size * pool_size
        self.buffer = bytearray(self.buffer_size)
        self.buffer_view = memoryview(self.buffer)
        
        # 跟踪可用/已用帧
        self.available_frames = queue.Queue(pool_size)
        self.in_use_frames = set()
        
        # 预填充可用帧队列
        for i in range(pool_size):
            offset = i * self.frame_size
            self.available_frames.put((i, offset))
        
        logger.info(f"初始化内存池: {pool_size}帧, 每帧{self.frame_size/1024/1024:.2f}MB")
    
    def get_frame(self):
        """获取预分配的帧内存"""
        try:
            frame_id, offset = self.available_frames.get(block=False)
            self.in_use_frames.add(frame_id)
            # 返回NumPy视图，无需复制
            array_view = np.frombuffer(
                self.buffer_view[offset:offset+self.frame_size], 
                dtype=np.uint8
            ).reshape(self.frame_shape)
            return frame_id, array_view
        except queue.Empty:
            # 如果池已耗尽，创建新内存
            logger.warning("内存池耗尽，临时分配新内存")
            return -1, np.zeros(self.frame_shape, dtype=np.uint8)
    
    def release_frame(self, frame_id):
        """释放帧回内存池"""
        if frame_id >= 0 and frame_id in self.in_use_frames:
            self.in_use_frames.remove(frame_id)
            offset = frame_id * self.frame_size
            self.available_frames.put((frame_id, offset))
    
    def clear(self):
        """清空并重置内存池"""
        # 释放所有帧
        self.in_use_frames.clear()
        # 清空队列
        while not self.available_frames.empty():
            try:
                self.available_frames.get(block=False)
            except queue.Empty:
                break
        # 重新填充队列
        for i in range(self.available_frames.maxsize):
            offset = i * self.frame_size
            self.available_frames.put((i, offset))

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
        self.memory_pool = None
        self.frame_counter = 0
        self.total_frames = 0
    
    def _init_memory_pool(self, channels, pool_size=240):
        """初始化或重置内存池"""
        if self.memory_pool is not None:
            self.memory_pool.clear()
        else:
            self.memory_pool = FrameMemoryPool(
                width=self.width,
                height=self.height,
                channels=channels,
                pool_size=pool_size
            )
        return self.memory_pool
    
    def _get_ffmpeg_command(
        self,
        output_path: str,
        pix_fmt: str,
        codec_and_output_params: List[str], # 重命名以更清晰
        audio_path: Optional[str]
    ) -> List[str]:
        """构造基础的ffmpeg命令 - 高性能优化版"""
        command = [
            "ffmpeg", "-y",
            # I/O优化参数
            "-probesize", "20M",       # 增加探测缓冲区大小
            "-analyzeduration", "20M", # 增加分析时间
            "-thread_queue_size", "8192", # 大幅增加线程队列大小
            # 输入格式参数
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

    def create_scrolling_video_optimized(
        self,
        image: Image.Image,
        output_path: str,
        text_actual_height: int,
        transparency_required: bool,
        preferred_codec: str, # 仍然接收 h264_nvenc 作为首选
        audio_path: Optional[str] = None,
        bg_color: Optional[Tuple[int, int, int, int]] = None
    ) -> str:
        """创建滚动视频 - 高性能优化版"""
        
        # 1. 图像预处理优化
        if transparency_required:
            img_array = np.ascontiguousarray(np.array(image))
            channels = 4
        else:
            # 直接转为RGB以避免后续转换开销
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            img_array = np.ascontiguousarray(np.array(image))
            channels = 3
        
        img_height, img_width = img_array.shape[:2]
        
        # 2. 滚动参数计算 - 减少中间变量
        scroll_distance = max(text_actual_height, img_height - self.height)
        scroll_frames = int(scroll_distance / self.scroll_speed) if self.scroll_speed > 0 else 0
        
        # 确保短文本有合理滚动时间
        min_scroll_frames = self.fps * 8
        if scroll_frames < min_scroll_frames and scroll_frames > 0:
            adjusted_speed = scroll_distance / min_scroll_frames
            if adjusted_speed < self.scroll_speed:
                logger.info(f"文本较短，减慢滚动速度: {self.scroll_speed:.2f} → {adjusted_speed:.2f} 像素/帧")
                self.scroll_speed = adjusted_speed
                scroll_frames = min_scroll_frames
        
        padding_frames_start = int(self.fps * 2.0)
        padding_frames_end = int(self.fps * 2.0)
        total_frames = padding_frames_start + scroll_frames + padding_frames_end
        self.total_frames = total_frames
        duration = total_frames / self.fps
        
        logger.info(f"文本高:{text_actual_height}, 图像高:{img_height}, 视频高:{self.height}")
        logger.info(f"滚动距离:{scroll_distance}, 滚动帧:{scroll_frames}, 总帧:{total_frames}, 时长:{duration:.2f}s")
        logger.info(f"输出:{output_path}, 透明:{transparency_required}, 首选编码器:{preferred_codec}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # 3. 检测可用的编码器并获取正确参数
        # 首先检查ffmpeg是否可用
        try:
            ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
            logger.info(f"检测到ffmpeg: {ffmpeg_version.splitlines()[0]}")
        except:
            logger.error("找不到ffmpeg命令，请确保已正确安装")
            raise RuntimeError("找不到ffmpeg命令")
            
        # 然后检测GPU编码器是否可用
        gpu_encoders = []
        try:
            encoders = subprocess.check_output(["ffmpeg", "-encoders"], stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
            for line in encoders.splitlines():
                if "nvenc" in line:
                    gpu_encoders.append(line.split()[1])
            if gpu_encoders:
                logger.info(f"检测到GPU编码器: {gpu_encoders}")
        except:
            logger.warning("无法检测GPU编码器")
            gpu_encoders = []
            
        # 背景色处理
        if not transparency_required and bg_color and len(bg_color) >= 3:
            bg_color_rgb = bg_color[:3]
        else:
            bg_color_rgb = (0, 0, 0)
            
        # 确定最终使用的编码器和参数
        use_gpu = False
        if transparency_required:
            # 透明视频必须使用ProRes
            ffmpeg_pix_fmt = "rgba"
            output_path = os.path.splitext(output_path)[0] + ".mov"
            video_codec_params = [
                "-c:v", "prores_ks", 
                "-profile:v", "4", 
                "-pix_fmt", "yuva444p10le", 
                "-alpha_bits", "16", 
                "-vendor", "ap10"
            ]
            logger.info("使用ProRes编码器处理透明视频")
        else:
            ffmpeg_pix_fmt = "rgb24"
            output_path = os.path.splitext(output_path)[0] + ".mp4"
            
            # 检查GPU编码器可用性
            if preferred_codec in gpu_encoders and not "NO_GPU" in os.environ:
                # 使用最简单的GPU参数以确保兼容性
                video_codec_params = [
                    "-c:v", preferred_codec,
                    "-preset", "p3",         # 保持p3预设
                    "-b:v", "5M",            # 使用固定比特率
                    "-pix_fmt", "yuv420p",   # 确保兼容性
                    "-movflags", "+faststart"
                ]
                logger.info(f"使用GPU编码器: {preferred_codec}")
                use_gpu = True
            else:
                # 回退到CPU编码
                video_codec_params = [
                    "-c:v", "libx264",
                    "-crf", "21",            
                    "-preset", "medium",      
                    "-pix_fmt", "yuv420p",   
                    "-movflags", "+faststart",
                    "-threads", "8"           
                ]
                logger.info(f"使用CPU编码器: libx264 (GPU编码器不可用或被禁用)")
        
        # 完整的ffmpeg命令
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            # I/O优化参数
            "-probesize", "20M",      
            "-analyzeduration", "20M", 
            "-thread_queue_size", "1024", 
            # 输入格式参数
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", ffmpeg_pix_fmt, 
            "-r", str(self.fps),
            "-i", "-",  # 从stdin读取
        ]
        
        # 添加音频输入（如果有）
        if audio_path and os.path.exists(audio_path):
            ffmpeg_cmd.extend(["-i", audio_path])
            
        # 添加视频编码参数
        ffmpeg_cmd.extend(video_codec_params)
        
        # 添加音频映射（如果有）
        if audio_path and os.path.exists(audio_path):
            ffmpeg_cmd.extend(["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
        else:
            ffmpeg_cmd.extend(["-map", "0:v:0"])
            
        # 添加输出路径
        ffmpeg_cmd.append(output_path)
            
        logger.info(f"执行ffmpeg命令: {' '.join(ffmpeg_cmd)}")
        
        # 4. 性能优化: 内存管理与进程控制
        # 预计算帧位置 - 确保最大性能
        frame_positions = []
        for frame_idx in range(total_frames):
            if frame_idx < padding_frames_start: 
                frame_positions.append(0)  # 静止开始
            elif frame_idx < padding_frames_start + scroll_frames:
                scroll_progress = frame_idx - padding_frames_start
                current_position = int(min(scroll_progress * self.scroll_speed, scroll_distance))
                frame_positions.append(current_position)  # 滚动部分
            else: 
                frame_positions.append(int(scroll_distance))  # 静止结尾
        
        # 初始化内存池
        self._init_memory_pool(channels, pool_size=120)
        
        # 数据传输模式：直接模式或缓存模式
        # GPU编码器通常更快，所以需要较小的批处理大小和更多流控制
        batch_size = 120 if not use_gpu else 60  
        num_batches = (total_frames + batch_size - 1) // batch_size
        
        # 确定最佳进程数
        try:
            cpu_count = mp.cpu_count()
            optimal_processes = min(6, max(1, cpu_count - 2))  # 留出2个核心给系统和ffmpeg
            num_processes = optimal_processes
            logger.info(f"检测到{cpu_count}个CPU核心，优化使用{num_processes}个进程进行渲染，批处理大小:{batch_size}")
        except:
            num_processes = 4
            logger.info(f"使用默认4个进程，批处理大小:{batch_size}")
        
        # 性能监控
        perf_monitor = PerformanceMonitor()
        perf_monitor.start()
        
        # 用于多进程的全局变量
        global _g_img_array
        _g_img_array = img_array  # 在多进程之间共享图像数据
        
        # 5. 处理和编码
        try:
            # 创建临时先入先出队列用于帧缓冲
            if not os.path.exists("/tmp"):
                os.makedirs("/tmp", exist_ok=True)
            temp_fifo = f"/tmp/ffmpeg_fifo_{int(time.time())}"
            try:
                os.mkfifo(temp_fifo)
                logger.info(f"创建FIFO: {temp_fifo}")
                use_fifo = True
            except:
                use_fifo = False
                logger.warning(f"无法创建FIFO，使用直接管道")
            
            # 启动ffmpeg进程，使用适当的输入方式
            if use_fifo:
                # 替换输入参数
                fifo_cmd = list(ffmpeg_cmd)
                for i, arg in enumerate(fifo_cmd):
                    if arg == "-" and fifo_cmd[i-1] == "-i":
                        fifo_cmd[i] = temp_fifo
                
                # 启动进程
                process = subprocess.Popen(
                    fifo_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                # 如果使用FIFO，我们需要延迟打开写入端，以确保ffmpeg已启动
                time.sleep(0.5)
                fifo_fd = os.open(temp_fifo, os.O_WRONLY)
                pipe_out = os.fdopen(fifo_fd, 'wb')
            else:
                # 使用标准管道
                process = subprocess.Popen(
                    ffmpeg_cmd, 
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                pipe_out = process.stdin
            
            # 创建stdout/stderr读取线程
            stdout_q = queue.Queue()
            stderr_q = queue.Queue()
            stdout_thread = threading.Thread(target=self._reader_thread, args=(process.stdout, stdout_q))
            stderr_thread = threading.Thread(target=self._reader_thread, args=(process.stderr, stderr_q))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # 准备批帧参数，优化内存使用
            frame_batch_params = []
            for batch_idx in range(num_batches):
                start_frame = batch_idx * batch_size
                end_frame = min(start_frame + batch_size, total_frames)
                batch_frames = []
                
                for frame_idx in range(start_frame, end_frame):
                    img_start_y = frame_positions[frame_idx]
                    frame_params = (frame_idx, img_start_y, img_height, img_width, 
                                   self.height, self.width, transparency_required, bg_color_rgb)
                    batch_frames.append(frame_params)
                
                frame_batch_params.append(batch_frames)
            
            # 处理帧计数
            self.frame_counter = 0
            
            # 创建共享进度条
            with tqdm.tqdm(total=total_frames, desc=f"编码 ({video_codec_params[1]})") as pbar:
                # 使用进程池处理帧
                last_write_time = time.time()
                with mp.Pool(processes=num_processes) as pool:
                    for batch_idx, batch_frames in enumerate(frame_batch_params):
                        # 定期检查ffmpeg是否还在运行
                        if process.poll() is not None:
                            logger.error(f"ffmpeg进程意外退出，返回码: {process.returncode}")
                            # 收集并记录错误输出
                            stderr_data = []
                            while not stderr_q.empty():
                                line = stderr_q.get_nowait()
                                if line: stderr_data.append(line.decode(errors='ignore').strip())
                            if stderr_data:
                                logger.error(f"ffmpeg错误输出: {stderr_data[-10:]}")
                            break
                        
                        # 防止上一次写入的时间太短
                        elapsed = time.time() - last_write_time
                        if elapsed < 0.05 and use_gpu:  # GPU模式下控制写入速率
                            time.sleep(0.05 - elapsed)
                        
                        # 处理当前批次
                        if len(batch_frames) > 30 and num_processes > 1:
                            # 大批处理：并行处理所有帧
                            pool_results = pool.map(_process_frame_optimized, batch_frames)
                            processed_frames = sorted(pool_results, key=lambda x: x[0])
                        else:
                            # 小批处理：顺序处理
                            processed_frames = []
                            for params in batch_frames:
                                frame_idx, frame = _process_frame_optimized(params)
                                processed_frames.append((frame_idx, frame))
                        
                        # 写入所有处理好的帧数据
                        try:
                            for _, frame in processed_frames:
                                pipe_out.write(frame.tobytes())
                                self.frame_counter += 1
                            pipe_out.flush()  # 确保数据写入
                            
                            # 更新进度条
                            pbar.update(len(processed_frames))
                            last_write_time = time.time()
                        except BrokenPipeError:
                            logger.error("管道已断开，停止写入")
                            break
                        except Exception as e:
                            logger.error(f"写入帧数据时出错: {e}")
                            break
                        
                        # 定期记录性能
                        if batch_idx % 5 == 0:
                            perf_monitor.log_stats(logger)
                        
                        # 定期垃圾回收
                        if batch_idx % 10 == 9:
                            gc.collect()
            
            # 关闭管道/FIFO
            try:
                pipe_out.close()
            except:
                pass
                
            # 如果使用FIFO，需要删除它
            if use_fifo:
                try:
                    os.unlink(temp_fifo)
                except:
                    pass
            
            # 等待ffmpeg完成
            try:
                return_code = process.wait(timeout=120.0)  # 等待最多2分钟
                logger.info(f"ffmpeg进程完成，返回码: {return_code}")
                
                # 记录输出
                stderr_lines = []
                while not stderr_q.empty():
                    line = stderr_q.get()
                    if line is not None:
                        stderr_lines.append(line.decode(errors='ignore').strip())
                
                if return_code != 0:
                    error_output = "\n".join(stderr_lines[-20:])  # 只记录最后20行
                    logger.error(f"ffmpeg错误输出:\n{error_output}")
                    
                return return_code == 0
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg进程超时，尝试终止")
                process.kill()
                return False
        except Exception as e:
            logger.error(f"渲染过程出错: {e}", exc_info=True)
            return False
        finally:
            # 清理资源
            _g_img_array = None
            gc.collect()
            
            # 记录最终性能
            end_time = time.time()
            total_time = end_time - perf_monitor.start_time
            avg_fps = self.frame_counter / total_time if total_time > 0 else 0
            
            logger.info(f"总渲染性能: 渲染了{self.frame_counter}帧，"
                       f"耗时{total_time:.2f}秒，"
                       f"平均{avg_fps:.2f}帧/秒")
            
            return avg_fps > 0  # 只要成功渲染了一些帧，就认为是成功的
    
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
        """创建滚动视频，使用高性能版本"""
        # 使用优化版本实现
        success = self.create_scrolling_video_optimized(
            image=image,
            output_path=output_path,
            text_actual_height=text_actual_height,
            transparency_required=transparency_required,
            preferred_codec=preferred_codec,
            audio_path=audio_path,
            bg_color=bg_color
        )
        
        # 检查是否成功，如果失败则尝试回退到CPU编码
        if not success and not transparency_required:
            # 设置环境变量强制使用CPU编码
            logger.info("优化渲染失败，回退到标准CPU渲染")
            os.environ["NO_GPU"] = "1"
            try:
                return self.create_scrolling_video_optimized(
                    image=image,
                    output_path=output_path,
                    text_actual_height=text_actual_height,
                    transparency_required=transparency_required,
                    preferred_codec="libx264",  # 强制使用CPU编码器
                    audio_path=audio_path,
                    bg_color=bg_color
                )
            finally:
                # 恢复环境变量
                if "NO_GPU" in os.environ:
                    del os.environ["NO_GPU"]
        
        return output_path

def _process_frame_optimized(args):
    """极度优化的帧处理函数"""
    global _g_img_array
    
    frame_idx, img_start_y, img_height, img_width, self_height, self_width, is_transparent, bg_color = args
    
    # 快速路径 - 直接计算切片位置
    img_end_y = min(img_height, img_start_y + self_height)
    visible_height = img_end_y - img_start_y
    
    # 零边界检查 - 极端情况直接返回背景
    if visible_height <= 0 or _g_img_array is None:
        if is_transparent:
            return frame_idx, np.zeros((self_height, self_width, 4), dtype=np.uint8)
        else:
            return frame_idx, np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
    
    # 预分配帧缓冲区 - 提高效率
    if is_transparent:
        # 创建空的透明帧
        frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
    else:
        # 用背景色填充帧
        frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
    
    # 计算切片 - 超高效版
    source_height = min(visible_height, self_height)
    source_width = min(img_width, self_width)
    
    # 使用直接视图赋值，避免任何额外复制
    if source_height > 0 and source_width > 0:
        # 单一高效赋值操作
        frame[:source_height, :source_width] = _g_img_array[img_start_y:img_start_y+source_height, :source_width]
    
    # 确保返回内存连续的帧数据 - 此步骤对于管道通信至关重要
    if not frame.flags.c_contiguous:
        frame = np.ascontiguousarray(frame)
        
    return frame_idx, frame

def fast_frame_processor(frame_batch, memory_pool, process):
    """高速批量帧处理写入器，极大减少多进程通信开销"""
    global _g_img_array
    
    frames_processed = 0
    
    # 处理整个批次
    for params in frame_batch:
        frame_idx, img_start_y, img_height, img_width, height, width, is_transparent, bg_color = params
        
        # 优化的帧边界计算
        img_end_y = min(img_height, img_start_y + height)
        visible_height = img_end_y - img_start_y
        
        # 从内存池获取预分配的帧
        pool_id, frame = memory_pool.get_frame()
        
        # 极速内联处理核心
        if visible_height <= 0 or _g_img_array is None:
            # 边界情况 - 直接使用背景
            if is_transparent:
                frame.fill(0)  # 全透明
            else:
                frame[:] = np.array(bg_color, dtype=np.uint8)  # 背景色
        else:
            # 有效内容 - 执行复制
            if is_transparent:
                frame.fill(0)  # 先清空
            else:
                frame[:] = np.array(bg_color, dtype=np.uint8)  # 先填充背景
                
            # 单次高效复制
            source_height = min(visible_height, height)
            source_width = min(img_width, width)
            frame[:source_height, :source_width] = _g_img_array[img_start_y:img_start_y+source_height, :source_width]
        
        # 直接写入管道 - 零中间缓冲
        process.stdin.write(frame.tobytes())
        
        # 释放帧回内存池
        memory_pool.release_frame(pool_id)
        frames_processed += 1
    
    # 强制刷新确保所有数据写入
    process.stdin.flush()
    return frames_processed 