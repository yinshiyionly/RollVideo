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
            # I/O优化参数
            "-probesize", "10M",       # 增加探测缓冲区大小
            "-analyzeduration", "10M", # 增加分析时间
            "-thread_queue_size", "1024", # 增加线程队列大小
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
                # 设置最大内存使用量（受限于30GB系统内存上限）
                max_batch_memory_mb = 25 * 1024  # 25GB，保留5GB系统内存空间
                # 计算适合的批处理大小，同时平衡I/O和CPU处理能力
                adaptive_batch_size = max(1, min(100, int(max_batch_memory_mb / frame_memory_mb)))
                # 根据分辨率优化批处理大小
                if self.width * self.height > 2073600: # 大于1080p
                    optimal_batch_size = min(adaptive_batch_size, 60)  # 高分辨率限制批大小
                elif self.width * self.height < 921600: # 小于720p
                    optimal_batch_size = min(adaptive_batch_size, 150) # 低分辨率允许更大批处理
                else:
                    optimal_batch_size = min(adaptive_batch_size, 90)  # 默认平衡点
                
                logger.info(f"单帧内存估算: {frame_memory_mb:.2f}MB, 优化批处理大小: {optimal_batch_size}")
                
                # 使用优化的批处理大小
                batch_size = optimal_batch_size
                num_batches = (total_frames + batch_size - 1) // batch_size
                
                # 决定使用多少个进程进行渲染
                try:
                    cpu_count = mp.cpu_count()
                    # 使用可用CPU核心数，但限制在8核以内
                    num_processes = min(8, max(1, cpu_count - 1))  # 限制最多使用8核，留出一个核心给系统和主进程
                    
                    # 分析系统负载状态，优化进程数
                    cpu_usage_factor = 1.0
                    try:
                        import psutil
                        # 获取当前系统CPU使用率
                        current_cpu_usage = psutil.cpu_percent(interval=0.1)
                        if current_cpu_usage > 70:
                            # 如果系统已经很忙，稍微减少进程数
                            cpu_usage_factor = 0.75
                        elif current_cpu_usage < 30:
                            # 如果系统很空闲，稍微增加进程数
                            cpu_usage_factor = 1.25
                    except ImportError:
                        # 无法进行系统监控，使用固定优化策略
                        logger.info("psutil未安装，使用静态优化策略")
                        # 高度优化策略 - 根据批处理大小选择合适的进程数
                        if batch_size <= 60:
                            cpu_usage_factor = 1.0  # 小批处理使用全部核心
                        elif batch_size <= 90:
                            cpu_usage_factor = 0.9  # 中型批处理略微减少
                        else:
                            cpu_usage_factor = 0.8  # 大批处理进一步减少核心数
                    
                    # 根据批处理大小和系统负载调整进程数
                    if batch_size >= 60 and num_processes > 3:
                        # 对于大批处理，使用较少但效率更高的进程
                        balanced_processes = max(3, min(num_processes - 2, int(num_processes * cpu_usage_factor)))
                        logger.info(f"检测到大批处理模式({batch_size}帧/批)，优化进程数: {num_processes} -> {balanced_processes}")
                        num_processes = balanced_processes
                    else:
                        # 适当优化进程数以匹配系统负载
                        adjusted_processes = max(2, min(num_processes, int(num_processes * cpu_usage_factor)))
                        if adjusted_processes != num_processes:
                            logger.info(f"根据系统负载调整进程数: {num_processes} -> {adjusted_processes}")
                            num_processes = adjusted_processes
                    
                    logger.info(f"检测到{cpu_count}个CPU核心，将使用{num_processes}个进程进行渲染，批处理大小:{batch_size}")
                except Exception as e:
                    # 如果无法检测CPU数量，默认使用4个进程
                    num_processes = 4
                    logger.info(f"无法检测CPU核心数，默认使用{num_processes}个进程: {e}")
                
                # 准备背景色参数（为了多进程）
                if not transparency_required and bg_color and len(bg_color) >= 3:
                    bg_color_rgb = bg_color[:3]
                else:
                    bg_color_rgb = (0, 0, 0)  # 默认黑色
                
                # 设置全局共享图像数组，用于多进程渲染
                global _g_img_array
                _g_img_array = img_array
                
                # 添加一个开始时间记录
                start_time = time.time()
                
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
                
                # 添加性能监控
                perf_monitor = PerformanceMonitor()
                perf_monitor.start()
                last_perf_log = time.time()
                
                # 添加一个循环队列作为帧缓冲
                frame_buffer = deque(maxlen=min(500, total_frames))  # 最多缓存500帧
                frame_buffer_lock = threading.Lock()
                
                # 创建一个事件来指示生产者已完成
                producer_done = threading.Event()
                
                # 创建一个函数来异步向ffmpeg发送帧数据，优化I/O操作
                def frame_sender():
                    # 批量写入缓冲区，减少系统调用
                    write_buffer = bytearray(self.width * self.height * (4 if transparency_required else 3) * 10)
                    write_buffer_view = memoryview(write_buffer)
                    current_pos = 0
                    max_buffer_size = len(write_buffer)
                    
                    while not (producer_done.is_set() and not frame_buffer):
                        # 尝试从缓冲区获取一批帧
                        frames_to_write = []
                        with frame_buffer_lock:
                            # 一次获取多个帧减少锁竞争
                            while frame_buffer and len(frames_to_write) < 5:
                                frames_to_write.append(frame_buffer.popleft())
                        
                        if frames_to_write:
                            try:
                                # 批量处理帧
                                for frame_idx, frame in frames_to_write:
                                    if frame is not None:
                                        # 获取帧数据大小
                                        frame_data = frame.tobytes()
                                        frame_size = len(frame_data)
                                        
                                        # 如果当前缓冲区不足，先刷新缓冲区
                                        if current_pos + frame_size > max_buffer_size:
                                            if current_pos > 0:
                                                # 刷新已缓冲数据
                                                process.stdin.write(write_buffer_view[:current_pos])
                                                current_pos = 0
                                        
                                        # 复制帧数据到缓冲区
                                        if frame_size <= max_buffer_size:
                                            # 使用memoryview高效复制
                                            write_buffer_view[current_pos:current_pos+frame_size] = frame_data
                                            current_pos += frame_size
                                        else:
                                            # 对于超大帧，直接写入
                                            process.stdin.write(frame_data)
                                        
                                        # 记录帧处理
                                        perf_monitor.record_frame()
                                
                                # 如果缓冲区有数据，定期刷新
                                if current_pos > 0 and (len(frames_to_write) < 5 or current_pos > max_buffer_size // 2):
                                    process.stdin.write(write_buffer_view[:current_pos])
                                    current_pos = 0
                                    
                            except (IOError, BrokenPipeError) as e:
                                logger.error(f"发送帧数据时出错: {e}")
                                return
                        else:
                            # 如果没有帧可发送，短暂等待但刷新已有数据
                            if current_pos > 0:
                                try:
                                    process.stdin.write(write_buffer_view[:current_pos])
                                    current_pos = 0
                                except (IOError, BrokenPipeError):
                                    return
                            time.sleep(0.001)
                    
                    # 最终刷新缓冲区
                    if current_pos > 0:
                        try:
                            process.stdin.write(write_buffer_view[:current_pos])
                        except (IOError, BrokenPipeError):
                            pass
                
                # 启动发送者线程
                sender_thread = threading.Thread(target=frame_sender, daemon=True)
                sender_thread.start()
                
                # 使用多进程处理帧
                with mp.Pool(processes=num_processes) as pool:
                    # 逐批次处理所有帧
                    for batch_idx in frame_iterator:
                        batch_start_time = time.time()
                        batch_frames = frame_batch_params[batch_idx]
                        
                        # 根据批次大小优化处理策略
                        if len(batch_frames) > 30 and num_processes > 1:
                            # 大批次处理
                            sub_batch_size = max(10, len(batch_frames) // num_processes)
                            all_processed = []
                            
                            # 分组处理
                            for i in range(0, len(batch_frames), sub_batch_size):
                                sub_batch = batch_frames[i:i+sub_batch_size]
                                # 异步处理子批次
                                sub_results = pool.map(_process_frame, sub_batch)
                                all_processed.extend(sub_results)
                            
                            # 排序结果
                            processed_frames = sorted(all_processed, key=lambda x: x[0])
                        else:
                            # 小批次处理
                            processed_frames = sorted(pool.map(_process_frame, batch_frames), key=lambda x: x[0])
                        
                        # 将帧添加到缓冲区
                        with frame_buffer_lock:
                            for frame_data in processed_frames:
                                frame_buffer.append(frame_data)
                        
                        # 记录批处理性能
                        batch_end_time = time.time()
                        batch_time = batch_end_time - batch_start_time
                        perf_monitor.record_batch(len(batch_frames))
                        
                        # 每10秒打印一次性能信息
                        if time.time() - last_perf_log > 10:
                            perf_monitor.log_stats(logger)
                            last_perf_log = time.time()
                        
                        # 垃圾回收
                        gc_counter += 1
                        if gc_counter >= 5:
                            gc_counter = 0
                            gc.collect()
                            
                            # 内存释放优化
                            if hasattr(os, 'posix_fadvise'):
                                try:
                                    # 通知操作系统可以释放内存
                                    os.posix_fadvise(process.stdin.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                                except:
                                    pass
                            
                            # macOS特定优化
                            if sys.platform == 'darwin':
                                try:
                                    # 对于macOS，使用资源限制优化
                                    resource.setrlimit(resource.RLIMIT_DATA, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
                                except:
                                    pass
                
                # 标记生产者完成
                producer_done.set()
                
                # 等待发送者线程完成
                sender_thread.join(timeout=60)  # 给予最多60秒完成
                
                # 记录最终性能
                perf_monitor.log_stats(logger)
                
                # 清理全局图像数组
                _g_img_array = None
                
                # 计算并输出总帧率
                end_time = time.time()
                total_time = end_time - start_time
                frames_per_second = total_frames / total_time if total_time > 0 else 0
                logger.info(f"总渲染性能: 渲染了{total_frames}帧，耗时{total_time:.2f}秒，平均{frames_per_second:.2f}帧/秒")
                
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
        # 为了获得最佳性能，提前创建输出目录并确保磁盘空间
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # 尝试预热磁盘缓存
        try:
            with open(output_dir + "/.cache_warmup", "wb") as f:
                # 写入一个4MB的文件来预热缓存
                f.write(b'\0' * 4 * 1024 * 1024)
                f.flush()
                os.fsync(f.fileno())
            os.remove(output_dir + "/.cache_warmup")
        except:
            pass
        
        # 尝试进行内存优化
        try:
            # 尝试强制垃圾回收
            gc.collect(generation=2)
            
            # 在macOS上尝试释放内存
            if sys.platform == 'darwin':
                try:
                    import resource
                    resource.setrlimit(resource.RLIMIT_DATA, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
                except:
                    pass
                    
            # 在Linux上尝试释放内存
            if sys.platform.startswith('linux'):
                try:
                    with open('/proc/self/oom_score_adj', 'w') as f:
                        f.write('-500')  # 降低OOM杀死此进程的概率
                except:
                    pass
        except:
            pass
        
        # 执行编码
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