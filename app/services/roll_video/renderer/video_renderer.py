"""视频渲染器模块"""

import os
import sys
import logging
import numpy as np
import subprocess
import threading
import queue
from tqdm import tqdm
import gc
import time
import multiprocessing as mp
from PIL import Image
from typing import Dict, Tuple, List, Optional, Union
import platform
from collections import defaultdict
import psutil
import traceback
import signal
from multiprocessing import shared_memory
import random
import string

from .memory_management import FrameMemoryPool, SharedMemoryFramePool, FrameBuffer
from .performance import PerformanceMonitor
from .frame_processors import (
    _process_frame,
    _process_frame_optimized,
    _process_frame_optimized_shm,
    fast_frame_processor,
    init_shared_memory,
    cleanup_shared_memory,
    init_worker,
    test_worker_shared_memory,
)
from .utils import time_tracker, get_memory_usage, optimize_memory, emergency_cleanup

logger = logging.getLogger(__name__)


class VideoRenderer:
    """视频渲染器，负责创建滚动效果的视频，使用ffmpeg管道和线程读取优化"""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 30,
        scroll_speed: int = 5,  # 每帧滚动的像素数（由service层基于行高和每秒滚动行数计算而来）
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

    def _init_memory_pool(self, channels=3, pool_size=120):
        """
        初始化内存池，预分配帧缓冲区

        Args:
            channels: 通道数，3表示RGB，4表示RGBA
            pool_size: 内存池大小
        """
        logger.info(
            f"初始化内存池: {pool_size}个{self.width}x{self.height}x{channels}帧缓冲区"
        )
        self.memory_pool = []

        try:
            for _ in range(pool_size):
                # 预分配连续内存
                frame = np.zeros(
                    (self.height, self.width, channels), dtype=np.uint8, order="C"
                )
                self.memory_pool.append(frame)
        except Exception as e:
            logger.warning(f"内存池初始化失败: {e}，将使用动态分配")
            # 如果内存不足，减小池大小重试
            if pool_size > 30:
                logger.info(f"尝试减小内存池大小至30")
                self._init_memory_pool(channels, 30)
        return self.memory_pool

    def _get_ffmpeg_command(
        self,
        output_path: str,
        pix_fmt: str,
        codec_and_output_params: List[str],  # 重命名以更清晰
        audio_path: Optional[str],
    ) -> List[str]:
        """构造基础的ffmpeg命令 - 高性能优化版"""
        command = [
            "ffmpeg",
            "-y",
            # I/O优化参数
            "-probesize",
            "20M",  # 增加探测缓冲区大小
            "-analyzeduration",
            "20M",  # 增加分析时间
            "-thread_queue_size",
            "8192",  # 大幅增加线程队列大小
            # 输入格式参数
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            pix_fmt,
            "-r",
            str(self.fps),
            "-i",
            "-",  # 从 stdin 读取
        ]
        if audio_path and os.path.exists(audio_path):
            command.extend(["-i", audio_path])

        # 添加视频编码器和特定的输出参数 (如 -movflags)
        command.extend(codec_and_output_params)

        if audio_path and os.path.exists(audio_path):
            command.extend(
                ["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
            )
        else:
            command.extend(["-map", "0:v:0"])

        command.append(output_path)
        return command

    def _reader_thread(self, pipe, output_queue):
        """读取管道输出并放入队列"""
        try:
            with pipe:
                for line in iter(pipe.readline, b""):
                    output_queue.put(line)
        finally:
            output_queue.put(None)  # 发送结束信号

    def create_scrolling_video_optimized(
        self,
        image: Image.Image,
        output_path: str,
        text_actual_height: int,
        transparency_required: bool,
        preferred_codec: str,
        audio_path: Optional[str] = None,
        bg_color: Optional[Tuple[int, int, int, int]] = None,
    ) -> str:
        """创建滚动视频 - 高性能优化版 (共享内存 + 异步处理)"""

        # 1. 图像预处理优化
        if transparency_required:
            img_array = np.ascontiguousarray(np.array(image))
            channels = 4
        else:
            # 直接转为RGB以避免后续转换开销
            if image.mode == "RGBA":
                image = image.convert("RGB")
            img_array = np.ascontiguousarray(np.array(image))
            channels = 3

        img_height, img_width = img_array.shape[:2]

        # 2. 滚动参数计算 - 减少中间变量
        scroll_distance = max(text_actual_height, img_height - self.height)
        scroll_frames = (
            int(scroll_distance / self.scroll_speed) if self.scroll_speed > 0 else 0
        )

        # 确保短文本有合理滚动时间
        min_scroll_frames = self.fps * 8
        if scroll_frames < min_scroll_frames and scroll_frames > 0:
            adjusted_speed = scroll_distance / min_scroll_frames
            if adjusted_speed < self.scroll_speed:
                logger.info(
                    f"文本较短，减慢滚动速度: {self.scroll_speed:.2f} → {adjusted_speed:.2f} 像素/帧"
                )
                self.scroll_speed = adjusted_speed
                scroll_frames = min_scroll_frames

        padding_frames_start = int(self.fps * 2.0)
        padding_frames_end = int(self.fps * 2.0)
        total_frames = padding_frames_start + scroll_frames + padding_frames_end
        self.total_frames = total_frames
        duration = total_frames / self.fps

        logger.info(
            f"文本高:{text_actual_height}, 图像高:{img_height}, 视频高:{self.height}"
        )
        logger.info(
            f"滚动距离:{scroll_distance}, 滚动帧:{scroll_frames}, 总帧:{total_frames}, 时长:{duration:.2f}s"
        )
        logger.info(
            f"输出:{output_path}, 透明:{transparency_required}, 首选编码器:{preferred_codec}"
        )

        # 确保输出目录存在
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 3. 检测可用的编码器并获取正确参数
        # 首先检查ffmpeg是否可用
        try:
            ffmpeg_version = subprocess.check_output(
                ["ffmpeg", "-version"], stderr=subprocess.STDOUT
            ).decode("utf-8", errors="ignore")
            logger.info(f"检测到ffmpeg: {ffmpeg_version.splitlines()[0]}")
        except:
            logger.error("找不到ffmpeg命令，请确保已正确安装")
            raise RuntimeError("找不到ffmpeg命令")

        # 然后检测GPU编码器是否可用
        gpu_encoders = []
        try:
            encoders = subprocess.check_output(
                ["ffmpeg", "-encoders"], stderr=subprocess.STDOUT
            ).decode("utf-8", errors="ignore")
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
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4",
                "-pix_fmt",
                "yuva444p10le",
                "-alpha_bits",
                "16",
                "-vendor",
                "ap10",
            ]
            logger.info("使用ProRes编码器处理透明视频")
        else:
            ffmpeg_pix_fmt = "rgb24"
            output_path = os.path.splitext(output_path)[0] + ".mp4"

            # 检查操作系统，在macOS上默认使用CPU编码
            is_macos = platform.system() == "Darwin"
            if is_macos:
                # macOS上强制使用CPU编码
                logger.info("检测到macOS系统，将使用CPU编码器")
                os.environ["NO_GPU"] = "1"

            # 检查GPU编码器可用性
            if preferred_codec in gpu_encoders and not "NO_GPU" in os.environ:
                # 使用更高性能的GPU参数
                video_codec_params = [
                    "-c:v",
                    preferred_codec,
                    "-preset",
                    "p4",  # 提升到p4预设（更高性能）
                    "-b:v",
                    "8M",  # 提升到8M比特率
                    "-pix_fmt",
                    "yuv420p",  # 确保兼容性
                    "-movflags",
                    "+faststart",
                ]
                logger.info(f"使用GPU编码器: {preferred_codec}，预设:p4，比特率:8M")
                use_gpu = True
            else:
                # 回退到CPU编码
                video_codec_params = [
                    "-c:v",
                    "libx264",
                    "-crf",
                    "21",
                    "-preset",
                    "medium",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-threads",
                    "8",
                ]
                logger.info(f"使用CPU编码器: libx264 (GPU编码器不可用或被禁用)")
                use_gpu = False

        # 完整的ffmpeg命令
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            # I/O优化参数
            "-probesize",
            "20M",
            "-analyzeduration",
            "20M",
            "-thread_queue_size",
            "8192",
            # 输入格式参数
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            ffmpeg_pix_fmt,
            "-r",
            str(self.fps),
            "-i",
            "-",  # 从stdin读取
        ]

        # 添加音频输入（如果有）
        if audio_path and os.path.exists(audio_path):
            ffmpeg_cmd.extend(["-i", audio_path])

        # 添加视频编码参数
        ffmpeg_cmd.extend(video_codec_params)

        # 添加音频映射（如果有）
        if audio_path and os.path.exists(audio_path):
            ffmpeg_cmd.extend(
                [
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-shortest",
                ]
            )
        else:
            ffmpeg_cmd.extend(["-map", "0:v:0"])

        # 添加输出路径
        ffmpeg_cmd.append(output_path)

        logger.info(f"执行ffmpeg命令: {' '.join(ffmpeg_cmd)}")

        # 4. 性能优化: 共享内存管理与多进程控制
        # 预计算帧位置 - 确保最大性能
        frame_positions = []
        for frame_idx in range(total_frames):
            if frame_idx < padding_frames_start:
                frame_positions.append(0)  # 静止开始
            elif frame_idx < padding_frames_start + scroll_frames:
                scroll_progress = frame_idx - padding_frames_start
                current_position = int(
                    min(scroll_progress * self.scroll_speed, scroll_distance)
                )
                frame_positions.append(current_position)  # 滚动部分
            else:
                frame_positions.append(int(scroll_distance))  # 静止结尾

        # 5. 使用共享内存存储图像数据
        try:
            # 初始化共享内存并存储图像数据
            logger.info("正在将图像数据存入共享内存...")
            shm_name, shm_shape, shm_dtype = init_shared_memory(img_array)
            logger.info(f"图像数据已存入共享内存: {shm_name}")
            
            # 初始化进程池与任务处理
            # 确定最佳进程数
            try:
                cpu_count = mp.cpu_count()
                # 减少使用的核心数，为系统和ffmpeg留下更多资源
                optimal_processes = min(8, max(2, cpu_count - 2))
                num_processes = optimal_processes
                logger.info(
                    f"检测到{cpu_count}个CPU核心，优化使用{num_processes}个进程进行渲染"
                )
            except:
                num_processes = 4  # 默认使用较少进程
                logger.info(f"使用默认{num_processes}个进程")

            # 批处理大小
            # GPU编码器通常更快，所以需要较小的批处理大小和更多流控制
            batch_size = 240 if not use_gpu else 120
            num_batches = (total_frames + batch_size - 1) // batch_size
            
            # 测试工作进程能否正确访问共享内存
            logger.info("测试工作进程能否正确访问共享内存...")
            
            with mp.Pool(processes=1) as test_pool:
                test_result = test_pool.apply(
                    test_worker_shared_memory,
                    args=(shm_name, shm_shape, shm_dtype)
                )
                success, message = test_result
                
                if not success:
                    logger.error(f"共享内存测试失败: {message}")
                    logger.error("切换到备用模式...")
                    # 这里可以添加备用模式的处理逻辑，例如使用旧的处理方法
                    raise RuntimeError(f"共享内存访问测试失败: {message}")
                
                logger.info(f"共享内存测试成功: {message}")
            
            # 6. 运行FFmpeg进程并设置线程
            try:
                # 启动ffmpeg进程
                logger.info("启动FFmpeg进程...")
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10 * 1024 * 1024,  # 10MB缓冲区
                )
                logger.info("FFmpeg进程已启动")

                # 创建队列用于捕获输出
                stdout_q = queue.Queue()
                stderr_q = queue.Queue()

                # 启动读取线程
                stdout_thread = threading.Thread(
                    target=self._reader_thread, args=(process.stdout, stdout_q)
                )
                stderr_thread = threading.Thread(
                    target=self._reader_thread, args=(process.stderr, stderr_q)
                )

                stdout_thread.daemon = True
                stderr_thread.daemon = True
                stdout_thread.start()
                stderr_thread.start()
                
                # 7. 创建帧处理任务并准备处理
                logger.info("准备处理帧...")
                frame_params = []
                for frame_idx in range(total_frames):
                    img_start_y = frame_positions[frame_idx]
                    frame_params.append(
                        (
                            frame_idx,
                            img_start_y,
                            img_height,
                            img_width,
                            self.height,
                            self.width,
                            transparency_required,
                            bg_color_rgb,
                        )
                    )
                
                # 8. 使用进程池异步处理帧
                logger.info(f"启动进程池处理 {total_frames} 帧...")
                
                # 为进程池初始化器提供更详细的日志
                def init_worker_with_logging(shm_name, shape, dtype):
                    worker_pid = os.getpid()
                    logger.info(f"初始化工作进程 {worker_pid}...")
                    result = init_worker(shm_name, shape, dtype)
                    if result:
                        logger.info(f"工作进程 {worker_pid} 初始化成功")
                    else:
                        logger.error(f"工作进程 {worker_pid} 初始化失败")
                    return result
                
                # 创建带有共享内存初始化的进程池
                with mp.Pool(
                    processes=num_processes,
                    initializer=init_worker_with_logging,
                    initargs=(shm_name, shm_shape, shm_dtype),
                ) as pool:
                    # 创建进度条
                    with tqdm(total=total_frames, desc=f"编码 ({video_codec_params[1]})") as pbar:
                        # 创建任务结果缓冲区
                        frame_buffer = {}
                        next_frame_to_write = 0
                        frames_in_process = 0  # 跟踪处理中的帧数
                        max_buffer_size = 120  # 最大缓冲区大小
                        
                        # 分批提交任务，避免一次性提交过多导致内存压力
                        chunk_size = min(100, total_frames // num_processes)
                        if chunk_size < 2:
                            chunk_size = 2
                        
                        logger.info(f"使用批次大小: {chunk_size}")
                        
                        # 创建异步结果对象集合
                        pending_results = []
                        batch_start = 0
                        
                        # 首先提交初始批次的任务
                        initial_batch_size = min(chunk_size * num_processes * 2, total_frames)
                        logger.info(f"提交初始批次: {initial_batch_size} 帧")
                        
                        # 添加初始批次的任务
                        for i in range(0, initial_batch_size):
                            result = pool.apply_async(
                                _process_frame_optimized_shm, 
                                args=(frame_params[i],)
                            )
                            pending_results.append((i, result))
                            frames_in_process += 1
                        
                        batch_start = initial_batch_size
                        
                        while pending_results or batch_start < total_frames:
                            # 检查ffmpeg进程是否还在运行
                            if process.poll() is not None:
                                logger.error(f"ffmpeg进程意外退出，返回码: {process.returncode}")
                                # 收集并记录错误输出
                                stderr_data = []
                                while not stderr_q.empty():
                                    line = stderr_q.get_nowait()
                                    if line:
                                        stderr_data.append(line.decode(errors="ignore").strip())
                                if stderr_data:
                                    logger.error(f"ffmpeg错误输出: {stderr_data[-10:]}")
                                break
                                
                            # 处理已完成的任务结果
                            completed_indices = []
                            for idx, result in pending_results:
                                if result.ready():
                                    try:
                                        frame_idx, frame_data = result.get(timeout=0.1)
                                        completed_indices.append(idx)
                                        frames_in_process -= 1
                                        
                                        # 如果帧数据为None，表示处理失败，使用空帧
                                        if frame_data is None:
                                            logger.warning(f"帧 {frame_idx} 处理失败，使用空帧")
                                            if transparency_required:
                                                frame_data = np.zeros((self.height, self.width, 4), dtype=np.uint8)
                                            else:
                                                frame_data = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(bg_color_rgb, dtype=np.uint8)
                                        
                                        # 将帧数据存入缓冲区
                                        frame_buffer[frame_idx] = frame_data
                                        
                                    except Exception as e:
                                        logger.error(f"获取帧 {idx} 结果失败: {e}")
                                        completed_indices.append(idx)
                                        frames_in_process -= 1
                            
                            # 从待处理列表中移除已完成的任务
                            pending_results = [(idx, result) for idx, result in pending_results 
                                               if idx not in completed_indices]
                            
                            # 提交新任务，保持进程池饱和
                            while batch_start < total_frames and frames_in_process < num_processes * 3:
                                result = pool.apply_async(
                                    _process_frame_optimized_shm, 
                                    args=(frame_params[batch_start],)
                                )
                                pending_results.append((batch_start, result))
                                frames_in_process += 1
                                batch_start += 1
                                
                                # 避免提交太多任务
                                if batch_start % 100 == 0:
                                    logger.debug(f"已提交 {batch_start}/{total_frames} 帧")
                            
                            # 按顺序写入FFmpeg
                            while next_frame_to_write in frame_buffer:
                                # 获取要写入的帧
                                next_frame = frame_buffer.pop(next_frame_to_write)
                                # 写入FFmpeg
                                try:
                                    process.stdin.write(next_frame.tobytes())
                                    # 定期刷新防止阻塞
                                    if next_frame_to_write % 30 == 0:
                                        process.stdin.flush()
                                except BrokenPipeError:
                                    logger.error("FFmpeg管道已断开")
                                    break
                                except Exception as e:
                                    logger.error(f"写入帧 {next_frame_to_write} 失败: {e}")
                                    break
                                
                                # 更新进度条
                                pbar.update(1)
                                # 移动到下一帧
                                next_frame_to_write += 1
                            
                            # 防止缓冲区过大导致内存压力
                            if len(frame_buffer) > max_buffer_size:
                                # 强制垃圾回收
                                gc.collect()
                            
                            # 短暂睡眠，避免CPU满负荷
                            time.sleep(0.001)
                        
                        # 处理剩余缓冲区中的帧（如果有）
                        logger.info(f"处理剩余的 {len(frame_buffer)} 个帧...")
                        remaining_frames = sorted(frame_buffer.keys())
                        for frame_idx in remaining_frames:
                            try:
                                process.stdin.write(frame_buffer[frame_idx].tobytes())
                                pbar.update(1)
                            except:
                                logger.error(f"写入最终帧 {frame_idx} 失败")
                        
                        # 清空缓冲区
                        frame_buffer.clear()
                
                # 关闭FFmpeg的stdin
                try:
                    process.stdin.close()
                except:
                    logger.warning("关闭FFmpeg stdin失败")
                
                # 等待FFmpeg完成
                logger.info("等待FFmpeg进程完成...")
                try:
                    return_code = process.wait(timeout=60)  # 设置超时以避免永久阻塞
                    if return_code != 0:
                        logger.error(f"FFmpeg进程返回错误码: {return_code}")
                        # 收集并记录错误输出
                        stderr_data = []
                        while not stderr_q.empty():
                            line = stderr_q.get_nowait()
                            if line:
                                stderr_data.append(line.decode(errors="ignore").strip())
                        if stderr_data:
                            logger.error(f"FFmpeg错误输出: {stderr_data[-10:]}")
                        raise RuntimeError(f"FFmpeg进程失败，返回码: {return_code}")
                except subprocess.TimeoutExpired:
                    logger.error("等待FFmpeg进程超时，强制终止")
                    process.kill()
                    raise RuntimeError("FFmpeg进程超时，已终止")
                
                logger.info(f"视频渲染完成: {output_path}")
                return output_path
                
            finally:
                # 确保FFmpeg进程被正确关闭
                if 'process' in locals() and process.poll() is None:
                    logger.info("终止FFmpeg进程...")
                    try:
                        process.stdin.close()
                    except:
                        pass
                    try:
                        process.terminate()
                        process.wait(timeout=2)
                        if process.poll() is None:
                            logger.warning("FFmpeg进程没有及时终止，强制结束")
                            process.kill()
                    except:
                        logger.error("终止FFmpeg进程失败")
                    
        finally:
            # 清理共享内存
            logger.info("清理资源...")
            cleanup_shared_memory()
            
            # 强制垃圾回收
            gc.collect()
            
        return output_path

    def create_scrolling_video(
        self,
        image,
        output_path,
        text_actual_height=None,
        transparency_required=False,
        preferred_codec="h264_nvenc",
        audio_path=None,
        bg_color=(0, 0, 0, 255),
    ):
        """
        创建滚动视频

        Args:
            image: PIL图像对象
            output_path: 输出视频路径
            text_actual_height: 文本实际高度（不含额外填充）
            transparency_required: 是否需要透明背景
            preferred_codec: 首选视频编码器，默认尝试GPU加速
            audio_path: 可选的音频文件路径
            bg_color: 背景颜色，用于非透明视频

        Returns:
            输出视频的路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 获取图像尺寸
        img_width, img_height = image.size
        logger.info(f"图像尺寸: {img_width}x{img_height}")

        # 如果未提供文本实际高度，则使用图像高度
        if text_actual_height is None:
            text_actual_height = img_height
            logger.info("未提供文本实际高度，使用图像高度")

        # 计算总帧数
        # 1. 开始静止时间（秒）
        start_static_time = 3
        # 2. 结束静止时间（秒）
        end_static_time = 3
        # 3. 滚动所需帧数 = (图像高度 - 视频高度) / 每帧滚动像素数
        scroll_frames = max(0, (img_height - self.height)) / self.scroll_speed
        # 4. 总帧数 = 开始静止帧数 + 滚动帧数 + 结束静止帧数
        total_frames = int(
            (start_static_time * self.fps)
            + scroll_frames
            + (end_static_time * self.fps)
        )

        logger.info(
            f"视频参数: {self.width}x{self.height}, {self.fps}fps, 滚动速度: {self.scroll_speed}像素/帧"
        )
        logger.info(
            f"总帧数: {total_frames} (开始静止: {start_static_time}秒, 滚动: {scroll_frames/self.fps:.2f}秒, 结束静止: {end_static_time}秒)"
        )

        # 确定像素格式和编码器
        if transparency_required:
            # 透明视频使用ProRes 4444
            ffmpeg_pix_fmt = "rgba"
            output_path = os.path.splitext(output_path)[0] + ".mov"
            video_codec_params = [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4",  # ProRes 4444
                "-pix_fmt",
                "yuva444p10le",
                "-alpha_bits",
                "16",
                "-vendor",
                "ap10",
                "-threads",
                "8",  # 充分利用多核CPU
            ]
            logger.info("使用ProRes 4444编码器处理透明视频")
        else:
            # 不透明视频，尝试使用GPU加速
            ffmpeg_pix_fmt = "rgb24"
            output_path = os.path.splitext(output_path)[0] + ".mp4"

            # 检查操作系统，在macOS上默认使用CPU编码
            is_macos = platform.system() == "Darwin"
            if is_macos:
                # macOS上强制使用CPU编码
                logger.info("检测到macOS系统，将使用CPU编码器")
                os.environ["NO_GPU"] = "1"

            # 检查GPU编码器可用性
            gpu_encoders = ["h264_nvenc", "hevc_nvenc"]
            if preferred_codec in gpu_encoders and not "NO_GPU" in os.environ:
                # 使用更高性能的GPU参数
                video_codec_params = [
                    "-c:v",
                    preferred_codec,
                    "-preset",
                    "p4",  # 提升到p4预设（更高性能）
                    "-b:v",
                    "8M",  # 提升到8M比特率
                    "-pix_fmt",
                    "yuv420p",  # 确保兼容性
                    "-movflags",
                    "+faststart",
                ]
                logger.info(f"使用GPU编码器: {preferred_codec}，预设:p4，比特率:8M")
                use_gpu = True
            else:
                # 回退到CPU编码，但使用更高性能设置
                video_codec_params = [
                    "-c:v",
                    "libx264",
                    "-crf",
                    "21",  # 略微降低质量以提高速度
                    "-preset",
                    "medium",  # 保持medium预设
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-threads",
                    "8",  # 充分利用8核CPU
                ]
                logger.info(f"使用CPU编码器: libx264 (GPU编码器不可用或被禁用)")
                use_gpu = False

        # 预分配大型数组，减少内存碎片
        try:
            # 预热内存，减少动态分配开销
            if not transparency_required:
                # 为RGB预分配
                batch_size = 240 if not use_gpu else 120  # 增大批处理大小
                warmup_buffer = np.zeros(
                    (batch_size, self.height, self.width, 3), dtype=np.uint8
                )
                del warmup_buffer
            else:
                # 为RGBA预分配
                batch_size = 120  # 透明视频使用较小的批处理大小
                warmup_buffer = np.zeros(
                    (batch_size, self.height, self.width, 4), dtype=np.uint8
                )
                del warmup_buffer

            # 强制垃圾回收
            gc.collect()
            logger.info("内存预热完成")
        except Exception as e:
            logger.warning(f"内存预热失败: {e}")
            batch_size = 60  # 回退到较小的批处理大小

        # 数据传输模式：直接模式或缓存模式
        # 根据GPU/CPU模式调整批处理大小
        if not "batch_size" in locals():
            batch_size = 60  # 降低批处理大小
        num_batches = (total_frames + batch_size - 1) // batch_size

        # 确定最佳进程数
        try:
            cpu_count = mp.cpu_count()
            # 减少使用的核心数，为系统和ffmpeg留下更多资源
            optimal_processes = min(8, max(2, cpu_count - 1))
            num_processes = optimal_processes
            logger.info(
                f"检测到{cpu_count}个CPU核心，优化使用{num_processes}个进程进行渲染，批处理大小:{batch_size}"
            )
        except:
            num_processes = 6  # 默认使用较少进程
            logger.info(f"使用默认{num_processes}个进程，批处理大小:{batch_size}")

        # 初始化内存池 - 减小池大小以降低内存压力
        channels = 4 if transparency_required else 3
        self._init_memory_pool(channels, pool_size=720)  # 增加内存池大小提高性能

        # 将图像转换为numpy数组
        img_array = np.array(image)

        # 完整的ffmpeg命令
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            # I/O优化参数
            "-probesize",
            "32M",  # 增加探测缓冲区大小
            "-analyzeduration",
            "32M",  # 增加分析时间
            "-thread_queue_size",
            "4096",  # 大幅增加线程队列大小
            # 输入格式参数
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            ffmpeg_pix_fmt,
            "-r",
            str(self.fps),
            "-i",
            "-",  # 从stdin读取
        ]

        # 添加音频输入（如果有）
        if audio_path and os.path.exists(audio_path):
            ffmpeg_cmd.extend(["-i", audio_path])

        # 添加视频编码参数
        ffmpeg_cmd.extend(video_codec_params)

        # 添加音频映射（如果有）
        if audio_path and os.path.exists(audio_path):
            ffmpeg_cmd.extend(
                [
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-shortest",
                ]
            )
        else:
            ffmpeg_cmd.extend(["-map", "0:v:0"])

        # 添加输出路径
        ffmpeg_cmd.append(output_path)

        logger.info(f"FFmpeg命令: {' '.join(ffmpeg_cmd)}")

        # 启动FFmpeg进程
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,  # 增大缓冲区
        )

        # 创建进度条
        pbar = tqdm(total=total_frames, desc="渲染进度")

        # 创建多进程池
        with mp.Pool(processes=num_processes) as pool:
            # 设置全局图像数组
            global _g_img_array
            _g_img_array = img_array

            # 帧计数器
            self.frame_counter = 0
            start_time = time.time()

            try:
                # 处理每个批次
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, total_frames)
                    batch_frames = []

                    # 准备批次帧参数
                    for frame_idx in range(batch_start, batch_end):
                        # 计算当前帧对应的图像Y坐标
                        if frame_idx < start_static_time * self.fps:
                            # 开始静止阶段
                            img_start_y = 0
                        elif frame_idx >= total_frames - end_static_time * self.fps:
                            # 结束静止阶段
                            img_start_y = max(0, img_height - self.height)
                        else:
                            # 滚动阶段
                            scroll_frame_idx = frame_idx - start_static_time * self.fps
                            img_start_y = min(
                                img_height - self.height,
                                int(scroll_frame_idx * self.scroll_speed),
                            )

                        # 添加到批次
                        batch_frames.append(
                            (
                                frame_idx,
                                img_start_y,
                                img_height,
                                img_width,
                                self.height,
                                self.width,
                                transparency_required,
                                bg_color[:3],  # 只传递RGB部分
                            )
                        )

                    # 处理当前批次
                    if len(batch_frames) > 60 and num_processes > 1:
                        # 大批处理：并行处理所有帧
                        pool_results = pool.map(_process_frame_optimized, batch_frames)
                        processed_frames = sorted(pool_results, key=lambda x: x[0])
                    else:
                        # 小批处理：使用fast_frame_processor直接处理
                        try:
                            frames_processed = fast_frame_processor(
                                batch_frames, self.memory_pool, process
                            )
                            self.frame_counter += frames_processed
                            pbar.update(frames_processed)
                            continue  # 跳过后续处理，因为帧已经直接写入
                        except Exception as e:
                            logger.error(f"快速帧处理器失败: {e}，回退到标准处理")
                            # 回退到标准处理
                            processed_frames = []
                            for params in batch_frames:
                                frame_idx, frame = _process_frame_optimized(params)
                                processed_frames.append((frame_idx, frame))

                    # 将处理后的帧写入FFmpeg
                    for _, frame in processed_frames:
                        process.stdin.write(frame.tobytes())
                        self.frame_counter += 1
                        pbar.update(1)

                # 关闭stdin，等待FFmpeg完成
                process.stdin.close()
                process.wait()

                # 检查FFmpeg是否成功
                if process.returncode != 0:
                    stderr = process.stderr.read().decode("utf-8", errors="ignore")
                    logger.error(f"FFmpeg错误: {stderr}")
                    raise Exception(f"FFmpeg处理失败，返回码: {process.returncode}")

                # 计算性能统计
                end_time = time.time()
                total_time = end_time - start_time
                fps = self.frame_counter / total_time if total_time > 0 else 0
                logger.info(
                    f"总渲染性能: 渲染了{self.frame_counter}帧，耗时{total_time:.2f}秒，平均{fps:.2f}帧/秒"
                )

                return output_path

            except Exception as e:
                logger.error(f"视频渲染失败: {str(e)}", exc_info=True)
                # 尝试终止FFmpeg进程
                try:
                    process.terminate()
                except:
                    pass
                raise e

            finally:
                # 清理
                pbar.close()
                _g_img_array = None
                gc.collect()  # 强制垃圾回收

            return output_path

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
                        bg_color=bg_color,
                    )
                finally:
                    # 恢复环境变量
                    if "NO_GPU" in os.environ:
                        del os.environ["NO_GPU"]

        return output_path

    def create_scrolling_video_optimized(self, text_image_path, output_path, 
                                      duration=None, audio_path=None, 
                                      scroll_speed=None, fps=None, width=None, height=None):
        """
        创建滚动文本视频 (优化版)
        
        使用共享内存和多进程优化处理较长文本的滚动视频生成。
        
        Args:
            text_image_path: 文本图像路径
            output_path: 输出视频路径
            duration: 视频持续时间(秒)
            audio_path: 音频文件路径(可选)
            scroll_speed: 滚动速度(每秒像素)
            fps: 帧率
            width: 视频宽度
            height: 视频高度
            
        Returns:
            bool: 成功返回True，否则返回False
        """
        start_time = time.time()
        logger.info(f"开始生成优化版滚动视频: {output_path}")
        
        try:
            # 性能监控
            self.performance_monitor.start_task("create_scrolling_video_optimized")
            self.performance_monitor.log_memory_usage("开始")
            
            # 设置参数
            fps = fps or self.default_fps
            width = width or self.default_width
            height = height or self.default_height
            
            with time_tracker("图像预处理"):
                # 1. 加载图像并进行预处理
                try:
                    img = Image.open(text_image_path)
                    
                    # 检查图像尺寸
                    img_width, img_height = img.size
                    logger.info(f"文本图像尺寸: {img_width}x{img_height}")
                    
                    # 处理透明图像
                    if img.mode == 'RGBA':
                        # 创建白色背景
                        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                        # 将原图合成到背景上
                        img = Image.alpha_composite(background, img)
                    
                    # 转换为RGB模式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 如果图像宽度不等于视频宽度，进行调整
                    if img_width != width:
                        new_height = int(img_height * (width / img_width))
                        img = img.resize((width, new_height), Image.LANCZOS)
                        img_width, img_height = width, new_height
                        logger.info(f"调整后图像尺寸: {img_width}x{img_height}")
                    
                    # 转换为numpy数组以便处理
                    img_array = np.array(img)
                    
                    # 优化: 确保图像数据是连续的
                    if not img_array.flags.c_contiguous:
                        img_array = np.ascontiguousarray(img_array)
                    
                    self.performance_monitor.log_memory_usage("图像加载后")
                except Exception as e:
                    logger.error(f"图像加载失败: {str(e)}")
                    return False
            
            # 2. 计算滚动参数
            with time_tracker("计算滚动参数"):
                # 计算滚动持续时间
                if duration is None:
                    # 根据文本量自动计算合理的持续时间
                    # 短文本至少有5秒钟的阅读时间
                    min_scroll_time = 5  # 最小滚动时间(秒)
                    reading_speed = 100  # 每分钟阅读字符数
                    
                    # 估算文本字符数 (每6像素1个字符)
                    estimated_chars = (img_height * img_width) / (6 * 12)
                    reading_time = (estimated_chars / reading_speed) * 60
                    
                    # 确保至少有最小滚动时间
                    duration = max(min_scroll_time, reading_time)
                    
                    logger.info(f"自动计算的视频持续时间: {duration:.2f}秒 (估计{estimated_chars:.0f}字符)")
                
                # 如果没有提供滚动速度，根据图像高度和持续时间计算
                if scroll_speed is None:
                    total_scroll_pixels = img_height - height
                    if total_scroll_pixels > 0:
                        scroll_speed = total_scroll_pixels / duration
                    else:
                        scroll_speed = 0  # 图像高度小于视频高度，不需要滚动
                    
                    logger.info(f"计算的滚动速度: {scroll_speed:.2f}像素/秒")
                
                # 计算总帧数
                total_frames = int(duration * fps)
                frames_info = {"total": total_frames, "processed": 0}
                
                logger.info(f"滚动参数: 速度={scroll_speed:.2f}像素/秒, 总帧数={total_frames}")
                
                # 短视频检查 - 如果文本太短，不需要滚动
                if img_height <= height:
                    logger.info("文本图像较短，生成静态视频")
                    # 居中显示图像
                    y_offset = (height - img_height) // 2
                    # 创建背景
                    background = np.ones((height, width, 3), dtype=np.uint8) * 255
                    # 放置图像
                    background[y_offset:y_offset+img_height, :, :] = img_array
                    # 生成静态视频
                    return self._create_static_video(background, output_path, duration, audio_path, fps)
            
            # 3. 检查编码器和设置
            with time_tracker("检查编码器"):
                # 选择适合的编码器
                encoder = 'libx264'  # 默认
                
                # 尝试使用GPU加速
                for enc in self.available_encoders:
                    if 'nvenc' in enc or 'qsv' in enc or 'videotoolbox' in enc or 'amf' in enc:
                        encoder = enc
                        logger.info(f"使用GPU加速编码器: {encoder}")
                        break
                
                # 编码器特定参数
                encoder_params = []
                if encoder == 'libx264':
                    # CPU编码设置
                    encoder_params.extend(['-preset', 'fast', '-crf', '23'])
                elif 'nvenc' in encoder:
                    # NVIDIA设置
                    encoder_params.extend(['-preset', 'p4', '-rc', 'vbr', '-b:v', '5M'])
                elif 'videotoolbox' in encoder:
                    # macOS设置
                    encoder_params.extend(['-b:v', '5M', '-allow_sw', '1'])
                else:
                    # 其他GPU编码器通用设置
                    encoder_params.extend(['-b:v', '5M'])
            
            # 4. 设置共享内存和多进程
            with time_tracker("配置共享内存"):
                frame_meta = {
                    'width': width,
                    'height': height,
                    'img_height': img_height,
                    'img_width': img_width,
                    'fps': fps,
                    'duration': duration,
                    'scroll_speed': scroll_speed,
                    'total_frames': total_frames
                }
                
                # 创建共享内存对象
                try:
                    # 创建唯一命名的共享内存段
                    shm_name = f"img_data_{time.time()}_{random.randint(1000, 9999)}"
                    img_size = img_array.nbytes
                    img_shape = img_array.shape
                    
                    # 初始化共享内存
                    logger.info(f"创建共享内存段: {shm_name}, 大小: {img_size/1024/1024:.2f}MB")
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=img_size)
                    except Exception as e:
                        logger.error(f"创建共享内存失败: {str(e)}")
                        emergency_cleanup()  # 清理资源
                        
                        # 第二次尝试，先删除同名共享内存
                        try:
                            cleanup_shared_memory(shm_name)
                            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=img_size)
                            logger.info("第二次尝试创建共享内存成功")
                        except Exception as e2:
                            logger.error(f"第二次尝试创建共享内存失败: {str(e2)}")
                            raise
                    
                    # 将图像数据复制到共享内存
                    shared_img = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)
                    np.copyto(shared_img, img_array)
                    
                    # 设置进程间共享的字典
                    shared_dict = {
                        'shm_name': shm_name,
                        'img_shape': img_shape,
                        'frame_meta': frame_meta,
                    }
                    
                    # 初始化共享内存管理
                    init_shared_memory(shared_dict)
                    
                    # 测试共享内存是否可以访问
                    logger.info("测试共享内存访问...")
                    test_result = test_worker_shared_memory(shared_dict)
                    if not test_result:
                        logger.error("共享内存测试失败，无法继续处理")
                        cleanup_shared_memory(shm_name)
                        shm.close()
                        shm.unlink()
                        return False
                    else:
                        logger.info("共享内存测试成功!")
                    
                    self.performance_monitor.log_memory_usage("共享内存设置后")
                except Exception as e:
                    logger.error(f"设置共享内存失败: {str(e)}\n{traceback.format_exc()}")
                    emergency_cleanup()
                    return False
            
            # 5. 准备FFMPEG命令
            with time_tracker("配置FFMPEG"):
                # 创建临时管道路径
                temp_dir = os.path.dirname(output_path)
                os.makedirs(temp_dir, exist_ok=True)
                pipe_path = os.path.join(temp_dir, f"videopipe_{int(time.time())}.yuv")
                
                # 设置FFmpeg命令
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # 覆盖输出文件
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{width}x{height}',
                    '-pix_fmt', 'rgb24',
                    '-r', str(fps),
                    '-i', pipe_path,
                    '-an'  # 无音频
                ]
                
                # 添加编码器特定参数
                ffmpeg_cmd.extend(['-c:v', encoder])
                ffmpeg_cmd.extend(encoder_params)
                
                # 如果有音频文件，添加音频
                if audio_path and os.path.exists(audio_path):
                    ffmpeg_cmd.extend([
                        '-i', audio_path,
                        '-c:a', 'aac',
                        '-shortest'  # 以最短的输入流长度为准
                    ])
                
                # 添加输出文件和日志级别
                ffmpeg_cmd.extend([
                    '-loglevel', 'warning',
                    output_path
                ])
                
                logger.info(f"FFMPEG命令: {' '.join(ffmpeg_cmd)}")
            
            # 6. 创建管道并启动FFMPEG
            with time_tracker("设置渲染管道"):
                try:
                    # 创建命名管道
                    if os.path.exists(pipe_path):
                        os.unlink(pipe_path)
                    os.mkfifo(pipe_path)
                    
                    # 启动FFMPEG进程
                    ffmpeg_process = subprocess.Popen(
                        ffmpeg_cmd, 
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=10**8  # 增大缓冲区
                    )
                    
                    # 打开管道
                    pipe_fd = os.open(pipe_path, os.O_WRONLY)
                    
                    logger.info(f"FFMPEG进程已启动，PID: {ffmpeg_process.pid}")
                except Exception as e:
                    logger.error(f"设置FFMPEG进程失败: {str(e)}")
                    if 'shm' in locals():
                        cleanup_shared_memory(shm_name)
                        shm.close()
                        shm.unlink()
                    emergency_cleanup()
                    return False
            
            # 7. 帧生成和处理
            with time_tracker("生成和处理帧"):
                try:
                    # 使用共享内存帧池
                    frame_pool = SharedMemoryFramePool(total_frames, width, height)
                    frame_buffer = FrameBuffer(total_frames, buffer_size=min(100, total_frames // 4))
                    
                    # 进度报告函数
                    def report_progress():
                        last_report_time = time.time()
                        last_frames = 0
                        
                        while frames_info["processed"] < total_frames:
                            time.sleep(1)
                            current_time = time.time()
                            current_frames = frames_info["processed"]
                            
                            if current_time - last_report_time >= 5:  # 每5秒报告一次
                                frames_delta = current_frames - last_frames
                                time_delta = current_time - last_report_time
                                fps_rate = frames_delta / time_delta if time_delta > 0 else 0
                                
                                percent_done = (current_frames / total_frames) * 100
                                
                                logger.info(f"进度: {percent_done:.1f}% ({current_frames}/{total_frames}), "
                                          f"速率: {fps_rate:.1f} 帧/秒")
                                
                                # 更新上次报告值
                                last_report_time = current_time
                                last_frames = current_frames
                                
                                # 检查内存使用
                                if current_frames % 1000 == 0:
                                    self.performance_monitor.log_memory_usage(f"处理{current_frames}帧后")
                    
                    # 启动进度报告线程
                    progress_thread = threading.Thread(target=report_progress)
                    progress_thread.daemon = True
                    progress_thread.start()
                    
                    # 防止卡死处理
                    def watchdog_handler():
                        logger.error("处理超时，强制终止")
                        emergency_cleanup()
                        os._exit(1)  # 强制退出进程
                    
                    # 设置60分钟超时
                    watchdog_timer = threading.Timer(60 * 60, watchdog_handler)
                    watchdog_timer.daemon = True
                    watchdog_timer.start()
                    
                    logger.info(f"开始处理 {total_frames} 帧...")
                    
                    # 初始化并启动进程池
                    with mp.Pool(
                        processes=self.optimal_processes,
                        initializer=init_worker,
                        initargs=(shared_dict,)
                    ) as pool:
                        # 确保所有工作进程都能访问共享内存
                        logger.info(f"启动 {self.optimal_processes} 个工作进程...")
                        
                        # 确认工作进程初始化成功
                        test_tasks = [pool.apply_async(test_worker_shared_memory, args=(shared_dict,)) 
                                     for _ in range(self.optimal_processes)]
                        
                        # 等待并验证所有测试任务
                        failed_tests = 0
                        for i, task in enumerate(test_tasks):
                            try:
                                result = task.get(timeout=10)
                                if not result:
                                    failed_tests += 1
                                    logger.error(f"工作进程 {i} 共享内存测试失败")
                            except Exception as e:
                                failed_tests += 1
                                logger.error(f"工作进程 {i} 测试异常: {str(e)}")
                        
                        if failed_tests > 0:
                            logger.error(f"{failed_tests}/{self.optimal_processes} 个工作进程测试失败")
                            raise RuntimeError("工作进程初始化失败")
                        else:
                            logger.info("所有工作进程成功初始化")
                        
                        # 提交所有帧处理任务
                        frame_args = [(i, frame_meta) for i in range(total_frames)]
                        
                        # 使用imap_unordered获取结果，可以提高处理速度
                        for processed_frame in pool.imap_unordered(_process_frame_optimized_shm, frame_args):
                            if processed_frame:
                                frame_idx, frame_data = processed_frame
                                frame_buffer.add_frame(frame_idx, frame_data)
                                frames_info["processed"] += 1
                                
                                if frames_info["processed"] % 100 == 0:
                                    # 每处理100帧输出一次内存使用情况
                                    if hasattr(psutil, 'Process'):
                                        try:
                                            process = psutil.Process()
                                            mem_info = process.memory_info()
                                            logger.debug(f"内存使用: {mem_info.rss / (1024*1024):.1f}MB, "
                                                       f"处理进度: {frames_info['processed']}/{total_frames}")
                                        except:
                                            pass
                        
                        logger.info(f"所有 {total_frames} 帧任务已提交完成")
                    
                    # 取消超时
                    watchdog_timer.cancel()
                    
                    # 开始按顺序写入帧
                    logger.info("开始写入帧到视频...")
                    
                    for frame_idx in range(total_frames):
                        try:
                            frame_data = frame_buffer.get_frame(frame_idx)
                            if frame_data is not None:
                                # 写入到管道
                                os.write(pipe_fd, frame_data.tobytes())
                            else:
                                # 如果帧丢失，写入黑色帧
                                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                                os.write(pipe_fd, black_frame.tobytes())
                                logger.warning(f"帧 {frame_idx} 缺失，使用黑色帧替代")
                        except Exception as e:
                            logger.error(f"写入帧 {frame_idx} 失败: {str(e)}")
                    
                    logger.info("所有帧已写入")
                    
                    # 关闭管道
                    os.close(pipe_fd)
                    
                    # 等待FFmpeg完成
                    stdout, stderr = ffmpeg_process.communicate()
                    
                    if ffmpeg_process.returncode != 0:
                        logger.error(f"FFMPEG错误: {stderr.decode('utf-8', errors='ignore')}")
                        result = False
                    else:
                        logger.info(f"FFMPEG成功完成编码")
                        result = True
                    
                    # 清理共享内存
                    cleanup_shared_memory(shm_name)
                    shm.close()
                    shm.unlink()
                    
                    # 删除管道
                    if os.path.exists(pipe_path):
                        os.unlink(pipe_path)
                        
                    # 计算处理时间
                    total_time = time.time() - start_time
                    logger.info(f"视频生成完成。总耗时: {total_time:.2f}秒, "
                              f"平均每帧: {(total_time*1000)/total_frames:.2f}毫秒")
                    
                    # 清理内存
                    frame_pool.cleanup()
                    frame_buffer.cleanup()
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"帧处理失败: {str(e)}\n{traceback.format_exc()}")
                    
                    # 清理资源
                    try:
                        if 'pipe_fd' in locals():
                            os.close(pipe_fd)
                        if 'pipe_path' in locals() and os.path.exists(pipe_path):
                            os.unlink(pipe_path)
                        if 'shm' in locals():
                            cleanup_shared_memory(shm_name)
                            shm.close()
                            shm.unlink()
                        if 'ffmpeg_process' in locals() and ffmpeg_process.poll() is None:
                            ffmpeg_process.terminate()
                    except Exception as cleanup_error:
                        logger.error(f"清理资源失败: {str(cleanup_error)}")
                    
                    emergency_cleanup()
                    return False
                
        except Exception as e:
            logger.error(f"视频生成过程中发生错误: {str(e)}\n{traceback.format_exc()}")
            emergency_cleanup()
            return False
        finally:
            # 记录性能数据
            self.performance_monitor.end_task("create_scrolling_video_optimized")
            self.performance_monitor.log_memory_usage("结束")
            
            # 最终强制清理
            optimize_memory()
            logger.info("视频渲染过程结束")
    
    def _create_static_video(self, image_array, output_path, duration, audio_path=None, fps=30):
        """创建静态视频（用于短文本）"""
        try:
            height, width = image_array.shape[:2]
            total_frames = int(duration * fps)
            
            # 设置FFmpeg命令
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(fps),
                '-i', '-',  # 从标准输入读取
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23'
            ]
            
            # 添加音频（如果有）
            if audio_path and os.path.exists(audio_path):
                ffmpeg_cmd.extend([
                    '-i', audio_path,
                    '-c:a', 'aac',
                    '-shortest'
                ])
            
            ffmpeg_cmd.extend([output_path])
            
            # 启动FFmpeg进程
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 写入相同的帧
            for _ in range(total_frames):
                process.stdin.write(image_array.tobytes())
            
            # 关闭输入并等待完成
            process.stdin.close()
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"静态视频创建失败: {stderr.decode('utf-8', errors='ignore')}")
                return False
            
            logger.info(f"静态视频创建成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建静态视频失败: {str(e)}")
            return False
