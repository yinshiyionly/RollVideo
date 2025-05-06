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

        # 创建输出目录
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 删除旧的输出文件
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"已删除旧的输出文件: {output_path}")
            except Exception as e:
                logger.warning(f"删除旧输出文件失败: {e}")

        # 3. 确定最适合的处理模式
        is_macos = platform.system() == "Darwin"
        is_windows = platform.system() == "Windows"
        is_linux = platform.system() == "Linux"

        # 根据平台选择最佳编码器
        final_codec = preferred_codec
        
        # 检查是否强制关闭GPU
        if "NO_GPU" in os.environ:
            if preferred_codec.endswith("_nvenc") or preferred_codec.endswith("_qsv"):
                final_codec = "libx264"
                logger.info("环境变量禁用GPU，切换到CPU编码器")
                
        # 编码器和输出参数
        if transparency_required:
            # 透明背景需要特殊处理
            pix_fmt = "rgba"
            if preferred_codec == "prores_ks":
                # ProRes 4444保留Alpha
                codec_params = [
                    "-c:v", "prores_ks", 
                    "-profile:v", "4444",
                    "-pix_fmt", "yuva444p10le", 
                    "-alpha_bits", "16",
                    "-vendor", "ap10", 
                    "-colorspace", "bt709",
                ]
                # .mov容器
                if not output_path.lower().endswith(".mov"):
                    output_path = os.path.splitext(output_path)[0] + ".mov"
                    logger.info(f"透明视频必须使用.mov格式，已修改路径: {output_path}")
            else:
                # 无法使用其他编码器处理透明度，强制使用ProRes
                logger.warning(f"透明视频不支持编码器 {preferred_codec}，强制使用ProRes")
                codec_params = [
                    "-c:v", "prores_ks", 
                    "-profile:v", "4444",
                    "-pix_fmt", "yuva444p10le", 
                    "-alpha_bits", "16",
                    "-vendor", "ap10", 
                    "-colorspace", "bt709",
                ]
                # .mov容器
                if not output_path.lower().endswith(".mov"):
                    output_path = os.path.splitext(output_path)[0] + ".mov"
                    logger.info(f"透明视频必须使用.mov格式，已修改路径: {output_path}")
        else:
            # 非透明视频，尝试优化编码器
            pix_fmt = "rgb24"
            
            # 根据平台和编码器选择参数
            if preferred_codec == "h264_nvenc":
                # NVIDIA GPU加速
                if is_windows or is_linux:
                    codec_params = [
                        "-c:v", "h264_nvenc",
                        "-preset", "p4",
                        "-rc", "vbr",
                        "-cq", "23",
                        "-b:v", "5M",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                    ]
                else:
                    # 不支持NVIDIA，回退到CPU
                    logger.info("平台不支持NVIDIA编码，切换到libx264")
                    codec_params = [
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                    ]
            elif preferred_codec == "h264_qsv":
                # Intel QuickSync加速
                if is_windows or is_linux:
                    codec_params = [
                        "-c:v", "h264_qsv",
                        "-preset", "medium",
                        "-b:v", "5M",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                    ]
                else:
                    # 不支持QSV，回退到CPU
                    logger.info("平台不支持Intel QuickSync，切换到libx264")
                    codec_params = [
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                    ]
            elif preferred_codec == "h264_videotoolbox" and is_macos:
                # macOS VideoToolbox加速
                codec_params = [
                    "-c:v", "h264_videotoolbox",
                    "-b:v", "5M",
                    "-allow_sw", "1",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                ]
            elif preferred_codec == "prores_ks":
                # ProRes (非透明)
                codec_params = [
                    "-c:v", "prores_ks",
                    "-profile:v", "3",  # ProRes 422 HQ
                    "-pix_fmt", "yuv422p10le",
                    "-vendor", "ap10",
                    "-colorspace", "bt709",
                ]
                # .mov容器
                if not output_path.lower().endswith(".mov"):
                    output_path = os.path.splitext(output_path)[0] + ".mov"
            else:
                # 默认使用libx264 (高质量CPU编码)
                codec_params = [
                    "-c:v", "libx264",
                    "-preset", "medium",  # 平衡速度和质量的预设
                    "-crf", "23",         # 恒定质量因子 (0-51, 越低质量越高)
                    "-pix_fmt", "yuv420p", # 兼容大多数播放器
                    "-movflags", "+faststart", # MP4优化
                ]
        
        # 组合核心参数
        ffmpeg_cmd = self._get_ffmpeg_command(
            output_path=output_path,
            pix_fmt=pix_fmt,
            codec_and_output_params=codec_params,
            audio_path=audio_path,
        )
        
        # 记录实际命令行
        cmd_str = " ".join(ffmpeg_cmd)
        logger.info(f"FFmpeg命令: {cmd_str}")

        # 4. 共享内存初始化
        logger.info(f"初始化共享内存并上传图像数据 ({img_width}x{img_height}x{channels})...")
        self._init_memory_pool(channels=channels, pool_size=min(30, total_frames))

        shm = None
        shm_name = None
        mp_context = mp.get_context("spawn")  # 使用spawn避免fork问题

        try:
            # 创建共享内存
            try:
                shm_name = f"shm_image_{int(time.time())}_{random.randint(1000, 9999)}"
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=img_array.nbytes)
                
                # 创建Numpy数组视图并复制数据
                shm_array = np.ndarray(img_array.shape, dtype=img_array.dtype, buffer=shm.buf)
                np.copyto(shm_array, img_array)
                
                logger.info(f"已将图像数据复制到共享内存 {shm_name}")
                
                # 储存共享内存信息
                shared_dict = {
                    'shm_name': shm_name,
                    'img_shape': img_array.shape,
                    'dtype': img_array.dtype.name,
                }
                
                # 初始化本进程的共享内存
                init_shared_memory(shared_dict)
                
                # 测试共享内存访问
                test_result = test_worker_shared_memory(shared_dict)
                if not test_result:
                    logger.warning("主进程中测试共享内存失败，尝试重建...")
                    # 清理并重建
                    cleanup_shared_memory(shm_name)
                    time.sleep(0.5)
                    
                    # 重建共享内存
                    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=img_array.nbytes)
                    shm_array = np.ndarray(img_array.shape, dtype=img_array.dtype, buffer=shm.buf)
                    np.copyto(shm_array, img_array)
                    
                    # 更新共享内存信息
                    shared_dict = {
                        'shm_name': shm_name,
                        'img_shape': img_array.shape,
                        'dtype': img_array.dtype.name,
                    }
                    
                    # 再次初始化
                    init_shared_memory(shared_dict)
                    logger.info("共享内存已重建")
                    
            except Exception as e:
                logger.error(f"创建共享内存失败: {str(e)}")
                # 尝试清理后重试一次
                try:
                    if shm_name:
                        cleanup_shared_memory(shm_name)
                except:
                    pass
                
                # 重新生成随机名称避免冲突
                shm_name = f"shm_retry_{int(time.time())}_{random.randint(1000, 9999)}"
                time.sleep(0.5)  # 等待一下以确保清理完成
                
                try:
                    # 第二次尝试创建共享内存
                    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=img_array.nbytes)
                    shm_array = np.ndarray(img_array.shape, dtype=img_array.dtype, buffer=shm.buf)
                    np.copyto(shm_array, img_array)
                    
                    # 储存共享内存信息
                    shared_dict = {
                        'shm_name': shm_name,
                        'img_shape': img_array.shape,
                        'dtype': img_array.dtype.name,
                    }
                    
                    # 再次初始化
                    init_shared_memory(shared_dict)
                    logger.info("第二次尝试创建共享内存成功")
                    
                except Exception as e2:
                    logger.error(f"第二次尝试创建共享内存也失败: {str(e2)}")
                    # 如果共享内存无法使用，回退到非共享内存方法
                    logger.info("共享内存无法使用，回退到标准CPU处理")
                    return self.create_scrolling_video(
                        image=image,
                        output_path=output_path,
                        text_actual_height=text_actual_height,
                        transparency_required=transparency_required,
                        preferred_codec=preferred_codec,
                        audio_path=audio_path,
                        bg_color=bg_color
                    )

            # 5. 创建子进程池
            # 核心数和池大小计算
            cpu_count = mp.cpu_count()
            pool_size = max(2, min(cpu_count - 1, 8))  # 至少2个，最多8个，保留1个核心给主进程

            logger.info(f"创建{pool_size}个进程的进程池（共享内存：{shm_name}）")
            
            # 创建管道和队列
            read_stdout, write_stdout = os.pipe()
            read_stderr, write_stderr = os.pipe()
            stdout_queue = queue.Queue()
            stderr_queue = queue.Queue()

            # 创建进程池和帧生成器
            try:
                # 创建进程池（使用spawn确保共享内存兼容性）
                with mp_context.Pool(processes=pool_size, initializer=init_worker, initargs=(shared_dict,)) as pool:
                    # 启动FFmpeg进程
                    proc = subprocess.Popen(
                        ffmpeg_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=10 * 1024 * 1024,  # 大缓冲区
                    )

                    # 启动读取线程
                    stdout_thread = threading.Thread(
                        target=self._reader_thread, args=(proc.stdout, stdout_queue)
                    )
                    stderr_thread = threading.Thread(
                        target=self._reader_thread, args=(proc.stderr, stderr_queue)
                    )
                    stdout_thread.daemon = True
                    stderr_thread.daemon = True
                    stdout_thread.start()
                    stderr_thread.start()

                    # 创建帧任务列表
                    frame_tasks = []
                    
                    # 记录开始时间（供进度报告使用）
                    processing_start_time = time.time()
                    
                    # 前面的静止帧
                    for i in range(padding_frames_start):
                        frame_meta = {
                            'width': self.width,
                            'height': self.height,
                            'img_height': img_height,
                            'scroll_speed': 0,  # 静止不滚动
                            'fps': self.fps,
                        }
                        frame_tasks.append((i, frame_meta))
                    
                    # 滚动帧
                    for i in range(scroll_frames):
                        # 计算滚动偏移量
                        frame_idx = padding_frames_start + i
                        frame_meta = {
                            'width': self.width,
                            'height': self.height,
                            'img_height': img_height,
                            'scroll_speed': self.scroll_speed,
                            'fps': self.fps,
                        }
                        frame_tasks.append((frame_idx, frame_meta))
                    
                    # 后面的静止帧
                    for i in range(padding_frames_end):
                        frame_idx = padding_frames_start + scroll_frames + i
                        frame_meta = {
                            'width': self.width,
                            'height': self.height,
                            'img_height': img_height,
                            'scroll_speed': 0,  # 静止不滚动
                            'fps': self.fps,
                        }
                        frame_tasks.append((frame_idx, frame_meta))

                    # 异步处理帧
                    logger.info(f"开始处理{len(frame_tasks)}帧...")
                    
                    # 进度报告函数
                    last_report_time = time.time()
                    frames_processed = 0
                    
                    # 设置看门狗定时器
                    watchdog_event = threading.Event()
                    
                    def report_progress():
                        nonlocal last_report_time, frames_processed, processing_start_time
                        current_time = time.time()
                        if current_time - last_report_time >= 2.0:  # 每2秒更新一次
                            elapsed = current_time - processing_start_time
                            fps = frames_processed / elapsed if elapsed > 0 else 0
                            percent = 100.0 * frames_processed / total_frames if total_frames > 0 else 0
                            
                            # 估计剩余时间
                            if fps > 0:
                                remaining_frames = total_frames - frames_processed
                                eta = remaining_frames / fps
                                eta_str = f", 预计剩余: {eta:.1f}秒" if eta > 0 else ""
                            else:
                                eta_str = ""
                                
                            logger.info(
                                f"进度: {frames_processed}/{total_frames} 帧 "
                                f"({percent:.1f}%, {fps:.1f} fps{eta_str})"
                            )
                            last_report_time = current_time
                            
                            # 重置看门狗
                            watchdog_event.set()
                    
                    # 看门狗线程函数
                    def watchdog_handler():
                        """监视进程进度，如果超过30秒没有进展，则终止进程"""
                        watchdog_timeout = 30.0  # 30秒超时
                        last_check_frames = 0
                        last_check_time = time.time()
                        
                        while True:
                            # 等待看门狗重置或超时
                            if watchdog_event.wait(watchdog_timeout / 2):
                                # 事件被设置，重置看门狗
                                watchdog_event.clear()
                                last_check_frames = frames_processed
                                last_check_time = time.time()
                            else:
                                # 检查是否有进展
                                if frames_processed == last_check_frames:
                                    current_time = time.time()
                                    if current_time - last_check_time > watchdog_timeout:
                                        logger.error(
                                            f"看门狗检测到处理卡住: {watchdog_timeout}秒内没有进度!"
                                            f"最后处理: {last_check_frames}/{total_frames}帧"
                                        )
                                        # 尝试停止处理
                                        try:
                                            proc.terminate()
                                        except:
                                            pass
                                        return  # 停止看门狗
                                else:
                                    # 有进展，重置
                                    last_check_frames = frames_processed
                                    last_check_time = time.time()
                    
                    # 启动看门狗线程
                    watchdog = threading.Thread(target=watchdog_handler)
                    watchdog.daemon = True
                    watchdog.start()
                    
                    # 初始化看门狗
                    watchdog_event.set()

                    # 5. 分批处理并流式输出到FFMPEG
                    chunk_size = 12  # 每批处理帧数
                    total_batches = (len(frame_tasks) + chunk_size - 1) // chunk_size

                    for batch_idx in range(total_batches):
                        # 获取当前批次任务
                        start_idx = batch_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, len(frame_tasks))
                        current_batch = frame_tasks[start_idx:end_idx]
                        
                        try:
                            # 并行处理当前批次
                            results = pool.map(_process_frame_optimized_shm, current_batch)
                            
                            # 按顺序写入FFmpeg
                            for result in results:
                                if result is not None:
                                    frame_idx, frame = result
                                    
                                    # 优化: 直接写入二进制数据，避免额外复制
                                    frame_bytes = frame.tobytes()
                                    proc.stdin.write(frame_bytes)
                                    proc.stdin.flush()
                                    
                                    # 更新进度
                                    frames_processed += 1
                                    
                                    # 报告进度
                                    report_progress()
                                else:
                                    logger.warning(f"批次 {batch_idx+1}/{total_batches} 中有帧处理失败")
                        
                        except Exception as e:
                            logger.error(f"处理批次 {batch_idx+1}/{total_batches} 时出错: {str(e)}\n{traceback.format_exc()}")
                            # 继续尝试处理其他批次
                        
                        # 检查FFmpeg是否仍在运行
                        if proc.poll() is not None:
                            logger.error(f"FFmpeg进程意外退出，返回码: {proc.returncode}")
                            # 读取剩余错误输出
                            try:
                                while True:
                                    err_line = stderr_queue.get_nowait()
                                    if err_line is None:
                                        break
                                    logger.error(f"FFmpeg错误: {err_line.decode('utf-8', errors='replace').strip()}")
                            except queue.Empty:
                                pass
                            break

                    # 6. 完成处理，关闭stdin管道
                    proc.stdin.close()
                    
                    # 等待FFmpeg完成
                    return_code = proc.wait()
                    
                    # 读取剩余输出
                    while True:
                        try:
                            err_line = stderr_queue.get_nowait()
                            if err_line is None:
                                break
                            # 只记录错误和警告
                            err_str = err_line.decode('utf-8', errors='replace').strip()
                            if "error" in err_str.lower() or "warning" in err_str.lower():
                                logger.warning(f"FFmpeg: {err_str}")
                        except queue.Empty:
                            break
                    
                    # 等待读取线程
                    stdout_thread.join(timeout=2)
                    stderr_thread.join(timeout=2)
                    
                    # 检查退出状态
                    if return_code != 0:
                        logger.error(f"FFmpeg进程异常退出，代码: {return_code}")
                        return None
                    else:
                        logger.info(
                            f"视频处理完成，用时: {time.time() - processing_start_time:.2f}秒，"
                            f"实际处理: {frames_processed}/{total_frames}帧"
                        )
                
            except Exception as e:
                logger.error(f"处理视频时出错: {str(e)}\n{traceback.format_exc()}")
                # 如果仍有FFmpeg进程，尝试终止
                try:
                    if 'proc' in locals() and proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=2)
                except:
                    pass
                return None
                
            finally:
                # 清理共享内存
                try:
                    if shm_name:
                        cleanup_shared_memory(shm_name)
                except Exception as e:
                    logger.warning(f"清理共享内存失败: {str(e)}")
                
                # 确保管道已关闭
                try:
                    if 'proc' in locals() and proc.poll() is None:
                        proc.terminate()
                except:
                    pass
                
                # 垃圾回收
                gc.collect()
        
        except Exception as e:
            logger.error(f"视频创建过程失败: {str(e)}\n{traceback.format_exc()}")
            # 回退到标准CPU处理
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

    def _create_static_video(self, image_array, output_path, duration, audio_path=None, fps=30):
        """创建静态视频，用于短文本不需要滚动的情况"""
