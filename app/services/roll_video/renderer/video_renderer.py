# renderer/video_renderer.py

import os
import numpy as np
import logging
import subprocess
import threading
import queue
from tqdm import tqdm
import gc
import time
import multiprocessing as mp
import multiprocessing.shared_memory as sm  # 导入 shared_memory
from PIL import Image
from typing import Dict, Tuple, List, Optional, Union
import platform
import ctypes  # 用于 SharedMemory 大小计算

from .memory_management import FrameMemoryPool  # 可选，如果后面要集成内存池管理
from .performance import PerformanceMonitor
from .frame_processors import (
    init_worker,  # 导入 worker 初始化函数
    _process_frame_optimized_shm,  # 导入要使用的帧处理函数 (指向 SHM 版本)
)

# from .frame_processors import fast_frame_processor # 这个在这个优化路径下不使用

logger = logging.getLogger(__name__)

# 移除旧的全局变量
# _g_img_array = None


class VideoRenderer:
    """视频渲染器，负责创建滚动效果的视频，使用ffmpeg管道和多进程/共享内存优化"""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 30,
        scroll_speed: float = 5,  # 保持 float，像素/帧
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
        # self.memory_pool = None # 内存池管理调整到主进程或专门 Manager
        self.frame_counter = 0
        self.total_frames = 0
        self._shm = None  # 用于存储共享内存对象
        self._shm_name = None

    # 移除 _init_memory_pool 方法，内存池的管理思路可能需要调整

    def _get_ffmpeg_command(
        self,
        output_path: str,
        pix_fmt: str,
        codec_and_output_params: List[str],
        audio_path: Optional[str],
    ) -> List[str]:
        """构造基础的ffmpeg命令 - 高性能优化版"""
        command = [
            "ffmpeg",
            "-y",
            # I/O优化参数
            "-probesize",
            "32M",  # 增加探测缓冲区大小
            "-analyzeduration",
            "32M",  # 增加分析时间
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
        """创建滚动视频 - 使用共享内存和优化多进程"""

        # 1. 图像预处理并存入共享内存
        logger.info("将文本渲染图像存入共享内存...")
        if transparency_required:
            # 确保是 RGBA
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            img_array = np.ascontiguousarray(np.array(image))
            channels = 4
            ffmpeg_pix_fmt = "rgba"  # FFmpeg input format for RGBA
        else:
            # 确保是 RGB
            if image.mode == "RGBA":
                image = image.convert("RGB")
            img_array = np.ascontiguousarray(np.array(image))
            channels = 3
            ffmpeg_pix_fmt = "rgb24"  # FFmpeg input format for RGB

        img_height, img_width = img_array.shape[:2]
        img_dtype = img_array.dtype
        img_size_bytes = img_array.nbytes

        try:
            # 创建共享内存
            self._shm = sm.SharedMemory(create=True, size=img_size_bytes)
            self._shm_name = self._shm.name
            # 将 NumPy 数组内容复制到共享内存
            shm_array = np.ndarray(
                img_array.shape, dtype=img_dtype, buffer=self._shm.buf
            )
            np.copyto(shm_array, img_array)
            del img_array  # 释放原始 NumPy 数组内存

            logger.info(
                f"创建共享内存 '{self._shm_name}' 成功，大小: {img_size_bytes / (1024*1024):.2f}MB"
            )

        except Exception as e:
            logger.error(f"创建共享内存失败: {e}. 回退到不使用共享内存 (性能会下降!)")
            # 这种回退需要全局变量的支持，或者重新设计 frame_processors 的参数传递
            # 为了简化，如果共享内存失败，直接报错退出，强制解决环境问题
            # 如果需要回退，需要在 frame_processors 中增加不使用 SHM 的路径，并在 init_worker 中不做任何事
            # 这里选择不回退，强制使用 SHM
            self._cleanup_shared_memory()  # 清理可能创建了一半的 SHM
            raise RuntimeError(
                f"创建共享内存失败: {e}. 请检查系统共享内存设置或可用内存。"
            )

        # 2. 滚动参数计算
        scroll_distance = max(text_actual_height, img_height - self.height)
        # 如果文本高度小于视频高度，scroll_distance 应该是 0，或者只是为了展示完整文本，
        # 这里的逻辑是基于文本会超出屏幕滚动的，如果文本很短，可能不需要滚动，或者只需要一点点来居中？
        # 假设 text_actual_height >= self.height，scroll_distance = text_actual_height - self.height
        # 如果 text_actual_height < self.height, scroll_distance = 0
        scroll_distance = max(
            0, text_actual_height - self.height
        )  # 只计算文本需要滚动的距离
        # 图像总高度 image.height > text_actual_height 因为底部有填充
        # 滚动的总像素距离应该是从顶部滚动到文本的底部刚刚滚出屏幕为止
        # 或者从文本顶部刚进入屏幕开始，到文本底部刚离开屏幕为止
        # 最简单的逻辑是，从顶部 0 滚到 img_height - self.height 的位置
        scroll_total_pixels = max(
            0, img_height - self.height
        )  # 滚动总距离是图像总高减去视频高

        scroll_frames = (
            int(scroll_total_pixels / self.scroll_speed) if self.scroll_speed > 0 else 0
        )

        # 确保短视频有最小时长
        min_scroll_duration_sec = 8.0  # 滚动部分的最小时间，排除头尾
        min_scroll_frames_content = int(min_scroll_duration_sec * self.fps)

        if scroll_frames < min_scroll_frames_content and scroll_frames > 0:
            # 需要滚动但滚动帧数不够，调整速度以达到最小滚动时间
            adjusted_speed = scroll_total_pixels / min_scroll_frames_content
            if adjusted_speed < self.scroll_speed:
                logger.info(
                    f"文本滚动部分 ({scroll_frames}帧) 短于最小要求 ({min_scroll_frames_content}帧)，"
                    f"调整滚动速度: {self.scroll_speed:.2f} → {adjusted_speed:.2f} 像素/帧"
                )
                self.scroll_speed = adjusted_speed
                scroll_frames = min_scroll_frames_content  # 更新滚动帧数以匹配调整后的速度和最小时间
            else:
                # 如果即使使用最小滚动时间，速度也比原来快，说明 original_speed 已经很快了，
                # 这种情况不太可能发生，除非 scroll_speed 初始设置得极低。
                # 如果发生了，说明 scroll_total_pixels 很大但 scroll_speed 也很低，
                # 或者 scroll_total_pixels 很小但 scroll_speed 不够低。
                # 此时保持原速，但可能需要增加静止时间
                pass  # 保持原速，不调整 scroll_frames

        padding_frames_start = int(self.fps * 2.0)
        padding_frames_end = int(self.fps * 2.0)

        # 总帧数 = 开始静止 + 滚动 + 结束静止
        total_frames = padding_frames_start + scroll_frames + padding_frames_end
        self.total_frames = total_frames
        duration = total_frames / self.fps

        logger.info(
            f"图像高:{img_height}, 视频高:{self.height}, 文本实际高:{text_actual_height}"
        )
        logger.info(
            f"滚动总像素距离:{scroll_total_pixels}, 滚动速度(调整后): {self.scroll_speed:.2f}像素/帧, 滚动帧:{scroll_frames}, "
            f"开始静止:{padding_frames_start}帧, 结束静止:{padding_frames_end}帧, 总帧:{total_frames}, 时长:{duration:.2f}s"
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
            self._cleanup_shared_memory()
            raise RuntimeError("找不到ffmpeg命令")

        # 然后检测GPU编码器是否可用
        gpu_encoders = []
        try:
            encoders = subprocess.check_output(
                ["ffmpeg", "-encoders"], stderr=subprocess.STDOUT
            ).decode("utf-8", errors="ignore")
            for line in encoders.splitlines():
                # Look for VCE (AMD), NVENC (NVIDIA), QSV (Intel)
                if "vce" in line or "nvenc" in line or "qsv" in line:
                    parts = line.split()
                    if len(parts) > 1:
                        gpu_encoders.append(parts[1])
            if gpu_encoders:
                logger.info(f"检测到GPU编码器: {', '.join(gpu_encoders)}")
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
            # 透明视频必须使用ProRes 或其他支持 alpha 的格式 (VP9 for webm)
            ffmpeg_pix_fmt_output = (
                "yuva444p10le"  # Output pixel format for ProRes with alpha
            )
            output_path = os.path.splitext(output_path)[0] + ".mov"
            video_codec_params = [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4",  # ProRes 4444
                "-pix_fmt",
                ffmpeg_pix_fmt_output,
                "-alpha_bits",
                "16",
                "-vendor",
                "ap10",
            ]
            logger.info("使用ProRes编码器处理透明视频 (.mov)")
        else:
            ffmpeg_pix_fmt_output = "yuv420p"  # Output pixel format for H.264/HEVC
            output_path = os.path.splitext(output_path)[0] + ".mp4"

            # 检查操作系统，在macOS上默认使用CPU编码（Apple Silicon有VideoToolbox，但需要专门参数）
            is_macos = platform.system() == "Darwin"
            # 尝试使用 VideoToolbox on macOS if available (for mp4)
            if is_macos:
                # Check if VT H264 encoder is available
                try:
                    vt_encoders = subprocess.check_output(
                        ["ffmpeg", "-encoders"], stderr=subprocess.STDOUT
                    ).decode("utf-8", errors="ignore")
                    if "h264_videotoolbox" in vt_encoders:
                        preferred_codec = "h264_videotoolbox"
                        gpu_encoders.append(preferred_codec)  # Add to list
                        logger.info("检测到VideoToolbox H.264编码器")
                    elif "hevc_videotoolbox" in vt_encoders:
                        preferred_codec = "hevc_videotoolbox"
                        gpu_encoders.append(preferred_codec)  # Add to list
                        logger.info("检测到VideoToolbox HEVC编码器")
                    else:
                        logger.info("未检测到VideoToolbox编码器")
                except:
                    logger.warning("无法检测VideoToolbox编码器")

            # 检查 preferred_codec 是否在检测到的 GPU 编码器列表中
            if preferred_codec in gpu_encoders:
                # 使用更高性能的GPU参数
                video_codec_params = [
                    "-c:v",
                    preferred_codec,
                    # GPU encoder presets might vary
                    "-preset",
                    (
                        "p4"
                        if preferred_codec in ["h264_nvenc", "hevc_nvenc"]
                        else "medium"
                    ),  # nvenc/hevc_nvenc use p1-p7, videotoolbox uses 'medium', 'fast', etc.
                    "-b:v",
                    "10M",  # 提升到10M比特率以提高质量
                    "-pix_fmt",
                    ffmpeg_pix_fmt_output,  # 输出格式
                    "-movflags",
                    "+faststart",
                ]
                if preferred_codec in ["h264_nvenc", "hevc_nvenc"]:
                    video_codec_params.extend(
                        ["-tune", "hq", "-g", str(self.fps * 2)]
                    )  # High quality tune, GOP every 2 seconds

                logger.info(
                    f"使用GPU编码器: {preferred_codec}，参数: {video_codec_params}"
                )
                use_gpu = True
            else:
                # 回退到CPU编码
                video_codec_params = [
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",  # 较低的CRF提高质量
                    "-preset",
                    "veryfast",  # 优先速度
                    "-pix_fmt",
                    ffmpeg_pix_fmt_output,
                    "-movflags",
                    "+faststart",
                    "-threads",
                    str(max(2, mp.cpu_count() - 2)),  # 使用大部分CPU核心
                ]
                logger.info(
                    f"使用CPU编码器: libx264 (GPU编码器不可用或被禁用)，参数: {video_codec_params}"
                )
                use_gpu = False

        # 完整的ffmpeg命令
        # Note: FFmpeg input pix_fmt should match the data written to stdin ('rgba' or 'rgb24')
        # FFmpeg output pix_fmt is set by -pix_fmt in video_codec_params ('yuva444p10le' or 'yuv420p')
        ffmpeg_cmd = self._get_ffmpeg_command(
            output_path,
            ffmpeg_pix_fmt,  # Input pix_fmt
            video_codec_params,
            audio_path,
        )

        logger.info(f"执行ffmpeg命令: {' '.join(ffmpeg_cmd)}")

        # 4. 启动 FFmpeg 进程
        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,  # 增大缓冲区
            )
        except FileNotFoundError:
            logger.error("无法启动FFmpeg进程。请确保FFmpeg已安装且在系统PATH中。")
            self._cleanup_shared_memory()
            raise RuntimeError("FFmpeg command not found.")
        except Exception as e:
            logger.error(f"启动FFmpeg进程失败: {e}")
            self._cleanup_shared_memory()
            raise RuntimeError(f"启动FFmpeg进程失败: {e}")

        # 创建stdout/stderr读取线程
        stdout_q = queue.Queue()
        stderr_q = queue.Queue()
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

        # 5. 启动进程池并处理帧
        # 确定最佳进程数
        try:
            cpu_count = mp.cpu_count()
            # 保留一些核心给 FFmpeg 和系统
            num_processes = max(2, cpu_count - 2)
            logger.info(
                f"检测到{cpu_count}个CPU核心，使用{num_processes}个进程进行帧处理"
            )
        except:
            num_processes = 6  # Default if detection fails
            logger.info(f"使用默认{num_processes}个进程进行帧处理")

        perf_monitor = PerformanceMonitor()
        perf_monitor.start()

        # 准备帧参数列表
        frame_params_list = []
        for frame_idx in range(total_frames):
            # 计算当前帧对应的图像Y坐标
            if frame_idx < padding_frames_start:
                # 开始静止阶段
                img_start_y = 0
            elif frame_idx >= total_frames - padding_frames_end:
                # 结束静止阶段
                img_start_y = max(0, img_height - self.height)
            else:
                # 滚动阶段
                scroll_frame_idx = frame_idx - padding_frames_start
                img_start_y = min(
                    img_height
                    - self.height,  # Ensure we don't scroll past the end of the image
                    int(scroll_frame_idx * self.scroll_speed),
                )

            # 添加到参数列表 (不包含 img_array 本身)
            frame_params_list.append(
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

        # 使用进程池处理帧，并迭代获取结果
        # 使用 imap_unordered 可以按完成顺序获取结果，但我们后面需要按帧索引排序
        # 或者直接使用 map 获取全部结果再排序，对于合理大小的视频，内存是足够的
        # 为了控制内存峰值，最好还是边生成边写入，使用 apply_async + 结果队列 + 写入线程/主进程循环
        # 简单起见，先尝试 map，如果内存问题再换 apply_async

        # 使用 pool.map 可能会一次性生成所有帧，内存压力大。
        # 更好的方法是 Producer-Consumer，但涉及 Queue, Manager, 锁定等，复杂。
        # 一个折衷是使用 imap_unordered 并手动排序写入。

        # 初始化一个临时的字典来存储乱序完成的帧
        processed_frames_buffer = {}
        next_frame_to_write = 0
        buffer_size_limit = num_processes * 2  # 缓冲区大小限制，防止内存无限增长

        logger.info("开始进程池帧处理...")
        with mp.Pool(
            processes=num_processes,
            initializer=init_worker,  # 设置 worker 初始化函数
            initargs=(self._shm_name, img_array.shape, img_dtype),  # 传递共享内存信息
        ) as pool:
            # 使用 imap_unordered 获取结果，按完成顺序返回
            results_iterator = pool.imap_unordered(
                _process_frame_to_use, frame_params_list
            )

            with tqdm(
                total=total_frames, desc=f"渲染({video_codec_params[1]})", unit="frame"
            ) as pbar:
                try:
                    for frame_idx, frame_data in results_iterator:
                        # 将完成的帧放入缓冲区
                        processed_frames_buffer[frame_idx] = frame_data

                        # 尝试写入连续的帧
                        while next_frame_to_write in processed_frames_buffer:
                            frame_to_write = processed_frames_buffer.pop(
                                next_frame_to_write
                            )
                            try:
                                process.stdin.write(frame_to_write.tobytes())
                                # Optional: process.stdin.flush() # Flushing per frame can be slow, let OS buffer
                                self.frame_counter += 1
                                pbar.update(1)
                                next_frame_to_write += 1

                                # 检查 ffmpeg 进程状态
                                if process.poll() is not None:
                                    raise BrokenPipeError(
                                        "FFmpeg process exited unexpectedly during write."
                                    )

                            except BrokenPipeError:
                                logger.error(f"管道写入失败，FFmpeg 进程可能已退出。")
                                # Stop processing and exit loop
                                self._cleanup_shared_memory()
                                # Try to read any remaining stderr for error details
                                stderr_output = []
                                while not stderr_q.empty():
                                    line = stderr_q.get_nowait()
                                    if line:
                                        stderr_output.append(
                                            line.decode(errors="ignore").strip()
                                        )
                                if stderr_output:
                                    logger.error(
                                        f"FFmpeg stderr (last 10 lines):\n"
                                        + "\n".join(stderr_output[-10:])
                                    )
                                raise RuntimeError(
                                    "FFmpeg process failed."
                                )  # Re-raise exception

                            except Exception as e:
                                logger.error(
                                    f"写入帧 {next_frame_to_write} 数据时出错: {e}"
                                )
                                # Stop processing and exit loop
                                self._cleanup_shared_memory()
                                raise RuntimeError(f"写入帧数据时出错: {e}")

                        # Optional: Check buffer size to prevent excessive memory usage
                        # if len(processed_frames_buffer) > buffer_size_limit:
                        #     logger.warning(f"帧写入缓冲区达到 {len(processed_frames_buffer)} 帧，可能存在瓶颈。")
                        #     time.sleep(0.1) # Wait a bit if buffer is too large

                    # 所有任务已提交，继续写入剩余缓冲区的帧
                    while next_frame_to_write in processed_frames_buffer:
                        frame_to_write = processed_frames_buffer.pop(
                            next_frame_to_write
                        )
                        try:
                            process.stdin.write(frame_to_write.tobytes())
                            self.frame_counter += 1
                            pbar.update(1)
                            next_frame_to_write += 1
                            if process.poll() is not None:
                                raise BrokenPipeError(
                                    "FFmpeg process exited unexpectedly during final write."
                                )

                        except BrokenPipeError:
                            logger.error(f"最终管道写入失败，FFmpeg 进程可能已退出。")
                            self._cleanup_shared_memory()
                            stderr_output = []
                            while not stderr_q.empty():
                                line = stderr_q.get_nowait()
                                if line:
                                    stderr_output.append(
                                        line.decode(errors="ignore").strip()
                                    )
                            if stderr_output:
                                logger.error(
                                    f"FFmpeg stderr (last 10 lines):\n"
                                    + "\n".join(stderr_output[-10:])
                                )
                            raise RuntimeError(
                                "FFmpeg process failed during final write."
                            )  # Re-raise exception
                        except Exception as e:
                            logger.error(
                                f"写入最终帧 {next_frame_to_write} 数据时出错: {e}"
                            )
                            self._cleanup_shared_memory()
                            raise RuntimeError(f"写入最终帧数据时出错: {e}")

                except Exception as e:
                    logger.error(f"进程池处理或写入循环出错: {e}", exc_info=True)
                    # Clean up immediately on error
                    pool.terminate()  # 终止所有 worker 进程
                    process.terminate()  # 终止 FFmpeg 进程
                    self._cleanup_shared_memory()
                    raise RuntimeError(
                        f"渲染管线处理失败: {e}"
                    )  # Re-raise for service layer

                finally:
                    # 确保所有 worker 完成
                    pool.close()
                    pool.join()
                    logger.info("进程池已关闭。")

            # 6. 关闭 FFmpeg 输入并等待完成
            logger.info("关闭 FFmpeg stdin...")
            try:
                process.stdin.close()
            except BrokenPipeError:
                logger.warning("FFmpeg stdin 管道已断开 (可能已提前退出)。")
            except Exception as e:
                logger.warning(f"关闭 FFmpeg stdin 时出错: {e}")

            logger.info("等待 FFmpeg 进程完成...")
            try:
                return_code = process.wait(
                    timeout=120.0 + duration
                )  # 等待 FFmpeg 完成，给足时间
                logger.info(f"ffmpeg进程完成，返回码: {return_code}")

                # 记录并检查输出/错误
                stderr_lines = []
                while not stderr_q.empty():
                    line = stderr_q.get_nowait()  # Use non-blocking read
                    if line is not None:
                        stderr_lines.append(line.decode(errors="ignore").strip())

                if return_code != 0:
                    error_output = "\n".join(stderr_lines[-30:])  # 记录最后30行
                    logger.error(f"ffmpeg处理失败，返回码: {return_code}")
                    logger.error(f"ffmpeg错误输出:\n{error_output}")
                    # Clean up shared memory *before* raising error
                    self._cleanup_shared_memory()
                    raise RuntimeError(
                        f"FFmpeg处理失败，返回码: {return_code}\n错误信息:\n{error_output}"
                    )

                logger.info(f"视频文件已生成: {output_path}")

            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg进程超时，尝试终止")
                process.kill()
                self._cleanup_shared_memory()
                raise RuntimeError("FFmpeg进程超时")
            except Exception as e:
                logger.error(f"等待 ffmpeg 或处理其输出时出错: {e}")
                self._cleanup_shared_memory()
                raise RuntimeError(f"等待 ffmpeg 或处理其输出时出错: {e}")

        # 7. 清理共享内存
        self._cleanup_shared_memory()
        gc.collect()  # 强制垃圾回收

        # 8. 记录最终性能
        end_time = time.time()
        total_time = end_time - perf_monitor.start_time
        avg_fps = self.frame_counter / total_time if total_time > 0 else 0

        logger.info(
            f"总渲染性能: 渲染了{self.frame_counter}帧，"
            f"耗时{total_time:.2f}秒，"
            f"平均{avg_fps:.2f}帧/秒"
        )

        return output_path  # Return the actual output path

    def _cleanup_shared_memory(self):
        """清理创建的共享内存"""
        if self._shm is not None:
            try:
                self._shm.close()
                if self._shm_name:
                    sm.SharedMemory(
                        name=self._shm_name
                    ).unlink()  # Unlink after closing
                    logger.info(f"共享内存 '{self._shm_name}' 已清理。")
                self._shm = None
                self._shm_name = None
            except FileNotFoundError:
                logger.warning(f"共享内存 '{self._shm_name}' 清理时未找到。")
            except Exception as e:
                logger.error(f"清理共享内存 '{self._shm_name}' 时出错: {e}")

    # 可以保留 create_scrolling_video 方法作为旧版本或备用，但不再需要在 optimized 中调用它
    # ... (原 create_scrolling_video 方法代码)
    # 如果不再需要旧方法，可以直接删除或大幅简化
