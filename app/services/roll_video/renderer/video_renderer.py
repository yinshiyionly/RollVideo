"""视频渲染器模块"""

import os
import logging
import numpy as np
import subprocess
import gc
import time
from PIL import Image
import platform
import traceback

from .performance import PerformanceMonitor

logger = logging.getLogger(__name__)


class VideoRenderer:
    """视频渲染器，负责创建滚动效果的视频，使用ffmpeg管道和线程读取优化"""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 30,
        roll_px: float = 1.6,  # 每帧滚动的像素数（由service层基于行高和每秒滚动行数计算而来）
    ):
        """
        初始化视频渲染器

        Args:
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率
            roll_px: 每帧滚动的像素数（由service层基于行高和每秒滚动行数计算而来）
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.roll_px = roll_px
        self.memory_pool = None
        self.frame_counter = 0
        self.total_frames = 0
        
        # 性能统计数据
        self.performance_stats = {
            "preparation_time": 0,     # 准备阶段时间
            "frame_processing_time": 0, # 帧处理阶段时间
            "encoding_time": 0,         # 视频编码阶段时间
            "total_time": 0,            # 总时间
            "frames_processed": 0,      # 处理的帧数
            "fps": 0,                   # 平均每秒处理的帧数
        }

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

    def _get_codec_parameters(self, preferred_codec, transparency_required, channels):
        """
        获取适合当前平台和需求的编码器参数
        
        Args:
            preferred_codec: 首选编码器
            transparency_required: 是否需要透明支持
            channels: 通道数（3=RGB, 4=RGBA）
            
        Returns:
            (codec_params, pix_fmt): 编码器参数列表和像素格式
        """
        # 检查系统平台
        is_macos = platform.system() == "Darwin"
        is_windows = platform.system() == "Windows"
        is_linux = platform.system() == "Linux"
        
        # 透明背景需要特殊处理
        if transparency_required or channels == 4:
            # 透明背景需要特殊处理
            pix_fmt = "rgba"
            # ProRes 4444保留Alpha
            codec_params = [
                "-c:v", "prores_ks", 
                "-profile:v", "4444",
                "-pix_fmt", "yuva444p10le", 
                "-alpha_bits", "16",
                "-vendor", "ap10", 
                "-colorspace", "bt709",
            ]
            logger.info("使用CPU编码器: ProRes 4444(透明视频)")
            return codec_params, pix_fmt
        
        # 不透明视频处理
        pix_fmt = "rgb24"
        
        # 检查是否强制使用CPU
        force_cpu = "NO_GPU" in os.environ
        
        # 根据平台和编码器选择参数
        if preferred_codec == "h264_nvenc" and not force_cpu:
            # NVIDIA GPU加速
            # todo ... 按照GPU的性能同步修改一下CPU的参数 
            if is_windows or is_linux:
                codec_params = [
                    "-c:v", "h264_nvenc",
                    "-preset", "p7",  # 1-7质量最高 
                    "-rc", "vbr",  # 使用VBR编码，平均码率，   -rc cbr -b:v 10M 这个组合是强制填充码率
                    "-cq", "15",  # 质量因子，小到大效果越来越低
                    "-b:v", "10M",  # 平均码率
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                ]
                logger.info("使用GPU编码器: h264_nvenc")
            else:
                codec_params = [
                    "-c:v", "libx264",
                    "-preset", "veryfast",  # 使用更快的预设
                    "-crf", "20",  # 略微降低质量以提高速度
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                ]
                logger.info("平台不支持NVIDIA编码，切换到CPU编码器: libx264")
        elif preferred_codec == "h264_videotoolbox":
            codec_params = [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]
            logger.info("不支持VideoToolbox，使用CPU编码器: libx264")
        elif preferred_codec == "prores_ks":
            # ProRes (非透明)
            codec_params = [
                "-c:v", "prores_ks",
                "-profile:v", "3",  # ProRes 422 HQ
                "-pix_fmt", "yuv422p10le",
                "-vendor", "ap10",
                "-colorspace", "bt709",
            ]
            logger.info("使用CPU编码器: ProRes(非透明)")
        else:
            # 默认使用libx264 (高质量CPU编码)
            codec_params = [
                "-c:v", "libx264",
                "-preset", "medium",  # 平衡速度和质量的预设
                "-crf", "20",         # 恒定质量因子 (0-51, 越低质量越高)
                "-pix_fmt", "yuv420p", # 兼容大多数播放器
                "-movflags", "+faststart", # MP4优化
            ]
            logger.info(f"使用CPU编码器: libx264")
        
        return codec_params, pix_fmt

    def create_scrolling_video_crop(
        self,
        image,
        output_path,
        text_actual_height,
        transparency_required=False,
        preferred_codec="libx264",
        audio_path=None,
        bg_color=(255, 255, 255)
    ):
        """
        使用FFmpeg的crop滤镜和时间表达式创建滚动视频，只支持从下到上滚动
        
        参数:
            image: 要滚动的图像 (PIL.Image或NumPy数组)
            output_path: 输出视频文件路径
            text_actual_height: 文本实际高度
            transparency_required: 是否需要透明通道
            preferred_codec: 首选视频编码器
            audio_path: 可选的音频文件路径
            bg_color: 背景颜色 (R,G,B) 或 (R,G,B,A)
        
        Returns:
            输出视频的路径
        """
        try:
            # 记录开始时间
            total_start_time = time.time()
            
            # 初始化性能统计
            self.performance_stats = {
                "preparation_time": 0,
                "encoding_time": 0,
                "total_time": 0,
                "frames_processed": 0,
                "fps": 0
            }
            
            # 1. 准备图像
            preparation_start_time = time.time()
            
            # 将输入图像转换为PIL.Image对象
            if isinstance(image, np.ndarray):
                # NumPy数组转PIL图像
                if image.shape[2] == 4:  # RGBA
                    pil_image = Image.fromarray(image, 'RGBA')
                else:  # RGB
                    pil_image = Image.fromarray(image, 'RGB')
            elif isinstance(image, Image.Image):
                # 直接使用PIL图像
                pil_image = image
            else:
                raise ValueError("不支持的图像类型，需要PIL.Image或numpy.ndarray")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 设置临时图像文件路径
            temp_img_path = f"{os.path.splitext(output_path)[0]}_temp.png"
            
            # 临时图像优化选项
            image_optimize_options = {
                "optimize": True,  # 优化图像存储
                "compress_level": 6,  # 中等压缩级别
            }
            
            # 使用PIL直接保存图像，保留原始格式和所有信息
            pil_image.save(temp_img_path, format="PNG", **image_optimize_options)
            
            # 获取图像尺寸
            img_width, img_height = pil_image.size
            
            # 清理内存中的大型对象，确保不会占用过多内存
            del pil_image
            gc.collect()
            
            # 2. 计算滚动参数
            # 滚动距离 = 图像高度 - 视频高度
            scroll_distance = max(0, img_height - self.height)
            
            # 确保至少有8秒的滚动时间
            min_scroll_duration = 8.0  # 秒
            scroll_duration = max(min_scroll_duration, scroll_distance / (self.roll_px * self.fps))
            
            # 前后各添加2秒静止时间
            start_static_time = 2.0  # 秒
            end_static_time = 2.0  # 秒
            total_duration = start_static_time + scroll_duration + end_static_time
            
            # 总帧数
            total_frames = int(total_duration * self.fps)
            self.total_frames = total_frames
            
            # 滚动起始和结束时间点
            scroll_start_time = start_static_time
            scroll_end_time = start_static_time + scroll_duration
            
            logger.info(f"视频参数: 宽度={self.width}, 高度={self.height}, 帧率={self.fps}")
            logger.info(f"滚动参数: 距离={scroll_distance}px, 速度={self.roll_px}px/帧, 持续={scroll_duration:.2f}秒")
            logger.info(f"时间设置: 总时长={total_duration:.2f}秒, 静止开始={start_static_time}秒, 静止结束={end_static_time}秒")
            
            # 3. 设置编码器参数
            codec_params, pix_fmt = self._get_codec_parameters(
                preferred_codec, transparency_required, 4 if transparency_required else 3
            )
            
            # 准备阶段结束
            preparation_end_time = time.time()
            self.performance_stats["preparation_time"] = preparation_end_time - preparation_start_time
            
            # 4. 创建FFmpeg命令，使用crop滤镜和表达式
            encoding_start_time = time.time()
            
            # 检查系统是否有支持的GPU加速
            gpu_support = {
                "nvidia": False
            }
            
            # 检测NVIDIA GPU
            try:
                nvidia_result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
                gpu_support["nvidia"] = nvidia_result.returncode == 0
                
                if gpu_support["nvidia"]:
                    logger.info("检测到NVIDIA GPU，将尝试使用NVENC编码器")
            except Exception as e:
                logger.info(f"NVIDIA GPU检测出错: {e}，将使用CPU处理")
            
            # 构建基本FFmpeg命令
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-loop", "1",  # 循环输入图像
                "-i", temp_img_path,  # 输入图像
                "-progress", "pipe:2",  # 输出进度信息到stderr
                "-stats",  # 启用统计信息
                "-stats_period", "1",  # 每1秒输出一次统计信息
                "-max_muxing_queue_size", "8192",  # 限制复用队列大小
            ]
            
            # 添加音频输入（如果有）
            if audio_path and os.path.exists(audio_path):
                ffmpeg_cmd.extend(["-i", audio_path])
                
            # 创建裁剪表达式
            crop_y_expr = f"'if(between(t,{scroll_start_time},{scroll_end_time}),min({img_height-self.height},(t-{scroll_start_time})/{scroll_duration}*{scroll_distance}),if(lt(t,{scroll_start_time}),0,{scroll_distance}))'"
            
            # 始终使用CPU的crop滤镜,GPU没有crop滤镜
            crop_expr = (
                f"crop=w={self.width}:h={self.height}:"
                f"x=0:y={crop_y_expr}"
            )
            
            # 添加滤镜
            ffmpeg_cmd.extend([
                "-filter_complex", crop_expr,
            ])
            
            # 基于GPU支持选择合适的编码器
            if gpu_support["nvidia"] and not transparency_required:
                # 尝试使用NVENC编码器
                if "libx264" in ' '.join(codec_params):
                    logger.info("切换到NVIDIA硬件编码器(h264_nvenc)")
                    for i, param in enumerate(codec_params):
                        if param == "libx264":
                            codec_params[i] = "h264_nvenc"
                            break
            else:
                logger.info("未检测到支持的GPU或使用透明视频，将使用CPU处理")
                
            # 添加公共参数
            ffmpeg_cmd.extend([
                "-t", str(total_duration),  # 设置总时长
                "-vsync", "1",  # 添加vsync参数，确保平滑的视频同步
                "-thread_queue_size", "8192",  # 限制线程队列大小，减少内存使用
            ])
            
            # 添加视频编码参数
            ffmpeg_cmd.extend(codec_params)
            
            # 设置帧率
            ffmpeg_cmd.extend(["-r", str(self.fps)])
            
            # 添加音频映射（如果有）
            if audio_path and os.path.exists(audio_path):
                ffmpeg_cmd.extend([
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-map", "0:v:0",  # 从第1个输入（索引0）获取视频
                    "-map", "1:a:0",  # 从第2个输入（索引1）获取音频
                    "-shortest",
                ])
            
            # 添加输出路径
            ffmpeg_cmd.append(output_path)

            logger.info(f"FFmpeg命令: {' '.join(ffmpeg_cmd)}")

            # 5. 执行FFmpeg命令
            try:
                # 启动进程
                logger.info("正在启动FFmpeg进程...")
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # 行缓冲
                    universal_newlines=True  # 使用通用换行符
                )
                
                # 使用新的进度条监控FFmpeg进度
                monitor_thread = PerformanceMonitor.monitor_ffmpeg_progress(
                    process=process,
                    total_duration=total_duration,
                    total_frames=total_frames,
                    encoding_start_time=encoding_start_time
                )
                
                # 获取输出和错误
                stdout, stderr = process.communicate()
                
                # 等待监控线程结束
                if monitor_thread and monitor_thread.is_alive():
                    monitor_thread.join(timeout=2.0)
                
                # 检查进程返回码
                if process.returncode != 0:
                    logger.error(f"FFmpeg处理失败: {stderr}")
                    # 分析常见错误原因，提供更详细的错误信息
                    if "No space left on device" in stderr:
                        raise Exception(f"FFmpeg处理失败: 设备存储空间不足")
                    elif "Invalid argument" in stderr:
                        raise Exception(f"FFmpeg处理失败: 参数无效，请检查命令")
                    elif "Error opening filters" in stderr:
                        raise Exception(f"FFmpeg处理失败: 滤镜配置错误，请检查滤镜表达式")
                    elif "CUDA error" in stderr or "CUDA failure" in stderr:
                        raise Exception("FFmpeg处理失败: CUDA错误，可能是GPU内存不足或驱动问题")
                    elif "Impossible to convert between the formats" in stderr:
                        raise Exception("FFmpeg处理失败: 滤镜之间的格式转换不兼容，这可能是CUDA格式问题")
                    elif "Function not implemented" in stderr:
                        raise Exception("FFmpeg处理失败: 功能未实现，可能是当前CUDA版本不支持某些操作") 
                    else:
                        raise Exception(f"FFmpeg处理失败，返回码: {process.returncode}")
            except Exception as e:
                logger.error(f"FFmpeg处理失败: {str(e)}")
                raise
                
            # 删除临时图像文件
            try:
                os.remove(temp_img_path)
                logger.info(f"已删除临时文件: {temp_img_path}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")
            
            # 更新性能统计信息
            encoding_end_time = time.time()
            self.performance_stats["encoding_time"] = encoding_end_time - encoding_start_time
            self.performance_stats["total_time"] = encoding_end_time - total_start_time
            self.performance_stats["frames_processed"] = total_frames
            self.performance_stats["fps"] = total_frames / max(0.001, self.performance_stats["encoding_time"])
            
            logger.info(f"视频处理完成: {output_path}")
            logger.info(f"性能统计: 准备={self.performance_stats['preparation_time']:.2f}秒, 编码={self.performance_stats['encoding_time']:.2f}秒")
            logger.info(f"总时间: {self.performance_stats['total_time']:.2f}秒, 平均帧率: {self.performance_stats['fps']:.1f}FPS")
            
            return output_path
            
        except Exception as e:
            logger.error(f"使用FFmpeg滤镜创建滚动视频时出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 记录总时间（即使发生错误）
            self.performance_stats["total_time"] = time.time() - total_start_time
            raise

    def create_scrolling_video_overlay_cuda(
        self,
        image,
        output_path,
        text_actual_height,
        transparency_required=False,
        preferred_codec="h264_nvenc",
        audio_path=None,
        bg_color=(0, 0, 0, 255)
    ):
        """
        使用FFmpeg的overlay_cuda滤镜创建GPU加速的滚动视频
        
        参数:
            image: 要滚动的图像 (PIL.Image或NumPy数组)
            output_path: 输出视频文件路径
            text_actual_height: 文本实际高度
            transparency_required: 是否需要透明通道
            preferred_codec: 首选视频编码器
            audio_path: 可选的音频文件路径
            bg_color: 背景颜色 (R,G,B) 或 (R,G,B,A)
            
        Returns:
            输出视频的路径
        """
        try:
            # 记录开始时间
            total_start_time = time.time()
            
            # 初始化性能统计
            self.performance_stats = {
                "preparation_time": 0,
                "encoding_time": 0,
                "total_time": 0,
                "frames_processed": 0,
                "fps": 0
            }
            
            # 1. 准备图像
            preparation_start_time = time.time()
            
            # 将输入图像转换为PIL.Image对象
            if isinstance(image, np.ndarray):
                # NumPy数组转PIL图像
                if image.shape[2] == 4:  # RGBA
                    pil_image = Image.fromarray(image, 'RGBA')
                else:  # RGB
                    pil_image = Image.fromarray(image, 'RGB')
            elif isinstance(image, Image.Image):
                # 直接使用PIL图像
                pil_image = image
            else:
                raise ValueError("不支持的图像类型，需要PIL.Image或numpy.ndarray")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 设置临时图像文件路径
            temp_img_path = f"{os.path.splitext(output_path)[0]}_temp.png"
            
            # 临时图像优化选项,False不优化
            image_optimize_options = {
                "optimize": False,
                "compress_level": 0,
            }
            
            # 使用PIL直接保存图像，保留原始格式和所有信息
            pil_image.save(temp_img_path, format="PNG", **image_optimize_options)
            
            # 获取图像尺寸
            img_width, img_height = pil_image.size
            
            # 清理内存中的大型对象，确保不会占用过多内存
            del pil_image
            gc.collect()
            
            # 2. 计算滚动参数
            # 滚动距离 = 图像高度 - 视频高度
            scroll_distance = max(0, img_height - self.height)
            
            # 确保至少有8秒的滚动时间
            min_scroll_duration = 8.0  # 秒
            scroll_duration = max(min_scroll_duration, scroll_distance / (self.roll_px * self.fps))
            
            # 前后各添加2秒静止时间
            start_static_time = 2.0  # 秒
            end_static_time = 2.0  # 秒
            total_duration = start_static_time + scroll_duration + end_static_time
            
            # 总帧数
            total_frames = int(total_duration * self.fps)
            self.total_frames = total_frames
            
            # 滚动起始和结束时间点
            scroll_start_time = start_static_time
            scroll_end_time = start_static_time + scroll_duration
            
            logger.info(f"视频参数: 宽度={self.width}, 高度={self.height}, 帧率={self.fps}")
            logger.info(f"滚动参数: 距离={scroll_distance}px, 速度={self.roll_px}px/帧, 持续={scroll_duration:.2f}秒")
            logger.info(f"时间设置: 总时长={total_duration:.2f}秒, 静止开始={start_static_time}秒, 静止结束={end_static_time}秒")
            
            # 3. 设置编码器参数
            codec_params, _ = self._get_codec_parameters(
                preferred_codec, transparency_required, 4 if transparency_required else 3
            )
            
            # 准备阶段结束
            preparation_end_time = time.time()
            self.performance_stats["preparation_time"] = preparation_end_time - preparation_start_time
            
            # 4. 确认系统是否支持CUDA和overlay_cuda滤镜
            encoding_start_time = time.time()
            has_cuda_support = False
            has_overlay_cuda = False
            
            try:
                # 1. 首先检测NVIDIA GPU是否存在
                nvidia_result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
                has_cuda_support = nvidia_result.returncode == 0
                
                if not has_cuda_support:
                    logger.warning("未检测到NVIDIA GPU，overlay_cuda滤镜需要NVIDIA GPU支持，将回退到使用CPU的crop滤镜方法")
                    return self.create_scrolling_video_crop(
                        image=image,
                        output_path=output_path,
                        text_actual_height=text_actual_height,
                        transparency_required=transparency_required,
                        preferred_codec=preferred_codec,
                        audio_path=audio_path,
                        bg_color=bg_color,
                    )
                
                # 2. 再检测是否支持overlay_cuda滤镜
                filter_check = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-filters"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                has_overlay_cuda = "overlay_cuda" in filter_check.stdout
                
                if not has_overlay_cuda:
                    logger.warning("系统不支持overlay_cuda滤镜，将回退到使用CPU的crop滤镜方法")
                    return self.create_scrolling_video_crop(
                        image=image,
                        output_path=output_path,
                        text_actual_height=text_actual_height,
                        transparency_required=transparency_required,
                        preferred_codec=preferred_codec,
                        audio_path=audio_path,
                        bg_color=bg_color,
                    )
                
                logger.info("overlay_cuda滤镜检测，通过✅")
                
            except Exception as e:
                logger.warning(f"检测CUDA或overlay_cuda滤镜时出错: {e}")
                logger.info("将回退到使用CPU的crop滤镜方法")
                return self.create_scrolling_video_crop(
                    image=image,
                    output_path=output_path,
                    text_actual_height=text_actual_height,
                    transparency_required=transparency_required,
                    preferred_codec=preferred_codec,
                    audio_path=audio_path,
                    bg_color=bg_color,
                )
            
            # 5. 构建基本FFmpeg命令 (CUDA加速版)
            # 将背景色从RGB(A)转换为十六进制格式 (#RRGGBB)
            bg_hex = "#{:02x}{:02x}{:02x}".format(bg_color[0], bg_color[1], bg_color[2])
            
            # 构建基础命令
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-f", "lavfi", 
                "-i", f"color=c={bg_hex}:s={self.width}x{self.height}:r={self.fps},format=yuv420p,hwupload_cuda",
                "-i", temp_img_path,
                "-progress", "pipe:2",  # 输出进度信息到stderr
                "-stats",  # 启用统计信息
                "-stats_period", "1",  # 每1秒输出一次统计信息
            ]
            
            # 添加音频输入（如果有）
            if audio_path and os.path.exists(audio_path):
                ffmpeg_cmd.extend(["-i", audio_path])
            
            # 根据用户提供的命令修改滚动表达式
            # overlay_cuda=x=0:y='if(between(t,2.0,458.85), -((t-2.0)/456.85)*27411, if(lt(t,2.0), 0, -27411))'
            y_expr = f"if(between(t,{scroll_start_time},{scroll_end_time}), -((t-{scroll_start_time})/{scroll_duration})*{img_height-self.height}, if(lt(t,{scroll_start_time}), 0, -{img_height-self.height}))"
            
            # 透明视频滤镜链
            if transparency_required:
                # 准备前景图像：先转换格式为RGBA，然后上传到CUDA
                img_filter = f"[1:v]fps={self.fps},format=rgba,hwupload_cuda[img_cuda];"
                bg_format = "rgba" # 如果前景是RGBA，背景也用RGBA以确保兼容性
                bg_filter = f"[0:v]format={bg_format},hwupload_cuda[bg_cuda];"
                # overlay_cuda 叠加
                overlay_filter = f"[bg_cuda][img_cuda]overlay_cuda=x=0:y=\"{y_expr}\"[out_cuda];"
                # 确保输出流格式正确 (RGBA输出需要hwdownload+format转换)
                output_filter = "[out_cuda]hwdownload,format=rgba[out]"
                logger.info("使用 overlay_cuda 处理透明视频 (RGBA格式)")
                filter_complex = (
                    bg_filter +
                    img_filter +
                    overlay_filter +
                    output_filter
                )
            # 非透明视频滤镜链
            else:
                filter_complex = f"[1:v]fps={self.fps},format=yuv420p,hwupload_cuda[img_cuda_no_alpha]; \
                                 [0:v][img_cuda_no_alpha]overlay_cuda=x=0:y='{y_expr}'[out_cuda]; \
                                 [out_cuda]hwdownload,format=yuv420p[out]"
                logger.info("使用 overlay_cuda 处理透明视频 (YUV420P格式)")

            ffmpeg_cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[out]"
            ])
            
            # 添加音频映射（如果有）
            if audio_path and os.path.exists(audio_path):
                ffmpeg_cmd.extend([
                    "-map", "2:a:0",  # 从第3个输入（索引2）获取音频
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                ])
            
            # 添加视频编码器参数
            ffmpeg_cmd.extend(codec_params)
            
            # 设置持续时间
            ffmpeg_cmd.extend(["-t", str(total_duration)])
            
            # 添加输出路径
            ffmpeg_cmd.append(output_path)
            
            logger.info(f"FFmpeg命令: {' '.join(ffmpeg_cmd)}")
            
            # 6. 执行FFmpeg命令
            try:
                # 启动进程
                logger.info("正在启动FFmpeg进程 (overlay_cuda)...")
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # 行缓冲
                    universal_newlines=True  # 使用通用换行符
                )
                
                # 使用性能监控线程
                monitor_thread = PerformanceMonitor.monitor_ffmpeg_progress(
                    process=process,
                    total_duration=total_duration,
                    total_frames=total_frames,
                    encoding_start_time=encoding_start_time
                )
                
                # 获取输出和错误
                stdout, stderr = process.communicate()
                
                # 等待监控线程结束
                if monitor_thread and monitor_thread.is_alive():
                    monitor_thread.join(timeout=2.0)
                
                # 检查进程返回码
                if process.returncode != 0:
                    logger.error(f"FFmpeg处理失败: {stderr}")
                    # 分析常见错误原因，提供更详细的错误信息
                    if "No space left on device" in stderr:
                        raise Exception("FFmpeg处理失败: 设备存储空间不足")
                    elif "Invalid argument" in stderr:
                        raise Exception("FFmpeg处理失败: 参数无效，请检查命令")
                    elif "Error opening filters" in stderr:
                        raise Exception("FFmpeg处理失败: 滤镜配置错误，请检查滤镜表达式")
                    elif "CUDA error" in stderr or "CUDA failure" in stderr:
                        raise Exception("FFmpeg处理失败: CUDA错误，可能是GPU内存不足或驱动问题")
                    elif "Impossible to convert between the formats" in stderr:
                        raise Exception("FFmpeg处理失败: 滤镜之间的格式转换不兼容，这可能是CUDA格式问题")
                    elif "Function not implemented" in stderr:
                        raise Exception("FFmpeg处理失败: 功能未实现，可能是当前CUDA版本不支持某些操作") 
                    else:
                        raise Exception(f"FFmpeg处理失败，返回码: {process.returncode}")
            except Exception as e:
                logger.error(f"FFmpeg处理失败: {str(e)}")
                raise
                
            # 删除临时图像文件
            try:
                os.remove(temp_img_path)
                logger.info(f"已删除临时文件: {temp_img_path}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")
            
            # 更新性能统计信息
            encoding_end_time = time.time()
            self.performance_stats["encoding_time"] = encoding_end_time - encoding_start_time
            self.performance_stats["total_time"] = encoding_end_time - total_start_time
            self.performance_stats["frames_processed"] = total_frames
            self.performance_stats["fps"] = total_frames / max(0.001, self.performance_stats["encoding_time"])
            
            logger.info(f"视频处理完成 (overlay_cuda): {output_path}")
            logger.info(f"性能统计: 准备={self.performance_stats['preparation_time']:.2f}秒, 编码={self.performance_stats['encoding_time']:.2f}秒")
            logger.info(f"总时间: {self.performance_stats['total_time']:.2f}秒, 平均帧率: {self.performance_stats['fps']:.1f}FPS")
            
            return output_path
            
        except Exception as e:
            logger.error(f"创建滚动视频失败 (overlay_cuda): {str(e)}")
            logger.error(traceback.format_exc())
            try:
                # 清理临时文件
                if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            except:
                pass
            raise
