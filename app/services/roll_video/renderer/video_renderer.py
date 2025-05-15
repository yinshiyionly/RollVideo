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
import requests

from .performance import PerformanceMonitor

logger = logging.getLogger(__name__)


class VideoRenderer:
    """视频渲染器，负责创建滚动效果的视频，使用ffmpeg管道和线程读取优化"""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 60,
        roll_px: float = 1.6,  # 每帧滚动的像素数（由service层基于行高和每秒滚动行数计算而来）
        top_blank: int = 0     # 顶部留白，单位 px
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
        self.top_blank = top_blank
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
        
        # 不透明视频处理
        pix_fmt = "rgb24"
        
        # 检查是否强制使用CPU
        force_cpu = "NO_GPU" in os.environ
        
        # 根据平台和编码器选择参数
        if preferred_codec == "h264_nvenc" and not force_cpu:
            if is_windows or is_linux:
                codec_params = [
                    "-c:v", "h264_nvenc",
                    "-preset", "p4",            # 1-7质量最高 
                    "-rc", "vbr",               # 使用VBR编码，平均码率，   -rc cbr -b:v 10M 这个组合是强制填充码率
                    "-cq", "25",                # 质量因子，小到大效果越来越低
                    "-b:v", "10M",              # 平均码率
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-bf", "3",                 # 视频B帧数量
                    "-g", "60",                 # 关键帧
                ]
                logger.info("使用GPU编码器: h264_nvenc")
            else:
                codec_params = [
                    "-c:v", "libx264",
                    "-preset", "medium",      # 质量最高的预设，对应NVENC的p7
                    "-crf", "18",               # 对应NVENC的cq=15，越低质量越高
                    "-b:v", "10M",              # 与NVENC参数保持一致的平均码率
                    "-maxrate", "15M",          # 最大码率限制
                    "-bufsize", "20M",          # 码率控制缓冲区
                    "-pix_fmt", "yuv420p",      # 与NVENC参数保持一致
                    "-movflags", "+faststart",  # MP4优化
                    "-bf", "3",                 # 视频B帧数量
                    "-g", "60",                 # 关键帧
                ]
                logger.info("平台不支持NVIDIA编码，切换到CPU编码器: libx264")
        else:
            # 默认使用libx264 (高质量CPU编码)
            codec_params = [
                "-c:v", "libx264",
                "-preset", "medium",      # 质量最高的预设，对应NVENC的p7
                "-crf", "18",               # 对应NVENC的cq=15，越低质量越高
                "-b:v", "10M",              # 与NVENC参数保持一致的平均码率
                "-maxrate", "15M",          # 最大码率限制
                "-bufsize", "20M",          # 码率控制缓冲区
                "-pix_fmt", "yuv420p",      # 与NVENC参数保持一致
                "-movflags", "+faststart",  # MP4优化
                "-bf", "3",                 # 视频B帧数量
                "-g", "60",                 # 关键帧
            ]
            logger.info(f"使用CPU编码器: libx264(高质量模式)")
        
        return codec_params, pix_fmt

    def _download_background_image(self, background_url, output_path):
        """
        从网络下载背景图片并保存到本地临时文件
        
        参数:
            background_url: 背景图片URL
            output_path: 输出视频文件路径，用于确定保存目录
            
        返回:
            local_background_path: 本地保存的图片路径，如果下载失败则返回None
        """
        if not background_url or not background_url.startswith(("http://", "https://", "ftp://")):
            # 如果不是网络URL，直接返回原路径
            return background_url
            
        try:
            import requests

            # 临时文件目录
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            # 设置临时图片名称
            local_background_path =  f"{os.path.splitext(output_path)[0]}_background.png"
            
            logger.info(f"开始下载背景图片: {background_url}")
            
            # 设置请求头，部分网站需要UA才能访问
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # 下载图片，带超时和重试
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = requests.get(background_url, stream=True, timeout=30, headers=headers)
                    response.raise_for_status()  # 确保请求成功
                    
                    # 检查内容类型是否为图片
                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith('image/'):
                        logger.warning(f"下载的内容不是图片，Content-Type: {content_type}")
                        # 如果不是图片，继续尝试，因为有些服务器可能不正确设置Content-Type
                    
                    with open(local_background_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # 验证文件是否为有效图片
                    try:
                        from PIL import Image
                        Image.open(local_background_path).verify()
                        logger.info(f"背景图片下载成功，保存至: {local_background_path}")
                        return local_background_path
                    except Exception as e:
                        logger.warning(f"下载的文件不是有效图片: {e}")
                        os.remove(local_background_path)
                        retry_count += 1
                        continue
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"下载背景图片超时，重试 {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    time.sleep(1)  # 等待1秒后重试
                except requests.exceptions.RequestException as e:
                    logger.warning(f"下载背景图片失败: {e}，重试 {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    time.sleep(1)  # 等待1秒后重试
            
            logger.error(f"经过 {max_retries} 次重试后仍无法下载背景图片")
            return None
            
        except Exception as e:
            logger.error(f"下载背景图片过程中出现异常: {str(e)}")
            return None

    def _build_ffmpeg_cmd(self, bg_hex, temp_img_path, background_url=None, fps=None, width=None, height=None, top_blank=None, output_path=None):
        """
        构建基础的 FFmpeg 命令，根据是否提供背景图片 URL 来确定使用纯色背景还是图片背景
        
        参数:
            bg_hex: 背景色十六进制格式
            temp_img_path: 临时图像文件路径
            background_url: 背景图片 URL，如果为 None 或不存在则使用纯色背景。支持网络URL，会自动下载
            fps: 帧率，默认使用实例的fps
            width: 视频宽度，默认使用实例的width
            height: 视频高度，默认使用实例的height
            top_blank: 顶部留白，默认使用实例的top_blank
            output_path: 输出视频文件路径，用于确定背景图片保存位置
        
        返回:
            构建好的基础 FFmpeg 命令列表
        """
        # 使用默认参数或传入的参数
        fps = fps or self.fps
        width = width or self.width
        height = height or self.height
        top_blank = top_blank or self.top_blank
        
        # 基础命令部分
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
        ]
        
        # 处理背景图片，如果是网络URL则下载到本地
        local_background_path = None
        if background_url:
            # 下载背景图片（如果是URL）
            local_background_path = self._download_background_image(background_url, output_path)
            
            if not local_background_path or not os.path.exists(local_background_path):
                logger.warning(f"背景图片路径不存在或下载失败: {background_url}，将回退到使用纯色背景")
                local_background_path = None  # 文件不存在则使用纯色背景
        
        # 根据是否有背景图片 URL 来决定输入源
        if local_background_path and os.path.exists(local_background_path):
            # 使用图片作为背景
            ffmpeg_cmd.extend([
                "-loop", "1",
                "-framerate", str(fps),
                "-i", local_background_path,     # 背景图片作为第一个输入
                "-framerate", str(fps),
                "-i", temp_img_path,             # 文本图片作为第二个输入
            ])
            
            logger.info(f"使用背景图片模式: {local_background_path}")
        else:
            # 使用纯色背景
            ffmpeg_cmd.extend([
                "-f", "lavfi", 
                "-i", f"color=c={bg_hex}:s={width}x{height}:r={fps},format=yuv420p,hwupload_cuda",
                "-i", temp_img_path,
                "-f", "lavfi",
                "-i", f"color=c={bg_hex}:s={width}x{top_blank}:r={fps},format=yuv420p,hwupload_cuda",
            ])
            logger.info("使用纯色背景模式")
        
        # 添加进度输出参数
        ffmpeg_cmd.extend([
            "-progress", "pipe:2",  # 输出进度信息到stderr
            "-stats",               # 启用统计信息
            "-stats_period", "1",   # 每1秒输出一次统计信息
        ])
        
        return ffmpeg_cmd, local_background_path

    def create_scrolling_video_overlay_cuda(
        self,
        image,
        output_path,
        text_actual_height,
        preferred_codec="h264_nvenc",
        audio_path=None,
        bg_color=(0, 0, 0, 255),
        background_url=None,
    ):
        """
        使用FFmpeg的overlay_cuda滤镜创建GPU加速的滚动视频
        
        参数:
            image: 要滚动的图像 (PIL.Image或NumPy数组)
            output_path: 输出视频文件路径
            text_actual_height: 文本实际高度
            preferred_codec: 首选视频编码器
            audio_path: 可选的音频文件路径
            bg_color: 背景颜色 (R,G,B) 或 (R,G,B,A)
            background_url: 可选的背景图片路径，如果提供则使用该图片作为背景，否则使用纯色背景
            
        Returns:
            输出视频的路径
        """
        # transparency_required: 是否需要透明通道 默认 False
        transparency_required = False

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
                "compress_level": 6, # 1~9：压缩级别逐步提高，默认6
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
                    err_msg = "未检测到NVIDIA GPU，overlay_cuda滤镜需要NVIDIA GPU支持，将回退到使用CPU的crop滤镜方法"
                    logger.warning(err_msg)
                    raise Exception(err_msg)
                
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
                    err_msg = "系统不支持overlay_cuda滤镜，将回退到使用CPU的crop滤镜方法"
                    logger.warning(err_msg)
                    raise Exception(err_msg)
                
                logger.info("overlay_cuda滤镜检测，通过✅")
                
            except Exception as e:
                err_msg = f"检测CUDA或overlay_cuda滤镜时出错: {e}"
                logger.warning(err_msg)
                raise Exception(err_msg)
            
            # 5. 构建基本FFmpeg命令 (CUDA加速版)
            # 将背景色从RGB(A)转换为十六进制格式 (#RRGGBB)
            bg_hex = "#{:02x}{:02x}{:02x}".format(bg_color[0], bg_color[1], bg_color[2])

            # 使用新函数构建命令
            ffmpeg_cmd, local_background_path = self._build_ffmpeg_cmd(
                bg_hex=bg_hex, 
                temp_img_path=temp_img_path, 
                background_url=background_url, 
                output_path=output_path
            )

            # 添加音频输入（如果有）
            if audio_path and os.path.exists(audio_path):
                ffmpeg_cmd.extend(["-i", audio_path])
            
            # 根据用户提供的命令修改滚动表达式
            # overlay_cuda=x=0:y='if(between(t,2.0,458.85), -((t-2.0)/456.85)*27411, if(lt(t,2.0), 0, -27411))'
            y_expr = f"if(between(t,{scroll_start_time},{scroll_end_time}), {self.top_blank} - ((t-{scroll_start_time})/{scroll_duration})*{img_height-self.height + self.top_blank}, if(lt(t,{scroll_start_time}), {self.top_blank}, -{img_height-self.height + self.top_blank}))"

            # 根据是否有背景图片URL来构建不同的滤镜链
            if local_background_path and os.path.exists(local_background_path):
                # 使用背景图片的滤镜链
                filter_complex = f"[0:v]fps={self.fps},format=yuv420p,hwupload_cuda[bg_cuda]; \
                                  [1:v]fps={self.fps},format=rgba,hwupload_cuda[scroll_cuda]; \
                                  [0:v]fps={self.fps},crop=iw:{self.top_blank}:0:0,hwupload_cuda[top_mask_cuda]; \
                                  [bg_cuda][scroll_cuda]overlay_cuda=x=0:y='{y_expr}'[overlayed_cuda]; \
                                  [overlayed_cuda][top_mask_cuda]overlay_cuda=x=0:y=0[final_cuda]; \
                                  [final_cuda]hwdownload,format=yuv420p[out]"
                
                logger.info("使用背景图片的滤镜链")
            else:
                # 使用纯色背景的滤镜链 - 修复了原始滤镜链中的错误
                filter_complex = f"[1:v]fps={self.fps},format=yuv420p,hwupload_cuda[img_cuda]; \
                                  [0:v][img_cuda]overlay_cuda=x=0:y='{y_expr}'[bg_with_scroll]; \
                                  [bg_with_scroll][2:v]overlay_cuda=x=0:y=0[final_cuda]; \
                                  [final_cuda]hwdownload,format=yuv420p[out]"
                
                logger.info("使用纯色背景的滤镜链")

            ffmpeg_cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[out]"
            ])
            
            # 添加音频映射（如果有）
            if audio_path and os.path.exists(audio_path):
                # 计算音频输入的索引，根据background_url是否存在而不同
                audio_index = 2  # 默认索引
                if background_url and os.path.exists(background_url):
                    audio_index = 2  # 在背景图片模式下，音频是第3个输入(索引2)
                else:
                    audio_index = 3  # 在纯色背景模式下，音频是第4个输入(索引3)
                
                ffmpeg_cmd.extend([
                    "-map", f"{audio_index}:a:0",  # 从指定索引获取音频
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                ])
            else:
                # 如果没有音频，不需要添加音频相关参数
                pass
            
            # 添加视频编码器参数
            ffmpeg_cmd.extend(codec_params)
            
            # 设置帧率
            ffmpeg_cmd.extend(["-r", str(self.fps)])
            
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
                # 删除文本图片临时文件
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                    logger.info(f"已删除临时文件: {temp_img_path}")
                
                # 删除临时下载的背景图片文件
                if local_background_path and os.path.exists(local_background_path):
                    os.remove(local_background_path)
                    logger.info(f"已删除临时背景图片文件: {local_background_path}")
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
                    logger.info(f"已删除临时文件: {temp_img_path}")
                
                # 清理临时背景图片文件
                if 'local_background_path' in locals() and local_background_path and os.path.exists(local_background_path):
                    os.remove(local_background_path)
                    logger.info(f"已删除临时背景图片文件: {local_background_path}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")
                pass
            raise
