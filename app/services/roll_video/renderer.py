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
import concurrent.futures
import time
from collections import deque

logger = logging.getLogger(__name__)

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
        scroll_speed: int = 2,  # 每帧滚动的像素数
        frame_buffer_size: int = 24, # 帧缓冲区大小，默认为fps的80%
        worker_threads: int = 4  # 工作线程数
    ):
        """
        初始化视频渲染器
        
        Args:
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率
            scroll_speed: 滚动速度(像素/帧)
            frame_buffer_size: 帧缓冲区大小
            worker_threads: 用于帧处理的工作线程数
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.scroll_speed = scroll_speed
        self.frame_buffer_size = min(frame_buffer_size, int(fps * 0.8)) # 限制缓冲区大小
        self.worker_threads = max(1, min(worker_threads, os.cpu_count() or 4)) # 限制线程数
    
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

    def _generate_frame(self, frame_data):
        """
        生成单个帧 - 这个函数会被并行调用
        """
        frame_idx, img_array, img_height, img_width, img_start_y, transparency_required, background_frame_rgb = frame_data
        
        img_end_y = min(img_height, img_start_y + self.height)
        frame_start_y = 0
        frame_end_y = img_end_y - img_start_y
        
        if img_start_y < img_end_y and frame_start_y < frame_end_y and frame_end_y <= self.height:
            img_h_slice = slice(img_start_y, img_end_y)
            img_w_slice = slice(0, min(self.width, img_width))
            frame_h_slice = slice(frame_start_y, frame_end_y)
            frame_w_slice = slice(0, min(self.width, img_width))
            source_section = img_array[img_h_slice, img_w_slice]
            
            if transparency_required:
                frame_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
                target_area = frame_rgba[frame_h_slice, frame_w_slice]
                if target_area.shape[:2] == source_section.shape[:2]: 
                    np.copyto(target_area, source_section)
                else: 
                    copy_width = min(target_area.shape[1], source_section.shape[1])
                    target_area[:target_area.shape[0], :copy_width] = source_section[:target_area.shape[0], :copy_width]
                output_frame_data = frame_rgba.tobytes()
            else:
                frame_rgb = background_frame_rgb.copy()
                target_area = frame_rgb[frame_h_slice, frame_w_slice]
                if target_area.shape[:2] == source_section.shape[:2]:
                    # 优化的alpha混合 - 矢量化计算
                    alpha = source_section[:, :, 3:4].astype(np.float32) / 255.0
                    blended = (source_section[:, :, :3].astype(np.float32) * alpha + 
                               target_area.astype(np.float32) * (1.0 - alpha))
                    np.copyto(target_area, blended.astype(np.uint8))
                else:
                    copy_width = min(target_area.shape[1], source_section.shape[1])
                    source_section_crop = source_section[:target_area.shape[0], :copy_width]
                    target_area_crop = target_area[:target_area.shape[0], :copy_width]
                    alpha = source_section_crop[:, :, 3:4].astype(np.float32) / 255.0
                    blended = (source_section_crop[:, :, :3].astype(np.float32) * alpha + 
                               target_area_crop.astype(np.float32) * (1.0 - alpha))
                    target_area[:target_area.shape[0], :copy_width] = blended.astype(np.uint8)
                output_frame_data = frame_rgb.tobytes()
        else:
            if transparency_required: 
                output_frame_data = np.zeros((self.height, self.width, 4), dtype=np.uint8).tobytes()
            else: 
                output_frame_data = background_frame_rgb.tobytes()
        
        return frame_idx, output_frame_data

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
        scroll_distance = max(0, text_actual_height - self.height)
        scroll_frames = int(scroll_distance / self.scroll_speed) if self.scroll_speed > 0 else 0
        padding_frames_start = int(self.fps * 0.5)
        padding_frames_end = int(self.fps * 0.5)
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
            # GPU 参数: 优先尝试H.265/HEVC (如果支持)，性能比H.264更好
            hevc_gpu_params = [
                "-c:v", "hevc_nvenc", # 尝试使用HEVC/H.265 GPU加速
                "-preset", "p3",      # 中性能预设
                "-rc:v", "vbr",       # 变速率控制
                "-cq:v", "24",        # 对于HEVC，质量参数略有不同
                "-b:v", "0",          # 让CQ控制比特率
                "-pix_fmt", "yuv420p",
                "-tag:v", "hvc1",     # 添加兼容标签
                "-movflags", "+faststart"
            ]
            
            # GPU H.264参数
            gpu_params = [
                "-c:v", preferred_codec, 
                "-preset", "p3",
                "-rc:v", "vbr",
                "-cq:v", "21", 
                "-b:v", "0",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart"
            ]
            
            # 选择最合适的GPU编码器
            # 如果首选是h264_nvenc，尝试先用hevc_nvenc
            if preferred_codec == "h264_nvenc":
                video_codec_and_output_params = hevc_gpu_params
                gpu_fallback_params = gpu_params
            else:
                video_codec_and_output_params = gpu_params
                gpu_fallback_params = hevc_gpu_params
            
            # CPU 参数：libx264，针对速度做更多优化
            cpu_fallback_codec_and_output_params = [
                "-c:v", "libx264",
                "-crf", "23",        # 略微降低质量提高速度
                "-preset", "fast",   # 比medium更快的预设
                "-tune", "fastdecode", # 优化解码速度
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart"
            ]
            logger.info(f"设置ffmpeg(不透明): 输入={ffmpeg_pix_fmt}, 输出={output_path}")
            logger.info(f"  首选GPU参数: {' '.join(video_codec_and_output_params)}")
            logger.info(f"  备选GPU参数: {' '.join(gpu_fallback_params)}")
            logger.info(f"  回退CPU参数: {' '.join(cpu_fallback_codec_and_output_params)}")
            
            # 准备背景色 (RGB)
            final_bg_color_rgb = (0, 0, 0)
            if bg_color and len(bg_color) >= 3:
                 final_bg_color_rgb = bg_color[:3]
            background_frame_rgb = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(final_bg_color_rgb, dtype=np.uint8)
        
        # 优化：预计算所有帧的y位置
        def calculate_position(frame_idx):
            if frame_idx < padding_frames_start:
                return 0
            elif frame_idx < padding_frames_start + scroll_frames:
                scroll_progress = frame_idx - padding_frames_start
                pos = scroll_progress * self.scroll_speed
                return min(pos, scroll_distance)
            else:
                return scroll_distance
        
        frame_positions = [calculate_position(i) for i in range(total_frames)]
        
        # --- 使用生产者-消费者模式处理帧生成和编码 ---
        def run_ffmpeg_with_pipe(current_codec_params: List[str], is_gpu_attempt: bool) -> bool:
            ffmpeg_cmd = self._get_ffmpeg_command(output_path, ffmpeg_pix_fmt, current_codec_params, audio_path)
            logger.info(f"执行ffmpeg命令: {' '.join(ffmpeg_cmd)}")
            
            # 创建队列和事件标志
            frame_queue = queue.Queue(maxsize=self.frame_buffer_size)
            error_event = threading.Event()
            finish_event = threading.Event()
            stdout_q = queue.Queue()
            stderr_q = queue.Queue()
            
            # ffmpeg进程
            process = None
            
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd, stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    bufsize=10*1024*1024  # 使用更大的缓冲区
                )
                
                # 启动输出读取线程
                stdout_thread = threading.Thread(target=self._reader_thread, args=(process.stdout, stdout_q), daemon=True)
                stderr_thread = threading.Thread(target=self._reader_thread, args=(process.stderr, stderr_q), daemon=True)
                stdout_thread.start()
                stderr_thread.start()
                
                # 帧生成线程 - 生产者
                def frame_producer():
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_threads) as executor:
                            # 提交所有帧生成任务
                            futures = []
                            for frame_idx in range(total_frames):
                                if error_event.is_set():
                                    break
                                
                                img_start_y = int(frame_positions[frame_idx])
                                frame_data = (frame_idx, img_array, img_height, img_width, 
                                             img_start_y, transparency_required, background_frame_rgb)
                                futures.append(executor.submit(self._generate_frame, frame_data))
                            
                            # 维护有序的结果队列
                            ordered_results = {}
                            next_frame_idx = 0
                            
                            # 处理完成的任务
                            for future in concurrent.futures.as_completed(futures):
                                if error_event.is_set():
                                    break
                                    
                                try:
                                    frame_idx, frame_bytes = future.result()
                                    ordered_results[frame_idx] = frame_bytes
                                    
                                    # 按顺序将帧放入队列
                                    while next_frame_idx in ordered_results and not error_event.is_set():
                                        frame_queue.put(ordered_results.pop(next_frame_idx))
                                        next_frame_idx += 1
                                except Exception as e:
                                    logger.error(f"帧生成错误: {e}")
                                    error_event.set()
                    except Exception as e:
                        logger.error(f"帧生成主线程错误: {e}")
                        error_event.set()
                    finally:
                        finish_event.set()  # 设置帧生成完成标志
                
                # 帧写入线程 - 消费者
                def frame_consumer():
                    codec_name = "unknown"
                    try: codec_name = current_codec_params[current_codec_params.index("-c:v") + 1]
                    except (ValueError, IndexError): pass
                    
                    progress_bar = tqdm.tqdm(total=total_frames, desc=f"编码 ({codec_name}) ")
                    
                    try:
                        frames_written = 0
                        while frames_written < total_frames:
                            if error_event.is_set():
                                break
                            
                            # 等待帧或结束信号
                            try:
                                frame_data = frame_queue.get(timeout=0.5)
                                process.stdin.write(frame_data)
                                frames_written += 1
                                progress_bar.update(1)
                                frame_queue.task_done()
                            except queue.Empty:
                                # 检查是否所有帧都已生成
                                if finish_event.is_set() and frames_written >= total_frames:
                                    break
                            except (BrokenPipeError, IOError) as e:
                                logger.error(f"写入ffmpeg管道错误: {e}")
                                error_event.set()
                                break
                    except Exception as e:
                        logger.error(f"帧写入线程错误: {e}")
                        error_event.set()
                    finally:
                        progress_bar.close()
                        # 确保stdin关闭
                        if process and process.stdin and not process.stdin.closed:
                            try:
                                process.stdin.close()
                            except Exception as e:
                                logger.error(f"关闭stdin错误: {e}")
                
                # 启动生产者和消费者线程
                producer_thread = threading.Thread(target=frame_producer, daemon=True)
                consumer_thread = threading.Thread(target=frame_consumer, daemon=True)
                producer_thread.start()
                consumer_thread.start()
                
                # 等待线程完成
                producer_thread.join()
                consumer_thread.join()
                
                # 检查是否发生错误
                if error_event.is_set():
                    logger.error("编码过程中发生错误")
                    return False
                
                # 等待ffmpeg进程完成
                logger.info("等待ffmpeg进程结束...")
                process.wait()
                return_code = process.returncode
                
                # 收集输出
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
                
                if return_code == 0:
                    logger.info(f"使用 {codec_name} 编码成功完成。")
                    return True
                else:
                    logger.error(f"ffmpeg ({codec_name}) 执行失败，返回码: {return_code}")
                    if is_gpu_attempt: logger.warning("GPU编码失败提示：检查ffmpeg版本/驱动/显存。")
                    return False
                
            except FileNotFoundError: 
                logger.error("ffmpeg 未找到。请确保已安装并加入PATH。")
                raise
            except Exception as e: 
                logger.error(f"执行 ffmpeg 时出错: {e}", exc_info=True)
                logger.error(f"命令: {' '.join(ffmpeg_cmd)}")
                # 尝试收集最后的 stderr
                stderr_lines_on_except = []
                while True:
                    try: line = stderr_q.get(timeout=0.1)
                    except queue.Empty: break
                    if line is None: break
                    stderr_lines_on_except.append(line.decode(errors='ignore').strip())
                
                if stderr_lines_on_except: 
                    stderr_content_on_except = "\n".join(stderr_lines_on_except)
                    logger.error(f"ffmpeg stderr (异常时):\n{stderr_content_on_except}")
                return False
            finally: 
                # 确保线程和进程清理
                error_event.set()  # 通知所有线程退出
                
                # 清理进程
                if process and process.poll() is None: 
                    logger.warning("尝试终止 ffmpeg 进程...")
                    try: process.terminate() 
                    except ProcessLookupError: pass
                    try: process.wait(timeout=1) 
                    except subprocess.TimeoutExpired: 
                        logger.warning("ffmpeg 进程超时，强制终止")
                        try: process.kill() 
                        except ProcessLookupError: pass
                    except Exception as e_wait: 
                        logger.error(f"等待终止时出错: {e_wait}")
                
                # 关闭管道
                for pipe in [process.stdin, process.stdout, process.stderr] if process else []:
                    if pipe and not pipe.closed:
                        try: pipe.close()
                        except: pass
        
        # --- 执行编码 --- 
        success = run_ffmpeg_with_pipe(video_codec_and_output_params, is_gpu_attempt=(not transparency_required))
        
        # 如果 HEVC 失败，尝试 H.264 GPU编码
        if not success and not transparency_required and preferred_codec == "h264_nvenc":
            logger.info(f"HEVC GPU编码失败，尝试回退到H.264 GPU...")
            success = run_ffmpeg_with_pipe(gpu_fallback_params, is_gpu_attempt=True)
        
        # 如果 GPU 失败，尝试 CPU 编码
        if not success and not transparency_required and cpu_fallback_codec_and_output_params:
            logger.info(f"GPU编码失败，尝试回退到CPU编码...")
            success = run_ffmpeg_with_pipe(cpu_fallback_codec_and_output_params, is_gpu_attempt=False)
            if not success:
                 logger.error("CPU回退编码也失败了。")
                 raise Exception("视频编码失败（GPU和CPU均失败）")
        elif not success and transparency_required:
             logger.error("透明视频 (CPU prores_ks) 编码失败。")
             raise Exception("透明视频编码失败")
             
        logger.info(f"视频渲染流程完成。输出文件: {output_path}")
        return output_path 