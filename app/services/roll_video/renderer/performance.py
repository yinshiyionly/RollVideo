"""性能监控模块"""

import time
import psutil
import threading
import logging
import numpy as np
from collections import deque
import os
import re
import sys
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控类，用于跟踪和报告渲染性能"""
    
    def __init__(self, history_size=120):
        """
        初始化性能监控器
        
        Args:
            history_size: 历史记录大小，用于计算平均值
        """
        self.start_time = time.time()
        self.process = psutil.Process()
        self.history_size = history_size
        
        # 性能指标
        self.cpu_percent_history = deque(maxlen=history_size)
        self.memory_usage_history = deque(maxlen=history_size)
        self.fps_history = deque(maxlen=history_size)
        self.frame_time_history = deque(maxlen=history_size)
        
        # 帧计数器
        self.total_frames = 0
        self.last_frame_time = time.time()
        self.last_report_time = time.time()
        self.last_frame_count = 0
        
        # 共享内存指标
        self.frame_buffer_size_history = deque(maxlen=history_size)
        self.worker_frame_times = {}  # worker_id -> 处理时间列表
        
        # 标志
        self.running = False
        self.monitor_thread = None
        
        # 状态变量
        self.peak_memory = 0
        self.peak_cpu = 0
        self.min_fps = float('inf')
        self.max_fps = 0
        self.avg_fps = 0
        
        self.stats = {}
        self.reset()

    def reset(self):
        """重置性能统计数据"""
        self.stats = {
            "start_time": 0,
            "end_time": 0,
            "duration": 0,
            "frames_processed": 0,
            "frames_per_second": 0,
            "memory_peak": 0,
            "cpu_percent_avg": 0,
        }
        self.timers = {}

    def start(self, name="default"):
        """开始计时"""
        self.timers[name] = {"start": time.time(), "end": 0, "duration": 0}
        if name == "default" or not self.start_time:
            self.start_time = time.time()
            self.stats["start_time"] = self.start_time

    def stop(self, name="default", frames_processed=None):
        """停止计时并记录性能数据"""
        if name in self.timers:
            self.timers[name]["end"] = time.time()
            self.timers[name]["duration"] = (
                self.timers[name]["end"] - self.timers[name]["start"]
            )

        if name == "default" or not self.stats["end_time"]:
            self.stats["end_time"] = time.time()
            self.stats["duration"] = self.stats["end_time"] - self.stats["start_time"]

        # 如果提供了已处理的帧数，计算FPS
        if frames_processed:
            self.stats["frames_processed"] = frames_processed
            self.stats["frames_per_second"] = (
                frames_processed / self.stats["duration"] if self.stats["duration"] > 0 else 0
            )

        # 获取进程信息
        process = psutil.Process(os.getpid())
        self.stats["memory_peak"] = process.memory_info().rss / (1024 * 1024)  # MB
        self.stats["cpu_percent_avg"] = process.cpu_percent()

        return self.stats

    def report(self):
        """生成性能报告"""
        logger.info("\n" + "=" * 50)
        logger.info("性能统计报告:")
        logger.info(f"总时间: {self.stats['duration']:.2f} 秒")
        logger.info(f"处理帧数: {self.stats['frames_processed']}")
        logger.info(f"平均帧率: {self.stats['frames_per_second']:.2f} FPS")
        logger.info(f"内存峰值: {self.stats['memory_peak']:.2f} MB")
        logger.info(f"CPU平均使用率: {self.stats['cpu_percent_avg']:.2f}%")

        # 输出各阶段计时
        if self.timers:
            logger.info("\n各阶段耗时:")
            for name, timer in self.timers.items():
                if name != "default" and "duration" in timer:
                    percent = (
                        timer["duration"] / self.stats["duration"] * 100
                        if self.stats["duration"] > 0
                        else 0
                    )
                    logger.info(
                        f"- {name}: {timer['duration']:.2f} 秒 ({percent:.1f}%)"
                    )

        logger.info("=" * 50 + "\n")
    
    def start(self, interval=1.0):
        """启动后台监控线程"""
        self.running = True
        self.start_time = time.time()
        
        def _monitor_loop():
            while self.running:
                try:
                    # 收集CPU使用率
                    self.cpu_percent_history.append(self.process.cpu_percent(interval=0.1))
                    self.peak_cpu = max(self.peak_cpu, self.cpu_percent_history[-1])
                    
                    # 收集内存使用
                    mem_info = self.process.memory_info()
                    current_memory = mem_info.rss / (1024 * 1024)  # MB
                    self.memory_usage_history.append(current_memory)
                    self.peak_memory = max(self.peak_memory, current_memory)
                    
                    # 计算平均值
                    avg_cpu = sum(self.cpu_percent_history) / len(self.cpu_percent_history) if self.cpu_percent_history else 0
                    avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else 0
                    
                    # 休眠
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"性能监控异常: {e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=_monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("性能监控已启动")
    
    def stop(self):
        """停止监控线程"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # 计算总体性能
        end_time = time.time()
        total_time = end_time - self.start_time
        self.avg_fps = self.total_frames / total_time if total_time > 0 else 0
        
        logger.info(f"性能监控已停止，总计: {self.total_frames}帧, {total_time:.2f}秒, {self.avg_fps:.2f}fps")
    
    def record_frame_processed(self, worker_id=None):
        """
        记录一帧处理完成
        
        Args:
            worker_id: 可选的工作进程ID
        """
        now = time.time()
        frame_time = now - self.last_frame_time
        self.last_frame_time = now
        self.frame_time_history.append(frame_time)
        self.total_frames += 1
        
        # 记录工作进程性能
        if worker_id is not None:
            if worker_id not in self.worker_frame_times:
                self.worker_frame_times[worker_id] = deque(maxlen=30)
            self.worker_frame_times[worker_id].append(frame_time)
        
        # 定期计算FPS
        elapsed = now - self.last_report_time
        if elapsed >= 1.0:  # 每秒更新一次
            frames_since_last = self.total_frames - self.last_frame_count
            current_fps = frames_since_last / elapsed
            self.fps_history.append(current_fps)
            
            # 更新统计
            self.min_fps = min(self.min_fps, current_fps) if self.min_fps != float('inf') else current_fps
            self.max_fps = max(self.max_fps, current_fps)
            
            # 重置
            self.last_report_time = now
            self.last_frame_count = self.total_frames
    
    def record_buffer_size(self, size):
        """记录帧缓冲区大小"""
        self.frame_buffer_size_history.append(size)
    
    def log_stats(self, logger, detailed=False):
        """
        记录当前性能统计信息
        
        Args:
            logger: 日志记录器
            detailed: 是否记录详细信息
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # 基本性能
        avg_cpu = sum(self.cpu_percent_history) / len(self.cpu_percent_history) if self.cpu_percent_history else 0
        avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else 0
        current_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        logger.info(
            f"性能 [{elapsed:.1f}s]: "
            f"CPU {avg_cpu:.1f}% (峰值 {self.peak_cpu:.1f}%), "
            f"内存 {avg_memory:.1f}MB (峰值 {self.peak_memory:.1f}MB), "
            f"FPS {current_fps:.2f} (共 {self.total_frames}帧)"
        )
        
        # 缓冲区使用情况
        if self.frame_buffer_size_history:
            avg_buffer = sum(self.frame_buffer_size_history) / len(self.frame_buffer_size_history)
            max_buffer = max(self.frame_buffer_size_history)
            logger.info(f"缓冲状态: 平均大小 {avg_buffer:.1f}, 峰值 {max_buffer}")
        
        # 详细信息
        if detailed and self.worker_frame_times:
            worker_stats = []
            for worker_id, times in self.worker_frame_times.items():
                if times:
                    avg_time = sum(times) / len(times)
                    worker_stats.append(f"W{worker_id}: {1/avg_time:.1f}fps")
            if worker_stats:
                logger.info(f"工作进程性能: {', '.join(worker_stats)}")
    
    def get_summary(self):
        """获取性能摘要"""
        elapsed = time.time() - self.start_time
        
        return {
            "elapsed_time": elapsed,
            "total_frames": self.total_frames,
            "avg_fps": self.total_frames / elapsed if elapsed > 0 else 0,
            "peak_memory_mb": self.peak_memory,
            "peak_cpu_percent": self.peak_cpu,
            "min_fps": self.min_fps if self.min_fps != float('inf') else 0,
            "max_fps": self.max_fps
        }

    @staticmethod
    def monitor_ffmpeg_progress(process, total_duration, total_frames, encoding_start_time=None):
        """
        监控FFmpeg进度并显示进度条
        
        参数:
            process: FFmpeg进程对象
            total_duration: 视频总时长（秒）
            total_frames: 视频总帧数
            encoding_start_time: 编码开始时间（如果为None则使用当前时间）
        
        返回:
            监控线程对象
        """
        if encoding_start_time is None:
            encoding_start_time = time.time()
            
        # 创建进度监控线程
        def progress_monitor_thread():
            # 保存进度正则表达式模式
            frame_pattern = re.compile(r"frame=\s*(\d+)")
            time_pattern = re.compile(r"time=\s*(\d+:\d+:\d+\.\d+)")
            fps_pattern = re.compile(r"fps=\s*(\d+\.?\d*)")
            speed_pattern = re.compile(r"speed=\s*([\d.]+)x")
            
            # 创建TQDM进度条（精简格式，自动适应屏幕宽度）
            pbar = tqdm(
                total=100, 
                desc="FFmpeg处理进度", 
                bar_format='{l_bar}{bar}| {n:3.1f}% [{elapsed}<{remaining}{postfix}]',
                unit="%",
                ncols=None
            )
            pbar.set_postfix(帧="0/0", 剩余="未知")
            
            last_progress = 0
            last_frame = 0
            
            try:
                while process.poll() is None:
                    # 非阻塞方式读取stderr
                    stderr_line = process.stderr.readline()
                    if not stderr_line:
                        time.sleep(0.1)  # 减少CPU使用
                        continue
                        
                    stderr_line = stderr_line.strip()
                    
                    # 解析进度信息
                    frame_match = frame_pattern.search(stderr_line)
                    time_match = time_pattern.search(stderr_line)
                    fps_match = fps_pattern.search(stderr_line)
                    speed_match = speed_pattern.search(stderr_line)
                    
                    # 更新进度信息
                    if time_match:
                        # 从时间字符串(HH:MM:SS.MS)解析时间
                        time_str = time_match.group(1)
                        time_parts = time_str.split(':')
                        if len(time_parts) == 3:
                            hours = int(time_parts[0])
                            minutes = int(time_parts[1])
                            seconds = float(time_parts[2])
                            current_time_sec = hours * 3600 + minutes * 60 + seconds
                            
                            # 计算实际进度百分比
                            progress = min(1.0, current_time_sec / total_duration) * 100
                            
                            # 更新进度条
                            if progress > last_progress:
                                pbar.update(progress - last_progress)
                                last_progress = progress
                            
                            # 获取当前帧数和速度
                            current_frame = int(frame_match.group(1)) if frame_match else 0
                            fps = float(fps_match.group(1)) if fps_match else 0
                            speed = float(speed_match.group(1)) if speed_match else 0
                            
                            # 计算已用时间和预计剩余时间
                            elapsed = time.time() - encoding_start_time
                            
                            if progress > 0:
                                eta = elapsed / (progress/100) - elapsed
                                eta_str = f"{eta:.1f}秒"
                            else:
                                eta_str = "未知"
                            
                            # 更新进度条后缀信息（精简格式）
                            pbar.set_postfix(
                                帧=f"{current_frame}/{total_frames}", 
                                剩余=eta_str
                            )
                            
                            # 记录日志（不那么频繁）
                            if current_frame - last_frame >= total_frames / 20:  # 每完成约5%记录一次
                                logger.info(
                                    f"FFmpeg进度: {progress:.1f}% | "
                                    f"时间: {time_str}/{total_duration:.1f}秒 | "
                                    f"帧: {current_frame}/{total_frames} | "
                                    f"速度: {speed:.1f}x | "
                                    f"FPS: {fps:.1f} | "
                                    f"预计剩余: {eta_str}"
                                )
                                last_frame = current_frame
                    
                # 处理完成，关闭进度条
                pbar.update(100 - last_progress)  # 确保进度达到100%
                pbar.close()
                
                # 输出完成信息
                elapsed = time.time() - encoding_start_time
                logger.info(f"FFmpeg处理完成，总用时: {elapsed:.2f}秒")
                
            except Exception as e:
                logger.error(f"监控FFmpeg进度时出错: {str(e)}")
                
                # 关闭进度条
                try:
                    pbar.close()
                except:
                    pass
            
        # 创建并启动监控线程
        monitor_thread = threading.Thread(target=progress_monitor_thread)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return monitor_thread

def log_system_info():
    """记录系统信息"""
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "current_frequency": psutil.cpu_freq(),
    }
    
    memory = psutil.virtual_memory()
    memory_info = {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "percent_used": memory.percent
    }
    
    disk = psutil.disk_usage('/')
    disk_info = {
        "total_gb": disk.total / (1024**3),
        "free_gb": disk.free / (1024**3),
        "percent_used": disk.percent
    }
    
    # 尝试获取GPU信息
    gpu_info = "未检测到GPU信息"
    try:
        # 尝试使用nvidia-smi获取信息
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
    except:
        pass
    
    logger.info(f"系统信息:")
    logger.info(f"CPU: {cpu_info['physical_cores']}物理核心, {cpu_info['logical_cores']}逻辑核心")
    logger.info(f"内存: 总计 {memory_info['total_gb']:.1f}GB, 可用 {memory_info['available_gb']:.1f}GB ({memory_info['percent_used']}% 已使用)")
    logger.info(f"磁盘: 总计 {disk_info['total_gb']:.1f}GB, 可用 {disk_info['free_gb']:.1f}GB ({disk_info['percent_used']}% 已使用)")
    logger.info(f"GPU: {gpu_info}")

class FrameProcessingTracker:
    """跟踪帧处理性能"""
    
    def __init__(self):
        self.processing_times = {}  # 帧索引 -> 处理时间
        self.start_times = {}       # 帧索引 -> 开始时间
        self.write_times = {}       # 帧索引 -> 写入时间
        self.queue_times = {}       # 帧索引 -> 排队时间
        self.lock = threading.RLock()
    
    def start_frame(self, frame_idx):
        """记录帧开始处理的时间"""
        with self.lock:
            self.start_times[frame_idx] = time.time()
    
    def end_frame(self, frame_idx):
        """记录帧处理完成的时间"""
        with self.lock:
            if frame_idx in self.start_times:
                self.processing_times[frame_idx] = time.time() - self.start_times[frame_idx]
                return self.processing_times[frame_idx]
        return 0
    
    def frame_queued(self, frame_idx):
        """记录帧排队的时间"""
        with self.lock:
            self.queue_times[frame_idx] = time.time()
    
    def frame_written(self, frame_idx):
        """记录帧写入的时间"""
        with self.lock:
            self.write_times[frame_idx] = time.time()
    
    def get_stats(self):
        """获取处理统计信息"""
        with self.lock:
            processing_times = list(self.processing_times.values())
            
            # 计算从排队到写入的延迟
            latencies = []
            for frame_idx in self.write_times:
                if frame_idx in self.queue_times:
                    latencies.append(self.write_times[frame_idx] - self.queue_times[frame_idx])
            
            stats = {
                "avg_processing_time": np.mean(processing_times) if processing_times else 0,
                "min_processing_time": min(processing_times) if processing_times else 0,
                "max_processing_time": max(processing_times) if processing_times else 0,
                "frames_processed": len(processing_times),
                "avg_latency": np.mean(latencies) if latencies else 0,
                "max_latency": max(latencies) if latencies else 0,
            }
            return stats
    
    def log_stats(self, logger):
        """记录处理统计信息"""
        stats = self.get_stats()
        
        logger.info(
            f"帧处理统计: "
            f"{stats['frames_processed']}帧, "
            f"平均处理时间: {stats['avg_processing_time']*1000:.1f}ms, "
            f"范围: [{stats['min_processing_time']*1000:.1f}-{stats['max_processing_time']*1000:.1f}]ms, "
            f"平均延迟: {stats['avg_latency']*1000:.1f}ms, "
            f"最大延迟: {stats['max_latency']*1000:.1f}ms"
        )