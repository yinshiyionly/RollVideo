"""性能监控模块"""

import time
import psutil
import threading
import logging
import numpy as np
from collections import deque
import os

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