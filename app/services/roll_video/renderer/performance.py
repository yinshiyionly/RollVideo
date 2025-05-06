"""性能监控模块"""

import time
import logging
import statistics
from collections import deque

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """监控和分析渲染性能"""
    
    def __init__(self, window_size=30):
        """
        初始化性能监控器
        
        Args:
            window_size: 记录性能数据的窗口大小
        """
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