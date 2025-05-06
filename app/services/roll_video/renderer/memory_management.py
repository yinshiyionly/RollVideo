"""内存管理模块"""

import numpy as np
import queue
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import threading

logger = logging.getLogger(__name__)

class SharedMemoryFramePool:
    """基于共享内存的帧内存池，支持多进程高效访问"""
    
    def __init__(self, width, height, channels, pool_size=120):
        """
        初始化基于共享内存的帧内存池
        
        Args:
            width: 帧宽度
            height: 帧高度
            channels: 颜色通道数(RGB=3, RGBA=4)
            pool_size: 内存池大小(帧数)
        """
        self.width = width
        self.height = height
        self.channels = channels
        self.frame_shape = (height, width, channels)
        self.nbytes_per_frame = height * width * channels
        self.pool_size = pool_size
        
        # 分配共享内存
        total_size = self.nbytes_per_frame * pool_size
        self.shm = shared_memory.SharedMemory(create=True, size=total_size)
        self.shm_name = self.shm.name
        
        logger.info(f"创建共享内存帧池: {pool_size}帧, 总大小={total_size/(1024*1024):.2f}MB, 名称={self.shm_name}")
        
        # 共享队列用于跟踪可用帧索引
        self.available_frames = mp.Queue(pool_size)
        self.lock = mp.Lock()
        self.in_use_count = mp.Value('i', 0)
        
        # 预填充帧索引
        for i in range(pool_size):
            self.available_frames.put(i)
    
    def get_frame(self):
        """获取一个共享内存帧，返回帧索引和NumPy数组视图"""
        try:
            with self.lock:
                frame_idx = self.available_frames.get(block=False)
                self.in_use_count.value += 1
            
            # 计算帧在共享内存中的起始位置
            offset = frame_idx * self.nbytes_per_frame
            
            # 创建指向该区域的NumPy数组视图
            array_view = np.ndarray(
                shape=self.frame_shape,
                dtype=np.uint8,
                buffer=self.shm.buf[offset:offset + self.nbytes_per_frame]
            )
            
            return frame_idx, array_view
        
        except queue.Empty:
            logger.warning("共享内存帧池耗尽，动态分配新帧")
            # 无法从池获取，分配新帧（不共享）
            return -1, np.zeros(self.frame_shape, dtype=np.uint8)
    
    def release_frame(self, frame_idx):
        """将帧索引归还给池"""
        if frame_idx >= 0 and frame_idx < self.pool_size:
            with self.lock:
                self.available_frames.put(frame_idx)
                self.in_use_count.value -= 1
    
    def cleanup(self):
        """清理共享内存资源"""
        try:
            logger.info(f"清理共享内存帧池: {self.shm_name}")
            self.shm.close()
            self.shm.unlink()
        except Exception as e:
            logger.error(f"清理共享内存帧池失败: {e}")
    
    def init_worker(self):
        """在新进程中初始化对共享内存的访问"""
        try:
            if not hasattr(self, 'shm_worker'):
                self.shm_worker = shared_memory.SharedMemory(name=self.shm_name)
                logger.debug(f"工作进程连接到共享内存帧池: {self.shm_name}")
        except Exception as e:
            logger.error(f"工作进程连接共享内存帧池失败: {e}")

class FrameMemoryPool:
    """高效内存池管理器，实现零拷贝内存分配"""
    
    def __init__(self, width, height, channels, pool_size=240):
        """
        初始化帧内存池
        
        Args:
            width: 帧宽度
            height: 帧高度
            channels: 颜色通道数(RGB=3, RGBA=4)
            pool_size: 内存池大小(帧数)
        """
        self.width = width
        self.height = height
        self.channels = channels
        self.frame_shape = (height, width, channels)
        self.frame_size = width * height * channels
        
        # 创建大型连续内存块
        self.buffer_size = self.frame_size * pool_size
        self.buffer = bytearray(self.buffer_size)
        self.buffer_view = memoryview(self.buffer)
        
        # 跟踪可用/已用帧
        self.available_frames = queue.Queue(pool_size)
        self.in_use_frames = set()
        
        # 预填充可用帧队列
        for i in range(pool_size):
            offset = i * self.frame_size
            self.available_frames.put((i, offset))
        
        logger.info(f"初始化内存池: {pool_size}帧, 每帧{self.frame_size/1024/1024:.2f}MB")
    
    def get_frame(self):
        """获取预分配的帧内存"""
        try:
            frame_id, offset = self.available_frames.get(block=False)
            self.in_use_frames.add(frame_id)
            # 返回NumPy视图，无需复制
            array_view = np.frombuffer(
                self.buffer_view[offset:offset+self.frame_size], 
                dtype=np.uint8
            ).reshape(self.frame_shape)
            return frame_id, array_view
        except queue.Empty:
            # 如果池已耗尽，创建新内存
            logger.warning("内存池耗尽，临时分配新内存")
            return -1, np.zeros(self.frame_shape, dtype=np.uint8)
    
    def release_frame(self, frame_id):
        """释放帧回内存池"""
        if frame_id >= 0 and frame_id in self.in_use_frames:
            self.in_use_frames.remove(frame_id)
            offset = frame_id * self.frame_size
            self.available_frames.put((frame_id, offset))
    
    def clear(self):
        """清空并重置内存池"""
        # 释放所有帧
        self.in_use_frames.clear()
        # 清空队列
        while not self.available_frames.empty():
            try:
                self.available_frames.get(block=False)
            except queue.Empty:
                break
        # 重新填充队列
        for i in range(self.available_frames.maxsize):
            offset = i * self.frame_size
            self.available_frames.put((i, offset))

class FrameBuffer:
    """线程安全的帧缓冲区，用于管理异步处理的帧"""
    
    def __init__(self, max_buffer_size=240):
        """
        初始化帧缓冲区
        
        Args:
            max_buffer_size: 缓冲区最大容量
        """
        self.buffer = {}  # 帧索引 -> 帧数据映射
        self.lock = threading.Lock()
        self.max_size = max_buffer_size
        self.next_frame_idx = 0  # 下一个要写入的帧索引
    
    def add_frame(self, frame_idx, frame_data):
        """添加帧到缓冲区"""
        with self.lock:
            self.buffer[frame_idx] = frame_data
            return len(self.buffer)
    
    def get_next_frame(self):
        """获取并移除下一个要处理的帧，如果不可用则返回None"""
        with self.lock:
            if self.next_frame_idx in self.buffer:
                frame = self.buffer.pop(self.next_frame_idx)
                self.next_frame_idx += 1
                return self.next_frame_idx - 1, frame
            return None, None
    
    def has_next_frame(self):
        """检查是否有下一帧可用"""
        with self.lock:
            return self.next_frame_idx in self.buffer
    
    def get_buffer_size(self):
        """获取当前缓冲区大小"""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()
    
    def get_all_frames_in_order(self):
        """获取所有缓冲的帧，按顺序排列"""
        with self.lock:
            sorted_keys = sorted(self.buffer.keys())
            return [(idx, self.buffer[idx]) for idx in sorted_keys] 