"""内存管理模块"""

import numpy as np
import queue
import logging

logger = logging.getLogger(__name__)

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