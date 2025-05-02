#!/usr/bin/env python
# 测试优化后的视频渲染器性能

from app.services.roll_video.renderer import (
    TextRenderer, 
    VideoRenderer, 
    fast_frame_processor,
    _process_frame_optimized
)
from PIL import Image
import numpy as np
import time
import os
import sys

def test_renderer_components():
    print("=== 组件性能测试 ===")
    
    # 创建测试图像
    print("创建测试图像...")
    img = Image.new('RGBA', (640, 480), (255, 255, 255, 255))
    img_array = np.array(img)
    
    # 初始化渲染器
    print("初始化VideoRenderer...")
    v = VideoRenderer(width=640, height=480, fps=30, scroll_speed=5)
    
    # 测试内存池
    print("测试内存池性能...")
    start = time.time()
    pool = v._init_memory_pool(channels=4, pool_size=120)
    
    # 分配和释放100个帧
    for i in range(100):
        frame_id, frame = pool.get_frame()
        frame.fill(i % 255)  # 做一些操作
        pool.release_frame(frame_id)
    
    mem_time = time.time() - start
    print(f"内存池操作 100 帧耗时: {mem_time:.6f}秒, {100/mem_time:.2f} 帧/秒")
    
    # 测试帧处理函数性能
    print("\n测试帧处理函数性能...")
    start = time.time()
    frames_to_process = 100
    
    # 设置全局图像数组
    import app.services.roll_video.renderer as renderer
    renderer._g_img_array = img_array
    
    # 处理多个帧
    for i in range(frames_to_process):
        frame_idx, frame = _process_frame_optimized(
            (i, i % 400, 480, 640, 480, 640, True, (0, 0, 0))
        )
    
    proc_time = time.time() - start
    print(f"处理 {frames_to_process} 帧耗时: {proc_time:.6f}秒, {frames_to_process/proc_time:.2f} 帧/秒")
    
    # 清理
    renderer._g_img_array = None
    
    print("\n=== 所有测试完成 ===")

if __name__ == "__main__":
    test_renderer_components() 