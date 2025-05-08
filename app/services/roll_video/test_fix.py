#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试滚动视频修复

此脚本测试修复后的overlay_cuda滤镜是否正确显示文本
"""

import os
import sys
import logging
import time
from datetime import datetime

# 添加父目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导入RollVideo服务
from app.services.roll_video.roll_video_service import RollVideoService

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def main():
    # 创建简单测试文本
    sample_text = """这是一个测试文本，
用于验证滚动视频是否正确显示。
如果修复成功，
文本应该从视频开始就能看到，
然后平滑滚动。

这是第二段文本，
确保滚动效果正常。
"""

    # 输出目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 创建服务实例
    service = RollVideoService()

    # 输出文件名
    base_file_name = f"test_fix_{time.strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join(output_dir, base_file_name + ".mp4") 

    # 记录开始时间
    start_time = time.time()
    
    # 测试修复的overlay_cuda滤镜
    result = service.create_roll_video_overlay_cuda(
        text=sample_text,
        output_path=output_path,
        width=720,
        height=1280,
        font_path="方正黑体简体.ttf",
        font_size=40,
        font_color=(0, 0, 0),
        bg_color=(255, 255, 255, 1.0),
        line_spacing=20,
        char_spacing=10,
        fps=30,
        scroll_speed=1,
        scroll_effect="basic",  # 使用基本滚动效果
        scroll_direction="bottom_to_top"  # 从下到上滚动
    )
    
    # 记录结束时间和总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    # 输出结果
    if result.get("status") == "success":
        logger.info(f"测试成功: 视频创建完成")
        logger.info(f"输出视频路径: {result.get('output_path')}")
        logger.info(f"总耗时: {total_time:.2f}秒")
    else:
        logger.error(f"测试失败: {result.get('message')}")

if __name__ == "__main__":
    main() 