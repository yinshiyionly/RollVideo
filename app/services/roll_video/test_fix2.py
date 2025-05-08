#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试滚动视频修复 - 版本2

此脚本专门测试从下到上滚动的修复效果
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
    sample_text = """示例文本第1行，
示例文本第2行，
示例文本第3行，
示例文本第4行，
示例文本第5行，
示例文本第6行，
示例文本第7行，
示例文本第8行，
示例文本第9行，
示例文本第10行，
示例文本第11行，
示例文本第12行，
示例文本第13行，
示例文本第14行，
示例文本第15行，
示例文本第16行，
示例文本第17行，
示例文本第18行，
示例文本第19行，
示例文本第20行。
"""

    # 输出目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 创建服务实例
    service = RollVideoService()

    # 分别测试crop和overlay_cuda
    test_methods = ["crop", "overlay_cuda"]
    
    for method in test_methods:
        logger.info(f"开始测试 {method} 方法...")
        
        # 输出文件名
        base_file_name = f"test_fix2_{method}_{time.strftime('%Y%m%d_%H%M%S')}"
        output_path = os.path.join(output_dir, base_file_name + ".mp4") 
    
        # 记录开始时间
        start_time = time.time()
        
        # 测试参数
        test_params = {
            "text": sample_text,
            "output_path": output_path,
            "width": 720,
            "height": 1280,
            "font_path": "方正黑体简体.ttf",
            "font_size": 40,
            "font_color": (0, 0, 0),
            "bg_color": (255, 255, 255, 1.0),
            "line_spacing": 20,
            "char_spacing": 10,
            "fps": 30,
            "scroll_speed": 1,
            "scroll_direction": "bottom_to_top"  # 从下到上滚动
        }
        
        if method == "overlay_cuda":
            # 添加overlay_cuda特有参数
            test_params["scroll_effect"] = "basic"
            result = service.create_roll_video_overlay_cuda(**test_params)
        else:
            # 使用crop方法
            result = service.create_roll_video_crop(**test_params)
        
        # 记录结束时间和总耗时
        end_time = time.time()
        total_time = end_time - start_time
        
        # 输出结果
        if result.get("status") == "success":
            logger.info(f"{method} 测试成功: 视频创建完成")
            logger.info(f"输出视频路径: {result.get('output_path')}")
            logger.info(f"总耗时: {total_time:.2f}秒")
        else:
            logger.error(f"{method} 测试失败: {result.get('message')}")
        
        logger.info(f"{'-'*50}")

if __name__ == "__main__":
    main() 