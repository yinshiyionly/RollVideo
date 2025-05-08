#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RollVideo测试示例

此脚本演示了如何使用RollVideo服务创建滚动文本视频
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
    # 示例文本 - 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_text_path = os.path.join(current_dir, "example.txt")
    sample_text = open(sample_text_path, "r").read()

    # 路径
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 创建服务实例
    service = RollVideoService()

    # 定义测试场景参数列表
    test_cases = [
        {
            "description": "白底黑字（GPU加速 -> MP4）- overlay_cuda GPU加速方法",
            "method": "overlay_cuda", 
            "params": {
                "text": sample_text,
                "width": 720,
                "height": 1280,
                "font_path": "方正黑体简体.ttf",
                "font_size": 30,
                "font_color": [0,0,0],
                "bg_color": [255,255,255,1.0],  # 不透明
                "line_spacing": 20,
                "char_spacing": 10,
                "fps": 30,
                "scroll_speed": 1,
                "scroll_effect": "basic",  # 基础匀速滚动
            }
        },
        {
            "description": "深蓝底白字（GPU加速 + 加减速效果 -> MP4）- overlay_cuda GPU加速方法",
            "method": "ffmpeg",
            "params": {
                "text": sample_text,
                "width": 720,
                "height": 1280,
                "font_path": "方正黑体简体.ttf",
                "font_size": 30,
                "font_color": [0,0,0],
                "bg_color": [255,255,255,1.0],
                "line_spacing": 20,
                "char_spacing": 10,
                "fps": 30,  # 更高帧率，更平滑的效果
                "scroll_speed": 1,
            }
        },
        {
            "description": "深蓝底白字（GPU加速 + 加减速效果 -> MP4）- overlay_cuda GPU加速方法",
            "method": "original", 
            "params": {
                "text": sample_text,
                "width": 720,
                "height": 1280,
                "font_path": "方正黑体简体.ttf",
                "font_size": 30,
                "font_color": [0,0,0],
                "bg_color": [255,255,255,1.0],
                "line_spacing": 20,
                "char_spacing": 10,
                "fps": 30,  # 更高帧率，更平滑的效果
                "scroll_speed": 1,
            }
        }
    ]

    # 循环生成不同场景的视频
    for i, test_case in enumerate(test_cases):
        logger.info(f"--- 开始生成场景 {i+1}: {test_case['description']} ---")
        
        # 输出文件名
        base_file_name = f"test_case_{i+1}_{time.strftime('%Y%m%d_%H%M%S')}"
        output_path_base = os.path.join(output_dir, base_file_name + ".tmp") 

        # 根据方法选择不同的生成函数
        method = test_case.get("method", "original")
        
        # 记录开始时间
        start_time = time.time()
        
        if method == "ffmpeg":
            # 使用FFmpeg滤镜方法
            result = service.create_roll_video_ffmpeg(
                output_path=output_path_base,
                **test_case['params']
            )
        elif method == "overlay_cuda":
            # 使用overlay_cuda GPU加速方法
            result = service.create_roll_video_overlay_cuda(
                output_path=output_path_base,
                **test_case['params']
            )
        else:
            # 使用原始方法
            result = service.create_roll_video(
                output_path=output_path_base,
                **test_case['params']
            )
        
        # 记录结束时间和总耗时
        end_time = time.time()
        total_time = end_time - start_time
        
        # 输出结果
        if result.get("status") == "success":
            logger.info(f"场景 {i+1} 成功: 视频创建完成")
            logger.info(f"最终输出视频路径: {result.get('output_path')}")
            logger.info(f"总耗时: {total_time:.2f}秒")
        else:
            logger.error(f"场景 {i+1} 失败: {result.get('message')}")

        logger.info(f"--- 场景 {i+1} 生成结束 ---")
        
        # 在不同场景之间添加一个间隔
        if i < len(test_cases) - 1:
            logger.info("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()