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
            "description": "overlay_cuda_with_bg_stretch",
            "method": "overlay_cuda", 
            "params": {
                "text": sample_text,
                "width": 720,
                "height": 1280,
                "font_path": "方正黑体简体.ttf",
                "font_size": 30,
                "font_color": [0,0,0],
                "bg_color": [255,255,255,1.0],  # 不透明白色背景
                "line_spacing": 20,
                "char_spacing": 10,
                "fps": 30,
                "scroll_speed": 1,
                "top_margin": 10, # 上边距
                "bottom_margin": 10, # 下边距
                "left_margin": 10, # 左边距
                "right_margin": 10, # 右边距
                "background_url": "https://aigc-miaobi.tos-cn-guangzhou.volces.com/clone_audio_user_demo/2025-05-08/b8b6dd60-2bf0-11f0-94a9-0daeec838b86.jpg", # 背景图
                "scale_mode": "stretch"  # 拉伸模式
            }
        },
        {
            "description": "overlay_cuda_with_bg_tile",
            "method": "overlay_cuda", 
            "params": {
                "text": sample_text,
                "width": 720,
                "height": 1280,
                "font_path": "方正黑体简体.ttf",
                "font_size": 30,
                "font_color": [0,0,0],
                "bg_color": [255,255,255,1.0],  # 不透明白色背景
                "line_spacing": 20,
                "char_spacing": 10,
                "fps": 30,
                "scroll_speed": 1,
                "top_margin": 10, # 上边距
                "bottom_margin": 10, # 下边距
                "left_margin": 10, # 左边距
                "right_margin": 10, # 右边距
                "background_url": "https://aigc-miaobi.tos-cn-guangzhou.volces.com/clone_audio_user_demo/2025-05-08/b8b6dd60-2bf0-11f0-94a9-0daeec838b86.jpg", # 背景图
                "scale_mode": "tile"  # 平铺模式
            }
        },
        {
            "description": "overlay_cuda_no_bg",
            "method": "overlay_cuda", 
            "params": {
                "text": sample_text,
                "width": 720,
                "height": 1280,
                "font_path": "方正黑体简体.ttf",
                "font_size": 30,
                "font_color": [0,0,0],
                "bg_color": [255,255,255,1.0],  # 不透明白色背景
                "line_spacing": 20,
                "char_spacing": 10,
                "fps": 30,
                "scroll_speed": 1,
                "top_margin": 10, # 上边距
                "bottom_margin": 10, # 下边距
                "left_margin": 10, # 左边距
                "right_margin": 10, # 右边距
                "background_url": None  # 不使用背景图
            }
        },
        {
            "description": "crop_with_bg_stretch",
            "method": "crop",
            "params": {
                "text": sample_text,
                "width": 720,
                "height": 1280,
                "font_path": "方正黑体简体.ttf",
                "font_size": 30,
                "font_color": [0,0,0],
                "bg_color": [255,255,255,1.0],  # 不透明白色背景
                "line_spacing": 20,
                "char_spacing": 10,
                "fps": 30,
                "scroll_speed": 1,
                "top_margin": 30, # 上边距
                "bottom_margin": 30, # 下边距
                "left_margin": 30, # 左边距
                "right_margin": 30, # 右边距
                "background_url": "https://aigc-miaobi.tos-cn-guangzhou.volces.com/clone_audio_user_demo/2025-05-08/b8b6dd60-2bf0-11f0-94a9-0daeec838b86.jpg", # 背景图
                "scale_mode": "stretch"  # 拉伸模式
            }
        },
        {
            "description": "crop_with_bg_tile",
            "method": "crop",
            "params": {
                "text": sample_text,
                "width": 720,
                "height": 1280,
                "font_path": "方正黑体简体.ttf",
                "font_size": 30,
                "font_color": [0,0,0],
                "bg_color": [255,255,255,1.0],  # 不透明白色背景
                "line_spacing": 20,
                "char_spacing": 10,
                "fps": 30,
                "scroll_speed": 1,
                "top_margin": 30, # 上边距
                "bottom_margin": 30, # 下边距
                "left_margin": 30, # 左边距
                "right_margin": 30, # 右边距
                "background_url": "https://aigc-miaobi.tos-cn-guangzhou.volces.com/clone_audio_user_demo/2025-05-08/b8b6dd60-2bf0-11f0-94a9-0daeec838b86.jpg", # 背景图
                "scale_mode": "tile"  # 平铺模式
            }
        },
        {
            "description": "crop_no_bg",
            "method": "crop",
            "params": {
                "text": sample_text,
                "width": 720,
                "height": 1280,
                "font_path": "方正黑体简体.ttf",
                "font_size": 30,
                "font_color": [0,0,0],
                "bg_color": [255,255,255,1.0],  # 不透明白色背景
                "line_spacing": 20,
                "char_spacing": 10,
                "fps": 30,
                "scroll_speed": 1,
                "top_margin": 30, # 上边距
                "bottom_margin": 30, # 下边距
                "left_margin": 30, # 左边距
                "right_margin": 30, # 右边距
                "background_url": None  # 不使用背景图
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
        method = test_case.get("method", "crop")
        
        # 记录开始时间
        start_time = time.time()
        
        if method == "crop":
            # crop滤镜
            result = service.create_roll_video_crop(
                output_path=output_path_base,
                **test_case['params']
            )
        elif method == "overlay_cuda":
            # overlay_cuda滤镜 (只支持基础匀速滚动和从下到上滚动)
            params = test_case['params'].copy()
            result = service.create_roll_video_overlay_cuda(
                output_path=output_path_base,
                **params
            )
        else:
            # overlay_cuda滤镜
            result = service.create_roll_video_overlay_cuda(
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