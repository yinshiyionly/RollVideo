#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
滚动视频服务使用示例
"""

import os
import logging
import time
import sys
from roll_video_service import RollVideoService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
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
    # 2025-05-02 22:12:00 旧版方案升级版 2025-05-02 14:10:52,850 - renderer - INFO - 总渲染性能: 渲染了5628帧，耗时28.91秒，平均194.70帧/秒
    test_cases = [
        {
            # 不透明 -> 自动尝试 GPU (h264_nvenc) -> 输出 .mp4
            "description": "白底黑字（自动GPU -> MP4）",
            "params": {
                "text": sample_text,
                "width": 720,
                "height": 1280,
                "font_path": "方正黑体简体.ttf",
                "font_size": 24,
                "font_color": [0,0,0],
                "bg_color":[255,255,255,1.0], # 不透明
                "line_spacing": 3,
                "char_spacing": 5,
                "fps": 30,
                "scroll_speed": 1,
            }
        }
        # ,{
        #     # 透明背景 -> 使用 ProRes 4444 -> 输出 .mov (高质量，大文件)
        #     "description": "透明背景黑字（ProRes 4444 -> MOV）",
        #     "params": {
        #         "text": sample_text[:2000],  # 使用较少文字以加快测试
        #         "width": 720,
        #         "height": 1280,
        #         "font_path": "方正黑体简体.ttf",
        #         "font_size": 24,
        #         "font_color": [0,0,0],
        #         "bg_color": [255,255,255,0.5],  # 半透明背景
        #         "line_spacing": 1.5,
        #         "char_spacing": 5,
        #         "fps": 30,
        #         "scroll_speed": 1,  # 较快的滚动速度
        #     }
        # },
        # {
        #     # 透明背景 -> 使用 ProRes 422HQ -> 输出 .mov (中等质量，中等文件大小)
        #     "description": "透明背景黑字（ProRes 422HQ -> MOV）",
        #     "params": {
        #         "text": sample_text[:2000],  # 使用较少文字以加快测试
        #         "width": 720,
        #         "height": 1280,
        #         "font_path": "方正黑体简体.ttf",
        #         "font_size": 24,
        #         "font_color": [0,0,0],
        #         "bg_color": [255,255,255,0.5],  # 半透明背景
        #         "line_spacing": 1.5,
        #         "char_spacing": 5,
        #         "fps": 30,
        #         "scroll_speed": 1, 
        #     }
        # },
        # {
        #     # 透明背景 -> 使用 VP9 -> 输出 .webm (较低质量，小文件)
        #     "description": "透明背景黑字（VP9 -> WEBM）",
        #     "params": {
        #         "text": sample_text[:2000],  # 使用较少文字以加快测试
        #         "width": 720,
        #         "height": 1280,
        #         "font_path": "方正黑体简体.ttf",
        #         "font_size": 24,
        #         "font_color": [0,0,0],
        #         "bg_color": [255,255,255,0.5],  # 半透明背景
        #         "line_spacing": 1.5,
        #         "char_spacing": 5,
        #         "fps": 30,
        #         "scroll_speed": 1,  # 更快的滚动速度
        #     }
        # }
    ]

    # 循环生成不同场景的视频
    for i, test_case in enumerate(test_cases):
        logger.info(f"--- 开始生成场景 {i+1}: {test_case['description']} ---")
        
        # 输出文件名不再需要指定扩展名，Service 会自动处理
        # 但我们仍然需要一个基础的文件名
        base_file_name = f"test_case_{i+1}_{time.strftime('%Y%m%d_%H%M%S')}"
        # 传递一个带任意（或无）扩展名的路径给 Service
        output_path_base = os.path.join(output_dir, base_file_name + ".tmp") 

        # 调用服务创建滚动视频
        # 使用 **test_case['params'] 来传递所有参数
        result = service.create_roll_video(
            output_path=output_path_base, # Service 会修正扩展名
            **test_case['params'] 
        )

        # 输出结果
        if result:
            logger.info(f"场景 {i+1} 成功: 视频创建完成")
            logger.info(f"最终输出视频路径: {result}") # 注意这里是最终路径
        else:
            logger.error(f"场景 {i+1} 失败: 未能创建视频")

        logger.info(f"--- 场景 {i+1} 生成结束 ---")

if __name__ == "__main__":
    main()