#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
滚动视频服务使用示例
"""

import os
import logging
import time
from app.services.roll_video.roll_video_service import RollVideoService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    # 示例文本
    sample_text = """滚动视频示例文本
感谢使用滚动视频服务!"""  # 更短的文本用于测试

    # 路径
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 创建服务实例
    service = RollVideoService()

    # 定义测试场景参数列表
    test_cases = [
        {
            # 不透明 -> 自动尝试 GPU (h264_nvenc) -> 输出 .mp4
            "description": "白底黑字（自动GPU -> MP4）",
            "params": {
                "text": sample_text,
                "width": 450,
                "height": 700,
                "font_path": "方正黑体简体.ttf",
                "font_size": 24,
                "font_color": (0, 0, 0),
                "bg_color": (255, 255, 255, 1.0), # 不透明
                "line_spacing": int(24 * 3) - 24,
                "char_spacing": 5,
                "fps": 30,
                "scroll_speed": 1,
            }
        },
        {
            # 不透明 -> 自动尝试 GPU (h264_nvenc) -> 输出 .mp4
            "description": "黑底白字（自动GPU -> MP4）",
            "params": {
                "text": sample_text,
                "width": 450,
                "height": 700,
                "font_path": "方正黑体简体.ttf",
                "font_size": 24,
                "font_color": (255, 255, 255), # 白色字体
                "bg_color": (0, 0, 0), # 不透明 (RGB)
                "line_spacing": int(24 * 3) - 24,
                "char_spacing": 5,
                "fps": 30,
                "scroll_speed": 1,
            }
        },
        {
            # 不透明 -> 自动尝试 GPU (h264_nvenc) -> 输出 .mp4
            "description": "短文本（自动GPU -> MP4）",
            "params": {
                "text": "这是一个非常短的文本，测试前后空白区域效果。",
                "width": 450,
                "height": 700,
                "font_path": "方正黑体简体.ttf",
                "font_size": 40,
                "font_color": (255, 255, 255),
                "bg_color": (50, 50, 50, 255), # 不透明
                "line_spacing": int(40 * 3) - 40,
                "char_spacing": 5,
                "fps": 30,
                "scroll_speed": 1,
            }
        },
        {
            # 透明 -> 自动 CPU (prores_ks) -> 输出 .mov
            "description": "透明背景红字（自动CPU -> MOV）",
            "params": {
                "text": sample_text,
                "width": 450,
                "height": 700,
                "font_path": "方正黑体简体.ttf",
                "font_size": 36,
                "font_color": (255, 0, 0),
                "bg_color": (0, 0, 0, 0), # 完全透明
                "line_spacing": int(36 * 3) - 36,
                "char_spacing": 5,
                "fps": 30,
                "scroll_speed": 1,
            }
        },
         {
            # 半透明 -> 自动 CPU (prores_ks) -> 输出 .mov
            "description": "半透明背景绿字（自动CPU -> MOV）",
            "params": {
                "text": sample_text,
                "width": 450,
                "height": 700,
                "font_path": "方正黑体简体.ttf",
                "font_size": 36,
                "font_color": (0, 255, 0),
                "bg_color": (0, 0, 0, 0.5), # 半透明 (float alpha)
                "line_spacing": int(36 * 3) - 36,
                "char_spacing": 5,
                "fps": 30,
                "scroll_speed": 1,
            }
        },
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
        if result["status"] == "success":
            logger.info(f"场景 {i+1} 成功: {result['message']}")
            logger.info(f"最终输出视频路径: {result['output_path']}") # 注意这里是最终路径
        else:
            logger.error(f"场景 {i+1} 失败: {result['message']}")

        logger.info(f"--- 场景 {i+1} 生成结束 ---")

if __name__ == "__main__":
    main()