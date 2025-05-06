#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试滚动视频服务功能
"""

import os
import logging
import time
import sys
from app.services.roll_video.roll_video_service import RollVideoService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("开始测试滚动视频服务...")
    
    # 简单的测试文本
    sample_text = """这是一个测试文本。
    我们正在测试滚动视频功能。
    这里包含多行文本。
    
    测试滚动视频的功能是否正常工作。
    如果一切顺利，应该能看到平滑滚动的文字。
    
    测试优化后的渲染性能，应该有显著提升。
    """ * 10  # 复制多次以获得更长的文本
    
    logger.info(f"测试文本长度: {len(sample_text)} 字符, {len(sample_text.split('\\n'))} 行")
    
    # 输出目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 创建服务实例
    logger.info("创建RollVideoService实例...")
    service = RollVideoService()
    
    # 输出文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"test_roll_{timestamp}.mp4")
    logger.info(f"输出文件: {output_path}")
    
    # 检查字体目录
    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "services", "roll_video", "fonts")
    if not os.path.exists(font_dir):
        logger.warning(f"字体目录不存在: {font_dir}, 将使用系统字体")
    else:
        logger.info(f"检测到字体目录: {font_dir}")
        fonts = [f for f in os.listdir(font_dir) if f.endswith(('.ttf', '.otf'))]
        logger.info(f"可用字体: {fonts}")
    
    # 渲染参数
    bg_color = (255, 255, 255, 255) # 不透明白色背景
    font_color = (0, 0, 0) # 黑色文字
    width = 720
    height = 1280
    
    logger.info(f"开始渲染视频: {width}x{height}, 背景色={bg_color}, 字体颜色={font_color}")
    
    # 调用服务创建滚动视频
    try:
        result = service.create_roll_video(
            text=sample_text,
            output_path=output_path,
            width=width,
            height=height,
            font_size=24,
            font_color=font_color,
            bg_color=bg_color,
            line_spacing=10,
            char_spacing=2,
            fps=30,
            scroll_speed=1.0  # 每秒滚动一行
        )
        
        # 输出结果
        logger.info(f"渲染结果: {result}")
        if result["status"] == "success":
            logger.info(f"测试成功: 视频已创建")
            logger.info(f"最终输出视频路径: {result['output_path']}")
        else:
            logger.error(f"测试失败: {result['message']}")
    except Exception as e:
        logger.error(f"渲染过程出现异常: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 