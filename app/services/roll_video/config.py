# -*- coding: utf-8 -*-

"""
滚动视频服务的默认配置参数
"""

# 滚动视频渲染和编码的默认调优参数
DEFAULT_OPTIMIZATION_CONFIG = {
    "worker_threads": 8,  # 默认工作线程数 (基于16核CPU调整)
    "transparent_codec": "prores_4444", # 默认透明视频编码器
}
