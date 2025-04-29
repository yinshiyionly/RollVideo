#!/usr/bin/env python
"""
Celery 工作进程启动文件

使用方法：
celery -A celery_worker worker --loglevel=info --concurrency=1 
"""

import os
import sys

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入 Celery 应用实例
from app.celery_app import celery_app

# 获取任务列表
tasks = celery_app.tasks
print(f"已注册任务: {list(tasks.keys())}")

if __name__ == '__main__':
    celery_app.start() 