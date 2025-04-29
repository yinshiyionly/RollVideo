"""
统一导入的 Celery 应用实例，解决导入路径冲突问题
"""

# 重新导出 app.celery_app 中的实例
from app.celery_app import celery_app

__all__ = ['celery_app'] 