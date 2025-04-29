from celery import Celery



REDIS_HOST="127.0.0.1"
REDIS_PORT=6379
REDIS_PASSWORD="GTO4mjZQXZkWYgspMWHHgla0Lf5yNew8zlgRyq"

# 创建Celery实例
celery_app = Celery(
    "roll-video",
    broker=f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/1",
    broker_transport_options={
        'visibility_timeout': 3600,
        'socket_timeout': 30,
        'socket_connect_timeout': 30,
    },
    include=['app.tasks.roll_video_tasks']  # 显式包含任务模块
)

# 配置Celery
celery_app.conf.update(
    # 基本配置
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    result_backend_transport_options={
        'retry_policy': {
            'timeout': 5.0
        }
    },
    redis_backend_health_check_interval = 5,
    
    # 任务执行设置
    result_expires=18000,
    worker_prefetch_multiplier=1,
    task_time_limit=18000,
    task_soft_time_limit=15000,
    
    # Redis 配置
    broker_transport_options={
        'visibility_timeout': 3600,
        'socket_timeout': 30,
        'socket_connect_timeout': 30,
        'socket_keepalive': True,
        'retry_on_timeout': True
    },
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    broker_connection_timeout=30,
    
    # 明确指定 broker 类型
    broker_transport='redis',
    result_backend_transport='redis',
    
    # 任务名称设置 - 确保与使用的路径一致
    task_create_missing_queues=True,
    task_default_queue="celery",
    task_default_exchange="celery",
    task_default_routing_key="celery",
)

# 自动发现任务 - 这里不再使用
# celery_app.autodiscover_tasks(['app'])

__all__ = ['celery_app']
