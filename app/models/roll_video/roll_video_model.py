from enum import Enum
from pydantic import BaseModel, StringConstraints
from typing import Optional, Dict, Any, Annotated

class TaskStatus(str, Enum):
    """任务状态枚举类"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class CreateTaskRequest(BaseModel):
    """创建任务请求模型"""
    video_url: str
    uid: Annotated[str, StringConstraints(min_length=1, max_length=64)]

class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    status: TaskStatus
    video_url: str
    uid: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None 