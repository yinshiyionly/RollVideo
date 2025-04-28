from enum import Enum
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


class TaskState(str, Enum):
    """任务状态枚举类
    
    用于表示滚动视频任务的不同状态
    """
    PENDING = "pending"      # 待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"         # 失败


class TaskStatus(int, Enum):
    """记录状态枚举类
    
    用于表示记录是否有效
    """
    NORMAL = 1  # 正常
    DELETED = 2  # 删除


class RollVideoTaskBase(BaseModel):
    """滚动视频任务基础模型
    
    包含创建和查询任务时共有的基础字段
    """
    task_id: str = Field(..., description="任务ID")
    uid: int = Field(..., description="用户ID")
    source: str = Field(..., description="来源")


class RollVideoTaskCreate(RollVideoTaskBase):
    """创建滚动视频任务请求模型
    
    用于接收创建滚动视频处理任务的请求参数
    """
    payload: Optional[Dict[str, Any]] = Field(None, description="任务请求参数")
    task_state: TaskState = Field(default=TaskState.PENDING, description="任务状态")


class RollVideoTaskResponse(RollVideoTaskBase):
    """滚动视频任务响应模型
    
    用于返回任务处理的状态和结果
    """
    id: int = Field(..., description="主键ID")
    task_state: TaskState = Field(..., description="任务状态")
    payload: Optional[Dict[str, Any]] = Field(None, description="任务请求参数")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")
    status: TaskStatus = Field(..., description="状态 1-正常 2-删除")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class RollVideoTaskUpdate(BaseModel):
    """更新滚动视频任务模型
    
    用于更新滚动视频任务的状态和结果
    """
    task_state: Optional[TaskState] = Field(None, description="任务状态")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")
    payload: Optional[Dict[str, Any]] = Field(None, description="任务请求参数")
    status: Optional[TaskStatus] = Field(None, description="状态 1-正常 2-删除") 