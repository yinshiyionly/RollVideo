from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
from uuid import uuid4
import json
from datetime import datetime

from models.response import success_response, error_response, StatusCode, StatusMessage
from models.request import PaginationParams, CreateVideoTaskRequest
# from services.roll_video.models.roll_video_task import RollVideoTaskCreate
from services.mysql.roll_video_task_db import RollVideoTaskDB
from tasks.roll_video_tasks import process_video_task
from utils.logger import Logger

# 创建路由器
router = APIRouter()

# 初始化日志系统
log = Logger()


# 创建任务
@router.post("/task/create")
async def create_task(
    uid: int,
    source: str,
    video_url: str
):
    """创建视频处理任务
    
    Args:
        uid: 用户ID
        source: 来源
        video_url: 视频URL
        
    Returns:
        任务创建结果
    """
    try:
        # 生成任务ID
        task_id = f"{uuid4()}"
        
        # 构建任务参数
        task_data = {
            "task_id": task_id,
            "uid": uid,
            "source": source,
            "payload": {"video_url": video_url}
        }
        
        # 创建任务对象
        task = RollVideoTaskCreate(**task_data)
        
        # 保存到数据库
        task_db = RollVideoTaskDB()
        saved_task_id = task_db.create_task(task)
        
        if not saved_task_id:
            return error_response(
                code=StatusCode.TASK_CREATION_FAILED,
                message=StatusMessage.TASK_CREATION_FAILED
            )
            
        # 将任务加入处理队列
        # process_video_task.delay(task_id)
        
        # 返回成功响应
        return success_response(
            data={"task_id": task_id},
            message="任务创建成功"
        )
        
    except Exception as e:
        log.error(f"创建任务失败: {str(e)}")
        return error_response(
            code=StatusCode.SERVER_ERROR,
            message=f"创建任务失败: {str(e)}"
        )


# 获取单个任务详情
@router.get("/task/{task_id}")
async def get_task_detail(
    task_id: str = Path(..., description="任务ID")
):
    """获取任务详情
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务详情信息，包括：
        - task_id: 任务ID
        - result: 处理结果
        - task_state: 任务状态
        - created_at: 创建时间
        - updated_at: 更新时间
    """
    try:
        # 获取任务信息
        task_db = RollVideoTaskDB()
        task = task_db.get_task(task_id)
        
        if not task:
            return error_response(
                code=StatusCode.TASK_NOT_FOUND,
                message=StatusMessage.TASK_NOT_FOUND
            )
        
        # 格式化任务数据
        task_detail = {
            "task_id": task.task_id,
            "task_state": task.task_state,
            "result": task.result if task.result else {},  # 确保result是字典类型
            "created_at": task.created_at.strftime("%Y-%m-%d %H:%M:%S") if task.created_at else None,
            "updated_at": task.updated_at.strftime("%Y-%m-%d %H:%M:%S") if task.updated_at else None
        }
        
        # 返回成功响应
        return success_response(
            data=task_detail,
            message="获取任务详情成功"
        )
        
    except Exception as e:
        log.error(f"获取任务详情失败: {str(e)}")
        return error_response(
            code=StatusCode.SERVER_ERROR,
            message=f"获取任务详情失败: {str(e)}"
        )
        