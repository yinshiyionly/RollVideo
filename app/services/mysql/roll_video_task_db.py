from typing import Optional, Dict, Any, List
from fastapi import HTTPException
from sqlalchemy import create_engine, text, exc
from sqlalchemy.exc import SQLAlchemyError
from app.config import settings
from app.utils.logger import Logger
from app.models.roll_video_task import TaskState, TaskStatus, RollVideoTaskCreate, RollVideoTaskResponse, RollVideoTaskUpdate
import json
from functools import wraps
import time
from datetime import datetime


logger = Logger("roll_video_task_db")

def retry_on_connection_error(max_retries=3, initial_delay=1):
    """数据库连接错误重试装饰器
    
    在发生数据库连接丢失时，自动重试操作
    
    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟时间(秒)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exc.OperationalError as e:
                    if "Lost connection" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"数据库连接丢失，正在重试 ({attempt + 1}/{max_retries})", {
                            "error": str(e),
                            "function": func.__name__
                        })
                        time.sleep(delay)
                        delay *= 2  # 指数退避
                        continue
                    raise
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RollVideoTaskDB:
    """滚动视频任务数据库操作类"""
    
    def __init__(self):
        """初始化数据库连接"""
        db_url = f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
        self.engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=1800,  # 30分钟回收连接
            pool_size=10,
            max_overflow=20
        )
    
    @retry_on_connection_error()
    def create_task(self, task: RollVideoTaskCreate) -> str:
        """创建新的滚动视频任务
        
        Args:
            task: 任务创建模型
            
        Returns:
            str: 任务ID
            
        Raises:
            HTTPException: 数据库操作失败时抛出异常
        """
        try:
            with self.engine.connect() as conn:
                query = text(
                    """
                    INSERT INTO roll_video_task 
                    (task_id, uid, source, task_state, payload)
                    VALUES (:task_id, :uid, :source, :task_state, :payload)
                    """
                )
                
                conn.execute(
                    query, {
                        "task_id": task.task_id,
                        "uid": task.uid,
                        "source": task.source,
                        "task_state": task.task_state.value,
                        "payload": json.dumps(task.payload) if task.payload else None
                    }
                )
                conn.commit()
                return task.task_id
        except SQLAlchemyError as e:
            logger.error(f"创建任务失败: {str(e)}", {"task_id": task.task_id})
            raise HTTPException(status_code=500, detail="创建任务失败")
    
    @retry_on_connection_error()
    def get_task(self, task_id: str) -> Optional[RollVideoTaskResponse]:
        """获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[RollVideoTaskResponse]: 任务响应模型，不存在则返回None
            
        Raises:
            HTTPException: 数据解析或数据库操作失败时抛出异常
        """
        try:
            with self.engine.connect() as conn:
                query = text(
                    """
                    SELECT id, task_id, uid, source, task_state, payload, result, status, created_at, updated_at
                    FROM roll_video_task
                    WHERE task_id = :task_id AND status = 1
                    """
                )
                result = conn.execute(query, {"task_id": task_id}).fetchone()
                
                if not result:
                    return None
                
                try:
                    payload = json.loads(result.payload) if result.payload else None
                except json.JSONDecodeError as e:
                    logger.error(f"payload JSON解析失败: {str(e)}", {
                        "task_id": task_id,
                        "payload": result.payload
                    })
                    payload = None
                
                try:
                    result_json = json.loads(result.result) if result.result else None
                except json.JSONDecodeError as e:
                    logger.error(f"result JSON解析失败: {str(e)}", {
                        "task_id": task_id,
                        "result": result.result
                    })
                    result_json = None
                
                return RollVideoTaskResponse(
                    id=result.id,
                    task_id=result.task_id,
                    uid=result.uid,
                    source=result.source,
                    task_state=result.task_state,
                    payload=payload,
                    result=result_json,
                    status=result.status,
                    created_at=result.created_at,
                    updated_at=result.updated_at
                )
        except HTTPException as http_ex:
            raise http_ex
        except SQLAlchemyError as e:
            logger.error(f"获取任务失败: {str(e)}", {"task_id": task_id})
            raise HTTPException(status_code=500, detail="获取任务失败")
        except Exception as e:
            logger.error(f"获取任务时发生未预期的错误: {str(e)}", {
                "task_id": task_id,
                "error_type": type(e).__name__
            })
            raise HTTPException(status_code=500, detail="获取任务时发生未预期的错误")
    
    @retry_on_connection_error()
    def update_task(self, task_id: str, task_update: RollVideoTaskUpdate) -> bool:
        """更新任务信息
        
        Args:
            task_id: 任务ID
            task_update: 任务更新模型
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self.engine.connect() as conn:
                # 构建动态更新条件
                update_fields = []
                params = {"task_id": task_id}
                
                if task_update.task_state is not None:
                    update_fields.append("task_state = :task_state")
                    params["task_state"] = task_update.task_state.value
                
                if task_update.result is not None:
                    update_fields.append("result = :result")
                    params["result"] = json.dumps(task_update.result)
                
                if task_update.payload is not None:
                    update_fields.append("payload = :payload")
                    params["payload"] = json.dumps(task_update.payload)
                
                if task_update.status is not None:
                    update_fields.append("status = :status")
                    params["status"] = task_update.status.value
                
                if not update_fields:
                    logger.warning("没有指定要更新的字段", {"task_id": task_id})
                    return False
                
                query = text(
                    f"""
                    UPDATE roll_video_task
                    SET {', '.join(update_fields)}
                    WHERE task_id = :task_id
                    """
                )
                
                conn.execute(query, params)
                conn.commit()
                return True
        except SQLAlchemyError as e:
            logger.error(f"更新任务失败: {str(e)}", {"task_id": task_id})
            return False
    
    @retry_on_connection_error()
    def update_task_state(self, task_id: str, task_state: TaskState, result: Optional[Dict[str, Any]] = None) -> bool:
        """更新任务状态
        
        Args:
            task_id: 任务ID
            task_state: 任务状态
            result: 任务结果（可选）
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self.engine.connect() as conn:
                params = {
                    "task_id": task_id,
                    "task_state": task_state.value,
                }
                
                if result is not None:
                    query = text(
                        """
                        UPDATE roll_video_task
                        SET task_state = :task_state, result = :result
                        WHERE task_id = :task_id
                        """
                    )
                    params["result"] = json.dumps(result)
                else:
                    query = text(
                        """
                        UPDATE roll_video_task
                        SET task_state = :task_state
                        WHERE task_id = :task_id
                        """
                    )
                
                conn.execute(query, params)
                conn.commit()
                return True
        except SQLAlchemyError as e:
            logger.error(f"更新任务状态失败: {str(e)}", {"task_id": task_id})
            return False
    
    @retry_on_connection_error()
    def delete_task(self, task_id: str) -> bool:
        """逻辑删除任务
        
        将任务状态修改为已删除
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            with self.engine.connect() as conn:
                query = text(
                    """
                    UPDATE roll_video_task
                    SET status = :status
                    WHERE task_id = :task_id
                    """
                )
                
                conn.execute(query, {
                    "task_id": task_id,
                    "status": TaskStatus.DELETED.value
                })
                conn.commit()
                return True
        except SQLAlchemyError as e:
            logger.error(f"删除任务失败: {str(e)}", {"task_id": task_id})
            return False
    
    @retry_on_connection_error()
    def list_tasks(self, uid: Optional[int] = None, source: Optional[str] = None, 
                   task_state: Optional[TaskState] = None, limit: int = 20, offset: int = 0) -> List[RollVideoTaskResponse]:
        """列出任务
        
        根据条件查询任务列表
        
        Args:
            uid: 用户ID（可选）
            source: 来源（可选）
            task_state: 任务状态（可选）
            limit: 返回记录数量限制
            offset: 分页偏移量
            
        Returns:
            List[RollVideoTaskResponse]: 任务响应模型列表
        """
        try:
            with self.engine.connect() as conn:
                conditions = ["status = 1"]  # 默认只查询正常状态的记录
                params = {}
                
                if uid is not None:
                    conditions.append("uid = :uid")
                    params["uid"] = uid
                
                if source is not None:
                    conditions.append("source = :source")
                    params["source"] = source
                
                if task_state is not None:
                    conditions.append("task_state = :task_state")
                    params["task_state"] = task_state.value
                
                where_clause = " AND ".join(conditions)
                
                query = text(
                    f"""
                    SELECT id, task_id, uid, source, task_state, payload, result, status, created_at, updated_at
                    FROM roll_video_task
                    WHERE {where_clause}
                    ORDER BY id DESC
                    LIMIT :limit OFFSET :offset
                    """
                )
                
                params["limit"] = limit
                params["offset"] = offset
                
                results = conn.execute(query, params).fetchall()
                
                tasks = []
                for row in results:
                    try:
                        payload = json.loads(row.payload) if row.payload else None
                    except json.JSONDecodeError:
                        payload = None
                        
                    try:
                        result_json = json.loads(row.result) if row.result else None
                    except json.JSONDecodeError:
                        result_json = None
                    
                    tasks.append(RollVideoTaskResponse(
                        id=row.id,
                        task_id=row.task_id,
                        uid=row.uid,
                        source=row.source,
                        task_state=row.task_state,
                        payload=payload,
                        result=result_json,
                        status=row.status,
                        created_at=row.created_at,
                        updated_at=row.updated_at
                    ))
                
                return tasks
        except SQLAlchemyError as e:
            logger.error(f"查询任务列表失败: {str(e)}")
            return [] 