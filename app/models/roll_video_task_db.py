from typing import Optional, Dict, Any, List
from fastapi import HTTPException
from sqlalchemy import exc
from sqlalchemy.exc import SQLAlchemyError
from app.utils.logger import Logger
from app.models.roll_video_task import TaskState, TaskStatus, RollVideoTaskCreate, RollVideoTaskResponse, RollVideoTaskUpdate
import json
from functools import wraps
import time
from datetime import datetime

from sqlalchemy import Column, String, Integer, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from app.utils.mysql_pool import MySQLPool

Base = declarative_base()

class RollVideoTaskModel(Base):
    """滚动视频任务数据库模型"""
    
    __tablename__ = "roll_video_tasks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), index=True, unique=True, nullable=False, comment="任务ID")
    uid = Column(Integer, index=True, nullable=False, comment="用户ID")
    source = Column(String(32), index=True, nullable=False, comment="来源")
    task_state = Column(String(32), index=True, nullable=False, comment="任务状态")
    payload = Column(JSON, nullable=True, comment="任务参数")
    result = Column(JSON, nullable=True, comment="任务结果")
    status = Column(Integer, index=True, nullable=False, default=1, comment="记录状态：1-正常，0-删除")
    created_at = Column(DateTime, nullable=False, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now, comment="更新时间")

logger = Logger("roll-video-task-db")

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
        # 使用 MySQLPool 单例获取连接池
        self.mysql_pool = MySQLPool()
        # 使用池化的 Session
        self.SessionLocal = self.mysql_pool.SessionLocal
    
    def get_session(self) -> Session:
        """获取新的会话"""
        return self.SessionLocal()
        
    def create_task(self, task: RollVideoTaskCreate) -> str:
        """创建新任务
        
        Args:
            task: 任务创建参数
            
        Returns:
            str: 任务ID
        """
        # 使用with语句自动管理session的生命周期
        with self.mysql_pool.session_scope() as session:
            # 构建数据库模型
            db_task = RollVideoTaskModel(
                task_id=task.task_id,
                uid=task.uid,
                source=task.source,
                task_state=TaskState.PENDING,
                payload=task.payload,
                status=TaskStatus.NORMAL.value
            )
            
            # 添加到会话
            session.add(db_task)
            
            # 会话自动提交和关闭
            return task.task_id
    
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
            with self.mysql_pool.session_scope() as session:
                query = session.query(RollVideoTaskModel).filter(RollVideoTaskModel.task_id == task_id).first()
                
                if not query:
                    return None
                
                # 处理payload字段
                payload = None
                if query.payload is not None:
                    if isinstance(query.payload, dict):
                        # 已经是字典，直接使用
                        payload = query.payload
                    elif isinstance(query.payload, str):
                        # 是字符串，需要解析
                        try:
                            payload = json.loads(query.payload)
                        except json.JSONDecodeError as e:
                            logger.error(f"payload JSON解析失败: {str(e)}", {
                                "task_id": task_id,
                                "payload": query.payload
                            })
                
                # 处理result字段
                result_json = None
                if query.result is not None:
                    if isinstance(query.result, dict):
                        # 已经是字典，直接使用
                        result_json = query.result
                    elif isinstance(query.result, str):
                        # 是字符串，需要解析
                        try:
                            result_json = json.loads(query.result)
                        except json.JSONDecodeError as e:
                            logger.error(f"result JSON解析失败: {str(e)}", {
                                "task_id": task_id,
                                "result": query.result
                            })
                
                return RollVideoTaskResponse(
                    id=query.id,
                    task_id=query.task_id,
                    uid=query.uid,
                    source=query.source,
                    task_state=query.task_state,
                    payload=payload,
                    result=result_json,
                    status=query.status,
                    created_at=query.created_at,
                    updated_at=query.updated_at
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
            with self.mysql_pool.session_scope() as session:
                query = session.query(RollVideoTaskModel).filter(RollVideoTaskModel.task_id == task_id).first()
                
                if not query:
                    logger.warning("任务不存在", {"task_id": task_id})
                    return False
                
                if task_update.task_state is not None:
                    query.task_state = task_update.task_state.value
                
                if task_update.result is not None:
                    # 尝试直接赋值，出错则使用JSON序列化
                    try:
                        query.result = task_update.result
                    except Exception as e:
                        logger.warning(f"直接赋值结果失败，尝试JSON序列化: {str(e)}")
                        query.result = json.dumps(task_update.result, ensure_ascii=False)
                
                if task_update.payload is not None:
                    # 尝试直接赋值，出错则使用JSON序列化
                    try:
                        query.payload = task_update.payload
                    except Exception as e:
                        logger.warning(f"直接赋值载荷失败，尝试JSON序列化: {str(e)}")
                        query.payload = json.dumps(task_update.payload, ensure_ascii=False)
                
                if task_update.status is not None:
                    query.status = task_update.status.value
                
                session.commit()
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
            with self.mysql_pool.session_scope() as session:
                query = session.query(RollVideoTaskModel).filter(RollVideoTaskModel.task_id == task_id).first()
                
                if not query:
                    logger.warning("任务不存在", {"task_id": task_id})
                    return False
                
                if task_state is not None:
                    query.task_state = task_state.value
                
                if result is not None:
                    # 对于SQLAlchemy JSON列，根据底层数据库驱动可能需要不同处理：
                    # MySQL (pymysql): 直接赋值字典
                    # 其他数据库: 可能需要使用json.dumps序列化
                    try:
                        # 尝试直接赋值字典
                        query.result = result
                    except Exception as e:
                        logger.warning(f"直接赋值结果失败，尝试JSON序列化: {str(e)}")
                        # 使用ensure_ascii=False确保中文不被转换为Unicode转义序列
                        query.result = json.dumps(result, ensure_ascii=False)
                
                session.commit()
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
            with self.mysql_pool.session_scope() as session:
                query = session.query(RollVideoTaskModel).filter(RollVideoTaskModel.task_id == task_id).first()
                
                if not query:
                    logger.warning("任务不存在", {"task_id": task_id})
                    return False
                
                query.status = TaskStatus.DELETED.value
                session.commit()
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
            with self.mysql_pool.session_scope() as session:
                conditions = ["status = 1"]  # 默认只查询正常状态的记录
                query = session.query(RollVideoTaskModel)
                
                if uid is not None:
                    conditions.append("uid = :uid")
                    query = query.filter(RollVideoTaskModel.uid == uid)
                
                if source is not None:
                    conditions.append("source = :source")
                    query = query.filter(RollVideoTaskModel.source == source)
                
                if task_state is not None:
                    conditions.append("task_state = :task_state")
                    query = query.filter(RollVideoTaskModel.task_state == task_state.value)
                
                query = query.filter(*conditions)
                
                query = query.order_by(RollVideoTaskModel.id.desc()).limit(limit).offset(offset)
                
                results = query.all()
                
                tasks = []
                for row in results:
                    # 处理payload字段
                    payload = None
                    if row.payload is not None:
                        if isinstance(row.payload, dict):
                            # 已经是字典，直接使用
                            payload = row.payload
                        elif isinstance(row.payload, str):
                            # 是字符串，需要解析
                            try:
                                payload = json.loads(row.payload)
                            except json.JSONDecodeError:
                                logger.error(f"list_task - payload JSON解析失败", {
                                    "task_id": row.task_id,
                                    "payload": row.payload
                                })
                    
                    # 处理result字段
                    result_json = None
                    if row.result is not None:
                        if isinstance(row.result, dict):
                            # 已经是字典，直接使用
                            result_json = row.result
                        elif isinstance(row.result, str):
                            # 是字符串，需要解析
                            try:
                                result_json = json.loads(row.result)
                            except json.JSONDecodeError:
                                logger.error(f"list_task - result JSON解析失败", {
                                    "task_id": row.task_id,
                                    "result": row.result
                                })
                    
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