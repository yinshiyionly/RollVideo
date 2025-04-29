import time
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from config import settings

# 设置日志
logger = logging.getLogger(__name__)

# SQLAlchemy基类
Base = declarative_base()

# 连接池异常类
class MySQLPoolError(Exception):
    """MySQL连接池异常基类"""
    pass

class ConnectionError(MySQLPoolError):
    """连接异常"""
    pass

class ExecuteError(MySQLPoolError):
    """执行SQL异常"""
    pass

class TransactionError(MySQLPoolError):
    """事务异常"""
    pass

class MySQLPool:
    """MySQL连接池管理器
    
    基于SQLAlchemy实现的MySQL连接池，支持异常重试和监控
    
    Attributes:
        engine: SQLAlchemy引擎
        max_connections: 最大连接数
        host: 数据库主机
        port: 数据库端口
        database: 数据库名
        user: 数据库用户名
        password: 数据库密码
        timeout: 连接超时时间(秒)
        charset: 字符集
        retry_count: 异常重试次数
        retry_delay: 重试延迟时间(秒)
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(MySQLPool, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
        max_connections: int = 10,
        min_connections: int = 5,
        timeout: int = 30,
        charset: str = 'utf8mb4',
        retry_count: int = 3,
        retry_delay: float = 0.5,
    ):
        """初始化连接池
        
        Args:
            host: 数据库主机，默认从配置文件获取
            port: 数据库端口，默认从配置文件获取
            database: 数据库名，默认从配置文件获取
            user: 用户名，默认从配置文件获取
            password: 密码，默认从配置文件获取
            max_connections: 最大连接数，默认10
            min_connections: 最小连接数，默认5
            timeout: 连接超时时间(秒)，默认30秒
            charset: 字符集，默认utf8mb4
            retry_count: 异常重试次数，默认3次
            retry_delay: 重试延迟时间(秒)，默认0.5秒
        """
        # 如果已经初始化过，则直接返回
        if hasattr(self, 'engine'):
            return
        
        # 默认使用配置文件中的数据库配置
        self.host = host or settings.MYSQL_HOST
        self.port = port or settings.MYSQL_PORT
        self.database = database or settings.MYSQL_DATABASE
        self.user = user or settings.MYSQL_USER
        self.password = password or settings.MYSQL_PASSWORD
        
        # 连接池配置
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.timeout = timeout
        self.charset = charset
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        # 初始化SQLAlchemy
        self._init_sqlalchemy()
        
        logger.info(
            f"MySQL连接池初始化成功 - 主机:{self.host} 端口:{self.port} "
            f"数据库:{self.database} 最大连接数:{self.max_connections}"
        )
    
    def _init_sqlalchemy(self) -> None:
        """初始化SQLAlchemy引擎和会话工厂"""
        connection_str = (
            f"mysql+pymysql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}?charset={self.charset}"
        )
        
        # 创建引擎，配置连接池
        self.engine = create_engine(
            connection_str,
            poolclass=QueuePool,  # 使用队列池
            pool_size=self.min_connections,
            max_overflow=self.max_connections - self.min_connections,
            pool_timeout=self.timeout,
            pool_recycle=3600,  # 一小时后回收连接
            pool_pre_ping=True,  # 自动检测连接是否有效
        )
        
        # 创建会话工厂
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_connection(self):
        """获取数据库连接
        
        Returns:
            sqlalchemy.engine.Connection: 数据库连接对象
        
        Raises:
            ConnectionError: 获取连接失败时抛出
        """
        for attempt in range(self.retry_count):
            try:
                connection = self.engine.connect()
                return connection
            except Exception as e:
                logger.warning(f"获取MySQL连接失败 (尝试 {attempt+1}/{self.retry_count}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"获取MySQL连接失败，已达到最大重试次数: {str(e)}")
                    raise ConnectionError(f"获取MySQL连接失败: {str(e)}")
    
    @contextmanager
    def get_cursor(self):
        """获取原生SQL执行上下文
        
        使用方法:
            with mysql_pool.get_cursor() as conn:
                result = conn.execute(text("SELECT * FROM table"))
                for row in result:
                    print(row)
        
        Yields:
            sqlalchemy.engine.Connection: 返回 SQLAlchemy 连接对象
        
        Raises:
            ConnectionError: 获取连接失败时抛出
        """
        conn = None
        try:
            conn = self.get_connection()
            yield conn
        except Exception as e:
            logger.error(f"获取SQL执行上下文失败: {str(e)}")
            raise ConnectionError(f"获取SQL执行上下文失败: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def transaction(self):
        """事务上下文管理器
        
        使用方法:
            with mysql_pool.transaction() as conn:
                conn.execute(text("INSERT INTO table VALUES (:id, :value)"), {"id": 1, "value": "test"})
                conn.execute(text("UPDATE table SET value = :value WHERE id = :id"), {"id": 1, "value": "new"})
        
        Yields:
            sqlalchemy.engine.Connection: 返回 SQLAlchemy 连接对象
        
        Raises:
            TransactionError: 事务执行失败时抛出
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.begin():  # 开始事务
                yield conn
                # 事务会自动提交或回滚
        except Exception as e:
            logger.error(f"MySQL事务执行失败: {str(e)}")
            raise TransactionError(f"MySQL事务执行失败: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    def execute(
        self, 
        sql: str, 
        params: Union[tuple, list, dict] = None, 
        commit: bool = True
    ) -> int:
        """执行SQL语句
        
        Args:
            sql: SQL语句
            params: 参数，可以是元组、列表或字典
            commit: 是否自动提交
            
        Returns:
            int: 影响的行数
            
        Raises:
            ExecuteError: SQL执行失败时抛出
        """
        for attempt in range(self.retry_count):
            try:
                with self.get_cursor() as conn:
                    result = conn.execute(text(sql), params)
                    if commit:
                        conn.commit()
                    return result.rowcount
            except Exception as e:
                logger.warning(f"执行SQL失败 (尝试 {attempt+1}/{self.retry_count}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"执行SQL失败，已达到最大重试次数: {str(e)}")
                    raise ExecuteError(f"执行SQL失败: {str(e)}")
    
    def executemany(
        self, 
        sql: str, 
        params_list: List[Union[tuple, dict]], 
        commit: bool = True
    ) -> int:
        """批量执行SQL语句
        
        Args:
            sql: SQL语句
            params_list: 参数列表，每个元素可以是元组或字典
            commit: 是否自动提交
            
        Returns:
            int: 影响的行数
            
        Raises:
            ExecuteError: SQL执行失败时抛出
        """
        for attempt in range(self.retry_count):
            try:
                with self.get_cursor() as conn:
                    result = conn.execute(text(sql), params_list)
                    if commit:
                        conn.commit()
                    return result.rowcount
            except Exception as e:
                logger.warning(f"批量执行SQL失败 (尝试 {attempt+1}/{self.retry_count}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"批量执行SQL失败，已达到最大重试次数: {str(e)}")
                    raise ExecuteError(f"批量执行SQL失败: {str(e)}")
    
    def query_one(self, sql: str, params: Union[tuple, list, dict] = None) -> Optional[Dict[str, Any]]:
        """查询单条记录
        
        Args:
            sql: SQL语句
            params: 参数，可以是元组、列表或字典
            
        Returns:
            Optional[Dict[str, Any]]: 查询结果，没有结果时返回None
            
        Raises:
            ExecuteError: SQL执行失败时抛出
        """
        for attempt in range(self.retry_count):
            try:
                with self.get_cursor() as conn:
                    result = conn.execute(text(sql), params)
                    row = result.fetchone()
                    return dict(row) if row else None
            except Exception as e:
                logger.warning(f"查询单条记录失败 (尝试 {attempt+1}/{self.retry_count}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"查询单条记录失败，已达到最大重试次数: {str(e)}")
                    raise ExecuteError(f"查询单条记录失败: {str(e)}")
    
    def query_all(self, sql: str, params: Union[tuple, list, dict] = None) -> List[Dict[str, Any]]:
        """查询多条记录
        
        Args:
            sql: SQL语句
            params: 参数，可以是元组、列表或字典
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表
            
        Raises:
            ExecuteError: SQL执行失败时抛出
        """
        for attempt in range(self.retry_count):
            try:
                with self.get_cursor() as conn:
                    result = conn.execute(text(sql), params)
                    rows = result.fetchall()
                    return [dict(row) for row in rows]
            except Exception as e:
                logger.warning(f"查询多条记录失败 (尝试 {attempt+1}/{self.retry_count}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"查询多条记录失败，已达到最大重试次数: {str(e)}")
                    raise ExecuteError(f"查询多条记录失败: {str(e)}")
    
    def get_session(self) -> Session:
        """获取新的会话
        
        Returns:
            Session: SQLAlchemy会话对象
        """
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Session:
        """会话上下文管理器
        
        使用方法:
            with mysql_pool.session_scope() as session:
                user = User(name="test")
                session.add(user)
                # 会话自动提交或回滚
        
        Yields:
            Session: SQLAlchemy会话对象
            
        Raises:
            Exception: 会话执行失败时抛出
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"会话执行失败: {str(e)}")
            raise
        finally:
            session.close()
    
    def check_connection(self) -> bool:
        """检查数据库连接是否可用
        
        Returns:
            bool: 连接是否可用
        """
        try:
            with self.get_cursor() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"数据库连接检查失败: {str(e)}")
            return False
    
    def close(self) -> None:
        """关闭连接池
        
        在应用程序结束时调用，释放所有连接
        """
        try:
            self.engine.dispose()
            logger.info("MySQL连接池已关闭")
        except Exception as e:
            logger.error(f"关闭MySQL连接池失败: {str(e)}")


def get_db():
    """依赖注入获取数据库会话
    
    用于FastAPI依赖注入，确保会话在请求结束后关闭
    
    Yields:
        Session: SQLAlchemy会话对象
    """
    db = MySQLPool().get_session()
    try:
        yield db
    finally:
        db.close() 