import time
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor
from DBUtils.PooledDB import PooledDB
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings

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
    
    基于DBUtils.PooledDB实现的MySQL连接池，支持异常重试和监控
    
    Attributes:
        pool: 数据库连接池
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
        if hasattr(self, 'pool'):
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
        
        # 初始化连接池
        self._init_pool()
        
        # 初始化SQLAlchemy
        self._init_sqlalchemy()
        
        logger.info(
            f"MySQL连接池初始化成功 - 主机:{self.host} 端口:{self.port} "
            f"数据库:{self.database} 最大连接数:{self.max_connections}"
        )
    
    def _init_pool(self) -> None:
        """初始化DBUtils连接池"""
        try:
            self.pool = PooledDB(
                creator=pymysql,
                maxconnections=self.max_connections,
                mincached=self.min_connections,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                autocommit=True,
                cursorclass=DictCursor,
                blocking=True,
                ping=1,  # 自动检测连接是否可用
            )
        except Exception as e:
            logger.error(f"MySQL连接池初始化失败: {str(e)}")
            raise ConnectionError(f"MySQL连接池初始化失败: {str(e)}")
    
    def _init_sqlalchemy(self) -> None:
        """初始化SQLAlchemy引擎和会话工厂"""
        connection_str = (
            f"mysql+pymysql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}?charset={self.charset}"
        )
        
        # 创建引擎，配置连接池
        self.engine = create_engine(
            connection_str,
            pool_size=self.min_connections,
            max_overflow=self.max_connections - self.min_connections,
            pool_timeout=self.timeout,
            pool_recycle=3600,  # 一小时后回收连接
            pool_pre_ping=True,  # 自动检测连接是否有效
        )
        
        # 创建会话工厂
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_connection(self) -> pymysql.connections.Connection:
        """获取数据库连接
        
        Returns:
            pymysql.connections.Connection: 数据库连接对象
        
        Raises:
            ConnectionError: 获取连接失败时抛出
        """
        for attempt in range(self.retry_count):
            try:
                connection = self.pool.connection()
                return connection
            except Exception as e:
                logger.warning(f"获取MySQL连接失败 (尝试 {attempt+1}/{self.retry_count}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"获取MySQL连接失败，已达到最大重试次数: {str(e)}")
                    raise ConnectionError(f"获取MySQL连接失败: {str(e)}")
    
    @contextmanager
    def get_cursor(self) -> DictCursor:
        """获取数据库游标上下文管理器
        
        使用方法:
            with mysql_pool.get_cursor() as cursor:
                cursor.execute("SELECT * FROM table")
                results = cursor.fetchall()
        
        Yields:
            DictCursor: 返回字典格式结果的游标对象
        
        Raises:
            ConnectionError: 获取连接或游标失败时抛出
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            yield cursor
        except Exception as e:
            logger.error(f"获取MySQL游标失败: {str(e)}")
            raise ConnectionError(f"获取MySQL游标失败: {str(e)}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    @contextmanager
    def transaction(self) -> DictCursor:
        """事务上下文管理器
        
        使用方法:
            with mysql_pool.transaction() as cursor:
                cursor.execute("INSERT INTO table VALUES (%s, %s)", (1, 'test'))
                cursor.execute("UPDATE table SET value = %s WHERE id = %s", ('new', 1))
        
        Yields:
            DictCursor: 返回字典格式结果的游标对象
        
        Raises:
            TransactionError: 事务执行失败时抛出
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            conn.begin()  # 开始事务
            cursor = conn.cursor()
            yield cursor
            conn.commit()  # 提交事务
        except Exception as e:
            if conn:
                conn.rollback()  # 回滚事务
            logger.error(f"MySQL事务执行失败: {str(e)}")
            raise TransactionError(f"MySQL事务执行失败: {str(e)}")
        finally:
            if cursor:
                cursor.close()
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
            params: SQL参数
            commit: 是否自动提交，默认True
        
        Returns:
            int: 影响的行数
        
        Raises:
            ExecuteError: SQL执行失败时抛出
        """
        with self.get_cursor() as cursor:
            try:
                affected_rows = cursor.execute(sql, params)
                if commit:
                    cursor.connection.commit()
                return affected_rows
            except Exception as e:
                if commit:
                    cursor.connection.rollback()
                logger.error(f"执行SQL失败: {str(e)}, SQL: {sql}, 参数: {params}")
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
            params_list: SQL参数列表
            commit: 是否自动提交，默认True
        
        Returns:
            int: 影响的行数
        
        Raises:
            ExecuteError: SQL批量执行失败时抛出
        """
        with self.get_cursor() as cursor:
            try:
                affected_rows = cursor.executemany(sql, params_list)
                if commit:
                    cursor.connection.commit()
                return affected_rows
            except Exception as e:
                if commit:
                    cursor.connection.rollback()
                logger.error(f"批量执行SQL失败: {str(e)}, SQL: {sql}")
                raise ExecuteError(f"批量执行SQL失败: {str(e)}")
    
    def query_one(self, sql: str, params: Union[tuple, list, dict] = None) -> Optional[Dict[str, Any]]:
        """查询单条记录
        
        Args:
            sql: SQL查询语句
            params: SQL参数
        
        Returns:
            Optional[Dict[str, Any]]: 查询结果字典，未找到时返回None
        
        Raises:
            ExecuteError: SQL执行失败时抛出
        """
        with self.get_cursor() as cursor:
            try:
                cursor.execute(sql, params)
                return cursor.fetchone()
            except Exception as e:
                logger.error(f"查询失败: {str(e)}, SQL: {sql}, 参数: {params}")
                raise ExecuteError(f"查询失败: {str(e)}")
    
    def query_all(self, sql: str, params: Union[tuple, list, dict] = None) -> List[Dict[str, Any]]:
        """查询多条记录
        
        Args:
            sql: SQL查询语句
            params: SQL参数
        
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        
        Raises:
            ExecuteError: SQL执行失败时抛出
        """
        with self.get_cursor() as cursor:
            try:
                cursor.execute(sql, params)
                return cursor.fetchall()
            except Exception as e:
                logger.error(f"查询失败: {str(e)}, SQL: {sql}, 参数: {params}")
                raise ExecuteError(f"查询失败: {str(e)}")
    
    def get_session(self) -> Session:
        """获取SQLAlchemy会话
        
        Returns:
            Session: SQLAlchemy会话对象
        """
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Session:
        """SQLAlchemy会话上下文管理器
        
        使用方法:
            with mysql_pool.session_scope() as session:
                user = User(name="test", email="test@example.com")
                session.add(user)
        
        Yields:
            Session: SQLAlchemy会话对象
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"SQLAlchemy会话操作失败: {str(e)}")
            raise
        finally:
            session.close()
    
    def check_connection(self) -> bool:
        """检查数据库连接是否正常
        
        Returns:
            bool: 连接正常返回True，否则返回False
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"数据库连接检查失败: {str(e)}")
            return False
    
    def close(self) -> None:
        """关闭连接池
        
        注意: 由于DBUtils.PooledDB没有提供直接关闭连接池的方法，
        该方法仅用于记录关闭操作，实际连接会在应用退出时自动关闭
        """
        logger.info("MySQL连接池准备关闭")


# 创建连接池单例
mysql_pool = MySQLPool()

# 提供SQLAlchemy ORM相关组件的快捷访问
def get_db():
    """获取数据库会话
    
    用于FastAPI的依赖注入
    
    Yields:
        Session: SQLAlchemy会话对象
    """
    db = mysql_pool.get_session()
    try:
        yield db
    finally:
        db.close() 