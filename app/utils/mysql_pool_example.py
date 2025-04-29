"""
MySQL连接池使用示例

本文件提供了MySQL连接池各种使用方法的示例代码，
包括直接使用连接池、使用SQLAlchemy ORM以及事务处理等。
"""

import logging
from sqlalchemy import Column, Integer, String, Boolean, DateTime, func
from datetime import datetime

from app.utils.mysql_pool import (
    mysql_pool, 
    get_db, 
    Base, 
    MySQLPoolError, 
    ConnectionError, 
    ExecuteError
)

# 设置日志
logger = logging.getLogger(__name__)

# 定义示例ORM模型
class User(Base):
    """用户模型示例"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


def example_raw_connection():
    """直接使用连接池示例"""
    try:
        # 查询单条记录
        user = mysql_pool.query_one("SELECT * FROM users WHERE id = %s", (1,))
        if user:
            logger.info(f"找到用户: {user['username']}")
        else:
            logger.info("未找到用户")
        
        # 查询多条记录
        active_users = mysql_pool.query_all(
            "SELECT * FROM users WHERE is_active = %s LIMIT %s", 
            (True, 10)
        )
        logger.info(f"找到 {len(active_users)} 个活跃用户")
        
        # 执行插入操作
        affected_rows = mysql_pool.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (%s, %s, %s)",
            ("newuser", "newuser@example.com", "hashed_password_here")
        )
        logger.info(f"插入了 {affected_rows} 条记录")
        
        # 执行更新操作
        affected_rows = mysql_pool.execute(
            "UPDATE users SET is_active = %s WHERE username = %s",
            (False, "newuser")
        )
        logger.info(f"更新了 {affected_rows} 条记录")
        
        # 批量插入
        users_data = [
            ("user1", "user1@example.com", "password1"),
            ("user2", "user2@example.com", "password2"),
            ("user3", "user3@example.com", "password3"),
        ]
        affected_rows = mysql_pool.executemany(
            "INSERT INTO users (username, email, hashed_password) VALUES (%s, %s, %s)",
            users_data
        )
        logger.info(f"批量插入了 {affected_rows} 条记录")
        
    except MySQLPoolError as e:
        logger.error(f"MySQL操作失败: {str(e)}")


def example_transaction():
    """事务处理示例"""
    try:
        with mysql_pool.transaction() as cursor:
            # 在事务中执行的所有操作要么全部成功，要么全部失败
            cursor.execute(
                "INSERT INTO users (username, email, hashed_password) VALUES (%s, %s, %s)",
                ("transaction_user", "transaction@example.com", "password")
            )
            
            cursor.execute(
                "UPDATE users SET is_active = %s WHERE id = %s",
                (True, 1)
            )
            
            # 如果这里抛出异常，上面的操作会被回滚
            # raise Exception("模拟事务失败")
        
        logger.info("事务成功完成")
        
    except MySQLPoolError as e:
        logger.error(f"事务操作失败: {str(e)}")


def example_sqlalchemy_orm():
    """SQLAlchemy ORM使用示例"""
    try:
        # 使用会话上下文管理器
        with mysql_pool.session_scope() as session:
            # 创建新用户
            new_user = User(
                username="orm_user",
                email="orm_user@example.com",
                hashed_password="hashed_orm_password"
            )
            session.add(new_user)
            
            # 查询用户
            users = session.query(User).filter(User.is_active == True).all()
            logger.info(f"ORM查询到 {len(users)} 个活跃用户")
            
            # 更新用户
            user_to_update = session.query(User).filter(User.username == "orm_user").first()
            if user_to_update:
                user_to_update.email = "updated_email@example.com"
                logger.info(f"更新了用户 {user_to_update.username} 的邮箱")
        
        logger.info("ORM操作成功完成")
        
    except Exception as e:
        logger.error(f"ORM操作失败: {str(e)}")


def example_fastapi_dependency():
    """FastAPI依赖注入示例（伪代码）"""
    # 在FastAPI路由中使用
    """
    from fastapi import Depends, FastAPI
    from sqlalchemy.orm import Session
    
    app = FastAPI()
    
    @app.get("/users/{user_id}")
    def read_user(user_id: int, db: Session = Depends(get_db)):
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            return {"message": "User not found"}
        return user
    
    @app.post("/users/")
    def create_user(username: str, email: str, password: str, db: Session = Depends(get_db)):
        user = User(username=username, email=email, hashed_password=password)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    """
    pass


def create_tables():
    """创建表结构示例"""
    # 创建所有定义的表
    Base.metadata.create_all(bind=mysql_pool.engine)
    logger.info("表结构创建完成")


def example_connection_monitoring():
    """连接监控示例"""
    # 检查数据库连接是否正常
    if mysql_pool.check_connection():
        logger.info("数据库连接正常")
    else:
        logger.error("数据库连接异常")


if __name__ == "__main__":
    # 创建表结构
    create_tables()
    
    # 测试连接监控
    example_connection_monitoring()
    
    # 测试原始连接操作
    example_raw_connection()
    
    # 测试事务处理
    example_transaction()
    
    # 测试ORM操作
    example_sqlalchemy_orm() 