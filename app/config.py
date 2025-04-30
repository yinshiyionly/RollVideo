from pydantic_settings import BaseSettings
from typing import Set


class Settings(BaseSettings):
    """应用程序配置类

    用于管理应用程序的所有配置项，包括基础设置、文件存储、音视频处理参数等。
    继承自pydantic.BaseSettings，支持从环境变量和.env文件加载配置。
    """

    # 基础配置
    APP_NAME: str  # 应用程序名称
    APP_HOST: str # 应用主机
    APP_PORT: int  # 应用端口
    API_V1_STR: str  # API版本前缀
    DEBUG: bool  # 调试模式开关
    X_TOKEN: str # 接口鉴权
    CLIENT_NOTIFY_URL:str  # 客户端接收推送事件地址

    # 视频处理配置
    VIDEO_TMP_DIR: str # 视频临时保存目录

    # MySQL配置
    MYSQL_HOST: str  # MySQL主机地址
    MYSQL_PORT: int  # MySQL端口
    MYSQL_DATABASE: str  # MySQL数据库名
    MYSQL_USER: str  # MySQL用户名
    MYSQL_PASSWORD: str  # MySQL密码
    MYSQL_ROOT_PASSWORD: str  # MySQL root密码

    # Redis配置
    REDIS_HOST: str  # Redis主机地址
    REDIS_PORT: int  # Redis端口
    REDIS_PASSWORD: str  # Redis密码

    # 火山引擎配置
    VOLC_AK: str  # 火山引擎访问密钥ID
    VOLC_SK: str  # 火山引擎访问密钥密码

    # TOS云存储配置
    TOS_REGION: str  # TOS区域
    TOS_ENDPOINT: str  # TOS终端节点
    TOS_BUCKET: str  # TOS存储桶名称
    TOS_BUCKET_HOST_PUB: str  # TOS公网访问域名
    TOS_BUCKET_HOST_PRI: str  # TOS内网访问域名
    TOS_SCHEME: str  # TOS访问协议
    TOS_CDN: str # TOS CDN或公网访问域名

    # 日志配置
    LOG_LEVEL: str  # 日志级别
    LOG_FORMAT: str  # 日志格式
    LOG_DIR: str  # 日志文件目录
    LOG_FILE_PREFIX: str  # 日志文件名前缀
    LOG_FILE_MAX_BYTES: int  # 单个日志文件最大大小
    LOG_FILE_BACKUP_COUNT: int  # 日志文件备份数量

    class Config:
        """配置类设置

        case_sensitive: 区分大小写
        env_file: 环境变量文件路径
        """

        case_sensitive = True
        env_file = ".env"


settings = Settings()  # 创建全局配置实例
