from typing import Generic, List, Optional, TypeVar, Dict, Any
from pydantic import BaseModel, Field, validator

# 定义分页请求模型
class PaginationParams(BaseModel):
    """分页参数模型
    
    用于所有需要分页的请求
    """
    page: int = Field(1, ge=1, description="页码，从1开始")
    page_size: int = Field(10, ge=1, le=100, description="每页条数，1-100")
    
    @validator('page')
    def page_must_be_positive(cls, v):
        if v < 1:
            return 1
        return v
    
    @validator('page_size')
    def page_size_must_be_valid(cls, v):
        if v < 1:
            return 10
        if v > 100:
            return 100
        return v

# 定义排序请求模型
class SortParams(BaseModel):
    """排序参数模型
    
    用于所有需要排序的请求
    """
    sort_by: str = Field(None, description="排序字段")
    sort_order: str = Field("desc", description="排序方向: asc或desc")
    
    @validator('sort_order')
    def sort_order_must_be_valid(cls, v):
        if v not in ["asc", "desc"]:
            return "desc"
        return v

# 定义搜索请求模型
class SearchParams(BaseModel):
    """搜索参数模型
    
    用于所有需要搜索的请求
    """
    keyword: Optional[str] = Field(None, description="搜索关键词")
    filter_by: Optional[Dict[str, Any]] = Field(None, description="过滤条件")

# 定义通用请求模型基类
class BaseRequest(BaseModel):
    """请求模型基类
    
    所有API请求模型的基类
    """
    class Config:
        # 额外属性处理策略：忽略额外属性
        extra = "ignore"
        # 允许字段别名
        allow_population_by_field_name = True
        # 验证时将字段值转换为适当类型
        validate_assignment = True
        # 必须使用声明过的字段
        arbitrary_types_allowed = False
        # 是否在响应中包含私有属性
        exclude_unset = True

# 定义任务创建请求模型示例
class CreateVideoTaskRequest(BaseRequest):
    """创建视频处理任务请求模型
    
    示例请求模型，实际使用时需根据业务需求调整
    """
    video_url: str = Field(..., description="视频URL")
    task_type: str = Field(..., description="任务类型")
    callback_url: Optional[str] = Field(None, description="回调URL")
    params: Optional[Dict[str, Any]] = Field(None, description="任务参数")
    
    @validator('task_type')
    def task_type_must_be_valid(cls, v):
        valid_types = ["transcription", "translation", "summary"]
        if v not in valid_types:
            raise ValueError(f"任务类型必须是以下之一: {', '.join(valid_types)}")
        return v
    
    @validator('video_url')
    def video_url_must_be_valid(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("视频URL必须以http://或https://开头")
        return v 