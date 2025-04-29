from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

T = TypeVar('T')

class ResponseModel(BaseModel, Generic[T]):
    """标准API响应模型
    
    所有API响应都使用这个标准格式:
    {
        "code": 200,             # 业务状态码
        "message": "success",    # 状态消息
        "data": {},              # 响应数据(可选)
    }
    """
    code: int = Field(200, description="业务状态码")
    message: str = Field("success", description="状态消息")
    data: Optional[T] = Field(None, description="响应数据")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": None
            }
        }

# 预定义的业务状态码
class StatusCode:
    SUCCESS = 200                  # 成功
    BAD_REQUEST = 400              # 请求参数错误
    UNAUTHORIZED = 401             # 未授权
    FORBIDDEN = 403                # 禁止访问
    NOT_FOUND = 404                # 资源不存在
    METHOD_NOT_ALLOWED = 405       # 方法不允许
    SERVER_ERROR = 500             # 服务器内部错误
    SERVICE_UNAVAILABLE = 503      # 服务不可用
    
    # 自定义业务状态码(建议从1000开始)
    TASK_NOT_FOUND = 1001          # 任务不存在
    TASK_ALREADY_EXISTS = 1002     # 任务已存在
    TASK_CREATION_FAILED = 1003    # 任务创建失败
    TASK_PROCESSING = 1004         # 任务处理中

# 预定义的状态消息
class StatusMessage:
    SUCCESS = "success"
    BAD_REQUEST = "请求参数错误"
    UNAUTHORIZED = "未授权访问"
    FORBIDDEN = "禁止访问"
    NOT_FOUND = "资源不存在"
    METHOD_NOT_ALLOWED = "方法不允许"
    SERVER_ERROR = "服务器内部错误"
    SERVICE_UNAVAILABLE = "服务暂不可用"
    
    # 自定义业务状态消息
    TASK_NOT_FOUND = "任务不存在"
    TASK_ALREADY_EXISTS = "任务已存在"
    TASK_CREATION_FAILED = "任务创建失败"
    TASK_PROCESSING = "任务处理中"

# 成功响应工厂函数
def success_response(data: Any = None, message: str = StatusMessage.SUCCESS) -> Dict:
    """创建成功响应"""
    return {
        "code": StatusCode.SUCCESS,
        "message": message,
        "data": data
    }

# 错误响应工厂函数
def error_response(
    code: int = StatusCode.SERVER_ERROR, 
    message: str = StatusMessage.SERVER_ERROR,
    data: Any = None,
    status_code: int = None
) -> JSONResponse:
    """创建错误响应
    
    Args:
        code: 业务状态码
        message: 错误消息
        data: 响应数据
        status_code: HTTP状态码，如果不指定则使用业务状态码
        
    Returns:
        JSONResponse: 标准化的错误响应
    """
    content = {
        "code": code,
        "message": message,
        "data": data
    }
    
    # 如果没有指定 status_code，则使用业务状态码
    if status_code is None:
        # 对于自定义业务状态码（>=1000），使用200作为HTTP状态码
        status_code = 200 if code >= 1000 else code
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )

# 分页响应模型
class PaginatedResponseModel(BaseModel, Generic[T]):
    """分页响应数据模型"""
    items: List[T] = Field(..., description="分页数据项")
    total: int = Field(..., description="总数据量")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页数据量")
    pages: int = Field(..., description="总页数")

# 分页响应工厂函数
def paginated_response(
    items: List, 
    total: int, 
    page: int, 
    page_size: int
) -> Dict:
    """创建分页响应"""
    return success_response(data={
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size
    }) 