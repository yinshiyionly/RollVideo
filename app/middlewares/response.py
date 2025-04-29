from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.utils.logger import Logger
from app.models.response import success_response, error_response, StatusCode, StatusMessage


class ResponseMiddleware(BaseHTTPMiddleware):
    """响应处理中间件
    
    确保所有API响应都遵循统一的响应格式:
    {
        "code": 200,             # 业务状态码
        "message": "success",    # 状态消息
        "data": {},              # 响应数据
    }
    """
    
    def __init__(self, app: ASGIApp, exclude_paths: list = None):
        """初始化中间件
        
        Args:
            app: ASGI应用
            exclude_paths: 排除路径列表，这些路径的响应不会被处理(如Swagger文档)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/docs", "/redoc", "/openapi.json"]
        self.logger = Logger()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求和响应
        
        Args:
            request: 请求对象
            call_next: 下一个中间件或路由处理函数
            
        Returns:
            Response: 标准化的响应对象
        """
        # 排除特定路径
        for path in self.exclude_paths:
            if request.url.path.startswith(path):
                return await call_next(request)
        
        # 调用下一个处理函数
        response = await call_next(request)
        
        # 只处理JSONResponse
        if not isinstance(response, JSONResponse):
            return response
        
        # 获取响应内容
        response_body = response.body
        response_json = {}
        
        try:
            # 解析响应内容
            import json
            response_json = json.loads(response_body)
            
            # 检查响应是否已经是标准格式
            if self._is_standard_response(response_json):
                return response
            
            # 根据状态码确定响应类型
            status_code = response.status_code
            if 200 <= status_code < 400:
                # 成功响应
                standard_response = success_response(
                    data=response_json,
                    message=StatusMessage.SUCCESS
                )
            else:
                # 错误响应
                error_msg = response_json.get("detail", StatusMessage.SERVER_ERROR) 
                if isinstance(error_msg, dict) and "msg" in error_msg:
                    error_msg = error_msg["msg"]
                
                standard_response = error_response(
                    code=self._map_http_to_business_code(status_code),
                    message=error_msg,
                    data=None
                )
            
            # 创建新的标准化响应
            return JSONResponse(
                status_code=200,  # HTTP状态码始终为200
                content=standard_response,
                headers=dict(response.headers.items())
            )
            
        except Exception as e:
            self.logger.error(f"响应中间件处理异常: {str(e)}")
            # 出现异常时返回原始响应
            return response
    
    def _is_standard_response(self, response_data: Dict[str, Any]) -> bool:
        """检查响应是否已符合标准格式
        
        Args:
            response_data: 响应数据
            
        Returns:
            bool: 是否符合标准格式
        """
        return (
            isinstance(response_data, dict) and
            "code" in response_data and
            "message" in response_data and
            "data" in response_data
        )
    
    def _map_http_to_business_code(self, http_status: int) -> int:
        """将HTTP状态码映射为业务状态码
        
        Args:
            http_status: HTTP状态码
            
        Returns:
            int: 业务状态码
        """
        code_map = {
            400: StatusCode.BAD_REQUEST,
            401: StatusCode.UNAUTHORIZED,
            403: StatusCode.FORBIDDEN,
            404: StatusCode.NOT_FOUND,
            405: StatusCode.METHOD_NOT_ALLOWED,
            500: StatusCode.SERVER_ERROR,
            503: StatusCode.SERVICE_UNAVAILABLE
        }
        return code_map.get(http_status, StatusCode.SERVER_ERROR) 