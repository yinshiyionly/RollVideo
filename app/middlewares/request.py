from typing import Callable, Dict, Any
import time
import json
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.utils.logger import Logger
from app.models.response import error_response, StatusCode, StatusMessage


class RequestMiddleware(BaseHTTPMiddleware):
    """请求处理中间件
    
    处理所有传入的请求:
    1. 记录请求信息
    2. 处理请求耗时统计
    3. 请求体验证
    """
    
    def __init__(self, app: ASGIApp, exclude_paths: list = None):
        """初始化中间件
        
        Args:
            app: ASGI应用
            exclude_paths: 排除路径列表，这些路径的请求不会被处理(如Swagger文档)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/docs", "/redoc", "/openapi.json", "/favicon.ico"]
        self.logger = Logger()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个中间件或路由处理函数
            
        Returns:
            Response: 响应对象
        """
        # 记录请求开始时间
        start_time = time.time()
        
        # 排除特定路径
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # 获取请求ID
        request_id = request.headers.get("X-Request-ID", "")
        
        # 记录请求信息
        await self._log_request(request, request_id)
        
        try:
            # 继续处理请求
            response = await call_next(request)
            
            # 记录响应信息
            process_time = time.time() - start_time
            self._log_response(request, response, process_time, request_id)
            
            return response
            
        except Exception as e:
            # 处理异常
            self.logger.error(f"请求处理异常: {str(e)}", {"request_id": request_id})
            process_time = time.time() - start_time
            
            # 返回错误响应
            error_resp = error_response(
                code=StatusCode.SERVER_ERROR,
                message=f"请求处理失败: {str(e)}"
            )
            
            return Response(
                content=json.dumps(error_resp),
                status_code=500,
                media_type="application/json"
            )
    
    async def _log_request(self, request: Request, request_id: str):
        """记录请求信息
        
        Args:
            request: 请求对象
            request_id: 请求ID
        """
        # 获取请求头
        headers = dict(request.headers.items())
        
        # 安全起见，移除敏感信息
        if "authorization" in headers:
            headers["authorization"] = "***"
        if "cookie" in headers:
            headers["cookie"] = "***"
        
        # 尝试获取请求体(对于某些请求类型可能无法获取)
        body = ""
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body_bytes = await request.body()
                body = body_bytes.decode()
                if len(body) > 1000:  # 限制请求体日志长度
                    body = body[:1000] + "... [截断]"
            except Exception:
                body = "<无法读取请求体>"
        
        # 构建日志信息
        request_info = {
            "request_id": request_id,
            "client_ip": request.client.host if request.client else "unknown",
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "headers": headers,
            "body": body if body else None,
            "query_params": dict(request.query_params)
        }
        
        # 记录请求信息
        self.logger.info("API请求开始", request_info)
    
    def _log_response(self, request: Request, response: Response, process_time: float, request_id: str):
        """记录响应信息
        
        Args:
            request: 请求对象
            response: 响应对象
            process_time: 处理时间(秒)
            request_id: 请求ID
        """
        # 构建日志信息
        response_info = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": f"{process_time:.6f}s"
        }
        
        # 根据状态码选择日志级别
        if response.status_code >= 500:
            self.logger.error("API请求完成", response_info)
        elif response.status_code >= 400:
            self.logger.warning("API请求完成", response_info)
        else:
            self.logger.info("API请求完成", response_info) 