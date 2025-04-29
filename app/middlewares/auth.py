from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.models.response import error_response, StatusCode, StatusMessage
from app.config import settings
from app.utils.logger import Logger

log = Logger('auth-middleware')

class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件
    
    处理所有需要认证的请求:
    1. 验证 x-token 是否存在
    2. 验证 x-token 是否有效
    """
    
    def __init__(self, app: ASGIApp, exclude_paths: list = None):
        """初始化中间件
        
        Args:
            app: ASGI应用
            exclude_paths: 排除路径列表，这些路径不需要认证(如Swagger文档)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or []
        self.valid_tokens = settings.X_TOKEN if isinstance(settings.X_TOKEN, list) else [settings.X_TOKEN]

    async def dispatch(self, request: Request, call_next):
        """处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个中间件或路由处理函数
        """
        try:
            # 检查是否为排除路径
            if any(request.url.path.startswith(path) for path in self.exclude_paths):
                return await call_next(request)

            # 验证token是否存在
            token = request.headers.get("x-token")
            if not token:
                return error_response(
                    code=StatusCode.UNAUTHORIZED,
                    message=StatusMessage.UNAUTHORIZED,
                    status_code=StatusCode.UNAUTHORIZED
                )
            
            # 验证token是否有效
            if token not in self.valid_tokens:
                return error_response(
                    code=StatusCode.FORBIDDEN,
                    message=StatusMessage.FORBIDDEN,
                    status_code=StatusCode.FORBIDDEN,
                )

            return await call_next(request)
        except Exception:
            # 异常情况
            return error_response(
                code=StatusCode.SERVER_ERROR,
                message=StatusMessage.SERVER_ERROR,
                status_code=StatusCode.SERVER_ERROR
            )
