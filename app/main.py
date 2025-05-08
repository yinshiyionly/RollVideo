import logging
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.config import settings
from app.utils.logger import Logger
from typing import Any

# 导入自定义中间件
from app.models.response import error_response, StatusCode, StatusMessage
from app.middlewares.auth import AuthMiddleware

# 导入路由
from app.routers import router

# 初始化日志系统
log = Logger()

# 创建FastAPI应用实例
app = FastAPI(
    title=settings.APP_NAME,  # 设置应用名称
    description="滚动视频API服务",
    version="1.0.0",
    debug=settings.DEBUG,  # 设置调试模式
)

# 配置 uvicorn 访问日志级别
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# 注册中间件（注意：中间件按照相反的顺序执行，最后添加的最先执行）

# 响应中间件：确保所有响应符合标准格式


# 配置CORS中间件
# 允许跨域资源共享，使前端应用能够安全地访问API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源访问
    allow_credentials=True,  # 允许携带认证信息
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)

# 注册鉴权中间件
app.add_middleware(
    AuthMiddleware,
    exclude_paths=["/docs", "/redoc", "/openapi.json"]  # 排除 Swagger 文档路径
)

# 注册示例路由
app.include_router(
    router.router,
    prefix=f"{settings.API_V1_STR}",
)

# HTTP 异常处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    try:
        log.warning(
            f"HTTP异常 - 路径: {request.url.path}\n"
            f"状态码: {exc.status_code}\n"
            f"详情: {exc.detail}"
        )

        return error_response(
            code=StatusCode.BAD_REQUEST,
            message=f"HTTP错误: {str(exc.detail)}",
            status_code=StatusCode.BAD_REQUEST,
        )
        
    except Exception as e:
        log.error(f"处理HTTP异常时出错: {str(e)}")
        return error_response(
            code=StatusCode.SERVER_ERROR,
            message=f"HTTP异常: {str(e)}",
            status_code=StatusCode.SERVER_ERROR,
        )
    
# 覆盖 404 处理器
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return error_response(
        code=StatusCode.NOT_FOUND,
        message=StatusMessage.NOT_FOUND,
        status_code=StatusCode.NOT_FOUND,
    )

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    log.error(f"全局异常: {str(exc)}", extra={"path": request.url.path})
    return error_response(
        code=StatusCode.SERVER_ERROR,
        message=f"服务异常: {str(exc)}",
        status_code=StatusCode.SERVER_ERROR,
    )


# 添加全局异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误，包括JSON解析错误

    Args:
        request: 请求对象
        exc: 异常对象

    Returns:
        统一的错误响应格式
    """
    # 获取具体的错误信息
    error_detail = exc.errors()[0] if exc.errors() else {}
    error_type = error_detail.get('type', '')
    error_msg = error_detail.get('msg', '未知错误')

    if 'json_invalid' in error_type:
        message = 'JSON格式错误：请确保所有字段名使用双引号，并检查JSON语法'
    elif 'type_error' in error_type:
        message = f'数据类型错误：{error_msg}'
    else:
        message = f'参数验证失败：{error_msg}'

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=error_response(
            code=StatusCode.INVALID_PARAMS,
            message=message
        )
    )

# 包装路由处理函数，添加异常保护
def safe_endpoint(func: Any):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            log.error(f"路由处理发生异常: {str(e)}")
            return error_response(
                code=StatusCode.SERVER_ERROR,
                message=f"服务错误: {str(e)}",
                status_code=StatusCode.SERVER_ERROR,
            )
    return wrapper

# 对于所有的路由处理函数，建议都添加 @safe_endpoint 装饰器：
# @app.post("/some-endpoint")
# @safe_endpoint
# async def some_endpoint():
#     # 你的代码
#     pass


# 应用启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    log.info(f"服务启动成功 - {settings.APP_NAME}", 
             extra={"mode": "DEBUG" if settings.DEBUG else "PRODUCTION"})

# 应用关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    log.info(f"服务正在关闭 - {settings.APP_NAME}")

if __name__ == "__main__":
    """开发环境启动入口"""
    import uvicorn
    
    # 启动服务
    uvicorn.run(
        "app.main:app", 
        host=settings.APP_HOST, 
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
