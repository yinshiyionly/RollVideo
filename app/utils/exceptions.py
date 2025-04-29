from fastapi import Request
from fastapi.responses import JSONResponse
from app.utils.logger import Logger
# 初始化日志系统
log = Logger()

async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled Exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "Internal Server Error", "data": None},
    )
