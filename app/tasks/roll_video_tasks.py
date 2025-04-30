import os
import datetime
import time
import requests
from app.models.roll_video_task_db import RollVideoTaskDB
from app.models.roll_video_task import TaskState
from app.utils.logger import Logger
from app.celery_app import celery_app
from app.services.roll_video.roll_video_service import RollVideoService
from app.utils.tos_client import TOSClient
from app.config import settings

# 初始化日志
log = Logger('roll-video-celery')

@celery_app.task(name='app.tasks.roll_video_tasks.push_event_to_client', bind=True)
def push_event_to_client(self, task_id: str, video_url: str, retry_count: int = 0, event_type: str = "video_generated", extra_data: dict = None):
    """向客户端推送事件
    
    Args:
        task_id: 任务ID
        video_url: 视频URL（成功时有值）
        retry_count: 重试次数
        event_type: 事件类型（video_generated, video_failed）
        extra_data: 额外数据
    """
    if not hasattr(settings, 'CLIENT_NOTIFY_URL') or not settings.CLIENT_NOTIFY_URL:
        log.warning(f"未配置CLIENT_NOTIFY_URL，无法推送事件")
        return
        
    log.info(f"开始向客户端推送事件: task_id={task_id}, event_type={event_type}, retry_count={retry_count}")
    
    # 获取任务信息
    task_db = RollVideoTaskDB()
    task = task_db.get_task(task_id)
    if not task:
        log.error(f"推送事件失败，任务不存在: {task_id}")
        return
        
    # 构建推送数据
    payload = {
        "task_id": task_id,
        "uid": task.uid,
        "source": task.source,
        "event_type": event_type,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 根据事件类型添加不同的数据
    if event_type == "video_generated" and video_url:
        payload["video_url"] = video_url
    
    # 添加额外数据
    if extra_data and isinstance(extra_data, dict):
        payload.update(extra_data)
        
    try:
        # 发送请求
        response = requests.post(
            settings.CLIENT_NOTIFY_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10  # 10秒超时
        )
        
        # 检查响应
        if response.status_code == 200 and response.text.strip().lower() == 'success':
            log.info(f"推送事件成功: task_id={task_id}")
            return True
            
        log.warning(f"推送事件失败: status_code={response.status_code}, response={response.text}, task_id={task_id}")
        
        # 如果失败并且重试次数未达到最大值，则安排重试
        if retry_count < 3:
            # 确定下次重试的延迟时间
            delay_minutes = [5, 10, 30][retry_count]
            delay_seconds = delay_minutes * 60
            
            log.info(f"将在{delay_minutes}分钟后重试推送事件: task_id={task_id}, retry_count={retry_count+1}")
            
            # 安排延迟任务
            push_event_to_client.apply_async(
                args=[task_id, video_url, retry_count + 1, event_type, extra_data],
                countdown=delay_seconds
            )
        else:
            log.error(f"推送事件失败，已达到最大重试次数: task_id={task_id}")
            
    except Exception as e:
        log.error(f"推送事件过程中发生异常: {str(e)}")
        
        # 如果出现异常并且重试次数未达到最大值，则安排重试
        if retry_count < 3:
            # 确定下次重试的延迟时间
            delay_minutes = [5, 10, 30][retry_count]
            delay_seconds = delay_minutes * 60
            
            log.info(f"将在{delay_minutes}分钟后重试推送事件: task_id={task_id}, retry_count={retry_count+1}")
            
            # 安排延迟任务
            push_event_to_client.apply_async(
                args=[task_id, video_url, retry_count + 1, event_type, extra_data],
                countdown=delay_seconds
            )
        else:
            log.error(f"推送事件失败，已达到最大重试次数: task_id={task_id}")

@celery_app.task(name='app.tasks.roll_video_tasks.generate_roll_video_task', bind=True)
def generate_roll_video_task(self, task_id: str):
    """处理视频任务
    
    Args:
        task_id: 任务ID
    """
    try:
        # 1. 获取任务信息
        task_db = RollVideoTaskDB()
        task = task_db.get_task(task_id)
        if not task:
            log.error(f"任务不存在: {task_id}")
            return
            
        # 2. 更新任务状态为处理中
        update_task_status(task_id, TaskState.PROCESSING, {"progress": 0})
        
        # 3. 生成滚动视频保存目录
        roll_video_filename = os.path.join(settings.VIDEO_TMP_DIR, f"{task_id}")
        log.info(f"roll_video_filename: {task.payload}")
        # 3.1 创建服务实例
        service = RollVideoService()
        
         # 3.2使用 task.payload 来传递所有参数
        result = service.create_roll_video(
            output_path=roll_video_filename, # Service 会修正扩展名
            **task.payload
        )

        # 3.3 输出结果
        if result["status"] == "success" and result["output_path"] and os.path.exists(result["output_path"]):
            log.info(f"生成视频成功, task_id: {task_id}, 生成结果: {result}")
            # 3.4 上传 tos
            try:
                object_key = upload_video_to_tos(task_id, result['output_path'])
                # 构建完整的CDN URL，确保正确拼接
                tos_cdn = settings.TOS_CDN.rstrip('/')  # 移除末尾可能的斜杠
                tos_url = f"{tos_cdn}/{object_key.lstrip('/')}"  # 确保object_key不以斜杠开头
                # 3.5 更新任务状态为完成
                update_task_status(task_id, TaskState.COMPLETED, {"tos_url": tos_url})
                # 3.6 向客户端推送成功事件
                if hasattr(settings, 'CLIENT_NOTIFY_URL') and settings.CLIENT_NOTIFY_URL:
                    push_event_to_client.delay(task_id, tos_url, 0, "video_generated")
            except Exception as e:
                log.error(f"上传TOS失败: {str(e)}")
                update_task_status(task_id, TaskState.FAILED, {"error": f"视频生成成功但上传失败: {str(e)}"})
                # 4.1 向客户端推送失败事件
                if hasattr(settings, 'CLIENT_NOTIFY_URL') and settings.CLIENT_NOTIFY_URL:
                    push_event_to_client.delay(task_id, "", 0, "video_failed", {"error": f"视频生成成功但上传失败: {str(e)}"})
            finally:
                # 清理临时文件
                cleanup_temp_file(result.get("output_path"))
        else:
            error_msg = f"生成视频失败, task_id: {task_id}"
            if result.get("message"):
                error_msg += f", 错误信息: {result['message']}"
            log.error(error_msg)
            update_task_status(task_id, TaskState.FAILED, {"error": error_msg})
            # 5.1 向客户端推送失败事件
            if hasattr(settings, 'CLIENT_NOTIFY_URL') and settings.CLIENT_NOTIFY_URL:
                push_event_to_client.delay(task_id, "", 0, "video_failed", {"error": error_msg})
            # 尝试清理可能存在的临时文件
            cleanup_temp_file(result.get("output_path"))
        
    except Exception as e:
        log.error(f"处理视频任务失败: {str(e)}")
        # 更新任务状态为失败
        update_task_status(task_id, TaskState.FAILED, {"error": str(e)})

def update_task_status(task_id: str, state: TaskState, result: dict = None):
    """更新任务状态
    
    Args:
        task_id: 任务ID
        state: 任务状态
        result: 任务结果
    """
    try:
        task_db = RollVideoTaskDB()
        task_db.update_task_state(
            task_id=task_id,
            task_state=state,
            result=result
        )
        log.info(f"已更新任务状态: task_id={task_id}, state={state}, result={result}")
    except Exception as e:
        log.error(f"更新任务状态失败: {str(e)}")

def cleanup_temp_file(file_path):
    """清理临时文件
    
    Args:
        file_path: 文件路径
    """
    if not file_path:
        return
        
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            log.info(f"已清理临时文件: {file_path}")
    except Exception as e:
        log.error(f"清理临时文件失败: {str(e)}")

def upload_video_to_tos(task_id: str, video_path: str):
    """上传文件至TOS

    Args:
        task_id: 任务ID
        video_path: 本地生成成功的视频文件路径
    """
    if not os.path.exists(video_path):
        raise Exception(f"上传文件不存在: {video_path}")
    
    # 最大重试3次
    max_retries = 3
    # 重试计数器
    retry_count = 0
    # 记录最后一次错误信息
    last_error = None
    
    # 重试逻辑
    while retry_count < max_retries:
        try:
            # 初始化 TOS 客户端
            tos_client = TOSClient()
            # 获取源文件扩展名
            ext = os.path.splitext(video_path)[1]  # 包含 `.` 例如 `.mp3`
            now = datetime.datetime.now()
            # 生成 object_key
            video_object_key = f"roll-video/{now.year}/{now.month:02d}/{task_id}{ext}"
            # 执行上传TOS操作
            tos_client.upload_file(
                local_file_path=video_path,
                object_key=video_object_key,
                metadata={"task_id": task_id},
            )
            # 返回
            return video_object_key
        except Exception as e:
            retry_count += 1
            last_error = e
            log.warning(f"上传TOS失败(第{retry_count}次), task_id: {task_id}, error: {str(e)}")
            if retry_count < max_retries:
                # 等待一段时间后重试
                time.sleep(5)
    
    # 所有重试都失败
    msg = f"上传TOS失败(已重试{max_retries}次), task_id: {task_id}, video_path: {video_path}, 最后错误: {str(last_error)}"
    log.error(msg)
    raise Exception(msg)