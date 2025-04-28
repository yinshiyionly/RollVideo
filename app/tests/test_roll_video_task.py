import pytest
from uuid import uuid4
from app.services.roll_video.models.roll_video_task import (
    TaskState, 
    TaskStatus, 
    RollVideoTaskCreate, 
    RollVideoTaskUpdate
)
from app.services.mysql.roll_video_task_db import RollVideoTaskDB


@pytest.fixture
def roll_video_task_db():
    """创建数据库操作实例"""
    return RollVideoTaskDB()


@pytest.fixture
def create_task_params():
    """创建测试任务参数"""
    task_id = f"test-{uuid4()}"
    return {
        "task_id": task_id,
        "uid": 10001,
        "source": "test",
        "payload": {"video_url": "https://example.com/test.mp4"}
    }


def test_create_and_get_task(roll_video_task_db, create_task_params):
    """测试创建和获取任务"""
    # 创建任务
    task = RollVideoTaskCreate(**create_task_params)
    task_id = roll_video_task_db.create_task(task)
    
    # 验证返回的任务ID正确
    assert task_id == create_task_params["task_id"]
    
    # 获取任务
    stored_task = roll_video_task_db.get_task(task_id)
    
    # 验证获取的任务数据正确
    assert stored_task is not None
    assert stored_task.task_id == task_id
    assert stored_task.uid == create_task_params["uid"]
    assert stored_task.source == create_task_params["source"]
    assert stored_task.task_state == TaskState.PENDING
    assert stored_task.payload == create_task_params["payload"]
    assert stored_task.status == TaskStatus.NORMAL


def test_update_task_state(roll_video_task_db, create_task_params):
    """测试更新任务状态"""
    # 创建任务
    task = RollVideoTaskCreate(**create_task_params)
    task_id = roll_video_task_db.create_task(task)
    
    # 更新任务状态
    result = {"progress": 50}
    success = roll_video_task_db.update_task_state(
        task_id, TaskState.PROCESSING, result
    )
    
    # 验证更新成功
    assert success is True
    
    # 验证更新后的数据
    updated_task = roll_video_task_db.get_task(task_id)
    assert updated_task.task_state == TaskState.PROCESSING
    assert updated_task.result == result


def test_update_task(roll_video_task_db, create_task_params):
    """测试更新任务信息"""
    # 创建任务
    task = RollVideoTaskCreate(**create_task_params)
    task_id = roll_video_task_db.create_task(task)
    
    # 更新任务
    update_data = RollVideoTaskUpdate(
        task_state=TaskState.COMPLETED,
        result={"output": "https://example.com/output.mp4"},
        payload={"video_url": "https://example.com/updated.mp4"}
    )
    success = roll_video_task_db.update_task(task_id, update_data)
    
    # 验证更新成功
    assert success is True
    
    # 验证更新后的数据
    updated_task = roll_video_task_db.get_task(task_id)
    assert updated_task.task_state == TaskState.COMPLETED
    assert updated_task.result == update_data.result
    assert updated_task.payload == update_data.payload


def test_delete_task(roll_video_task_db, create_task_params):
    """测试删除任务"""
    # 创建任务
    task = RollVideoTaskCreate(**create_task_params)
    task_id = roll_video_task_db.create_task(task)
    
    # 删除任务
    success = roll_video_task_db.delete_task(task_id)
    
    # 验证删除成功
    assert success is True
    
    # 验证任务已被标记为删除
    deleted_task = roll_video_task_db.get_task(task_id)
    assert deleted_task is None  # 因为get_task只返回status=1的记录


def test_list_tasks(roll_video_task_db):
    """测试列出任务"""
    # 创建多个测试任务
    uid = 10002
    source = "test_list"
    
    # 创建3个任务
    tasks = []
    for i in range(3):
        task_params = {
            "task_id": f"list-{uuid4()}",
            "uid": uid,
            "source": source,
            "payload": {"index": i}
        }
        task = RollVideoTaskCreate(**task_params)
        roll_video_task_db.create_task(task)
        tasks.append(task_params)
    
    # 测试按用户ID查询
    user_tasks = roll_video_task_db.list_tasks(uid=uid)
    assert len(user_tasks) >= 3
    
    # 测试按来源查询
    source_tasks = roll_video_task_db.list_tasks(source=source)
    assert len(source_tasks) >= 3
    
    # 测试按状态查询
    pending_tasks = roll_video_task_db.list_tasks(task_state=TaskState.PENDING)
    assert len(pending_tasks) >= 3
    
    # 测试分页
    paged_tasks = roll_video_task_db.list_tasks(limit=2)
    assert len(paged_tasks) == 2 