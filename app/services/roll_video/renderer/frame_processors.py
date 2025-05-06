"""帧处理模块"""

import numpy as np
import logging
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import sys
import traceback
import time
import random

logger = logging.getLogger(__name__)

# 全局共享内存字典（进程内共享）
_SHARED_MEMORY_DICT = {}
_LOCAL_PROCESS_ID = None

def init_shared_memory(shared_dict):
    """
    初始化共享内存管理
    
    Args:
        shared_dict: 包含共享内存信息的字典
    """
    global _SHARED_MEMORY_DICT
    try:
        logger.info(f"初始化共享内存: {shared_dict['shm_name']}")
        _SHARED_MEMORY_DICT = shared_dict
        return True
    except Exception as e:
        logger.error(f"初始化共享内存失败: {str(e)}\n{traceback.format_exc()}")
        return False

def cleanup_shared_memory(shm_name=None):
    """
    清理共享内存
    
    Args:
        shm_name: 指定要清理的共享内存名称，如果为None则清理所有
    """
    global _SHARED_MEMORY_DICT
    try:
        if shm_name:
            logger.info(f"清理共享内存: {shm_name}")
            try:
                # 尝试找到并关闭指定的共享内存
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
                logger.info(f"成功清理共享内存: {shm_name}")
            except Exception as e:
                logger.warning(f"清理共享内存 {shm_name} 时出错: {str(e)}")
        else:
            # 清理_SHARED_MEMORY_DICT中记录的所有共享内存
            if _SHARED_MEMORY_DICT and 'shm_name' in _SHARED_MEMORY_DICT:
                try:
                    shm_name = _SHARED_MEMORY_DICT['shm_name']
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink()
                    logger.info(f"成功清理共享内存: {shm_name}")
                except Exception as e:
                    logger.warning(f"清理共享内存 {shm_name} 时出错: {str(e)}")
            
            _SHARED_MEMORY_DICT.clear()
            
        return True
    except Exception as e:
        logger.error(f"清理共享内存过程出错: {str(e)}\n{traceback.format_exc()}")
        return False

def init_worker(shared_dict):
    """
    初始化工作进程
    
    Args:
        shared_dict: 共享内存信息字典
    """
    global _SHARED_MEMORY_DICT, _LOCAL_PROCESS_ID
    
    # 设置进程名，便于调试
    _LOCAL_PROCESS_ID = os.getpid()
    try:
        mp.current_process().name = f"FrameProcessor-{_LOCAL_PROCESS_ID}"
    except:
        pass
    
    try:
        # 存储共享内存信息
        _SHARED_MEMORY_DICT = shared_dict
        
        # 在工作进程中记录初始化
        logger.info(f"工作进程 {_LOCAL_PROCESS_ID} 初始化，连接到共享内存 {shared_dict.get('shm_name', 'unknown')}")
        
        # 测试是否能访问共享内存
        if test_worker_shared_memory(shared_dict):
            logger.info(f"工作进程 {_LOCAL_PROCESS_ID} 成功连接到共享内存")
            return True
        else:
            logger.error(f"工作进程 {_LOCAL_PROCESS_ID} 连接共享内存失败")
            return False
    except Exception as e:
        logger.error(f"工作进程 {_LOCAL_PROCESS_ID} 初始化失败: {str(e)}\n{traceback.format_exc()}")
        return False

def test_worker_shared_memory(shared_dict):
    """
    测试工作进程共享内存访问
    
    Args:
        shared_dict: 共享内存信息字典
    
    Returns:
        bool: 测试成功返回True，否则返回False
    """
    pid = os.getpid()
    try:
        # 尝试访问共享内存
        shm_name = shared_dict.get('shm_name')
        img_shape = shared_dict.get('img_shape')
        
        if not shm_name or not img_shape:
            logger.error(f"进程 {pid}: 共享内存信息不完整")
            return False
        
        # 尝试连接到共享内存
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except Exception as e:
            logger.error(f"进程 {pid}: 无法连接到共享内存 {shm_name}: {str(e)}")
            return False
        
        # 尝试创建numpy数组读取共享内存数据
        try:
            shared_img = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)
            
            # 验证数据是否可读
            test_value = shared_img[0, 0, 0]
            
            # 检查dtype是否正确
            if shared_img.dtype != np.uint8:
                logger.error(f"进程 {pid}: 共享内存数据类型错误: {shared_img.dtype}")
                shm.close()
                return False
            
            # 关闭但不销毁
            shm.close()
            
            # 生成随机ID以标识此进程
            test_id = random.randint(10000, 99999)
            logger.info(f"进程 {pid} (ID:{test_id}): 共享内存测试成功，可以读取图像数据")
            return True
            
        except Exception as e:
            logger.error(f"进程 {pid}: 读取共享内存数据失败: {str(e)}")
            shm.close()
            return False
            
    except Exception as e:
        logger.error(f"进程 {pid}: 共享内存测试出错: {str(e)}\n{traceback.format_exc()}")
        return False

def _process_frame_optimized_shm(args):
    """
    使用共享内存优化的帧处理函数
    
    Args:
        args: 包含帧索引和帧元数据的元组
    
    Returns:
        元组 (帧索引, 帧数据) 或 None (如果处理失败)
    """
    global _SHARED_MEMORY_DICT, _LOCAL_PROCESS_ID
    
    frame_idx, frame_meta = args
    
    try:
        # 获取共享内存信息
        shm_name = _SHARED_MEMORY_DICT.get('shm_name')
        img_shape = _SHARED_MEMORY_DICT.get('img_shape')
        
        if not shm_name or not img_shape:
            logger.error(f"进程 {_LOCAL_PROCESS_ID}: 帧 {frame_idx} 缺少共享内存信息")
            return None
        
        # 尝试访问共享内存
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except Exception as e:
            logger.error(f"进程 {_LOCAL_PROCESS_ID}: 帧 {frame_idx} 无法访问共享内存: {str(e)}")
            return None
        
        try:
            # 创建对共享内存的视图
            source_img = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)
            
            # 从元数据获取参数
            width = frame_meta['width']
            height = frame_meta['height']
            img_height = frame_meta['img_height']
            scroll_speed = frame_meta['scroll_speed']
            fps = frame_meta['fps']
            
            # 创建输出帧
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # 计算当前帧的垂直位置
            if scroll_speed > 0:
                # 计算滚动偏移量
                total_scroll_distance = img_height - height
                current_position = min(
                    total_scroll_distance,
                    max(0, int(frame_idx * scroll_speed / fps))
                )
                
                # 计算源图像和目标帧的重叠区域
                src_start_y = current_position
                src_end_y = min(img_height, src_start_y + height)
                dst_start_y = 0
                dst_end_y = src_end_y - src_start_y
                
                # 复制图像数据
                frame[dst_start_y:dst_end_y, :, :] = source_img[src_start_y:src_end_y, :, :]
            else:
                # 如果不需要滚动，则居中放置图像
                if img_height < height:
                    start_y = (height - img_height) // 2
                    frame[start_y:start_y+img_height, :, :] = source_img
                else:
                    # 如果图像高度超过视频高度，仅显示顶部
                    frame[:, :, :] = source_img[:height, :, :]
            
            # 关闭共享内存（不销毁）
            shm.close()
            
            return (frame_idx, frame)
            
        except Exception as e:
            logger.error(f"进程 {_LOCAL_PROCESS_ID}: 处理帧 {frame_idx} 时出错: {str(e)}\n{traceback.format_exc()}")
            shm.close()
            # 返回黑色帧而不是None，以避免视频中断
            return (frame_idx, np.zeros((frame_meta['height'], frame_meta['width'], 3), dtype=np.uint8))
            
    except Exception as e:
        logger.error(f"进程 {_LOCAL_PROCESS_ID}: 处理帧 {frame_idx} 时发生致命错误: {str(e)}\n{traceback.format_exc()}")
        return None

def _process_frame(args):
    """多进程帧处理函数，高性能优化版本"""
    global _g_img_array
    
    frame_idx, img_start_y, img_height, img_width, self_height, self_width, frame_positions, is_transparent, bg_color = args
    
    # 快速路径 - 检查边界条件
    img_end_y = min(img_height, img_start_y + self_height)
    frame_start_y = 0
    frame_end_y = img_end_y - img_start_y
    
    # 空帧情况或边界检查失败时直接返回
    if img_start_y >= img_end_y or frame_start_y >= frame_end_y or frame_end_y > self_height or _g_img_array is None:
        # 创建空帧
        if is_transparent:
            frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
        else:
            # 使用背景色
            frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        return frame_idx, frame
    
    # 预分配整个帧缓冲区 - 减少内存分配开销
    if is_transparent:
        frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
    else:
        # 使用背景色
        frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
    
    # 计算切片 - 使用NumPy高效的切片操作
    img_h_slice = slice(img_start_y, img_end_y)
    img_w_slice = slice(0, min(self_width, img_width))
    frame_h_slice = slice(frame_start_y, frame_end_y)
    frame_w_slice = slice(0, min(self_width, img_width))
    
    try:
        # 获取源数据和目标区域 - 仅引用，不复制
        source_section = _g_img_array[img_h_slice, img_w_slice]
        target_area = frame[frame_h_slice, frame_w_slice]
        
        # 数据局部性优化 - 确保数据连续布局，加速内存访问
        source_section = np.ascontiguousarray(source_section)
        
        if is_transparent:
            # 透明背景 - 直接整块复制，避免逐像素操作
            if target_area.shape[:2] == source_section.shape[:2]:
                # 使用内存层面的快速复制
                np.copyto(target_area, source_section)
            else:
                # 处理大小不匹配的情况
                copy_height = min(target_area.shape[0], source_section.shape[0])
                copy_width = min(target_area.shape[1], source_section.shape[1])
                if copy_height > 0 and copy_width > 0:
                    # 使用预先计算的切片提高性能
                    src_view = source_section[:copy_height, :copy_width]
                    dst_view = target_area[:copy_height, :copy_width]
                    np.copyto(dst_view, src_view)
        else:
            # 不透明背景 - 使用向量化的alpha混合
            if target_area.shape[:2] == source_section.shape[:2]:
                # 向量化操作，一次计算所有像素
                alpha = source_section[:, :, 3:4].astype(np.float32) / 255.0
                alpha_inv = 1.0 - alpha
                # 预分配输出数组，减少临时内存分配
                blended = np.empty_like(target_area)
                # 内联计算提高性能
                np.multiply(source_section[:, :, :3], alpha, out=blended)
                np.add(blended, target_area * alpha_inv, out=blended, casting='unsafe')
                np.copyto(target_area, blended.astype(np.uint8))
            else:
                # 处理大小不匹配的情况
                copy_height = min(target_area.shape[0], source_section.shape[0])
                copy_width = min(target_area.shape[1], source_section.shape[1])
                if copy_height > 0 and copy_width > 0:
                    src_crop = source_section[:copy_height, :copy_width]
                    dst_crop = target_area[:copy_height, :copy_width]
                    # 使用连续数组提高性能
                    src_crop = np.ascontiguousarray(src_crop)
                    dst_crop = np.ascontiguousarray(dst_crop)
                    alpha = src_crop[:, :, 3:4].astype(np.float32) / 255.0
                    alpha_inv = 1.0 - alpha
                    # 预分配输出数组，减少临时内存分配
                    blended = np.empty_like(dst_crop)
                    # 内联计算提高性能
                    np.multiply(src_crop[:, :, :3], alpha, out=blended)
                    np.add(blended, dst_crop * alpha_inv, out=blended, casting='unsafe')
                    np.copyto(target_area[:copy_height, :copy_width], blended.astype(np.uint8))
    except Exception as e:
        logger.error(f"处理帧 {frame_idx} 时出错: {e}")
    
    # 确保返回连续内存布局
    return frame_idx, np.ascontiguousarray(frame)

def _process_frame_optimized(args):
    """极度优化的帧处理函数"""
    global _g_img_array
    
    frame_idx, img_start_y, img_height, img_width, self_height, self_width, is_transparent, bg_color = args
    
    # 快速路径 - 直接计算切片位置
    img_end_y = min(img_height, img_start_y + self_height)
    visible_height = img_end_y - img_start_y
    
    # 零边界检查 - 极端情况直接返回背景
    if visible_height <= 0 or _g_img_array is None:
        if is_transparent:
            # 预分配连续内存
            frame = np.zeros((self_height, self_width, 4), dtype=np.uint8, order='C')
        else:
            # 使用背景色
            frame = np.ones((self_height, self_width, 3), dtype=np.uint8, order='C') * np.array(bg_color, dtype=np.uint8)
        return frame_idx, frame
    
    # 预分配帧缓冲区 - 提高效率，确保内存连续
    if is_transparent:
        # 创建空的透明帧
        frame = np.zeros((self_height, self_width, 4), dtype=np.uint8, order='C')
    else:
        # 用背景色填充帧
        frame = np.ones((self_height, self_width, 3), dtype=np.uint8, order='C') * np.array(bg_color, dtype=np.uint8)
    
    # 计算切片 - 超高效版
    source_height = min(visible_height, self_height)
    source_width = min(img_width, self_width)
    
    # 使用直接视图赋值，避免任何额外复制
    if source_height > 0 and source_width > 0:
        # 单一高效赋值操作
        frame[:source_height, :source_width] = _g_img_array[img_start_y:img_start_y+source_height, :source_width]
    
    # 确保返回内存连续的帧数据 - 此步骤对于管道通信至关重要
    # 由于我们已经在创建时指定order='C'，这里可以省略额外的检查
    return frame_idx, frame

def fast_frame_processor(batch_frames, memory_pool, ffmpeg_process):
    """
    高性能帧处理器，直接将帧写入FFmpeg进程
    
    Args:
        batch_frames: 批处理帧参数列表
        memory_pool: 内存池
        ffmpeg_process: FFmpeg进程
        
    Returns:
        处理的帧数
    """
    global _g_img_array
    
    if _g_img_array is None:
        return 0
        
    frames_processed = 0
    
    for args in batch_frames:
        frame_idx, img_start_y, img_height, img_width, self_height, self_width, is_transparent, bg_color = args
        
        # 快速路径计算
        img_end_y = min(img_height, img_start_y + self_height)
        visible_height = img_end_y - img_start_y
        
        # 从内存池获取帧缓冲区
        if memory_pool is not None and len(memory_pool) > 0:
            frame = memory_pool.pop()
            if is_transparent:
                frame.fill(0)  # 透明帧填充0
            else:
                # RGB帧填充背景色
                frame[:, :, 0].fill(bg_color[0])
                frame[:, :, 1].fill(bg_color[1])
                frame[:, :, 2].fill(bg_color[2])
        else:
            # 内存池为空，创建新帧
            if is_transparent:
                frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
            else:
                frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        
        # 计算切片并复制数据
        source_height = min(visible_height, self_height)
        source_width = min(img_width, self_width)
        
        if source_height > 0 and source_width > 0:
            frame[:source_height, :source_width] = _g_img_array[img_start_y:img_start_y+source_height, :source_width]
        
        # 直接写入FFmpeg进程
        ffmpeg_process.stdin.write(frame.tobytes())
        
        # 将帧缓冲区放回内存池
        if memory_pool is not None:
            memory_pool.append(frame)
            
        frames_processed += 1
        
    return frames_processed