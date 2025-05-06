"""帧处理模块"""

import numpy as np
import logging
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)

# 在多线程/多进程中共享的全局变量
_g_img_array = None  # 全局共享的图像数组

# 共享内存变量
_shm = None  # 共享内存对象引用
_shm_name = None  # 共享内存名称
_shm_shape = None  # 图像形状
_shm_dtype = np.uint8  # 数据类型

def init_shared_memory(image_array):
    """初始化共享内存并存储图像数据
    
    Args:
        image_array: 源图像的NumPy数组
        
    Returns:
        shared_memory_name: 共享内存段的名称
        shape: 图像数组的形状
    """
    global _shm, _shm_name, _shm_shape
    
    # 获取数组信息
    nbytes = image_array.nbytes
    shape = image_array.shape
    dtype = image_array.dtype
    
    # 创建共享内存段
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    
    # 创建共享内存的NumPy视图并复制数据
    shm_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np.copyto(shm_array, image_array)
    
    logger.info(f"创建共享内存: 大小={nbytes/(1024*1024):.2f}MB, 形状={shape}")
    
    # 保存引用以便稍后清理
    _shm = shm
    _shm_name = shm.name
    _shm_shape = shape
    
    return shm.name, shape, dtype

def cleanup_shared_memory():
    """清理共享内存资源"""
    global _shm
    
    if _shm is not None:
        try:
            logger.info(f"正在清理共享内存: {_shm.name}")
            _shm.close()
            _shm.unlink()
            _shm = None
        except Exception as e:
            logger.error(f"清理共享内存失败: {e}")

def init_worker(shm_name, shape, dtype):
    """初始化工作进程，连接到共享内存
    
    Args:
        shm_name: 共享内存段的名称
        shape: 图像数组的形状
        dtype: 数据类型
    """
    global _g_img_array, _shm_name, _shm_shape, _shm_dtype
    
    try:
        # 保存共享内存信息
        _shm_name = shm_name
        _shm_shape = shape
        _shm_dtype = dtype
        
        # 连接到现有共享内存
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        
        # 创建NumPy数组，它使用共享内存作为底层缓冲区
        _g_img_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
        logger.debug(f"工作进程连接到共享内存: {shm_name}, 形状={shape}")
    except Exception as e:
        logger.error(f"工作进程共享内存初始化失败: {e}")

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

def _process_frame_optimized_shm(args):
    """使用共享内存的极度优化帧处理函数"""
    global _g_img_array
    
    # 如果_g_img_array为None，尝试重新连接到共享内存
    if _g_img_array is None and _shm_name is not None:
        try:
            existing_shm = shared_memory.SharedMemory(name=_shm_name)
            _g_img_array = np.ndarray(_shm_shape, dtype=_shm_dtype, buffer=existing_shm.buf)
        except Exception as e:
            logger.error(f"重新连接共享内存失败: {e}")
            return args[0], None  # 返回帧索引和None表示处理失败
    
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
    
    # 确保返回内存连续的帧数据
    return frame_idx, frame

# 尝试添加JIT编译支持（如果可用）
try:
    import numba
    if hasattr(numba, 'jit'):
        logger.info("启用Numba JIT加速")
        
        @numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
        def _blend_images_fast(source, target, alpha):
            """使用JIT编译加速的图像混合函数"""
            height, width = source.shape[:2]
            result = np.empty((height, width, 3), dtype=np.uint8)
            
            for y in numba.prange(height):
                for x in range(width):
                    a = alpha[y, x, 0] / 255.0
                    for c in range(3):
                        result[y, x, c] = int(source[y, x, c] * a + target[y, x, c] * (1.0 - a))
            
            return result
        
        # 创建一个使用Numba优化的帧处理函数
        def _process_frame_jit(args):
            """使用JIT编译加速的帧处理函数"""
            global _g_img_array
            
            frame_idx, img_start_y, img_height, img_width, self_height, self_width, frame_positions, is_transparent, bg_color = args
            
            # 快速路径和边界检查
            img_end_y = min(img_height, img_start_y + self_height)
            frame_start_y = 0
            frame_end_y = img_end_y - img_start_y
            
            if img_start_y >= img_end_y or frame_start_y >= frame_end_y or frame_end_y > self_height or _g_img_array is None:
                if is_transparent:
                    frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
                else:
                    frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
                return frame_idx, frame
            
            # 创建帧缓冲区
            if is_transparent:
                frame = np.zeros((self_height, self_width, 4), dtype=np.uint8)
            else:
                frame = np.ones((self_height, self_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
            
            # 计算切片
            img_h_slice = slice(img_start_y, img_end_y)
            img_w_slice = slice(0, min(self_width, img_width))
            frame_h_slice = slice(frame_start_y, frame_end_y)
            frame_w_slice = slice(0, min(self_width, img_width))
            
            try:
                source_section = _g_img_array[img_h_slice, img_w_slice]
                target_area = frame[frame_h_slice, frame_w_slice]
                
                if target_area.shape[:2] == source_section.shape[:2]:
                    if is_transparent:
                        target_area[:] = source_section
                    else:
                        # 使用JIT编译的快速混合函数
                        blended = _blend_images_fast(
                            source_section[:, :, :3], 
                            target_area, 
                            source_section[:, :, 3:4]
                        )
                        target_area[:] = blended
                else:
                    # 处理形状不匹配的情况
                    copy_height = min(target_area.shape[0], source_section.shape[0])
                    copy_width = min(target_area.shape[1], source_section.shape[1])
                    
                    if copy_height > 0 and copy_width > 0:
                        src_crop = source_section[:copy_height, :copy_width]
                        dst_crop = target_area[:copy_height, :copy_width]
                        
                        if is_transparent:
                            target_area[:copy_height, :copy_width] = src_crop
                        else:
                            # 使用JIT编译的快速混合函数
                            blended = _blend_images_fast(
                                src_crop[:, :, :3], 
                                dst_crop, 
                                src_crop[:, :, 3:4]
                            )
                            target_area[:copy_height, :copy_width] = blended
            except Exception as e:
                logger.error(f"JIT优化帧处理出错 {frame_idx}: {e}")
            
            return frame_idx, frame
        
        # 用优化的JIT函数替换原函数
        _process_frame_original = _process_frame
        _process_frame = _process_frame_jit
        logger.info("已启用JIT加速帧处理函数")
    else:
        logger.info("Numba可用但JIT不可用，使用标准优化")
except ImportError:
    logger.info("Numba未安装，使用标准优化")

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