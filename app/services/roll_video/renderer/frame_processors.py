# renderer/frame_processors.py

import numpy as np
import logging
import ctypes
import multiprocessing.shared_memory as sm

logger = logging.getLogger(__name__)

# 全局共享的图像数组视图（在每个子进程中创建，连接到主进程共享的内存）
_g_img_shm_view = None

# 用于 Numba JIT 加速
try:
    import numba

    if hasattr(numba, "jit"):
        logger.info("启用Numba JIT加速 (尝试)")

        @numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
        def _blend_images_fast(source, target, alpha):
            """使用JIT编译加速的图像混合函数 for RGB"""
            height, width = source.shape[:2]
            # result = np.empty((height, width, 3), dtype=np.uint8) # Numba doesn't like empty
            # Manual blend into target directly
            for y in numba.prange(height):
                for x in range(width):
                    a = alpha[y, x, 0] / 255.0
                    a_inv = 1.0 - a
                    for c in range(3):
                        target[y, x, c] = int(
                            source[y, x, c] * a + target[y, x, c] * a_inv
                        )

        # 修正 JIT 应用逻辑，让 JIT 版本处理不透明混合
        def _blend_opaque_optimized(source_section, target_area):
            """Wrapper to use JIT blend for opaque"""
            if target_area.shape[:2] == source_section.shape[:2]:
                _blend_images_fast(
                    source_section[:, :, :3], target_area, source_section[:, :, 3:4]
                )
            else:
                # Handle size mismatch for blend
                copy_height = min(target_area.shape[0], source_section.shape[0])
                copy_width = min(target_area.shape[1], source_section.shape[1])
                if copy_height > 0 and copy_width > 0:
                    src_crop = np.ascontiguousarray(
                        source_section[:copy_height, :copy_width]
                    )  # Ensure contiguous for Numba
                    dst_crop = np.ascontiguousarray(
                        target_area[:copy_height, :copy_width]
                    )
                    _blend_images_fast(
                        src_crop[:, :, :3], dst_crop, src_crop[:, :, 3:4]
                    )
                    target_area[:copy_height, :copy_width] = (
                        dst_crop  # Copy blended result back
                    )

        logger.info("Numba JIT blending function defined.")

    else:
        _blend_images_fast = None
        _blend_opaque_optimized = None
        logger.info("Numba available but JIT not, or JIT blend not defined.")
except ImportError:
    _blend_images_fast = None
    _blend_opaque_optimized = None
    logger.info("Numba not installed.")


def init_worker(shm_name, shm_shape, shm_dtype):
    """
    进程池 worker 初始化函数，连接共享内存
    """
    global _g_img_shm_view
    existing_shm = sm.SharedMemory(name=shm_name)
    # 使用缓冲区的全部字节创建 NumPy 数组，然后 reshape 到原始图像形状
    _g_img_shm_view = np.ndarray(shm_shape, dtype=shm_dtype, buffer=existing_shm.buf)
    logger.debug(f"Worker {mp.current_process().pid} connected to shared memory.")


def _process_frame_optimized_shm(args):
    """
    优化后的帧处理函数，使用共享内存访问原始图像数据
    参数中移除 img_array 本身
    """
    (
        frame_idx,
        img_start_y,
        img_height,
        img_width,
        self_height,
        self_width,
        is_transparent,
        bg_color,
    ) = args

    # 确保在 worker 中已连接到共享内存
    global _g_img_shm_view
    if _g_img_shm_view is None:
        # Fallback or error - this shouldn't happen if pool initializer works
        logger.error("Worker not initialized with shared memory!")
        if is_transparent:
            frame = np.zeros((self_height, self_width, 4), dtype=np.uint8, order="C")
        else:
            frame = np.ones(
                (self_height, self_width, 3), dtype=np.uint8, order="C"
            ) * np.array(bg_color, dtype=np.uint8)
        return frame_idx, frame  # Return blank frame

    img_array = _g_img_shm_view  # Access image data via shared memory view

    # --- 现有 _process_frame_optimized 的逻辑 ---
    # 快速路径 - 直接计算切片位置
    img_end_y = min(img_height, img_start_y + self_height)
    visible_height = img_end_y - img_start_y

    # 零边界检查 - 极端情况直接返回背景
    if visible_height <= 0:
        if is_transparent:
            frame = np.zeros((self_height, self_width, 4), dtype=np.uint8, order="C")
        else:
            frame = np.ones(
                (self_height, self_width, 3), dtype=np.uint8, order="C"
            ) * np.array(bg_color, dtype=np.uint8)
        return frame_idx, frame

    # 预分配帧缓冲区 - 提高效率，确保内存连续
    if is_transparent:
        frame = np.zeros((self_height, self_width, 4), dtype=np.uint8, order="C")
    else:
        frame = np.ones(
            (self_height, self_width, 3), dtype=np.uint8, order="C"
        ) * np.array(bg_color, dtype=np.uint8)

    # 计算切片 - 超高效版
    source_height = min(visible_height, self_height)
    source_width = min(img_width, self_width)

    # 使用直接视图赋值，避免任何额外复制
    if source_height > 0 and source_width > 0:
        # 获取源图像的当前可见部分切片
        source_section = img_array[
            img_start_y : img_start_y + source_height, :source_width
        ]

        if is_transparent:
            # 透明背景 - 直接复制 RGBA 数据
            frame[:source_height, :source_width] = source_section
        else:
            # 不透明背景 - 需要 Alpha 混合
            # source_section 此时是 RGBA (from img_array)
            # frame 此时是 RGB (with bg_color)
            target_area = frame[:source_height, :source_width]

            # 尝试使用 Numba 加速混合（如果可用）
            if _blend_opaque_optimized is not None:
                try:
                    _blend_opaque_optimized(source_section, target_area)
                except Exception as e:
                    logger.warning(
                        f"Numba blend failed for frame {frame_idx}: {e}, falling back to NumPy"
                    )
                    # Fallback to NumPy if Numba fails
                    alpha = source_section[:, :, 3:4].astype(np.float32) / 255.0
                    alpha_inv = 1.0 - alpha
                    # Blend source RGB onto target RGB
                    np.multiply(
                        source_section[:, :, :3],
                        alpha,
                        out=target_area,
                        casting="unsafe",
                    )  # Blend foreground onto target
                    np.add(
                        target_area,
                        frame[:source_height, :source_width] * alpha_inv,
                        out=target_area,
                        casting="unsafe",
                    )  # Blend background
            else:
                # 使用 NumPy 向量化混合
                alpha = source_section[:, :, 3:4].astype(np.float32) / 255.0
                alpha_inv = 1.0 - alpha
                # Blend source RGB onto target RGB
                np.multiply(
                    source_section[:, :, :3], alpha, out=target_area, casting="unsafe"
                )  # Blend foreground onto target
                np.add(
                    target_area,
                    frame[:source_height, :source_width] * alpha_inv,
                    out=target_area,
                    casting="unsafe",
                )  # Blend background

    # 确保返回内存连续的帧数据 - 对于管道通信至关重要
    # 由于我们在创建时指定 order='C'，这里通常不需要额外的 copy，但为安全可以保留 ascontiguousarray
    return frame_idx, np.ascontiguousarray(frame)  # 返回帧数据


# 将进程池要使用的函数指向这个新的 SHM 版本
_process_frame_to_use = _process_frame_optimized_shm

# 可以移除旧的 _process_frame 和 _process_frame_optimized 函数，或者保留但不再使用。
# 重要的是确保进程池调用的是 _process_frame_optimized_shm
