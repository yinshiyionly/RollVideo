"""工具函数模块"""

import os
import sys
import logging
import resource
import psutil
import gc
import traceback
import time
from contextlib import contextmanager

# 尝试导入资源模块
try:
    import resource
    HAS_RESOURCE_MODULE = True
except ImportError:
    HAS_RESOURCE_MODULE = False

# 尝试导入mmap模块
try:
    import mmap
    HAS_MMAP = True
except ImportError:
    HAS_MMAP = False

logger = logging.getLogger(__name__)

# 设置NumPy环境变量以提高性能
def setup_numpy_performance():
    """配置NumPy高性能环境变量"""
    try:
        # 针对性能优化的环境变量
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = '8'  # 使用全部8核心
        if 'MKL_NUM_THREADS' not in os.environ:
            os.environ['MKL_NUM_THREADS'] = '8'  # 使用全部8核心
        if 'NUMEXPR_NUM_THREADS' not in os.environ:
            os.environ['NUMEXPR_NUM_THREADS'] = '8'  # 使用全部8核心
        if 'OPENBLAS_NUM_THREADS' not in os.environ:
            os.environ['OPENBLAS_NUM_THREADS'] = '8'  # 使用全部8核心
        
        # 高级线程优化 - 如果系统支持则启用
        try:
            os.environ['OMP_WAIT_POLICY'] = 'ACTIVE'  # 主动等待，减少线程唤醒延迟
            os.environ['OMP_PROC_BIND'] = 'spread'    # 优化线程绑定
            os.environ['OMP_SCHEDULE'] = 'dynamic,16' # 动态调度，减少线程不平衡
            
            # 尝试开启NumPy高级优化
            np.seterr(all='ignore')                   # 忽略NumPy警告提高性能
            np.set_printoptions(precision=3, suppress=True)
        except:
            pass
            
        logger.info(f"已设置NumPy高性能模式: OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}")
    except Exception as e:
        logger.warning(f"设置NumPy线程优化失败: {e}")

def limit_resources(mem_limit_gb=None, cpu_limit=None):
    """
    限制进程的系统资源使用
    
    Args:
        mem_limit_gb: 内存限制 (GB)
        cpu_limit: CPU限制（核心数）
    """
    try:
        if mem_limit_gb:
            # 尝试设置内存限制 (bytes)
            mem_limit_bytes = int(mem_limit_gb * 1024 * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
            logger.info(f"设置内存限制: {mem_limit_gb}GB")
            
        if cpu_limit:
            # 尝试限制CPU使用
            p = psutil.Process()
            if hasattr(p, 'cpu_affinity'):
                try:
                    # 获取所有可用CPU
                    all_cpus = list(range(psutil.cpu_count()))
                    # 限制使用的CPU数量
                    limited_cpus = all_cpus[:cpu_limit]
                    p.cpu_affinity(limited_cpus)
                    logger.info(f"限制CPU使用: {cpu_limit}核")
                except:
                    logger.warning("无法设置CPU亲和性")
    except Exception as e:
        logger.warning(f"设置资源限制失败: {str(e)}")

def optimize_memory():
    """
    优化内存使用，减少内存碎片
    """
    # 强制垃圾回收
    gc.collect()
    
    # 尝试减少内存碎片
    try:
        if hasattr(gc, 'set_threshold'):
            old_threshold = gc.get_threshold()
            # 设置较高的阈值，减少GC频率
            gc.set_threshold(25000, 10, 10)
            logger.info(f"优化GC阈值: {old_threshold} -> {gc.get_threshold()}")
    except Exception as e:
        logger.warning(f"优化内存使用失败: {str(e)}")

def get_memory_usage():
    """
    获取当前进程的内存使用情况
    
    Returns:
        Dict: 内存使用信息
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # 物理内存使用
            "vms_mb": memory_info.vms / (1024 * 1024),  # 虚拟内存使用
            "percent": process.memory_percent(),  # 使用百分比
            "system_available_gb": virtual_memory.available / (1024 * 1024 * 1024),  # 系统可用内存
        }
    except Exception as e:
        logger.error(f"获取内存使用信息失败: {str(e)}")
        return {"error": str(e)}

@contextmanager
def time_tracker(description="操作"):
    """
    跟踪代码块执行时间的上下文管理器
    
    Args:
        description: 操作描述
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"{description}耗时: {elapsed:.2f}秒")

def setup_logging(level=logging.INFO, log_file=None):
    """
    设置日志
    
    Args:
        level: 日志级别
        log_file: 日志文件路径
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 创建文件处理器（如果提供了日志文件路径）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"日志已设置，级别: {logging.getLevelName(level)}")
    if log_file:
        logger.info(f"日志文件: {log_file}")

def emergency_cleanup():
    """紧急清理进程资源，用于异常情况"""
    # 强制清理残留的ffmpeg进程
    try:
        process = psutil.Process()
        for child in process.children(recursive=True):
            if 'ffmpeg' in child.name().lower():
                logger.warning(f"强制终止残留的FFmpeg进程: {child.pid}")
                try:
                    child.terminate()
                    child.wait(timeout=1)
                    if child.is_running():
                        child.kill()
                except:
                    pass
    except:
        pass
    
    # 强制垃圾回收
    gc.collect()
    
    # 输出调试信息
    try:
        mem_info = get_memory_usage()
        logger.info(f"紧急清理后内存使用: {mem_info['rss_mb']:.2f}MB")
    except:
        pass

# 在导入时尝试设置NumPy性能
setup_numpy_performance()

# 在导入时尝试设置资源限制
try:
    limit_resources()
except Exception as e:
    logger.warning(f"尝试限制资源时发生错误: {e}") 