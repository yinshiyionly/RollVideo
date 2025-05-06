"""工具函数模块"""

import os
import logging
import numpy as np

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

def limit_resources():
    """尝试限制进程资源使用"""
    if HAS_RESOURCE_MODULE:
        try:
            # 设置内存限制 (30GB)
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            memory_limit = 30 * 1024 * 1024 * 1024  # 30GB in bytes
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))
            
            # 设置CPU时间限制 (限制单CPU使用)
            cpu_time_limit = 24 * 60 * 60  # 24小时（非常宽松的限制）
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, hard))
            
            logger.info(f"已设置资源限制: 内存={memory_limit/(1024*1024*1024):.1f}GB, CPU时间={cpu_time_limit/3600:.1f}小时")
        except Exception as e:
            logger.warning(f"设置资源限制失败: {e}")
    else:
        logger.info("由于缺少resource模块，资源限制将通过其他方式实现")

# 在导入时尝试设置NumPy性能
setup_numpy_performance()

# 在导入时尝试设置资源限制
try:
    limit_resources()
except Exception as e:
    logger.warning(f"尝试限制资源时发生错误: {e}") 