import os
import logging
from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import oss2
from app.config import settings
from app.utils.logger import Logger

# 配置日志记录器
logger = Logger('oss-client')


class OSSClient:
    def __init__(self):
        """初始化 OSS 客户端

        从应用设置中加载所需配置，初始化阿里云 OSS 客户端
        """
        self.access_key = settings.OSS_ACCESS_KEY
        self.secret_key = settings.OSS_SECRET_KEY
        self.endpoint = settings.OSS_ENDPOINT
        self.bucket_name = settings.OSS_BUCKET

        if not all([self.access_key, self.secret_key, self.endpoint, self.bucket_name]):
            raise ValueError("OSS配置信息不完整，请检查环境变量")

        try:
            # 创建 Auth 对象
            self.auth = oss2.Auth(self.access_key, self.secret_key)
            # 创建 Bucket 对象
            self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)
            logger.info("OSS客户端初始化成功")
        except Exception as e:
            logger.error(f"OSS客户端初始化失败: {str(e)}")
            raise

    def upload_file(
            self,
            local_file_path: Union[str, Path],
            object_key: str,
            metadata: Optional[dict] = None,
    ) -> dict:
        """上传文件到 OSS

        Args:
            local_file_path: 本地文件路径
            object_key: 对象存储中的文件路径
            metadata: 用户自定义元数据

        Returns:
            dict: 上传结果，包含状态码、ETag等信息

        Raises:
            FileNotFoundError: 本地文件不存在
            oss2.exceptions.OssError: OSS 操作错误
            Exception: 其他未知错误
        """
        try:
            # 标准化文件路径
            local_file_path = Path(local_file_path)

            # 检查文件是否存在
            if not local_file_path.exists():
                raise FileNotFoundError(f"文件不存在: {local_file_path}")

            # 获取文件大小
            file_size = local_file_path.stat().st_size

            # 对于大文件使用分片上传
            if file_size > 100 * 1024 * 1024:  # 大于 100MB 使用分片上传
                return self._multipart_upload(
                    str(local_file_path), object_key, metadata
                )

            # 小文件直接上传
            result = self.bucket.put_object_from_file(
                object_key,
                str(local_file_path)
            )

            logger.info(f"文件上传成功: {object_key}, 大小: {file_size} 字节")
            return {
                "status": "success",
                "etag": result.etag,
                "request_id": result.request_id,
                "object_key": object_key,
            }

        except FileNotFoundError as e:
            err = f"文件不存在: {str(e)}"
            logger.error(err)
            raise Exception(err)
        except oss2.exceptions.OssError as e:
            err = f"OSS操作错误: {e.code}, {e.message}, 请求ID: {e.request_id}"
            logger.error(err)
            raise Exception(err)
        except Exception as e:
            err = f"文件上传失败-未知错误: {str(e)}"
            logger.error(err)
            raise Exception(err)

    def _multipart_upload(
            self,
            local_file_path: str,
            object_key: str,
            metadata: Optional[dict] = None,
    ) -> dict:
        """分片上传大文件

        Args:
            local_file_path: 本地文件路径
            object_key: 对象存储中的文件路径
            metadata: 用户自定义元数据

        Returns:
            dict: 上传结果

        Raises:
            Exception: 上传过程中的任何错误
        """
        # 分片大小，单位为字节，最小100KB
        part_size = 10 * 1024 * 1024  # 10MB
        upload_id = None

        try:
            # 输出当前使用的OSS凭证信息（部分隐藏）
            logger.info(f"使用的AccessKey前缀: {self.access_key[:4]}****")
            logger.info(f"使用的Endpoint: {self.endpoint}")
            logger.info(f"使用的Bucket: {self.bucket_name}")

            # 配置头信息
            headers = {}
            if metadata:
                for k, v in metadata.items():
                    headers[f'x-oss-meta-{k}'] = str(v)

            # 确保所有分片请求使用相同的初始化参数
            self.auth = oss2.Auth(self.access_key, self.secret_key)
            self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)

            # 初始化分片上传任务
            upload_id = self.bucket.init_multipart_upload(object_key, headers=headers).upload_id
            logger.info(f"初始化分片上传任务: {upload_id}")

            # 计算分片数量
            file_size = os.path.getsize(local_file_path)
            parts_count = (file_size + part_size - 1) // part_size

            # 上传分片
            parts = []
            for i in range(parts_count):
                part_number = i + 1
                start = i * part_size
                end = min(start + part_size, file_size)
                size = end - start

                # 分片上传，支持断点续传
                with open(local_file_path, 'rb') as f:
                    f.seek(start)
                    # 使用同一个 bucket 对象来确保签名计算一致
                    result = self.bucket.upload_part(
                        object_key,
                        upload_id,
                        part_number,
                        f.read(size)
                    )

                parts.append(oss2.models.PartInfo(part_number, result.etag))
                logger.info(f"分片 {part_number}/{parts_count} 上传完成")

            # 完成分片上传
            result = self.bucket.complete_multipart_upload(object_key, upload_id, parts)

            logger.info(f"分片上传完成: {object_key}, 大小: {file_size} 字节")
            return {
                "status": "success",
                "etag": result.etag,
                "request_id": result.request_id,
                "object_key": object_key,
            }

        except oss2.exceptions.ServerError as e:
            logger.error(f"OSS服务器错误: {e.status}, {e.code}, {e.message}, 请求ID: {e.request_id}")
            # 签名错误特殊处理
            if e.code == 'SignatureDoesNotMatch':
                logger.error(f"签名不匹配错误: {e.message}")
                logger.error(f"请检查AccessKey和SecretKey是否正确，以及签名计算方法")

            # 清理未完成的分片上传
            if upload_id:
                try:
                    self.bucket.abort_multipart_upload(object_key, upload_id)
                    logger.info(f"已清理未完成的分片上传: {upload_id}")
                except Exception as abort_e:
                    logger.error(f"清理未完成的分片上传失败: {str(abort_e)}")

            raise
        except Exception as e:
            logger.error(f"分片上传失败: {str(e)}")

            # 清理未完成的分片上传
            if upload_id:
                try:
                    self.bucket.abort_multipart_upload(object_key, upload_id)
                    logger.info(f"已清理未完成的分片上传: {upload_id}")
                except Exception as abort_e:
                    logger.error(f"清理未完成的分片上传失败: {str(abort_e)}")

            raise

    def upload_with_retry(
            self,
            local_file_path: Union[str, Path],
            object_key: str,
            metadata: Optional[dict] = None,
            max_retries: int = 3
    ) -> dict:
        """带重试机制的文件上传

        Args:
            local_file_path: 本地文件路径
            object_key: 对象存储中的文件路径
            metadata: 用户自定义元数据
            max_retries: 最大重试次数

        Returns:
            dict: 上传结果

        Raises:
            Exception: 所有重试都失败后抛出最后一次异常
        """
        last_error = None
        for retry in range(max_retries):
            try:
                # 每次重试前重新初始化客户端连接
                if retry > 0:
                    logger.info("重新初始化OSS客户端连接")
                    self.auth = oss2.Auth(self.access_key, self.secret_key)
                    self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)

                result = self.upload_file(local_file_path, object_key, metadata)
                logger.info(f"上传成功(尝试 {retry + 1}/{max_retries}): {object_key}")
                return result
            except oss2.exceptions.ServerError as e:
                last_error = e
                # 签名错误特殊处理
                if hasattr(e, 'code') and e.code == 'SignatureDoesNotMatch':
                    logger.warning(f"签名不匹配错误(尝试 {retry + 1}/{max_retries}): {e.message}")
                    # 可能是时间同步问题，或者密钥问题
                    if retry < max_retries - 1:
                        # 使用更长的等待时间
                        import time
                        wait_time = (2 ** retry) * 3
                        logger.info(f"将在 {wait_time} 秒后重试上传，重新初始化连接")
                        time.sleep(wait_time)
                else:
                    logger.warning(f"服务器错误(尝试 {retry + 1}/{max_retries}): {e.status}, {e.code}, {e.message}")
                    if retry < max_retries - 1:
                        import time
                        wait_time = (2 ** retry) * 2
                        logger.info(f"将在 {wait_time} 秒后重试上传")
                        time.sleep(wait_time)
            except Exception as e:
                last_error = e
                logger.warning(f"上传失败(尝试 {retry + 1}/{max_retries}): {str(e)}")
                # 如果不是最后一次尝试，则等待后重试
                if retry < max_retries - 1:
                    import time
                    # 指数退避策略，每次重试等待时间增加
                    wait_time = (2 ** retry) * 2
                    logger.info(f"将在 {wait_time} 秒后重试上传")
                    time.sleep(wait_time)

        # 所有重试都失败
        logger.error(f"上传失败(已尝试 {max_retries} 次): {str(last_error)}")
        raise last_error

    def check_file_exists(self, object_key: str) -> bool:
        """检查文件是否存在于 OSS

        Args:
            object_key: 对象存储中的文件路径

        Returns:
            bool: 文件是否存在
        """
        try:
            self.bucket.head_object(object_key)
            return True
        except oss2.exceptions.NoSuchKey:
            return False
        except Exception as e:
            logger.error(f"检查文件存在时发生错误: {str(e)}")
            return False

    def verify_credentials(self) -> dict:
        """验证OSS凭证是否有效

        尝试列出Bucket中的对象来验证凭证是否有效

        Returns:
            dict: 验证结果，包含状态和消息
        """
        try:
            # 创建新的连接进行验证
            auth = oss2.Auth(self.access_key, self.secret_key)
            bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

            # 尝试列出Bucket中的对象（最多1个）
            result = bucket.list_objects(max_keys=1)

            # 检查请求是否成功
            return {
                "status": "success",
                "message": f"OSS凭证验证成功，Bucket: {self.bucket_name}，服务器时间: {result.headers.get('date', '未知')}",
                "request_id": result.request_id
            }
        except oss2.exceptions.ServerError as e:
            # 服务器错误，如签名不匹配
            error_message = f"OSS凭证验证失败: {e.status}, {e.code}, {e.message}"
            logger.error(error_message)
            if e.code == 'SignatureDoesNotMatch':
                # 提供更详细的错误信息
                return {
                    "status": "error",
                    "message": f"{error_message}. 请检查: 1.AccessKey和SecretKey是否正确; 2.服务器时间是否同步; 3.Endpoint是否匹配Bucket区域",
                    "error_code": e.code,
                    "request_id": e.request_id
                }
            return {
                "status": "error",
                "message": error_message,
                "error_code": e.code,
                "request_id": e.request_id
            }
        except Exception as e:
            error_message = f"OSS凭证验证失败-未知错误: {str(e)}"
            logger.error(error_message)
            return {
                "status": "error",
                "message": error_message
            }

    def delete_file(self, object_key: str) -> bool:
        """删除 OSS 中的文件

        Args:
            object_key: 对象存储中的文件路径

        Returns:
            bool: 是否删除成功
        """
        try:
            self.bucket.delete_object(object_key)
            logger.info(f"文件删除成功: {object_key}")
            return True
        except Exception as e:
            logger.error(f"删除文件失败: {str(e)}")
            return False
