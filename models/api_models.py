"""API响应模型定义

本模块定义了统一的API响应格式、错误处理和分页结构。
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic, Union, Callable, Literal, Set
from datetime import datetime, timezone, timedelta
from enum import Enum
from pydantic import BaseModel, Field, validator, model_validator, constr, conint, confloat
from pydantic.config import ConfigDict
import uuid
import time
from decimal import Decimal
import json
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 泛型类型变量
T = TypeVar('T')

# 性能优化配置
@dataclass
class SerializationConfig:
    """序列化配置"""
    enable_cache: bool = True
    cache_size: int = 1000
    enable_compression: bool = False
    max_workers: int = 4
    async_threshold: int = 100  # 超过此数量的列表使用异步处理

# 全局序列化配置
default_serialization_config = SerializationConfig()

# 序列化缓存
@lru_cache(maxsize=1000)
def _cached_model_dump(model_hash: str, model_data: str) -> str:
    """缓存的模型序列化"""
    return model_data

class SerializationMixin(ABC):
    """序列化混入类，提供性能优化的序列化方法"""
    
    def model_dump_optimized(self, 
                           config: Optional[SerializationConfig] = None,
                           **kwargs) -> Dict[str, Any]:
        """优化的模型序列化"""
        config = config or default_serialization_config
        
        if config.enable_cache:
            # 生成模型哈希用于缓存
            model_str = str(self.__dict__)
            model_hash = hashlib.md5(model_str.encode()).hexdigest()
            
            try:
                cached_result = _cached_model_dump(model_hash, model_str)
                if cached_result:
                    return json.loads(cached_result)
            except:
                pass
        
        # 标准序列化
        result = self.model_dump(**kwargs)
        
        if config.enable_cache:
            try:
                _cached_model_dump(model_hash, json.dumps(result, default=str))
            except:
                pass
        
        return result
    
    @classmethod
    async def model_validate_async(cls, data: Dict[str, Any]):
        """异步模型验证"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, cls.model_validate, data)
    
    @classmethod
    def model_validate_batch(cls, 
                           data_list: List[Dict[str, Any]], 
                           config: Optional[SerializationConfig] = None) -> List['SerializationMixin']:
        """批量模型验证"""
        config = config or default_serialization_config
        
        if len(data_list) < config.async_threshold:
            return [cls.model_validate(data) for data in data_list]
        
        # 使用线程池进行并行处理
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [executor.submit(cls.model_validate, data) for data in data_list]
            return [future.result() for future in futures]
    
    @classmethod
    async def model_validate_batch_async(cls, 
                                       data_list: List[Dict[str, Any]], 
                                       config: Optional[SerializationConfig] = None) -> List['SerializationMixin']:
        """异步批量模型验证"""
        config = config or default_serialization_config
        
        tasks = [cls.model_validate_async(data) for data in data_list]
        return await asyncio.gather(*tasks)


class ResponseStatus(str, Enum):
    """响应状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"
    PENDING = "pending"
    PROCESSING = "processing"
    TIMEOUT = "timeout"


class ErrorCode(str, Enum):
    """错误代码枚举"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    CONFLICT_ERROR = "CONFLICT_ERROR"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"


class ErrorDetail(BaseModel):
    """错误详情模型"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    code: ErrorCode = Field(..., description="错误代码")
    message: constr(min_length=1, max_length=1000) = Field(..., description="错误消息")
    field: Optional[constr(min_length=1, max_length=100)] = Field(None, description="相关字段")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="错误时间")
    trace_id: Optional[str] = Field(None, description="追踪ID")
    
    @validator('details')
    def validate_details(cls, v):
        if v is not None and len(str(v)) > 5000:
            raise ValueError("错误详情过长")
        return v


class PaginationInfo(BaseModel):
    """分页信息模型"""
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        frozen=True  # 分页信息不可变
    )
    
    page: conint(ge=1) = Field(..., description="当前页码")
    page_size: conint(ge=1, le=1000) = Field(..., description="每页大小")
    total: conint(ge=0) = Field(..., description="总记录数")
    total_pages: conint(ge=0) = Field(..., description="总页数")
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")
    offset: conint(ge=0) = Field(..., description="偏移量")
    next_cursor: Optional[str] = Field(None, description="下一页游标")
    prev_cursor: Optional[str] = Field(None, description="上一页游标")
    
    @model_validator(mode='before')
    @classmethod
    def validate_pagination(cls, values):
        page = values.get('page', 1)
        page_size = values.get('page_size', 10)
        total = values.get('total', 0)
        
        # 计算总页数
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        values['total_pages'] = total_pages
        
        # 计算偏移量
        values['offset'] = (page - 1) * page_size
        
        # 计算是否有上下页
        values['has_next'] = page < total_pages
        values['has_prev'] = page > 1
        
        return values
    
    @classmethod
    def create(cls, page: int, page_size: int, total: int, 
               next_cursor: Optional[str] = None, 
               prev_cursor: Optional[str] = None) -> "PaginationInfo":
        """创建分页信息"""
        return cls(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=0,  # 将由validator计算
            has_next=False,  # 将由validator计算
            has_prev=False,  # 将由validator计算
            offset=0,  # 将由validator计算
            next_cursor=next_cursor,
            prev_cursor=prev_cursor
        )


class BaseResponse(BaseModel, SerializationMixin, Generic[T]):
    """基础响应模型"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        # 性能优化配置
        validate_default=True,
        arbitrary_types_allowed=False
    )
    
    status: ResponseStatus = Field(..., description="响应状态")
    message: constr(min_length=1, max_length=500) = Field(..., description="响应消息")
    data: Optional[T] = Field(None, description="响应数据")
    errors: List[ErrorDetail] = Field(default_factory=list, description="错误列表")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="请求ID")
    processing_time: Optional[confloat(ge=0)] = Field(None, description="处理时间（毫秒）")
    server_time: Optional[datetime] = Field(default_factory=datetime.utcnow, description="服务器时间")
    version: Optional[str] = Field(None, description="API版本")
    
    @validator('errors')
    def validate_errors(cls, v):
        if len(v) > 100:  # 限制错误数量
            raise ValueError("错误数量过多")
        return v
    
    @model_validator(mode='before')
    @classmethod
    def validate_response(cls, values):
        status = values.get('status')
        errors = values.get('errors', [])
        data = values.get('data')
        
        # 状态一致性验证
        if status == ResponseStatus.ERROR and not errors:
            raise ValueError("错误状态必须包含错误信息")
        if status == ResponseStatus.SUCCESS and errors:
            raise ValueError("成功状态不应包含错误信息")
        
        return values
    
    @classmethod
    def success(cls, data: Optional[T] = None, message: str = "操作成功", 
                processing_time: Optional[float] = None, 
                version: Optional[str] = None) -> "BaseResponse[T]":
        """创建成功响应"""
        return cls(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=data,
            processing_time=processing_time,
            version=version
        )
    
    @classmethod
    def error(cls, message: str, errors: List[ErrorDetail] = None, 
              error_code: Optional[ErrorCode] = None,
              processing_time: Optional[float] = None) -> "BaseResponse[T]":
        """创建错误响应"""
        error_list = errors or []
        if error_code and not error_list:
            error_list = [ErrorDetail(code=error_code, message=message)]
        
        return cls(
            status=ResponseStatus.ERROR,
            message=message,
            errors=error_list,
            processing_time=processing_time
        )
    
    @classmethod
    def warning(cls, data: Optional[T] = None, message: str = "操作完成但有警告", 
                errors: List[ErrorDetail] = None,
                processing_time: Optional[float] = None) -> "BaseResponse[T]":
        """创建警告响应"""
        return cls(
            status=ResponseStatus.WARNING,
            message=message,
            data=data,
            errors=errors or [],
            processing_time=processing_time
        )
    
    @classmethod
    def partial(cls, data: Optional[T] = None, message: str = "部分成功", 
                errors: List[ErrorDetail] = None,
                processing_time: Optional[float] = None) -> "BaseResponse[T]":
        """创建部分成功响应"""
        return cls(
            status=ResponseStatus.PARTIAL,
            message=message,
            data=data,
            errors=errors or [],
            processing_time=processing_time
        )


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应模型"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        validate_default=True
    )
    
    status: ResponseStatus = Field(..., description="响应状态")
    message: constr(min_length=1, max_length=500) = Field(..., description="响应消息")
    data: List[T] = Field(..., description="数据列表")
    pagination: PaginationInfo = Field(..., description="分页信息")
    errors: List[ErrorDetail] = Field(default_factory=list, description="错误列表")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="请求ID")
    processing_time: Optional[confloat(ge=0)] = Field(None, description="处理时间（毫秒）")
    total_count: conint(ge=0) = Field(..., description="总记录数")
    filtered_count: Optional[conint(ge=0)] = Field(None, description="过滤后记录数")
    
    @validator('data')
    def validate_data_count(cls, v, values):
        pagination = values.get('pagination')
        if pagination and len(v) > pagination.page_size:
            raise ValueError(f"数据数量({len(v)})超过页面大小({pagination.page_size})")
        return v
    
    @model_validator(mode='before')
    @classmethod
    def validate_pagination_consistency(cls, values):
        data = values.get('data', [])
        pagination = values.get('pagination')
        total_count = values.get('total_count', 0)
        
        if pagination:
            # 验证总数一致性
            if pagination.total != total_count:
                values['pagination'] = PaginationInfo.create(
                    page=pagination.page,
                    page_size=pagination.page_size,
                    total=total_count,
                    next_cursor=pagination.next_cursor,
                    prev_cursor=pagination.prev_cursor
                )
        
        return values
    
    @classmethod
    def success(
        cls,
        data: List[T],
        pagination: PaginationInfo,
        message: str = "查询成功",
        request_id: Optional[str] = None
    ) -> "PaginatedResponse[T]":
        """创建成功的分页响应"""
        return cls(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=data,
            pagination=pagination,
            request_id=request_id
        )


class ServiceStatus(str, Enum):
    """服务状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    STARTING = "starting"
    STOPPING = "stopping"


class DependencyStatus(BaseModel):
    """依赖服务状态模型"""
    model_config = ConfigDict(extra="forbid")
    
    name: constr(min_length=1, max_length=100) = Field(..., description="依赖名称")
    status: ServiceStatus = Field(..., description="依赖状态")
    response_time: Optional[confloat(ge=0)] = Field(None, description="响应时间（毫秒）")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="最后检查时间")
    error_message: Optional[constr(max_length=500)] = Field(None, description="错误信息")
    version: Optional[str] = Field(None, description="依赖版本")
    endpoint: Optional[str] = Field(None, description="检查端点")


class SystemMetrics(BaseModel):
    """系统指标模型"""
    model_config = ConfigDict(extra="forbid")
    
    cpu_usage: Optional[confloat(ge=0, le=100)] = Field(None, description="CPU使用率(%)")
    memory_usage: Optional[confloat(ge=0, le=100)] = Field(None, description="内存使用率(%)")
    disk_usage: Optional[confloat(ge=0, le=100)] = Field(None, description="磁盘使用率(%)")
    active_connections: Optional[conint(ge=0)] = Field(None, description="活跃连接数")
    request_rate: Optional[confloat(ge=0)] = Field(None, description="请求速率(req/s)")
    error_rate: Optional[confloat(ge=0, le=100)] = Field(None, description="错误率(%)")
    avg_response_time: Optional[confloat(ge=0)] = Field(None, description="平均响应时间（毫秒）")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    status: ServiceStatus = Field(..., description="服务状态")
    version: constr(min_length=1, max_length=50) = Field(..., description="服务版本")
    uptime: confloat(ge=0) = Field(..., description="运行时间（秒）")
    dependencies: List[DependencyStatus] = Field(default_factory=list, description="依赖服务状态")
    system_metrics: Optional[SystemMetrics] = Field(None, description="系统指标")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="检查时间")
    environment: Optional[constr(max_length=50)] = Field(None, description="运行环境")
    build_info: Optional[Dict[str, str]] = Field(None, description="构建信息")
    
    @validator('dependencies')
    def validate_dependencies(cls, v):
        if len(v) > 50:  # 限制依赖数量
            raise ValueError("依赖服务数量过多")
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def determine_overall_status(cls, values):
        dependencies = values.get('dependencies', [])
        current_status = values.get('status')
        
        # 如果有依赖服务不健康，降级整体状态
        if dependencies:
            unhealthy_deps = [dep for dep in dependencies if dep.status == ServiceStatus.UNHEALTHY]
            degraded_deps = [dep for dep in dependencies if dep.status == ServiceStatus.DEGRADED]
            
            if unhealthy_deps and current_status == ServiceStatus.HEALTHY:
                values['status'] = ServiceStatus.DEGRADED
            elif len(unhealthy_deps) > len(dependencies) // 2:
                values['status'] = ServiceStatus.UNHEALTHY
        
        return values


class BatchOperationResult(BaseModel):
    """批量操作结果模型"""
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        frozen=True  # 批量操作结果不可变
    )
    
    total: conint(ge=0) = Field(..., description="总数量")
    success: conint(ge=0) = Field(..., description="成功数量")
    failed: conint(ge=0) = Field(..., description="失败数量")
    skipped: conint(ge=0) = Field(..., description="跳过数量")
    errors: List[ErrorDetail] = Field(default_factory=list, description="错误列表")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")
    processing_time: confloat(ge=0) = Field(..., description="处理时间（毫秒）")
    start_time: datetime = Field(..., description="开始时间")
    end_time: datetime = Field(..., description="结束时间")
    success_rate: confloat(ge=0, le=100) = Field(..., description="成功率(%)")
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_counts(cls, values):
        total = values.get('total', 0)
        success = values.get('success', 0)
        failed = values.get('failed', 0)
        skipped = values.get('skipped', 0)
        
        # 验证数量一致性
        if success + failed + skipped != total:
            raise ValueError("成功、失败、跳过数量之和必须等于总数量")
        
        # 计算成功率
        if total > 0:
            values['success_rate'] = (success / total) * 100
        else:
            values['success_rate'] = 0.0
        
        # 验证时间
        start_time = values.get('start_time')
        end_time = values.get('end_time')
        if start_time and end_time and end_time < start_time:
            raise ValueError("结束时间不能早于开始时间")
        
        return values
    
    @validator('errors')
    def validate_errors_count(cls, v, values):
        failed = values.get('failed', 0)
        if len(v) > failed:
            raise ValueError("错误数量不能超过失败数量")
        return v


class FileUploadResponse(BaseModel):
    """文件上传响应模型"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    file_id: constr(min_length=1, max_length=100) = Field(..., description="文件ID")
    filename: constr(min_length=1, max_length=255) = Field(..., description="文件名")
    original_filename: Optional[constr(max_length=255)] = Field(None, description="原始文件名")
    file_size: conint(ge=0, le=5*1024*1024*1024) = Field(..., description="文件大小（字节）")  # 最大5GB
    file_type: constr(min_length=1, max_length=100) = Field(..., description="文件类型")
    upload_url: Optional[str] = Field(None, description="上传URL")
    download_url: Optional[str] = Field(None, description="下载URL")
    preview_url: Optional[str] = Field(None, description="预览URL")
    thumbnail_url: Optional[str] = Field(None, description="缩略图URL")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文件元数据")
    upload_time: datetime = Field(default_factory=datetime.utcnow, description="上传时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    checksum: Optional[constr(max_length=128)] = Field(None, description="文件校验和")
    storage_path: Optional[str] = Field(None, description="存储路径")
    is_public: bool = Field(default=False, description="是否公开")
    
    @validator('filename', 'original_filename')
    def validate_filename(cls, v):
        if v and ('/' in v or '\\' in v or '..' in v):
            raise ValueError("文件名包含非法字符")
        return v
    
    @validator('file_type')
    def validate_content_type(cls, v):
        allowed_types = {
            'image/', 'video/', 'audio/', 'text/', 'application/pdf',
            'application/json', 'application/xml', 'application/zip',
            'application/msword', 'application/vnd.ms-excel'
        }
        if not any(v.startswith(allowed) for allowed in allowed_types):
            raise ValueError(f"不支持的文件类型: {v}")
        return v
    
    @validator('file_size')
    def validate_size(cls, v, values):
        file_type = values.get('file_type', '')
        # 根据文件类型设置不同的大小限制
        if file_type.startswith('image/') and v > 50*1024*1024:  # 50MB for images
            raise ValueError("图片文件大小不能超过50MB")
        elif file_type.startswith('video/') and v > 1024*1024*1024:  # 1GB for videos
            raise ValueError("视频文件大小不能超过1GB")
        return v


class ExportStatus(str, Enum):
    """导出状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ExportFormat(str, Enum):
    """导出格式枚举"""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    XML = "xml"
    PDF = "pdf"
    ZIP = "zip"


class ExportResponse(BaseModel):
    """导出响应模型"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    export_id: constr(min_length=1, max_length=100) = Field(..., description="导出ID")
    status: ExportStatus = Field(..., description="导出状态")
    format: ExportFormat = Field(..., description="导出格式")
    download_url: Optional[str] = Field(None, description="下载链接")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    file_size: Optional[conint(ge=0)] = Field(None, description="文件大小（字节）")
    progress: confloat(ge=0, le=100) = Field(default=0, description="导出进度")
    total_records: Optional[conint(ge=0)] = Field(None, description="总记录数")
    processed_records: Optional[conint(ge=0)] = Field(None, description="已处理记录数")
    error_message: Optional[constr(max_length=500)] = Field(None, description="错误信息")
    estimated_completion: Optional[datetime] = Field(None, description="预计完成时间")
    
    @validator('progress')
    def validate_progress_status(cls, v, values):
        status = values.get('status')
        if status == ExportStatus.COMPLETED and v != 100:
            return 100
        elif status == ExportStatus.FAILED and v > 0:
            return v  # 保持失败时的进度
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_export_data(cls, values):
        status = values.get('status')
        download_url = values.get('download_url')
        error_message = values.get('error_message')
        started_at = values.get('started_at')
        completed_at = values.get('completed_at')
        created_at = values.get('created_at')
        
        # 状态一致性验证
        if status == ExportStatus.COMPLETED and not download_url:
            raise ValueError("完成状态必须提供下载链接")
        if status == ExportStatus.FAILED and not error_message:
            raise ValueError("失败状态必须提供错误信息")
        
        # 时间一致性验证
        if started_at and created_at and started_at < created_at:
            raise ValueError("开始时间不能早于创建时间")
        if completed_at and started_at and completed_at < started_at:
            raise ValueError("完成时间不能早于开始时间")
        
        return values


class StatisticsPeriod(str, Enum):
    """统计周期枚举"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


class StatisticsResponse(BaseModel):
    """统计信息响应模型"""
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        frozen=True  # 统计数据不可变
    )
    
    period: StatisticsPeriod = Field(..., description="统计周期")
    start_date: datetime = Field(..., description="统计开始时间")
    end_date: datetime = Field(..., description="统计结束时间")
    total_users: conint(ge=0) = Field(..., description="总用户数")
    active_users: conint(ge=0) = Field(..., description="活跃用户数")
    new_users: conint(ge=0) = Field(default=0, description="新用户数")
    total_sessions: conint(ge=0) = Field(..., description="总会话数")
    total_messages: conint(ge=0) = Field(..., description="总消息数")
    avg_session_duration: confloat(ge=0) = Field(..., description="平均会话时长（分钟）")
    avg_messages_per_session: confloat(ge=0) = Field(default=0, description="平均每会话消息数")
    user_retention_rate: confloat(ge=0, le=100) = Field(default=0, description="用户留存率(%)")
    peak_concurrent_users: conint(ge=0) = Field(default=0, description="峰值并发用户数")
    total_tokens_used: conint(ge=0) = Field(default=0, description="总令牌使用量")
    total_cost: confloat(ge=0) = Field(default=0, description="总成本")
    metrics: Dict[str, Union[int, float, str]] = Field(default_factory=dict, description="其他指标")
    charts: Optional[Dict[str, List[Dict[str, Any]]]] = Field(None, description="图表数据")
    summary: Optional[str] = Field(None, description="统计摘要")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="生成时间")
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_statistics(cls, values):
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        total_users = values.get('total_users', 0)
        active_users = values.get('active_users', 0)
        total_sessions = values.get('total_sessions', 0)
        total_messages = values.get('total_messages', 0)
        
        # 验证时间范围
        if start_date and end_date and end_date <= start_date:
            raise ValueError("结束时间必须晚于开始时间")
        
        # 验证用户数据一致性
        if active_users > total_users:
            raise ValueError("活跃用户数不能超过总用户数")
        
        # 计算平均每会话消息数
        if total_sessions > 0:
            values['avg_messages_per_session'] = total_messages / total_sessions
        
        return values
    
    @validator('metrics')
    def validate_metrics(cls, v):
        # 限制metrics字典的大小
        if len(v) > 50:
            raise ValueError("指标数量不能超过50个")
        return v


class WebSocketMessageType(str, Enum):
    """WebSocket消息类型枚举"""
    CHAT_MESSAGE = "chat_message"
    SYSTEM_NOTIFICATION = "system_notification"
    USER_STATUS = "user_status"
    SESSION_UPDATE = "session_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    CONNECTION_ACK = "connection_ack"
    TYPING_INDICATOR = "typing_indicator"
    AGENT_STATUS = "agent_status"
    STREAM_CHUNK = "stream_chunk"
    MEMORY_SAVED = "memory_saved"  # 记忆保存事件


class WebSocketMessage(BaseModel):
    """WebSocket消息模型"""
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    type: WebSocketMessageType = Field(..., description="消息类型")
    event: str = Field(..., description="事件名称")
    data: Optional[Dict[str, Any]] = Field(None, description="消息数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    session_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="会话ID")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    request_id: Optional[str] = Field(None, description="请求ID")
    message_id: constr(min_length=1, max_length=100) = Field(..., description="消息ID")
    correlation_id: Optional[constr(max_length=100)] = Field(None, description="关联ID")
    priority: conint(ge=0, le=10) = Field(default=5, description="消息优先级")
    ttl: Optional[conint(ge=1)] = Field(None, description="消息生存时间（秒）")
    retry_count: conint(ge=0, le=5) = Field(default=0, description="重试次数")
    
    @validator('data')
    def validate_data_size(cls, v):
        if v is not None:
            # 限制数据大小，防止内存溢出
            import json
            data_str = json.dumps(v, default=str)
            if len(data_str) > 1024 * 1024:  # 1MB limit
                raise ValueError("消息数据大小不能超过1MB")
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_message_requirements(cls, values):
        msg_type = values.get('type')
        session_id = values.get('session_id')
        user_id = values.get('user_id')
        
        # 某些消息类型需要会话ID
        if msg_type in [WebSocketMessageType.CHAT_MESSAGE, WebSocketMessageType.SESSION_UPDATE] and not session_id:
            raise ValueError(f"消息类型 {msg_type} 需要提供会话ID")
        
        # 某些消息类型需要用户ID
        if msg_type in [WebSocketMessageType.USER_STATUS, WebSocketMessageType.TYPING_INDICATOR] and not user_id:
            raise ValueError(f"消息类型 {msg_type} 需要提供用户ID")
        
        return values


class HTTPMethod(str, Enum):
    """HTTP方法枚举"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class APIMetrics(BaseModel):
    """API指标模型"""
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        frozen=True  # 指标数据不可变
    )
    
    endpoint: constr(min_length=1, max_length=200) = Field(..., description="API端点")
    method: HTTPMethod = Field(..., description="HTTP方法")
    response_time: confloat(ge=0) = Field(..., description="响应时间（毫秒）")
    status_code: conint(ge=100, le=599) = Field(..., description="HTTP状态码")
    request_size: Optional[conint(ge=0)] = Field(None, description="请求大小（字节）")
    response_size: Optional[conint(ge=0)] = Field(None, description="响应大小（字节）")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    session_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="会话ID")
    ip_address: Optional[str] = Field(None, description="客户端IP地址")
    user_agent: Optional[constr(max_length=500)] = Field(None, description="用户代理")
    error_message: Optional[constr(max_length=1000)] = Field(None, description="错误信息")
    error_type: Optional[constr(max_length=100)] = Field(None, description="错误类型")
    request_id: constr(min_length=1, max_length=100) = Field(..., description="请求ID")
    cache_hit: bool = Field(default=False, description="是否命中缓存")
    database_queries: conint(ge=0) = Field(default=0, description="数据库查询次数")
    external_api_calls: conint(ge=0) = Field(default=0, description="外部API调用次数")
    
    @validator('endpoint')
    def validate_endpoint(cls, v):
        # 验证端点格式
        if not v.startswith('/'):
            raise ValueError("API端点必须以/开头")
        return v
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        if v:
            import ipaddress
            try:
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError("无效的IP地址格式")
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_error_consistency(cls, values):
        status_code = values.get('status_code')
        error_message = values.get('error_message')
        
        # 4xx和5xx状态码应该有错误信息
        if status_code >= 400 and not error_message:
            values['error_message'] = f"HTTP {status_code} Error"
        
        return values


class ValidationErrorType(str, Enum):
    """验证错误类型枚举"""
    VALUE_ERROR = "value_error"
    TYPE_ERROR = "type_error"
    MISSING = "missing"
    EXTRA_FORBIDDEN = "extra_forbidden"
    STRING_TOO_SHORT = "string_too_short"
    STRING_TOO_LONG = "string_too_long"
    NUMBER_TOO_SMALL = "number_too_small"
    NUMBER_TOO_LARGE = "number_too_large"
    INVALID_FORMAT = "invalid_format"
    ENUM_INVALID = "enum_invalid"
    LIST_TOO_SHORT = "list_too_short"
    LIST_TOO_LONG = "list_too_long"
    DICT_INVALID = "dict_invalid"
    CUSTOM = "custom"


class ValidationErrorResponse(BaseModel):
    """验证错误响应模型"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        frozen=True  # 错误信息不可变
    )
    
    field: constr(min_length=1, max_length=200) = Field(..., description="字段路径")
    message: constr(min_length=1, max_length=500) = Field(..., description="错误信息")
    value: Any = Field(..., description="错误值")
    type: ValidationErrorType = Field(..., description="错误类型")
    code: Optional[constr(max_length=50)] = Field(None, description="错误代码")
    context: Optional[Dict[str, Any]] = Field(None, description="错误上下文")
    suggestion: Optional[constr(max_length=200)] = Field(None, description="修复建议")
    
    @classmethod
    def from_pydantic_error(cls, error: Dict[str, Any]) -> "ValidationErrorResponse":
        """从Pydantic验证错误创建响应"""
        error_type = error.get("type", "unknown")
        
        # 映射Pydantic错误类型到自定义枚举
        type_mapping = {
            "value_error": ValidationErrorType.VALUE_ERROR,
            "type_error": ValidationErrorType.TYPE_ERROR,
            "missing": ValidationErrorType.MISSING,
            "extra_forbidden": ValidationErrorType.EXTRA_FORBIDDEN,
            "string_too_short": ValidationErrorType.STRING_TOO_SHORT,
            "string_too_long": ValidationErrorType.STRING_TOO_LONG,
            "greater_than_equal": ValidationErrorType.NUMBER_TOO_SMALL,
            "less_than_equal": ValidationErrorType.NUMBER_TOO_LARGE,
            "enum": ValidationErrorType.ENUM_INVALID,
        }
        
        mapped_type = type_mapping.get(error_type, ValidationErrorType.CUSTOM)
        
        # 生成修复建议
        suggestion = cls._generate_suggestion(error_type, error)
        
        return cls(
            field="." + ".".join(str(loc) for loc in error.get("loc", [])),
            message=error.get("msg", "Validation error"),
            value=error.get("input"),
            type=mapped_type,
            code=error_type,
            context=error.get("ctx"),
            suggestion=suggestion
        )
    
    @staticmethod
    def _generate_suggestion(error_type: str, error: Dict[str, Any]) -> Optional[str]:
        """生成修复建议"""
        suggestions = {
            "missing": "请提供必需的字段值",
            "string_too_short": "请提供更长的字符串",
            "string_too_long": "请缩短字符串长度",
            "greater_than_equal": "请提供更大的数值",
            "less_than_equal": "请提供更小的数值",
            "enum": "请选择有效的枚举值",
            "type_error": "请检查数据类型是否正确",
        }
        return suggestions.get(error_type)
    
    @validator('value')
    def validate_value_size(cls, v):
        # 限制错误值的大小，防止内存问题
        if isinstance(v, str) and len(v) > 1000:
            return v[:1000] + "...(truncated)"
        return v


class RateLimitWindow(str, Enum):
    """限流时间窗口枚举"""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


class RateLimitResponse(BaseModel):
    """限流响应模型"""
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        frozen=True  # 限流信息不可变
    )
    
    limit: conint(ge=1) = Field(..., description="限制数量")
    remaining: conint(ge=0) = Field(..., description="剩余数量")
    used: conint(ge=0) = Field(..., description="已使用数量")
    reset_time: datetime = Field(..., description="重置时间")
    retry_after: Optional[conint(ge=1)] = Field(None, description="重试间隔（秒）")
    window: RateLimitWindow = Field(..., description="时间窗口")
    window_size: conint(ge=1) = Field(..., description="时间窗口大小")
    identifier: constr(min_length=1, max_length=100) = Field(..., description="限流标识符")
    policy: constr(min_length=1, max_length=50) = Field(..., description="限流策略")
    scope: constr(min_length=1, max_length=50) = Field(default="global", description="限流范围")
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_rate_limit_consistency(cls, values):
        limit = values.get('limit', 0)
        remaining = values.get('remaining', 0)
        used = values.get('used', 0)
        
        # 验证数量一致性
        if used + remaining != limit:
            # 自动修正used值
            values['used'] = limit - remaining
        
        # 验证剩余数量不超过限制
        if remaining > limit:
            raise ValueError("剩余数量不能超过限制数量")
        
        # 如果剩余为0，设置重试时间
        if remaining == 0 and not values.get('retry_after'):
            reset_time = values.get('reset_time')
            if reset_time:
                now = datetime.utcnow()
                values['retry_after'] = max(1, int((reset_time - now).total_seconds()))
        
        return values
    
    @validator('reset_time')
    def validate_reset_time(cls, v):
        # 重置时间不能是过去时间
        if v < datetime.utcnow():
            raise ValueError("重置时间不能是过去时间")
        return v
    
    def is_exceeded(self) -> bool:
        """检查是否超出限制"""
        return self.remaining == 0
    
    def get_reset_seconds(self) -> int:
        """获取距离重置的秒数"""
        now = datetime.utcnow()
        return max(0, int((self.reset_time - now).total_seconds()))
    
    def get_usage_percentage(self) -> float:
        """获取使用率百分比"""
        return (self.used / self.limit) * 100 if self.limit > 0 else 0


# 新增业务模型

class AuthenticationStatus(str, Enum):
    """认证状态"""
    AUTHENTICATED = "authenticated"
    UNAUTHENTICATED = "unauthenticated"
    EXPIRED = "expired"
    INVALID = "invalid"
    LOCKED = "locked"
    PENDING = "pending"

class PermissionLevel(str, Enum):
    """权限级别"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class AuditAction(str, Enum):
    """审计操作类型"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"
    IMPORT = "import"
    BACKUP = "backup"
    RESTORE = "restore"

class NotificationType(str, Enum):
    """通知类型"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    SYSTEM = "system"
    USER = "user"

class UserAuthResponse(BaseModel, SerializationMixin):
    """用户认证响应"""
    user_id: str = Field(..., description="用户ID", max_length=64)
    username: str = Field(..., description="用户名", max_length=100)
    email: Optional[str] = Field(None, description="邮箱", max_length=255)
    status: AuthenticationStatus = Field(..., description="认证状态")
    access_token: Optional[str] = Field(None, description="访问令牌", max_length=2048)
    refresh_token: Optional[str] = Field(None, description="刷新令牌", max_length=2048)
    expires_at: Optional[datetime] = Field(None, description="令牌过期时间")
    permissions: List[PermissionLevel] = Field(default_factory=list, description="用户权限")
    roles: List[str] = Field(default_factory=list, description="用户角色", max_items=20)
    last_login: Optional[datetime] = Field(None, description="最后登录时间")
    login_attempts: int = Field(default=0, description="登录尝试次数", ge=0, le=10)
    is_locked: bool = Field(default=False, description="账户是否锁定")
    lock_until: Optional[datetime] = Field(None, description="锁定到期时间")
    profile: Optional[Dict[str, Any]] = Field(default=None, description="用户资料")
    preferences: Optional[Dict[str, Any]] = Field(default=None, description="用户偏好设置")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        validate_assignment=True
    )
    
    @validator('email')
    def validate_email(cls, v):
        """验证邮箱格式"""
        if v and '@' not in v:
            raise ValueError("邮箱格式不正确")
        return v
    
    @validator('access_token', 'refresh_token')
    def validate_tokens(cls, v):
        """验证令牌格式"""
        if v and len(v) < 10:
            raise ValueError("令牌长度不能少于10个字符")
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_token_consistency(cls, values):
        """验证令牌一致性"""
        status = values.get('status')
        access_token = values.get('access_token')
        expires_at = values.get('expires_at')
        
        if status == AuthenticationStatus.AUTHENTICATED:
            if not access_token:
                raise ValueError("认证成功时必须提供访问令牌")
            if not expires_at:
                # 默认1小时过期
                values['expires_at'] = datetime.now(timezone.utc) + timedelta(hours=1)
        
        return values

class AuditLogEntry(BaseModel, SerializationMixin):
    """审计日志条目"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="日志ID")
    user_id: Optional[str] = Field(None, description="操作用户ID", max_length=64)
    username: Optional[str] = Field(None, description="操作用户名", max_length=100)
    action: AuditAction = Field(..., description="操作类型")
    resource_type: str = Field(..., description="资源类型", max_length=100)
    resource_id: Optional[str] = Field(None, description="资源ID", max_length=64)
    resource_name: Optional[str] = Field(None, description="资源名称", max_length=255)
    old_values: Optional[Dict[str, Any]] = Field(None, description="修改前的值")
    new_values: Optional[Dict[str, Any]] = Field(None, description="修改后的值")
    ip_address: Optional[str] = Field(None, description="IP地址", max_length=45)
    user_agent: Optional[str] = Field(None, description="用户代理", max_length=500)
    session_id: Optional[str] = Field(None, description="会话ID", max_length=64)
    request_id: Optional[str] = Field(None, description="请求ID", max_length=64)
    success: bool = Field(default=True, description="操作是否成功")
    error_message: Optional[str] = Field(None, description="错误信息", max_length=1000)
    duration_ms: Optional[int] = Field(None, description="操作耗时(毫秒)", ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="操作时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        validate_assignment=True
    )
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        """验证IP地址格式"""
        if v:
            import ipaddress
            try:
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError("IP地址格式不正确")
        return v
    
    @validator('metadata')
    def validate_metadata_size(cls, v):
        """验证元数据大小"""
        if v:
            metadata_str = json.dumps(v, default=str)
            if len(metadata_str) > 10240:  # 10KB
                raise ValueError("元数据大小不能超过10KB")
        return v

class NotificationMessage(BaseModel, SerializationMixin):
    """通知消息"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="通知ID")
    type: NotificationType = Field(..., description="通知类型")
    title: str = Field(..., description="通知标题", max_length=200)
    content: str = Field(..., description="通知内容", max_length=2000)
    recipient_id: Optional[str] = Field(None, description="接收者ID", max_length=64)
    recipient_type: str = Field(default="user", description="接收者类型", max_length=50)
    sender_id: Optional[str] = Field(None, description="发送者ID", max_length=64)
    channel: str = Field(default="system", description="通知渠道", max_length=50)
    priority: int = Field(default=1, description="优先级(1-5)", ge=1, le=5)
    is_read: bool = Field(default=False, description="是否已读")
    read_at: Optional[datetime] = Field(None, description="阅读时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")
    actions: List[Dict[str, str]] = Field(default_factory=list, description="可执行操作", max_items=10)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        validate_assignment=True
    )
    
    @validator('title', 'content')
    def validate_text_content(cls, v):
        """验证文本内容"""
        if not v or not v.strip():
            raise ValueError("标题和内容不能为空")
        return v.strip()
    
    @validator('actions')
    def validate_actions(cls, v):
        """验证操作列表"""
        for action in v:
            if not isinstance(action, dict) or 'type' not in action or 'label' not in action:
                raise ValueError("操作必须包含type和label字段")
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_read_consistency(cls, values):
        """验证阅读状态一致性"""
        is_read = values.get('is_read')
        read_at = values.get('read_at')
        
        if is_read and not read_at:
            values['read_at'] = datetime.now(timezone.utc)
        elif not is_read and read_at:
            values['is_read'] = True
        
        return values

class SystemConfigResponse(BaseModel, SerializationMixin):
    """系统配置响应"""
    config_key: str = Field(..., description="配置键", max_length=100)
    config_value: Any = Field(..., description="配置值")
    config_type: str = Field(..., description="配置类型", max_length=50)
    description: Optional[str] = Field(None, description="配置描述", max_length=500)
    is_sensitive: bool = Field(default=False, description="是否敏感配置")
    is_readonly: bool = Field(default=False, description="是否只读")
    default_value: Optional[Any] = Field(None, description="默认值")
    valid_values: Optional[List[Any]] = Field(None, description="有效值列表")
    min_value: Optional[Union[int, float]] = Field(None, description="最小值")
    max_value: Optional[Union[int, float]] = Field(None, description="最大值")
    pattern: Optional[str] = Field(None, description="值的正则模式")
    category: str = Field(default="general", description="配置分类", max_length=50)
    environment: str = Field(default="production", description="环境", max_length=20)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    updated_by: Optional[str] = Field(None, description="更新者", max_length=100)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        validate_assignment=True
    )
    
    @validator('config_value')
    def validate_config_value(cls, v, values):
        """验证配置值"""
        config_type = values.get('config_type')
        
        if config_type == 'int':
            try:
                return int(v)
            except (ValueError, TypeError):
                raise ValueError("配置值必须是整数")
        elif config_type == 'float':
            try:
                return float(v)
            except (ValueError, TypeError):
                raise ValueError("配置值必须是浮点数")
        elif config_type == 'bool':
            if isinstance(v, str):
                return v.lower() in ('true', '1', 'yes', 'on')
            return bool(v)
        elif config_type == 'json':
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    raise ValueError("配置值必须是有效的JSON")
        
        return v
    
    def model_dump(self, **kwargs):
        """序列化时处理敏感配置"""
        data = super().model_dump(**kwargs)
        if self.is_sensitive:
            data['config_value'] = "***HIDDEN***"
        return data

# 常用响应类型别名
StringResponse = BaseResponse[str]
IntResponse = BaseResponse[int]
BoolResponse = BaseResponse[bool]
DictResponse = BaseResponse[Dict[str, Any]]
ListResponse = BaseResponse[List[Any]]
UserAuthResponseType = BaseResponse[UserAuthResponse]
AuditLogResponseType = BaseResponse[List[AuditLogEntry]]
NotificationResponseType = BaseResponse[List[NotificationMessage]]
SystemConfigResponseType = BaseResponse[List[SystemConfigResponse]]

# 导出所有模型
__all__ = [
    "ResponseStatus",
    "ErrorCode",
    "ErrorDetail",
    "PaginationInfo",
    "BaseResponse",
    "PaginatedResponse",
    "HealthCheckResponse",
    "BatchOperationResult",
    "FileUploadResponse",
    "ExportResponse",
    "StatisticsResponse",
    "WebSocketMessage",
    "APIMetrics",
    "ValidationErrorResponse",
    "RateLimitResponse",
    "StringResponse",
    "IntResponse",
    "BoolResponse",
    "DictResponse",
    "ListResponse"
]