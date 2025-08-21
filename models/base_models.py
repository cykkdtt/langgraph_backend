"""
基础响应模型

定义API响应的基础数据结构。
"""

from typing import Optional, Dict, Any, List, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')


class BaseResponse(BaseModel, Generic[T]):
    """基础响应模型"""
    success: bool = Field(description="请求是否成功")
    message: str = Field(description="响应消息")
    data: Optional[T] = Field(None, description="响应数据")
    error: Optional[str] = Field(None, description="错误信息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间")
    request_id: Optional[str] = Field(None, description="请求ID")
    
    @classmethod
    def success(cls, data: T = None, message: str = "操作成功") -> "BaseResponse[T]":
        """创建成功响应"""
        return cls(success=True, message=message, data=data)
    
    @classmethod
    def error(cls, message: str, error: str = None) -> "BaseResponse[T]":
        """创建错误响应"""
        return cls(success=False, message=message, error=error)


class SuccessResponse(BaseResponse):
    """成功响应模型"""
    success: bool = Field(True, description="请求成功")
    error: Optional[str] = Field(None, description="错误信息")


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = Field(False, description="请求失败")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")


class PaginationInfo(BaseModel):
    """分页信息"""
    page: int = Field(ge=1, description="当前页码")
    page_size: int = Field(ge=1, le=100, description="每页大小")
    total: int = Field(ge=0, description="总记录数")
    total_pages: int = Field(ge=0, description="总页数")
    has_next: bool = Field(description="是否有下一页")
    has_prev: bool = Field(description="是否有上一页")


class PaginatedResponse(BaseResponse, Generic[T]):
    """分页响应模型"""
    items: List[T] = Field(description="数据项列表")
    pagination: PaginationInfo = Field(description="分页信息")


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(description="服务状态")
    version: str = Field(description="服务版本")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="检查时间")
    components: Dict[str, Dict[str, Any]] = Field(description="组件状态")
    uptime: float = Field(description="运行时间（秒）")


class ValidationError(BaseModel):
    """验证错误"""
    field: str = Field(description="字段名")
    message: str = Field(description="错误消息")
    value: Any = Field(description="错误值")


class ValidationErrorResponse(ErrorResponse):
    """验证错误响应"""
    validation_errors: List[ValidationError] = Field(description="验证错误列表")