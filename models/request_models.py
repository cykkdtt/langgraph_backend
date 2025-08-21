"""API请求模型定义

本模块定义了各个API端点的请求参数验证模型，包括创建、更新、查询等操作的请求体。
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator, EmailStr
from uuid import UUID


# 基础请求模型
class PaginationRequest(BaseModel):
    """分页请求模型"""
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")
    sort_by: Optional[str] = Field(None, description="排序字段")
    sort_order: Optional[str] = Field(default="desc", regex="^(asc|desc)$", description="排序方向")


class SearchRequest(PaginationRequest):
    """搜索请求模型"""
    query: str = Field(..., min_length=1, max_length=500, description="搜索查询")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    facets: Optional[List[str]] = Field(None, description="分面字段")
    highlight: bool = Field(default=True, description="是否高亮")


# 用户相关请求模型
class UserCreateRequest(BaseModel):
    """用户创建请求模型"""
    username: str = Field(..., min_length=3, max_length=50, regex="^[a-zA-Z0-9_-]+$", description="用户名")
    email: EmailStr = Field(..., description="邮箱")
    password: str = Field(..., min_length=8, max_length=128, description="密码")
    full_name: Optional[str] = Field(None, max_length=100, description="全名")
    bio: Optional[str] = Field(None, max_length=500, description="个人简介")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    timezone: Optional[str] = Field(None, description="时区")
    language: Optional[str] = Field(default="zh-CN", description="语言")

    @validator('password')
    def validate_password(cls, v):
        """密码强度验证"""
        if not any(c.isupper() for c in v):
            raise ValueError('密码必须包含至少一个大写字母')
        if not any(c.islower() for c in v):
            raise ValueError('密码必须包含至少一个小写字母')
        if not any(c.isdigit() for c in v):
            raise ValueError('密码必须包含至少一个数字')
        return v


class UserUpdateRequest(BaseModel):
    """用户更新请求模型"""
    full_name: Optional[str] = Field(None, max_length=100, description="全名")
    bio: Optional[str] = Field(None, max_length=500, description="个人简介")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    location: Optional[str] = Field(None, max_length=100, description="位置")
    website: Optional[str] = Field(None, description="网站")
    timezone: Optional[str] = Field(None, description="时区")
    language: Optional[str] = Field(None, description="语言")
    settings: Optional[Dict[str, Any]] = Field(None, description="用户设置")


class UserLoginRequest(BaseModel):
    """用户登录请求模型"""
    username_or_email: str = Field(..., min_length=1, description="用户名或邮箱")
    password: str = Field(..., min_length=1, description="密码")
    remember_me: bool = Field(default=False, description="记住我")


class PasswordChangeRequest(BaseModel):
    """密码修改请求模型"""
    current_password: str = Field(..., description="当前密码")
    new_password: str = Field(..., min_length=8, max_length=128, description="新密码")
    confirm_password: str = Field(..., description="确认密码")

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('密码确认不匹配')
        return v


class PasswordResetRequest(BaseModel):
    """密码重置请求模型"""
    email: EmailStr = Field(..., description="邮箱")


class PasswordResetConfirmRequest(BaseModel):
    """密码重置确认请求模型"""
    token: str = Field(..., description="重置令牌")
    new_password: str = Field(..., min_length=8, max_length=128, description="新密码")
    confirm_password: str = Field(..., description="确认密码")

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('密码确认不匹配')
        return v


# 会话相关请求模型
class SessionCreateRequest(BaseModel):
    """会话创建请求模型"""
    title: str = Field(..., min_length=1, max_length=200, description="会话标题")
    description: Optional[str] = Field(None, max_length=1000, description="会话描述")
    settings: Optional[Dict[str, Any]] = Field(None, description="会话设置")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    expires_at: Optional[datetime] = Field(None, description="过期时间")


class SessionUpdateRequest(BaseModel):
    """会话更新请求模型"""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="会话标题")
    description: Optional[str] = Field(None, max_length=1000, description="会话描述")
    is_active: Optional[bool] = Field(None, description="是否激活")
    settings: Optional[Dict[str, Any]] = Field(None, description="会话设置")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    expires_at: Optional[datetime] = Field(None, description="过期时间")


# 线程相关请求模型
class ThreadCreateRequest(BaseModel):
    """线程创建请求模型"""
    session_id: Optional[str] = Field(None, description="会话ID")
    title: str = Field(..., min_length=1, max_length=200, description="线程标题")
    description: Optional[str] = Field(None, max_length=1000, description="线程描述")
    settings: Optional[Dict[str, Any]] = Field(None, description="线程设置")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class ThreadUpdateRequest(BaseModel):
    """线程更新请求模型"""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="线程标题")
    description: Optional[str] = Field(None, max_length=1000, description="线程描述")
    is_active: Optional[bool] = Field(None, description="是否激活")
    is_archived: Optional[bool] = Field(None, description="是否归档")
    settings: Optional[Dict[str, Any]] = Field(None, description="线程设置")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


# 消息相关请求模型
class MessageCreateRequest(BaseModel):
    """消息创建请求模型"""
    thread_id: str = Field(..., description="线程ID")
    role: str = Field(..., regex="^(user|assistant|system|function)$", description="消息角色")
    message_type: str = Field(default="text", regex="^(text|image|file|audio|video|code|markdown)$", description="消息类型")
    content: str = Field(..., min_length=1, description="消息内容")
    content_data: Optional[Dict[str, Any]] = Field(None, description="结构化内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    parent_id: Optional[str] = Field(None, description="父消息ID")
    reply_to_id: Optional[str] = Field(None, description="回复消息ID")


class MessageUpdateRequest(BaseModel):
    """消息更新请求模型"""
    content: Optional[str] = Field(None, min_length=1, description="消息内容")
    content_data: Optional[Dict[str, Any]] = Field(None, description="结构化内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    is_pinned: Optional[bool] = Field(None, description="是否置顶")


class MessageBatchDeleteRequest(BaseModel):
    """消息批量删除请求模型"""
    message_ids: List[str] = Field(..., min_items=1, max_items=100, description="消息ID列表")
    hard_delete: bool = Field(default=False, description="是否硬删除")


# 工作流相关请求模型
class WorkflowCreateRequest(BaseModel):
    """工作流创建请求模型"""
    name: str = Field(..., min_length=1, max_length=100, description="工作流名称")
    description: Optional[str] = Field(None, max_length=1000, description="工作流描述")
    definition: Dict[str, Any] = Field(..., description="工作流定义")
    config: Optional[Dict[str, Any]] = Field(None, description="配置信息")
    is_public: bool = Field(default=False, description="是否公开")
    is_template: bool = Field(default=False, description="是否模板")
    tags: Optional[List[str]] = Field(None, description="标签")
    category: Optional[str] = Field(None, max_length=50, description="分类")


class WorkflowUpdateRequest(BaseModel):
    """工作流更新请求模型"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="工作流名称")
    description: Optional[str] = Field(None, max_length=1000, description="工作流描述")
    definition: Optional[Dict[str, Any]] = Field(None, description="工作流定义")
    config: Optional[Dict[str, Any]] = Field(None, description="配置信息")
    status: Optional[str] = Field(None, regex="^(draft|active|inactive|deprecated)$", description="工作流状态")
    is_public: Optional[bool] = Field(None, description="是否公开")
    is_template: Optional[bool] = Field(None, description="是否模板")
    tags: Optional[List[str]] = Field(None, description="标签")
    category: Optional[str] = Field(None, max_length=50, description="分类")


class WorkflowExecuteRequest(BaseModel):
    """工作流执行请求模型"""
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    context: Optional[Dict[str, Any]] = Field(None, description="执行上下文")
    async_execution: bool = Field(default=True, description="是否异步执行")


# 记忆相关请求模型
class MemoryCreateRequest(BaseModel):
    """记忆创建请求模型"""
    thread_id: Optional[str] = Field(None, description="线程ID")
    memory_type: str = Field(..., regex="^(semantic|episodic|procedural|working)$", description="记忆类型")
    title: Optional[str] = Field(None, max_length=200, description="记忆标题")
    content: str = Field(..., min_length=1, description="记忆内容")
    summary: Optional[str] = Field(None, max_length=500, description="记忆摘要")
    data: Optional[Dict[str, Any]] = Field(None, description="结构化数据")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性评分")
    weight: float = Field(default=1.0, ge=0.0, description="权重")
    tags: Optional[List[str]] = Field(None, description="标签")
    category: Optional[str] = Field(None, max_length=50, description="分类")


class MemoryUpdateRequest(BaseModel):
    """记忆更新请求模型"""
    title: Optional[str] = Field(None, max_length=200, description="记忆标题")
    content: Optional[str] = Field(None, min_length=1, description="记忆内容")
    summary: Optional[str] = Field(None, max_length=500, description="记忆摘要")
    data: Optional[Dict[str, Any]] = Field(None, description="结构化数据")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    importance: Optional[float] = Field(None, ge=0.0, le=1.0, description="重要性评分")
    weight: Optional[float] = Field(None, ge=0.0, description="权重")
    tags: Optional[List[str]] = Field(None, description="标签")
    category: Optional[str] = Field(None, max_length=50, description="分类")


class MemorySearchRequest(SearchRequest):
    """记忆搜索请求模型"""
    memory_types: Optional[List[str]] = Field(None, description="记忆类型过滤")
    importance_min: Optional[float] = Field(None, ge=0.0, le=1.0, description="最小重要性")
    importance_max: Optional[float] = Field(None, ge=0.0, le=1.0, description="最大重要性")
    tags: Optional[List[str]] = Field(None, description="标签过滤")
    category: Optional[str] = Field(None, description="分类过滤")
    thread_id: Optional[str] = Field(None, description="线程ID过滤")


# 时间旅行相关请求模型
class TimeTravelCreateRequest(BaseModel):
    """时间旅行快照创建请求模型"""
    thread_id: Optional[str] = Field(None, description="线程ID")
    snapshot_name: str = Field(..., min_length=1, max_length=100, description="快照名称")
    description: Optional[str] = Field(None, max_length=500, description="快照描述")
    snapshot_data: Dict[str, Any] = Field(..., description="快照数据")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    snapshot_type: str = Field(default="manual", regex="^(manual|auto|checkpoint)$", description="快照类型")


class TimeTravelRestoreRequest(BaseModel):
    """时间旅行恢复请求模型"""
    snapshot_id: str = Field(..., description="快照ID")
    restore_options: Optional[Dict[str, Any]] = Field(None, description="恢复选项")


# 附件相关请求模型
class AttachmentUploadRequest(BaseModel):
    """附件上传请求模型"""
    message_id: Optional[str] = Field(None, description="消息ID")
    filename: str = Field(..., min_length=1, max_length=255, description="文件名")
    file_type: str = Field(..., description="文件类型")
    file_size: int = Field(..., gt=0, description="文件大小")
    is_public: bool = Field(default=False, description="是否公开")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


# 用户偏好相关请求模型
class UserPreferenceCreateRequest(BaseModel):
    """用户偏好创建请求模型"""
    preference_key: str = Field(..., min_length=1, max_length=100, description="偏好键")
    preference_value: Any = Field(..., description="偏好值")
    preference_type: str = Field(default="string", regex="^(string|number|boolean|object|array)$", description="偏好类型")
    description: Optional[str] = Field(None, max_length=500, description="描述")
    is_public: bool = Field(default=False, description="是否公开")


class UserPreferenceUpdateRequest(BaseModel):
    """用户偏好更新请求模型"""
    preference_value: Optional[Any] = Field(None, description="偏好值")
    description: Optional[str] = Field(None, max_length=500, description="描述")
    is_public: Optional[bool] = Field(None, description="是否公开")


class UserPreferenceBatchUpdateRequest(BaseModel):
    """用户偏好批量更新请求模型"""
    preferences: Dict[str, Any] = Field(..., description="偏好设置字典")


# 系统配置相关请求模型
class SystemConfigCreateRequest(BaseModel):
    """系统配置创建请求模型"""
    config_key: str = Field(..., min_length=1, max_length=100, description="配置键")
    config_value: Any = Field(..., description="配置值")
    config_type: str = Field(default="string", regex="^(string|number|boolean|object|array)$", description="配置类型")
    description: Optional[str] = Field(None, max_length=500, description="描述")
    is_encrypted: bool = Field(default=False, description="是否加密")


class SystemConfigUpdateRequest(BaseModel):
    """系统配置更新请求模型"""
    config_value: Optional[Any] = Field(None, description="配置值")
    description: Optional[str] = Field(None, max_length=500, description="描述")
    is_active: Optional[bool] = Field(None, description="是否激活")
    is_encrypted: Optional[bool] = Field(None, description="是否加密")


# 批量操作请求模型
class BatchDeleteRequest(BaseModel):
    """批量删除请求模型"""
    ids: List[str] = Field(..., min_items=1, max_items=100, description="ID列表")
    hard_delete: bool = Field(default=False, description="是否硬删除")


class BatchUpdateRequest(BaseModel):
    """批量更新请求模型"""
    ids: List[str] = Field(..., min_items=1, max_items=100, description="ID列表")
    update_data: Dict[str, Any] = Field(..., description="更新数据")


# 导出请求模型
class ExportRequest(BaseModel):
    """导出请求模型"""
    export_type: str = Field(..., regex="^(json|csv|xlsx|pdf)$", description="导出格式")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    fields: Optional[List[str]] = Field(None, description="导出字段")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="日期范围")
    include_metadata: bool = Field(default=False, description="是否包含元数据")


# 统计查询请求模型
class StatisticsRequest(BaseModel):
    """统计查询请求模型"""
    metrics: List[str] = Field(..., min_items=1, description="统计指标")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="日期范围")
    group_by: Optional[List[str]] = Field(None, description="分组字段")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    aggregation: str = Field(default="sum", regex="^(sum|avg|count|min|max)$", description="聚合方式")


# WebSocket请求模型
class WebSocketConnectRequest(BaseModel):
    """WebSocket连接请求模型"""
    session_id: Optional[str] = Field(None, description="会话ID")
    thread_id: Optional[str] = Field(None, description="线程ID")
    subscribe_events: Optional[List[str]] = Field(None, description="订阅事件")


class WebSocketMessageRequest(BaseModel):
    """WebSocket消息请求模型"""
    type: str = Field(..., description="消息类型")
    event: str = Field(..., description="事件名称")
    data: Dict[str, Any] = Field(default_factory=dict, description="消息数据")
    request_id: Optional[str] = Field(None, description="请求ID")


# 导出所有请求模型
__all__ = [
    "PaginationRequest",
    "SearchRequest",
    "UserCreateRequest",
    "UserUpdateRequest",
    "UserLoginRequest",
    "PasswordChangeRequest",
    "PasswordResetRequest",
    "PasswordResetConfirmRequest",
    "SessionCreateRequest",
    "SessionUpdateRequest",
    "ThreadCreateRequest",
    "ThreadUpdateRequest",
    "MessageCreateRequest",
    "MessageUpdateRequest",
    "MessageBatchDeleteRequest",
    "WorkflowCreateRequest",
    "WorkflowUpdateRequest",
    "WorkflowExecuteRequest",
    "MemoryCreateRequest",
    "MemoryUpdateRequest",
    "MemorySearchRequest",
    "TimeTravelCreateRequest",
    "TimeTravelRestoreRequest",
    "AttachmentUploadRequest",
    "UserPreferenceCreateRequest",
    "UserPreferenceUpdateRequest",
    "UserPreferenceBatchUpdateRequest",
    "SystemConfigCreateRequest",
    "SystemConfigUpdateRequest",
    "BatchDeleteRequest",
    "BatchUpdateRequest",
    "ExportRequest",
    "StatisticsRequest",
    "WebSocketConnectRequest",
    "WebSocketMessageRequest"
]