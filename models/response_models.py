"""业务响应模型定义

本模块定义了各个业务实体的API响应模型，包括用户、会话、线程、消息、工作流、记忆等。
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from uuid import UUID


class UserResponse(BaseModel):
    """用户响应模型"""
    id: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    email: str = Field(..., description="邮箱")
    full_name: Optional[str] = Field(None, description="全名")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    status: str = Field(..., description="用户状态")
    role: str = Field(..., description="用户角色")
    is_active: bool = Field(..., description="是否激活")
    is_verified: bool = Field(..., description="是否验证")
    last_login_at: Optional[datetime] = Field(None, description="最后登录时间")
    settings: Dict[str, Any] = Field(default_factory=dict, description="用户设置")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="用户偏好")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="用户统计")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class UserProfileResponse(UserResponse):
    """用户详细信息响应模型"""
    bio: Optional[str] = Field(None, description="个人简介")
    location: Optional[str] = Field(None, description="位置")
    website: Optional[str] = Field(None, description="网站")
    timezone: Optional[str] = Field(None, description="时区")
    language: Optional[str] = Field(None, description="语言")
    session_count: int = Field(default=0, description="会话数量")
    thread_count: int = Field(default=0, description="线程数量")
    message_count: int = Field(default=0, description="消息数量")


class SessionResponse(BaseModel):
    """会话响应模型"""
    id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")
    title: str = Field(..., description="会话标题")
    description: Optional[str] = Field(None, description="会话描述")
    is_active: bool = Field(..., description="是否激活")
    settings: Dict[str, Any] = Field(default_factory=dict, description="会话设置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    last_activity_at: Optional[datetime] = Field(None, description="最后活动时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    message_count: int = Field(default=0, description="消息数量")
    thread_count: int = Field(default=0, description="线程数量")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class ThreadResponse(BaseModel):
    """线程响应模型"""
    id: str = Field(..., description="线程ID")
    user_id: str = Field(..., description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    title: str = Field(..., description="线程标题")
    description: Optional[str] = Field(None, description="线程描述")
    is_active: bool = Field(..., description="是否激活")
    is_archived: bool = Field(..., description="是否归档")
    settings: Dict[str, Any] = Field(default_factory=dict, description="线程设置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    last_message_at: Optional[datetime] = Field(None, description="最后消息时间")
    message_count: int = Field(default=0, description="消息数量")
    token_count: int = Field(default=0, description="令牌数量")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class MessageResponse(BaseModel):
    """消息响应模型"""
    id: str = Field(..., description="消息ID")
    thread_id: str = Field(..., description="线程ID")
    user_id: str = Field(..., description="用户ID")
    role: str = Field(..., description="消息角色")
    message_type: str = Field(..., description="消息类型")
    content: str = Field(..., description="消息内容")
    content_data: Dict[str, Any] = Field(default_factory=dict, description="结构化内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    parent_id: Optional[str] = Field(None, description="父消息ID")
    reply_to_id: Optional[str] = Field(None, description="回复消息ID")
    is_edited: bool = Field(default=False, description="是否已编辑")
    is_deleted: bool = Field(default=False, description="是否已删除")
    is_pinned: bool = Field(default=False, description="是否置顶")
    token_count: int = Field(default=0, description="令牌数量")
    character_count: int = Field(default=0, description="字符数量")
    edit_count: int = Field(default=0, description="编辑次数")
    edited_at: Optional[datetime] = Field(None, description="编辑时间")
    deleted_at: Optional[datetime] = Field(None, description="删除时间")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class MessageWithAttachmentsResponse(MessageResponse):
    """包含附件的消息响应模型"""
    attachments: List['AttachmentResponse'] = Field(default_factory=list, description="附件列表")


class WorkflowResponse(BaseModel):
    """工作流响应模型"""
    id: str = Field(..., description="工作流ID")
    user_id: str = Field(..., description="用户ID")
    name: str = Field(..., description="工作流名称")
    description: Optional[str] = Field(None, description="工作流描述")
    version: str = Field(..., description="版本")
    definition: Dict[str, Any] = Field(..., description="工作流定义")
    config: Dict[str, Any] = Field(default_factory=dict, description="配置信息")
    status: str = Field(..., description="工作流状态")
    is_public: bool = Field(default=False, description="是否公开")
    is_template: bool = Field(default=False, description="是否模板")
    execution_count: int = Field(default=0, description="执行次数")
    success_count: int = Field(default=0, description="成功次数")
    failure_count: int = Field(default=0, description="失败次数")
    tags: List[str] = Field(default_factory=list, description="标签")
    category: Optional[str] = Field(None, description="分类")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class WorkflowExecutionResponse(BaseModel):
    """工作流执行响应模型"""
    id: str = Field(..., description="执行ID")
    workflow_id: str = Field(..., description="工作流ID")
    user_id: str = Field(..., description="用户ID")
    status: str = Field(..., description="执行状态")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="输出数据")
    context: Dict[str, Any] = Field(default_factory=dict, description="执行上下文")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    duration: Optional[float] = Field(None, description="执行时长")
    error_message: Optional[str] = Field(None, description="错误消息")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    step_count: int = Field(default=0, description="步骤总数")
    completed_steps: int = Field(default=0, description="已完成步骤")
    failed_steps: int = Field(default=0, description="失败步骤")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class WorkflowStepResponse(BaseModel):
    """工作流步骤响应模型"""
    id: str = Field(..., description="步骤ID")
    execution_id: str = Field(..., description="执行ID")
    step_name: str = Field(..., description="步骤名称")
    step_type: str = Field(..., description="步骤类型")
    step_order: int = Field(..., description="步骤顺序")
    status: str = Field(..., description="步骤状态")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="输出数据")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    duration: Optional[float] = Field(None, description="执行时长")
    error_message: Optional[str] = Field(None, description="错误消息")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    retry_count: int = Field(default=0, description="重试次数")
    max_retries: int = Field(default=3, description="最大重试次数")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class MemoryResponse(BaseModel):
    """记忆响应模型"""
    id: str = Field(..., description="记忆ID")
    user_id: str = Field(..., description="用户ID")
    thread_id: Optional[str] = Field(None, description="线程ID")
    memory_type: str = Field(..., description="记忆类型")
    title: Optional[str] = Field(None, description="记忆标题")
    content: str = Field(..., description="记忆内容")
    summary: Optional[str] = Field(None, description="记忆摘要")
    data: Dict[str, Any] = Field(default_factory=dict, description="结构化数据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    importance: float = Field(..., description="重要性评分")
    weight: float = Field(default=1.0, description="权重")
    accessed_at: datetime = Field(..., description="最后访问时间")
    access_count: int = Field(default=0, description="访问次数")
    tags: List[str] = Field(default_factory=list, description="标签")
    category: Optional[str] = Field(None, description="分类")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class MemoryVectorResponse(BaseModel):
    """记忆向量响应模型"""
    id: str = Field(..., description="向量ID")
    memory_id: str = Field(..., description="记忆ID")
    vector_type: str = Field(..., description="向量类型")
    model_name: str = Field(..., description="模型名称")
    vector_data: List[float] = Field(..., description="向量数据")
    dimension: int = Field(..., description="向量维度")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class TimeTravelResponse(BaseModel):
    """时间旅行响应模型"""
    id: str = Field(..., description="快照ID")
    user_id: str = Field(..., description="用户ID")
    thread_id: Optional[str] = Field(None, description="线程ID")
    snapshot_name: str = Field(..., description="快照名称")
    description: Optional[str] = Field(None, description="快照描述")
    snapshot_data: Dict[str, Any] = Field(..., description="快照数据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    snapshot_type: str = Field(..., description="快照类型")
    is_active: bool = Field(..., description="是否激活")
    data_size: int = Field(default=0, description="数据大小")
    restore_count: int = Field(default=0, description="恢复次数")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class AttachmentResponse(BaseModel):
    """附件响应模型"""
    id: str = Field(..., description="附件ID")
    user_id: str = Field(..., description="用户ID")
    message_id: Optional[str] = Field(None, description="消息ID")
    filename: str = Field(..., description="文件名")
    original_filename: str = Field(..., description="原始文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小")
    file_type: str = Field(..., description="文件类型")
    mime_type: str = Field(..., description="MIME类型")
    file_hash: str = Field(..., description="文件哈希")
    checksum: Optional[str] = Field(None, description="校验和")
    is_public: bool = Field(default=False, description="是否公开")
    is_processed: bool = Field(default=False, description="是否已处理")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    download_count: int = Field(default=0, description="下载次数")
    download_url: Optional[str] = Field(None, description="下载链接")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class UserPreferenceResponse(BaseModel):
    """用户偏好响应模型"""
    id: str = Field(..., description="偏好ID")
    user_id: str = Field(..., description="用户ID")
    preference_key: str = Field(..., description="偏好键")
    preference_value: Any = Field(..., description="偏好值")
    preference_type: str = Field(..., description="偏好类型")
    description: Optional[str] = Field(None, description="描述")
    is_public: bool = Field(default=False, description="是否公开")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class SystemConfigResponse(BaseModel):
    """系统配置响应模型"""
    id: str = Field(..., description="配置ID")
    config_key: str = Field(..., description="配置键")
    config_value: Any = Field(..., description="配置值")
    config_type: str = Field(..., description="配置类型")
    description: Optional[str] = Field(None, description="描述")
    is_active: bool = Field(..., description="是否激活")
    is_encrypted: bool = Field(..., description="是否加密")
    version: str = Field(..., description="版本")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


# 统计响应模型
class UserStatisticsResponse(BaseModel):
    """用户统计响应模型"""
    total_users: int = Field(..., description="总用户数")
    active_users: int = Field(..., description="活跃用户数")
    new_users_today: int = Field(..., description="今日新用户")
    verified_users: int = Field(..., description="已验证用户")
    user_growth_rate: float = Field(..., description="用户增长率")
    average_session_duration: float = Field(..., description="平均会话时长")
    top_active_users: List[Dict[str, Any]] = Field(default_factory=list, description="最活跃用户")


class MessageStatisticsResponse(BaseModel):
    """消息统计响应模型"""
    total_messages: int = Field(..., description="总消息数")
    messages_today: int = Field(..., description="今日消息数")
    messages_by_role: Dict[str, int] = Field(default_factory=dict, description="按角色分类的消息数")
    messages_by_type: Dict[str, int] = Field(default_factory=dict, description="按类型分类的消息数")
    average_message_length: float = Field(..., description="平均消息长度")
    total_tokens: int = Field(..., description="总令牌数")
    peak_message_time: Optional[str] = Field(None, description="消息高峰时间")


class WorkflowStatisticsResponse(BaseModel):
    """工作流统计响应模型"""
    total_workflows: int = Field(..., description="总工作流数")
    active_workflows: int = Field(..., description="活跃工作流数")
    workflows_by_status: Dict[str, int] = Field(default_factory=dict, description="按状态分类的工作流数")
    total_executions: int = Field(..., description="总执行次数")
    success_rate: float = Field(..., description="成功率")
    average_execution_time: float = Field(..., description="平均执行时间")
    popular_workflows: List[Dict[str, Any]] = Field(default_factory=list, description="热门工作流")


class MemoryStatisticsResponse(BaseModel):
    """记忆统计响应模型"""
    total_memories: int = Field(..., description="总记忆数")
    memories_by_type: Dict[str, int] = Field(default_factory=dict, description="按类型分类的记忆数")
    memories_today: int = Field(..., description="今日记忆数")
    average_importance: float = Field(..., description="平均重要性")
    most_accessed_memories: List[Dict[str, Any]] = Field(default_factory=list, description="最常访问的记忆")
    memory_categories: List[Dict[str, Any]] = Field(default_factory=list, description="记忆分类统计")


# 搜索响应模型
class SearchResult(BaseModel):
    """搜索结果项"""
    id: str = Field(..., description="结果ID")
    type: str = Field(..., description="结果类型")
    title: str = Field(..., description="标题")
    content: str = Field(..., description="内容")
    score: float = Field(..., description="相关性评分")
    highlights: List[str] = Field(default_factory=list, description="高亮片段")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: datetime = Field(..., description="创建时间")


class SearchResponse(BaseModel):
    """搜索响应模型"""
    query: str = Field(..., description="搜索查询")
    total: int = Field(..., description="总结果数")
    results: List[SearchResult] = Field(..., description="搜索结果")
    facets: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="分面统计")
    suggestions: List[str] = Field(default_factory=list, description="搜索建议")
    search_time: float = Field(..., description="搜索耗时")


# 导出所有响应模型
__all__ = [
    "UserResponse",
    "UserProfileResponse",
    "SessionResponse",
    "ThreadResponse",
    "MessageResponse",
    "MessageWithAttachmentsResponse",
    "WorkflowResponse",
    "WorkflowExecutionResponse",
    "WorkflowStepResponse",
    "MemoryResponse",
    "MemoryVectorResponse",
    "TimeTravelResponse",
    "AttachmentResponse",
    "UserPreferenceResponse",
    "SystemConfigResponse",
    "UserStatisticsResponse",
    "MessageStatisticsResponse",
    "WorkflowStatisticsResponse",
    "MemoryStatisticsResponse",
    "SearchResult",
    "SearchResponse"
]