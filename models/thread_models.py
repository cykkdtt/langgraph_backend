"""对话线程管理相关数据模型

本模块定义了LangGraph多智能体系统中对话线程管理的数据模型，包括：
- 线程创建、更新、删除请求
- 线程信息响应
- 线程搜索和过滤
- 线程统计和分析
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import uuid

from models.chat_models import MessageRole, MessageType, ChatMode


class ThreadStatus(str, Enum):
    """线程状态"""
    ACTIVE = "active"          # 活跃
    ARCHIVED = "archived"      # 已归档
    DELETED = "deleted"        # 已删除
    PAUSED = "paused"          # 暂停
    COMPLETED = "completed"    # 已完成


class ThreadPriority(str, Enum):
    """线程优先级"""
    LOW = "low"               # 低优先级
    NORMAL = "normal"         # 普通优先级
    HIGH = "high"             # 高优先级
    URGENT = "urgent"         # 紧急


class ThreadType(str, Enum):
    """线程类型"""
    CHAT = "chat"             # 普通聊天
    TASK = "task"             # 任务执行
    WORKFLOW = "workflow"     # 工作流
    ANALYSIS = "analysis"     # 分析任务
    COLLABORATION = "collaboration"  # 协作


class SortOrder(str, Enum):
    """排序方式"""
    ASC = "asc"               # 升序
    DESC = "desc"             # 降序


class ThreadSortBy(str, Enum):
    """线程排序字段"""
    CREATED_AT = "created_at"  # 创建时间
    UPDATED_AT = "updated_at"  # 更新时间
    TITLE = "title"           # 标题
    PRIORITY = "priority"     # 优先级
    MESSAGE_COUNT = "message_count"  # 消息数量
    LAST_ACTIVITY = "last_activity"  # 最后活动时间


class ThreadCreateRequest(BaseModel):
    """创建线程请求"""
    title: str = Field(..., min_length=1, max_length=200, description="线程标题")
    description: Optional[str] = Field(None, max_length=1000, description="线程描述")
    thread_type: ThreadType = Field(ThreadType.CHAT, description="线程类型")
    priority: ThreadPriority = Field(ThreadPriority.NORMAL, description="优先级")
    mode: ChatMode = Field(ChatMode.SINGLE_AGENT, description="聊天模式")
    agent_type: Optional[str] = Field(None, description="指定智能体类型")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    initial_message: Optional[str] = Field(None, description="初始消息")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    
    @validator('tags')
    def validate_tags(cls, v):
        if len(v) > 10:
            raise ValueError('标签数量不能超过10个')
        return [tag.strip() for tag in v if tag.strip()]
    
    @validator('title')
    def validate_title(cls, v):
        return v.strip()


class ThreadUpdateRequest(BaseModel):
    """更新线程请求"""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="线程标题")
    description: Optional[str] = Field(None, max_length=1000, description="线程描述")
    status: Optional[ThreadStatus] = Field(None, description="线程状态")
    priority: Optional[ThreadPriority] = Field(None, description="优先级")
    tags: Optional[List[str]] = Field(None, description="标签列表")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    
    @validator('tags')
    def validate_tags(cls, v):
        if v is not None and len(v) > 10:
            raise ValueError('标签数量不能超过10个')
        return [tag.strip() for tag in v if tag.strip()] if v else v
    
    @validator('title')
    def validate_title(cls, v):
        return v.strip() if v else v


class ThreadSearchRequest(BaseModel):
    """线程搜索请求"""
    query: Optional[str] = Field(None, description="搜索关键词")
    status: Optional[List[ThreadStatus]] = Field(None, description="状态过滤")
    thread_type: Optional[List[ThreadType]] = Field(None, description="类型过滤")
    priority: Optional[List[ThreadPriority]] = Field(None, description="优先级过滤")
    tags: Optional[List[str]] = Field(None, description="标签过滤")
    agent_type: Optional[str] = Field(None, description="智能体类型过滤")
    created_after: Optional[datetime] = Field(None, description="创建时间起始")
    created_before: Optional[datetime] = Field(None, description="创建时间结束")
    updated_after: Optional[datetime] = Field(None, description="更新时间起始")
    updated_before: Optional[datetime] = Field(None, description="更新时间结束")
    sort_by: ThreadSortBy = Field(ThreadSortBy.UPDATED_AT, description="排序字段")
    sort_order: SortOrder = Field(SortOrder.DESC, description="排序方式")
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页数量")


class ThreadMessageSummary(BaseModel):
    """线程消息摘要"""
    message_id: str = Field(..., description="消息ID")
    role: MessageRole = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    message_type: MessageType = Field(..., description="消息类型")
    created_at: datetime = Field(..., description="创建时间")
    agent_type: Optional[str] = Field(None, description="智能体类型")


class ThreadParticipant(BaseModel):
    """线程参与者"""
    user_id: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    role: str = Field(..., description="参与角色")
    joined_at: datetime = Field(..., description="加入时间")
    last_activity: Optional[datetime] = Field(None, description="最后活动时间")
    message_count: int = Field(0, description="消息数量")


class ThreadStatistics(BaseModel):
    """线程统计信息"""
    total_messages: int = Field(0, description="总消息数")
    user_messages: int = Field(0, description="用户消息数")
    agent_messages: int = Field(0, description="智能体消息数")
    tool_calls: int = Field(0, description="工具调用次数")
    participants_count: int = Field(0, description="参与者数量")
    avg_response_time: Optional[float] = Field(None, description="平均响应时间(秒)")
    first_message_at: Optional[datetime] = Field(None, description="首条消息时间")
    last_message_at: Optional[datetime] = Field(None, description="最后消息时间")
    active_duration: Optional[int] = Field(None, description="活跃时长(秒)")


class ThreadInfo(BaseModel):
    """线程信息"""
    thread_id: str = Field(..., description="线程ID")
    title: str = Field(..., description="线程标题")
    description: Optional[str] = Field(None, description="线程描述")
    thread_type: ThreadType = Field(..., description="线程类型")
    status: ThreadStatus = Field(..., description="线程状态")
    priority: ThreadPriority = Field(..., description="优先级")
    mode: ChatMode = Field(..., description="聊天模式")
    owner_id: str = Field(..., description="创建者ID")
    owner_username: str = Field(..., description="创建者用户名")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    last_activity: Optional[datetime] = Field(None, description="最后活动时间")
    statistics: ThreadStatistics = Field(..., description="统计信息")
    participants: List[ThreadParticipant] = Field(default_factory=list, description="参与者列表")
    recent_messages: List[ThreadMessageSummary] = Field(default_factory=list, description="最近消息")


class ThreadListResponse(BaseModel):
    """线程列表响应"""
    threads: List[ThreadInfo] = Field(..., description="线程列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页数量")
    total_pages: int = Field(..., description="总页数")
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")


class ThreadCreateResponse(BaseModel):
    """创建线程响应"""
    thread_id: str = Field(..., description="线程ID")
    title: str = Field(..., description="线程标题")
    status: ThreadStatus = Field(..., description="线程状态")
    created_at: datetime = Field(..., description="创建时间")
    message: str = Field("线程创建成功", description="响应消息")


class ThreadUpdateResponse(BaseModel):
    """更新线程响应"""
    thread_id: str = Field(..., description="线程ID")
    updated_fields: List[str] = Field(..., description="更新的字段")
    updated_at: datetime = Field(..., description="更新时间")
    message: str = Field("线程更新成功", description="响应消息")


class ThreadDeleteResponse(BaseModel):
    """删除线程响应"""
    thread_id: str = Field(..., description="线程ID")
    deleted_at: datetime = Field(..., description="删除时间")
    message: str = Field("线程删除成功", description="响应消息")


class ThreadArchiveRequest(BaseModel):
    """归档线程请求"""
    reason: Optional[str] = Field(None, max_length=500, description="归档原因")
    archive_messages: bool = Field(True, description="是否归档消息")


class ThreadArchiveResponse(BaseModel):
    """归档线程响应"""
    thread_id: str = Field(..., description="线程ID")
    archived_at: datetime = Field(..., description="归档时间")
    archived_messages_count: int = Field(0, description="归档的消息数量")
    message: str = Field("线程归档成功", description="响应消息")


class ThreadRestoreRequest(BaseModel):
    """恢复线程请求"""
    reason: Optional[str] = Field(None, max_length=500, description="恢复原因")
    restore_messages: bool = Field(True, description="是否恢复消息")


class ThreadRestoreResponse(BaseModel):
    """恢复线程响应"""
    thread_id: str = Field(..., description="线程ID")
    restored_at: datetime = Field(..., description="恢复时间")
    restored_messages_count: int = Field(0, description="恢复的消息数量")
    message: str = Field("线程恢复成功", description="响应消息")


class ThreadBatchOperation(BaseModel):
    """批量操作请求"""
    thread_ids: List[str] = Field(..., min_items=1, max_items=100, description="线程ID列表")
    operation: str = Field(..., description="操作类型: delete, archive, restore, update_status")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="操作参数")


class ThreadBatchOperationResponse(BaseModel):
    """批量操作响应"""
    operation: str = Field(..., description="操作类型")
    total_requested: int = Field(..., description="请求处理的总数")
    successful: int = Field(..., description="成功处理的数量")
    failed: int = Field(..., description="失败的数量")
    failed_thread_ids: List[str] = Field(default_factory=list, description="失败的线程ID")
    errors: List[str] = Field(default_factory=list, description="错误信息")
    processed_at: datetime = Field(..., description="处理时间")


class ThreadExportRequest(BaseModel):
    """导出线程请求"""
    thread_ids: Optional[List[str]] = Field(None, description="指定线程ID列表")
    search_criteria: Optional[ThreadSearchRequest] = Field(None, description="搜索条件")
    export_format: str = Field("json", description="导出格式: json, csv, markdown")
    include_messages: bool = Field(True, description="是否包含消息")
    include_metadata: bool = Field(True, description="是否包含元数据")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="日期范围")


class ThreadExportResponse(BaseModel):
    """导出线程响应"""
    export_id: str = Field(..., description="导出任务ID")
    download_url: str = Field(..., description="下载链接")
    file_size: int = Field(..., description="文件大小(字节)")
    thread_count: int = Field(..., description="导出的线程数量")
    export_format: str = Field(..., description="导出格式")
    created_at: datetime = Field(..., description="创建时间")
    expires_at: datetime = Field(..., description="过期时间")


class ThreadAnalyticsRequest(BaseModel):
    """线程分析请求"""
    date_range: Dict[str, datetime] = Field(..., description="日期范围")
    group_by: str = Field("day", description="分组方式: hour, day, week, month")
    metrics: List[str] = Field(
        default_factory=lambda: ["thread_count", "message_count", "user_activity"],
        description="分析指标"
    )
    filters: Optional[ThreadSearchRequest] = Field(None, description="过滤条件")


class ThreadAnalyticsData(BaseModel):
    """线程分析数据点"""
    timestamp: datetime = Field(..., description="时间点")
    thread_count: int = Field(0, description="线程数量")
    message_count: int = Field(0, description="消息数量")
    user_activity: int = Field(0, description="用户活跃度")
    avg_response_time: Optional[float] = Field(None, description="平均响应时间")
    completion_rate: Optional[float] = Field(None, description="完成率")


class ThreadAnalyticsResponse(BaseModel):
    """线程分析响应"""
    data_points: List[ThreadAnalyticsData] = Field(..., description="数据点列表")
    summary: Dict[str, Any] = Field(..., description="汇总统计")
    trends: Dict[str, Any] = Field(..., description="趋势分析")
    generated_at: datetime = Field(..., description="生成时间")