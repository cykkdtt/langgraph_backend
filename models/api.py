"""API响应模型定义

包含所有API接口的请求和响应数据模型，提供数据验证和序列化功能。
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, model_validator, EmailStr


class ResponseStatus(str, Enum):
    """响应状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class SortOrder(str, Enum):
    """排序方向枚举"""
    ASC = "asc"
    DESC = "desc"


# ============================================================================
# 基础响应模型
# ============================================================================

class BaseResponse(BaseModel):
    """基础响应模型"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    message: str = "操作成功"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataResponse(BaseResponse):
    """数据响应模型"""
    data: Any = None


class ListResponse(BaseResponse):
    """列表响应模型"""
    data: List[Any] = []
    total: int = 0
    page: int = 1
    page_size: int = 20
    has_next: bool = False
    has_prev: bool = False


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    status: ResponseStatus = ResponseStatus.ERROR
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None


# ============================================================================
# 分页和查询模型
# ============================================================================

class PaginationParams(BaseModel):
    """分页参数"""
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页数量")
    
    @property
    def offset(self) -> int:
        """计算偏移量"""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """获取限制数量"""
        return self.page_size


class SortParams(BaseModel):
    """排序参数"""
    sort_by: str = Field("created_at", description="排序字段")
    sort_order: SortOrder = Field(SortOrder.DESC, description="排序方向")


class FilterParams(BaseModel):
    """过滤参数"""
    search: Optional[str] = Field(None, description="搜索关键词")
    tags: Optional[List[str]] = Field(None, description="标签过滤")
    category: Optional[str] = Field(None, description="分类过滤")
    status: Optional[str] = Field(None, description="状态过滤")
    date_from: Optional[datetime] = Field(None, description="开始日期")
    date_to: Optional[datetime] = Field(None, description="结束日期")


class QueryParams(PaginationParams, SortParams, FilterParams):
    """查询参数（组合分页、排序、过滤）"""
    pass


# ============================================================================
# 用户相关模型
# ============================================================================

class UserBase(BaseModel):
    """用户基础模型"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱地址")
    full_name: Optional[str] = Field(None, max_length=100, description="全名")
    bio: Optional[str] = Field(None, max_length=500, description="个人简介")
    avatar_url: Optional[str] = Field(None, max_length=500, description="头像URL")
    
    @validator('username')
    def validate_username(cls, v):
        """验证用户名格式"""
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('用户名只能包含字母、数字、下划线和连字符')
        return v.lower()


class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(..., min_length=8, max_length=128, description="密码")
    confirm_password: str = Field(..., description="确认密码")
    
    @validator('password')
    def validate_password(cls, v):
        """验证密码强度"""
        import re
        if not re.search(r'[A-Z]', v):
            raise ValueError('密码必须包含至少一个大写字母')
        if not re.search(r'[a-z]', v):
            raise ValueError('密码必须包含至少一个小写字母')
        if not re.search(r'\d', v):
            raise ValueError('密码必须包含至少一个数字')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('密码必须包含至少一个特殊字符')
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_passwords_match(cls, values):
        """验证密码确认"""
        password = values.get('password')
        confirm_password = values.get('confirm_password')
        if password and confirm_password and password != confirm_password:
            raise ValueError('密码和确认密码不匹配')
        return values


class UserUpdate(BaseModel):
    """用户更新模型"""
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = Field(None, max_length=500)
    settings: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None


class UserResponse(UserBase):
    """用户响应模型"""
    id: UUID
    status: str
    is_admin: bool
    is_verified: bool
    last_login_at: Optional[datetime]
    email_verified_at: Optional[datetime]
    settings: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}
    login_count: int = 0
    message_count: int = 0
    thread_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class UserLogin(BaseModel):
    """用户登录模型"""
    username: str = Field(..., description="用户名或邮箱")
    password: str = Field(..., description="密码")
    remember_me: bool = Field(False, description="记住我")


class UserLoginResponse(BaseResponse):
    """用户登录响应模型"""
    data: Dict[str, Any] = Field(default_factory=dict)
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class PasswordChange(BaseModel):
    """密码修改模型"""
    current_password: str = Field(..., description="当前密码")
    new_password: str = Field(..., min_length=8, max_length=128, description="新密码")
    confirm_password: str = Field(..., description="确认新密码")
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_passwords_match(cls, values):
        """验证密码确认"""
        new_password = values.get('new_password')
        confirm_password = values.get('confirm_password')
        if new_password and confirm_password and new_password != confirm_password:
            raise ValueError('新密码和确认密码不匹配')
        return values


class PasswordReset(BaseModel):
    """密码重置模型"""
    email: EmailStr = Field(..., description="邮箱地址")


class PasswordResetConfirm(BaseModel):
    """密码重置确认模型"""
    token: str = Field(..., description="重置令牌")
    new_password: str = Field(..., min_length=8, max_length=128, description="新密码")
    confirm_password: str = Field(..., description="确认新密码")
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_passwords_match(cls, values):
        """验证密码确认"""
        new_password = values.get('new_password')
        confirm_password = values.get('confirm_password')
        if new_password and confirm_password and new_password != confirm_password:
            raise ValueError('新密码和确认密码不匹配')
        return values


# ============================================================================
# 会话相关模型
# ============================================================================

class SessionBase(BaseModel):
    """会话基础模型"""
    title: str = Field(..., min_length=1, max_length=200, description="会话标题")
    description: Optional[str] = Field(None, max_length=1000, description="会话描述")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="会话设置")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")


class SessionCreate(SessionBase):
    """会话创建模型"""
    expires_in: Optional[int] = Field(None, ge=3600, description="过期时间（秒）")


class SessionUpdate(BaseModel):
    """会话更新模型"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    is_active: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(SessionBase):
    """会话响应模型"""
    id: UUID
    user_id: UUID
    is_active: bool
    last_activity_at: datetime
    expires_at: Optional[datetime]
    message_count: int = 0
    thread_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# 线程相关模型
# ============================================================================

class ThreadBase(BaseModel):
    """线程基础模型"""
    title: str = Field(..., min_length=1, max_length=200, description="线程标题")
    description: Optional[str] = Field(None, max_length=1000, description="线程描述")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="线程设置")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")


class ThreadCreate(ThreadBase):
    """线程创建模型"""
    session_id: Optional[UUID] = Field(None, description="所属会话ID")


class ThreadUpdate(BaseModel):
    """线程更新模型"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    is_active: Optional[bool] = None
    is_archived: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ThreadResponse(ThreadBase):
    """线程响应模型"""
    id: UUID
    user_id: UUID
    session_id: Optional[UUID]
    is_active: bool
    is_archived: bool
    last_message_at: Optional[datetime]
    message_count: int = 0
    token_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# 消息相关模型
# ============================================================================

class MessageBase(BaseModel):
    """消息基础模型"""
    role: str = Field(..., description="消息角色")
    message_type: str = Field("text", description="消息类型")
    content: str = Field(..., min_length=1, description="消息内容")
    content_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="结构化内容")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")
    
    @validator('role')
    def validate_role(cls, v):
        """验证消息角色"""
        valid_roles = ['user', 'assistant', 'system', 'function']
        if v not in valid_roles:
            raise ValueError(f'角色必须是以下之一: {", ".join(valid_roles)}')
        return v
    
    @validator('message_type')
    def validate_message_type(cls, v):
        """验证消息类型"""
        valid_types = ['text', 'image', 'file', 'audio', 'video', 'multimodal']
        if v not in valid_types:
            raise ValueError(f'消息类型必须是以下之一: {", ".join(valid_types)}')
        return v


class MessageCreate(MessageBase):
    """消息创建模型"""
    thread_id: UUID = Field(..., description="线程ID")
    parent_id: Optional[UUID] = Field(None, description="父消息ID")
    reply_to_id: Optional[UUID] = Field(None, description="回复消息ID")


class MessageUpdate(BaseModel):
    """消息更新模型"""
    content: Optional[str] = Field(None, min_length=1)
    content_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    is_pinned: Optional[bool] = None


class MessageResponse(MessageBase):
    """消息响应模型"""
    id: UUID
    thread_id: UUID
    user_id: UUID
    parent_id: Optional[UUID]
    reply_to_id: Optional[UUID]
    is_edited: bool
    is_deleted: bool
    is_pinned: bool
    token_count: int = 0
    character_count: int = 0
    edit_count: int = 0
    edited_at: Optional[datetime]
    deleted_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class MessageList(BaseModel):
    """消息列表模型"""
    messages: List[MessageResponse]
    total: int
    has_more: bool = False
    next_cursor: Optional[str] = None
    prev_cursor: Optional[str] = None


# ============================================================================
# 工作流相关模型
# ============================================================================

class WorkflowBase(BaseModel):
    """工作流基础模型"""
    name: str = Field(..., min_length=1, max_length=200, description="工作流名称")
    description: Optional[str] = Field(None, max_length=1000, description="工作流描述")
    version: str = Field("1.0.0", description="版本号")
    definition: Dict[str, Any] = Field(..., description="工作流定义")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="配置信息")
    tags: Optional[List[str]] = Field(default_factory=list, description="标签")
    category: Optional[str] = Field(None, max_length=50, description="分类")


class WorkflowCreate(WorkflowBase):
    """工作流创建模型"""
    is_public: bool = Field(False, description="是否公开")
    is_template: bool = Field(False, description="是否为模板")


class WorkflowUpdate(BaseModel):
    """工作流更新模型"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    version: Optional[str] = None
    definition: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    is_public: Optional[bool] = None
    is_template: Optional[bool] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None


class WorkflowResponse(WorkflowBase):
    """工作流响应模型"""
    id: UUID
    user_id: UUID
    status: str
    is_public: bool
    is_template: bool
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class WorkflowExecutionCreate(BaseModel):
    """工作流执行创建模型"""
    workflow_id: UUID = Field(..., description="工作流ID")
    input_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="输入数据")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="执行上下文")


class WorkflowExecutionResponse(BaseModel):
    """工作流执行响应模型"""
    id: UUID
    workflow_id: UUID
    user_id: UUID
    status: str
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration: Optional[float]
    error_message: Optional[str]
    error_details: Optional[Dict[str, Any]]
    step_count: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# 记忆相关模型
# ============================================================================

class MemoryBase(BaseModel):
    """记忆基础模型"""
    memory_type: str = Field(..., description="记忆类型")
    title: Optional[str] = Field(None, max_length=200, description="记忆标题")
    content: str = Field(..., min_length=1, description="记忆内容")
    summary: Optional[str] = Field(None, max_length=1000, description="记忆摘要")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="结构化数据")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="重要性评分")
    weight: float = Field(1.0, ge=0.0, description="权重")
    tags: Optional[List[str]] = Field(default_factory=list, description="标签")
    category: Optional[str] = Field(None, max_length=50, description="分类")
    
    @validator('memory_type')
    def validate_memory_type(cls, v):
        """验证记忆类型"""
        valid_types = ['semantic', 'episodic', 'procedural', 'working']
        if v not in valid_types:
            raise ValueError(f'记忆类型必须是以下之一: {", ".join(valid_types)}')
        return v


class MemoryCreate(MemoryBase):
    """记忆创建模型"""
    thread_id: Optional[UUID] = Field(None, description="关联线程ID")


class MemoryUpdate(BaseModel):
    """记忆更新模型"""
    title: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    summary: Optional[str] = Field(None, max_length=1000)
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    weight: Optional[float] = Field(None, ge=0.0)
    tags: Optional[List[str]] = None
    category: Optional[str] = None


class MemoryResponse(MemoryBase):
    """记忆响应模型"""
    id: UUID
    user_id: UUID
    thread_id: Optional[UUID]
    accessed_at: datetime
    access_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class MemorySearch(BaseModel):
    """记忆搜索模型"""
    query: str = Field(..., min_length=1, description="搜索查询")
    memory_types: Optional[List[str]] = Field(None, description="记忆类型过滤")
    categories: Optional[List[str]] = Field(None, description="分类过滤")
    tags: Optional[List[str]] = Field(None, description="标签过滤")
    min_importance: Optional[float] = Field(None, ge=0.0, le=1.0, description="最小重要性")
    limit: int = Field(10, ge=1, le=100, description="返回数量限制")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="相似度阈值")


# ============================================================================
# 时间旅行相关模型
# ============================================================================

class TimeTravelBase(BaseModel):
    """时间旅行基础模型"""
    snapshot_name: str = Field(..., min_length=1, max_length=200, description="快照名称")
    description: Optional[str] = Field(None, max_length=1000, description="快照描述")
    snapshot_type: str = Field("manual", description="快照类型")
    
    @validator('snapshot_type')
    def validate_snapshot_type(cls, v):
        """验证快照类型"""
        valid_types = ['manual', 'auto', 'checkpoint']
        if v not in valid_types:
            raise ValueError(f'快照类型必须是以下之一: {", ".join(valid_types)}')
        return v


class TimeTravelCreate(TimeTravelBase):
    """时间旅行创建模型"""
    thread_id: Optional[UUID] = Field(None, description="关联线程ID")
    snapshot_data: Dict[str, Any] = Field(..., description="快照数据")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")


class TimeTravelResponse(TimeTravelBase):
    """时间旅行响应模型"""
    id: UUID
    user_id: UUID
    thread_id: Optional[UUID]
    snapshot_data: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    is_active: bool
    data_size: int = 0
    restore_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class TimeTravelRestore(BaseModel):
    """时间旅行恢复模型"""
    snapshot_id: UUID = Field(..., description="快照ID")
    restore_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="恢复选项")


# ============================================================================
# 附件相关模型
# ============================================================================

class AttachmentResponse(BaseModel):
    """附件响应模型"""
    id: UUID
    user_id: UUID
    message_id: Optional[UUID]
    filename: str
    original_filename: str
    file_size: int
    file_type: str
    mime_type: str
    file_hash: str
    is_public: bool
    is_processed: bool
    metadata: Dict[str, Any] = {}
    download_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# 统计和分析模型
# ============================================================================

class UserStats(BaseModel):
    """用户统计模型"""
    total_users: int = 0
    active_users: int = 0
    new_users_today: int = 0
    new_users_week: int = 0
    new_users_month: int = 0
    verified_users: int = 0
    admin_users: int = 0


class MessageStats(BaseModel):
    """消息统计模型"""
    total_messages: int = 0
    messages_today: int = 0
    messages_week: int = 0
    messages_month: int = 0
    avg_messages_per_user: float = 0.0
    avg_message_length: float = 0.0
    total_tokens: int = 0


class ThreadStats(BaseModel):
    """线程统计模型"""
    total_threads: int = 0
    active_threads: int = 0
    archived_threads: int = 0
    threads_today: int = 0
    threads_week: int = 0
    threads_month: int = 0
    avg_messages_per_thread: float = 0.0


class WorkflowStats(BaseModel):
    """工作流统计模型"""
    total_workflows: int = 0
    active_workflows: int = 0
    public_workflows: int = 0
    template_workflows: int = 0
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float = 0.0


class SystemStats(BaseModel):
    """系统统计模型"""
    user_stats: UserStats
    message_stats: MessageStats
    thread_stats: ThreadStats
    workflow_stats: WorkflowStats
    system_uptime: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    disk_usage: float = 0.0


# ============================================================================
# WebSocket消息模型
# ============================================================================

class WebSocketMessage(BaseModel):
    """WebSocket消息模型"""
    type: str = Field(..., description="消息类型")
    data: Any = Field(..., description="消息数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatMessage(BaseModel):
    """聊天消息模型"""
    thread_id: UUID
    message: MessageCreate
    stream: bool = Field(False, description="是否流式响应")
    
    class Config:
        json_encoders = {
            UUID: str
        }


class ChatResponse(BaseModel):
    """聊天响应模型"""
    message_id: UUID
    content: str
    is_complete: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            UUID: str
        }


# ============================================================================
# 健康检查和系统信息模型
# ============================================================================

class HealthCheck(BaseModel):
    """健康检查模型"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    uptime: float = 0.0
    checks: Dict[str, Any] = {}
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemInfo(BaseModel):
    """系统信息模型"""
    name: str = "LangGraph Study API"
    version: str = "1.0.0"
    description: str = "LangGraph学习平台后端API"
    environment: str = "development"
    python_version: str
    dependencies: Dict[str, str] = {}
    features: List[str] = []
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }