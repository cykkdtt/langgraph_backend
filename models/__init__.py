"""数据库模型模块

包含所有数据库模型定义、API响应模型和数据转换器，提供优化的数据库结构和索引。"""

# 导入数据库模型
from .auth_models import UserStatus
from .chat_models import MessageRole, MessageType
from .workflow_models import WorkflowStatus
from .memory_models import MemoryType
from .database_models import (
    
    # 数据库模型
    User,
    Session,
    Message,
    ToolCall,
    AgentState,
    SystemLog,
    Workflow,
    WorkflowExecution,
    Memory
)

# 导入API模型
from .api import (
    # 基础响应模型
    BaseResponse,
    DataResponse,
    ListResponse,
    ErrorResponse,
    ResponseStatus,
    
    # 查询和分页模型
    PaginationParams,
    SortParams,
    FilterParams,
    QueryParams,
    SortOrder,
    
    # 用户相关模型
    UserBase,
    UserCreate,
    UserUpdate,
    UserResponse,
    UserLogin,
    UserLoginResponse,
    PasswordChange,
    PasswordReset,
    PasswordResetConfirm,
    
    # 会话相关模型
    SessionBase,
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    
    # 线程相关模型
    ThreadBase,
    ThreadCreate,
    ThreadUpdate,
    ThreadResponse,
    
    # 消息相关模型
    MessageBase,
    MessageCreate,
    MessageUpdate,
    MessageResponse,
    MessageList,
    
    # 工作流相关模型
    WorkflowBase,
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowResponse,
    WorkflowExecutionCreate,
    WorkflowExecutionResponse,
    
    # 记忆相关模型
    MemoryBase,
    MemoryCreate,
    MemoryUpdate,
    MemoryResponse,
    MemorySearch,
    
    # 时间旅行相关模型
    TimeTravelBase,
    TimeTravelCreate,
    TimeTravelResponse,
    TimeTravelRestore,
    
    # 附件相关模型
    AttachmentResponse,
    
    # 统计模型
    UserStats,
    MessageStats,
    ThreadStats,
    WorkflowStats,
    SystemStats,
    
    # WebSocket模型
    WebSocketMessage,
    ChatMessage,
    ChatResponse,
    
    # 系统模型
    HealthCheck,
    SystemInfo
)

# 导入转换器
from .converters import (
    # 转换器类
    BaseConverter,
    UserConverter,
    SessionConverter,
    # ThreadConverter,  # 已注释
    MessageConverter,
    WorkflowConverter,
    WorkflowExecutionConverter,
    MemoryConverter,
    # TimeTravelConverter,  # 已注释
    # AttachmentConverter,  # 已注释
    ConverterRegistry
)

# 便捷函数从ConverterRegistry导入
convert_to_response = ConverterRegistry.convert_to_response
convert_to_response_list = ConverterRegistry.convert_to_response_list

__all__ = [
    # 枚举类型
    "UserStatus",
    "MessageRole",
    "MessageType",
    "WorkflowStatus",
    "MemoryType",
    
    # 数据库模型
    "User",
    "Session",
    "Message",
    "ToolCall",
    "AgentState",
    "SystemLog",
    "Workflow",
    "WorkflowExecution",
    "Memory",
    
    # 基础响应模型
    "BaseResponse",
    "DataResponse",
    "ListResponse",
    "ErrorResponse",
    "ResponseStatus",
    
    # 查询和分页模型
    "PaginationParams",
    "SortParams",
    "FilterParams",
    "QueryParams",
    "SortOrder",
    
    # 用户相关模型
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserLogin",
    "UserLoginResponse",
    "PasswordChange",
    "PasswordReset",
    "PasswordResetConfirm",
    
    # 会话相关模型
    "SessionBase",
    "SessionCreate",
    "SessionUpdate",
    "SessionResponse",
    
    # 线程相关模型
    "ThreadBase",
    "ThreadCreate",
    "ThreadUpdate",
    "ThreadResponse",
    
    # 消息相关模型
    "MessageBase",
    "MessageCreate",
    "MessageUpdate",
    "MessageResponse",
    "MessageList",
    
    # 工作流相关模型
    "WorkflowBase",
    "WorkflowCreate",
    "WorkflowUpdate",
    "WorkflowResponse",
    "WorkflowExecutionCreate",
    "WorkflowExecutionResponse",
    
    # 记忆相关模型
    "MemoryBase",
    "MemoryCreate",
    "MemoryUpdate",
    "MemoryResponse",
    "MemorySearch",
    
    # 时间旅行相关模型
    "TimeTravelBase",
    "TimeTravelCreate",
    "TimeTravelResponse",
    "TimeTravelRestore",
    
    # 附件相关模型
    "AttachmentResponse",
    
    # 统计模型
    "UserStats",
    "MessageStats",
    "ThreadStats",
    "WorkflowStats",
    "SystemStats",
    
    # WebSocket模型
    "WebSocketMessage",
    "ChatMessage",
    "ChatResponse",
    
    # 系统模型
    "HealthCheck",
    "SystemInfo",
    
    # 转换器类
    "BaseConverter",
    "UserConverter",
    "SessionConverter",
    # "ThreadConverter",  # 已注释
    "MessageConverter",
    "WorkflowConverter",
    "WorkflowExecutionConverter",
    "MemoryConverter",
    # "TimeTravelConverter",  # 已注释
    # "AttachmentConverter",  # 已注释
    "ConverterRegistry",
    
    # 便捷函数
    "convert_to_response",
    "convert_to_response_list"
]