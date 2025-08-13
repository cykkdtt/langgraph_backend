# 核心模块初始化文件
"""
多智能体LangGraph项目 - 核心模块

本模块包含项目的核心功能组件：
- 智能体管理和协作
- 工具管理和执行
- 记忆管理
- 流式处理
- 中断处理
- 工作流编排
- 时间旅行功能
"""

# 智能体相关
from .agents import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentCapability,
    AgentMetadata,
    ChatRequest,
    ChatResponse,
    StreamChunk,
    MemoryEnhancedAgent,
    AgentConfig,
    AgentInstance,
    AgentRegistry,
    AgentFactory,
    AgentPerformanceMetrics,
    AgentManager,
    CollaborationMode,
    MessageType,
    CollaborationMessage,
    CollaborationTask,
    CollaborationContext,
    AgentCollaborationOrchestrator,
    TaskScheduler,
    LoadBalancer,
    get_agent_registry,
    get_agent_factory,
    get_agent_manager,
    get_collaboration_orchestrator,
    initialize_agent_system,
    initialize_agent_manager
)

# 工具相关
from .tools import (
    ToolCategory,
    ToolPermission,
    ToolMetadata,
    ToolExecutionResult,
    ToolExecutionContext,
    BaseManagedTool,
    ToolRegistry,
    get_tool_registry,
    managed_tool
)

# 尝试导入可选组件
try:
    from .tools import (
        MCPManager,
        get_mcp_manager,
        initialize_mcp_manager
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

try:
    from .tools import (
        ToolExecutionMode,
        ToolValidationLevel,
        EnhancedToolExecutionContext,
        EnhancedToolExecutionResult,
        ToolValidator,
        EnhancedToolManager,
        get_enhanced_tool_manager
    )
    ENHANCED_TOOL_MANAGER_AVAILABLE = True
except ImportError:
    ENHANCED_TOOL_MANAGER_AVAILABLE = False

# 记忆管理
from .memory import (
    MemoryType,
    MemoryScope,
    MemoryItem,
    MemoryQuery,
    MemoryNamespace,
    LangMemManager
)

# 流式处理
from .streaming import (
    StreamManager,
    StreamMode,
    StreamEvent,
    StreamChunk as StreamingChunk
)

# 中断处理
from .interrupts import (
    InterruptType,
    InterruptStatus,
    InterruptPriority,
    InterruptRequest,
    InterruptResponse,
    InterruptContext,
    EnhancedInterruptManager
)

# 工作流编排
from .workflows import (
    WorkflowBuilder,
    WorkflowDefinition,
    WorkflowStep,
    Condition
)

# 时间旅行
from .time_travel import (
    TimeTravelManager,
    CheckpointManager,
    RollbackManager,
    StateHistoryManager
)

# 缓存管理
from .cache import (
    RedisManager,
    SessionCache,
    CacheManager,
    get_redis_manager,
    get_session_cache,
    get_cache_manager
)

# 检查点管理
from .checkpoint import (
    CheckpointMetadata,
    CheckpointInfo,
    CheckpointManager as CoreCheckpointManager,
    get_checkpoint_manager,
    checkpoint_manager_context
)

# 数据库管理
from .database import (
    DatabaseManager,
    database_manager,
    get_database_manager,
    initialize_database,
    get_postgres_connection,
    get_redis_connection,
    get_async_session
)

# 错误处理
from .error import (
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    ErrorInfo,
    BaseError,
    SystemError,
    ConfigurationError,
    DatabaseError,
    ConnectionError,
    APIError,
    AuthenticationError,
    AuthorizationError,
    ErrorHandler,
    get_error_handler,
    handle_errors,
    handle_async_errors,
    PerformanceMetric,
    PerformanceMonitor,
    get_performance_monitor,
    monitor_performance,
    monitor_async_performance
)

# 日志管理
from .logging import (
    StructuredFormatter,
    LoggerManager,
    logger_manager,
    get_logger,
    get_context_logger,
    initialize_logging
)

# 导出列表
__all__ = [
    # 智能体相关
    "BaseAgent",
    "AgentType",
    "AgentStatus", 
    "AgentCapability",
    "AgentMetadata",
    "ChatRequest",
    "ChatResponse",
    "StreamChunk",
    "MemoryEnhancedAgent",
    "AgentConfig",
    "AgentInstance",
    "AgentRegistry",
    "AgentFactory",
    "AgentPerformanceMetrics",
    "AgentManager",
    "CollaborationMode",
    "MessageType",
    "CollaborationMessage",
    "CollaborationTask",
    "CollaborationContext",
    "AgentCollaborationOrchestrator",
    "TaskScheduler",
    "LoadBalancer",
    "get_agent_registry",
    "get_agent_factory",
    "get_agent_manager",
    "get_collaboration_orchestrator",
    "initialize_agent_system",
    "initialize_agent_manager",
    
    # 工具相关
    "ToolCategory",
    "ToolPermission",
    "ToolMetadata",
    "ToolExecutionResult",
    "ToolExecutionContext",
    "BaseManagedTool",
    "ToolRegistry",
    "get_tool_registry",
    "managed_tool",
    
    # 记忆管理
    "MemoryType",
    "MemoryScope",
    "MemoryItem",
    "MemoryQuery",
    "MemoryNamespace",
    "LangMemManager",
    
    # 流式处理
    "StreamManager",
    "StreamMode",
    "StreamEvent",
    "StreamingChunk",
    
    # 中断处理
    "InterruptType",
    "InterruptStatus",
    "InterruptPriority",
    "InterruptRequest",
    "InterruptResponse",
    "InterruptContext",
    "EnhancedInterruptManager",
    
    # 工作流编排
    "WorkflowBuilder",
    "WorkflowDefinition",
    "WorkflowStep",
    "Condition",
    
    # 时间旅行
    "TimeTravelManager",
    "CheckpointManager",
    "RollbackManager",
    "StateHistoryManager",
    
    # 缓存管理
    "RedisManager",
    "SessionCache", 
    "CacheManager",
    "get_redis_manager",
    "get_session_cache",
    "get_cache_manager",
    
    # 检查点管理
    "CheckpointMetadata",
    "CheckpointInfo",
    "CoreCheckpointManager",
    "get_checkpoint_manager",
    "checkpoint_manager_context",
    
    # 数据库管理
    "DatabaseManager",
    "database_manager",
    "get_database_manager",
    "initialize_database",
    "get_postgres_connection",
    "get_redis_connection",
    "get_async_session",
    
    # 错误处理
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    "ErrorInfo",
    "BaseError",
    "SystemError",
    "ConfigurationError",
    "DatabaseError",
    "ConnectionError",
    "APIError",
    "AuthenticationError",
    "AuthorizationError",
    "ErrorHandler",
    "get_error_handler",
    "handle_errors",
    "handle_async_errors",
    "PerformanceMetric",
    "PerformanceMonitor",
    "get_performance_monitor",
    "monitor_performance",
    "monitor_async_performance",
    
    # 日志管理
    "StructuredFormatter",
    "LoggerManager",
    "logger_manager",
    "get_logger",
    "get_context_logger",
    "initialize_logging",
]

# 根据可用性动态添加可选组件
if MCP_AVAILABLE:
    __all__.extend([
        "MCPManager",
        "get_mcp_manager",
        "initialize_mcp_manager"
    ])

if ENHANCED_TOOL_MANAGER_AVAILABLE:
    __all__.extend([
        "ToolExecutionMode",
        "ToolValidationLevel",
        "EnhancedToolExecutionContext",
        "EnhancedToolExecutionResult",
        "ToolValidator",
        "EnhancedToolManager",
        "get_enhanced_tool_manager"
    ])