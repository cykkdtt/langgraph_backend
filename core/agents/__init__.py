# 智能体模块初始化文件
"""多智能体LangGraph项目 - 智能体模块

本模块包含所有智能体相关的类和功能：
- BaseAgent: 智能体抽象基类
- 各种具体的智能体实现
- 智能体管理和注册功能
"""

from .base import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentCapability,
    AgentMetadata,
    ChatRequest,
    ChatResponse,
    StreamChunk
)

from .memory_enhanced import MemoryEnhancedAgent

from .registry import (
    AgentConfig,
    AgentInstance,
    AgentRegistry,
    AgentFactory,
    get_agent_registry,
    get_agent_factory,
    initialize_agent_system
)

from .manager import (
    AgentPerformanceMetrics,
    AgentManager,
    get_agent_manager,
    initialize_agent_manager
)

# 协作优化器
from .collaboration_optimizer import (
    CollaborationMode,
    MessageType,
    CollaborationMessage,
    CollaborationTask,
    CollaborationContext,
    AgentCollaborationOrchestrator,
    TaskScheduler,
    LoadBalancer,
    get_collaboration_orchestrator
)

# 暂时注释掉，因为这些类还没有实现
# from .collaborative import SupervisorAgent, ResearchAgent, ChartAgent
# from .rag import RAGAgent
# from .specialized import CodeAgent, AnalysisAgent

__all__ = [
    # 基础类
    "BaseAgent",
    "AgentType", 
    "AgentStatus",
    "AgentCapability",
    "AgentMetadata",
    "ChatRequest",
    "ChatResponse", 
    "StreamChunk",
    
    # 具体实现
    "MemoryEnhancedAgent",
    
    # 注册表和工厂
    "AgentConfig",
    "AgentInstance", 
    "AgentRegistry",
    "AgentFactory",
    "get_agent_registry",
    "get_agent_factory",
    "initialize_agent_system",
    
    # 管理器
    "AgentPerformanceMetrics",
    "AgentManager",
    "get_agent_manager", 
    "initialize_agent_manager",
    
    # 协作优化器
    "CollaborationMode",
    "MessageType",
    "CollaborationMessage",
    "CollaborationTask",
    "CollaborationContext",
    "AgentCollaborationOrchestrator",
    "TaskScheduler",
    "LoadBalancer",
    "get_collaboration_orchestrator",
    
    # 暂时注释掉
    # "SupervisorAgent",
    # "ResearchAgent", 
    # "ChartAgent",
    # "RAGAgent",
    # "CodeAgent",
    # "AnalysisAgent"
]