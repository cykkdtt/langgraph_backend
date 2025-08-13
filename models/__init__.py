"""
API数据模型

定义API接口中使用的请求和响应模型。
"""

from .base_models import BaseResponse, PaginatedResponse
from .chat_models import ChatRequest, ChatResponse, StreamChunk, ThreadInfo
from .agent_models import (
    AgentInstanceRequest, AgentInstanceResponse, 
    CreateAgentRequest, AgentTypeInfo
)
from .memory_models import (
    MemoryManageRequest, MemorySearchRequest, 
    MemoryItem, MemorySearchResponse
)
from .rag_models import (
    DocumentModel, DocumentUploadRequest, DocumentSearchRequest,
    RetrievalConfig, VectorSearchRequest
)

__all__ = [
    # 基础模型
    "BaseResponse",
    "PaginatedResponse",
    
    # 聊天模型
    "ChatRequest",
    "ChatResponse", 
    "StreamChunk",
    "ThreadInfo",
    
    # 智能体模型
    "AgentInstanceRequest",
    "AgentInstanceResponse",
    "CreateAgentRequest", 
    "AgentTypeInfo",
    
    # 记忆模型
    "MemoryManageRequest",
    "MemorySearchRequest",
    "MemoryItem",
    "MemorySearchResponse",
    
    # RAG模型
    "DocumentModel",
    "DocumentUploadRequest",
    "DocumentSearchRequest",
    "RetrievalConfig",
    "VectorSearchRequest"
]