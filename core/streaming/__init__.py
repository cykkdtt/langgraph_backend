"""
流式处理模块

提供增强的流式处理功能，支持：
- 多种流式模式
- 实时事件处理
- 流式状态管理
- 错误处理和恢复
- LangGraph官方流式处理集成
"""

from .stream_manager import StreamManager
from .stream_types import StreamChunk, StreamEvent, StreamMode, StreamEventType, StreamConfig
from .sse_handler import SSEHandler
from .websocket_handler import WebSocketHandler
from .langgraph_adapter import LangGraphStreamAdapter, StreamWriterIntegration

# 全局流式管理器实例
_streaming_manager = None

def get_streaming_manager() -> StreamManager:
    """获取全局流式管理器实例"""
    global _streaming_manager
    if _streaming_manager is None:
        _streaming_manager = StreamManager()
    return _streaming_manager

__all__ = [
    "StreamManager",
    "StreamChunk", 
    "StreamEvent",
    "StreamMode",
    "StreamEventType",
    "StreamConfig",
    "SSEHandler",
    "WebSocketHandler",
    "LangGraphStreamAdapter",
    "StreamWriterIntegration",
    "get_streaming_manager"
]