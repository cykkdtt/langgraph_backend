"""
流式处理数据类型定义

定义流式处理中使用的数据结构和枚举类型。
"""

from enum import Enum
from typing import Any, Dict, Optional, Union, List
from pydantic import BaseModel
from datetime import datetime


class StreamMode(str, Enum):
    """流式模式枚举"""
    VALUES = "values"           # 节点输出值
    EVENTS = "events"           # 事件流
    UPDATES = "updates"         # 状态更新
    MESSAGES = "messages"       # 消息流
    DEBUG = "debug"             # 调试信息
    ALL = "all"                 # 所有模式


class StreamEventType(str, Enum):
    """流式事件类型"""
    NODE_START = "node_start"           # 节点开始执行
    NODE_END = "node_end"               # 节点执行完成
    NODE_ERROR = "node_error"           # 节点执行错误
    TOOL_START = "tool_start"           # 工具调用开始
    TOOL_END = "tool_end"               # 工具调用完成
    TOOL_ERROR = "tool_error"           # 工具调用错误
    MESSAGE_CHUNK = "message_chunk"     # 消息块
    STATE_UPDATE = "state_update"       # 状态更新
    INTERRUPT = "interrupt"             # 中断事件
    RESUME = "resume"                   # 恢复事件
    COMPLETE = "complete"               # 完成事件
    ERROR = "error"                     # 错误事件
    CUSTOM_EVENT = "custom_event"       # 自定义事件
    PROGRESS_UPDATE = "progress_update" # 进度更新
    DEBUG = "debug"                     # 调试信息


class StreamChunk(BaseModel):
    """流式响应块"""
    chunk_type: StreamEventType
    content: str
    metadata: Dict[str, Any] = {}
    timestamp: datetime = None
    node_id: Optional[str] = None
    run_id: Optional[str] = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)


class StreamEvent(BaseModel):
    """流式事件"""
    event_type: StreamEventType
    event_id: str
    run_id: str
    node_id: Optional[str] = None
    data: Dict[str, Any] = {}
    timestamp: datetime = None
    parent_event_id: Optional[str] = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)


class StreamState(BaseModel):
    """流式状态"""
    run_id: str
    status: str = "running"
    current_node: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
    started_at: datetime = None
    updated_at: datetime = None
    
    def __init__(self, **data):
        now = datetime.utcnow()
        if 'started_at' not in data:
            data['started_at'] = now
        if 'updated_at' not in data:
            data['updated_at'] = now
        super().__init__(**data)


class InterruptRequest(BaseModel):
    """中断请求"""
    run_id: str
    node_id: str
    interrupt_type: str = "human_input"
    message: str
    context: Dict[str, Any] = {}
    timeout: Optional[int] = None  # 超时时间（秒）
    created_at: datetime = None
    
    def __init__(self, **data):
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        super().__init__(**data)


class InterruptResponse(BaseModel):
    """中断响应"""
    request_id: str
    response_data: Dict[str, Any]
    approved: bool = True
    message: Optional[str] = None
    responded_at: datetime = None
    
    def __init__(self, **data):
        if 'responded_at' not in data:
            data['responded_at'] = datetime.utcnow()
        super().__init__(**data)


class StreamConfig(BaseModel):
    """流式配置"""
    modes: List[StreamMode] = [StreamMode.VALUES]
    include_events: List[StreamEventType] = []
    exclude_events: List[StreamEventType] = []
    buffer_size: int = 100
    timeout: int = 30
    enable_compression: bool = False
    enable_heartbeat: bool = True
    heartbeat_interval: int = 30  # 秒