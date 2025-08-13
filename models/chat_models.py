"""
聊天相关数据模型

定义聊天API的请求和响应模型。
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from core.streaming.stream_types import StreamMode


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageType(str, Enum):
    """消息类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class MessageContent(BaseModel):
    """消息内容"""
    type: MessageType = Field(description="内容类型")
    text: Optional[str] = Field(None, description="文本内容")
    image_url: Optional[str] = Field(None, description="图片URL")
    file_url: Optional[str] = Field(None, description="文件URL")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class ToolCall(BaseModel):
    """工具调用"""
    id: str = Field(description="工具调用ID")
    name: str = Field(description="工具名称")
    arguments: Dict[str, Any] = Field(description="工具参数")


class ToolResult(BaseModel):
    """工具结果"""
    tool_call_id: str = Field(description="工具调用ID")
    result: Any = Field(description="工具执行结果")
    error: Optional[str] = Field(None, description="错误信息")


class Message(BaseModel):
    """消息模型"""
    id: Optional[str] = Field(None, description="消息ID")
    role: MessageRole = Field(description="消息角色")
    content: Union[str, List[MessageContent]] = Field(description="消息内容")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="工具调用")
    tool_results: Optional[List[ToolResult]] = Field(None, description="工具结果")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="消息时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(description="用户消息")
    thread_id: Optional[str] = Field(None, description="会话线程ID")
    agent_id: str = Field(description="智能体ID")
    stream: bool = Field(False, description="是否流式响应")
    stream_mode: StreamMode = Field(StreamMode.VALUES, description="流式模式")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(None, ge=1, description="最大token数")
    tools: Optional[List[str]] = Field(None, description="可用工具列表")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class ChatResponse(BaseModel):
    """聊天响应"""
    message: Message = Field(description="助手回复")
    thread_id: str = Field(description="会话线程ID")
    agent_id: str = Field(description="智能体ID")
    usage: Optional[Dict[str, int]] = Field(None, description="token使用情况")
    finish_reason: Optional[str] = Field(None, description="结束原因")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class StreamChunk(BaseModel):
    """流式响应块"""
    id: str = Field(description="块ID")
    type: str = Field(description="块类型")
    content: str = Field(description="内容")
    delta: Optional[str] = Field(None, description="增量内容")
    finish_reason: Optional[str] = Field(None, description="结束原因")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class ThreadInfo(BaseModel):
    """会话线程信息"""
    thread_id: str = Field(description="线程ID")
    title: Optional[str] = Field(None, description="线程标题")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    message_count: int = Field(description="消息数量")
    agent_id: str = Field(description="智能体ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class CreateThreadRequest(BaseModel):
    """创建线程请求"""
    title: Optional[str] = Field(None, description="线程标题")
    agent_id: str = Field(description="智能体ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class UpdateThreadRequest(BaseModel):
    """更新线程请求"""
    title: Optional[str] = Field(None, description="线程标题")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class ThreadListRequest(BaseModel):
    """线程列表请求"""
    agent_id: Optional[str] = Field(None, description="智能体ID")
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页大小")
    search: Optional[str] = Field(None, description="搜索关键词")


class MessageListRequest(BaseModel):
    """消息列表请求"""
    thread_id: str = Field(description="线程ID")
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(50, ge=1, le=100, description="每页大小")
    before: Optional[str] = Field(None, description="在此消息之前")
    after: Optional[str] = Field(None, description="在此消息之后")