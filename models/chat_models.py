"""聊天相关的数据模型

本模块定义了聊天API和WebSocket通信所需的数据模型，包括：
- 聊天请求和响应模型
- WebSocket消息模型
- 流式响应模型
- 智能体选择和配置模型
"""

from typing import Dict, List, Optional, Any, Union, Literal, Set
from pydantic import BaseModel, Field, validator, model_validator, constr, conint, confloat
from datetime import datetime, timezone
from enum import Enum
import uuid
import re
import json
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ipaddress import ip_address, AddressValueError

# 导入序列化混入类
try:
    from .api_models import SerializationMixin, SerializationConfig, default_serialization_config
except ImportError:
    # 如果无法导入，定义简化版本
    class SerializationMixin:
        pass
    
    @dataclass
    class SerializationConfig:
        enable_cache: bool = True
        cache_size: int = 1000
        enable_compression: bool = False
        thread_pool_workers: int = 4
        async_threshold: int = 100
    
    default_serialization_config = SerializationConfig()


class MessageRole(str, Enum):
    """消息角色枚举"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageType(str, Enum):
    """消息类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM_INFO = "system_info"


class ChatMode(str, Enum):
    """聊天模式枚举"""
    SINGLE_AGENT = "single_agent"  # 单智能体模式
    MULTI_AGENT = "multi_agent"    # 多智能体协作模式
    WORKFLOW = "workflow"          # 工作流模式
    AUTO_SELECT = "auto_select"    # 自动选择智能体


class StreamEventType(str, Enum):
    """流式事件类型枚举"""
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_END = "message_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    AGENT_SWITCH = "agent_switch"
    ERROR = "error"
    SYSTEM_INFO = "system_info"


class WebSocketMessageType(str, Enum):
    """WebSocket消息类型枚举"""
    CHAT_MESSAGE = "chat_message"
    STREAM_CHUNK = "stream_chunk"
    AGENT_STATUS = "agent_status"
    SESSION_UPDATE = "session_update"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    HEARTBEAT = "heartbeat"
    RECONNECT = "reconnect"
    RATE_LIMIT = "rate_limit"
    MEMORY_SAVED = "memory_saved"


class MessagePriority(str, Enum):
    """消息优先级枚举"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AgentCapability(str, Enum):
    """智能体能力枚举"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    IMAGE_ANALYSIS = "image_analysis"
    FILE_PROCESSING = "file_processing"
    WEB_SEARCH = "web_search"
    TOOL_CALLING = "tool_calling"
    WORKFLOW_EXECUTION = "workflow_execution"


class SessionStatus(str, Enum):
    """会话状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class ToolCall(BaseModel, SerializationMixin):
    """工具调用模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: constr(min_length=1, max_length=100, pattern=r'^[a-zA-Z][a-zA-Z0-9_]*$') = Field(..., description="工具名称")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="工具参数")
    result: Optional[Any] = Field(None, description="工具执行结果")
    status: Literal["pending", "running", "completed", "failed"] = Field(default="pending")
    error: Optional[constr(max_length=1000)] = Field(None, description="错误信息")
    execution_time: Optional[confloat(ge=0.0)] = Field(None, description="执行时间（秒）")
    retry_count: conint(ge=0, le=5) = Field(default=0, description="重试次数")
    timeout: Optional[confloat(gt=0.0, le=300.0)] = Field(None, description="超时时间（秒）")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="优先级")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = Field(None, description="开始执行时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    
    @validator('arguments')
    def validate_arguments(cls, v):
        """验证工具参数"""
        if len(str(v)) > 10000:  # 限制参数大小
            raise ValueError("工具参数过大")
        return v
    
    @validator('result')
    def validate_result(cls, v):
        """验证工具结果"""
        if v is not None and len(str(v)) > 50000:  # 限制结果大小
            raise ValueError("工具执行结果过大")
        return v
    
    @model_validator(mode='before')
    @classmethod
    def validate_status_consistency(cls, values):
        """验证状态一致性"""
        status = values.get('status')
        error = values.get('error')
        result = values.get('result')
        started_at = values.get('started_at')
        completed_at = values.get('completed_at')
        
        # 失败状态必须有错误信息
        if status == 'failed' and not error:
            raise ValueError("失败状态必须提供错误信息")
        
        # 成功状态不应该有错误信息
        if status == 'completed' and error:
            raise ValueError("完成状态不应该有错误信息")
        
        # 时间一致性验证
        if started_at and completed_at and started_at > completed_at:
            raise ValueError("开始时间不能晚于完成时间")
        
        # 运行中状态应该有开始时间
        if status in ['running', 'completed', 'failed'] and not started_at:
            values['started_at'] = datetime.now(timezone.utc)
        
        # 完成状态应该有完成时间
        if status in ['completed', 'failed'] and not completed_at:
            values['completed_at'] = datetime.now(timezone.utc)
        
        return values


class ChatMessage(BaseModel, SerializationMixin):
    """聊天消息模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = Field(..., description="消息角色")
    content: constr(min_length=1, max_length=50000) = Field(..., description="消息内容")
    message_type: MessageType = Field(default=MessageType.TEXT, description="消息类型")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="消息元数据")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="工具调用列表")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="附件列表")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="消息优先级")
    thread_id: Optional[str] = Field(None, description="线程ID")
    parent_id: Optional[str] = Field(None, description="父消息ID")
    edit_count: conint(ge=0) = Field(default=0, description="编辑次数")
    token_count: Optional[conint(ge=0)] = Field(None, description="令牌数量")
    language: Optional[constr(min_length=2, max_length=10)] = Field(None, description="语言代码")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = Field(None, description="过期时间")

    @validator('content')
    def validate_content(cls, v):
        """验证消息内容"""
        if not v or not v.strip():
            raise ValueError("消息内容不能为空")
        # 检查是否包含恶意内容（简单示例）
        forbidden_patterns = [r'<script.*?>', r'javascript:', r'data:text/html']
        for pattern in forbidden_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("消息内容包含不允许的脚本")
        return v.strip()
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 5000:  # 限制元数据大小
            raise ValueError("元数据过大")
        return v
    
    @validator('tool_calls')
    def validate_tool_calls(cls, v):
        """验证工具调用列表"""
        if len(v) > 10:  # 限制工具调用数量
            raise ValueError("工具调用数量过多")
        return v
    
    @validator('attachments')
    def validate_attachments(cls, v):
        """验证附件列表"""
        if len(v) > 20:  # 限制附件数量
            raise ValueError("附件数量过多")
        for attachment in v:
            if 'size' in attachment and attachment['size'] > 100 * 1024 * 1024:  # 100MB
                raise ValueError("附件大小超过限制")
        return v
    
    @validator('language')
    def validate_language(cls, v):
        """验证语言代码"""
        if v and not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise ValueError("无效的语言代码格式")
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_message_consistency(cls, values):
        """验证消息一致性"""
        message_type = values.get('message_type')
        content = values.get('content')
        tool_calls = values.get('tool_calls', [])
        attachments = values.get('attachments', [])
        
        # 工具调用消息必须有工具调用
        if message_type == MessageType.TOOL_CALL and not tool_calls:
            raise ValueError("工具调用消息必须包含工具调用")
        
        # 文件消息必须有附件
        if message_type == MessageType.FILE and not attachments:
            raise ValueError("文件消息必须包含附件")
        
        # 更新时间处理
        if values.get('edit_count', 0) > 0 and not values.get('updated_at'):
            values['updated_at'] = datetime.now(timezone.utc)
        
        return values


class AgentConfig(BaseModel, SerializationMixin):
    """智能体配置模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None,
            set: lambda v: list(v) if v else []
        }
    }
    
    agent_type: constr(min_length=1, max_length=50, pattern=r'^[a-zA-Z][a-zA-Z0-9_]*$') = Field(..., description="智能体类型")
    model_name: Optional[constr(min_length=1, max_length=100)] = Field(None, description="使用的模型名称")
    temperature: confloat(ge=0.0, le=2.0) = Field(default=0.7, description="温度参数")
    max_tokens: Optional[conint(gt=0, le=100000)] = Field(None, description="最大令牌数")
    top_p: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Top-p采样参数")
    frequency_penalty: Optional[confloat(ge=-2.0, le=2.0)] = Field(None, description="频率惩罚")
    presence_penalty: Optional[confloat(ge=-2.0, le=2.0)] = Field(None, description="存在惩罚")
    tools: List[constr(min_length=1, max_length=100)] = Field(default_factory=list, description="可用工具列表")
    capabilities: Set[AgentCapability] = Field(default_factory=set, description="智能体能力")
    system_prompt: Optional[constr(max_length=10000)] = Field(None, description="系统提示词")
    max_conversation_turns: conint(ge=1, le=1000) = Field(default=100, description="最大对话轮数")
    timeout: confloat(gt=0.0, le=600.0) = Field(default=30.0, description="超时时间（秒）")
    retry_attempts: conint(ge=0, le=5) = Field(default=3, description="重试次数")
    rate_limit: Optional[conint(gt=0)] = Field(None, description="速率限制（请求/分钟）")
    custom_config: Dict[str, Any] = Field(default_factory=dict, description="自定义配置")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    
    @validator('tools')
    def validate_tools(cls, v):
        """验证工具列表"""
        if len(v) > 50:  # 限制工具数量
            raise ValueError("工具数量过多")
        # 检查重复工具
        if len(v) != len(set(v)):
            raise ValueError("工具列表包含重复项")
        return v
    
    @validator('capabilities')
    def validate_capabilities(cls, v):
        """验证智能体能力"""
        if len(v) > 10:  # 限制能力数量
            raise ValueError("智能体能力过多")
        return v
    
    @validator('custom_config')
    def validate_custom_config(cls, v):
        """验证自定义配置"""
        if len(str(v)) > 10000:  # 限制配置大小
            raise ValueError("自定义配置过大")
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_config_consistency(cls, values):
        """验证配置一致性"""
        tools = values.get('tools', [])
        capabilities = values.get('capabilities', set())
        
        # 根据工具推断能力
        tool_capability_map = {
            'code_interpreter': AgentCapability.CODE_GENERATION,
            'web_search': AgentCapability.WEB_SEARCH,
            'image_analyzer': AgentCapability.IMAGE_ANALYSIS,
            'file_processor': AgentCapability.FILE_PROCESSING
        }
        
        inferred_capabilities = set()
        for tool in tools:
            if tool in tool_capability_map:
                inferred_capabilities.add(tool_capability_map[tool])
        
        # 自动添加推断的能力
        values['capabilities'] = capabilities.union(inferred_capabilities)
        
        # 如果有工具调用相关的工具，添加工具调用能力
        if tools:
            values['capabilities'].add(AgentCapability.TOOL_CALLING)
        
        return values


class ChatRequest(BaseModel, SerializationMixin):
    """聊天请求模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    message: constr(min_length=1, max_length=50000) = Field(..., description="用户消息")
    session_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="会话ID")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    mode: ChatMode = Field(default=ChatMode.AUTO_SELECT, description="聊天模式")
    agent_config: Optional[AgentConfig] = Field(None, description="智能体配置")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    stream: bool = Field(default=False, description="是否使用流式响应")
    max_history: conint(ge=1, le=1000) = Field(default=20, description="最大历史消息数")
    include_tools: bool = Field(default=True, description="是否包含工具调用")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="请求优先级")
    language: Optional[constr(min_length=2, max_length=10)] = Field(None, description="首选语言")
    client_info: Dict[str, Any] = Field(default_factory=dict, description="客户端信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="请求元数据")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="请求ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @validator('message')
    def validate_message(cls, v):
        """验证消息内容"""
        if not v or not v.strip():
            raise ValueError("消息内容不能为空")
        # 检查恶意内容
        forbidden_patterns = [r'<script.*?>', r'javascript:', r'data:text/html']
        for pattern in forbidden_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("消息内容包含不允许的脚本")
        return v.strip()
    
    @validator('context')
    def validate_context(cls, v):
        """验证上下文信息"""
        if len(str(v)) > 10000:  # 限制上下文大小
            raise ValueError("上下文信息过大")
        return v
    
    @validator('client_info')
    def validate_client_info(cls, v):
        """验证客户端信息"""
        if len(str(v)) > 2000:  # 限制客户端信息大小
            raise ValueError("客户端信息过大")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 5000:  # 限制元数据大小
            raise ValueError("元数据过大")
        return v
    
    @validator('language')
    def validate_language(cls, v):
        """验证语言代码"""
        if v and not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise ValueError("无效的语言代码格式")
        return v


class ChatResponse(BaseModel, SerializationMixin):
    """聊天响应模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    message: constr(min_length=1, max_length=100000) = Field(..., description="AI响应消息")
    session_id: constr(min_length=1, max_length=100) = Field(..., description="会话ID")
    message_id: constr(min_length=1, max_length=100) = Field(..., description="消息ID")
    request_id: constr(min_length=1, max_length=100) = Field(..., description="请求ID")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    agent_type: constr(min_length=1, max_length=50) = Field(..., description="智能体类型")
    model_name: constr(min_length=1, max_length=100) = Field(..., description="使用的模型名称")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="工具调用列表")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="附件列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="响应元数据")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time: confloat(ge=0.0, le=300.0) = Field(..., description="处理时间（秒）")
    token_usage: Dict[str, conint(ge=0)] = Field(default_factory=dict, description="Token使用情况")
    confidence: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="置信度")
    sources: List[constr(min_length=1, max_length=500)] = Field(default_factory=list, description="信息来源")
    suggestions: List[constr(min_length=1, max_length=200)] = Field(default_factory=list, description="建议问题")
    error: Optional[constr(min_length=1, max_length=1000)] = Field(None, description="错误信息")
    warning: Optional[constr(min_length=1, max_length=1000)] = Field(None, description="警告信息")
    language: Optional[constr(min_length=2, max_length=10)] = Field(None, description="响应语言")
    finish_reason: Optional[constr(min_length=1, max_length=50)] = Field(None, description="完成原因")
    cost: Optional[confloat(ge=0.0)] = Field(None, description="请求成本")

    @validator('message')
    def validate_message(cls, v):
        """验证响应消息"""
        if not v or not v.strip():
            raise ValueError("响应消息不能为空")
        return v.strip()
    
    @validator('tool_calls')
    def validate_tool_calls(cls, v):
        """验证工具调用数量"""
        if len(v) > 20:  # 限制工具调用数量
            raise ValueError("工具调用数量过多")
        return v
    
    @validator('attachments')
    def validate_attachments(cls, v):
        """验证附件"""
        if len(v) > 10:  # 限制附件数量
            raise ValueError("附件数量过多")
        total_size = sum(att.get('size', 0) for att in v)
        if total_size > 100 * 1024 * 1024:  # 100MB限制
            raise ValueError("附件总大小超过限制")
        return v
    
    @validator('sources')
    def validate_sources(cls, v):
        """验证信息来源数量"""
        if len(v) > 20:  # 限制来源数量
            raise ValueError("信息来源数量过多")
        return v
    
    @validator('suggestions')
    def validate_suggestions(cls, v):
        """验证建议问题数量"""
        if len(v) > 10:  # 限制建议数量
            raise ValueError("建议问题数量过多")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 10000:  # 限制元数据大小
            raise ValueError("元数据过大")
        return v
    
    @validator('language')
    def validate_language(cls, v):
        """验证语言代码"""
        if v and not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise ValueError("无效的语言代码格式")
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_response_consistency(cls, values):
        """验证响应一致性"""
        error = values.get('error')
        message = values.get('message')
        tool_calls = values.get('tool_calls', [])
        
        # 如果有错误，消息应该包含错误信息
        if error and not any(word in message.lower() for word in ['error', 'failed', 'unable', '错误', '失败']):
            values['message'] = f"处理请求时发生错误: {error}"
        
        # 验证token使用情况
        token_usage = values.get('token_usage', {})
        if token_usage:
            total_tokens = token_usage.get('total_tokens', 0)
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            
            if total_tokens > 0 and total_tokens != prompt_tokens + completion_tokens:
                if prompt_tokens > 0 and completion_tokens > 0:
                    token_usage['total_tokens'] = prompt_tokens + completion_tokens
                    values['token_usage'] = token_usage
        
        return values


class StreamChunk(BaseModel, SerializationMixin):
    """流式响应块模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    chunk_id: constr(min_length=1, max_length=100) = Field(..., description="块ID")
    session_id: constr(min_length=1, max_length=100) = Field(..., description="会话ID")
    message_id: constr(min_length=1, max_length=100) = Field(..., description="消息ID")
    request_id: constr(min_length=1, max_length=100) = Field(..., description="请求ID")
    content: constr(max_length=10000) = Field(..., description="内容块")
    chunk_type: constr(min_length=1, max_length=50) = Field(default="text", description="块类型")
    chunk_index: conint(ge=0) = Field(..., description="块索引")
    total_chunks: Optional[conint(ge=1)] = Field(None, description="总块数")
    is_final: bool = Field(default=False, description="是否为最后一块")
    delta: Optional[Dict[str, Any]] = Field(None, description="增量数据")
    finish_reason: Optional[constr(min_length=1, max_length=50)] = Field(None, description="完成原因")
    token_count: Optional[conint(ge=0)] = Field(None, description="当前块token数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="块元数据")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time: Optional[confloat(ge=0.0)] = Field(None, description="处理时间")

    @validator('content')
    def validate_content(cls, v):
        """验证内容块"""
        if len(v) > 10000:
            raise ValueError("内容块过大")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 2000:  # 限制元数据大小
            raise ValueError("元数据过大")
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_chunk_consistency(cls, values):
        """验证块一致性"""
        is_final = values.get('is_final', False)
        finish_reason = values.get('finish_reason')
        chunk_index = values.get('chunk_index', 0)
        total_chunks = values.get('total_chunks')
        
        # 如果是最后一块，应该有完成原因
        if is_final and not finish_reason:
            values['finish_reason'] = 'completed'
        
        # 验证块索引和总数的一致性
        if total_chunks is not None and chunk_index >= total_chunks:
            raise ValueError("块索引不能大于等于总块数")
        
        # 如果是最后一块且有总块数，验证索引
        if is_final and total_chunks is not None and chunk_index != total_chunks - 1:
            values['chunk_index'] = total_chunks - 1
        
        return values


class WebSocketMessage(BaseModel, SerializationMixin):
    """WebSocket消息模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    type: WebSocketMessageType = Field(..., description="消息类型")
    data: Dict[str, Any] = Field(..., description="消息数据")
    session_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="会话ID")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    message_id: constr(min_length=1, max_length=100) = Field(default_factory=lambda: str(uuid.uuid4()), description="消息ID")
    correlation_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="关联ID")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="消息优先级")
    ttl: Optional[conint(ge=1, le=3600)] = Field(None, description="生存时间（秒）")
    retry_count: conint(ge=0, le=5) = Field(default=0, description="重试次数")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    client_info: Dict[str, Any] = Field(default_factory=dict, description="客户端信息")

    @validator('data')
    def validate_data_size(cls, v):
        """验证数据大小"""
        if len(str(v)) > 100000:  # 100KB限制
            raise ValueError("消息数据过大")
        return v
    
    @validator('client_info')
    def validate_client_info(cls, v):
        """验证客户端信息"""
        if len(str(v)) > 5000:  # 限制客户端信息大小
            raise ValueError("客户端信息过大")
        return v

    @model_validator(mode='before')


    @classmethod
    def validate_message_requirements(cls, values):
        """验证消息要求"""
        msg_type = values.get('type')
        session_id = values.get('session_id')
        user_id = values.get('user_id')
        ttl = values.get('ttl')
        timestamp = values.get('timestamp')
        
        # 某些消息类型需要session_id或user_id
        if msg_type in [WebSocketMessageType.CHAT_MESSAGE] and not (session_id or user_id):
            raise ValueError(f"消息类型 {msg_type} 需要提供 session_id 或 user_id")
        
        # 设置过期时间
        if ttl and timestamp and not values.get('expires_at'):
            from datetime import timedelta
            values['expires_at'] = timestamp + timedelta(seconds=ttl)
        
        # 验证重试次数
        retry_count = values.get('retry_count', 0)
        if retry_count > 0 and msg_type in [WebSocketMessageType.HEARTBEAT, WebSocketMessageType.PONG]:
            raise ValueError("心跳和PONG消息不应该重试")
        
        return values


class AgentStatus(BaseModel, SerializationMixin):
    """智能体状态模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None,
            set: lambda v: list(v) if v else []
        }
    }
    
    agent_id: constr(min_length=1, max_length=100) = Field(..., description="智能体ID")
    agent_type: constr(min_length=1, max_length=50) = Field(..., description="智能体类型")
    status: Literal["idle", "thinking", "tool_calling", "responding", "error", "offline", "starting", "stopping", "maintenance"] = Field(..., description="状态")
    session_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="会话ID")
    current_task: Optional[constr(min_length=1, max_length=500)] = Field(None, description="当前任务")
    progress: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="进度")
    capabilities: Set[AgentCapability] = Field(default_factory=set, description="智能体能力")
    load_factor: confloat(ge=0.0, le=1.0) = Field(default=0.0, description="负载因子")
    queue_size: conint(ge=0) = Field(default=0, description="队列大小")
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = Field(None, description="启动时间")
    error_count: conint(ge=0) = Field(default=0, description="错误次数")
    success_count: conint(ge=0) = Field(default=0, description="成功次数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="状态元数据")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 5000:  # 限制元数据大小
            raise ValueError("元数据过大")
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_status_consistency(cls, values):
        """验证状态一致性"""
        status = values.get('status')
        current_task = values.get('current_task')
        progress = values.get('progress')
        queue_size = values.get('queue_size', 0)
        
        # 如果状态是忙碌相关，应该有当前任务
        if status in ['thinking', 'tool_calling', 'responding'] and not current_task:
            values['current_task'] = '处理中...'
        
        # 如果状态是空闲，清除当前任务和进度
        if status == 'idle':
            values['current_task'] = None
            values['progress'] = None
        
        # 如果状态是离线，清除队列
        if status == 'offline':
            values['queue_size'] = 0
            values['current_task'] = None
            values['progress'] = None
        
        return values


class SessionInfo(BaseModel, SerializationMixin):
    """会话信息模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    session_id: constr(min_length=1, max_length=100) = Field(..., description="会话ID")
    user_id: constr(min_length=1, max_length=100) = Field(..., description="用户ID")
    title: Optional[constr(min_length=1, max_length=200)] = Field(None, description="会话标题")
    description: Optional[constr(max_length=1000)] = Field(None, description="会话描述")
    mode: ChatMode = Field(..., description="聊天模式")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="会话状态")
    active_agents: List[constr(min_length=1, max_length=50)] = Field(default_factory=list, description="活跃智能体列表")
    message_count: conint(ge=0) = Field(default=0, description="消息数量")
    token_count: conint(ge=0) = Field(default=0, description="总token数")
    cost: confloat(ge=0.0) = Field(default=0.0, description="总成本")
    max_messages: conint(ge=1, le=10000) = Field(default=1000, description="最大消息数")
    timeout_minutes: conint(ge=1, le=1440) = Field(default=60, description="超时时间（分钟）")
    language: Optional[constr(min_length=2, max_length=10)] = Field(None, description="会话语言")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="创建时间")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="更新时间")
    ended_at: Optional[datetime] = Field(None, description="结束时间")
    client_info: Dict[str, Any] = Field(default_factory=dict, description="客户端信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="会话元数据")
    
    @validator('active_agents')
    def validate_active_agents(cls, v):
        """验证活跃智能体列表"""
        if len(v) > 10:  # 限制智能体数量
            raise ValueError("活跃智能体数量过多")
        # 检查重复
        if len(v) != len(set(v)):
            raise ValueError("活跃智能体列表包含重复项")
        return v
    
    @validator('language')
    def validate_language(cls, v):
        """验证语言代码"""
        if v and not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise ValueError("无效的语言代码格式")
        return v
    
    @validator('client_info')
    def validate_client_info(cls, v):
        """验证客户端信息"""
        if len(str(v)) > 5000:  # 限制客户端信息大小
            raise ValueError("客户端信息过大")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 10000:  # 限制元数据大小
            raise ValueError("元数据过大")
        return v
    
    @model_validator(mode='before')

    
    @classmethod
    def validate_session_consistency(cls, values):
        """验证会话一致性"""
        status = values.get('status')
        ended_at = values.get('ended_at')
        created_at = values.get('created_at')
        updated_at = values.get('updated_at')
        message_count = values.get('message_count', 0)
        max_messages = values.get('max_messages', 1000)
        
        # 如果会话已结束，应该有结束时间
        if status in [SessionStatus.EXPIRED, SessionStatus.TERMINATED] and not ended_at:
            values['ended_at'] = datetime.now(timezone.utc)
        
        # 如果会话未结束，清除结束时间
        if status not in [SessionStatus.EXPIRED, SessionStatus.TERMINATED] and ended_at:
            values['ended_at'] = None
        
        # 验证时间逻辑
        if created_at and updated_at and updated_at < created_at:
            values['updated_at'] = created_at
        
        if ended_at and created_at and ended_at < created_at:
            values['ended_at'] = datetime.now(timezone.utc)
        
        # 检查消息数量限制
        if message_count >= max_messages and status == SessionStatus.ACTIVE:
            values['status'] = SessionStatus.EXPIRED
            values['ended_at'] = datetime.now(timezone.utc)
        
        return values


class ChatHistory(BaseModel, SerializationMixin):
    """聊天历史模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    session_id: constr(min_length=1, max_length=100) = Field(..., description="会话ID")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    messages: List[ChatMessage] = Field(default_factory=list, description="消息列表")
    total_messages: conint(ge=0) = Field(default=0, description="总消息数")
    total_tokens: conint(ge=0) = Field(default=0, description="总token数")
    total_cost: confloat(ge=0.0) = Field(default=0.0, description="总成本")
    page: conint(ge=1) = Field(default=1, description="页码")
    page_size: conint(ge=1, le=100) = Field(default=20, description="每页大小")
    has_more: bool = Field(default=False, description="是否有更多消息")
    next_cursor: Optional[str] = Field(None, description="下一页游标")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    first_message_at: Optional[datetime] = Field(None, description="首条消息时间")
    last_message_at: Optional[datetime] = Field(None, description="最后消息时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="历史元数据")

    @validator('messages')
    def validate_messages(cls, v):
        """验证消息列表"""
        if len(v) > 1000:
            raise ValueError("消息数量过多")
        
        # 验证消息时间顺序
        for i in range(1, len(v)):
            if v[i].created_at < v[i-1].created_at:
                raise ValueError("消息时间顺序不正确")
        
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 10000:  # 限制元数据大小
            raise ValueError("元数据过大")
        return v

    @model_validator(mode='before')


    @classmethod
    def validate_history_consistency(cls, values):
        """验证历史一致性"""
        messages = values.get('messages', [])
        page = values.get('page', 1)
        page_size = values.get('page_size', 20)
        
        # 更新总消息数
        values['total_messages'] = len(messages)
        
        # 计算总token数和成本
        total_tokens = sum(msg.token_count or 0 for msg in messages)
        values['total_tokens'] = total_tokens
        
        # 设置首条和最后消息时间
        if messages:
            values['first_message_at'] = messages[0].created_at
            values['last_message_at'] = messages[-1].created_at
            values['updated_at'] = messages[-1].created_at
        
        # 验证分页逻辑
        expected_messages = min(page_size, values['total_messages'])
        if len(messages) > expected_messages:
            # 截取到正确的页面大小
            values['messages'] = messages[:expected_messages]
        
        # 设置是否有更多消息
        values['has_more'] = values['total_messages'] > page * page_size
        
        return values


class ChatMessageRequest(BaseModel, SerializationMixin):
    """聊天消息请求模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    content: constr(min_length=1, max_length=50000) = Field(..., description="消息内容")
    role: constr(min_length=1, max_length=20) = Field(default="user", description="角色")
    session_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="会话ID")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    parent_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="父消息ID")
    thread_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="线程ID")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="消息优先级")
    language: Optional[constr(min_length=2, max_length=10)] = Field(None, description="语言代码")
    client_info: Optional[Dict[str, Any]] = Field(None, description="客户端信息")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="消息元数据")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="附件列表")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="工具调用列表")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @validator('content')
    def validate_content(cls, v):
        """验证消息内容"""
        if not v.strip():
            raise ValueError("消息内容不能为空")
        
        # 检查恶意内容
        malicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'document\.cookie'
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("消息内容包含潜在恶意代码")
        
        return v.strip()
    
    @validator('role')
    def validate_role(cls, v):
        """验证角色"""
        allowed_roles = {'user', 'assistant', 'system', 'function'}
        if v not in allowed_roles:
            raise ValueError(f"角色必须是 {allowed_roles} 之一")
        return v
    
    @validator('language')
    def validate_language(cls, v):
        """验证语言代码"""
        if v and not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise ValueError("语言代码格式不正确")
        return v
    
    @validator('client_info')
    def validate_client_info(cls, v):
        """验证客户端信息"""
        if v and len(str(v)) > 5000:
            raise ValueError("客户端信息过大")
        return v
    
    @validator('context')
    def validate_context(cls, v):
        """验证上下文信息"""
        if v and len(str(v)) > 10000:
            raise ValueError("上下文信息过大")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 5000:
            raise ValueError("元数据过大")
        return v
    
    @validator('attachments')
    def validate_attachments(cls, v):
        """验证附件"""
        if len(v) > 10:
            raise ValueError("附件数量过多")
        
        total_size = sum(att.get('size', 0) for att in v)
        if total_size > 100 * 1024 * 1024:  # 100MB
            raise ValueError("附件总大小超过限制")
        
        return v
    
    @validator('tool_calls')
    def validate_tool_calls(cls, v):
        """验证工具调用"""
        if len(v) > 5:
            raise ValueError("工具调用数量过多")
        return v

    @model_validator(mode='before')


    @classmethod
    def validate_request_consistency(cls, values):
        """验证请求一致性"""
        role = values.get('role')
        content = values.get('content', '')
        tool_calls = values.get('tool_calls', [])
        attachments = values.get('attachments', [])
        
        # 系统角色不应有工具调用或附件
        if role == 'system' and (tool_calls or attachments):
            raise ValueError("系统消息不应包含工具调用或附件")
        
        # 功能角色必须有工具调用结果
        if role == 'function' and not tool_calls:
            raise ValueError("功能消息必须包含工具调用")
        
        # 设置过期时间（如果未设置）
        if not values.get('expires_at'):
            from datetime import timedelta
            values['expires_at'] = values.get('created_at') + timedelta(hours=24)
        
        return values


class BatchChatRequest(BaseModel, SerializationMixin):
    """批量聊天请求模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    requests: List[ChatRequest] = Field(..., description="请求列表")
    batch_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="批次ID")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="优先级")
    timeout: conint(ge=1, le=300) = Field(default=30, description="超时时间（秒）")
    max_concurrent: conint(ge=1, le=10) = Field(default=5, description="最大并发数")
    fail_fast: bool = Field(default=False, description="快速失败模式")
    retry_failed: bool = Field(default=True, description="重试失败请求")
    callback_url: Optional[str] = Field(None, description="回调URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="批次元数据")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(None, description="过期时间")

    @validator('requests')
    def validate_requests(cls, v):
        """验证请求列表"""
        if not v:
            raise ValueError("请求列表不能为空")
        if len(v) > 100:
            raise ValueError("批量请求数量过多")
        
        # 检查重复的请求ID
        request_ids = [req.request_id for req in v if req.request_id]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("存在重复的请求ID")
        
        return v
    
    @validator('callback_url')
    def validate_callback_url(cls, v):
        """验证回调URL"""
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("回调URL格式不正确")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 5000:
            raise ValueError("元数据过大")
        return v

    @model_validator(mode='before')


    @classmethod
    def validate_batch_consistency(cls, values):
        """验证批次一致性"""
        requests = values.get('requests', [])
        timeout = values.get('timeout', 30)
        max_concurrent = values.get('max_concurrent', 5)
        
        # 生成批次ID（如果未提供）
        if not values.get('batch_id'):
            import uuid
            values['batch_id'] = f"batch_{uuid.uuid4().hex[:8]}"
        
        # 设置过期时间
        if not values.get('expires_at'):
            from datetime import timedelta
            values['expires_at'] = values.get('created_at') + timedelta(seconds=timeout + 60)
        
        # 验证并发数不超过请求数
        if max_concurrent > len(requests):
            values['max_concurrent'] = len(requests)
        
        return values


class BatchChatResponse(BaseModel, SerializationMixin):
    """批量聊天响应模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    batch_id: constr(min_length=1, max_length=100) = Field(..., description="批次ID")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    responses: List[ChatResponse] = Field(..., description="响应列表")
    total_count: conint(ge=0) = Field(..., description="总请求数")
    success_count: conint(ge=0) = Field(..., description="成功数")
    failed_count: conint(ge=0) = Field(..., description="失败数")
    pending_count: conint(ge=0) = Field(default=0, description="待处理数")
    processing_time: confloat(ge=0.0) = Field(..., description="处理时间")
    average_response_time: confloat(ge=0.0) = Field(default=0.0, description="平均响应时间")
    total_tokens: conint(ge=0) = Field(default=0, description="总token数")
    total_cost: confloat(ge=0.0) = Field(default=0.0, description="总成本")
    status: str = Field(default="completed", description="批次状态")
    errors: List[str] = Field(default_factory=list, description="错误列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="批次元数据")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")

    @validator('status')
    def validate_status(cls, v):
        """验证状态"""
        allowed_statuses = {'pending', 'processing', 'completed', 'failed', 'cancelled'}
        if v not in allowed_statuses:
            raise ValueError(f"状态必须是 {allowed_statuses} 之一")
        return v
    
    @validator('errors')
    def validate_errors(cls, v):
        """验证错误列表"""
        if len(v) > 100:
            raise ValueError("错误数量过多")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 10000:
            raise ValueError("元数据过大")
        return v

    @model_validator(mode='before')


    @classmethod
    def validate_batch_response_consistency(cls, values):
        """验证批次响应一致性"""
        responses = values.get('responses', [])
        status = values.get('status', 'completed')
        
        # 计算统计信息
        success_responses = [r for r in responses if not r.error]
        failed_responses = [r for r in responses if r.error]
        
        values['total_count'] = len(responses)
        values['success_count'] = len(success_responses)
        values['failed_count'] = len(failed_responses)
        
        # 计算平均响应时间
        if responses:
            total_time = sum(r.processing_time or 0 for r in responses)
            values['average_response_time'] = total_time / len(responses)
        
        # 计算总token数和成本
        total_tokens = 0
        total_cost = 0.0
        for response in responses:
            if response.token_usage:
                total_tokens += response.token_usage.get('total', 0)
            if response.cost:
                total_cost += response.cost
        
        values['total_tokens'] = total_tokens
        values['total_cost'] = total_cost
        
        # 设置完成时间
        if status == 'completed' and not values.get('completed_at'):
            values['completed_at'] = datetime.now(timezone.utc)
        
        # 验证时间逻辑
        created_at = values.get('created_at')
        started_at = values.get('started_at')
        completed_at = values.get('completed_at')
        
        if started_at and created_at and started_at < created_at:
            raise ValueError("开始时间不能早于创建时间")
        
        if completed_at and started_at and completed_at < started_at:
            raise ValueError("完成时间不能早于开始时间")
        
        return values


class ChatError(BaseModel, SerializationMixin):
    """聊天错误模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    error_code: constr(min_length=1, max_length=50, pattern=r'^[A-Z0-9_]+$') = Field(..., description="错误代码")
    error_message: constr(min_length=1, max_length=1000) = Field(..., description="错误消息")
    error_type: constr(min_length=1, max_length=50) = Field(..., description="错误类型")
    severity: constr(min_length=1, max_length=20) = Field(default="error", description="严重程度")
    category: constr(min_length=1, max_length=50) = Field(default="general", description="错误分类")
    request_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="请求ID")
    session_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="会话ID")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    agent_type: Optional[constr(min_length=1, max_length=50)] = Field(None, description="智能体类型")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = Field(default_factory=dict, description="错误上下文")
    stack_trace: Optional[constr(max_length=10000)] = Field(None, description="堆栈跟踪")
    user_message: Optional[constr(max_length=500)] = Field(None, description="用户友好错误消息")
    retry_after: Optional[conint(ge=0, le=3600)] = Field(None, description="重试间隔（秒）")
    resolution_steps: List[constr(max_length=200)] = Field(default_factory=list, description="解决步骤")
    related_errors: List[constr(max_length=100)] = Field(default_factory=list, description="相关错误ID")
    details: Dict[str, Any] = Field(default_factory=dict, description="错误详情")

    @validator('error_code')
    def validate_error_code(cls, v):
        """验证错误代码格式"""
        if not v.isupper():
            raise ValueError("错误代码必须为大写")
        return v
    
    @validator('severity')
    def validate_severity(cls, v):
        """验证严重程度"""
        allowed_severities = {'debug', 'info', 'warning', 'error', 'critical'}
        if v.lower() not in allowed_severities:
            raise ValueError(f"严重程度必须是 {allowed_severities} 之一")
        return v.lower()
    
    @validator('category')
    def validate_category(cls, v):
        """验证错误分类"""
        allowed_categories = {
            'general', 'validation', 'authentication', 'authorization', 
            'network', 'database', 'timeout', 'rate_limit', 'system'
        }
        if v.lower() not in allowed_categories:
            raise ValueError(f"错误分类必须是 {allowed_categories} 之一")
        return v.lower()
    
    @validator('context')
    def validate_context(cls, v):
        """验证错误上下文"""
        if len(str(v)) > 5000:
            raise ValueError("错误上下文过大")
        return v
    
    @validator('stack_trace')
    def validate_stack_trace(cls, v):
        """验证堆栈跟踪"""
        if v and len(v) > 10000:
            raise ValueError("堆栈跟踪过长")
        return v
    
    @validator('resolution_steps')
    def validate_resolution_steps(cls, v):
        """验证解决步骤"""
        if len(v) > 10:
            raise ValueError("解决步骤过多")
        return v
    
    @validator('related_errors')
    def validate_related_errors(cls, v):
        """验证相关错误"""
        if len(v) > 5:
            raise ValueError("相关错误过多")
        return v
    
    @validator('details')
    def validate_details(cls, v):
        """验证错误详情"""
        if len(str(v)) > 5000:
            raise ValueError("错误详情过大")
        return v


class ConnectionInfo(BaseModel, SerializationMixin):
    """连接信息模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    connection_id: constr(min_length=1, max_length=100) = Field(..., description="连接ID")
    user_id: constr(min_length=1, max_length=100) = Field(..., description="用户ID")
    session_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="会话ID")
    status: constr(min_length=1, max_length=20) = Field(default="active", description="连接状态")
    connection_type: constr(min_length=1, max_length=20) = Field(default="websocket", description="连接类型")
    ip_address: Optional[constr(min_length=7, max_length=45)] = Field(None, description="IP地址")
    user_agent: Optional[constr(max_length=500)] = Field(None, description="用户代理")
    connected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    disconnected_at: Optional[datetime] = Field(None, description="断开连接时间")
    heartbeat_interval: conint(ge=1, le=300) = Field(default=30, description="心跳间隔（秒）")
    last_heartbeat: Optional[datetime] = Field(None, description="最后心跳时间")
    bandwidth_usage: confloat(ge=0.0) = Field(default=0.0, description="带宽使用量（MB）")
    message_count: conint(ge=0) = Field(default=0, description="消息数量")
    error_count: conint(ge=0) = Field(default=0, description="错误数量")
    client_info: Dict[str, Any] = Field(default_factory=dict, description="客户端信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="连接元数据")

    @validator('status')
    def validate_status(cls, v):
        """验证连接状态"""
        allowed_statuses = {'active', 'inactive', 'disconnected', 'error', 'timeout'}
        if v.lower() not in allowed_statuses:
            raise ValueError(f"连接状态必须是 {allowed_statuses} 之一")
        return v.lower()
    
    @validator('connection_type')
    def validate_connection_type(cls, v):
        """验证连接类型"""
        allowed_types = {'websocket', 'http', 'grpc', 'tcp', 'udp'}
        if v.lower() not in allowed_types:
            raise ValueError(f"连接类型必须是 {allowed_types} 之一")
        return v.lower()
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        """验证IP地址格式"""
        if v:
            import re
            # 简单的IPv4和IPv6验证
            ipv4_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
            ipv6_pattern = r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
            if not (re.match(ipv4_pattern, v) or re.match(ipv6_pattern, v)):
                raise ValueError("IP地址格式不正确")
        return v
    
    @validator('client_info')
    def validate_client_info(cls, v):
        """验证客户端信息"""
        if len(str(v)) > 2000:
            raise ValueError("客户端信息过大")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 5000:
            raise ValueError("元数据过大")
        return v

    @model_validator(mode='before')


    @classmethod
    def validate_connection_consistency(cls, values):
        """验证连接一致性"""
        connected_at = values.get('connected_at')
        last_activity = values.get('last_activity')
        disconnected_at = values.get('disconnected_at')
        last_heartbeat = values.get('last_heartbeat')
        status = values.get('status', 'active')
        
        # 验证时间逻辑
        if last_activity and connected_at and last_activity < connected_at:
            raise ValueError("最后活动时间不能早于连接时间")
        
        if disconnected_at and connected_at and disconnected_at < connected_at:
            raise ValueError("断开连接时间不能早于连接时间")
        
        if last_heartbeat and connected_at and last_heartbeat < connected_at:
            raise ValueError("最后心跳时间不能早于连接时间")
        
        # 状态一致性验证
        if status == 'disconnected' and not disconnected_at:
            values['disconnected_at'] = datetime.now(timezone.utc)
        
        if status == 'active' and disconnected_at:
            raise ValueError("活跃连接不应有断开时间")
        
        # 设置最后心跳时间
        if status == 'active' and not last_heartbeat:
            values['last_heartbeat'] = connected_at
        
        return values


class MessageInfo(BaseModel, SerializationMixin):
    """消息信息模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    id: constr(min_length=1, max_length=100) = Field(..., description="消息ID")
    role: MessageRole = Field(..., description="消息角色")
    content: constr(min_length=1, max_length=50000) = Field(..., description="消息内容")
    message_type: MessageType = Field(default=MessageType.TEXT, description="消息类型")
    session_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="会话ID")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    parent_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="父消息ID")
    thread_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="线程ID")
    status: constr(min_length=1, max_length=20) = Field(default="sent", description="消息状态")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="消息优先级")
    language: Optional[constr(min_length=2, max_length=10)] = Field(None, description="消息语言")
    encoding: constr(min_length=1, max_length=20) = Field(default="utf-8", description="消息编码")
    content_hash: Optional[constr(min_length=1, max_length=64)] = Field(None, description="内容哈希")
    word_count: conint(ge=0) = Field(default=0, description="字数统计")
    char_count: conint(ge=0) = Field(default=0, description="字符数统计")
    token_count: Optional[conint(ge=0)] = Field(None, description="Token数量")
    processing_time: Optional[confloat(ge=0.0)] = Field(None, description="处理时间（秒）")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="附件列表")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="工具调用")
    context: Dict[str, Any] = Field(default_factory=dict, description="消息上下文")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="消息元数据")

    @validator('status')
    def validate_status(cls, v):
        """验证消息状态"""
        allowed_statuses = {'draft', 'sent', 'delivered', 'read', 'failed', 'deleted'}
        if v.lower() not in allowed_statuses:
            raise ValueError(f"消息状态必须是 {allowed_statuses} 之一")
        return v.lower()
    
    @validator('language')
    def validate_language(cls, v):
        """验证语言代码"""
        if v:
            import re
            # 验证ISO 639-1语言代码格式
            if not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
                raise ValueError("语言代码格式不正确，应为ISO 639-1格式")
        return v
    
    @validator('encoding')
    def validate_encoding(cls, v):
        """验证编码格式"""
        allowed_encodings = {'utf-8', 'utf-16', 'ascii', 'iso-8859-1'}
        if v.lower() not in allowed_encodings:
            raise ValueError(f"编码格式必须是 {allowed_encodings} 之一")
        return v.lower()
    
    @validator('content')
    def validate_content(cls, v):
        """验证消息内容"""
        # 检查恶意内容（简单示例）
        malicious_patterns = ['<script', 'javascript:', 'data:text/html']
        content_lower = v.lower()
        for pattern in malicious_patterns:
            if pattern in content_lower:
                raise ValueError("消息内容包含潜在恶意代码")
        return v
    
    @validator('attachments')
    def validate_attachments(cls, v):
        """验证附件"""
        if len(v) > 10:
            raise ValueError("附件数量不能超过10个")
        
        total_size = 0
        for attachment in v:
            if 'size' in attachment:
                total_size += attachment.get('size', 0)
        
        if total_size > 100 * 1024 * 1024:  # 100MB
            raise ValueError("附件总大小不能超过100MB")
        
        return v
    
    @validator('tool_calls')
    def validate_tool_calls(cls, v):
        """验证工具调用"""
        if len(v) > 5:
            raise ValueError("工具调用数量不能超过5个")
        return v
    
    @validator('context')
    def validate_context(cls, v):
        """验证消息上下文"""
        if len(str(v)) > 5000:
            raise ValueError("消息上下文过大")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 5000:
            raise ValueError("元数据过大")
        return v

    @model_validator(mode='before')


    @classmethod
    def validate_message_consistency(cls, values):
        """验证消息一致性"""
        content = values.get('content', '')
        role = values.get('role')
        message_type = values.get('message_type')
        tool_calls = values.get('tool_calls', [])
        attachments = values.get('attachments', [])
        created_at = values.get('created_at')
        updated_at = values.get('updated_at')
        expires_at = values.get('expires_at')
        
        # 计算字数和字符数
        if content:
            values['char_count'] = len(content)
            values['word_count'] = len(content.split())
        
        # 生成内容哈希
        if content and not values.get('content_hash'):
            import hashlib
            values['content_hash'] = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # 验证角色和工具调用的一致性
        if role == MessageRole.FUNCTION and not tool_calls:
            raise ValueError("功能角色消息必须包含工具调用")
        
        if role == MessageRole.SYSTEM and (tool_calls or attachments):
            raise ValueError("系统角色消息不应包含工具调用或附件")
        
        # 验证时间逻辑
        if updated_at and created_at and updated_at < created_at:
            raise ValueError("更新时间不能早于创建时间")
        
        if expires_at and created_at and expires_at < created_at:
            raise ValueError("过期时间不能早于创建时间")
        
        # 设置默认过期时间（24小时后）
        if not expires_at and created_at:
            from datetime import timedelta
            values['expires_at'] = created_at + timedelta(hours=24)
        
        return values


class ChatMessageResponse(BaseModel, SerializationMixin):
    """聊天消息响应模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    user_message: MessageInfo = Field(..., description="用户消息")
    ai_message: MessageInfo = Field(..., description="AI消息")
    session_id: constr(min_length=1, max_length=100) = Field(..., description="会话ID")
    user_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="用户ID")
    agent_type: Optional[constr(min_length=1, max_length=50)] = Field(None, description="智能体类型")
    conversation_id: Optional[constr(min_length=1, max_length=100)] = Field(None, description="对话ID")
    status: constr(min_length=1, max_length=20) = Field(default="completed", description="响应状态")
    processing_time: Optional[confloat(ge=0.0, le=300.0)] = Field(None, description="处理时间（秒）")
    token_usage: Dict[str, conint(ge=0)] = Field(default_factory=dict, description="Token使用情况")
    cost: Optional[confloat(ge=0.0)] = Field(None, description="响应成本")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="模型信息")
    quality_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="响应质量评分")
    confidence_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="置信度评分")
    feedback: Optional[Dict[str, Any]] = Field(None, description="用户反馈")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="响应元数据")

    @validator('status')
    def validate_status(cls, v):
        """验证响应状态"""
        allowed_statuses = {'pending', 'processing', 'completed', 'failed', 'timeout'}
        if v.lower() not in allowed_statuses:
            raise ValueError(f"响应状态必须是 {allowed_statuses} 之一")
        return v.lower()
    
    @validator('token_usage')
    def validate_token_usage(cls, v):
        """验证Token使用情况"""
        required_keys = {'prompt_tokens', 'completion_tokens', 'total_tokens'}
        if v and not all(key in v for key in required_keys):
            # 如果提供了token_usage，确保包含必要的键
            for key in required_keys:
                if key not in v:
                    v[key] = 0
        return v
    
    @validator('model_info')
    def validate_model_info(cls, v):
        """验证模型信息"""
        if len(str(v)) > 2000:
            raise ValueError("模型信息过大")
        return v
    
    @validator('feedback')
    def validate_feedback(cls, v):
        """验证用户反馈"""
        if v and len(str(v)) > 2000:
            raise ValueError("用户反馈过大")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 5000:
            raise ValueError("元数据过大")
        return v

    @model_validator(mode='before')


    @classmethod
    def validate_response_consistency(cls, values):
        """验证响应一致性"""
        user_message = values.get('user_message')
        ai_message = values.get('ai_message')
        session_id = values.get('session_id')
        status = values.get('status', 'completed')
        processing_time = values.get('processing_time')
        token_usage = values.get('token_usage', {})
        created_at = values.get('created_at')
        completed_at = values.get('completed_at')
        
        # 验证消息的会话ID一致性
        if user_message and user_message.session_id and user_message.session_id != session_id:
            raise ValueError("用户消息的会话ID与响应会话ID不一致")
        
        if ai_message and ai_message.session_id and ai_message.session_id != session_id:
            raise ValueError("AI消息的会话ID与响应会话ID不一致")
        
        # 设置完成时间
        if status == 'completed' and not completed_at:
            values['completed_at'] = datetime.now(timezone.utc)
        
        # 验证时间逻辑
        if completed_at and created_at and completed_at < created_at:
            raise ValueError("完成时间不能早于创建时间")
        
        # 计算成本（如果未提供）
        if not values.get('cost') and token_usage:
            total_tokens = token_usage.get('total_tokens', 0)
            # 简单的成本计算示例（实际应根据模型定价）
            values['cost'] = total_tokens * 0.0001
        
        # 验证处理时间合理性
        if processing_time and processing_time > 300:
            raise ValueError("处理时间过长，可能存在问题")
        
        return values


class AgentMetrics(BaseModel, SerializationMixin):
    """智能体指标模型"""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    agent_id: constr(min_length=1, max_length=100) = Field(..., description="智能体ID")
    agent_type: constr(min_length=1, max_length=50) = Field(..., description="智能体类型")
    agent_version: constr(min_length=1, max_length=20) = Field(default="1.0.0", description="智能体版本")
    status: constr(min_length=1, max_length=20) = Field(default="active", description="智能体状态")
    total_requests: conint(ge=0) = Field(default=0, description="总请求数")
    successful_requests: conint(ge=0) = Field(default=0, description="成功请求数")
    failed_requests: conint(ge=0) = Field(default=0, description="失败请求数")
    pending_requests: conint(ge=0) = Field(default=0, description="待处理请求数")
    timeout_requests: conint(ge=0) = Field(default=0, description="超时请求数")
    average_response_time: confloat(ge=0.0) = Field(default=0.0, description="平均响应时间")
    min_response_time: confloat(ge=0.0) = Field(default=0.0, description="最小响应时间")
    max_response_time: confloat(ge=0.0) = Field(default=0.0, description="最大响应时间")
    total_tokens_used: conint(ge=0) = Field(default=0, description="总令牌使用量")
    total_cost: confloat(ge=0.0) = Field(default=0.0, description="总成本")
    success_rate: confloat(ge=0.0, le=1.0) = Field(default=0.0, description="成功率")
    error_rate: confloat(ge=0.0, le=1.0) = Field(default=0.0, description="错误率")
    throughput: confloat(ge=0.0) = Field(default=0.0, description="吞吐量（请求/秒）")
    cpu_usage: confloat(ge=0.0, le=100.0) = Field(default=0.0, description="CPU使用率（%）")
    memory_usage: confloat(ge=0.0) = Field(default=0.0, description="内存使用量（MB）")
    disk_usage: confloat(ge=0.0) = Field(default=0.0, description="磁盘使用量（MB）")
    network_in: confloat(ge=0.0) = Field(default=0.0, description="网络入流量（MB）")
    network_out: confloat(ge=0.0) = Field(default=0.0, description="网络出流量（MB）")
    queue_size: conint(ge=0) = Field(default=0, description="队列大小")
    max_queue_size: conint(ge=0) = Field(default=100, description="最大队列大小")
    concurrent_sessions: conint(ge=0) = Field(default=0, description="并发会话数")
    max_concurrent_sessions: conint(ge=0) = Field(default=10, description="最大并发会话数")
    last_used: Optional[datetime] = Field(None, description="最后使用时间")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="启动时间")
    uptime: confloat(ge=0.0) = Field(default=0.0, description="运行时间（秒）")
    health_score: confloat(ge=0.0, le=1.0) = Field(default=1.0, description="健康评分")
    performance_score: confloat(ge=0.0, le=1.0) = Field(default=1.0, description="性能评分")
    error_details: List[Dict[str, Any]] = Field(default_factory=list, description="错误详情")
    capabilities: List[constr(max_length=50)] = Field(default_factory=list, description="智能体能力")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="指标元数据")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @validator('status')
    def validate_status(cls, v):
        """验证智能体状态"""
        allowed_statuses = {'active', 'inactive', 'maintenance', 'error', 'stopped'}
        if v.lower() not in allowed_statuses:
            raise ValueError(f"智能体状态必须是 {allowed_statuses} 之一")
        return v.lower()
    
    @validator('agent_version')
    def validate_agent_version(cls, v):
        """验证版本格式"""
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError("版本格式应为 x.y.z")
        return v
    
    @validator('capabilities')
    def validate_capabilities(cls, v):
        """验证智能体能力"""
        if len(v) > 20:
            raise ValueError("智能体能力数量不能超过20个")
        return v
    
    @validator('error_details')
    def validate_error_details(cls, v):
        """验证错误详情"""
        if len(v) > 50:
            raise ValueError("错误详情数量过多")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """验证元数据"""
        if len(str(v)) > 5000:
            raise ValueError("元数据过大")
        return v

    @model_validator(mode='before')


    @classmethod
    def validate_metrics_consistency(cls, values):
        """验证指标一致性"""
        total_requests = values.get('total_requests', 0)
        successful_requests = values.get('successful_requests', 0)
        failed_requests = values.get('failed_requests', 0)
        pending_requests = values.get('pending_requests', 0)
        timeout_requests = values.get('timeout_requests', 0)
        started_at = values.get('started_at')
        last_used = values.get('last_used')
        status = values.get('status', 'active')
        
        # 验证请求数量一致性
        if successful_requests + failed_requests + timeout_requests > total_requests:
            raise ValueError("成功、失败和超时请求数之和不能超过总请求数")
        
        # 计算成功率和错误率
        if total_requests > 0:
            values['success_rate'] = successful_requests / total_requests
            values['error_rate'] = (failed_requests + timeout_requests) / total_requests
        else:
            values['success_rate'] = 0.0
            values['error_rate'] = 0.0
        
        # 计算运行时间
        if started_at:
            current_time = datetime.now(timezone.utc)
            values['uptime'] = (current_time - started_at).total_seconds()
        
        # 计算吞吐量
        uptime = values.get('uptime', 0)
        if uptime > 0:
            values['throughput'] = total_requests / (uptime / 3600)  # 请求/小时转换为请求/秒
        
        # 验证时间逻辑
        if last_used and started_at and last_used < started_at:
            raise ValueError("最后使用时间不能早于启动时间")
        
        # 验证队列大小
        queue_size = values.get('queue_size', 0)
        max_queue_size = values.get('max_queue_size', 100)
        if queue_size > max_queue_size:
            raise ValueError("当前队列大小不能超过最大队列大小")
        
        # 验证并发会话数
        concurrent_sessions = values.get('concurrent_sessions', 0)
        max_concurrent_sessions = values.get('max_concurrent_sessions', 10)
        if concurrent_sessions > max_concurrent_sessions:
            raise ValueError("当前并发会话数不能超过最大并发会话数")
        
        # 计算健康评分
        health_factors = [
            values.get('success_rate', 0),
            1 - values.get('error_rate', 0),
            1 - (values.get('cpu_usage', 0) / 100),
            1 - min(values.get('queue_size', 0) / max_queue_size, 1)
        ]
        values['health_score'] = sum(health_factors) / len(health_factors)
        
        # 计算性能评分
        avg_response_time = values.get('average_response_time', 0)
        performance_factors = [
            values.get('success_rate', 0),
            max(0, 1 - (avg_response_time / 10)),  # 假设10秒为基准
            values.get('throughput', 0) / 100 if values.get('throughput', 0) < 100 else 1
        ]
        values['performance_score'] = sum(performance_factors) / len(performance_factors)
        
        # 状态一致性验证
        if status == 'error' and values.get('error_rate', 0) == 0:
            raise ValueError("错误状态的智能体应该有错误记录")
        
        if status == 'inactive' and concurrent_sessions > 0:
            raise ValueError("非活跃状态的智能体不应有并发会话")
        
        return values