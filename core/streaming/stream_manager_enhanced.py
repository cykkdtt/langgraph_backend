"""
增强的流管理器

提供完整的流式处理管理功能。
"""

import asyncio
import json
import gzip
from typing import Optional, Dict, Any, List, Callable, AsyncIterator, Union
from datetime import datetime
import uuid
import weakref
from collections import defaultdict
from pydantic import BaseModel

from .stream_types import (
    StreamMode, StreamEvent, StreamChunk, StreamConfig, StreamEventType
)
from typing import NamedTuple
from enum import Enum

# 为了保持兼容性，定义一些额外的类型
class StreamType(str, Enum):
    """流类型"""
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    EVENT = "event"
    CHUNK = "chunk"
    DELTA = "delta"

class StreamStatus(str, Enum):
    """流状态"""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StreamMessage(BaseModel):
    """流消息"""
    id: str
    stream_id: str
    type: StreamType
    content: Any
    sequence: int
    is_final: bool = False
    timestamp: datetime = None
    metadata: Dict[str, Any] = {}
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now()
        super().__init__(**data)

class StreamMetrics(BaseModel):
    """流指标"""
    stream_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    chunks_sent: int = 0
    chunks_received: int = 0
    errors: int = 0
    retries: int = 0
    
    def calculate_metrics(self) -> None:
        """计算派生指标"""
        if self.start_time and self.end_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

class StreamFilter(BaseModel):
    """流过滤器"""
    event_types: Optional[List[str]] = None
    min_timestamp: Optional[datetime] = None
    max_timestamp: Optional[datetime] = None
    metadata_filters: Dict[str, Any] = {}

class StreamSubscription(BaseModel):
    """流订阅"""
    id: str
    stream_id: str
    subscriber_id: str
    filter: Optional[StreamFilter] = None
    created_at: datetime = None
    is_active: bool = True
    
    def __init__(self, **data):
        if 'created_at' not in data:
            data['created_at'] = datetime.now()
        super().__init__(**data)

class StreamBuffer(BaseModel):
    """流缓冲区"""
    stream_id: str
    buffer_id: str
    capacity: int
    current_size: int = 0
    messages: List[StreamMessage] = []
    is_full: bool = False
    overflow_strategy: str = "drop_oldest"
    
    def add_message(self, message: StreamMessage) -> bool:
        """添加消息到缓冲区"""
        if self.is_full and self.overflow_strategy == "drop_new":
            return False
        
        if self.is_full and self.overflow_strategy == "drop_oldest":
            self.messages.pop(0)
            self.current_size -= 1
        
        self.messages.append(message)
        self.current_size += 1
        self.is_full = self.current_size >= self.capacity
        return True
    
    def get_messages(self, count: Optional[int] = None) -> List[StreamMessage]:
        """获取消息"""
        if count is None:
            return self.messages.copy()
        return self.messages[-count:] if count > 0 else []
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.messages.clear()
        self.current_size = 0
        self.is_full = False


class StreamSession:
    """流会话"""
    
    def __init__(
        self,
        session_id: str,
        config: StreamConfig,
        manager: "StreamManager"
    ):
        self.session_id = session_id
        self.config = config
        self.manager = weakref.ref(manager)
        
        # 状态管理
        self.status = StreamStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        
        # 数据管理
        self.buffer = StreamBuffer(
            stream_id=session_id,
            buffer_id=f"{session_id}_buffer",
            capacity=config.buffer_size
        )
        self.metrics = StreamMetrics(
            stream_id=session_id,
            start_time=self.created_at
        )
        
        # 订阅管理
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # 控制标志
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # 初始为非暂停状态
        
        # 错误处理
        self.error_count = 0
        self.last_error: Optional[Exception] = None
    
    async def start(self) -> None:
        """启动流会话"""
        if self.status != StreamStatus.PENDING:
            raise ValueError(f"无法启动流会话，当前状态: {self.status}")
        
        self.status = StreamStatus.ACTIVE
        self.started_at = datetime.now()
        self.metrics.start_time = self.started_at
        
        await self._emit_event("session_started", {"session_id": self.session_id})
    
    async def stop(self) -> None:
        """停止流会话"""
        if self.status in [StreamStatus.COMPLETED, StreamStatus.CANCELLED]:
            return
        
        self.status = StreamStatus.COMPLETED
        self.ended_at = datetime.now()
        self.metrics.end_time = self.ended_at
        self.metrics.calculate_metrics()
        
        self._stop_event.set()
        await self._emit_event("session_stopped", {"session_id": self.session_id})
    
    async def pause(self) -> None:
        """暂停流会话"""
        if self.status != StreamStatus.ACTIVE:
            return
        
        self.status = StreamStatus.PAUSED
        self._pause_event.clear()
        await self._emit_event("session_paused", {"session_id": self.session_id})
    
    async def resume(self) -> None:
        """恢复流会话"""
        if self.status != StreamStatus.PAUSED:
            return
        
        self.status = StreamStatus.ACTIVE
        self._pause_event.set()
        await self._emit_event("session_resumed", {"session_id": self.session_id})
    
    async def send_message(self, message: StreamMessage) -> None:
        """发送消息"""
        await self._pause_event.wait()  # 等待非暂停状态
        
        if self.status != StreamStatus.ACTIVE:
            raise ValueError(f"无法发送消息，流会话状态: {self.status}")
        
        # 添加到缓冲区
        self.buffer.add_message(message)
        
        # 更新指标
        self.metrics.messages_sent += 1
        if hasattr(message.content, '__len__'):
            self.metrics.bytes_sent += len(str(message.content))
        
        # 通知订阅者
        await self._notify_subscribers(message)
        
        # 触发事件
        await self._emit_event("message_sent", {
            "message_id": message.id,
            "type": message.type,
            "sequence": message.sequence
        })
    
    async def send_chunk(self, chunk: StreamChunk) -> None:
        """发送数据块"""
        await self._pause_event.wait()
        
        if self.status != StreamStatus.ACTIVE:
            raise ValueError(f"无法发送数据块，流会话状态: {self.status}")
        
        # 更新指标
        self.metrics.chunks_sent += 1
        self.metrics.bytes_sent += chunk.size
        
        # 触发事件
        await self._emit_event("chunk_sent", {
            "chunk_id": chunk.id,
            "size": chunk.size,
            "is_last": chunk.is_last
        })
    
    async def subscribe(
        self,
        subscriber_id: str,
        filter: Optional[StreamFilter] = None
    ) -> str:
        """订阅流"""
        subscription_id = str(uuid.uuid4())
        
        subscription = StreamSubscription(
            id=subscription_id,
            stream_id=self.session_id,
            subscriber_id=subscriber_id,
            filter=filter
        )
        
        self.subscriptions[subscription_id] = subscription
        
        await self._emit_event("subscription_created", {
            "subscription_id": subscription_id,
            "subscriber_id": subscriber_id
        })
        
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            
            await self._emit_event("subscription_removed", {
                "subscription_id": subscription_id
            })
            
            return True
        return False
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """添加事件处理器"""
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> None:
        """移除事件处理器"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    async def get_messages(
        self,
        count: Optional[int] = None,
        filter: Optional[StreamFilter] = None
    ) -> List[StreamMessage]:
        """获取消息"""
        messages = self.buffer.get_messages(count)
        
        if filter:
            # 简化过滤逻辑，实际应用中可能需要更复杂的实现
            filtered_messages = []
            for msg in messages:
                # 创建临时事件用于过滤
                temp_event = StreamEvent(
                    id=msg.id,
                    type=msg.type.value,
                    data=msg.content,
                    timestamp=msg.timestamp,
                    metadata=msg.metadata
                )
                if filter.matches(temp_event):
                    filtered_messages.append(msg)
            return filtered_messages
        
        return messages
    
    def get_metrics(self) -> StreamMetrics:
        """获取指标"""
        # 更新当前指标
        if self.status == StreamStatus.ACTIVE:
            self.metrics.duration = (datetime.now() - self.metrics.start_time).total_seconds()
        
        return self.metrics
    
    async def _notify_subscribers(self, message: StreamMessage) -> None:
        """通知订阅者"""
        for subscription in self.subscriptions.values():
            if not subscription.is_active:
                continue
            
            # 应用过滤器
            if subscription.filter:
                temp_event = StreamEvent(
                    id=message.id,
                    type=message.type.value,
                    data=message.content,
                    timestamp=message.timestamp,
                    metadata=message.metadata
                )
                if not subscription.filter.matches(temp_event):
                    continue
            
            # 更新订阅活动时间
            subscription.last_activity = datetime.now()
            
            # 通知管理器
            manager = self.manager()
            if manager:
                await manager._notify_subscriber(subscription.subscriber_id, message)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """触发事件"""
        event = StreamEvent(
            id=str(uuid.uuid4()),
            type=event_type,
            data=data,
            metadata={"session_id": self.session_id}
        )
        
        # 调用事件处理器
        for handler in self.event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                print(f"事件处理器错误: {e}")
        
        # 通知管理器
        manager = self.manager()
        if manager:
            await manager._handle_session_event(event)


class StreamManager:
    """增强的流管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, StreamSession] = {}
        self.global_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.subscriber_handlers: Dict[str, Callable] = {}
        
        # 全局配置
        self.default_config = StreamConfig(
            stream_id="default",
            mode=StreamMode.SSE,
            buffer_size=1000,
            max_chunk_size=1024*1024,
            timeout=300,
            retry_attempts=3
        )
        
        # 统计信息
        self.total_sessions_created = 0
        self.active_sessions_count = 0
        
        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    async def create_session(
        self,
        session_id: Optional[str] = None,
        config: Optional[StreamConfig] = None
    ) -> StreamSession:
        """创建流会话"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id in self.sessions:
            raise ValueError(f"会话已存在: {session_id}")
        
        # 使用提供的配置或默认配置
        session_config = config or self.default_config.copy()
        session_config.stream_id = session_id
        
        # 创建会话
        session = StreamSession(session_id, session_config, self)
        self.sessions[session_id] = session
        
        # 更新统计
        self.total_sessions_created += 1
        self.active_sessions_count += 1
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[StreamSession]:
        """获取流会话"""
        return self.sessions.get(session_id)
    
    async def remove_session(self, session_id: str) -> bool:
        """移除流会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            await session.stop()
            del self.sessions[session_id]
            self.active_sessions_count -= 1
            return True
        return False
    
    async def list_sessions(
        self,
        status_filter: Optional[StreamStatus] = None
    ) -> List[Dict[str, Any]]:
        """列出流会话"""
        sessions_info = []
        
        for session in self.sessions.values():
            if status_filter is None or session.status == status_filter:
                sessions_info.append({
                    "session_id": session.session_id,
                    "status": session.status,
                    "created_at": session.created_at,
                    "started_at": session.started_at,
                    "config": session.config.dict(),
                    "metrics": session.get_metrics().dict(),
                    "subscription_count": len(session.subscriptions)
                })
        
        return sessions_info
    
    async def broadcast_message(
        self,
        message: StreamMessage,
        session_filter: Optional[Callable[[StreamSession], bool]] = None
    ) -> int:
        """广播消息到多个会话"""
        sent_count = 0
        
        for session in self.sessions.values():
            if session.status != StreamStatus.ACTIVE:
                continue
            
            if session_filter and not session_filter(session):
                continue
            
            try:
                await session.send_message(message)
                sent_count += 1
            except Exception as e:
                print(f"广播消息到会话 {session.session_id} 失败: {e}")
        
        return sent_count
    
    async def create_stream_iterator(
        self,
        session_id: str,
        message_filter: Optional[StreamFilter] = None
    ) -> AsyncIterator[StreamMessage]:
        """创建流迭代器"""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"会话不存在: {session_id}")
        
        # 简化实现，实际应用中需要更复杂的流处理
        while session.status == StreamStatus.ACTIVE:
            messages = await session.get_messages(count=10, filter=message_filter)
            for message in messages:
                yield message
            
            await asyncio.sleep(0.1)  # 避免过于频繁的轮询
    
    def register_global_handler(self, event_type: str, handler: Callable) -> None:
        """注册全局事件处理器"""
        self.global_handlers[event_type].append(handler)
    
    def register_subscriber_handler(self, subscriber_id: str, handler: Callable) -> None:
        """注册订阅者处理器"""
        self.subscriber_handlers[subscriber_id] = handler
    
    async def get_global_metrics(self) -> Dict[str, Any]:
        """获取全局指标"""
        total_messages = 0
        total_bytes = 0
        total_errors = 0
        
        for session in self.sessions.values():
            metrics = session.get_metrics()
            total_messages += metrics.messages_sent + metrics.messages_received
            total_bytes += metrics.bytes_sent + metrics.bytes_received
            total_errors += metrics.errors
        
        return {
            "total_sessions_created": self.total_sessions_created,
            "active_sessions": self.active_sessions_count,
            "total_messages": total_messages,
            "total_bytes": total_bytes,
            "total_errors": total_errors,
            "sessions_by_status": self._get_sessions_by_status()
        }
    
    async def cleanup_inactive_sessions(self, max_age_hours: int = 24) -> int:
        """清理非活跃会话"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if (session.status in [StreamStatus.COMPLETED, StreamStatus.FAILED, StreamStatus.CANCELLED] and
                session.created_at.timestamp() < cutoff_time):
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            await self.remove_session(session_id)
        
        return len(sessions_to_remove)
    
    async def _notify_subscriber(self, subscriber_id: str, message: StreamMessage) -> None:
        """通知订阅者"""
        if subscriber_id in self.subscriber_handlers:
            handler = self.subscriber_handlers[subscriber_id]
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                print(f"订阅者 {subscriber_id} 处理器错误: {e}")
    
    async def _handle_session_event(self, event: StreamEvent) -> None:
        """处理会话事件"""
        # 调用全局事件处理器
        for handler in self.global_handlers.get(event.type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                print(f"全局事件处理器错误: {e}")
    
    def _get_sessions_by_status(self) -> Dict[str, int]:
        """按状态统计会话数量"""
        status_counts = defaultdict(int)
        for session in self.sessions.values():
            status_counts[session.status.value] += 1
        return dict(status_counts)
    
    def _start_cleanup_task(self) -> None:
        """启动清理任务"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(3600)  # 每小时清理一次
                    await self.cleanup_inactive_sessions()
                except Exception as e:
                    print(f"清理任务错误: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def shutdown(self) -> None:
        """关闭管理器"""
        # 停止所有会话
        for session in list(self.sessions.values()):
            await session.stop()
        
        # 取消清理任务
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.sessions.clear()