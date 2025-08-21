"""事件处理器模块

提供各种业务事件的处理逻辑。
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .event_bus import Event, AsyncEventHandler
from .event_store import get_event_store
from ..utils.cache import get_cache_manager
from ..utils.performance_monitoring import get_metrics_collector
from ..utils.validation import ValidationException

logger = logging.getLogger(__name__)


class BaseEventHandler(AsyncEventHandler):
    """基础事件处理器"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.event_store = get_event_store()
        self.cache_manager = get_cache_manager()
        self.metrics_collector = get_metrics_collector()
    
    async def handle(self, event: Event) -> None:
        """处理事件"""
        try:
            # 记录性能指标
            start_time = datetime.now()
            
            # 验证事件
            await self.validate_event(event)
            
            # 处理事件
            await self.process_event(event)
            
            # 记录处理时间
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics_collector.record_custom_metric(
                name=f"event_handler_{self.name}_duration",
                value=duration,
                tags={"event_type": event.event_type, "handler": self.name}
            )
            
            logger.debug(f"Event {event.event_id} processed by {self.name}")
        
        except Exception as e:
            logger.error(f"Failed to process event {event.event_id} in {self.name}: {e}")
            
            # 记录错误指标
            self.metrics_collector.record_custom_metric(
                name=f"event_handler_{self.name}_errors",
                value=1,
                tags={"event_type": event.event_type, "handler": self.name, "error": str(e)}
            )
            
            # 可以选择重新抛出异常或记录错误
            raise
    
    async def validate_event(self, event: Event) -> None:
        """验证事件"""
        if not event.event_id:
            raise ValidationException("Event ID is required")
        
        if not event.event_type:
            raise ValidationException("Event type is required")
        
        if not event.timestamp:
            raise ValidationException("Event timestamp is required")
    
    @abstractmethod
    async def process_event(self, event: Event) -> None:
        """处理事件的具体逻辑"""
        pass


class UserEventHandler(BaseEventHandler):
    """用户事件处理器"""
    
    def __init__(self):
        super().__init__("UserEventHandler")
        self.supported_events = {
            "user.registered",
            "user.login",
            "user.logout",
            "user.profile_updated",
            "user.password_changed",
            "user.deleted"
        }
    
    def can_handle(self, event: Event) -> bool:
        """检查是否可以处理该事件"""
        return event.event_type in self.supported_events
    
    async def process_event(self, event: Event) -> None:
        """处理用户事件"""
        if event.event_type == "user.registered":
            await self._handle_user_registered(event)
        elif event.event_type == "user.login":
            await self._handle_user_login(event)
        elif event.event_type == "user.logout":
            await self._handle_user_logout(event)
        elif event.event_type == "user.profile_updated":
            await self._handle_user_profile_updated(event)
        elif event.event_type == "user.password_changed":
            await self._handle_user_password_changed(event)
        elif event.event_type == "user.deleted":
            await self._handle_user_deleted(event)
    
    async def _handle_user_registered(self, event: Event) -> None:
        """处理用户注册事件"""
        user_data = event.data
        user_id = user_data.get("user_id")
        
        logger.info(f"User {user_id} registered")
        
        # 清除相关缓存
        await self._clear_user_cache(user_id)
        
        # 可以在这里添加其他逻辑，如发送欢迎邮件等
    
    async def _handle_user_login(self, event: Event) -> None:
        """处理用户登录事件"""
        user_data = event.data
        user_id = user_data.get("user_id")
        
        logger.info(f"User {user_id} logged in")
        
        # 更新用户最后登录时间缓存
        self.cache_manager.set(
            f"user:{user_id}:last_login",
            datetime.now().isoformat(),
            ttl=86400  # 24小时
        )
    
    async def _handle_user_logout(self, event: Event) -> None:
        """处理用户登出事件"""
        user_data = event.data
        user_id = user_data.get("user_id")
        session_id = user_data.get("session_id")
        
        logger.info(f"User {user_id} logged out")
        
        # 清除会话缓存
        if session_id:
            self.cache_manager.delete(f"session:{session_id}")
    
    async def _handle_user_profile_updated(self, event: Event) -> None:
        """处理用户资料更新事件"""
        user_data = event.data
        user_id = user_data.get("user_id")
        
        logger.info(f"User {user_id} profile updated")
        
        # 清除用户相关缓存
        await self._clear_user_cache(user_id)
    
    async def _handle_user_password_changed(self, event: Event) -> None:
        """处理用户密码更改事件"""
        user_data = event.data
        user_id = user_data.get("user_id")
        
        logger.info(f"User {user_id} password changed")
        
        # 清除所有用户会话
        self.cache_manager.clear_by_tags([f"user:{user_id}"])
    
    async def _handle_user_deleted(self, event: Event) -> None:
        """处理用户删除事件"""
        user_data = event.data
        user_id = user_data.get("user_id")
        
        logger.info(f"User {user_id} deleted")
        
        # 清除所有用户相关数据
        await self._clear_user_cache(user_id)
    
    async def _clear_user_cache(self, user_id: str) -> None:
        """清除用户相关缓存"""
        cache_keys = [
            f"user:{user_id}",
            f"user:{user_id}:profile",
            f"user:{user_id}:permissions",
            f"user:{user_id}:last_login"
        ]
        
        for key in cache_keys:
            self.cache_manager.delete(key)
        
        # 按标签清除
        self.cache_manager.clear_by_tags([f"user:{user_id}"])


class SessionEventHandler(BaseEventHandler):
    """会话事件处理器"""
    
    def __init__(self):
        super().__init__("SessionEventHandler")
        self.supported_events = {
            "session.created",
            "session.updated",
            "session.deleted",
            "session.expired"
        }
    
    def can_handle(self, event: Event) -> bool:
        """检查是否可以处理该事件"""
        return event.event_type in self.supported_events
    
    async def process_event(self, event: Event) -> None:
        """处理会话事件"""
        if event.event_type == "session.created":
            await self._handle_session_created(event)
        elif event.event_type == "session.updated":
            await self._handle_session_updated(event)
        elif event.event_type == "session.deleted":
            await self._handle_session_deleted(event)
        elif event.event_type == "session.expired":
            await self._handle_session_expired(event)
    
    async def _handle_session_created(self, event: Event) -> None:
        """处理会话创建事件"""
        session_data = event.data
        session_id = session_data.get("session_id")
        user_id = session_data.get("user_id")
        
        logger.info(f"Session {session_id} created for user {user_id}")
        
        # 缓存会话信息
        self.cache_manager.set(
            f"session:{session_id}",
            session_data,
            ttl=3600,  # 1小时
            tags=[f"user:{user_id}", "session"]
        )
    
    async def _handle_session_updated(self, event: Event) -> None:
        """处理会话更新事件"""
        session_data = event.data
        session_id = session_data.get("session_id")
        
        logger.info(f"Session {session_id} updated")
        
        # 更新缓存
        existing_data = self.cache_manager.get(f"session:{session_id}")
        if existing_data:
            existing_data.update(session_data)
            self.cache_manager.set(
                f"session:{session_id}",
                existing_data,
                ttl=3600
            )
    
    async def _handle_session_deleted(self, event: Event) -> None:
        """处理会话删除事件"""
        session_data = event.data
        session_id = session_data.get("session_id")
        
        logger.info(f"Session {session_id} deleted")
        
        # 删除缓存
        self.cache_manager.delete(f"session:{session_id}")
    
    async def _handle_session_expired(self, event: Event) -> None:
        """处理会话过期事件"""
        session_data = event.data
        session_id = session_data.get("session_id")
        
        logger.info(f"Session {session_id} expired")
        
        # 删除缓存
        self.cache_manager.delete(f"session:{session_id}")


class MessageEventHandler(BaseEventHandler):
    """消息事件处理器"""
    
    def __init__(self):
        super().__init__("MessageEventHandler")
        self.supported_events = {
            "message.sent",
            "message.received",
            "message.updated",
            "message.deleted"
        }
    
    def can_handle(self, event: Event) -> bool:
        """检查是否可以处理该事件"""
        return event.event_type in self.supported_events
    
    async def process_event(self, event: Event) -> None:
        """处理消息事件"""
        if event.event_type == "message.sent":
            await self._handle_message_sent(event)
        elif event.event_type == "message.received":
            await self._handle_message_received(event)
        elif event.event_type == "message.updated":
            await self._handle_message_updated(event)
        elif event.event_type == "message.deleted":
            await self._handle_message_deleted(event)
    
    async def _handle_message_sent(self, event: Event) -> None:
        """处理消息发送事件"""
        message_data = event.data
        message_id = message_data.get("message_id")
        thread_id = message_data.get("thread_id")
        user_id = message_data.get("user_id")
        
        logger.info(f"Message {message_id} sent in thread {thread_id}")
        
        # 更新线程最后活动时间
        if thread_id:
            self.cache_manager.set(
                f"thread:{thread_id}:last_activity",
                datetime.now().isoformat(),
                ttl=86400,
                tags=[f"thread:{thread_id}", f"user:{user_id}"]
            )
        
        # 清除相关缓存
        cache_keys = [
            f"thread:{thread_id}:messages",
            f"user:{user_id}:recent_messages"
        ]
        
        for key in cache_keys:
            self.cache_manager.delete(key)
    
    async def _handle_message_received(self, event: Event) -> None:
        """处理消息接收事件"""
        message_data = event.data
        message_id = message_data.get("message_id")
        thread_id = message_data.get("thread_id")
        
        logger.info(f"Message {message_id} received in thread {thread_id}")
        
        # 清除相关缓存
        if thread_id:
            self.cache_manager.delete(f"thread:{thread_id}:messages")
    
    async def _handle_message_updated(self, event: Event) -> None:
        """处理消息更新事件"""
        message_data = event.data
        message_id = message_data.get("message_id")
        thread_id = message_data.get("thread_id")
        
        logger.info(f"Message {message_id} updated")
        
        # 清除相关缓存
        cache_keys = [
            f"message:{message_id}",
            f"thread:{thread_id}:messages"
        ]
        
        for key in cache_keys:
            self.cache_manager.delete(key)
    
    async def _handle_message_deleted(self, event: Event) -> None:
        """处理消息删除事件"""
        message_data = event.data
        message_id = message_data.get("message_id")
        thread_id = message_data.get("thread_id")
        
        logger.info(f"Message {message_id} deleted")
        
        # 清除相关缓存
        cache_keys = [
            f"message:{message_id}",
            f"thread:{thread_id}:messages"
        ]
        
        for key in cache_keys:
            self.cache_manager.delete(key)


class WorkflowEventHandler(BaseEventHandler):
    """工作流事件处理器"""
    
    def __init__(self):
        super().__init__("WorkflowEventHandler")
        self.supported_events = {
            "workflow.created",
            "workflow.started",
            "workflow.completed",
            "workflow.failed",
            "workflow.cancelled",
            "workflow.step_completed",
            "workflow.step_failed"
        }
    
    def can_handle(self, event: Event) -> bool:
        """检查是否可以处理该事件"""
        return event.event_type in self.supported_events
    
    async def process_event(self, event: Event) -> None:
        """处理工作流事件"""
        if event.event_type == "workflow.created":
            await self._handle_workflow_created(event)
        elif event.event_type == "workflow.started":
            await self._handle_workflow_started(event)
        elif event.event_type == "workflow.completed":
            await self._handle_workflow_completed(event)
        elif event.event_type == "workflow.failed":
            await self._handle_workflow_failed(event)
        elif event.event_type == "workflow.cancelled":
            await self._handle_workflow_cancelled(event)
        elif event.event_type == "workflow.step_completed":
            await self._handle_workflow_step_completed(event)
        elif event.event_type == "workflow.step_failed":
            await self._handle_workflow_step_failed(event)
    
    async def _handle_workflow_created(self, event: Event) -> None:
        """处理工作流创建事件"""
        workflow_data = event.data
        workflow_id = workflow_data.get("workflow_id")
        user_id = workflow_data.get("user_id")
        
        logger.info(f"Workflow {workflow_id} created by user {user_id}")
        
        # 缓存工作流信息
        self.cache_manager.set(
            f"workflow:{workflow_id}",
            workflow_data,
            ttl=3600,
            tags=[f"user:{user_id}", "workflow"]
        )
    
    async def _handle_workflow_started(self, event: Event) -> None:
        """处理工作流开始事件"""
        workflow_data = event.data
        workflow_id = workflow_data.get("workflow_id")
        
        logger.info(f"Workflow {workflow_id} started")
        
        # 更新工作流状态缓存
        self.cache_manager.set(
            f"workflow:{workflow_id}:status",
            "running",
            ttl=3600
        )
    
    async def _handle_workflow_completed(self, event: Event) -> None:
        """处理工作流完成事件"""
        workflow_data = event.data
        workflow_id = workflow_data.get("workflow_id")
        
        logger.info(f"Workflow {workflow_id} completed")
        
        # 更新工作流状态缓存
        self.cache_manager.set(
            f"workflow:{workflow_id}:status",
            "completed",
            ttl=86400  # 保留24小时
        )
        
        # 清除运行时缓存
        self.cache_manager.delete(f"workflow:{workflow_id}:runtime")
    
    async def _handle_workflow_failed(self, event: Event) -> None:
        """处理工作流失败事件"""
        workflow_data = event.data
        workflow_id = workflow_data.get("workflow_id")
        error = workflow_data.get("error")
        
        logger.error(f"Workflow {workflow_id} failed: {error}")
        
        # 更新工作流状态缓存
        self.cache_manager.set(
            f"workflow:{workflow_id}:status",
            "failed",
            ttl=86400
        )
        
        # 记录错误信息
        self.cache_manager.set(
            f"workflow:{workflow_id}:error",
            error,
            ttl=86400
        )
    
    async def _handle_workflow_cancelled(self, event: Event) -> None:
        """处理工作流取消事件"""
        workflow_data = event.data
        workflow_id = workflow_data.get("workflow_id")
        
        logger.info(f"Workflow {workflow_id} cancelled")
        
        # 更新工作流状态缓存
        self.cache_manager.set(
            f"workflow:{workflow_id}:status",
            "cancelled",
            ttl=86400
        )
        
        # 清除运行时缓存
        self.cache_manager.delete(f"workflow:{workflow_id}:runtime")
    
    async def _handle_workflow_step_completed(self, event: Event) -> None:
        """处理工作流步骤完成事件"""
        step_data = event.data
        workflow_id = step_data.get("workflow_id")
        step_id = step_data.get("step_id")
        
        logger.info(f"Workflow {workflow_id} step {step_id} completed")
        
        # 更新步骤状态缓存
        self.cache_manager.set(
            f"workflow:{workflow_id}:step:{step_id}:status",
            "completed",
            ttl=3600
        )
    
    async def _handle_workflow_step_failed(self, event: Event) -> None:
        """处理工作流步骤失败事件"""
        step_data = event.data
        workflow_id = step_data.get("workflow_id")
        step_id = step_data.get("step_id")
        error = step_data.get("error")
        
        logger.error(f"Workflow {workflow_id} step {step_id} failed: {error}")
        
        # 更新步骤状态缓存
        self.cache_manager.set(
            f"workflow:{workflow_id}:step:{step_id}:status",
            "failed",
            ttl=3600
        )
        
        # 记录错误信息
        self.cache_manager.set(
            f"workflow:{workflow_id}:step:{step_id}:error",
            error,
            ttl=3600
        )


class MemoryEventHandler(BaseEventHandler):
    """记忆事件处理器"""
    
    def __init__(self):
        super().__init__("MemoryEventHandler")
        self.supported_events = {
            "memory.created",
            "memory.updated",
            "memory.deleted",
            "memory.accessed"
        }
    
    def can_handle(self, event: Event) -> bool:
        """检查是否可以处理该事件"""
        return event.event_type in self.supported_events
    
    async def process_event(self, event: Event) -> None:
        """处理记忆事件"""
        if event.event_type == "memory.created":
            await self._handle_memory_created(event)
        elif event.event_type == "memory.updated":
            await self._handle_memory_updated(event)
        elif event.event_type == "memory.deleted":
            await self._handle_memory_deleted(event)
        elif event.event_type == "memory.accessed":
            await self._handle_memory_accessed(event)
    
    async def _handle_memory_created(self, event: Event) -> None:
        """处理记忆创建事件"""
        memory_data = event.data
        memory_id = memory_data.get("memory_id")
        user_id = memory_data.get("user_id")
        memory_type = memory_data.get("memory_type")
        
        logger.info(f"Memory {memory_id} of type {memory_type} created for user {user_id}")
        
        # 清除相关缓存
        cache_keys = [
            f"user:{user_id}:memories",
            f"user:{user_id}:memories:{memory_type}"
        ]
        
        for key in cache_keys:
            self.cache_manager.delete(key)
    
    async def _handle_memory_updated(self, event: Event) -> None:
        """处理记忆更新事件"""
        memory_data = event.data
        memory_id = memory_data.get("memory_id")
        user_id = memory_data.get("user_id")
        memory_type = memory_data.get("memory_type")
        
        logger.info(f"Memory {memory_id} updated")
        
        # 清除相关缓存
        cache_keys = [
            f"memory:{memory_id}",
            f"user:{user_id}:memories",
            f"user:{user_id}:memories:{memory_type}"
        ]
        
        for key in cache_keys:
            self.cache_manager.delete(key)
    
    async def _handle_memory_deleted(self, event: Event) -> None:
        """处理记忆删除事件"""
        memory_data = event.data
        memory_id = memory_data.get("memory_id")
        user_id = memory_data.get("user_id")
        memory_type = memory_data.get("memory_type")
        
        logger.info(f"Memory {memory_id} deleted")
        
        # 清除相关缓存
        cache_keys = [
            f"memory:{memory_id}",
            f"user:{user_id}:memories",
            f"user:{user_id}:memories:{memory_type}"
        ]
        
        for key in cache_keys:
            self.cache_manager.delete(key)
    
    async def _handle_memory_accessed(self, event: Event) -> None:
        """处理记忆访问事件"""
        memory_data = event.data
        memory_id = memory_data.get("memory_id")
        user_id = memory_data.get("user_id")
        
        logger.debug(f"Memory {memory_id} accessed by user {user_id}")
        
        # 更新访问时间缓存
        self.cache_manager.set(
            f"memory:{memory_id}:last_accessed",
            datetime.now().isoformat(),
            ttl=86400
        )


# 便捷函数
def create_default_handlers() -> List[BaseEventHandler]:
    """创建默认事件处理器列表"""
    return [
        UserEventHandler(),
        SessionEventHandler(),
        MessageEventHandler(),
        WorkflowEventHandler(),
        MemoryEventHandler()
    ]


def register_default_handlers(event_bus) -> None:
    """注册默认事件处理器到事件总线"""
    handlers = create_default_handlers()
    
    for handler in handlers:
        # 为每个处理器注册其支持的事件类型
        for event_type in handler.supported_events:
            event_bus.subscribe(
                event_type=event_type,
                handler=handler,
                priority=1  # 默认优先级
            )
    
    logger.info(f"Registered {len(handlers)} default event handlers")