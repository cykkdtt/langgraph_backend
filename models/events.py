"""模型事件系统模块

本模块提供模型生命周期事件的监听、触发和处理功能。
"""

from typing import (
    Dict, Any, Optional, List, Callable, Type, Union, 
    get_type_hints, Awaitable
)
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps
import asyncio
import threading
import weakref
import logging
from contextlib import contextmanager
from collections import defaultdict, deque
from sqlalchemy import event
from sqlalchemy.orm import Session
from sqlalchemy.orm.events import InstanceEvents

# 导入项目模型
from .models import (
    User, Session as ChatSession, Thread, Message, Workflow, 
    WorkflowExecution, WorkflowStep, Memory, MemoryVector, 
    TimeTravel, Attachment, UserPreference, SystemConfig
)


logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型枚举"""
    # 模型生命周期事件
    BEFORE_CREATE = "before_create"
    AFTER_CREATE = "after_create"
    BEFORE_UPDATE = "before_update"
    AFTER_UPDATE = "after_update"
    BEFORE_DELETE = "before_delete"
    AFTER_DELETE = "after_delete"
    
    # 数据库事件
    BEFORE_FLUSH = "before_flush"
    AFTER_FLUSH = "after_flush"
    BEFORE_COMMIT = "before_commit"
    AFTER_COMMIT = "after_commit"
    BEFORE_ROLLBACK = "before_rollback"
    AFTER_ROLLBACK = "after_rollback"
    
    # 业务事件
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    MEMORY_CREATED = "memory_created"
    MEMORY_RETRIEVED = "memory_retrieved"
    TIME_TRAVEL_EXECUTED = "time_travel_executed"
    
    # 系统事件
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGED = "config_changed"
    ERROR_OCCURRED = "error_occurred"


class EventPriority(Enum):
    """事件优先级枚举"""
    HIGHEST = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    LOWEST = 5


class EventStatus(Enum):
    """事件状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EventContext:
    """事件上下文"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source: str
    target: Optional[Any] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session: Optional[Session] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    status: EventStatus = EventStatus.PENDING
    error: Optional[Exception] = None
    result: Optional[Any] = None
    duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'target': str(self.target) if self.target else None,
            'data': self.data,
            'metadata': self.metadata,
            'user_id': self.user_id,
            'request_id': self.request_id,
            'correlation_id': self.correlation_id,
            'parent_event_id': self.parent_event_id,
            'priority': self.priority.value,
            'status': self.status.value,
            'error': str(self.error) if self.error else None,
            'result': self.result,
            'duration': self.duration
        }


class EventHandler(ABC):
    """事件处理器基类"""
    
    def __init__(self, name: str, priority: EventPriority = EventPriority.NORMAL):
        self.name = name
        self.priority = priority
        self.enabled = True
    
    @abstractmethod
    def handle(self, context: EventContext) -> Any:
        """处理事件"""
        pass
    
    def can_handle(self, context: EventContext) -> bool:
        """检查是否可以处理事件"""
        return self.enabled
    
    def __lt__(self, other):
        """用于优先级排序"""
        return self.priority.value < other.priority.value


class AsyncEventHandler(EventHandler):
    """异步事件处理器"""
    
    @abstractmethod
    async def handle_async(self, context: EventContext) -> Any:
        """异步处理事件"""
        pass
    
    def handle(self, context: EventContext) -> Any:
        """同步处理事件（调用异步方法）"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.handle_async(context))


class FunctionEventHandler(EventHandler):
    """函数事件处理器"""
    
    def __init__(self, name: str, func: Callable[[EventContext], Any], 
                 priority: EventPriority = EventPriority.NORMAL):
        super().__init__(name, priority)
        self.func = func
    
    def handle(self, context: EventContext) -> Any:
        """处理事件"""
        return self.func(context)


class ConditionalEventHandler(EventHandler):
    """条件事件处理器"""
    
    def __init__(self, name: str, handler: EventHandler, 
                 condition: Callable[[EventContext], bool]):
        super().__init__(name, handler.priority)
        self.handler = handler
        self.condition = condition
    
    def can_handle(self, context: EventContext) -> bool:
        """检查是否可以处理事件"""
        return self.enabled and self.condition(context)
    
    def handle(self, context: EventContext) -> Any:
        """处理事件"""
        return self.handler.handle(context)


class EventBus:
    """事件总线"""
    
    def __init__(self, max_history: int = 1000):
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
        self._event_history: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._enabled = True
        self._middleware: List[Callable] = []
        self._filters: List[Callable[[EventContext], bool]] = []
        self._interceptors: List[Callable[[EventContext], EventContext]] = []
    
    def register_handler(self, event_type: EventType, handler: EventHandler) -> None:
        """注册事件处理器"""
        with self._lock:
            self._handlers[event_type].append(handler)
            # 按优先级排序
            self._handlers[event_type].sort()
    
    def register_global_handler(self, handler: EventHandler) -> None:
        """注册全局事件处理器"""
        with self._lock:
            self._global_handlers.append(handler)
            self._global_handlers.sort()
    
    def unregister_handler(self, event_type: EventType, handler: EventHandler) -> None:
        """注销事件处理器"""
        with self._lock:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
    
    def unregister_global_handler(self, handler: EventHandler) -> None:
        """注销全局事件处理器"""
        with self._lock:
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)
    
    def add_middleware(self, middleware: Callable) -> None:
        """添加中间件"""
        self._middleware.append(middleware)
    
    def add_filter(self, filter_func: Callable[[EventContext], bool]) -> None:
        """添加事件过滤器"""
        self._filters.append(filter_func)
    
    def add_interceptor(self, interceptor: Callable[[EventContext], EventContext]) -> None:
        """添加事件拦截器"""
        self._interceptors.append(interceptor)
    
    def emit(self, context: EventContext) -> List[Any]:
        """发布事件"""
        if not self._enabled:
            return []
        
        # 应用过滤器
        for filter_func in self._filters:
            if not filter_func(context):
                return []
        
        # 应用拦截器
        for interceptor in self._interceptors:
            context = interceptor(context)
        
        # 记录事件历史
        self._event_history.append(context)
        
        context.status = EventStatus.PROCESSING
        start_time = datetime.now()
        
        results = []
        
        try:
            # 获取处理器
            handlers = []
            
            # 特定事件类型的处理器
            if context.event_type in self._handlers:
                handlers.extend(self._handlers[context.event_type])
            
            # 全局处理器
            handlers.extend(self._global_handlers)
            
            # 按优先级排序
            handlers.sort()
            
            # 执行处理器
            for handler in handlers:
                if handler.can_handle(context):
                    try:
                        # 应用中间件
                        for middleware in self._middleware:
                            middleware(context, handler)
                        
                        result = handler.handle(context)
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error in event handler {handler.name}: {e}")
                        context.error = e
                        context.status = EventStatus.FAILED
                        raise
            
            context.status = EventStatus.COMPLETED
            context.result = results
            
        except Exception as e:
            context.status = EventStatus.FAILED
            context.error = e
            logger.error(f"Error processing event {context.event_type}: {e}")
            raise
        
        finally:
            end_time = datetime.now()
            context.duration = (end_time - start_time).total_seconds()
        
        return results
    
    def emit_async(self, context: EventContext) -> Awaitable[List[Any]]:
        """异步发布事件"""
        async def _emit_async():
            return self.emit(context)
        
        return _emit_async()
    
    def get_event_history(self, event_type: Optional[EventType] = None, 
                         limit: Optional[int] = None) -> List[EventContext]:
        """获取事件历史"""
        history = list(self._event_history)
        
        if event_type:
            history = [ctx for ctx in history if ctx.event_type == event_type]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def clear_history(self) -> None:
        """清空事件历史"""
        self._event_history.clear()
    
    def enable(self) -> None:
        """启用事件总线"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用事件总线"""
        self._enabled = False
    
    def get_handler_count(self, event_type: Optional[EventType] = None) -> int:
        """获取处理器数量"""
        if event_type:
            return len(self._handlers.get(event_type, []))
        else:
            return sum(len(handlers) for handlers in self._handlers.values()) + len(self._global_handlers)


class ModelEventManager:
    """模型事件管理器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._model_listeners: Dict[Type, List[Callable]] = defaultdict(list)
        self._setup_sqlalchemy_events()
    
    def _setup_sqlalchemy_events(self) -> None:
        """设置SQLAlchemy事件监听"""
        # 注册SQLAlchemy事件
        @event.listens_for(Session, 'before_flush')
        def before_flush(session, flush_context, instances):
            context = EventContext(
                event_id=f"flush_{id(session)}_{datetime.now().timestamp()}",
                event_type=EventType.BEFORE_FLUSH,
                timestamp=datetime.now(),
                source="sqlalchemy",
                session=session,
                data={'instances': instances}
            )
            self.event_bus.emit(context)
        
        @event.listens_for(Session, 'after_flush')
        def after_flush(session, flush_context):
            context = EventContext(
                event_id=f"flush_{id(session)}_{datetime.now().timestamp()}",
                event_type=EventType.AFTER_FLUSH,
                timestamp=datetime.now(),
                source="sqlalchemy",
                session=session
            )
            self.event_bus.emit(context)
        
        @event.listens_for(Session, 'before_commit')
        def before_commit(session):
            context = EventContext(
                event_id=f"commit_{id(session)}_{datetime.now().timestamp()}",
                event_type=EventType.BEFORE_COMMIT,
                timestamp=datetime.now(),
                source="sqlalchemy",
                session=session
            )
            self.event_bus.emit(context)
        
        @event.listens_for(Session, 'after_commit')
        def after_commit(session):
            context = EventContext(
                event_id=f"commit_{id(session)}_{datetime.now().timestamp()}",
                event_type=EventType.AFTER_COMMIT,
                timestamp=datetime.now(),
                source="sqlalchemy",
                session=session
            )
            self.event_bus.emit(context)
    
    def register_model_listener(self, model_class: Type, event_type: EventType, 
                               listener: Callable) -> None:
        """注册模型监听器"""
        # 创建SQLAlchemy事件监听器
        if event_type == EventType.BEFORE_CREATE:
            @event.listens_for(model_class, 'before_insert')
            def before_insert(mapper, connection, target):
                context = EventContext(
                    event_id=f"create_{model_class.__name__}_{id(target)}_{datetime.now().timestamp()}",
                    event_type=EventType.BEFORE_CREATE,
                    timestamp=datetime.now(),
                    source=model_class.__name__,
                    target=target
                )
                listener(context)
        
        elif event_type == EventType.AFTER_CREATE:
            @event.listens_for(model_class, 'after_insert')
            def after_insert(mapper, connection, target):
                context = EventContext(
                    event_id=f"create_{model_class.__name__}_{id(target)}_{datetime.now().timestamp()}",
                    event_type=EventType.AFTER_CREATE,
                    timestamp=datetime.now(),
                    source=model_class.__name__,
                    target=target
                )
                listener(context)
        
        elif event_type == EventType.BEFORE_UPDATE:
            @event.listens_for(model_class, 'before_update')
            def before_update(mapper, connection, target):
                context = EventContext(
                    event_id=f"update_{model_class.__name__}_{id(target)}_{datetime.now().timestamp()}",
                    event_type=EventType.BEFORE_UPDATE,
                    timestamp=datetime.now(),
                    source=model_class.__name__,
                    target=target
                )
                listener(context)
        
        elif event_type == EventType.AFTER_UPDATE:
            @event.listens_for(model_class, 'after_update')
            def after_update(mapper, connection, target):
                context = EventContext(
                    event_id=f"update_{model_class.__name__}_{id(target)}_{datetime.now().timestamp()}",
                    event_type=EventType.AFTER_UPDATE,
                    timestamp=datetime.now(),
                    source=model_class.__name__,
                    target=target
                )
                listener(context)
        
        elif event_type == EventType.BEFORE_DELETE:
            @event.listens_for(model_class, 'before_delete')
            def before_delete(mapper, connection, target):
                context = EventContext(
                    event_id=f"delete_{model_class.__name__}_{id(target)}_{datetime.now().timestamp()}",
                    event_type=EventType.BEFORE_DELETE,
                    timestamp=datetime.now(),
                    source=model_class.__name__,
                    target=target
                )
                listener(context)
        
        elif event_type == EventType.AFTER_DELETE:
            @event.listens_for(model_class, 'after_delete')
            def after_delete(mapper, connection, target):
                context = EventContext(
                    event_id=f"delete_{model_class.__name__}_{id(target)}_{datetime.now().timestamp()}",
                    event_type=EventType.AFTER_DELETE,
                    timestamp=datetime.now(),
                    source=model_class.__name__,
                    target=target
                )
                listener(context)
        
        # 记录监听器
        self._model_listeners[model_class].append(listener)
    
    def emit_business_event(self, event_type: EventType, source: str, 
                           target: Any = None, data: Dict[str, Any] = None, 
                           **kwargs) -> List[Any]:
        """发布业务事件"""
        context = EventContext(
            event_id=f"{event_type.value}_{datetime.now().timestamp()}",
            event_type=event_type,
            timestamp=datetime.now(),
            source=source,
            target=target,
            data=data or {},
            **kwargs
        )
        return self.event_bus.emit(context)


class EventDecorator:
    """事件装饰器"""
    
    def __init__(self, event_manager: ModelEventManager):
        self.event_manager = event_manager
    
    def emit_event(self, event_type: EventType, source: str = None):
        """发布事件装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 发布前置事件
                before_event_type = EventType(f"before_{event_type.value.split('_', 1)[-1]}")
                self.event_manager.emit_business_event(
                    before_event_type,
                    source or func.__name__,
                    data={'args': args, 'kwargs': kwargs}
                )
                
                try:
                    result = func(*args, **kwargs)
                    
                    # 发布后置事件
                    after_event_type = EventType(f"after_{event_type.value.split('_', 1)[-1]}")
                    self.event_manager.emit_business_event(
                        after_event_type,
                        source or func.__name__,
                        data={'result': result}
                    )
                    
                    return result
                
                except Exception as e:
                    # 发布错误事件
                    self.event_manager.emit_business_event(
                        EventType.ERROR_OCCURRED,
                        source or func.__name__,
                        data={'error': str(e), 'args': args, 'kwargs': kwargs}
                    )
                    raise
            
            return wrapper
        return decorator
    
    def track_performance(self, source: str = None):
        """性能跟踪装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = datetime.now()
                
                try:
                    result = func(*args, **kwargs)
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # 发布性能事件
                    self.event_manager.emit_business_event(
                        EventType.SYSTEM_STARTUP,  # 使用系统事件类型
                        source or func.__name__,
                        data={
                            'function': func.__name__,
                            'duration': duration,
                            'start_time': start_time.isoformat(),
                            'end_time': end_time.isoformat()
                        }
                    )
                    
                    return result
                
                except Exception as e:
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # 发布错误性能事件
                    self.event_manager.emit_business_event(
                        EventType.ERROR_OCCURRED,
                        source or func.__name__,
                        data={
                            'function': func.__name__,
                            'duration': duration,
                            'error': str(e)
                        }
                    )
                    raise
            
            return wrapper
        return decorator


# 全局事件总线和管理器
event_bus = EventBus()
event_manager = ModelEventManager(event_bus)
event_decorator = EventDecorator(event_manager)


# 便捷装饰器
def emit_event(event_type: EventType, source: str = None):
    """发布事件装饰器"""
    return event_decorator.emit_event(event_type, source)


def track_performance(source: str = None):
    """性能跟踪装饰器"""
    return event_decorator.track_performance(source)


# 便捷函数
def register_handler(event_type: EventType, handler: EventHandler) -> None:
    """注册事件处理器"""
    event_bus.register_handler(event_type, handler)


def register_function_handler(event_type: EventType, name: str, 
                             func: Callable[[EventContext], Any],
                             priority: EventPriority = EventPriority.NORMAL) -> None:
    """注册函数事件处理器"""
    handler = FunctionEventHandler(name, func, priority)
    event_bus.register_handler(event_type, handler)


def emit_business_event(event_type: EventType, source: str, 
                       target: Any = None, data: Dict[str, Any] = None, 
                       **kwargs) -> List[Any]:
    """发布业务事件"""
    return event_manager.emit_business_event(event_type, source, target, data, **kwargs)


# 导出所有类和函数
__all__ = [
    "EventType",
    "EventPriority",
    "EventStatus",
    "EventContext",
    "EventHandler",
    "AsyncEventHandler",
    "FunctionEventHandler",
    "ConditionalEventHandler",
    "EventBus",
    "ModelEventManager",
    "EventDecorator",
    "event_bus",
    "event_manager",
    "event_decorator",
    "emit_event",
    "track_performance",
    "register_handler",
    "register_function_handler",
    "emit_business_event"
]