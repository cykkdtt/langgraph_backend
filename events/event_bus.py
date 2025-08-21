"""事件总线模块

实现事件发布订阅机制，支持同步和异步事件处理。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union, Awaitable
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """事件优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """事件基类"""
    event_type: str
    data: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'priority': self.priority.value,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建事件"""
        timestamp = datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp']
        priority = EventPriority(data.get('priority', EventPriority.NORMAL.value))
        
        return cls(
            event_id=data['event_id'],
            event_type=data['event_type'],
            data=data['data'],
            timestamp=timestamp,
            source=data.get('source'),
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            priority=priority,
            metadata=data.get('metadata', {}),
            correlation_id=data.get('correlation_id'),
            causation_id=data.get('causation_id')
        )


class EventHandler(ABC):
    """事件处理器抽象基类"""
    
    def __init__(self, handler_id: str = None):
        self.handler_id = handler_id or str(uuid4())
        self.is_async = asyncio.iscoroutinefunction(self.handle)
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """处理事件"""
        pass
    
    def can_handle(self, event: Event) -> bool:
        """检查是否可以处理该事件"""
        return True
    
    def get_priority(self) -> int:
        """获取处理器优先级（数字越大优先级越高）"""
        return 0


class SyncEventHandler(EventHandler):
    """同步事件处理器"""
    
    @abstractmethod
    def handle_sync(self, event: Event) -> None:
        """同步处理事件"""
        pass
    
    async def handle(self, event: Event) -> None:
        """异步包装同步处理"""
        self.handle_sync(event)


class AsyncEventHandler(EventHandler):
    """异步事件处理器"""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """异步处理事件"""
        pass


@dataclass
class EventSubscription:
    """事件订阅信息"""
    handler: EventHandler
    event_types: Set[str]
    filter_func: Optional[Callable[[Event], bool]] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    is_active: bool = True


class EventBus:
    """事件总线"""
    
    def __init__(self, max_workers: int = 10):
        self._subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self._global_handlers: List[EventSubscription] = []
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._is_running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._max_workers = max_workers
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()
        self._event_history: List[Event] = []
        self._max_history_size = 1000
        
        # 统计信息
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'handlers_executed': 0
        }
    
    def subscribe(self, 
                 handler: EventHandler,
                 event_types: Union[str, List[str]] = None,
                 filter_func: Optional[Callable[[Event], bool]] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: Optional[float] = None) -> str:
        """订阅事件"""
        with self._lock:
            if isinstance(event_types, str):
                event_types = [event_types]
            
            subscription = EventSubscription(
                handler=handler,
                event_types=set(event_types) if event_types else set(),
                filter_func=filter_func,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout
            )
            
            if event_types:
                for event_type in event_types:
                    self._subscriptions[event_type].append(subscription)
            else:
                # 全局处理器
                self._global_handlers.append(subscription)
            
            logger.info(f"Handler {handler.handler_id} subscribed to events: {event_types or 'ALL'}")
            return handler.handler_id
    
    def unsubscribe(self, handler_id: str) -> bool:
        """取消订阅"""
        with self._lock:
            removed = False
            
            # 从特定事件类型订阅中移除
            for event_type, subscriptions in self._subscriptions.items():
                self._subscriptions[event_type] = [
                    sub for sub in subscriptions 
                    if sub.handler.handler_id != handler_id
                ]
                if len(subscriptions) != len(self._subscriptions[event_type]):
                    removed = True
            
            # 从全局处理器中移除
            original_count = len(self._global_handlers)
            self._global_handlers = [
                sub for sub in self._global_handlers 
                if sub.handler.handler_id != handler_id
            ]
            if len(self._global_handlers) != original_count:
                removed = True
            
            if removed:
                logger.info(f"Handler {handler_id} unsubscribed")
            
            return removed
    
    async def publish(self, event: Event) -> None:
        """发布事件"""
        self._stats['events_published'] += 1
        
        # 添加到历史记录
        self._add_to_history(event)
        
        # 添加到队列
        await self._event_queue.put(event)
        
        logger.debug(f"Event {event.event_id} of type {event.event_type} published")
    
    async def publish_sync(self, event: Event) -> None:
        """同步发布事件（立即处理）"""
        self._stats['events_published'] += 1
        self._add_to_history(event)
        
        await self._process_event(event)
    
    def publish_nowait(self, event: Event) -> None:
        """非阻塞发布事件"""
        try:
            self._event_queue.put_nowait(event)
            self._stats['events_published'] += 1
            self._add_to_history(event)
            logger.debug(f"Event {event.event_id} of type {event.event_type} published (nowait)")
        except asyncio.QueueFull:
            logger.warning(f"Event queue is full, dropping event {event.event_id}")
    
    async def start(self) -> None:
        """启动事件总线"""
        if self._is_running:
            return
        
        self._is_running = True
        
        # 启动工作线程
        for i in range(self._max_workers):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        logger.info(f"Event bus started with {self._max_workers} workers")
    
    async def stop(self) -> None:
        """停止事件总线"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # 等待所有工作任务完成
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks.clear()
        
        # 关闭线程池
        self._thread_pool.shutdown(wait=True)
        
        logger.info("Event bus stopped")
    
    async def _worker_loop(self, worker_name: str) -> None:
        """工作线程循环"""
        logger.info(f"Event worker {worker_name} started")
        
        while self._is_running:
            try:
                # 等待事件，设置超时以便定期检查运行状态
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._process_event(event)
                self._event_queue.task_done()
            
            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            
            except Exception as e:
                logger.error(f"Error in worker {worker_name}: {e}")
                await asyncio.sleep(0.1)  # 短暂延迟避免快速循环
        
        logger.info(f"Event worker {worker_name} stopped")
    
    async def _process_event(self, event: Event) -> None:
        """处理事件"""
        try:
            handlers = self._get_handlers_for_event(event)
            
            if not handlers:
                logger.debug(f"No handlers found for event {event.event_id}")
                return
            
            # 按优先级排序处理器
            handlers.sort(key=lambda h: h.handler.get_priority(), reverse=True)
            
            # 并发执行所有处理器
            tasks = []
            for subscription in handlers:
                if subscription.is_active:
                    task = asyncio.create_task(
                        self._execute_handler(subscription, event)
                    )
                    tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self._stats['events_processed'] += 1
            logger.debug(f"Event {event.event_id} processed by {len(tasks)} handlers")
        
        except Exception as e:
            self._stats['events_failed'] += 1
            logger.error(f"Error processing event {event.event_id}: {e}")
    
    def _get_handlers_for_event(self, event: Event) -> List[EventSubscription]:
        """获取事件的处理器"""
        handlers = []
        
        # 获取特定事件类型的处理器
        if event.event_type in self._subscriptions:
            handlers.extend(self._subscriptions[event.event_type])
        
        # 添加全局处理器
        handlers.extend(self._global_handlers)
        
        # 应用过滤器
        filtered_handlers = []
        for subscription in handlers:
            try:
                if subscription.filter_func is None or subscription.filter_func(event):
                    if subscription.handler.can_handle(event):
                        filtered_handlers.append(subscription)
            except Exception as e:
                logger.warning(f"Error in event filter: {e}")
        
        return filtered_handlers
    
    async def _execute_handler(self, subscription: EventSubscription, event: Event) -> None:
        """执行事件处理器"""
        handler = subscription.handler
        retries = 0
        
        while retries <= subscription.max_retries:
            try:
                # 设置超时
                if subscription.timeout:
                    await asyncio.wait_for(
                        handler.handle(event),
                        timeout=subscription.timeout
                    )
                else:
                    await handler.handle(event)
                
                self._stats['handlers_executed'] += 1
                logger.debug(f"Handler {handler.handler_id} processed event {event.event_id}")
                return
            
            except asyncio.TimeoutError:
                logger.warning(f"Handler {handler.handler_id} timed out processing event {event.event_id}")
                break
            
            except Exception as e:
                retries += 1
                logger.warning(
                    f"Handler {handler.handler_id} failed to process event {event.event_id} "
                    f"(attempt {retries}/{subscription.max_retries + 1}): {e}"
                )
                
                if retries <= subscription.max_retries:
                    await asyncio.sleep(subscription.retry_delay * retries)
                else:
                    logger.error(
                        f"Handler {handler.handler_id} failed permanently for event {event.event_id}"
                    )
    
    def _add_to_history(self, event: Event) -> None:
        """添加事件到历史记录"""
        self._event_history.append(event)
        
        # 限制历史记录大小
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            'queue_size': self._event_queue.qsize(),
            'active_subscriptions': sum(len(subs) for subs in self._subscriptions.values()),
            'global_handlers': len(self._global_handlers),
            'is_running': self._is_running,
            'worker_count': len(self._worker_tasks)
        }
    
    def get_event_history(self, limit: int = 100) -> List[Event]:
        """获取事件历史"""
        return self._event_history[-limit:]
    
    def clear_history(self) -> None:
        """清空事件历史"""
        self._event_history.clear()
    
    async def wait_for_queue_empty(self, timeout: Optional[float] = None) -> bool:
        """等待队列为空"""
        try:
            await asyncio.wait_for(self._event_queue.join(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


# 全局事件总线实例
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取全局事件总线"""
    global _global_event_bus
    
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    
    return _global_event_bus


async def publish_event(event_type: str, data: Dict[str, Any], 
                       source: str = None, user_id: str = None,
                       session_id: str = None, priority: EventPriority = EventPriority.NORMAL,
                       metadata: Dict[str, Any] = None) -> str:
    """发布事件的便捷函数"""
    event = Event(
        event_type=event_type,
        data=data,
        source=source,
        user_id=user_id,
        session_id=session_id,
        priority=priority,
        metadata=metadata or {}
    )
    
    event_bus = get_event_bus()
    await event_bus.publish(event)
    
    return event.event_id


def subscribe_to_events(handler: EventHandler, 
                       event_types: Union[str, List[str]] = None,
                       filter_func: Optional[Callable[[Event], bool]] = None) -> str:
    """订阅事件的便捷函数"""
    event_bus = get_event_bus()
    return event_bus.subscribe(handler, event_types, filter_func)