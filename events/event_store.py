"""事件存储模块

提供事件持久化存储、查询和重放功能。
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from uuid import uuid4
from sqlalchemy import Column, String, DateTime, Text, Integer, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import JSONB

from ..database.base import BaseModel
from .event_bus import Event, EventPriority

logger = logging.getLogger(__name__)

Base = declarative_base()


@dataclass
class EventQuery:
    """事件查询条件"""
    event_types: Optional[List[str]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    limit: int = 100
    offset: int = 0
    order_by: str = "timestamp"
    order_desc: bool = True


@dataclass
class EventStats:
    """事件统计信息"""
    total_events: int
    event_types: Dict[str, int]
    events_by_hour: Dict[str, int]
    events_by_day: Dict[str, int]
    top_sources: Dict[str, int]
    top_users: Dict[str, int]


class StoredEvent(BaseModel):
    """存储的事件模型"""
    __tablename__ = "events"
    
    event_id = Column(String(36), primary_key=True, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    data = Column(JSONB, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    source = Column(String(100), index=True)
    user_id = Column(String(36), index=True)
    session_id = Column(String(36), index=True)
    priority = Column(Integer, default=EventPriority.NORMAL.value)
    metadata = Column(JSONB, default={})
    correlation_id = Column(String(36), index=True)
    causation_id = Column(String(36), index=True)
    
    # 复合索引
    __table_args__ = (
        Index('idx_events_type_timestamp', 'event_type', 'timestamp'),
        Index('idx_events_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_events_session_timestamp', 'session_id', 'timestamp'),
        Index('idx_events_correlation', 'correlation_id'),
        Index('idx_events_causation', 'causation_id'),
    )
    
    def to_event(self) -> Event:
        """转换为事件对象"""
        return Event(
            event_id=self.event_id,
            event_type=self.event_type,
            data=self.data,
            timestamp=self.timestamp,
            source=self.source,
            user_id=self.user_id,
            session_id=self.session_id,
            priority=EventPriority(self.priority),
            metadata=self.metadata or {},
            correlation_id=self.correlation_id,
            causation_id=self.causation_id
        )
    
    @classmethod
    def from_event(cls, event: Event) -> 'StoredEvent':
        """从事件对象创建"""
        return cls(
            event_id=event.event_id,
            event_type=event.event_type,
            data=event.data,
            timestamp=event.timestamp,
            source=event.source,
            user_id=event.user_id,
            session_id=event.session_id,
            priority=event.priority.value,
            metadata=event.metadata,
            correlation_id=event.correlation_id,
            causation_id=event.causation_id
        )


class EventStore(ABC):
    """事件存储抽象基类"""
    
    @abstractmethod
    async def store_event(self, event: Event) -> None:
        """存储事件"""
        pass
    
    @abstractmethod
    async def get_event(self, event_id: str) -> Optional[Event]:
        """获取事件"""
        pass
    
    @abstractmethod
    async def query_events(self, query: EventQuery) -> List[Event]:
        """查询事件"""
        pass
    
    @abstractmethod
    async def count_events(self, query: EventQuery) -> int:
        """统计事件数量"""
        pass
    
    @abstractmethod
    async def get_event_stream(self, query: EventQuery) -> AsyncIterator[Event]:
        """获取事件流"""
        pass
    
    @abstractmethod
    async def delete_events(self, query: EventQuery) -> int:
        """删除事件"""
        pass
    
    @abstractmethod
    async def get_stats(self, start_time: datetime = None, 
                       end_time: datetime = None) -> EventStats:
        """获取统计信息"""
        pass


class DatabaseEventStore(EventStore):
    """数据库事件存储实现"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    async def store_event(self, event: Event) -> None:
        """存储事件"""
        try:
            stored_event = StoredEvent.from_event(event)
            self.db_session.add(stored_event)
            self.db_session.commit()
            
            logger.debug(f"Event {event.event_id} stored successfully")
        
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to store event {event.event_id}: {e}")
            raise
    
    async def get_event(self, event_id: str) -> Optional[Event]:
        """获取事件"""
        try:
            stored_event = self.db_session.query(StoredEvent).filter(
                StoredEvent.event_id == event_id
            ).first()
            
            return stored_event.to_event() if stored_event else None
        
        except Exception as e:
            logger.error(f"Failed to get event {event_id}: {e}")
            return None
    
    async def query_events(self, query: EventQuery) -> List[Event]:
        """查询事件"""
        try:
            db_query = self.db_session.query(StoredEvent)
            
            # 应用过滤条件
            db_query = self._apply_filters(db_query, query)
            
            # 排序
            if query.order_by == "timestamp":
                if query.order_desc:
                    db_query = db_query.order_by(StoredEvent.timestamp.desc())
                else:
                    db_query = db_query.order_by(StoredEvent.timestamp.asc())
            
            # 分页
            db_query = db_query.offset(query.offset).limit(query.limit)
            
            stored_events = db_query.all()
            return [event.to_event() for event in stored_events]
        
        except Exception as e:
            logger.error(f"Failed to query events: {e}")
            return []
    
    async def count_events(self, query: EventQuery) -> int:
        """统计事件数量"""
        try:
            db_query = self.db_session.query(StoredEvent)
            db_query = self._apply_filters(db_query, query)
            
            return db_query.count()
        
        except Exception as e:
            logger.error(f"Failed to count events: {e}")
            return 0
    
    async def get_event_stream(self, query: EventQuery) -> AsyncIterator[Event]:
        """获取事件流"""
        try:
            db_query = self.db_session.query(StoredEvent)
            db_query = self._apply_filters(db_query, query)
            
            # 排序
            if query.order_by == "timestamp":
                if query.order_desc:
                    db_query = db_query.order_by(StoredEvent.timestamp.desc())
                else:
                    db_query = db_query.order_by(StoredEvent.timestamp.asc())
            
            # 分批获取
            batch_size = 100
            offset = query.offset
            
            while True:
                batch = db_query.offset(offset).limit(batch_size).all()
                
                if not batch:
                    break
                
                for stored_event in batch:
                    yield stored_event.to_event()
                
                offset += batch_size
                
                # 如果有限制且已达到限制
                if query.limit > 0 and offset >= query.offset + query.limit:
                    break
        
        except Exception as e:
            logger.error(f"Failed to get event stream: {e}")
    
    async def delete_events(self, query: EventQuery) -> int:
        """删除事件"""
        try:
            db_query = self.db_session.query(StoredEvent)
            db_query = self._apply_filters(db_query, query)
            
            count = db_query.count()
            db_query.delete(synchronize_session=False)
            self.db_session.commit()
            
            logger.info(f"Deleted {count} events")
            return count
        
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to delete events: {e}")
            return 0
    
    async def get_stats(self, start_time: datetime = None, 
                       end_time: datetime = None) -> EventStats:
        """获取统计信息"""
        try:
            # 基础查询
            base_query = self.db_session.query(StoredEvent)
            
            if start_time:
                base_query = base_query.filter(StoredEvent.timestamp >= start_time)
            if end_time:
                base_query = base_query.filter(StoredEvent.timestamp <= end_time)
            
            # 总事件数
            total_events = base_query.count()
            
            # 按事件类型统计
            event_types = {}
            type_stats = base_query.with_entities(
                StoredEvent.event_type,
                self.db_session.query(StoredEvent).filter(
                    StoredEvent.event_type == StoredEvent.event_type
                ).count().label('count')
            ).group_by(StoredEvent.event_type).all()
            
            for event_type, count in type_stats:
                event_types[event_type] = count
            
            # 按小时统计（简化实现）
            events_by_hour = {}
            events_by_day = {}
            
            # 按来源统计
            top_sources = {}
            source_stats = base_query.with_entities(
                StoredEvent.source,
                self.db_session.query(StoredEvent).filter(
                    StoredEvent.source == StoredEvent.source
                ).count().label('count')
            ).filter(StoredEvent.source.isnot(None)).group_by(
                StoredEvent.source
            ).order_by(self.db_session.text('count DESC')).limit(10).all()
            
            for source, count in source_stats:
                top_sources[source] = count
            
            # 按用户统计
            top_users = {}
            user_stats = base_query.with_entities(
                StoredEvent.user_id,
                self.db_session.query(StoredEvent).filter(
                    StoredEvent.user_id == StoredEvent.user_id
                ).count().label('count')
            ).filter(StoredEvent.user_id.isnot(None)).group_by(
                StoredEvent.user_id
            ).order_by(self.db_session.text('count DESC')).limit(10).all()
            
            for user_id, count in user_stats:
                top_users[user_id] = count
            
            return EventStats(
                total_events=total_events,
                event_types=event_types,
                events_by_hour=events_by_hour,
                events_by_day=events_by_day,
                top_sources=top_sources,
                top_users=top_users
            )
        
        except Exception as e:
            logger.error(f"Failed to get event stats: {e}")
            return EventStats(
                total_events=0,
                event_types={},
                events_by_hour={},
                events_by_day={},
                top_sources={},
                top_users={}
            )
    
    def _apply_filters(self, query, event_query: EventQuery):
        """应用查询过滤条件"""
        if event_query.event_types:
            query = query.filter(StoredEvent.event_type.in_(event_query.event_types))
        
        if event_query.user_id:
            query = query.filter(StoredEvent.user_id == event_query.user_id)
        
        if event_query.session_id:
            query = query.filter(StoredEvent.session_id == event_query.session_id)
        
        if event_query.source:
            query = query.filter(StoredEvent.source == event_query.source)
        
        if event_query.start_time:
            query = query.filter(StoredEvent.timestamp >= event_query.start_time)
        
        if event_query.end_time:
            query = query.filter(StoredEvent.timestamp <= event_query.end_time)
        
        if event_query.correlation_id:
            query = query.filter(StoredEvent.correlation_id == event_query.correlation_id)
        
        if event_query.causation_id:
            query = query.filter(StoredEvent.causation_id == event_query.causation_id)
        
        return query


class MemoryEventStore(EventStore):
    """内存事件存储实现（用于测试）"""
    
    def __init__(self):
        self._events: Dict[str, Event] = {}
        self._events_by_type: Dict[str, List[str]] = {}
        self._events_by_user: Dict[str, List[str]] = {}
        self._events_by_session: Dict[str, List[str]] = {}
    
    async def store_event(self, event: Event) -> None:
        """存储事件"""
        self._events[event.event_id] = event
        
        # 更新索引
        if event.event_type not in self._events_by_type:
            self._events_by_type[event.event_type] = []
        self._events_by_type[event.event_type].append(event.event_id)
        
        if event.user_id:
            if event.user_id not in self._events_by_user:
                self._events_by_user[event.user_id] = []
            self._events_by_user[event.user_id].append(event.event_id)
        
        if event.session_id:
            if event.session_id not in self._events_by_session:
                self._events_by_session[event.session_id] = []
            self._events_by_session[event.session_id].append(event.event_id)
    
    async def get_event(self, event_id: str) -> Optional[Event]:
        """获取事件"""
        return self._events.get(event_id)
    
    async def query_events(self, query: EventQuery) -> List[Event]:
        """查询事件"""
        events = list(self._events.values())
        
        # 应用过滤条件
        filtered_events = []
        for event in events:
            if self._matches_query(event, query):
                filtered_events.append(event)
        
        # 排序
        if query.order_by == "timestamp":
            filtered_events.sort(
                key=lambda e: e.timestamp,
                reverse=query.order_desc
            )
        
        # 分页
        start = query.offset
        end = start + query.limit if query.limit > 0 else len(filtered_events)
        
        return filtered_events[start:end]
    
    async def count_events(self, query: EventQuery) -> int:
        """统计事件数量"""
        count = 0
        for event in self._events.values():
            if self._matches_query(event, query):
                count += 1
        return count
    
    async def get_event_stream(self, query: EventQuery) -> AsyncIterator[Event]:
        """获取事件流"""
        events = await self.query_events(query)
        for event in events:
            yield event
    
    async def delete_events(self, query: EventQuery) -> int:
        """删除事件"""
        to_delete = []
        for event_id, event in self._events.items():
            if self._matches_query(event, query):
                to_delete.append(event_id)
        
        for event_id in to_delete:
            del self._events[event_id]
            # 清理索引（简化实现）
        
        return len(to_delete)
    
    async def get_stats(self, start_time: datetime = None, 
                       end_time: datetime = None) -> EventStats:
        """获取统计信息"""
        events = list(self._events.values())
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # 统计事件类型
        event_types = {}
        for event in events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        # 统计来源
        top_sources = {}
        for event in events:
            if event.source:
                top_sources[event.source] = top_sources.get(event.source, 0) + 1
        
        # 统计用户
        top_users = {}
        for event in events:
            if event.user_id:
                top_users[event.user_id] = top_users.get(event.user_id, 0) + 1
        
        return EventStats(
            total_events=len(events),
            event_types=event_types,
            events_by_hour={},
            events_by_day={},
            top_sources=dict(sorted(top_sources.items(), key=lambda x: x[1], reverse=True)[:10]),
            top_users=dict(sorted(top_users.items(), key=lambda x: x[1], reverse=True)[:10])
        )
    
    def _matches_query(self, event: Event, query: EventQuery) -> bool:
        """检查事件是否匹配查询条件"""
        if query.event_types and event.event_type not in query.event_types:
            return False
        
        if query.user_id and event.user_id != query.user_id:
            return False
        
        if query.session_id and event.session_id != query.session_id:
            return False
        
        if query.source and event.source != query.source:
            return False
        
        if query.start_time and event.timestamp < query.start_time:
            return False
        
        if query.end_time and event.timestamp > query.end_time:
            return False
        
        if query.correlation_id and event.correlation_id != query.correlation_id:
            return False
        
        if query.causation_id and event.causation_id != query.causation_id:
            return False
        
        return True


# 全局事件存储实例
_global_event_store: Optional[EventStore] = None


def get_event_store() -> EventStore:
    """获取全局事件存储"""
    global _global_event_store
    
    if _global_event_store is None:
        # 默认使用内存存储
        _global_event_store = MemoryEventStore()
    
    return _global_event_store


def set_event_store(event_store: EventStore) -> None:
    """设置全局事件存储"""
    global _global_event_store
    _global_event_store = event_store