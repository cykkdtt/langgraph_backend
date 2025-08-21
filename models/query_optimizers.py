"""数据库查询优化模块

本模块提供数据库查询优化功能，包括：
- 高效的查询构建器
- 缓存策略实现
- 批量操作优化
- 分页查询优化
- 索引使用建议
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Type
from datetime import datetime, timedelta
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.orm import Query, Session, joinedload, selectinload
from sqlalchemy.sql import Select
import redis
import json
import hashlib
from functools import wraps
import logging

# 导入数据库模型
from .database_models import (
    User, Session as DBSession, Message, ToolCall, AgentState,
    SystemLog, Workflow, WorkflowExecution, Memory
)

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """查询优化器基类"""
    
    def __init__(self, db_session: Session, redis_client: Optional[redis.Redis] = None):
        self.db_session = db_session
        self.redis_client = redis_client
        self.cache_ttl = 300  # 默认缓存5分钟
    
    def _generate_cache_key(self, prefix: str, **kwargs) -> str:
        """生成缓存键"""
        key_data = json.dumps(kwargs, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if not self.redis_client:
            return None
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"缓存读取失败: {e}")
        return None
    
    def _set_to_cache(self, cache_key: str, data: Any, ttl: Optional[int] = None) -> None:
        """设置缓存数据"""
        if not self.redis_client:
            return
        try:
            ttl = ttl or self.cache_ttl
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}")
    
    def _invalidate_cache_pattern(self, pattern: str) -> None:
        """清除匹配模式的缓存"""
        if not self.redis_client:
            return
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"缓存清除失败: {e}")


class UserQueryOptimizer(QueryOptimizer):
    """用户查询优化器"""
    
    def get_user_by_id(self, user_id: str, use_cache: bool = True) -> Optional[User]:
        """根据ID获取用户（带缓存）"""
        cache_key = self._generate_cache_key("user", user_id=user_id)
        
        if use_cache:
            cached_user = self._get_from_cache(cache_key)
            if cached_user:
                return User(**cached_user)
        
        user = self.db_session.query(User).filter(User.id == user_id).first()
        
        if user and use_cache:
            user_dict = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "display_name": user.display_name,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat() if user.created_at else None
            }
            self._set_to_cache(cache_key, user_dict)
        
        return user
    
    def get_active_users_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        search_term: Optional[str] = None
    ) -> Tuple[List[User], int]:
        """分页获取活跃用户"""
        query = self.db_session.query(User).filter(User.is_active == True)
        
        if search_term:
            search_filter = or_(
                User.username.ilike(f"%{search_term}%"),
                User.email.ilike(f"%{search_term}%"),
                User.display_name.ilike(f"%{search_term}%")
            )
            query = query.filter(search_filter)
        
        # 获取总数
        total = query.count()
        
        # 分页查询
        users = query.order_by(desc(User.created_at))\
                    .offset((page - 1) * page_size)\
                    .limit(page_size)\
                    .all()
        
        return users, total
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        cache_key = self._generate_cache_key("user_stats", user_id=user_id)
        
        cached_stats = self._get_from_cache(cache_key)
        if cached_stats:
            return cached_stats
        
        # 使用子查询优化性能
        session_count = self.db_session.query(func.count(DBSession.id))\
                                      .filter(DBSession.user_id == user_id)\
                                      .scalar()
        
        message_count = self.db_session.query(func.count(Message.id))\
                                      .filter(Message.user_id == user_id)\
                                      .scalar()
        
        tool_call_count = self.db_session.query(func.count(ToolCall.id))\
                                        .join(Message)\
                                        .filter(Message.user_id == user_id)\
                                        .scalar()
        
        stats = {
            "session_count": session_count or 0,
            "message_count": message_count or 0,
            "tool_call_count": tool_call_count or 0,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        self._set_to_cache(cache_key, stats, ttl=600)  # 缓存10分钟
        return stats


class SessionQueryOptimizer(QueryOptimizer):
    """会话查询优化器"""
    
    def get_user_sessions_with_messages(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        include_inactive: bool = False
    ) -> Tuple[List[DBSession], int]:
        """获取用户会话及消息（优化版）"""
        query = self.db_session.query(DBSession)\
                              .filter(DBSession.user_id == user_id)
        
        if not include_inactive:
            query = query.filter(DBSession.is_active == True)
        
        # 预加载消息以减少N+1查询问题
        query = query.options(
            selectinload(DBSession.messages).selectinload(Message.tool_calls)
        )
        
        total = query.count()
        
        sessions = query.order_by(desc(DBSession.updated_at))\
                       .offset((page - 1) * page_size)\
                       .limit(page_size)\
                       .all()
        
        return sessions, total
    
    def get_session_with_recent_messages(
        self,
        session_id: str,
        message_limit: int = 50
    ) -> Optional[DBSession]:
        """获取会话及最近消息"""
        cache_key = self._generate_cache_key(
            "session_messages",
            session_id=session_id,
            limit=message_limit
        )
        
        # 尝试从缓存获取
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            # 这里需要重新构建对象，实际实现中可能需要更复杂的序列化
            pass
        
        session = self.db_session.query(DBSession)\
                                .filter(DBSession.id == session_id)\
                                .first()
        
        if session:
            # 获取最近的消息
            recent_messages = self.db_session.query(Message)\
                                            .filter(Message.session_id == session_id)\
                                            .order_by(desc(Message.created_at))\
                                            .limit(message_limit)\
                                            .all()
            
            # 手动设置关系以避免额外查询
            session.messages = recent_messages
        
        return session
    
    def update_session_activity(self, session_id: str) -> None:
        """更新会话活动时间（批量优化）"""
        # 使用批量更新而不是单个对象更新
        self.db_session.query(DBSession)\
                      .filter(DBSession.id == session_id)\
                      .update({
                          DBSession.last_activity_at: datetime.utcnow(),
                          DBSession.updated_at: datetime.utcnow()
                      })
        
        # 清除相关缓存
        self._invalidate_cache_pattern(f"session_messages:{session_id}:*")


class MessageQueryOptimizer(QueryOptimizer):
    """消息查询优化器"""
    
    def get_messages_with_context(
        self,
        session_id: str,
        before_message_id: Optional[str] = None,
        limit: int = 20,
        include_tool_calls: bool = True
    ) -> List[Message]:
        """获取带上下文的消息列表"""
        query = self.db_session.query(Message)\
                              .filter(Message.session_id == session_id)
        
        if before_message_id:
            # 获取指定消息的创建时间
            before_message = self.db_session.query(Message.created_at)\
                                           .filter(Message.id == before_message_id)\
                                           .first()
            if before_message:
                query = query.filter(Message.created_at < before_message.created_at)
        
        if include_tool_calls:
            query = query.options(selectinload(Message.tool_calls))
        
        messages = query.order_by(desc(Message.created_at))\
                       .limit(limit)\
                       .all()
        
        return list(reversed(messages))  # 返回时间正序
    
    def search_messages(
        self,
        user_id: str,
        search_term: str,
        session_id: Optional[str] = None,
        message_type: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Message], int]:
        """搜索消息（全文搜索优化）"""
        query = self.db_session.query(Message)\
                              .filter(Message.user_id == user_id)
        
        # 全文搜索（使用数据库特定的全文搜索功能）
        if search_term:
            # PostgreSQL示例，其他数据库需要调整
            query = query.filter(
                func.to_tsvector('english', Message.content)
                    .match(func.plainto_tsquery('english', search_term))
            )
        
        if session_id:
            query = query.filter(Message.session_id == session_id)
        
        if message_type:
            query = query.filter(Message.message_type == message_type)
        
        if date_from:
            query = query.filter(Message.created_at >= date_from)
        
        if date_to:
            query = query.filter(Message.created_at <= date_to)
        
        total = query.count()
        
        messages = query.order_by(desc(Message.created_at))\
                       .offset((page - 1) * page_size)\
                       .limit(page_size)\
                       .all()
        
        return messages, total
    
    def get_message_analytics(
        self,
        session_id: str,
        time_range: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """获取消息分析数据"""
        cache_key = self._generate_cache_key(
            "message_analytics",
            session_id=session_id,
            days=time_range.days
        )
        
        cached_analytics = self._get_from_cache(cache_key)
        if cached_analytics:
            return cached_analytics
        
        start_date = datetime.utcnow() - time_range
        
        # 使用聚合查询获取统计信息
        analytics = self.db_session.query(
            func.count(Message.id).label('total_messages'),
            func.avg(Message.content_length).label('avg_content_length'),
            func.avg(Message.processing_time).label('avg_processing_time'),
            func.avg(Message.quality_score).label('avg_quality_score')
        ).filter(
            Message.session_id == session_id,
            Message.created_at >= start_date
        ).first()
        
        # 按角色统计
        role_stats = self.db_session.query(
            Message.role,
            func.count(Message.id).label('count')
        ).filter(
            Message.session_id == session_id,
            Message.created_at >= start_date
        ).group_by(Message.role).all()
        
        result = {
            "total_messages": analytics.total_messages or 0,
            "avg_content_length": float(analytics.avg_content_length or 0),
            "avg_processing_time": float(analytics.avg_processing_time or 0),
            "avg_quality_score": float(analytics.avg_quality_score or 0),
            "role_distribution": {role: count for role, count in role_stats},
            "time_range_days": time_range.days,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        self._set_to_cache(cache_key, result, ttl=1800)  # 缓存30分钟
        return result


class WorkflowQueryOptimizer(QueryOptimizer):
    """工作流查询优化器"""
    
    def get_popular_workflows(
        self,
        category: Optional[str] = None,
        limit: int = 10,
        time_range: timedelta = timedelta(days=30)
    ) -> List[Workflow]:
        """获取热门工作流"""
        cache_key = self._generate_cache_key(
            "popular_workflows",
            category=category,
            limit=limit,
            days=time_range.days
        )
        
        cached_workflows = self._get_from_cache(cache_key)
        if cached_workflows:
            # 需要重新查询以获取完整对象
            workflow_ids = [w['id'] for w in cached_workflows]
            return self.db_session.query(Workflow)\
                                 .filter(Workflow.id.in_(workflow_ids))\
                                 .all()
        
        query = self.db_session.query(Workflow)\
                              .filter(Workflow.is_active == True)
        
        if category:
            query = query.filter(Workflow.category == category)
        
        # 按使用量和评分排序
        workflows = query.order_by(
            desc(Workflow.usage_count),
            desc(Workflow.rating),
            desc(Workflow.execution_count)
        ).limit(limit).all()
        
        # 缓存结果
        workflow_data = [{
            "id": w.id,
            "name": w.name,
            "category": w.category,
            "usage_count": w.usage_count,
            "rating": w.rating
        } for w in workflows]
        
        self._set_to_cache(cache_key, workflow_data, ttl=3600)  # 缓存1小时
        
        return workflows
    
    def get_workflow_execution_history(
        self,
        workflow_id: str,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[WorkflowExecution], int]:
        """获取工作流执行历史"""
        query = self.db_session.query(WorkflowExecution)\
                              .filter(WorkflowExecution.workflow_id == workflow_id)
        
        if user_id:
            query = query.filter(WorkflowExecution.user_id == user_id)
        
        if status:
            query = query.filter(WorkflowExecution.status == status)
        
        total = query.count()
        
        executions = query.order_by(desc(WorkflowExecution.created_at))\
                         .offset((page - 1) * page_size)\
                         .limit(page_size)\
                         .all()
        
        return executions, total


class Mem