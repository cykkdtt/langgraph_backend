"""会话仓储模块

提供会话相关的数据访问接口和业务逻辑。
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session, selectinload

from ..models.database import Session as SessionModel
from ..models.api import SessionCreate, SessionUpdate
from ..database.repository import CRUDRepository, EntityNotFoundError
from ..database.query_builder import QueryFilter
from ..database.session import SessionManager

logger = logging.getLogger(__name__)


class SessionRepository(CRUDRepository[SessionModel]):
    """会话仓储类"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        super().__init__(SessionModel, session_manager)
        # 设置默认加载选项
        self.set_default_load_options(
            selectinload(SessionModel.user),
            selectinload(SessionModel.threads)
        )
    
    def create_session(
        self, 
        session_create: SessionCreate, 
        user_id: int,
        session: Optional[Session] = None
    ) -> SessionModel:
        """创建会话"""
        session_data = session_create.model_dump()
        session_data["user_id"] = user_id
        session_data["is_active"] = True
        
        return self.create(session_data, session)
    
    def get_user_sessions(
        self, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 100,
        include_inactive: bool = False,
        session: Optional[Session] = None
    ) -> List[SessionModel]:
        """获取用户的会话列表"""
        filters = QueryFilter().eq("user_id", user_id)
        
        if not include_inactive:
            filters = filters.and_(QueryFilter().eq("is_active", True))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("updated_at", "desc")],
            session=session
        )
    
    def get_active_sessions(
        self, 
        user_id: Optional[int] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[SessionModel]:
        """获取活跃会话"""
        filters = QueryFilter().eq("is_active", True)
        
        if user_id:
            filters = filters.and_(QueryFilter().eq("user_id", user_id))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("updated_at", "desc")],
            session=session
        )
    
    def update_session_activity(
        self, 
        session_id: int,
        session: Optional[Session] = None
    ) -> Optional[SessionModel]:
        """更新会话活动时间"""
        try:
            session_obj = self.get(session_id, session)
            if not session_obj:
                raise EntityNotFoundError(f"Session with ID {session_id} not found")
            
            update_data = {
                "last_activity_at": datetime.utcnow()
            }
            
            return self.update(session_obj, update_data, session)
            
        except Exception as e:
            logger.error(f"Error updating session activity for session {session_id}: {e}")
            return None
    
    def deactivate_session(
        self, 
        session_id: int,
        session: Optional[Session] = None
    ) -> bool:
        """停用会话"""
        try:
            session_obj = self.get(session_id, session)
            if not session_obj:
                raise EntityNotFoundError(f"Session with ID {session_id} not found")
            
            update_data = {
                "is_active": False,
                "ended_at": datetime.utcnow()
            }
            
            self.update(session_obj, update_data, session)
            
            logger.info(f"Session {session_id} deactivated")
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating session {session_id}: {e}")
            return False
    
    def activate_session(
        self, 
        session_id: int,
        session: Optional[Session] = None
    ) -> bool:
        """激活会话"""
        try:
            session_obj = self.get(session_id, session)
            if not session_obj:
                raise EntityNotFoundError(f"Session with ID {session_id} not found")
            
            update_data = {
                "is_active": True,
                "ended_at": None,
                "last_activity_at": datetime.utcnow()
            }
            
            self.update(session_obj, update_data, session)
            
            logger.info(f"Session {session_id} activated")
            return True
            
        except Exception as e:
            logger.error(f"Error activating session {session_id}: {e}")
            return False
    
    def get_session_by_title(
        self, 
        user_id: int, 
        title: str,
        session: Optional[Session] = None
    ) -> Optional[SessionModel]:
        """根据标题获取会话"""
        filters = QueryFilter().and_(
            QueryFilter().eq("user_id", user_id),
            QueryFilter().eq("title", title)
        )
        
        sessions = self.get_multi(
            limit=1,
            filters=filters,
            session=session
        )
        
        return sessions[0] if sessions else None
    
    def search_sessions(
        self, 
        user_id: int,
        query: str, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[SessionModel]:
        """搜索用户会话"""
        filters = QueryFilter().and_(
            QueryFilter().eq("user_id", user_id),
            QueryFilter().or_(
                QueryFilter().ilike("title", f"%{query}%"),
                QueryFilter().ilike("description", f"%{query}%")
            )
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("updated_at", "desc")],
            session=session
        )
    
    def get_recent_sessions(
        self, 
        user_id: int,
        days: int = 7, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[SessionModel]:
        """获取最近的会话"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        filters = QueryFilter().and_(
            QueryFilter().eq("user_id", user_id),
            QueryFilter().gte("last_activity_at", cutoff_date)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("last_activity_at", "desc")],
            session=session
        )
    
    def get_sessions_by_date_range(
        self, 
        user_id: int,
        start_date: datetime,
        end_date: datetime,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[SessionModel]:
        """根据日期范围获取会话"""
        filters = QueryFilter().and_(
            QueryFilter().eq("user_id", user_id),
            QueryFilter().gte("created_at", start_date),
            QueryFilter().lte("created_at", end_date)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_session_statistics(
        self, 
        user_id: Optional[int] = None,
        session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """获取会话统计信息"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            base_filters = QueryFilter()
            if user_id:
                base_filters = base_filters.eq("user_id", user_id)
            
            # 总会话数
            total_sessions = self.count(filters=base_filters, session=db_session)
            
            # 活跃会话数
            active_filters = base_filters.and_(QueryFilter().eq("is_active", True))
            active_sessions = self.count(filters=active_filters, session=db_session)
            
            # 今日创建的会话数
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_filters = base_filters.and_(QueryFilter().gte("created_at", today_start))
            today_sessions = self.count(filters=today_filters, session=db_session)
            
            # 最近7天活跃的会话数
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_filters = base_filters.and_(QueryFilter().gte("last_activity_at", week_ago))
            recent_active_sessions = self.count(filters=recent_filters, session=db_session)
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "today_sessions": today_sessions,
                "recent_active_sessions": recent_active_sessions
            }
            
        except Exception as e:
            logger.error(f"Error getting session statistics: {e}")
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "today_sessions": 0,
                "recent_active_sessions": 0
            }
        finally:
            if not session:
                db_session.close()
    
    def cleanup_inactive_sessions(
        self, 
        days: int = 30,
        session: Optional[Session] = None
    ) -> int:
        """清理长期不活跃的会话"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # 查找长期不活跃的会话
            filters = QueryFilter().and_(
                QueryFilter().eq("is_active", True),
                QueryFilter().or_(
                    QueryFilter().is_null("last_activity_at"),
                    QueryFilter().lt("last_activity_at", cutoff_date)
                )
            )
            
            inactive_sessions = self.get_multi(
                filters=filters,
                limit=1000,  # 批量处理
                session=session
            )
            
            # 停用这些会话
            deactivated_count = 0
            for session_obj in inactive_sessions:
                if self.deactivate_session(session_obj.id, session):
                    deactivated_count += 1
            
            logger.info(f"Cleaned up {deactivated_count} inactive sessions")
            return deactivated_count
            
        except Exception as e:
            logger.error(f"Error cleaning up inactive sessions: {e}")
            return 0
    
    def bulk_deactivate_user_sessions(
        self, 
        user_id: int,
        session: Optional[Session] = None
    ) -> int:
        """批量停用用户的所有活跃会话"""
        try:
            # 获取用户的所有活跃会话
            active_sessions = self.get_active_sessions(
                user_id=user_id,
                limit=1000,
                session=session
            )
            
            # 批量停用
            deactivated_count = 0
            for session_obj in active_sessions:
                if self.deactivate_session(session_obj.id, session):
                    deactivated_count += 1
            
            logger.info(f"Bulk deactivated {deactivated_count} sessions for user {user_id}")
            return deactivated_count
            
        except Exception as e:
            logger.error(f"Error bulk deactivating sessions for user {user_id}: {e}")
            return 0
    
    def get_session_duration_stats(
        self, 
        user_id: Optional[int] = None,
        session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """获取会话持续时间统计"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 构建查询
            query = db_session.query(SessionModel)
            
            if user_id:
                query = query.filter(SessionModel.user_id == user_id)
            
            # 只统计已结束的会话
            query = query.filter(SessionModel.ended_at.isnot(None))
            
            sessions = query.all()
            
            if not sessions:
                return {
                    "total_sessions": 0,
                    "avg_duration_minutes": 0,
                    "min_duration_minutes": 0,
                    "max_duration_minutes": 0
                }
            
            # 计算持续时间
            durations = []
            for session_obj in sessions:
                if session_obj.created_at and session_obj.ended_at:
                    duration = session_obj.ended_at - session_obj.created_at
                    durations.append(duration.total_seconds() / 60)  # 转换为分钟
            
            if not durations:
                return {
                    "total_sessions": len(sessions),
                    "avg_duration_minutes": 0,
                    "min_duration_minutes": 0,
                    "max_duration_minutes": 0
                }
            
            return {
                "total_sessions": len(sessions),
                "avg_duration_minutes": sum(durations) / len(durations),
                "min_duration_minutes": min(durations),
                "max_duration_minutes": max(durations)
            }
            
        except Exception as e:
            logger.error(f"Error getting session duration stats: {e}")
            return {
                "total_sessions": 0,
                "avg_duration_minutes": 0,
                "min_duration_minutes": 0,
                "max_duration_minutes": 0
            }
        finally:
            if not session:
                db_session.close()