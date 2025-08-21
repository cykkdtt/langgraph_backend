"""线程仓储模块

提供对话线程相关的数据访问接口和业务逻辑。
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session, selectinload

from ..models.database import Thread
from ..models.api import ThreadCreate, ThreadUpdate, ThreadStats
from ..database.repository import CRUDRepository, EntityNotFoundError
from ..database.query_builder import QueryFilter
from ..database.session import SessionManager

logger = logging.getLogger(__name__)


class ThreadRepository(CRUDRepository[Thread]):
    """线程仓储类"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        super().__init__(Thread, session_manager)
        # 设置软删除字段
        self.set_soft_delete_field("deleted_at")
        # 设置默认加载选项
        self.set_default_load_options(
            selectinload(Thread.session),
            selectinload(Thread.messages),
            selectinload(Thread.memories)
        )
    
    def create_thread(
        self, 
        thread_create: ThreadCreate, 
        session_id: int,
        session: Optional[Session] = None
    ) -> Thread:
        """创建线程"""
        thread_data = thread_create.model_dump()
        thread_data["session_id"] = session_id
        thread_data["is_active"] = True
        
        return self.create(thread_data, session)
    
    def get_session_threads(
        self, 
        session_id: int, 
        skip: int = 0, 
        limit: int = 100,
        include_inactive: bool = False,
        session: Optional[Session] = None
    ) -> List[Thread]:
        """获取会话的线程列表"""
        filters = QueryFilter().eq("session_id", session_id)
        
        if not include_inactive:
            filters = filters.and_(QueryFilter().eq("is_active", True))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_user_threads(
        self, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 100,
        include_inactive: bool = False,
        session: Optional[Session] = None
    ) -> List[Thread]:
        """获取用户的所有线程"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 通过session表连接查询
            query = db_session.query(Thread).join(Thread.session)
            query = query.filter(Thread.session.has(user_id=user_id))
            
            if not include_inactive:
                query = query.filter(Thread.is_active == True)
            
            query = query.order_by(Thread.created_at.desc())
            query = query.offset(skip).limit(limit)
            
            return query.all()
            
        except Exception as e:
            logger.error(f"Error getting user threads for user {user_id}: {e}")
            return []
        finally:
            if not session:
                db_session.close()
    
    def get_active_threads(
        self, 
        session_id: Optional[int] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Thread]:
        """获取活跃线程"""
        filters = QueryFilter().eq("is_active", True)
        
        if session_id:
            filters = filters.and_(QueryFilter().eq("session_id", session_id))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("updated_at", "desc")],
            session=session
        )
    
    def update_thread_activity(
        self, 
        thread_id: int,
        session: Optional[Session] = None
    ) -> Optional[Thread]:
        """更新线程活动时间"""
        try:
            thread = self.get(thread_id, session)
            if not thread:
                raise EntityNotFoundError(f"Thread with ID {thread_id} not found")
            
            update_data = {
                "last_message_at": datetime.utcnow()
            }
            
            return self.update(thread, update_data, session)
            
        except Exception as e:
            logger.error(f"Error updating thread activity for thread {thread_id}: {e}")
            return None
    
    def deactivate_thread(
        self, 
        thread_id: int,
        session: Optional[Session] = None
    ) -> bool:
        """停用线程"""
        try:
            thread = self.get(thread_id, session)
            if not thread:
                raise EntityNotFoundError(f"Thread with ID {thread_id} not found")
            
            update_data = {
                "is_active": False
            }
            
            self.update(thread, update_data, session)
            
            logger.info(f"Thread {thread_id} deactivated")
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating thread {thread_id}: {e}")
            return False
    
    def activate_thread(
        self, 
        thread_id: int,
        session: Optional[Session] = None
    ) -> bool:
        """激活线程"""
        try:
            thread = self.get(thread_id, session)
            if not thread:
                raise EntityNotFoundError(f"Thread with ID {thread_id} not found")
            
            update_data = {
                "is_active": True
            }
            
            self.update(thread, update_data, session)
            
            logger.info(f"Thread {thread_id} activated")
            return True
            
        except Exception as e:
            logger.error(f"Error activating thread {thread_id}: {e}")
            return False
    
    def get_thread_by_title(
        self, 
        session_id: int, 
        title: str,
        session: Optional[Session] = None
    ) -> Optional[Thread]:
        """根据标题获取线程"""
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
            QueryFilter().eq("title", title)
        )
        
        threads = self.get_multi(
            limit=1,
            filters=filters,
            session=session
        )
        
        return threads[0] if threads else None
    
    def search_threads(
        self, 
        session_id: int,
        query: str, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Thread]:
        """搜索会话线程"""
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
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
    
    def search_user_threads(
        self, 
        user_id: int,
        query: str, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Thread]:
        """搜索用户的所有线程"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 通过session表连接查询
            query_obj = db_session.query(Thread).join(Thread.session)
            query_obj = query_obj.filter(Thread.session.has(user_id=user_id))
            query_obj = query_obj.filter(
                or_(
                    Thread.title.ilike(f"%{query}%"),
                    Thread.description.ilike(f"%{query}%")
                )
            )
            
            query_obj = query_obj.order_by(Thread.updated_at.desc())
            query_obj = query_obj.offset(skip).limit(limit)
            
            return query_obj.all()
            
        except Exception as e:
            logger.error(f"Error searching user threads for user {user_id}: {e}")
            return []
        finally:
            if not session:
                db_session.close()
    
    def get_recent_threads(
        self, 
        session_id: int,
        days: int = 7, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Thread]:
        """获取最近的线程"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
            QueryFilter().gte("last_message_at", cutoff_date)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("last_message_at", "desc")],
            session=session
        )
    
    def get_threads_by_date_range(
        self, 
        session_id: int,
        start_date: datetime,
        end_date: datetime,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Thread]:
        """根据日期范围获取线程"""
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
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
    
    def get_thread_statistics(
        self, 
        session_id: Optional[int] = None,
        user_id: Optional[int] = None,
        session: Optional[Session] = None
    ) -> ThreadStats:
        """获取线程统计信息"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            base_filters = QueryFilter()
            
            if session_id:
                base_filters = base_filters.eq("session_id", session_id)
            elif user_id:
                # 通过session表连接查询用户的线程
                query = db_session.query(Thread).join(Thread.session)
                query = query.filter(Thread.session.has(user_id=user_id))
                
                total_threads = query.count()
                active_threads = query.filter(Thread.is_active == True).count()
                
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                today_threads = query.filter(Thread.created_at >= today_start).count()
                
                week_ago = datetime.utcnow() - timedelta(days=7)
                recent_active_threads = query.filter(Thread.last_message_at >= week_ago).count()
                
                return ThreadStats(
                    total_threads=total_threads,
                    active_threads=active_threads,
                    today_threads=today_threads,
                    recent_active_threads=recent_active_threads
                )
            
            # 总线程数
            total_threads = self.count(filters=base_filters, session=db_session)
            
            # 活跃线程数
            active_filters = base_filters.and_(QueryFilter().eq("is_active", True))
            active_threads = self.count(filters=active_filters, session=db_session)
            
            # 今日创建的线程数
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_filters = base_filters.and_(QueryFilter().gte("created_at", today_start))
            today_threads = self.count(filters=today_filters, session=db_session)
            
            # 最近7天活跃的线程数
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_filters = base_filters.and_(QueryFilter().gte("last_message_at", week_ago))
            recent_active_threads = self.count(filters=recent_filters, session=db_session)
            
            return ThreadStats(
                total_threads=total_threads,
                active_threads=active_threads,
                today_threads=today_threads,
                recent_active_threads=recent_active_threads
            )
            
        except Exception as e:
            logger.error(f"Error getting thread statistics: {e}")
            return ThreadStats(
                total_threads=0,
                active_threads=0,
                today_threads=0,
                recent_active_threads=0
            )
        finally:
            if not session:
                db_session.close()
    
    def update_message_count(
        self, 
        thread_id: int,
        increment: int = 1,
        session: Optional[Session] = None
    ) -> Optional[Thread]:
        """更新线程消息数量"""
        try:
            thread = self.get(thread_id, session)
            if not thread:
                raise EntityNotFoundError(f"Thread with ID {thread_id} not found")
            
            new_count = max(0, (thread.message_count or 0) + increment)
            update_data = {
                "message_count": new_count,
                "last_message_at": datetime.utcnow()
            }
            
            return self.update(thread, update_data, session)
            
        except Exception as e:
            logger.error(f"Error updating message count for thread {thread_id}: {e}")
            return None
    
    def cleanup_inactive_threads(
        self, 
        days: int = 30,
        session: Optional[Session] = None
    ) -> int:
        """清理长期不活跃的线程"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # 查找长期不活跃的线程
            filters = QueryFilter().and_(
                QueryFilter().eq("is_active", True),
                QueryFilter().or_(
                    QueryFilter().is_null("last_message_at"),
                    QueryFilter().lt("last_message_at", cutoff_date)
                )
            )
            
            inactive_threads = self.get_multi(
                filters=filters,
                limit=1000,  # 批量处理
                session=session
            )
            
            # 停用这些线程
            deactivated_count = 0
            for thread in inactive_threads:
                if self.deactivate_thread(thread.id, session):
                    deactivated_count += 1
            
            logger.info(f"Cleaned up {deactivated_count} inactive threads")
            return deactivated_count
            
        except Exception as e:
            logger.error(f"Error cleaning up inactive threads: {e}")
            return 0
    
    def bulk_deactivate_session_threads(
        self, 
        session_id: int,
        session: Optional[Session] = None
    ) -> int:
        """批量停用会话的所有活跃线程"""
        try:
            # 获取会话的所有活跃线程
            active_threads = self.get_active_threads(
                session_id=session_id,
                limit=1000,
                session=session
            )
            
            # 批量停用
            deactivated_count = 0
            for thread in active_threads:
                if self.deactivate_thread(thread.id, session):
                    deactivated_count += 1
            
            logger.info(f"Bulk deactivated {deactivated_count} threads for session {session_id}")
            return deactivated_count
            
        except Exception as e:
            logger.error(f"Error bulk deactivating threads for session {session_id}: {e}")
            return 0
    
    def get_threads_with_message_count(
        self, 
        session_id: int,
        min_messages: int = 1,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Thread]:
        """获取有指定消息数量的线程"""
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
            QueryFilter().gte("message_count", min_messages)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("message_count", "desc")],
            session=session
        )
    
    def get_most_active_threads(
        self, 
        session_id: Optional[int] = None,
        days: int = 7,
        skip: int = 0, 
        limit: int = 10,
        session: Optional[Session] = None
    ) -> List[Thread]:
        """获取最活跃的线程"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        filters = QueryFilter().and_(
            QueryFilter().eq("is_active", True),
            QueryFilter().gte("last_message_at", cutoff_date)
        )
        
        if session_id:
            filters = filters.and_(QueryFilter().eq("session_id", session_id))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("message_count", "desc"), ("last_message_at", "desc")],
            session=session
        )