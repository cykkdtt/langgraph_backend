"""消息仓储模块

提供消息相关的数据访问接口和业务逻辑。
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.orm import Session, selectinload

from ..models.database import Message, MessageType, MessageRole
from ..models.api import MessageCreate, MessageUpdate, MessageStats
from ..database.repository import CRUDRepository, EntityNotFoundError
from ..database.query_builder import QueryFilter
from ..database.session import SessionManager

logger = logging.getLogger(__name__)


class MessageRepository(CRUDRepository[Message]):
    """消息仓储类"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        super().__init__(Message, session_manager)
        # 设置软删除字段
        self.set_soft_delete_field("deleted_at")
        # 设置默认加载选项
        self.set_default_load_options(
            selectinload(Message.thread),
            selectinload(Message.parent_message),
            selectinload(Message.child_messages),
            selectinload(Message.attachments)
        )
    
    def create_message(
        self, 
        message_create: MessageCreate, 
        thread_id: int,
        parent_id: Optional[int] = None,
        session: Optional[Session] = None
    ) -> Message:
        """创建消息"""
        message_data = message_create.model_dump()
        message_data["thread_id"] = thread_id
        
        if parent_id:
            message_data["parent_id"] = parent_id
        
        # 设置消息序号
        message_data["sequence_number"] = self._get_next_sequence_number(
            thread_id, session
        )
        
        return self.create(message_data, session)
    
    def _get_next_sequence_number(
        self, 
        thread_id: int,
        session: Optional[Session] = None
    ) -> int:
        """获取下一个序号"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 获取线程中最大的序号
            max_seq = db_session.query(func.max(Message.sequence_number)).filter(
                Message.thread_id == thread_id
            ).scalar()
            
            return (max_seq or 0) + 1
            
        except Exception as e:
            logger.error(f"Error getting next sequence number for thread {thread_id}: {e}")
            return 1
        finally:
            if not session:
                db_session.close()
    
    def get_thread_messages(
        self, 
        thread_id: int, 
        skip: int = 0, 
        limit: int = 100,
        order_by: str = "asc",
        session: Optional[Session] = None
    ) -> List[Message]:
        """获取线程的消息列表"""
        filters = QueryFilter().eq("thread_id", thread_id)
        
        order_direction = "asc" if order_by.lower() == "asc" else "desc"
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("sequence_number", order_direction)],
            session=session
        )
    
    def get_messages_by_role(
        self, 
        thread_id: int,
        role: MessageRole,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Message]:
        """根据角色获取消息"""
        filters = QueryFilter().and_(
            QueryFilter().eq("thread_id", thread_id),
            QueryFilter().eq("role", role)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("sequence_number", "asc")],
            session=session
        )
    
    def get_messages_by_type(
        self, 
        thread_id: int,
        message_type: MessageType,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Message]:
        """根据类型获取消息"""
        filters = QueryFilter().and_(
            QueryFilter().eq("thread_id", thread_id),
            QueryFilter().eq("message_type", message_type)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("sequence_number", "asc")],
            session=session
        )
    
    def get_conversation_history(
        self, 
        thread_id: int,
        limit: int = 50,
        before_message_id: Optional[int] = None,
        session: Optional[Session] = None
    ) -> List[Message]:
        """获取对话历史"""
        filters = QueryFilter().eq("thread_id", thread_id)
        
        if before_message_id:
            # 获取指定消息的序号
            before_message = self.get(before_message_id, session)
            if before_message:
                filters = filters.and_(
                    QueryFilter().lt("sequence_number", before_message.sequence_number)
                )
        
        return self.get_multi(
            limit=limit,
            filters=filters,
            order_by=[("sequence_number", "desc")],
            session=session
        )
    
    def get_recent_messages(
        self, 
        thread_id: int,
        minutes: int = 30,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Message]:
        """获取最近的消息"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        filters = QueryFilter().and_(
            QueryFilter().eq("thread_id", thread_id),
            QueryFilter().gte("created_at", cutoff_time)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def search_messages(
        self, 
        thread_id: int,
        query: str, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Message]:
        """搜索消息内容"""
        filters = QueryFilter().and_(
            QueryFilter().eq("thread_id", thread_id),
            QueryFilter().ilike("content", f"%{query}%")
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("sequence_number", "desc")],
            session=session
        )
    
    def search_user_messages(
        self, 
        user_id: int,
        query: str, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Message]:
        """搜索用户的所有消息"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 通过thread和session表连接查询
            query_obj = db_session.query(Message).join(Message.thread).join(Message.thread.session)
            query_obj = query_obj.filter(Message.thread.session.has(user_id=user_id))
            query_obj = query_obj.filter(Message.content.ilike(f"%{query}%"))
            
            query_obj = query_obj.order_by(Message.created_at.desc())
            query_obj = query_obj.offset(skip).limit(limit)
            
            return query_obj.all()
            
        except Exception as e:
            logger.error(f"Error searching user messages for user {user_id}: {e}")
            return []
        finally:
            if not session:
                db_session.close()
    
    def get_message_thread(
        self, 
        message_id: int,
        session: Optional[Session] = None
    ) -> List[Message]:
        """获取消息线程（父子消息链）"""
        try:
            message = self.get(message_id, session)
            if not message:
                return []
            
            # 找到根消息
            root_message = message
            while root_message.parent_id:
                parent = self.get(root_message.parent_id, session)
                if parent:
                    root_message = parent
                else:
                    break
            
            # 获取整个消息树
            return self._get_message_tree(root_message.id, session)
            
        except Exception as e:
            logger.error(f"Error getting message thread for message {message_id}: {e}")
            return []
    
    def _get_message_tree(
        self, 
        root_id: int,
        session: Optional[Session] = None
    ) -> List[Message]:
        """递归获取消息树"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 使用递归CTE查询消息树
            from sqlalchemy import text
            
            query = text("""
                WITH RECURSIVE message_tree AS (
                    -- 根消息
                    SELECT id, parent_id, thread_id, content, role, message_type, 
                           sequence_number, created_at, updated_at, 0 as level
                    FROM messages 
                    WHERE id = :root_id AND deleted_at IS NULL
                    
                    UNION ALL
                    
                    -- 子消息
                    SELECT m.id, m.parent_id, m.thread_id, m.content, m.role, m.message_type,
                           m.sequence_number, m.created_at, m.updated_at, mt.level + 1
                    FROM messages m
                    INNER JOIN message_tree mt ON m.parent_id = mt.id
                    WHERE m.deleted_at IS NULL
                )
                SELECT * FROM message_tree ORDER BY level, sequence_number
            """)
            
            result = db_session.execute(query, {"root_id": root_id})
            
            # 将结果转换为Message对象
            messages = []
            for row in result:
                message = db_session.get(Message, row.id)
                if message:
                    messages.append(message)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting message tree for root {root_id}: {e}")
            return []
        finally:
            if not session:
                db_session.close()
    
    def get_message_statistics(
        self, 
        thread_id: Optional[int] = None,
        user_id: Optional[int] = None,
        session: Optional[Session] = None
    ) -> MessageStats:
        """获取消息统计信息"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            base_filters = QueryFilter()
            
            if thread_id:
                base_filters = base_filters.eq("thread_id", thread_id)
            elif user_id:
                # 通过thread和session表连接查询用户的消息
                query = db_session.query(Message).join(Message.thread).join(Message.thread.session)
                query = query.filter(Message.thread.session.has(user_id=user_id))
                
                total_messages = query.count()
                
                user_messages = query.filter(Message.role == MessageRole.USER).count()
                assistant_messages = query.filter(Message.role == MessageRole.ASSISTANT).count()
                system_messages = query.filter(Message.role == MessageRole.SYSTEM).count()
                
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                today_messages = query.filter(Message.created_at >= today_start).count()
                
                return MessageStats(
                    total_messages=total_messages,
                    user_messages=user_messages,
                    assistant_messages=assistant_messages,
                    system_messages=system_messages,
                    today_messages=today_messages
                )
            
            # 总消息数
            total_messages = self.count(filters=base_filters, session=db_session)
            
            # 按角色统计
            user_filters = base_filters.and_(QueryFilter().eq("role", MessageRole.USER))
            user_messages = self.count(filters=user_filters, session=db_session)
            
            assistant_filters = base_filters.and_(QueryFilter().eq("role", MessageRole.ASSISTANT))
            assistant_messages = self.count(filters=assistant_filters, session=db_session)
            
            system_filters = base_filters.and_(QueryFilter().eq("role", MessageRole.SYSTEM))
            system_messages = self.count(filters=system_filters, session=db_session)
            
            # 今日消息数
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_filters = base_filters.and_(QueryFilter().gte("created_at", today_start))
            today_messages = self.count(filters=today_filters, session=db_session)
            
            return MessageStats(
                total_messages=total_messages,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                system_messages=system_messages,
                today_messages=today_messages
            )
            
        except Exception as e:
            logger.error(f"Error getting message statistics: {e}")
            return MessageStats(
                total_messages=0,
                user_messages=0,
                assistant_messages=0,
                system_messages=0,
                today_messages=0
            )
        finally:
            if not session:
                db_session.close()
    
    def get_messages_by_date_range(
        self, 
        thread_id: int,
        start_date: datetime,
        end_date: datetime,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Message]:
        """根据日期范围获取消息"""
        filters = QueryFilter().and_(
            QueryFilter().eq("thread_id", thread_id),
            QueryFilter().gte("created_at", start_date),
            QueryFilter().lte("created_at", end_date)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("sequence_number", "asc")],
            session=session
        )
    
    def get_message_context(
        self, 
        message_id: int,
        context_size: int = 5,
        session: Optional[Session] = None
    ) -> Tuple[List[Message], Message, List[Message]]:
        """获取消息上下文（前后消息）"""
        try:
            message = self.get(message_id, session)
            if not message:
                return [], None, []
            
            thread_id = message.thread_id
            sequence_number = message.sequence_number
            
            # 获取前面的消息
            before_filters = QueryFilter().and_(
                QueryFilter().eq("thread_id", thread_id),
                QueryFilter().lt("sequence_number", sequence_number)
            )
            
            before_messages = self.get_multi(
                limit=context_size,
                filters=before_filters,
                order_by=[("sequence_number", "desc")],
                session=session
            )
            before_messages.reverse()  # 恢复正序
            
            # 获取后面的消息
            after_filters = QueryFilter().and_(
                QueryFilter().eq("thread_id", thread_id),
                QueryFilter().gt("sequence_number", sequence_number)
            )
            
            after_messages = self.get_multi(
                limit=context_size,
                filters=after_filters,
                order_by=[("sequence_number", "asc")],
                session=session
            )
            
            return before_messages, message, after_messages
            
        except Exception as e:
            logger.error(f"Error getting message context for message {message_id}: {e}")
            return [], None, []
    
    def update_message_metadata(
        self, 
        message_id: int,
        metadata: Dict[str, Any],
        session: Optional[Session] = None
    ) -> Optional[Message]:
        """更新消息元数据"""
        try:
            message = self.get(message_id, session)
            if not message:
                raise EntityNotFoundError(f"Message with ID {message_id} not found")
            
            # 合并元数据
            current_metadata = message.metadata or {}
            current_metadata.update(metadata)
            
            update_data = {
                "metadata": current_metadata
            }
            
            return self.update(message, update_data, session)
            
        except Exception as e:
            logger.error(f"Error updating message metadata for message {message_id}: {e}")
            return None
    
    def mark_message_as_edited(
        self, 
        message_id: int,
        session: Optional[Session] = None
    ) -> Optional[Message]:
        """标记消息为已编辑"""
        try:
            message = self.get(message_id, session)
            if not message:
                raise EntityNotFoundError(f"Message with ID {message_id} not found")
            
            update_data = {
                "is_edited": True,
                "edited_at": datetime.utcnow()
            }
            
            return self.update(message, update_data, session)
            
        except Exception as e:
            logger.error(f"Error marking message as edited for message {message_id}: {e}")
            return None
    
    def get_edited_messages(
        self, 
        thread_id: int,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Message]:
        """获取已编辑的消息"""
        filters = QueryFilter().and_(
            QueryFilter().eq("thread_id", thread_id),
            QueryFilter().eq("is_edited", True)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("edited_at", "desc")],
            session=session
        )
    
    def get_messages_with_attachments(
        self, 
        thread_id: int,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Message]:
        """获取有附件的消息"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 查询有附件的消息
            query = db_session.query(Message).join(Message.attachments)
            query = query.filter(Message.thread_id == thread_id)
            query = query.order_by(Message.sequence_number.desc())
            query = query.offset(skip).limit(limit)
            
            return query.all()
            
        except Exception as e:
            logger.error(f"Error getting messages with attachments for thread {thread_id}: {e}")
            return []
        finally:
            if not session:
                db_session.close()
    
    def cleanup_old_messages(
        self, 
        days: int = 90,
        session: Optional[Session] = None
    ) -> int:
        """清理旧消息（软删除）"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # 查找旧消息
            filters = QueryFilter().lt("created_at", cutoff_date)
            
            old_messages = self.get_multi(
                filters=filters,
                limit=1000,  # 批量处理
                session=session
            )
            
            # 软删除这些消息
            deleted_count = 0
            for message in old_messages:
                if self.soft_delete(message.id, session):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old messages")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old messages: {e}")
            return 0
    
    def get_message_word_count(
        self, 
        message_id: int,
        session: Optional[Session] = None
    ) -> int:
        """获取消息字数"""
        try:
            message = self.get(message_id, session)
            if not message or not message.content:
                return 0
            
            # 简单的字数统计（按空格分割）
            words = message.content.split()
            return len(words)
            
        except Exception as e:
            logger.error(f"Error getting word count for message {message_id}: {e}")
            return 0
    
    def get_thread_word_count(
        self, 
        thread_id: int,
        session: Optional[Session] = None
    ) -> int:
        """获取线程总字数"""
        try:
            messages = self.get_thread_messages(thread_id, limit=10000, session=session)
            
            total_words = 0
            for message in messages:
                if message.content:
                    words = message.content.split()
                    total_words += len(words)
            
            return total_words
            
        except Exception as e:
            logger.error(f"Error getting word count for thread {thread_id}: {e}")
            return 0