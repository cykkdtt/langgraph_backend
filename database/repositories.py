"""具体仓储实现类

为每个数据库模型提供专门的仓储操作方法，包括用户、消息、工作流、记忆等。
实现特定的业务逻辑和复杂查询。
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import and_, or_, func, select, update, desc, asc
from sqlalchemy.orm import Session, selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database_models import (
    User, Session as UserSession, Message, ToolCall, AgentState,
    SystemLog, Workflow, WorkflowExecution, Memory
)
from ..models.workflow_models import (
    WorkflowCreateRequest, WorkflowUpdateRequest, WorkflowExecuteRequest,
    ExecutionNodeStatus, WorkflowInfo, WorkflowStatistics
)
from ..utils.validation import (
    ValidationException, ResourceNotFoundException, BusinessRuleException
)
from .repository import SyncRepository, AsyncRepository, QueryFilter, QuerySort


class UserRepository(SyncRepository[User]):
    """用户仓储"""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(User, session)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        return self.get_by_field("email", email)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        return self.get_by_field("username", username)
    
    def get_by_api_key_hash(self, api_key_hash: str) -> Optional[User]:
        """根据API密钥哈希获取用户"""
        return self.get_by_field("api_key_hash", api_key_hash)
    
    def check_email_exists(self, email: str, exclude_user_id: Optional[UUID] = None) -> bool:
        """检查邮箱是否已存在"""
        query = select(User).where(User.email == email)
        if exclude_user_id:
            query = query.where(User.id != exclude_user_id)
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalar_one_or_none() is not None
    
    def check_username_exists(self, username: str, exclude_user_id: Optional[UUID] = None) -> bool:
        """检查用户名是否已存在"""
        query = select(User).where(User.username == username)
        if exclude_user_id:
            query = query.where(User.id != exclude_user_id)
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalar_one_or_none() is not None
    
    def get_active_users(self, days: int = 30) -> List[User]:
        """获取活跃用户（指定天数内有活动）"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = (
            select(User)
            .where(User.last_login_at >= cutoff_date)
            .where(User.is_active == True)
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_users_by_rate_limit_tier(self, tier: str) -> List[User]:
        """根据限流等级获取用户"""
        query = select(User).where(User.rate_limit_tier == tier)
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def update_last_login(self, user_id: UUID) -> bool:
        """更新最后登录时间"""
        try:
            user = self.get_or_404(user_id)
            user.last_login_at = datetime.utcnow()
            user.login_count = (user.login_count or 0) + 1
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Failed to update last login for user {user_id}: {e}")
            return False
    
    def deactivate_user(self, user_id: UUID, reason: Optional[str] = None) -> bool:
        """停用用户"""
        try:
            user = self.get_or_404(user_id)
            user.is_active = False
            user.deactivated_at = datetime.utcnow()
            if reason:
                user.deactivation_reason = reason
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Failed to deactivate user {user_id}: {e}")
            return False


class SessionRepository(SyncRepository[UserSession]):
    """会话仓储"""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(UserSession, session)
    
    def get_by_token(self, token: str) -> Optional[UserSession]:
        """根据令牌获取会话"""
        return self.get_by_field("token", token)
    
    def get_active_sessions(self, user_id: UUID) -> List[UserSession]:
        """获取用户的活跃会话"""
        query = (
            select(UserSession)
            .where(UserSession.user_id == user_id)
            .where(UserSession.is_active == True)
            .where(UserSession.expires_at > datetime.utcnow())
        )
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def cleanup_expired_sessions(self) -> int:
        """清理过期会话"""
        try:
            stmt = (
                update(UserSession)
                .where(UserSession.expires_at <= datetime.utcnow())
                .values(is_active=False)
            )
            result = self.session.execute(stmt)
            self.session.commit()
            
            cleaned_count = result.rowcount
            self.logger.info(f"Cleaned up {cleaned_count} expired sessions")
            return cleaned_count
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def revoke_user_sessions(self, user_id: UUID, exclude_session_id: Optional[UUID] = None) -> int:
        """撤销用户的所有会话（可排除指定会话）"""
        try:
            query = update(UserSession).where(UserSession.user_id == user_id)
            if exclude_session_id:
                query = query.where(UserSession.id != exclude_session_id)
            
            stmt = query.values(is_active=False, revoked_at=datetime.utcnow())
            result = self.session.execute(stmt)
            self.session.commit()
            
            revoked_count = result.rowcount
            self.logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
            return revoked_count
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Failed to revoke sessions for user {user_id}: {e}")
            return 0


class MessageRepository(SyncRepository[Message]):
    """消息仓储"""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(Message, session)
    
    def get_by_session(self, session_id: UUID, limit: int = 50) -> List[Message]:
        """获取会话的消息列表"""
        query = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(desc(Message.created_at))
            .limit(limit)
            .options(selectinload(Message.tool_calls))
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_conversation_history(
        self,
        session_id: UUID,
        before_message_id: Optional[UUID] = None,
        limit: int = 20
    ) -> List[Message]:
        """获取对话历史"""
        query = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(desc(Message.created_at))
        )
        
        if before_message_id:
            # 获取指定消息之前的消息
            before_message = self.get_or_404(before_message_id)
            query = query.where(Message.created_at < before_message.created_at)
        
        query = query.limit(limit).options(selectinload(Message.tool_calls))
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_messages_by_status(self, status: str, limit: int = 100) -> List[Message]:
        """根据状态获取消息"""
        query = (
            select(Message)
            .where(Message.status == status)
            .order_by(desc(Message.created_at))
            .limit(limit)
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_high_priority_messages(self, limit: int = 50) -> List[Message]:
        """获取高优先级消息"""
        query = (
            select(Message)
            .where(Message.priority.in_(['high', 'urgent']))
            .order_by(desc(Message.priority), desc(Message.created_at))
            .limit(limit)
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def search_messages(
        self,
        query_text: str,
        session_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        limit: int = 50
    ) -> List[Message]:
        """搜索消息内容"""
        query = select(Message).where(
            or_(
                Message.content.ilike(f"%{query_text}%"),
                Message.metadata.op('->>')('summary').ilike(f"%{query_text}%")
            )
        )
        
        if session_id:
            query = query.where(Message.session_id == session_id)
        
        if user_id:
            query = query.where(Message.user_id == user_id)
        
        query = (
            query.order_by(desc(Message.created_at))
            .limit(limit)
            .options(selectinload(Message.tool_calls))
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_message_statistics(self, user_id: Optional[UUID] = None) -> Dict[str, Any]:
        """获取消息统计信息"""
        base_query = select(Message)
        if user_id:
            base_query = base_query.where(Message.user_id == user_id)
        base_query = self._apply_soft_delete_filter(base_query)
        
        # 总消息数
        total_count = self.session.execute(
            select(func.count()).select_from(base_query.subquery())
        ).scalar()
        
        # 按角色统计
        role_stats = self.session.execute(
            select(Message.role, func.count())
            .select_from(base_query.subquery())
            .group_by(Message.role)
        ).all()
        
        # 按状态统计
        status_stats = self.session.execute(
            select(Message.status, func.count())
            .select_from(base_query.subquery())
            .group_by(Message.status)
        ).all()
        
        # 平均内容长度
        avg_length = self.session.execute(
            select(func.avg(Message.content_length))
            .select_from(base_query.subquery())
        ).scalar() or 0
        
        return {
            "total_messages": total_count,
            "by_role": dict(role_stats),
            "by_status": dict(status_stats),
            "average_content_length": float(avg_length)
        }
    
    def update_message_quality_score(self, message_id: UUID, score: float) -> bool:
        """更新消息质量评分"""
        try:
            message = self.get_or_404(message_id)
            message.quality_score = max(0.0, min(1.0, score))  # 确保分数在0-1之间
            message.updated_at = datetime.utcnow()
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Failed to update quality score for message {message_id}: {e}")
            return False


class WorkflowRepository(SyncRepository[Workflow]):
    """工作流仓储"""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(Workflow, session)
    
    def get_by_name(self, name: str, user_id: Optional[UUID] = None) -> Optional[Workflow]:
        """根据名称获取工作流"""
        query = select(Workflow).where(Workflow.name == name)
        if user_id:
            query = query.where(Workflow.created_by == user_id)
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalar_one_or_none()
    
    def get_published_workflows(self, category: Optional[str] = None) -> List[Workflow]:
        """获取已发布的工作流"""
        query = (
            select(Workflow)
            .where(Workflow.is_published == True)
            .where(Workflow.status == 'active')
        )
        
        if category:
            query = query.where(Workflow.category == category)
        
        query = (
            query.order_by(desc(Workflow.popularity_score), desc(Workflow.created_at))
            .options(selectinload(Workflow.executions))
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_user_workflows(self, user_id: UUID, include_shared: bool = True) -> List[Workflow]:
        """获取用户的工作流"""
        query = select(Workflow)
        
        if include_shared:
            # 包括用户创建的和共享给用户的工作流
            query = query.where(
                or_(
                    Workflow.created_by == user_id,
                    Workflow.shared_with.op('@>')([str(user_id)])
                )
            )
        else:
            query = query.where(Workflow.created_by == user_id)
        
        query = (
            query.order_by(desc(Workflow.updated_at))
            .options(selectinload(Workflow.executions))
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_workflows_by_category(self, category: str) -> List[Workflow]:
        """根据分类获取工作流"""
        query = (
            select(Workflow)
            .where(Workflow.category == category)
            .where(Workflow.status == 'active')
            .order_by(desc(Workflow.popularity_score))
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def search_workflows(
        self,
        query_text: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[UUID] = None
    ) -> List[Workflow]:
        """搜索工作流"""
        query = select(Workflow).where(
            or_(
                Workflow.name.ilike(f"%{query_text}%"),
                Workflow.description.ilike(f"%{query_text}%")
            )
        )
        
        if category:
            query = query.where(Workflow.category == category)
        
        if tags:
            for tag in tags:
                query = query.where(Workflow.tags.op('@>')([tag]))
        
        if user_id:
            query = query.where(
                or_(
                    Workflow.created_by == user_id,
                    and_(
                        Workflow.is_published == True,
                        Workflow.status == 'active'
                    )
                )
            )
        else:
            query = query.where(
                and_(
                    Workflow.is_published == True,
                    Workflow.status == 'active'
                )
            )
        
        query = query.order_by(desc(Workflow.popularity_score))
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_workflow_statistics(self, workflow_id: UUID) -> Dict[str, Any]:
        """获取工作流统计信息"""
        workflow = self.get_or_404(workflow_id)
        
        # 执行统计
        execution_stats = self.session.execute(
            select(
                func.count(WorkflowExecution.id).label('total_executions'),
                func.count().filter(WorkflowExecution.status == 'completed').label('successful_executions'),
                func.count().filter(WorkflowExecution.status == 'failed').label('failed_executions'),
                func.avg(WorkflowExecution.execution_time).label('avg_execution_time')
            )
            .where(WorkflowExecution.workflow_id == workflow_id)
        ).first()
        
        # 最近执行
        recent_executions = self.session.execute(
            select(WorkflowExecution)
            .where(WorkflowExecution.workflow_id == workflow_id)
            .order_by(desc(WorkflowExecution.created_at))
            .limit(10)
        ).scalars().all()
        
        return {
            "workflow_id": str(workflow_id),
            "name": workflow.name,
            "total_executions": execution_stats.total_executions or 0,
            "successful_executions": execution_stats.successful_executions or 0,
            "failed_executions": execution_stats.failed_executions or 0,
            "success_rate": (
                (execution_stats.successful_executions or 0) / max(execution_stats.total_executions or 1, 1)
            ) * 100,
            "average_execution_time": float(execution_stats.avg_execution_time or 0),
            "popularity_score": workflow.popularity_score,
            "recent_executions": [
                {
                    "id": str(exec.id),
                    "status": exec.status,
                    "created_at": exec.created_at.isoformat(),
                    "execution_time": exec.execution_time
                }
                for exec in recent_executions
            ]
        }
    
    def update_popularity_score(self, workflow_id: UUID) -> bool:
        """更新工作流热度评分"""
        try:
            workflow = self.get_or_404(workflow_id)
            
            # 计算热度评分（基于执行次数、成功率、最近活跃度等）
            stats = self.get_workflow_statistics(workflow_id)
            
            # 简单的热度计算公式
            execution_score = min(stats['total_executions'] * 0.1, 50)  # 执行次数权重
            success_score = stats['success_rate'] * 0.3  # 成功率权重
            recency_score = 20 if stats['recent_executions'] else 0  # 最近活跃度权重
            
            new_score = execution_score + success_score + recency_score
            
            workflow.popularity_score = min(new_score, 100)  # 最大100分
            workflow.updated_at = datetime.utcnow()
            self.session.commit()
            
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Failed to update popularity score for workflow {workflow_id}: {e}")
            return False


class WorkflowExecutionRepository(SyncRepository[WorkflowExecution]):
    """工作流执行仓储"""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(WorkflowExecution, session)
    
    def get_by_workflow(self, workflow_id: UUID, limit: int = 50) -> List[WorkflowExecution]:
        """获取工作流的执行记录"""
        query = (
            select(WorkflowExecution)
            .where(WorkflowExecution.workflow_id == workflow_id)
            .order_by(desc(WorkflowExecution.created_at))
            .limit(limit)
        )
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_running_executions(self, user_id: Optional[UUID] = None) -> List[WorkflowExecution]:
        """获取正在运行的执行"""
        query = (
            select(WorkflowExecution)
            .where(WorkflowExecution.status.in_(['running', 'pending']))
        )
        
        if user_id:
            query = query.where(WorkflowExecution.user_id == user_id)
        
        query = query.order_by(desc(WorkflowExecution.created_at))
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_failed_executions(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[WorkflowExecution]:
        """获取失败的执行记录"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = (
            select(WorkflowExecution)
            .where(WorkflowExecution.status == 'failed')
            .where(WorkflowExecution.created_at >= cutoff_time)
            .order_by(desc(WorkflowExecution.created_at))
            .limit(limit)
        )
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def cleanup_old_executions(self, days: int = 30) -> int:
        """清理旧的执行记录"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # 只删除已完成或失败的执行记录
            deleted_count = self.bulk_delete(
                self.session.execute(
                    select(WorkflowExecution.id)
                    .where(WorkflowExecution.created_at < cutoff_date)
                    .where(WorkflowExecution.status.in_(['completed', 'failed', 'cancelled']))
                ).scalars().all(),
                soft_delete=False
            )
            
            self.logger.info(f"Cleaned up {deleted_count} old workflow executions")
            return deleted_count
        except Exception as e:
            self.logger.error(f"Failed to cleanup old executions: {e}")
            return 0


class MemoryRepository(SyncRepository[Memory]):
    """记忆仓储"""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(Memory, session)
    
    def get_by_session(self, session_id: UUID, memory_type: Optional[str] = None) -> List[Memory]:
        """获取会话的记忆"""
        query = select(Memory).where(Memory.session_id == session_id)
        
        if memory_type:
            query = query.where(Memory.memory_type == memory_type)
        
        query = (
            query.order_by(desc(Memory.importance_score), desc(Memory.created_at))
            .options(selectinload(Memory.related_memories))
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_by_user(self, user_id: UUID, memory_type: Optional[str] = None) -> List[Memory]:
        """获取用户的记忆"""
        query = select(Memory).where(Memory.user_id == user_id)
        
        if memory_type:
            query = query.where(Memory.memory_type == memory_type)
        
        query = (
            query.order_by(desc(Memory.importance_score), desc(Memory.last_accessed_at))
            .options(selectinload(Memory.related_memories))
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def search_memories(
        self,
        query_text: str,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[Memory]:
        """搜索记忆内容"""
        query = select(Memory).where(
            or_(
                Memory.content.ilike(f"%{query_text}%"),
                Memory.summary.ilike(f"%{query_text}%"),
                Memory.tags.op('@>')([query_text])
            )
        )
        
        if user_id:
            query = query.where(Memory.user_id == user_id)
        
        if session_id:
            query = query.where(Memory.session_id == session_id)
        
        if memory_type:
            query = query.where(Memory.memory_type == memory_type)
        
        if min_importance > 0:
            query = query.where(Memory.importance_score >= min_importance)
        
        query = (
            query.order_by(desc(Memory.importance_score), desc(Memory.relevance_score))
            .options(selectinload(Memory.related_memories))
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def get_related_memories(self, memory_id: UUID, limit: int = 10) -> List[Memory]:
        """获取相关记忆"""
        memory = self.get_or_404(memory_id)
        
        # 基于标签和内容相似性查找相关记忆
        query = (
            select(Memory)
            .where(Memory.id != memory_id)
            .where(Memory.user_id == memory.user_id)
        )
        
        # 如果有标签，优先匹配标签
        if memory.tags:
            query = query.where(
                or_(*[Memory.tags.op('@>')([tag]) for tag in memory.tags])
            )
        
        query = (
            query.order_by(desc(Memory.importance_score))
            .limit(limit)
        )
        query = self._apply_soft_delete_filter(query)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def update_access_info(self, memory_id: UUID) -> bool:
        """更新记忆访问信息"""
        try:
            memory = self.get_or_404(memory_id)
            memory.last_accessed_at = datetime.utcnow()
            memory.access_count = (memory.access_count or 0) + 1
            
            # 根据访问频率调整重要性评分
            if memory.access_count > 10:
                memory.importance_score = min(memory.importance_score * 1.1, 1.0)
            
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Failed to update access info for memory {memory_id}: {e}")
            return False
    
    def apply_memory_decay(self, days_threshold: int = 30) -> int:
        """应用记忆衰减机制"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            # 降低长时间未访问记忆的重要性评分
            stmt = (
                update(Memory)
                .where(Memory.last_accessed_at < cutoff_date)
                .where(Memory.importance_score > 0.1)
                .values(
                    importance_score=Memory.importance_score * 0.9,
                    updated_at=datetime.utcnow()
                )
            )
            
            result = self.session.execute(stmt)
            self.session.commit()
            
            decayed_count = result.rowcount
            self.logger.info(f"Applied decay to {decayed_count} memories")
            return decayed_count
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Failed to apply memory decay: {e}")
            return 0
    
    def consolidate_memories(self, user_id: UUID, session_id: Optional[UUID] = None) -> int:
        """整合记忆（合并相似记忆）"""
        try:
            # 获取候选记忆（相同用户、相似标签）
            query = (
                select(Memory)
                .where(Memory.user_id == user_id)
                .where(Memory.importance_score < 0.5)  # 只整合低重要性记忆
            )
            
            if session_id:
                query = query.where(Memory.session_id == session_id)
            
            query = self._apply_soft_delete_filter(query)
            memories = self.session.execute(query).scalars().all()
            
            # 简单的整合逻辑：按标签分组，合并相似记忆
            consolidated_count = 0
            tag_groups = {}
            
            for memory in memories:
                if memory.tags:
                    key = tuple(sorted(memory.tags))
                    if key not in tag_groups:
                        tag_groups[key] = []
                    tag_groups[key].append(memory)
            
            # 合并每个标签组中的记忆
            for tag_key, group_memories in tag_groups.items():
                if len(group_memories) > 1:
                    # 保留重要性最高的记忆，删除其他
                    group_memories.sort(key=lambda m: m.importance_score, reverse=True)
                    primary_memory = group_memories[0]
                    
                    # 合并内容和更新统计
                    for memory in group_memories[1:]:
                        primary_memory.access_count += memory.access_count or 0
                        if memory.last_accessed_at > primary_memory.last_accessed_at:
                            primary_memory.last_accessed_at = memory.last_accessed_at
                        
                        # 软删除被合并的记忆
                        memory.deleted_at = datetime.utcnow()
                        consolidated_count += 1
            
            self.session.commit()
            self.logger.info(f"Consolidated {consolidated_count} memories for user {user_id}")
            return consolidated_count
            
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Failed to consolidate memories for user {user_id}: {e}")
            return 0


# 仓储管理器
class RepositoryManager:
    """仓储管理器"""
    
    def __init__(self, session: Optional[Session] = None):
        self.session = session
        self._repositories = {}
    
    def get_user_repository(self) -> UserRepository:
        """获取用户仓储"""
        if 'user' not in self._repositories:
            self._repositories['user'] = UserRepository(self.session)
        return self._repositories['user']
    
    def get_session_repository(self) -> SessionRepository:
        """获取会话仓储"""
        if 'session' not in self._repositories:
            self._repositories['session'] = SessionRepository(self.session)
        return self._repositories['session']
    
    def get_message_repository(self) -> MessageRepository:
        """获取消息仓储"""
        if 'message' not in self._repositories:
            self._repositories['message'] = MessageRepository(self.session)
        return self._repositories['message']
    
    def get_workflow_repository(self) -> WorkflowRepository:
        """获取工作流仓储"""
        if 'workflow' not in self._repositories:
            self._repositories['workflow'] = WorkflowRepository(self.session)
        return self._repositories['workflow']
    
    def get_workflow_execution_repository(self) -> WorkflowExecutionRepository:
        """获取工作流执行仓储"""
        if 'workflow_execution' not in self._repositories:
            self._repositories['workflow_execution'] = WorkflowExecutionRepository(self.session)
        return self._repositories['workflow_execution']
    
    def get_memory_repository(self) -> MemoryRepository:
        """获取记忆仓储"""
        if 'memory' not in self._repositories:
            self._repositories['memory'] = MemoryRepository(self.session)
        return self._repositories['memory']


# 全局仓储管理器实例
_repository_manager = None


def get_repository_manager(session: Optional[Session] = None) -> RepositoryManager:
    """获取仓储管理器实例"""
    global _repository_manager
    if _repository_manager is None or session is not None:
        _repository_manager = RepositoryManager(session)
    return _repository_manager