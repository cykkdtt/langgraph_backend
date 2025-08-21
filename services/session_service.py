"""会话服务

提供会话管理相关的业务逻辑，包括会话创建、更新、删除、权限管理等功能。
"""

import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from pydantic import BaseModel, validator
from sqlalchemy.orm import Session as DBSession

from .base import BaseService, ServiceError, CacheConfig, publish_event
from ..models.database_models import Session as ChatSession, User, Message
from ..models.response_models import SessionResponse, BaseResponse
from ..database.repositories import SessionRepository, UserRepository, MessageRepository
from ..utils.validation import (
    ValidationException, BusinessRuleException, 
    PermissionDeniedException, DataValidator
)
from ..utils.performance_monitoring import monitor_performance


class SessionCreateSchema(BaseModel):
    """会话创建模式"""
    title: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None
    is_persistent: Optional[bool] = True
    max_messages: Optional[int] = 1000
    auto_archive_days: Optional[int] = 30
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None and not DataValidator.validate_content_length(v, max_length=200):
            raise ValueError("Title exceeds maximum length")
        return v
    
    @validator('description')
    def validate_description(cls, v):
        if v is not None and not DataValidator.validate_content_length(v, max_length=1000):
            raise ValueError("Description exceeds maximum length")
        return v
    
    @validator('max_messages')
    def validate_max_messages(cls, v):
        if v is not None and (v < 1 or v > 10000):
            raise ValueError("Max messages must be between 1 and 10000")
        return v
    
    @validator('auto_archive_days')
    def validate_auto_archive_days(cls, v):
        if v is not None and (v < 1 or v > 365):
            raise ValueError("Auto archive days must be between 1 and 365")
        return v


class SessionUpdateSchema(BaseModel):
    """会话更新模式"""
    title: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None
    status: Optional[str] = None  # active, paused, archived, deleted
    is_persistent: Optional[bool] = None
    max_messages: Optional[int] = None
    auto_archive_days: Optional[int] = None
    
    @validator('status')
    def validate_status(cls, v):
        if v is not None:
            valid_statuses = ["active", "paused", "archived", "deleted"]
            if v not in valid_statuses:
                raise ValueError(f"Status must be one of: {valid_statuses}")
        return v


class SessionSearchSchema(BaseModel):
    """会话搜索模式"""
    user_id: Optional[UUID] = None
    status: Optional[str] = None
    title_query: Optional[str] = None
    description_query: Optional[str] = None
    created_from: Optional[datetime] = None
    created_to: Optional[datetime] = None
    updated_from: Optional[datetime] = None
    updated_to: Optional[datetime] = None
    is_persistent: Optional[bool] = None
    has_messages: Optional[bool] = None
    min_message_count: Optional[int] = None
    max_message_count: Optional[int] = None


class SessionStatsResponse(BaseModel):
    """会话统计响应"""
    total_sessions: int
    sessions_by_status: Dict[str, int]
    active_sessions: int
    archived_sessions: int
    deleted_sessions: int
    total_messages: int
    average_messages_per_session: float
    sessions_today: int
    sessions_this_week: int
    sessions_this_month: int
    most_active_sessions: List[Dict[str, Any]]
    recent_sessions: List[Dict[str, Any]]
    generated_at: datetime


class SessionAnalyticsResponse(BaseModel):
    """会话分析响应"""
    session_id: UUID
    title: Optional[str]
    status: str
    created_at: datetime
    last_activity: Optional[datetime]
    duration_hours: Optional[float]
    message_count: int
    user_message_count: int
    assistant_message_count: int
    system_message_count: int
    tool_message_count: int
    average_message_length: float
    total_content_length: int
    conversation_turns: int
    response_times: List[float]
    average_response_time: Optional[float]
    quality_metrics: Dict[str, float]
    engagement_score: float
    cost_estimation: Dict[str, float]


class SessionService(BaseService[ChatSession, SessionCreateSchema, SessionUpdateSchema, SessionResponse]):
    """会话服务"""
    
    def __init__(
        self, 
        repository: SessionRepository,
        user_repository: UserRepository,
        message_repository: MessageRepository,
        session: Optional[DBSession] = None
    ):
        cache_config = CacheConfig(
            enabled=True,
            ttl=600,  # 10分钟
            key_prefix="session_service",
            invalidate_on_update=True
        )
        super().__init__(repository, SessionResponse, cache_config, session)
        self.session_repository = repository
        self.user_repository = user_repository
        self.message_repository = message_repository
    
    def _generate_session_title(self, user_id: UUID) -> str:
        """生成会话标题"""
        # 获取用户的会话数量
        session_count = self.session_repository.count_user_sessions(user_id)
        return f"对话 {session_count + 1}"
    
    def _calculate_engagement_score(self, session_data: Dict[str, Any]) -> float:
        """计算参与度评分"""
        # 基于消息数量、对话轮次、响应时间等计算参与度
        message_count = session_data.get("message_count", 0)
        conversation_turns = session_data.get("conversation_turns", 0)
        avg_response_time = session_data.get("average_response_time", 0)
        duration_hours = session_data.get("duration_hours", 0)
        
        # 归一化各项指标
        message_score = min(message_count / 50, 1.0)  # 50条消息为满分
        turn_score = min(conversation_turns / 25, 1.0)  # 25轮对话为满分
        time_score = max(0, 1.0 - (avg_response_time / 300))  # 5分钟内响应为满分
        duration_score = min(duration_hours / 2, 1.0)  # 2小时为满分
        
        # 加权平均
        engagement_score = (
            message_score * 0.3 +
            turn_score * 0.3 +
            time_score * 0.2 +
            duration_score * 0.2
        )
        
        return round(engagement_score, 3)
    
    def _validate_business_rules(self, data: Any, action: str = "create"):
        """验证业务规则"""
        if action == "create":
            # 检查用户是否存在
            if self.current_user_id:
                user = self.user_repository.get(self.current_user_id)
                if not user:
                    raise BusinessRuleException("User not found")
                
                # 检查用户是否已达到会话限制
                active_sessions = self.session_repository.count_active_sessions(self.current_user_id)
                max_sessions = getattr(user, 'max_sessions', 100)  # 默认最多100个活跃会话
                
                if active_sessions >= max_sessions:
                    raise BusinessRuleException(f"Maximum number of active sessions ({max_sessions}) reached")
    
    def _check_permission(self, action: str, resource: Any = None):
        """检查权限"""
        if not self.current_user_id:
            raise PermissionDeniedException("Authentication required")
        
        if action in ["create", "update", "delete", "archive"] and resource:
            # 检查会话是否属于当前用户
            if str(resource.user_id) != str(self.current_user_id):
                # 检查是否是管理员
                if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                    raise PermissionDeniedException("Can only access your own sessions")
        
        if action in ["admin_stats", "admin_search", "admin_analytics"]:
            # 需要管理员权限
            if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                raise PermissionDeniedException("Admin privileges required")
    
    @monitor_performance
    @publish_event("session_created", "session")
    def create_session(self, data: SessionCreateSchema) -> BaseResponse[SessionResponse]:
        """创建会话"""
        try:
            # 验证业务规则
            self._validate_business_rules(data, "create")
            
            # 生成会话标题（如果未提供）
            title = data.title or self._generate_session_title(UUID(self.current_user_id))
            
            # 创建会话数据
            session_data = {
                "id": uuid4(),
                "user_id": UUID(self.current_user_id),
                "title": title,
                "description": data.description,
                "metadata": json.dumps(data.metadata) if data.metadata else None,
                "settings": json.dumps(data.settings) if data.settings else None,
                "status": "active",
                "is_persistent": data.is_persistent,
                "max_messages": data.max_messages or 1000,
                "auto_archive_days": data.auto_archive_days or 30,
                "message_count": 0,
                "last_activity_at": datetime.utcnow(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # 创建会话
            session = self.session_repository.create(session_data)
            
            # 转换为响应模型
            response_data = self._transform_to_response(session)
            
            return self._create_success_response(
                response_data,
                "Session created successfully"
            )
            
        except (ValidationException, BusinessRuleException, PermissionDeniedException):
            raise
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            raise ServiceError("Failed to create session")
    
    @monitor_performance
    def get_user_sessions(
        self, 
        user_id: Optional[UUID] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> BaseResponse[List[SessionResponse]]:
        """获取用户会话列表"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_search")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取会话列表
            sessions = self.session_repository.get_user_sessions(
                user_id=user_id,
                status=status,
                limit=limit,
                offset=offset
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(session) for session in sessions
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting user sessions: {e}")
            raise ServiceError("Failed to get user sessions")
    
    @monitor_performance
    def search_sessions(
        self, 
        search_params: SessionSearchSchema
    ) -> BaseResponse[List[SessionResponse]]:
        """搜索会话"""
        try:
            # 如果指定了用户ID，检查权限
            if search_params.user_id:
                if str(search_params.user_id) != str(self.current_user_id):
                    self._check_permission("admin_search")
            
            # 如果没有指定用户ID，默认只搜索当前用户的会话
            if not search_params.user_id:
                search_params.user_id = UUID(self.current_user_id)
            
            # 执行搜索
            sessions = self.session_repository.search_sessions(
                user_id=search_params.user_id,
                status=search_params.status,
                title_query=search_params.title_query,
                description_query=search_params.description_query,
                created_from=search_params.created_from,
                created_to=search_params.created_to,
                updated_from=search_params.updated_from,
                updated_to=search_params.updated_to,
                is_persistent=search_params.is_persistent,
                has_messages=search_params.has_messages,
                min_message_count=search_params.min_message_count,
                max_message_count=search_params.max_message_count
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(session) for session in sessions
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error searching sessions: {e}")
            raise ServiceError("Failed to search sessions")
    
    @monitor_performance
    @publish_event("session_updated", "session")
    def update_session_status(
        self, 
        session_id: UUID, 
        status: str
    ) -> BaseResponse[SessionResponse]:
        """更新会话状态"""
        try:
            # 获取会话
            session = self.session_repository.get_or_404(session_id)
            
            # 权限检查
            self._check_permission("update", session)
            
            # 验证状态
            valid_statuses = ["active", "paused", "archived", "deleted"]
            if status not in valid_statuses:
                raise ValidationException(f"Invalid status. Must be one of: {valid_statuses}")
            
            # 更新状态
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            # 如果是归档状态，设置归档时间
            if status == "archived":
                update_data["archived_at"] = datetime.utcnow()
            
            # 更新会话
            updated_session = self.session_repository.update(session_id, update_data)
            
            # 转换为响应模型
            response_data = self._transform_to_response(updated_session)
            
            return self._create_success_response(
                response_data,
                f"Session status updated to {status}"
            )
            
        except (ValidationException, PermissionDeniedException):
            raise
        except Exception as e:
            self.logger.error(f"Error updating session status {session_id}: {e}")
            raise ServiceError("Failed to update session status")
    
    @monitor_performance
    @publish_event("session_archived", "session")
    def archive_session(self, session_id: UUID) -> BaseResponse[SessionResponse]:
        """归档会话"""
        return self.update_session_status(session_id, "archived")
    
    @monitor_performance
    @publish_event("session_deleted", "session")
    def delete_session(self, session_id: UUID, soft_delete: bool = True) -> BaseResponse[None]:
        """删除会话"""
        try:
            # 获取会话
            session = self.session_repository.get_or_404(session_id)
            
            # 权限检查
            self._check_permission("delete", session)
            
            if soft_delete:
                # 软删除：更新状态为deleted
                self.session_repository.update(session_id, {
                    "status": "deleted",
                    "deleted_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                })
            else:
                # 硬删除：从数据库中删除
                self.session_repository.delete(session_id)
            
            return self._create_success_response(
                None,
                "Session deleted successfully"
            )
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting session {session_id}: {e}")
            raise ServiceError("Failed to delete session")
    
    @monitor_performance
    def get_active_sessions(
        self, 
        user_id: Optional[UUID] = None,
        limit: int = 50
    ) -> BaseResponse[List[SessionResponse]]:
        """获取活跃会话"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_search")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取活跃会话
            sessions = self.session_repository.get_active_sessions(
                user_id=user_id,
                limit=limit
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(session) for session in sessions
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting active sessions: {e}")
            raise ServiceError("Failed to get active sessions")
    
    @monitor_performance
    def update_last_activity(
        self, 
        session_id: UUID
    ) -> BaseResponse[SessionResponse]:
        """更新会话最后活动时间"""
        try:
            # 获取会话
            session = self.session_repository.get_or_404(session_id)
            
            # 权限检查
            self._check_permission("update", session)
            
            # 更新最后活动时间
            updated_session = self.session_repository.update_last_activity(session_id)
            
            # 转换为响应模型
            response_data = self._transform_to_response(updated_session)
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error updating last activity for session {session_id}: {e}")
            raise ServiceError("Failed to update last activity")
    
    @monitor_performance
    def get_session_analytics(
        self, 
        session_id: UUID
    ) -> BaseResponse[SessionAnalyticsResponse]:
        """获取会话分析"""
        try:
            # 获取会话
            session = self.session_repository.get_or_404(session_id)
            
            # 权限检查
            self._check_permission("read", session)
            
            # 获取会话统计
            stats = self.message_repository.get_session_statistics(session_id)
            
            # 计算持续时间
            duration_hours = None
            if session.last_activity_at and session.created_at:
                duration = session.last_activity_at - session.created_at
                duration_hours = duration.total_seconds() / 3600
            
            # 计算参与度评分
            engagement_data = {
                "message_count": stats.get("total_messages", 0),
                "conversation_turns": stats.get("conversation_turns", 0),
                "average_response_time": stats.get("avg_response_time", 0),
                "duration_hours": duration_hours or 0
            }
            engagement_score = self._calculate_engagement_score(engagement_data)
            
            # 构建分析响应
            analytics = SessionAnalyticsResponse(
                session_id=session_id,
                title=session.title,
                status=session.status,
                created_at=session.created_at,
                last_activity=session.last_activity_at,
                duration_hours=duration_hours,
                message_count=stats.get("total_messages", 0),
                user_message_count=stats.get("user_messages", 0),
                assistant_message_count=stats.get("assistant_messages", 0),
                system_message_count=stats.get("system_messages", 0),
                tool_message_count=stats.get("tool_messages", 0),
                average_message_length=stats.get("avg_message_length", 0.0),
                total_content_length=stats.get("total_content_length", 0),
                conversation_turns=stats.get("conversation_turns", 0),
                response_times=stats.get("response_times", []),
                average_response_time=stats.get("avg_response_time"),
                quality_metrics={
                    "average_quality_score": stats.get("avg_quality_score", 0.0),
                    "quality_variance": stats.get("quality_variance", 0.0),
                    "high_quality_ratio": stats.get("high_quality_ratio", 0.0)
                },
                engagement_score=engagement_score,
                cost_estimation={
                    "total_cost": stats.get("total_cost", 0.0),
                    "input_cost": stats.get("input_cost", 0.0),
                    "output_cost": stats.get("output_cost", 0.0)
                }
            )
            
            return self._create_success_response(analytics)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting session analytics {session_id}: {e}")
            raise ServiceError("Failed to get session analytics")
    
    @monitor_performance
    def get_session_statistics(
        self, 
        user_id: Optional[UUID] = None
    ) -> BaseResponse[SessionStatsResponse]:
        """获取会话统计信息"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_stats")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取统计数据
            stats = self.session_repository.get_user_session_statistics(user_id)
            
            # 计算时间范围
            now = datetime.utcnow()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = today_start - timedelta(days=now.weekday())
            month_start = today_start.replace(day=1)
            
            # 获取时间段统计
            sessions_today = self.session_repository.count_sessions_in_period(
                user_id, today_start, now
            )
            sessions_this_week = self.session_repository.count_sessions_in_period(
                user_id, week_start, now
            )
            sessions_this_month = self.session_repository.count_sessions_in_period(
                user_id, month_start, now
            )
            
            # 获取最活跃的会话
            most_active_sessions = self.session_repository.get_most_active_sessions(
                user_id, limit=10
            )
            
            # 获取最近的会话
            recent_sessions = self.session_repository.get_recent_sessions(
                user_id, limit=10
            )
            
            session_stats = SessionStatsResponse(
                total_sessions=stats.get("total_sessions", 0),
                sessions_by_status=stats.get("sessions_by_status", {}),
                active_sessions=stats.get("active_sessions", 0),
                archived_sessions=stats.get("archived_sessions", 0),
                deleted_sessions=stats.get("deleted_sessions", 0),
                total_messages=stats.get("total_messages", 0),
                average_messages_per_session=stats.get("avg_messages_per_session", 0.0),
                sessions_today=sessions_today,
                sessions_this_week=sessions_this_week,
                sessions_this_month=sessions_this_month,
                most_active_sessions=most_active_sessions,
                recent_sessions=recent_sessions,
                generated_at=now
            )
            
            return self._create_success_response(session_stats)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting session statistics: {e}")
            raise ServiceError("Failed to get session statistics")
    
    @monitor_performance
    def cleanup_expired_sessions(self) -> BaseResponse[Dict[str, int]]:
        """清理过期会话"""
        try:
            # 需要管理员权限
            self._check_permission("admin_search")
            
            # 清理过期会话
            cleanup_result = self.session_repository.cleanup_expired_sessions()
            
            return self._create_success_response(
                cleanup_result,
                f"Cleaned up {cleanup_result.get('total_cleaned', 0)} expired sessions"
            )
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error cleaning up expired sessions: {e}")
            raise ServiceError("Failed to cleanup expired sessions")
    
    @monitor_performance
    def bulk_update_sessions(
        self, 
        session_ids: List[UUID],
        update_data: SessionUpdateSchema
    ) -> BaseResponse[List[SessionResponse]]:
        """批量更新会话"""
        try:
            updated_sessions = []
            
            for session_id in session_ids:
                try:
                    # 获取会话并检查权限
                    session = self.session_repository.get_or_404(session_id)
                    self._check_permission("update", session)
                    
                    # 准备更新数据
                    update_dict = update_data.dict(exclude_unset=True)
                    if update_dict:
                        update_dict["updated_at"] = datetime.utcnow()
                        
                        # 处理JSON字段
                        if "metadata" in update_dict and update_dict["metadata"]:
                            update_dict["metadata"] = json.dumps(update_dict["metadata"])
                        if "settings" in update_dict and update_dict["settings"]:
                            update_dict["settings"] = json.dumps(update_dict["settings"])
                        
                        # 更新会话
                        updated_session = self.session_repository.update(session_id, update_dict)
                        updated_sessions.append(updated_session)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to update session {session_id}: {e}")
                    continue
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(session) for session in updated_sessions
            ]
            
            return self._create_success_response(
                response_data,
                f"Updated {len(response_data)} sessions"
            )
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error bulk updating sessions: {e}")
            raise ServiceError("Failed to bulk update sessions")


# 便捷函数
def create_session_service(session: Optional[DBSession] = None) -> SessionService:
    """创建会话服务实例"""
    from ..database.repositories import get_repository_manager
    
    repo_manager = get_repository_manager()
    session_repository = repo_manager.get_session_repository(session)
    user_repository = repo_manager.get_user_repository(session)
    message_repository = repo_manager.get_message_repository(session)
    
    return SessionService(session_repository, user_repository, message_repository, session)