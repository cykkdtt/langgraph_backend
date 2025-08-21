"""消息服务

提供消息管理相关的业务逻辑，包括消息创建、查询、搜索、统计等功能。
"""

import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from uuid import UUID

from pydantic import BaseModel, validator
from sqlalchemy.orm import Session

from .base import BaseService, ServiceError, CacheConfig, publish_event
from ..models.database_models import Message, User, Session as ChatSession
from ..models.response_models import MessageResponse, BaseResponse
from ..database.repositories import MessageRepository, SessionRepository
from ..utils.validation import (
    ValidationException, BusinessRuleException, 
    PermissionDeniedException, DataValidator
)
from ..utils.performance_monitoring import monitor_performance


class MessageCreateSchema(BaseModel):
    """消息创建模式"""
    session_id: UUID
    role: str  # user, assistant, system, tool
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    parent_message_id: Optional[UUID] = None
    priority: Optional[str] = "normal"  # low, normal, high, urgent
    
    @validator('role')
    def validate_role(cls, v):
        valid_roles = ["user", "assistant", "system", "tool"]
        if v not in valid_roles:
            raise ValueError(f"Role must be one of: {valid_roles}")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        if v is not None:
            valid_priorities = ["low", "normal", "high", "urgent"]
            if v not in valid_priorities:
                raise ValueError(f"Priority must be one of: {valid_priorities}")
        return v
    
    @validator('content')
    def validate_content(cls, v):
        if not DataValidator.validate_content_length(v, max_length=50000):
            raise ValueError("Content exceeds maximum length")
        return v


class MessageUpdateSchema(BaseModel):
    """消息更新模式"""
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[str] = None  # pending, processing, completed, failed, cancelled
    priority: Optional[str] = None
    quality_score: Optional[float] = None
    
    @validator('status')
    def validate_status(cls, v):
        if v is not None:
            valid_statuses = ["pending", "processing", "completed", "failed", "cancelled"]
            if v not in valid_statuses:
                raise ValueError(f"Status must be one of: {valid_statuses}")
        return v
    
    @validator('quality_score')
    def validate_quality_score(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Quality score must be between 0.0 and 1.0")
        return v


class MessageSearchSchema(BaseModel):
    """消息搜索模式"""
    session_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    role: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    content_query: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_quality_score: Optional[float] = None
    has_tool_calls: Optional[bool] = None
    parent_message_id: Optional[UUID] = None


class ConversationHistorySchema(BaseModel):
    """对话历史模式"""
    session_id: UUID
    limit: Optional[int] = 50
    before_message_id: Optional[UUID] = None
    after_message_id: Optional[UUID] = None
    include_system: Optional[bool] = True
    include_tool_calls: Optional[bool] = True


class MessageStatsResponse(BaseModel):
    """消息统计响应"""
    total_messages: int
    messages_by_role: Dict[str, int]
    messages_by_status: Dict[str, int]
    messages_by_priority: Dict[str, int]
    average_quality_score: Optional[float]
    total_content_length: int
    messages_today: int
    messages_this_week: int
    messages_this_month: int
    top_sessions: List[Dict[str, Any]]
    generated_at: datetime


class MessageAnalyticsResponse(BaseModel):
    """消息分析响应"""
    session_id: UUID
    total_messages: int
    user_messages: int
    assistant_messages: int
    system_messages: int
    tool_messages: int
    average_response_time: Optional[float]
    conversation_duration: Optional[float]
    quality_metrics: Dict[str, float]
    token_usage: Dict[str, int]
    cost_estimation: Dict[str, float]


class MessageService(BaseService[Message, MessageCreateSchema, MessageUpdateSchema, MessageResponse]):
    """消息服务"""
    
    def __init__(
        self, 
        repository: MessageRepository,
        session_repository: SessionRepository,
        session: Optional[Session] = None
    ):
        cache_config = CacheConfig(
            enabled=True,
            ttl=300,  # 5分钟
            key_prefix="message_service",
            invalidate_on_update=True
        )
        super().__init__(repository, MessageResponse, cache_config, session)
        self.message_repository = repository
        self.session_repository = session_repository
    
    def _calculate_content_metrics(self, content: str) -> Dict[str, Any]:
        """计算内容指标"""
        return {
            "character_count": len(content),
            "word_count": len(content.split()),
            "line_count": content.count('\n') + 1,
            "estimated_tokens": len(content) // 4  # 粗略估算
        }
    
    def _estimate_cost(self, content: str, role: str) -> float:
        """估算消息成本"""
        # 基于内容长度和角色的简单成本估算
        base_cost_per_1k_tokens = {
            "user": 0.001,
            "assistant": 0.002,
            "system": 0.0005,
            "tool": 0.001
        }
        
        estimated_tokens = len(content) // 4
        cost_per_token = base_cost_per_1k_tokens.get(role, 0.001) / 1000
        
        return estimated_tokens * cost_per_token
    
    def _validate_business_rules(self, data: Any, action: str = "create"):
        """验证业务规则"""
        if action == "create":
            # 检查会话是否存在
            session = self.session_repository.get(data.session_id)
            if not session:
                raise BusinessRuleException("Session not found")
            
            # 检查会话是否属于当前用户
            if self.current_user_id and str(session.user_id) != str(self.current_user_id):
                raise PermissionDeniedException("Cannot create message in another user's session")
            
            # 检查父消息是否存在且属于同一会话
            if data.parent_message_id:
                parent_message = self.message_repository.get(data.parent_message_id)
                if not parent_message:
                    raise BusinessRuleException("Parent message not found")
                if str(parent_message.session_id) != str(data.session_id):
                    raise BusinessRuleException("Parent message must be in the same session")
    
    def _check_permission(self, action: str, resource: Any = None):
        """检查权限"""
        if not self.current_user_id:
            raise PermissionDeniedException("Authentication required")
        
        if action in ["create", "update", "delete"] and resource:
            # 检查消息是否属于当前用户的会话
            if hasattr(resource, 'session'):
                session = resource.session
            else:
                session = self.session_repository.get(resource.session_id)
            
            if session and str(session.user_id) != str(self.current_user_id):
                # 检查是否是管理员
                if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                    raise PermissionDeniedException("Can only access your own messages")
        
        if action in ["admin_stats", "admin_search"]:
            # 需要管理员权限
            if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                raise PermissionDeniedException("Admin privileges required")
    
    @monitor_performance
    @publish_event("message_created", "message")
    def create_message(self, data: MessageCreateSchema) -> BaseResponse[MessageResponse]:
        """创建消息"""
        try:
            # 验证业务规则
            self._validate_business_rules(data, "create")
            
            # 计算内容指标
            content_metrics = self._calculate_content_metrics(data.content)
            
            # 估算成本
            estimated_cost = self._estimate_cost(data.content, data.role)
            
            # 创建消息数据
            message_data = {
                "session_id": data.session_id,
                "role": data.role,
                "content": data.content,
                "tool_calls": json.dumps(data.tool_calls) if data.tool_calls else None,
                "metadata": json.dumps(data.metadata) if data.metadata else None,
                "parent_message_id": data.parent_message_id,
                "status": "pending",
                "priority": data.priority or "normal",
                "content_length": content_metrics["character_count"],
                "estimated_cost": estimated_cost,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # 创建消息
            message = self.message_repository.create(message_data)
            
            # 转换为响应模型
            response_data = self._transform_to_response(message)
            
            return self._create_success_response(
                response_data,
                "Message created successfully"
            )
            
        except (ValidationException, BusinessRuleException, PermissionDeniedException):
            raise
        except Exception as e:
            self.logger.error(f"Error creating message: {e}")
            raise ServiceError("Failed to create message")
    
    @monitor_performance
    def get_conversation_history(
        self, 
        data: ConversationHistorySchema
    ) -> BaseResponse[List[MessageResponse]]:
        """获取对话历史"""
        try:
            # 检查会话权限
            session = self.session_repository.get_or_404(data.session_id)
            self._check_permission("read", session)
            
            # 获取对话历史
            messages = self.message_repository.get_conversation_history(
                session_id=data.session_id,
                limit=data.limit,
                before_message_id=data.before_message_id,
                after_message_id=data.after_message_id,
                include_system=data.include_system,
                include_tool_calls=data.include_tool_calls
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(message) for message in messages
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {e}")
            raise ServiceError("Failed to get conversation history")
    
    @monitor_performance
    def search_messages(
        self, 
        search_params: MessageSearchSchema
    ) -> BaseResponse[List[MessageResponse]]:
        """搜索消息"""
        try:
            # 如果指定了用户ID，检查权限
            if search_params.user_id:
                if str(search_params.user_id) != str(self.current_user_id):
                    self._check_permission("admin_search")
            
            # 如果指定了会话ID，检查权限
            if search_params.session_id:
                session = self.session_repository.get_or_404(search_params.session_id)
                self._check_permission("read", session)
            
            # 如果没有指定用户ID或会话ID，默认只搜索当前用户的消息
            if not search_params.user_id and not search_params.session_id:
                search_params.user_id = UUID(self.current_user_id)
            
            # 执行搜索
            messages = self.message_repository.search_messages(
                user_id=search_params.user_id,
                session_id=search_params.session_id,
                role=search_params.role,
                status=search_params.status,
                priority=search_params.priority,
                content_query=search_params.content_query,
                date_from=search_params.date_from,
                date_to=search_params.date_to,
                min_quality_score=search_params.min_quality_score,
                has_tool_calls=search_params.has_tool_calls,
                parent_message_id=search_params.parent_message_id
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(message) for message in messages
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error searching messages: {e}")
            raise ServiceError("Failed to search messages")
    
    @monitor_performance
    def update_message_status(
        self, 
        message_id: UUID, 
        status: str,
        quality_score: Optional[float] = None
    ) -> BaseResponse[MessageResponse]:
        """更新消息状态"""
        try:
            # 获取消息
            message = self.message_repository.get_or_404(message_id)
            
            # 权限检查
            self._check_permission("update", message)
            
            # 验证状态
            valid_statuses = ["pending", "processing", "completed", "failed", "cancelled"]
            if status not in valid_statuses:
                raise ValidationException(f"Invalid status. Must be one of: {valid_statuses}")
            
            # 更新数据
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if quality_score is not None:
                if quality_score < 0.0 or quality_score > 1.0:
                    raise ValidationException("Quality score must be between 0.0 and 1.0")
                update_data["quality_score"] = quality_score
            
            # 更新消息
            updated_message = self.message_repository.update(message_id, update_data)
            
            # 转换为响应模型
            response_data = self._transform_to_response(updated_message)
            
            return self._create_success_response(
                response_data,
                "Message status updated successfully"
            )
            
        except (ValidationException, PermissionDeniedException):
            raise
        except Exception as e:
            self.logger.error(f"Error updating message status {message_id}: {e}")
            raise ServiceError("Failed to update message status")
    
    @monitor_performance
    def get_messages_by_status(
        self, 
        status: str,
        user_id: Optional[UUID] = None,
        limit: int = 100
    ) -> BaseResponse[List[MessageResponse]]:
        """根据状态获取消息"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_search")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取消息
            messages = self.message_repository.get_messages_by_status(
                status=status,
                user_id=user_id,
                limit=limit
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(message) for message in messages
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting messages by status {status}: {e}")
            raise ServiceError(f"Failed to get messages by status {status}")
    
    @monitor_performance
    def get_messages_by_priority(
        self, 
        priority: str,
        user_id: Optional[UUID] = None,
        limit: int = 100
    ) -> BaseResponse[List[MessageResponse]]:
        """根据优先级获取消息"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_search")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取消息
            messages = self.message_repository.get_messages_by_priority(
                priority=priority,
                user_id=user_id,
                limit=limit
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(message) for message in messages
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting messages by priority {priority}: {e}")
            raise ServiceError(f"Failed to get messages by priority {priority}")
    
    @monitor_performance
    def get_session_analytics(
        self, 
        session_id: UUID
    ) -> BaseResponse[MessageAnalyticsResponse]:
        """获取会话分析"""
        try:
            # 检查会话权限
            session = self.session_repository.get_or_404(session_id)
            self._check_permission("read", session)
            
            # 获取会话统计
            stats = self.message_repository.get_session_statistics(session_id)
            
            # 计算分析数据
            analytics = MessageAnalyticsResponse(
                session_id=session_id,
                total_messages=stats.get("total_messages", 0),
                user_messages=stats.get("user_messages", 0),
                assistant_messages=stats.get("assistant_messages", 0),
                system_messages=stats.get("system_messages", 0),
                tool_messages=stats.get("tool_messages", 0),
                average_response_time=stats.get("avg_response_time"),
                conversation_duration=stats.get("conversation_duration"),
                quality_metrics={
                    "average_quality_score": stats.get("avg_quality_score", 0.0),
                    "quality_variance": stats.get("quality_variance", 0.0)
                },
                token_usage={
                    "total_tokens": stats.get("total_tokens", 0),
                    "input_tokens": stats.get("input_tokens", 0),
                    "output_tokens": stats.get("output_tokens", 0)
                },
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
    def get_message_statistics(
        self, 
        user_id: Optional[UUID] = None
    ) -> BaseResponse[MessageStatsResponse]:
        """获取消息统计信息"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_stats")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取统计数据
            stats = self.message_repository.get_user_message_statistics(user_id)
            
            # 计算时间范围
            now = datetime.utcnow()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = today_start - timedelta(days=now.weekday())
            month_start = today_start.replace(day=1)
            
            # 获取时间段统计
            messages_today = self.message_repository.count_messages_in_period(
                user_id, today_start, now
            )
            messages_this_week = self.message_repository.count_messages_in_period(
                user_id, week_start, now
            )
            messages_this_month = self.message_repository.count_messages_in_period(
                user_id, month_start, now
            )
            
            # 获取热门会话
            top_sessions = self.message_repository.get_top_sessions_by_message_count(
                user_id, limit=10
            )
            
            message_stats = MessageStatsResponse(
                total_messages=stats.get("total_messages", 0),
                messages_by_role=stats.get("messages_by_role", {}),
                messages_by_status=stats.get("messages_by_status", {}),
                messages_by_priority=stats.get("messages_by_priority", {}),
                average_quality_score=stats.get("average_quality_score"),
                total_content_length=stats.get("total_content_length", 0),
                messages_today=messages_today,
                messages_this_week=messages_this_week,
                messages_this_month=messages_this_month,
                top_sessions=top_sessions,
                generated_at=now
            )
            
            return self._create_success_response(message_stats)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting message statistics: {e}")
            raise ServiceError("Failed to get message statistics")
    
    @monitor_performance
    def update_quality_scores(
        self, 
        session_id: UUID,
        quality_updates: List[Dict[str, Any]]
    ) -> BaseResponse[List[MessageResponse]]:
        """批量更新消息质量评分"""
        try:
            # 检查会话权限
            session = self.session_repository.get_or_404(session_id)
            self._check_permission("update", session)
            
            updated_messages = []
            
            for update in quality_updates:
                message_id = update.get("message_id")
                quality_score = update.get("quality_score")
                
                if not message_id or quality_score is None:
                    continue
                
                # 验证质量评分
                if quality_score < 0.0 or quality_score > 1.0:
                    continue
                
                # 更新消息
                try:
                    updated_message = self.message_repository.update_quality_score(
                        message_id, quality_score
                    )
                    if updated_message:
                        updated_messages.append(updated_message)
                except Exception as e:
                    self.logger.warning(f"Failed to update quality score for message {message_id}: {e}")
                    continue
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(message) for message in updated_messages
            ]
            
            return self._create_success_response(
                response_data,
                f"Updated quality scores for {len(response_data)} messages"
            )
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error updating quality scores: {e}")
            raise ServiceError("Failed to update quality scores")


# 便捷函数
def create_message_service(session: Optional[Session] = None) -> MessageService:
    """创建消息服务实例"""
    from ..database.repositories import get_repository_manager
    
    repo_manager = get_repository_manager()
    message_repository = repo_manager.get_message_repository(session)
    session_repository = repo_manager.get_session_repository(session)
    
    return MessageService(message_repository, session_repository, session)