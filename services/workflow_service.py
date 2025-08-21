"""工作流服务

提供工作流管理相关的业务逻辑，包括工作流创建、执行、监控、版本管理等功能。
"""

import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, validator
from sqlalchemy.orm import Session

from .base import BaseService, ServiceError, CacheConfig, publish_event
from ..models.database_models import Workflow, WorkflowExecution, User
from ..models.response_models import WorkflowResponse, BaseResponse
from ..database.repositories import WorkflowRepository, WorkflowExecutionRepository, UserRepository
from ..utils.validation import (
    ValidationException, BusinessRuleException, 
    PermissionDeniedException, DataValidator
)
from ..utils.performance_monitoring import monitor_performance


class WorkflowStatus(str, Enum):
    """工作流状态枚举"""
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ExecutionStatus(str, Enum):
    """执行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class WorkflowCreateSchema(BaseModel):
    """工作流创建模式"""
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    definition: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    is_public: Optional[bool] = False
    version: Optional[str] = "1.0.0"
    
    @validator('name')
    def validate_name(cls, v):
        if not DataValidator.validate_content_length(v, min_length=1, max_length=100):
            raise ValueError("Name must be between 1 and 100 characters")
        return v
    
    @validator('description')
    def validate_description(cls, v):
        if v is not None and not DataValidator.validate_content_length(v, max_length=1000):
            raise ValueError("Description exceeds maximum length")
        return v
    
    @validator('definition')
    def validate_definition(cls, v):
        # 验证工作流定义的基本结构
        required_fields = ['nodes', 'edges']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Workflow definition must contain '{field}' field")
        
        # 验证节点
        nodes = v.get('nodes', [])
        if not isinstance(nodes, list) or len(nodes) == 0:
            raise ValueError("Workflow must have at least one node")
        
        # 验证边
        edges = v.get('edges', [])
        if not isinstance(edges, list):
            raise ValueError("Edges must be a list")
        
        return v


class WorkflowUpdateSchema(BaseModel):
    """工作流更新模式"""
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    definition: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[WorkflowStatus] = None
    is_public: Optional[bool] = None
    version: Optional[str] = None


class WorkflowExecuteSchema(BaseModel):
    """工作流执行模式"""
    workflow_id: UUID
    input_data: Optional[Dict[str, Any]] = None
    config_override: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = 300
    priority: Optional[str] = "normal"  # low, normal, high, urgent
    
    @validator('priority')
    def validate_priority(cls, v):
        if v is not None:
            valid_priorities = ["low", "normal", "high", "urgent"]
            if v not in valid_priorities:
                raise ValueError(f"Priority must be one of: {valid_priorities}")
        return v
    
    @validator('timeout_seconds')
    def validate_timeout(cls, v):
        if v is not None and (v < 1 or v > 3600):  # 1秒到1小时
            raise ValueError("Timeout must be between 1 and 3600 seconds")
        return v


class WorkflowSearchSchema(BaseModel):
    """工作流搜索模式"""
    user_id: Optional[UUID] = None
    name_query: Optional[str] = None
    description_query: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[WorkflowStatus] = None
    is_public: Optional[bool] = None
    created_from: Optional[datetime] = None
    created_to: Optional[datetime] = None
    min_popularity_score: Optional[float] = None


class ExecutionSearchSchema(BaseModel):
    """执行搜索模式"""
    workflow_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    status: Optional[ExecutionStatus] = None
    priority: Optional[str] = None
    started_from: Optional[datetime] = None
    started_to: Optional[datetime] = None
    completed_from: Optional[datetime] = None
    completed_to: Optional[datetime] = None


class WorkflowStatsResponse(BaseModel):
    """工作流统计响应"""
    total_workflows: int
    workflows_by_status: Dict[str, int]
    workflows_by_category: Dict[str, int]
    public_workflows: int
    private_workflows: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time: float
    most_popular_workflows: List[Dict[str, Any]]
    recent_workflows: List[Dict[str, Any]]
    generated_at: datetime


class ExecutionStatsResponse(BaseModel):
    """执行统计响应"""
    total_executions: int
    executions_by_status: Dict[str, int]
    executions_by_priority: Dict[str, int]
    average_execution_time: float
    success_rate: float
    executions_today: int
    executions_this_week: int
    executions_this_month: int
    top_workflows: List[Dict[str, Any]]
    recent_executions: List[Dict[str, Any]]
    generated_at: datetime


class WorkflowService(BaseService[Workflow, WorkflowCreateSchema, WorkflowUpdateSchema, WorkflowResponse]):
    """工作流服务"""
    
    def __init__(
        self, 
        repository: WorkflowRepository,
        execution_repository: WorkflowExecutionRepository,
        user_repository: UserRepository,
        session: Optional[Session] = None
    ):
        cache_config = CacheConfig(
            enabled=True,
            ttl=600,  # 10分钟
            key_prefix="workflow_service",
            invalidate_on_update=True
        )
        super().__init__(repository, WorkflowResponse, cache_config, session)
        self.workflow_repository = repository
        self.execution_repository = execution_repository
        self.user_repository = user_repository
    
    def _validate_workflow_definition(self, definition: Dict[str, Any]) -> bool:
        """验证工作流定义"""
        try:
            # 检查必需字段
            required_fields = ['nodes', 'edges']
            for field in required_fields:
                if field not in definition:
                    return False
            
            # 验证节点
            nodes = definition.get('nodes', [])
            if not nodes:
                return False
            
            node_ids = set()
            for node in nodes:
                if not isinstance(node, dict):
                    return False
                if 'id' not in node or 'type' not in node:
                    return False
                node_ids.add(node['id'])
            
            # 验证边
            edges = definition.get('edges', [])
            for edge in edges:
                if not isinstance(edge, dict):
                    return False
                if 'source' not in edge or 'target' not in edge:
                    return False
                # 检查边引用的节点是否存在
                if edge['source'] not in node_ids or edge['target'] not in node_ids:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_popularity_score(self, workflow_data: Dict[str, Any]) -> float:
        """计算工作流热度评分"""
        # 基于执行次数、成功率、最近活动等计算热度
        execution_count = workflow_data.get("execution_count", 0)
        success_rate = workflow_data.get("success_rate", 0.0)
        days_since_created = workflow_data.get("days_since_created", 0)
        days_since_last_execution = workflow_data.get("days_since_last_execution", 999)
        
        # 归一化各项指标
        execution_score = min(execution_count / 100, 1.0)  # 100次执行为满分
        success_score = success_rate
        recency_score = max(0, 1.0 - (days_since_last_execution / 30))  # 30天内有执行为满分
        age_penalty = max(0.1, 1.0 - (days_since_created / 365))  # 年龄惩罚
        
        # 加权平均
        popularity_score = (
            execution_score * 0.4 +
            success_score * 0.3 +
            recency_score * 0.2 +
            age_penalty * 0.1
        )
        
        return round(popularity_score, 3)
    
    def _validate_business_rules(self, data: Any, action: str = "create"):
        """验证业务规则"""
        if action == "create":
            # 检查工作流名称是否重复（同一用户）
            if hasattr(data, 'name'):
                existing_workflow = self.workflow_repository.get_by_name(
                    data.name, UUID(self.current_user_id)
                )
                if existing_workflow:
                    raise BusinessRuleException(f"Workflow with name '{data.name}' already exists")
            
            # 验证工作流定义
            if hasattr(data, 'definition'):
                if not self._validate_workflow_definition(data.definition):
                    raise BusinessRuleException("Invalid workflow definition")
    
    def _check_permission(self, action: str, resource: Any = None):
        """检查权限"""
        if not self.current_user_id:
            raise PermissionDeniedException("Authentication required")
        
        if action in ["create", "update", "delete", "execute"] and resource:
            # 检查工作流是否属于当前用户或是公开的
            if hasattr(resource, 'user_id'):
                if str(resource.user_id) != str(self.current_user_id):
                    # 如果不是自己的工作流，检查是否是公开的
                    if action in ["update", "delete"] or not getattr(resource, 'is_public', False):
                        # 检查是否是管理员
                        if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                            raise PermissionDeniedException("Insufficient permissions")
        
        if action in ["admin_stats", "admin_search"]:
            # 需要管理员权限
            if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                raise PermissionDeniedException("Admin privileges required")
    
    @monitor_performance
    @publish_event("workflow_created", "workflow")
    def create_workflow(self, data: WorkflowCreateSchema) -> BaseResponse[WorkflowResponse]:
        """创建工作流"""
        try:
            # 验证业务规则
            self._validate_business_rules(data, "create")
            
            # 创建工作流数据
            workflow_data = {
                "id": uuid4(),
                "user_id": UUID(self.current_user_id),
                "name": data.name,
                "description": data.description,
                "category": data.category,
                "tags": json.dumps(data.tags) if data.tags else None,
                "definition": json.dumps(data.definition),
                "config": json.dumps(data.config) if data.config else None,
                "status": WorkflowStatus.DRAFT,
                "is_public": data.is_public or False,
                "version": data.version or "1.0.0",
                "execution_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "popularity_score": 0.0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # 创建工作流
            workflow = self.workflow_repository.create(workflow_data)
            
            # 转换为响应模型
            response_data = self._transform_to_response(workflow)
            
            return self._create_success_response(
                response_data,
                "Workflow created successfully"
            )
            
        except (ValidationException, BusinessRuleException, PermissionDeniedException):
            raise
        except Exception as e:
            self.logger.error(f"Error creating workflow: {e}")
            raise ServiceError("Failed to create workflow")
    
    @monitor_performance
    @publish_event("workflow_published", "workflow")
    def publish_workflow(self, workflow_id: UUID) -> BaseResponse[WorkflowResponse]:
        """发布工作流"""
        try:
            # 获取工作流
            workflow = self.workflow_repository.get_or_404(workflow_id)
            
            # 权限检查
            self._check_permission("update", workflow)
            
            # 检查当前状态
            if workflow.status != WorkflowStatus.DRAFT:
                raise BusinessRuleException("Only draft workflows can be published")
            
            # 验证工作流定义
            definition = json.loads(workflow.definition) if workflow.definition else {}
            if not self._validate_workflow_definition(definition):
                raise BusinessRuleException("Cannot publish workflow with invalid definition")
            
            # 更新状态
            updated_workflow = self.workflow_repository.update(workflow_id, {
                "status": WorkflowStatus.PUBLISHED,
                "published_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            # 转换为响应模型
            response_data = self._transform_to_response(updated_workflow)
            
            return self._create_success_response(
                response_data,
                "Workflow published successfully"
            )
            
        except (ValidationException, BusinessRuleException, PermissionDeniedException):
            raise
        except Exception as e:
            self.logger.error(f"Error publishing workflow {workflow_id}: {e}")
            raise ServiceError("Failed to publish workflow")
    
    @monitor_performance
    def search_workflows(
        self, 
        search_params: WorkflowSearchSchema
    ) -> BaseResponse[List[WorkflowResponse]]:
        """搜索工作流"""
        try:
            # 如果指定了用户ID，检查权限
            if search_params.user_id:
                if str(search_params.user_id) != str(self.current_user_id):
                    self._check_permission("admin_search")
            
            # 执行搜索
            workflows = self.workflow_repository.search_workflows(
                user_id=search_params.user_id,
                name_query=search_params.name_query,
                description_query=search_params.description_query,
                category=search_params.category,
                tags=search_params.tags,
                status=search_params.status,
                is_public=search_params.is_public,
                created_from=search_params.created_from,
                created_to=search_params.created_to,
                min_popularity_score=search_params.min_popularity_score,
                current_user_id=UUID(self.current_user_id) if self.current_user_id else None
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(workflow) for workflow in workflows
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error searching workflows: {e}")
            raise ServiceError("Failed to search workflows")
    
    @monitor_performance
    def get_public_workflows(
        self, 
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> BaseResponse[List[WorkflowResponse]]:
        """获取公开工作流"""
        try:
            # 获取公开工作流
            workflows = self.workflow_repository.get_published_workflows(
                category=category,
                limit=limit,
                offset=offset
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(workflow) for workflow in workflows
            ]
            
            return self._create_success_response(response_data)
            
        except Exception as e:
            self.logger.error(f"Error getting public workflows: {e}")
            raise ServiceError("Failed to get public workflows")
    
    @monitor_performance
    def get_user_workflows(
        self, 
        user_id: Optional[UUID] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> BaseResponse[List[WorkflowResponse]]:
        """获取用户工作流"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_search")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取用户工作流
            workflows = self.workflow_repository.get_user_workflows(
                user_id=user_id,
                status=status,
                limit=limit,
                offset=offset
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(workflow) for workflow in workflows
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting user workflows: {e}")
            raise ServiceError("Failed to get user workflows")
    
    @monitor_performance
    @publish_event("workflow_executed", "workflow_execution")
    def execute_workflow(
        self, 
        data: WorkflowExecuteSchema
    ) -> BaseResponse[Dict[str, Any]]:
        """执行工作流"""
        try:
            # 获取工作流
            workflow = self.workflow_repository.get_or_404(data.workflow_id)
            
            # 权限检查
            self._check_permission("execute", workflow)
            
            # 检查工作流状态
            if workflow.status not in [WorkflowStatus.PUBLISHED]:
                raise BusinessRuleException("Only published workflows can be executed")
            
            # 创建执行记录
            execution_data = {
                "id": uuid4(),
                "workflow_id": data.workflow_id,
                "user_id": UUID(self.current_user_id),
                "input_data": json.dumps(data.input_data) if data.input_data else None,
                "config": json.dumps(data.config_override) if data.config_override else None,
                "status": ExecutionStatus.PENDING,
                "priority": data.priority or "normal",
                "timeout_seconds": data.timeout_seconds or 300,
                "started_at": datetime.utcnow(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # 创建执行记录
            execution = self.execution_repository.create(execution_data)
            
            # 更新工作流执行计数
            self.workflow_repository.increment_execution_count(data.workflow_id)
            
            # TODO: 这里应该启动实际的工作流执行引擎
            # 目前返回执行ID，实际执行将在后台进行
            
            result = {
                "execution_id": str(execution.id),
                "workflow_id": str(data.workflow_id),
                "status": execution.status,
                "started_at": execution.started_at.isoformat(),
                "message": "Workflow execution started"
            }
            
            return self._create_success_response(
                result,
                "Workflow execution started successfully"
            )
            
        except (ValidationException, BusinessRuleException, PermissionDeniedException):
            raise
        except Exception as e:
            self.logger.error(f"Error executing workflow {data.workflow_id}: {e}")
            raise ServiceError("Failed to execute workflow")
    
    @monitor_performance
    def get_execution_status(
        self, 
        execution_id: UUID
    ) -> BaseResponse[Dict[str, Any]]:
        """获取执行状态"""
        try:
            # 获取执行记录
            execution = self.execution_repository.get_or_404(execution_id)
            
            # 权限检查
            if str(execution.user_id) != str(self.current_user_id):
                if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                    raise PermissionDeniedException("Can only access your own executions")
            
            # 构建状态响应
            status_data = {
                "execution_id": str(execution.id),
                "workflow_id": str(execution.workflow_id),
                "status": execution.status,
                "priority": execution.priority,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_seconds": execution.duration_seconds,
                "progress_percentage": execution.progress_percentage,
                "current_step": execution.current_step,
                "total_steps": execution.total_steps,
                "error_message": execution.error_message,
                "result_data": json.loads(execution.result_data) if execution.result_data else None
            }
            
            return self._create_success_response(status_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting execution status {execution_id}: {e}")
            raise ServiceError("Failed to get execution status")
    
    @monitor_performance
    def cancel_execution(
        self, 
        execution_id: UUID
    ) -> BaseResponse[Dict[str, Any]]:
        """取消执行"""
        try:
            # 获取执行记录
            execution = self.execution_repository.get_or_404(execution_id)
            
            # 权限检查
            if str(execution.user_id) != str(self.current_user_id):
                if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                    raise PermissionDeniedException("Can only cancel your own executions")
            
            # 检查执行状态
            if execution.status not in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
                raise BusinessRuleException("Can only cancel pending or running executions")
            
            # 更新执行状态
            updated_execution = self.execution_repository.update(execution_id, {
                "status": ExecutionStatus.CANCELLED,
                "completed_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            # TODO: 这里应该通知执行引擎取消执行
            
            result = {
                "execution_id": str(execution_id),
                "status": updated_execution.status,
                "cancelled_at": updated_execution.completed_at.isoformat(),
                "message": "Execution cancelled successfully"
            }
            
            return self._create_success_response(
                result,
                "Execution cancelled successfully"
            )
            
        except (BusinessRuleException, PermissionDeniedException):
            raise
        except Exception as e:
            self.logger.error(f"Error cancelling execution {execution_id}: {e}")
            raise ServiceError("Failed to cancel execution")
    
    @monitor_performance
    def get_workflow_statistics(
        self, 
        user_id: Optional[UUID] = None
    ) -> BaseResponse[WorkflowStatsResponse]:
        """获取工作流统计信息"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_stats")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取统计数据
            stats = self.workflow_repository.get_user_workflow_statistics(user_id)
            
            # 获取最受欢迎的工作流
            popular_workflows = self.workflow_repository.get_popular_workflows(
                user_id, limit=10
            )
            
            # 获取最近的工作流
            recent_workflows = self.workflow_repository.get_recent_workflows(
                user_id, limit=10
            )
            
            workflow_stats = WorkflowStatsResponse(
                total_workflows=stats.get("total_workflows", 0),
                workflows_by_status=stats.get("workflows_by_status", {}),
                workflows_by_category=stats.get("workflows_by_category", {}),
                public_workflows=stats.get("public_workflows", 0),
                private_workflows=stats.get("private_workflows", 0),
                total_executions=stats.get("total_executions", 0),
                successful_executions=stats.get("successful_executions", 0),
                failed_executions=stats.get("failed_executions", 0),
                average_execution_time=stats.get("avg_execution_time", 0.0),
                most_popular_workflows=popular_workflows,
                recent_workflows=recent_workflows,
                generated_at=datetime.utcnow()
            )
            
            return self._create_success_response(workflow_stats)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting workflow statistics: {e}")
            raise ServiceError("Failed to get workflow statistics")
    
    @monitor_performance
    def get_execution_statistics(
        self, 
        user_id: Optional[UUID] = None,
        workflow_id: Optional[UUID] = None
    ) -> BaseResponse[ExecutionStatsResponse]:
        """获取执行统计信息"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_stats")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取统计数据
            stats = self.execution_repository.get_execution_statistics(
                user_id=user_id,
                workflow_id=workflow_id
            )
            
            # 计算时间范围
            now = datetime.utcnow()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = today_start - timedelta(days=now.weekday())
            month_start = today_start.replace(day=1)
            
            # 获取时间段统计
            executions_today = self.execution_repository.count_executions_in_period(
                user_id, today_start, now, workflow_id
            )
            executions_this_week = self.execution_repository.count_executions_in_period(
                user_id, week_start, now, workflow_id
            )
            executions_this_month = self.execution_repository.count_executions_in_period(
                user_id, month_start, now, workflow_id
            )
            
            # 获取热门工作流
            top_workflows = self.execution_repository.get_top_workflows_by_execution_count(
                user_id, limit=10
            )
            
            # 获取最近的执行
            recent_executions = self.execution_repository.get_recent_executions(
                user_id, limit=10, workflow_id=workflow_id
            )
            
            execution_stats = ExecutionStatsResponse(
                total_executions=stats.get("total_executions", 0),
                executions_by_status=stats.get("executions_by_status", {}),
                executions_by_priority=stats.get("executions_by_priority", {}),
                average_execution_time=stats.get("avg_execution_time", 0.0),
                success_rate=stats.get("success_rate", 0.0),
                executions_today=executions_today,
                executions_this_week=executions_this_week,
                executions_this_month=executions_this_month,
                top_workflows=top_workflows,
                recent_executions=recent_executions,
                generated_at=now
            )
            
            return self._create_success_response(execution_stats)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting execution statistics: {e}")
            raise ServiceError("Failed to get execution statistics")
    
    @monitor_performance
    def update_popularity_scores(self) -> BaseResponse[Dict[str, int]]:
        """更新工作流热度评分"""
        try:
            # 需要管理员权限
            self._check_permission("admin_stats")
            
            # 获取所有工作流的统计数据
            workflows_stats = self.workflow_repository.get_all_workflows_statistics()
            
            updated_count = 0
            for workflow_stat in workflows_stats:
                # 计算热度评分
                popularity_score = self._calculate_popularity_score(workflow_stat)
                
                # 更新热度评分
                self.workflow_repository.update_popularity_score(
                    workflow_stat["id"], popularity_score
                )
                updated_count += 1
            
            result = {
                "updated_workflows": updated_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return self._create_success_response(
                result,
                f"Updated popularity scores for {updated_count} workflows"
            )
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error updating popularity scores: {e}")
            raise ServiceError("Failed to update popularity scores")


# 便捷函数
def create_workflow_service(session: Optional[Session] = None) -> WorkflowService:
    """创建工作流服务实例"""
    from ..database.repositories import get_repository_manager
    
    repo_manager = get_repository_manager()
    workflow_repository = repo_manager.get_workflow_repository(session)
    execution_repository = repo_manager.get_workflow_execution_repository(session)
    user_repository = repo_manager.get_user_repository(session)
    
    return WorkflowService(workflow_repository, execution_repository, user_repository, session)