"""API控制器

定义各种API端点的控制器，处理HTTP请求和响应。
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ..models.response_models import (
    BaseResponse, UserResponse, SessionResponse, MessageResponse, 
    WorkflowResponse, MemoryResponse, PaginatedResponse
)
from ..services.user_service import (
    UserService, UserCreateSchema, UserUpdateSchema, 
    PasswordChangeSchema, UserLoginSchema, UserStatsResponse
)
from ..services.session_service import (
    SessionService, SessionCreateSchema, SessionUpdateSchema,
    SessionSearchSchema, SessionStatsResponse, SessionAnalyticsResponse
)
from ..services.message_service import (
    MessageService, MessageCreateSchema, MessageUpdateSchema,
    MessageSearchSchema, ConversationHistorySchema, 
    MessageStatsResponse, MessageAnalyticsResponse
)
from ..services.workflow_service import (
    WorkflowService, WorkflowCreateSchema, WorkflowUpdateSchema,
    WorkflowExecuteSchema, WorkflowSearchSchema, ExecutionSearchSchema,
    WorkflowStatsResponse, ExecutionStatsResponse
)
from ..services.memory_service import (
    MemoryService, MemoryCreateSchema, MemoryUpdateSchema,
    MemorySearchSchema, MemoryRetrievalSchema, MemoryConsolidationSchema,
    MemoryStatsResponse, MemoryAnalyticsResponse, MemoryType
)
from ..database.connection import get_db_session
from ..utils.validation import ValidationException, BusinessRuleException, PermissionDeniedException
from ..utils.performance_monitoring import monitor_performance
from ..config.settings import get_settings

# 安全认证
security = HTTPBearer()
settings = get_settings()


class BaseController:
    """基础控制器类"""
    
    def __init__(self):
        self.router = APIRouter()
    
    def _handle_service_error(self, error: Exception) -> HTTPException:
        """处理服务层错误"""
        if isinstance(error, ValidationException):
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error)
            )
        elif isinstance(error, BusinessRuleException):
            return HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error)
            )
        elif isinstance(error, PermissionDeniedException):
            return HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(error)
            )
        else:
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    def _get_current_user_id(self, credentials: HTTPAuthorizationCredentials) -> str:
        """从JWT令牌获取当前用户ID"""
        # TODO: 实现JWT令牌解析和验证
        # 这里应该解析JWT令牌并返回用户ID
        # 暂时返回一个示例用户ID
        return "user_123"


class UserController(BaseController):
    """用户控制器"""
    
    def __init__(self):
        super().__init__()
        self.router.prefix = "/users"
        self.router.tags = ["users"]
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.post("/register", response_model=BaseResponse[UserResponse])
        @monitor_performance
        async def register_user(
            user_data: UserCreateSchema,
            db: Session = Depends(get_db_session)
        ):
            """用户注册"""
            try:
                from ..services.user_service import create_user_service
                user_service = create_user_service(db)
                result = user_service.register_user(user_data)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.post("/login", response_model=BaseResponse[Dict[str, Any]])
        @monitor_performance
        async def login_user(
            login_data: UserLoginSchema,
            db: Session = Depends(get_db_session)
        ):
            """用户登录"""
            try:
                from ..services.user_service import create_user_service
                user_service = create_user_service(db)
                result = user_service.authenticate_user(login_data)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/profile", response_model=BaseResponse[UserResponse])
        @monitor_performance
        async def get_user_profile(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取用户资料"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.user_service import create_user_service
                user_service = create_user_service(db)
                user_service.current_user_id = user_id
                result = user_service.get_by_id(UUID(user_id))
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.put("/profile", response_model=BaseResponse[UserResponse])
        @monitor_performance
        async def update_user_profile(
            user_data: UserUpdateSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """更新用户资料"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.user_service import create_user_service
                user_service = create_user_service(db)
                user_service.current_user_id = user_id
                result = user_service.update(UUID(user_id), user_data)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.post("/change-password", response_model=BaseResponse[Dict[str, str]])
        @monitor_performance
        async def change_password(
            password_data: PasswordChangeSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """修改密码"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.user_service import create_user_service
                user_service = create_user_service(db)
                user_service.current_user_id = user_id
                result = user_service.change_password(UUID(user_id), password_data)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.post("/api-keys", response_model=BaseResponse[Dict[str, str]])
        @monitor_performance
        async def generate_api_key(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """生成API密钥"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.user_service import create_user_service
                user_service = create_user_service(db)
                user_service.current_user_id = user_id
                result = user_service.generate_api_key(UUID(user_id))
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/stats", response_model=BaseResponse[UserStatsResponse])
        @monitor_performance
        async def get_user_stats(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取用户统计信息"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.user_service import create_user_service
                user_service = create_user_service(db)
                user_service.current_user_id = user_id
                result = user_service.get_user_statistics(UUID(user_id))
                return result
            except Exception as e:
                raise self._handle_service_error(e)


class SessionController(BaseController):
    """会话控制器"""
    
    def __init__(self):
        super().__init__()
        self.router.prefix = "/sessions"
        self.router.tags = ["sessions"]
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.post("/", response_model=BaseResponse[SessionResponse])
        @monitor_performance
        async def create_session(
            session_data: SessionCreateSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """创建会话"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.session_service import create_session_service
                session_service = create_session_service(db)
                session_service.current_user_id = user_id
                result = session_service.create_session(session_data)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/", response_model=BaseResponse[List[SessionResponse]])
        @monitor_performance
        async def get_user_sessions(
            limit: int = Query(50, ge=1, le=100),
            offset: int = Query(0, ge=0),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取用户会话列表"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.session_service import create_session_service
                session_service = create_session_service(db)
                session_service.current_user_id = user_id
                result = session_service.get_user_sessions(UUID(user_id), limit, offset)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/{session_id}", response_model=BaseResponse[SessionResponse])
        @monitor_performance
        async def get_session(
            session_id: UUID = Path(...),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取会话详情"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.session_service import create_session_service
                session_service = create_session_service(db)
                session_service.current_user_id = user_id
                result = session_service.get_by_id(session_id)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.put("/{session_id}", response_model=BaseResponse[SessionResponse])
        @monitor_performance
        async def update_session(
            session_id: UUID = Path(...),
            session_data: SessionUpdateSchema = Body(...),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """更新会话"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.session_service import create_session_service
                session_service = create_session_service(db)
                session_service.current_user_id = user_id
                result = session_service.update(session_id, session_data)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.delete("/{session_id}", response_model=BaseResponse[Dict[str, str]])
        @monitor_performance
        async def delete_session(
            session_id: UUID = Path(...),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """删除会话"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.session_service import create_session_service
                session_service = create_session_service(db)
                session_service.current_user_id = user_id
                result = session_service.delete_session(session_id)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.post("/search", response_model=BaseResponse[List[SessionResponse]])
        @monitor_performance
        async def search_sessions(
            search_params: SessionSearchSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """搜索会话"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.session_service import create_session_service
                session_service = create_session_service(db)
                session_service.current_user_id = user_id
                result = session_service.search_sessions(search_params)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/stats/overview", response_model=BaseResponse[SessionStatsResponse])
        @monitor_performance
        async def get_session_stats(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取会话统计信息"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.session_service import create_session_service
                session_service = create_session_service(db)
                session_service.current_user_id = user_id
                result = session_service.get_session_statistics(UUID(user_id))
                return result
            except Exception as e:
                raise self._handle_service_error(e)


class MessageController(BaseController):
    """消息控制器"""
    
    def __init__(self):
        super().__init__()
        self.router.prefix = "/messages"
        self.router.tags = ["messages"]
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.post("/", response_model=BaseResponse[MessageResponse])
        @monitor_performance
        async def create_message(
            message_data: MessageCreateSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """创建消息"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.message_service import create_message_service
                message_service = create_message_service(db)
                message_service.current_user_id = user_id
                result = message_service.create_message(message_data)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/session/{session_id}", response_model=BaseResponse[List[MessageResponse]])
        @monitor_performance
        async def get_session_messages(
            session_id: UUID = Path(...),
            limit: int = Query(50, ge=1, le=100),
            offset: int = Query(0, ge=0),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取会话消息"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.message_service import create_message_service
                message_service = create_message_service(db)
                message_service.current_user_id = user_id
                result = message_service.get_session_messages(session_id, limit, offset)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/conversation/{session_id}", response_model=BaseResponse[ConversationHistorySchema])
        @monitor_performance
        async def get_conversation_history(
            session_id: UUID = Path(...),
            limit: int = Query(50, ge=1, le=100),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取对话历史"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.message_service import create_message_service
                message_service = create_message_service(db)
                message_service.current_user_id = user_id
                result = message_service.get_conversation_history(session_id, limit)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.post("/search", response_model=BaseResponse[List[MessageResponse]])
        @monitor_performance
        async def search_messages(
            search_params: MessageSearchSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """搜索消息"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.message_service import create_message_service
                message_service = create_message_service(db)
                message_service.current_user_id = user_id
                result = message_service.search_messages(search_params)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.put("/{message_id}", response_model=BaseResponse[MessageResponse])
        @monitor_performance
        async def update_message(
            message_id: UUID = Path(...),
            message_data: MessageUpdateSchema = Body(...),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """更新消息"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.message_service import create_message_service
                message_service = create_message_service(db)
                message_service.current_user_id = user_id
                result = message_service.update(message_id, message_data)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/stats/overview", response_model=BaseResponse[MessageStatsResponse])
        @monitor_performance
        async def get_message_stats(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取消息统计信息"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.message_service import create_message_service
                message_service = create_message_service(db)
                message_service.current_user_id = user_id
                result = message_service.get_message_statistics(UUID(user_id))
                return result
            except Exception as e:
                raise self._handle_service_error(e)


class WorkflowController(BaseController):
    """工作流控制器"""
    
    def __init__(self):
        super().__init__()
        self.router.prefix = "/workflows"
        self.router.tags = ["workflows"]
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.post("/", response_model=BaseResponse[WorkflowResponse])
        @monitor_performance
        async def create_workflow(
            workflow_data: WorkflowCreateSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """创建工作流"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.workflow_service import create_workflow_service
                workflow_service = create_workflow_service(db)
                workflow_service.current_user_id = user_id
                result = workflow_service.create_workflow(workflow_data)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/", response_model=BaseResponse[List[WorkflowResponse]])
        @monitor_performance
        async def get_user_workflows(
            limit: int = Query(50, ge=1, le=100),
            offset: int = Query(0, ge=0),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取用户工作流"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.workflow_service import create_workflow_service
                workflow_service = create_workflow_service(db)
                workflow_service.current_user_id = user_id
                result = workflow_service.get_user_workflows(UUID(user_id), limit, offset)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/public", response_model=BaseResponse[List[WorkflowResponse]])
        @monitor_performance
        async def get_public_workflows(
            limit: int = Query(50, ge=1, le=100),
            offset: int = Query(0, ge=0),
            db: Session = Depends(get_db_session)
        ):
            """获取公开工作流"""
            try:
                from ..services.workflow_service import create_workflow_service
                workflow_service = create_workflow_service(db)
                result = workflow_service.get_public_workflows(limit, offset)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/{workflow_id}", response_model=BaseResponse[WorkflowResponse])
        @monitor_performance
        async def get_workflow(
            workflow_id: UUID = Path(...),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取工作流详情"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.workflow_service import create_workflow_service
                workflow_service = create_workflow_service(db)
                workflow_service.current_user_id = user_id
                result = workflow_service.get_by_id(workflow_id)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.post("/{workflow_id}/execute", response_model=BaseResponse[Dict[str, Any]])
        @monitor_performance
        async def execute_workflow(
            workflow_id: UUID = Path(...),
            execution_data: WorkflowExecuteSchema = Body(...),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """执行工作流"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.workflow_service import create_workflow_service
                workflow_service = create_workflow_service(db)
                workflow_service.current_user_id = user_id
                result = workflow_service.execute_workflow(workflow_id, execution_data)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.post("/search", response_model=BaseResponse[List[WorkflowResponse]])
        @monitor_performance
        async def search_workflows(
            search_params: WorkflowSearchSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """搜索工作流"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.workflow_service import create_workflow_service
                workflow_service = create_workflow_service(db)
                workflow_service.current_user_id = user_id
                result = workflow_service.search_workflows(search_params)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/stats/overview", response_model=BaseResponse[WorkflowStatsResponse])
        @monitor_performance
        async def get_workflow_stats(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取工作流统计信息"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.workflow_service import create_workflow_service
                workflow_service = create_workflow_service(db)
                workflow_service.current_user_id = user_id
                result = workflow_service.get_workflow_statistics(UUID(user_id))
                return result
            except Exception as e:
                raise self._handle_service_error(e)


class MemoryController(BaseController):
    """记忆控制器"""
    
    def __init__(self):
        super().__init__()
        self.router.prefix = "/memories"
        self.router.tags = ["memories"]
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.post("/", response_model=BaseResponse[MemoryResponse])
        @monitor_performance
        async def create_memory(
            memory_data: MemoryCreateSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """创建记忆"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.memory_service import create_memory_service
                memory_service = create_memory_service(db)
                memory_service.current_user_id = user_id
                result = memory_service.create_memory(memory_data)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.post("/retrieve", response_model=BaseResponse[List[MemoryResponse]])
        @monitor_performance
        async def retrieve_memories(
            retrieval_params: MemoryRetrievalSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """检索相关记忆"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.memory_service import create_memory_service
                memory_service = create_memory_service(db)
                memory_service.current_user_id = user_id
                result = memory_service.retrieve_memories(retrieval_params)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.post("/search", response_model=BaseResponse[List[MemoryResponse]])
        @monitor_performance
        async def search_memories(
            search_params: MemorySearchSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """搜索记忆"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.memory_service import create_memory_service
                memory_service = create_memory_service(db)
                memory_service.current_user_id = user_id
                result = memory_service.search_memories(search_params)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/session/{session_id}", response_model=BaseResponse[List[MemoryResponse]])
        @monitor_performance
        async def get_session_memories(
            session_id: UUID = Path(...),
            memory_type: Optional[MemoryType] = Query(None),
            limit: int = Query(100, ge=1, le=200),
            offset: int = Query(0, ge=0),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取会话记忆"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.memory_service import create_memory_service
                memory_service = create_memory_service(db)
                memory_service.current_user_id = user_id
                result = memory_service.get_session_memories(session_id, memory_type, limit, offset)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/{memory_id}/related", response_model=BaseResponse[List[MemoryResponse]])
        @monitor_performance
        async def get_related_memories(
            memory_id: UUID = Path(...),
            max_results: int = Query(10, ge=1, le=50),
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取相关记忆"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.memory_service import create_memory_service
                memory_service = create_memory_service(db)
                memory_service.current_user_id = user_id
                result = memory_service.get_related_memories(memory_id, max_results)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.post("/consolidate", response_model=BaseResponse[Dict[str, Any]])
        @monitor_performance
        async def consolidate_memories(
            consolidation_params: MemoryConsolidationSchema,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """整合记忆"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.memory_service import create_memory_service
                memory_service = create_memory_service(db)
                memory_service.current_user_id = user_id
                result = memory_service.consolidate_memories(consolidation_params)
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.post("/decay", response_model=BaseResponse[Dict[str, Any]])
        @monitor_performance
        async def apply_memory_decay(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """应用记忆衰减"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.memory_service import create_memory_service
                memory_service = create_memory_service(db)
                memory_service.current_user_id = user_id
                result = memory_service.apply_memory_decay()
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/stats/overview", response_model=BaseResponse[MemoryStatsResponse])
        @monitor_performance
        async def get_memory_stats(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取记忆统计信息"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.memory_service import create_memory_service
                memory_service = create_memory_service(db)
                memory_service.current_user_id = user_id
                result = memory_service.get_memory_statistics()
                return result
            except Exception as e:
                raise self._handle_service_error(e)
        
        @self.router.get("/analytics", response_model=BaseResponse[MemoryAnalyticsResponse])
        @monitor_performance
        async def get_memory_analytics(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(get_db_session)
        ):
            """获取记忆分析"""
            try:
                user_id = self._get_current_user_id(credentials)
                from ..services.memory_service import create_memory_service
                memory_service = create_memory_service(db)
                memory_service.current_user_id = user_id
                result = memory_service.get_memory_analytics()
                return result
            except Exception as e:
                raise self._handle_service_error(e)


# 创建控制器实例
def create_controllers() -> Dict[str, BaseController]:
    """创建所有控制器实例"""
    return {
        "users": UserController(),
        "sessions": SessionController(),
        "messages": MessageController(),
        "workflows": WorkflowController(),
        "memories": MemoryController()
    }


# 获取所有路由器
def get_all_routers() -> List[APIRouter]:
    """获取所有API路由器"""
    controllers = create_controllers()
    return [controller.router for controller in controllers.values()]