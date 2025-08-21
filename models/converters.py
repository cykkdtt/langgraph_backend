"""数据转换工具模块

本模块提供数据库模型与API响应模型之间的转换功能，确保数据的一致性和安全性。
"""

from typing import List, Optional, Dict, Any, Type, TypeVar
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import inspect

from .database_models import (
    User, Session as DBSession, Message, ToolCall, AgentState, SystemLog,
    Workflow, WorkflowExecution, Memory
)
from .response_models import (
    UserResponse, UserProfileResponse, SessionResponse, ThreadResponse,
    MessageResponse, MessageWithAttachmentsResponse, WorkflowResponse,
    WorkflowExecutionResponse, WorkflowStepResponse, MemoryResponse,
    MemoryVectorResponse, TimeTravelResponse, AttachmentResponse,
    UserPreferenceResponse, SystemConfigResponse
)

# 泛型类型变量
T = TypeVar('T')
R = TypeVar('R')


class BaseConverter:
    """基础转换器类"""
    
    @staticmethod
    def filter_sensitive_data(data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """过滤敏感数据"""
        return {k: v for k, v in data.items() if k not in sensitive_fields}
    
    @staticmethod
    def ensure_datetime_format(dt: Optional[datetime]) -> Optional[datetime]:
        """确保日期时间格式正确"""
        return dt if dt else None
    
    @staticmethod
    def safe_json_loads(data: Any, default: Any = None) -> Any:
        """安全的JSON加载"""
        if data is None:
            return default or {}
        if isinstance(data, (dict, list)):
            return data
        try:
            import json
            return json.loads(data) if isinstance(data, str) else data
        except (json.JSONDecodeError, TypeError):
            return default or {}


class UserConverter(BaseConverter):
    """用户模型转换器"""
    
    SENSITIVE_FIELDS = ['password_hash', 'salt', 'reset_token', 'verification_token']
    
    @classmethod
    def to_response(cls, user: User, include_profile: bool = False) -> UserResponse:
        """将用户数据库模型转换为响应模型"""
        if not user:
            return None
            
        base_data = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': user.full_name,
            'avatar_url': user.avatar_url,
            'status': user.status.value if user.status else 'inactive',
            'role': user.role.value if user.role else 'user',
            'is_active': user.is_active,
            'is_verified': user.is_verified,
            'last_login_at': cls.ensure_datetime_format(user.last_login_at),
            'settings': cls.safe_json_loads(user.settings),
            'preferences': cls.safe_json_loads(user.preferences),
            'statistics': cls.safe_json_loads(user.statistics),
            'created_at': user.created_at,
            'updated_at': user.updated_at
        }
        
        if include_profile:
            profile_data = {
                'bio': user.bio,
                'location': user.location,
                'website': user.website,
                'timezone': user.timezone,
                'language': user.language,
                'session_count': len(user.sessions) if user.sessions else 0,
                'thread_count': len(user.threads) if user.threads else 0,
                'message_count': len(user.messages) if user.messages else 0
            }
            base_data.update(profile_data)
            return UserProfileResponse(**base_data)
        
        return UserResponse(**base_data)
    
    @classmethod
    def to_response_list(cls, users: List[User], include_profile: bool = False) -> List[UserResponse]:
        """批量转换用户列表"""
        return [cls.to_response(user, include_profile) for user in users if user]


class SessionConverter(BaseConverter):
    """会话模型转换器"""
    
    @classmethod
    def to_response(cls, session: DBSession) -> SessionResponse:
        """将会话数据库模型转换为响应模型"""
        if not session:
            return None
            
        return SessionResponse(
            id=session.id,
            user_id=session.user_id,
            title=session.title,
            description=session.description,
            is_active=session.is_active,
            settings=cls.safe_json_loads(session.settings),
            metadata=cls.safe_json_loads(session.metadata),
            last_activity_at=cls.ensure_datetime_format(session.last_activity_at),
            expires_at=cls.ensure_datetime_format(session.expires_at),
            message_count=session.message_count,
            thread_count=session.thread_count,
            created_at=session.created_at,
            updated_at=session.updated_at
        )
    
    @classmethod
    def to_response_list(cls, sessions: List[DBSession]) -> List[SessionResponse]:
        """批量转换会话列表"""
        return [cls.to_response(session) for session in sessions if session]


# class ThreadConverter(BaseConverter):
#     """线程模型转换器"""
#     
#     @classmethod
#     def to_response(cls, thread: Thread) -> ThreadResponse:
#         """将线程数据库模型转换为响应模型"""
#         if not thread:
#             return None
#             
#         return ThreadResponse(
#             id=thread.id,
#             user_id=thread.user_id,
#             session_id=thread.session_id,
#             title=thread.title,
#             description=thread.description,
#             is_active=thread.is_active,
#             is_archived=thread.is_archived,
#             settings=cls.safe_json_loads(thread.settings),
#             metadata=cls.safe_json_loads(thread.metadata),
#             last_message_at=cls.ensure_datetime_format(thread.last_message_at),
#             message_count=thread.message_count,
#             token_count=thread.token_count,
#             created_at=thread.created_at,
#             updated_at=thread.updated_at
#         )
#     
#     @classmethod
#     def to_response_list(cls, threads: List[Thread]) -> List[ThreadResponse]:
#         """批量转换线程列表"""
#         return [cls.to_response(thread) for thread in threads if thread]


class MessageConverter(BaseConverter):
    """消息模型转换器"""
    
    @classmethod
    def to_response(cls, message: Message, include_attachments: bool = False) -> MessageResponse:
        """将消息数据库模型转换为响应模型"""
        if not message:
            return None
            
        base_data = {
            'id': message.id,
            'thread_id': message.thread_id,
            'user_id': message.user_id,
            'role': message.role.value if message.role else 'user',
            'message_type': message.message_type.value if message.message_type else 'text',
            'content': message.content,
            'content_data': cls.safe_json_loads(message.content_data),
            'metadata': cls.safe_json_loads(message.metadata),
            'parent_id': message.parent_id,
            'reply_to_id': message.reply_to_id,
            'is_edited': message.is_edited,
            'is_deleted': message.is_deleted,
            'is_pinned': message.is_pinned,
            'token_count': message.token_count,
            'character_count': message.character_count,
            'edit_count': message.edit_count,
            'edited_at': cls.ensure_datetime_format(message.edited_at),
            'deleted_at': cls.ensure_datetime_format(message.deleted_at),
            'created_at': message.created_at,
            'updated_at': message.updated_at
        }
        
        if include_attachments and hasattr(message, 'attachments'):
            attachments = [AttachmentConverter.to_response(att) for att in message.attachments]
            base_data['attachments'] = attachments
            return MessageWithAttachmentsResponse(**base_data)
        
        return MessageResponse(**base_data)
    
    @classmethod
    def to_response_list(cls, messages: List[Message], include_attachments: bool = False) -> List[MessageResponse]:
        """批量转换消息列表"""
        return [cls.to_response(message, include_attachments) for message in messages if message]


class WorkflowConverter(BaseConverter):
    """工作流模型转换器"""
    
    @classmethod
    def to_response(cls, workflow: Workflow) -> WorkflowResponse:
        """将工作流数据库模型转换为响应模型"""
        if not workflow:
            return None
            
        return WorkflowResponse(
            id=workflow.id,
            user_id=workflow.user_id,
            name=workflow.name,
            description=workflow.description,
            version=workflow.version,
            definition=cls.safe_json_loads(workflow.definition),
            config=cls.safe_json_loads(workflow.config),
            status=workflow.status.value if workflow.status else 'draft',
            is_public=workflow.is_public,
            is_template=workflow.is_template,
            execution_count=workflow.execution_count,
            success_count=workflow.success_count,
            failure_count=workflow.failure_count,
            tags=workflow.tags or [],
            category=workflow.category,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at
        )
    
    @classmethod
    def to_response_list(cls, workflows: List[Workflow]) -> List[WorkflowResponse]:
        """批量转换工作流列表"""
        return [cls.to_response(workflow) for workflow in workflows if workflow]


class WorkflowExecutionConverter(BaseConverter):
    """工作流执行模型转换器"""
    
    @classmethod
    def to_response(cls, execution: WorkflowExecution) -> WorkflowExecutionResponse:
        """将工作流执行数据库模型转换为响应模型"""
        if not execution:
            return None
            
        return WorkflowExecutionResponse(
            id=execution.id,
            workflow_id=execution.workflow_id,
            user_id=execution.user_id,
            status=execution.status.value if execution.status else 'pending',
            input_data=cls.safe_json_loads(execution.input_data),
            output_data=cls.safe_json_loads(execution.output_data),
            context=cls.safe_json_loads(execution.context),
            started_at=cls.ensure_datetime_format(execution.started_at),
            completed_at=cls.ensure_datetime_format(execution.completed_at),
            duration=execution.duration,
            error_message=execution.error_message,
            error_details=cls.safe_json_loads(execution.error_details),
            step_count=execution.step_count,
            completed_steps=execution.completed_steps,
            failed_steps=execution.failed_steps,
            created_at=execution.created_at,
            updated_at=execution.updated_at
        )
    
    @classmethod
    def to_response_list(cls, executions: List[WorkflowExecution]) -> List[WorkflowExecutionResponse]:
        """批量转换工作流执行列表"""
        return [cls.to_response(execution) for execution in executions if execution]


# class WorkflowStepConverter(BaseConverter):
#     """工作流步骤模型转换器"""
#     
#     @classmethod
#     def to_response(cls, step: WorkflowStep) -> WorkflowStepResponse:
#         """将工作流步骤数据库模型转换为响应模型"""
#         if not step:
#             return None
#             
#         return WorkflowStepResponse(
#             id=step.id,
#             execution_id=step.execution_id,
#             step_name=step.step_name,
#             step_type=step.step_type,
#             step_order=step.step_order,
#             status=step.status.value if step.status else 'pending',
#             input_data=cls.safe_json_loads(step.input_data),
#             output_data=cls.safe_json_loads(step.output_data),
#             started_at=cls.ensure_datetime_format(step.started_at),
#             completed_at=cls.ensure_datetime_format(step.completed_at),
#             duration=step.duration,
#             error_message=step.error_message,
#             error_details=cls.safe_json_loads(step.error_details),
#             retry_count=step.retry_count,
#             max_retries=step.max_retries,
#             created_at=step.created_at,
#             updated_at=step.updated_at
#         )
#     
#     @classmethod
#     def to_response_list(cls, steps: List[WorkflowStep]) -> List[WorkflowStepResponse]:
#         """批量转换工作流步骤列表"""
#         return [cls.to_response(step) for step in steps if step]


class MemoryConverter(BaseConverter):
    """记忆模型转换器"""
    
    @classmethod
    def to_response(cls, memory: Memory) -> MemoryResponse:
        """将记忆数据库模型转换为响应模型"""
        if not memory:
            return None
            
        return MemoryResponse(
            id=memory.id,
            user_id=memory.user_id,
            thread_id=memory.thread_id,
            memory_type=memory.memory_type.value if memory.memory_type else 'semantic',
            title=memory.title,
            content=memory.content,
            summary=memory.summary,
            data=cls.safe_json_loads(memory.data),
            metadata=cls.safe_json_loads(memory.metadata),
            importance=memory.importance,
            weight=memory.weight,
            accessed_at=memory.accessed_at,
            access_count=memory.access_count,
            tags=memory.tags or [],
            category=memory.category,
            created_at=memory.created_at,
            updated_at=memory.updated_at
        )
    
    @classmethod
    def to_response_list(cls, memories: List[Memory]) -> List[MemoryResponse]:
        """批量转换记忆列表"""
        return [cls.to_response(memory) for memory in memories if memory]


# class MemoryVectorConverter(BaseConverter):
#     """记忆向量模型转换器"""
#     
#     @classmethod
#     def to_response(cls, vector: MemoryVector) -> MemoryVectorResponse:
#         """将记忆向量数据库模型转换为响应模型"""
#         if not vector:
#             return None
#             
#         return MemoryVectorResponse(
#             id=vector.id,
#             memory_id=vector.memory_id,
#             vector_type=vector.vector_type,
#             model_name=vector.model_name,
#             vector_data=vector.vector_data or [],
#             dimension=vector.dimension,
#             metadata=cls.safe_json_loads(vector.metadata),
#             created_at=vector.created_at,
#             updated_at=vector.updated_at
#         )
#     
#     @classmethod
#     def to_response_list(cls, vectors: List[MemoryVector]) -> List[MemoryVectorResponse]:
#         """批量转换记忆向量列表"""
#         return [cls.to_response(vector) for vector in vectors if vector]


# class TimeTravelConverter(BaseConverter):
#     """时间旅行模型转换器"""
#     
#     @classmethod
#     def to_response(cls, timetravel: TimeTravel) -> TimeTravelResponse:
#         """将时间旅行数据库模型转换为响应模型"""
#         if not timetravel:
#             return None
#             
#         return TimeTravelResponse(
#             id=timetravel.id,
#             user_id=timetravel.user_id,
#             thread_id=timetravel.thread_id,
#             snapshot_name=timetravel.snapshot_name,
#             description=timetravel.description,
#             snapshot_data=cls.safe_json_loads(timetravel.snapshot_data),
#             metadata=cls.safe_json_loads(timetravel.metadata),
#             snapshot_type=timetravel.snapshot_type,
#             is_active=timetravel.is_active,
#             data_size=timetravel.data_size,
#             restore_count=timetravel.restore_count,
#             created_at=timetravel.created_at,
#             updated_at=timetravel.updated_at
#         )
#     
#     @classmethod
#     def to_response_list(cls, timetravels: List[TimeTravel]) -> List[TimeTravelResponse]:
#         """批量转换时间旅行列表"""
#         return [cls.to_response(timetravel) for timetravel in timetravels if timetravel]


# class AttachmentConverter(BaseConverter):
#     """附件模型转换器"""
#     
#     @classmethod
#     def to_response(cls, attachment: Attachment) -> AttachmentResponse:
#         """将附件数据库模型转换为响应模型"""
#         if not attachment:
#             return None
#             
#         # 生成下载链接（这里可以根据实际需求实现）
#         download_url = f"/api/attachments/{attachment.id}/download" if attachment.id else None
#         
#         return AttachmentResponse(
#             id=attachment.id,
#             user_id=attachment.user_id,
#             message_id=attachment.message_id,
#             filename=attachment.filename,
#             original_filename=attachment.original_filename,
#             file_path=attachment.file_path,
#             file_size=attachment.file_size,
#             file_type=attachment.file_type,
#             mime_type=attachment.mime_type,
#             file_hash=attachment.file_hash,
#             checksum=attachment.checksum,
#             is_public=attachment.is_public,
#             is_processed=attachment.is_processed,
#             metadata=cls.safe_json_loads(attachment.metadata),
#             download_count=attachment.download_count,
#             download_url=download_url,
#             created_at=attachment.created_at,
#             updated_at=attachment.updated_at
#         )
#     
#     @classmethod
#     def to_response_list(cls, attachments: List[Attachment]) -> List[AttachmentResponse]:
#         """批量转换附件列表"""
#         return [cls.to_response(attachment) for attachment in attachments if attachment]


# class UserPreferenceConverter(BaseConverter):
#     """用户偏好模型转换器"""
#     
#     @classmethod
#     def to_response(cls, preference: UserPreference) -> UserPreferenceResponse:
#         """将用户偏好数据库模型转换为响应模型"""
#         if not preference:
#             return None
#             
#         return UserPreferenceResponse(
#             id=preference.id,
#             user_id=preference.user_id,
#             preference_key=preference.preference_key,
#             preference_value=cls.safe_json_loads(preference.preference_value),
#             preference_type=preference.preference_type,
#             description=preference.description,
#             is_public=preference.is_public,
#             created_at=preference.created_at,
#             updated_at=preference.updated_at
#         )
#     
#     @classmethod
#     def to_response_list(cls, preferences: List[UserPreference]) -> List[UserPreferenceResponse]:
#         """批量转换用户偏好列表"""
#         return [cls.to_response(preference) for preference in preferences if preference]


# class SystemConfigConverter(BaseConverter):
#     """系统配置模型转换器"""
#     
#     SENSITIVE_KEYS = ['password', 'secret', 'key', 'token', 'credential']
#     
#     @classmethod
#     def to_response(cls, config: SystemConfig, include_sensitive: bool = False) -> SystemConfigResponse:
#         """将系统配置数据库模型转换为响应模型"""
#         if not config:
#             return None
#         
#         # 检查是否包含敏感信息
#         config_value = cls.safe_json_loads(config.config_value)
#         if not include_sensitive and any(key in config.config_key.lower() for key in cls.SENSITIVE_KEYS):
#             config_value = "[HIDDEN]"
#             
#         return SystemConfigResponse(
#             id=config.id,
#             config_key=config.config_key,
#             config_value=config_value,
#             config_type=config.config_type,
#             description=config.description,
#             is_active=config.is_active,
#             is_encrypted=config.is_encrypted,
#             version=config.version,
#             created_at=config.created_at,
#             updated_at=config.updated_at
#         )
#     
#     @classmethod
#     def to_response_list(cls, configs: List[SystemConfig], include_sensitive: bool = False) -> List[SystemConfigResponse]:
#         """批量转换系统配置列表"""
#         return [cls.to_response(config, include_sensitive) for config in configs if config]


class ConverterRegistry:
    """转换器注册表"""
    
    _converters = {
        User: UserConverter,
        DBSession: SessionConverter,
        # Thread: ThreadConverter,  # Thread模型不存在
        Message: MessageConverter,
        Workflow: WorkflowConverter,
        WorkflowExecution: WorkflowExecutionConverter,
        # WorkflowStep: WorkflowStepConverter,  # WorkflowStep模型不存在
        Memory: MemoryConverter,
        # MemoryVector: MemoryVectorConverter,  # MemoryVector模型不存在
        # TimeTravel: TimeTravelConverter,  # TimeTravel模型不存在
        # Attachment: AttachmentConverter,  # Attachment模型不存在
        # UserPreference: UserPreferenceConverter,  # UserPreference模型不存在
        # SystemConfig: SystemConfigConverter  # SystemConfig模型不存在
    }
    
    @classmethod
    def get_converter(cls, model_class: Type[T]) -> Optional[Type[BaseConverter]]:
        """获取模型对应的转换器"""
        return cls._converters.get(model_class)
    
    @classmethod
    def convert_to_response(cls, model_instance: T, **kwargs) -> Optional[R]:
        """通用转换方法"""
        if not model_instance:
            return None
            
        converter = cls.get_converter(type(model_instance))
        if not converter:
            raise ValueError(f"No converter found for model {type(model_instance)}")
            
        return converter.to_response(model_instance, **kwargs)
    
    @classmethod
    def convert_to_response_list(cls, model_instances: List[T], **kwargs) -> List[R]:
        """批量转换方法"""
        if not model_instances:
            return []
            
        if not model_instances:
            return []
            
        converter = cls.get_converter(type(model_instances[0]))
        if not converter:
            raise ValueError(f"No converter found for model {type(model_instances[0])}")
            
        return converter.to_response_list(model_instances, **kwargs)


# 导出所有转换器
__all__ = [
    "BaseConverter",
    "UserConverter",
    "SessionConverter",
    "ThreadConverter",
    "MessageConverter",
    "WorkflowConverter",
    "WorkflowExecutionConverter",
    "WorkflowStepConverter",
    "MemoryConverter",
    "MemoryVectorConverter",
    "TimeTravelConverter",
    "AttachmentConverter",
    "UserPreferenceConverter",
    "SystemConfigConverter",
    "ConverterRegistry"
]