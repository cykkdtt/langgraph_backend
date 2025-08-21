"""模型转换工具

本模块提供数据库模型到API模型的转换方法，确保数据一致性和格式标准化。
包括：
- 数据库模型到API响应模型的转换
- 数据验证和清洗
- 性能优化的批量转换
- 缓存策略支持
- 错误处理和数据验证
- 异步批量处理
"""

from typing import List, Dict, Any, Optional, Union, Type, TypeVar, Callable, Tuple
from datetime import datetime, timezone
from pydantic import BaseModel, ValidationError
import hashlib
import json
import asyncio
import logging
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

# 导入数据库模型
from .database_models import (
    User as DBUser,
    Session as DBSession,
    Message as DBMessage,
    ToolCall as DBToolCall,
    AgentState as DBAgentState,
    SystemLog as DBSystemLog,
    Workflow as DBWorkflow,
    WorkflowExecution as DBWorkflowExecution,
    Memory as DBMemory
)

# 导入API模型
from .api_models import BaseResponse, PaginatedResponse
from .chat_models import (
    ChatMessage,
    ToolCall,
    AgentStatus,
    SessionInfo,
    MessageInfo,
    AgentMetrics
)

T = TypeVar('T', bound=BaseModel)
DBModel = TypeVar('DBModel')
APIModel = TypeVar('APIModel')

# 配置日志
logger = logging.getLogger(__name__)


class ConversionError(Exception):
    """转换错误异常"""
    def __init__(self, message: str, model_type: str = None, model_id: str = None, original_error: Exception = None):
        self.message = message
        self.model_type = model_type
        self.model_id = model_id
        self.original_error = original_error
        super().__init__(self.message)


class ValidationError(Exception):
    """数据验证错误异常"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class ConversionMode(Enum):
    """转换模式枚举"""
    STRICT = "strict"  # 严格模式，验证失败抛出异常
    LENIENT = "lenient"  # 宽松模式，验证失败记录警告但继续
    SILENT = "silent"  # 静默模式，忽略验证错误


@dataclass
class ConversionConfig:
    """转换配置"""
    mode: ConversionMode = ConversionMode.STRICT
    enable_cache: bool = True
    cache_ttl: int = 300  # 缓存TTL（秒）
    batch_size: int = 100  # 批量处理大小
    max_workers: int = 4  # 最大工作线程数
    enable_validation: bool = True
    enable_performance_tracking: bool = False


@dataclass
class ConversionResult:
    """转换结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    warnings: List[str] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class BatchConversionResult:
    """批量转换结果"""
    total_count: int
    success_count: int
    error_count: int
    results: List[ConversionResult]
    total_processing_time: float
    errors: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / self.total_count if self.total_count > 0 else 0.0


def performance_tracker(func: Callable) -> Callable:
    """性能跟踪装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            # 记录性能指标
            if hasattr(args[0], 'config') and args[0].config.enable_performance_tracking:
                logger.info(f"{func.__name__} 执行时间: {processing_time:.4f}秒")
            
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"{func.__name__} 执行失败，耗时: {processing_time:.4f}秒，错误: {str(e)}")
            raise
    return wrapper


class ModelConverter:
    """模型转换器基类"""
    
    def __init__(self, config: ConversionConfig = None):
        self.config = config or ConversionConfig()
        self._cache = {}
        self._cache_timestamps = {}
        self._performance_stats = defaultdict(list)
    
    def _safe_convert_datetime(self, dt: Optional[datetime]) -> Optional[str]:
        """安全转换日期时间为ISO格式字符串"""
        try:
            if dt is None:
                return None
            # 确保时区信息
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception as e:
            self._log_warning(f"日期时间转换失败: {str(e)}")
            return None
    
    def _safe_get_dict(self, data: Any, default: Optional[Dict] = None) -> Dict[str, Any]:
        """安全获取字典数据"""
        try:
            if data is None:
                return default or {}
            if isinstance(data, dict):
                return data
            if isinstance(data, str):
                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict):
                        return parsed
                    else:
                        self._log_warning(f"解析的JSON不是字典类型: {type(parsed)}")
                        return default or {}
                except (json.JSONDecodeError, ValueError) as e:
                    self._log_warning(f"JSON解析失败: {str(e)}")
                    return default or {}
            return default or {}
        except Exception as e:
            self._log_warning(f"字典数据获取失败: {str(e)}")
            return default or {}
    
    def _calculate_content_hash(self, content: str) -> str:
        """计算内容哈希值"""
        try:
            if not content:
                return ""
            return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        except Exception as e:
            self._log_warning(f"内容哈希计算失败: {str(e)}")
            return ""
    
    def _validate_input(self, data: Any, model_type: str) -> bool:
        """验证输入数据"""
        if not self.config.enable_validation:
            return True
        
        try:
            if data is None:
                raise ValidationError(f"{model_type} 数据不能为空")
            
            # 检查必要的属性
            if hasattr(data, 'id') and not data.id:
                raise ValidationError(f"{model_type} ID不能为空")
            
            return True
        except ValidationError as e:
            if self.config.mode == ConversionMode.STRICT:
                raise
            elif self.config.mode == ConversionMode.LENIENT:
                self._log_warning(f"验证警告: {e.message}")
            return False
    
    def _log_warning(self, message: str):
        """记录警告信息"""
        if self.config.mode != ConversionMode.SILENT:
            logger.warning(message)
    
    def _log_error(self, message: str, error: Exception = None):
        """记录错误信息"""
        if error:
            logger.error(f"{message}: {str(error)}")
        else:
            logger.error(message)
    
    def _get_cache_key(self, model_type: str, model_id: str, model_hash: str = None) -> str:
        """生成缓存键"""
        if model_hash:
            return f"{model_type}:{model_id}:{model_hash}"
        return f"{model_type}:{model_id}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if not self.config.enable_cache:
            return False
        
        if cache_key not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[cache_key]
        return (time.time() - cache_time) < self.config.cache_ttl
    
    def _get_from_cache(self, cache_key: str) -> Any:
        """从缓存获取数据"""
        if self._is_cache_valid(cache_key):
            return self._cache.get(cache_key)
        return None
    
    def _set_cache(self, cache_key: str, data: Any):
        """设置缓存数据"""
        if self.config.enable_cache:
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = time.time()
    
    def _clear_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if (current_time - timestamp) >= self.config.cache_ttl
        ]
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    @performance_tracker
    def convert_single(self, db_model: DBModel, converter_func: Callable) -> ConversionResult:
        """单个模型转换"""
        start_time = time.time()
        
        try:
            # 验证输入
            model_type = type(db_model).__name__
            if not self._validate_input(db_model, model_type):
                return ConversionResult(
                    success=False,
                    error=f"{model_type} 验证失败",
                    processing_time=time.time() - start_time
                )
            
            # 检查缓存
            model_id = getattr(db_model, 'id', None)
            cache_key = self._get_cache_key(model_type, str(model_id)) if model_id else None
            
            if cache_key:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    return ConversionResult(
                        success=True,
                        data=cached_result,
                        processing_time=time.time() - start_time
                    )
            
            # 执行转换
            result = converter_func(db_model)
            
            # 设置缓存
            if cache_key:
                self._set_cache(cache_key, result)
            
            return ConversionResult(
                success=True,
                data=result,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self._log_error(f"转换失败", e)
            return ConversionResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def convert_batch(self, db_models: List[DBModel], converter_func: Callable) -> BatchConversionResult:
        """批量模型转换"""
        start_time = time.time()
        total_count = len(db_models)
        results = []
        errors = []
        
        # 分批处理
        for i in range(0, total_count, self.config.batch_size):
            batch = db_models[i:i + self.config.batch_size]
            
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_model = {
                    executor.submit(self.convert_single, model, converter_func): model 
                    for model in batch
                }
                
                for future in as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if not result.success:
                            errors.append({
                                'model_type': type(model).__name__,
                                'model_id': getattr(model, 'id', None),
                                'error': result.error
                            })
                    except Exception as e:
                        error_result = ConversionResult(
                            success=False,
                            error=str(e),
                            processing_time=0.0
                        )
                        results.append(error_result)
                        errors.append({
                            'model_type': type(model).__name__,
                            'model_id': getattr(model, 'id', None),
                            'error': str(e)
                        })
        
        success_count = sum(1 for r in results if r.success)
        error_count = total_count - success_count
        
        return BatchConversionResult(
            total_count=total_count,
            success_count=success_count,
            error_count=error_count,
            results=results,
            total_processing_time=time.time() - start_time,
            errors=errors
        )
    
    async def convert_batch_async(self, db_models: List[DBModel], converter_func: Callable) -> BatchConversionResult:
        """异步批量模型转换"""
        start_time = time.time()
        total_count = len(db_models)
        results = []
        errors = []
        
        async def convert_async(model):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.convert_single, model, converter_func)
        
        # 分批异步处理
        for i in range(0, total_count, self.config.batch_size):
            batch = db_models[i:i + self.config.batch_size]
            
            # 创建异步任务
            tasks = [convert_async(model) for model in batch]
            
            # 等待批次完成
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                model = batch[j]
                if isinstance(result, Exception):
                    error_result = ConversionResult(
                        success=False,
                        error=str(result),
                        processing_time=0.0
                    )
                    results.append(error_result)
                    errors.append({
                        'model_type': type(model).__name__,
                        'model_id': getattr(model, 'id', None),
                        'error': str(result)
                    })
                else:
                    results.append(result)
                    if not result.success:
                        errors.append({
                            'model_type': type(model).__name__,
                            'model_id': getattr(model, 'id', None),
                            'error': result.error
                        })
        
        success_count = sum(1 for r in results if r.success)
        error_count = total_count - success_count
        
        return BatchConversionResult(
            total_count=total_count,
            success_count=success_count,
            error_count=error_count,
            results=results,
            total_processing_time=time.time() - start_time,
            errors=errors
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            'cache_size': len(self._cache),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'performance_stats': dict(self._performance_stats)
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        # 这里需要实现缓存命中率的计算逻辑
        return 0.0
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_timestamps.clear()


class UserConverter(ModelConverter):
    """用户模型转换器"""
    
    def __init__(self, config: ConversionConfig = None):
        super().__init__(config)
    
    def db_to_api(self, db_user: DBUser) -> Dict[str, Any]:
        """数据库用户模型转换为API响应格式"""
        try:
            return {
                "id": db_user.id,
                "username": db_user.username,
                "email": db_user.email,
                "display_name": db_user.display_name,
                "avatar_url": db_user.avatar_url,
                "is_active": db_user.is_active,
                "is_verified": db_user.is_verified,
                "role": db_user.role,
                "preferences": self._safe_get_dict(db_user.preferences),
                "profile": self._safe_get_dict(db_user.profile),
                "created_at": self._safe_convert_datetime(db_user.created_at),
                "updated_at": self._safe_convert_datetime(db_user.updated_at),
                "last_login_at": self._safe_convert_datetime(db_user.last_login_at),
                "timezone": db_user.timezone,
                "language": db_user.language,
                "subscription_tier": db_user.subscription_tier,
                "usage_stats": self._safe_get_dict(db_user.usage_stats)
            }
        except Exception as e:
            raise ConversionError(f"用户模型转换失败: {str(e)}", "User", str(db_user.id), e)
    
    def db_list_to_api(self, db_users: List[DBUser]) -> List[Dict[str, Any]]:
        """批量转换用户列表"""
        result = self.convert_batch(db_users, self.db_to_api)
        return [r.data for r in result.results if r.success]
    
    async def db_list_to_api_async(self, db_users: List[DBUser]) -> BatchConversionResult:
        """异步批量转换用户列表"""
        return await self.convert_batch_async(db_users, self.db_to_api)
    
    def validate_user_data(self, db_user: DBUser) -> bool:
        """验证用户数据"""
        try:
            if not db_user.username or len(db_user.username.strip()) == 0:
                raise ValidationError("用户名不能为空", "username", db_user.username)
            
            if db_user.email and '@' not in db_user.email:
                raise ValidationError("邮箱格式无效", "email", db_user.email)
            
            return True
        except ValidationError as e:
            if self.config.mode == ConversionMode.STRICT:
                raise
            elif self.config.mode == ConversionMode.LENIENT:
                self._log_warning(f"用户数据验证警告: {e.message}")
            return False


class SessionConverter(ModelConverter):
    """会话模型转换器"""
    
    def __init__(self, config: ConversionConfig = None):
        super().__init__(config)
    
    def db_to_api(self, db_session: DBSession) -> SessionInfo:
        """数据库会话模型转换为API会话信息"""
        try:
            return SessionInfo(
                session_id=db_session.id,
                user_id=db_session.user_id,
                title=db_session.title,
                description=db_session.description,
                mode=db_session.mode or "auto_select",
                active_agents=self._safe_get_dict(db_session.agent_config, {}).get("active_agents", []),
                message_count=db_session.message_count or 0,
                created_at=db_session.created_at,
                updated_at=db_session.updated_at,
                metadata=self._safe_get_dict(db_session.metadata)
            )
        except Exception as e:
            raise ConversionError(f"会话模型转换失败: {str(e)}", "Session", str(db_session.id), e)
    
    def db_to_dict(self, db_session: DBSession) -> Dict[str, Any]:
        """数据库会话模型转换为字典格式"""
        try:
            return {
                "id": db_session.id,
                "user_id": db_session.user_id,
                "title": db_session.title,
                "description": db_session.description,
                "mode": db_session.mode,
                "status": db_session.status,
                "is_active": db_session.is_active,
                "agent_config": self._safe_get_dict(db_session.agent_config),
                "context_data": self._safe_get_dict(db_session.context_data),
                "settings": self._safe_get_dict(db_session.settings),
                "metadata": self._safe_get_dict(db_session.metadata),
                "message_count": db_session.message_count,
                "token_usage": db_session.token_usage,
                "created_at": self._safe_convert_datetime(db_session.created_at),
                "updated_at": self._safe_convert_datetime(db_session.updated_at),
                "last_activity_at": self._safe_convert_datetime(db_session.last_activity_at),
                "expires_at": self._safe_convert_datetime(db_session.expires_at)
            }
        except Exception as e:
            raise ConversionError(f"会话字典转换失败: {str(e)}", "Session", str(db_session.id), e)
    
    def db_list_to_api(self, db_sessions: List[DBSession]) -> List[SessionInfo]:
        """批量转换会话列表为API格式"""
        result = self.convert_batch(db_sessions, self.db_to_api)
        return [r.data for r in result.results if r.success]
    
    def db_list_to_dict(self, db_sessions: List[DBSession]) -> List[Dict[str, Any]]:
        """批量转换会话列表为字典格式"""
        result = self.convert_batch(db_sessions, self.db_to_dict)
        return [r.data for r in result.results if r.success]
    
    async def db_list_to_api_async(self, db_sessions: List[DBSession]) -> BatchConversionResult:
        """异步批量转换会话列表"""
        return await self.convert_batch_async(db_sessions, self.db_to_api)
    
    def validate_session_data(self, db_session: DBSession) -> bool:
        """验证会话数据"""
        try:
            if not db_session.title or len(db_session.title.strip()) == 0:
                raise ValidationError("会话标题不能为空", "title", db_session.title)
            
            if db_session.message_count and db_session.message_count < 0:
                raise ValidationError("消息数量不能为负数", "message_count", db_session.message_count)
            
            return True
        except ValidationError as e:
            if self.config.mode == ConversionMode.STRICT:
                raise
            elif self.config.mode == ConversionMode.LENIENT:
                self._log_warning(f"会话数据验证警告: {e.message}")
            return False


class MessageConverter(ModelConverter):
    """消息模型转换器"""
    
    def __init__(self, config: ConversionConfig = None):
        super().__init__(config)
    
    def db_to_chat_message(self, db_message: DBMessage) -> ChatMessage:
        """数据库消息模型转换为聊天消息"""
        try:
            return ChatMessage(
                id=db_message.id,
                role=db_message.role,
                content=db_message.content,
                message_type=db_message.message_type or "text",
                metadata=self._safe_get_dict(db_message.metadata),
                tool_calls=[],  # 需要单独查询和转换
                attachments=self._safe_get_dict(db_message.attachments, {}).get("files", []),
                created_at=db_message.created_at,
                updated_at=db_message.updated_at
            )
        except Exception as e:
            raise ConversionError(f"聊天消息转换失败: {str(e)}", "Message", str(db_message.id), e)
    
    def db_to_message_info(self, db_message: DBMessage) -> MessageInfo:
        """数据库消息模型转换为消息信息"""
        try:
            return MessageInfo(
                id=db_message.id,
                role=db_message.role,
                content=db_message.content,
                message_type=db_message.message_type or "text",
                created_at=db_message.created_at,
                metadata=self._safe_get_dict(db_message.metadata)
            )
        except Exception as e:
            raise ConversionError(f"消息信息转换失败: {str(e)}", "Message", str(db_message.id), e)
    
    def db_to_dict(self, db_message: DBMessage) -> Dict[str, Any]:
        """数据库消息模型转换为字典格式"""
        try:
            return {
                "id": db_message.id,
                "session_id": db_message.session_id,
                "user_id": db_message.user_id,
                "parent_message_id": db_message.parent_message_id,
                "role": db_message.role,
                "content": db_message.content,
                "message_type": db_message.message_type,
                "status": db_message.status,
                "priority": db_message.priority,
                "content_length": db_message.content_length,
                "token_count": db_message.token_count,
                "processing_time": db_message.processing_time,
                "response_time": db_message.response_time,
                "quality_score": db_message.quality_score,
                "relevance_score": db_message.relevance_score,
                "version": db_message.version,
                "edit_count": db_message.edit_count,
                "metadata": self._safe_get_dict(db_message.metadata),
                "attachments": self._safe_get_dict(db_message.attachments),
                "performance_metrics": self._safe_get_dict(db_message.performance_metrics),
                "created_at": self._safe_convert_datetime(db_message.created_at),
                "updated_at": self._safe_convert_datetime(db_message.updated_at),
                "deleted_at": self._safe_convert_datetime(db_message.deleted_at)
            }
        except Exception as e:
            raise ConversionError(f"消息字典转换失败: {str(e)}", "Message", str(db_message.id), e)
    
    def db_list_to_chat_messages(self, db_messages: List[DBMessage]) -> List[ChatMessage]:
        """批量转换消息列表为聊天消息格式"""
        result = self.convert_batch(db_messages, self.db_to_chat_message)
        return [r.data for r in result.results if r.success]
    
    def db_list_to_message_info(self, db_messages: List[DBMessage]) -> List[MessageInfo]:
        """批量转换消息列表为消息信息格式"""
        result = self.convert_batch(db_messages, self.db_to_message_info)
        return [r.data for r in result.results if r.success]
    
    def db_list_to_dict(self, db_messages: List[DBMessage]) -> List[Dict[str, Any]]:
        """批量转换消息列表为字典格式"""
        result = self.convert_batch(db_messages, self.db_to_dict)
        return [r.data for r in result.results if r.success]
    
    async def db_list_to_chat_messages_async(self, db_messages: List[DBMessage]) -> BatchConversionResult:
        """异步批量转换消息列表为聊天消息"""
        return await self.convert_batch_async(db_messages, self.db_to_chat_message)
    
    def validate_message_data(self, db_message: DBMessage) -> bool:
        """验证消息数据"""
        try:
            if not db_message.content or len(db_message.content.strip()) == 0:
                raise ValidationError("消息内容不能为空", "content", db_message.content)
            
            if db_message.role not in ["user", "assistant", "system", "tool"]:
                raise ValidationError("消息角色无效", "role", db_message.role)
            
            if db_message.token_count and db_message.token_count < 0:
                raise ValidationError("令牌数量不能为负数", "token_count", db_message.token_count)
            
            return True
        except ValidationError as e:
            if self.config.mode == ConversionMode.STRICT:
                raise
            elif self.config.mode == ConversionMode.LENIENT:
                self._log_warning(f"消息数据验证警告: {e.message}")
            return False


class ToolCallConverter(ModelConverter):
    """工具调用模型转换器"""
    
    def __init__(self, config: ConversionConfig = None):
        super().__init__(config)
    
    def db_to_api(self, db_tool_call: DBToolCall) -> ToolCall:
        """数据库工具调用模型转换为API工具调用"""
        try:
            return ToolCall(
                id=db_tool_call.id,
                name=db_tool_call.tool_name,
                arguments=self._safe_get_dict(db_tool_call.input_data),
                result=self._safe_get_dict(db_tool_call.output_data),
                status=db_tool_call.status,
                error=db_tool_call.error_message,
                execution_time=db_tool_call.execution_time,
                created_at=db_tool_call.created_at
            )
        except Exception as e:
            raise ConversionError(f"工具调用转换失败: {str(e)}", "ToolCall", str(db_tool_call.id), e)
    
    def db_to_dict(self, db_tool_call: DBToolCall) -> Dict[str, Any]:
        """数据库工具调用模型转换为字典格式"""
        try:
            return {
                "id": db_tool_call.id,
                "message_id": db_tool_call.message_id,
                "tool_name": db_tool_call.tool_name,
                "function_name": db_tool_call.function_name,
                "input_data": self._safe_get_dict(db_tool_call.input_data),
                "output_data": self._safe_get_dict(db_tool_call.output_data),
                "status": db_tool_call.status,
                "error_message": db_tool_call.error_message,
                "execution_time": db_tool_call.execution_time,
                "created_at": self._safe_convert_datetime(db_tool_call.created_at),
                "completed_at": self._safe_convert_datetime(db_tool_call.completed_at),
                "metadata": self._safe_get_dict(db_tool_call.metadata)
            }
        except Exception as e:
            raise ConversionError(f"工具调用字典转换失败: {str(e)}", "ToolCall", str(db_tool_call.id), e)
    
    def db_list_to_api(self, db_tool_calls: List[DBToolCall]) -> List[ToolCall]:
        """批量转换工具调用列表"""
        result = self.convert_batch(db_tool_calls, self.db_to_api)
        return [r.data for r in result.results if r.success]
    
    def db_list_to_dict(self, db_tool_calls: List[DBToolCall]) -> List[Dict[str, Any]]:
        """批量转换工具调用列表为字典格式"""
        result = self.convert_batch(db_tool_calls, self.db_to_dict)
        return [r.data for r in result.results if r.success]
    
    async def db_list_to_api_async(self, db_tool_calls: List[DBToolCall]) -> BatchConversionResult:
        """异步批量转换工具调用列表"""
        return await self.convert_batch_async(db_tool_calls, self.db_to_api)
    
    def validate_tool_call_data(self, db_tool_call: DBToolCall) -> bool:
        """验证工具调用数据"""
        try:
            if not db_tool_call.tool_name or len(db_tool_call.tool_name.strip()) == 0:
                raise ValidationError("工具名称不能为空", "tool_name", db_tool_call.tool_name)
            
            if not db_tool_call.function_name or len(db_tool_call.function_name.strip()) == 0:
                raise ValidationError("函数名称不能为空", "function_name", db_tool_call.function_name)
            
            if db_tool_call.execution_time and db_tool_call.execution_time < 0:
                raise ValidationError("执行时间不能为负数", "execution_time", db_tool_call.execution_time)
            
            return True
        except ValidationError as e:
            if self.config.mode == ConversionMode.STRICT:
                raise
            elif self.config.mode == ConversionMode.LENIENT:
                self._log_warning(f"工具调用数据验证警告: {e.message}")
            return False


class AgentStateConverter(ModelConverter):
    """智能体状态模型转换器"""
    
    def __init__(self, config: ConversionConfig = None):
        super().__init__(config)
    
    def db_to_api(self, db_agent_state: DBAgentState) -> AgentStatus:
        """数据库智能体状态转换为API智能体状态"""
        try:
            return AgentStatus(
                agent_type=db_agent_state.agent_type or "unknown",
                agent_id=db_agent_state.agent_id,
                status=db_agent_state.status,
                current_task=db_agent_state.current_task,
                progress=None,  # 需要从performance_metrics中提取
                metadata=self._safe_get_dict(db_agent_state.context_data),
                updated_at=db_agent_state.updated_at or db_agent_state.created_at
            )
        except Exception as e:
            raise ConversionError(f"智能体状态转换失败: {str(e)}", "AgentState", str(db_agent_state.id), e)
    
    def db_to_metrics(self, db_agent_state: DBAgentState) -> AgentMetrics:
        """数据库智能体状态转换为智能体指标"""
        try:
            performance_metrics = self._safe_get_dict(db_agent_state.performance_metrics)
            
            return AgentMetrics(
                agent_type=db_agent_state.agent_type or "unknown",
                total_requests=performance_metrics.get("total_requests", 0),
                successful_requests=performance_metrics.get("successful_requests", 0),
                failed_requests=performance_metrics.get("failed_requests", 0),
                average_response_time=performance_metrics.get("average_response_time", 0.0),
                total_tokens_used=performance_metrics.get("total_tokens_used", 0),
                last_used=db_agent_state.last_activity_at,
                uptime=performance_metrics.get("uptime", 0.0)
            )
        except Exception as e:
            raise ConversionError(f"智能体指标转换失败: {str(e)}", "AgentState", str(db_agent_state.id), e)
    
    def db_list_to_api(self, db_agent_states: List[DBAgentState]) -> List[AgentStatus]:
        """批量转换智能体状态列表"""
        result = self.convert_batch(db_agent_states, self.db_to_api)
        return [r.data for r in result.results if r.success]
    
    def db_list_to_metrics(self, db_agent_states: List[DBAgentState]) -> List[AgentMetrics]:
        """批量转换智能体状态为指标"""
        result = self.convert_batch(db_agent_states, self.db_to_metrics)
        return [r.data for r in result.results if r.success]
    
    async def db_list_to_api_async(self, db_agent_states: List[DBAgentState]) -> BatchConversionResult:
        """异步批量转换智能体状态列表"""
        return await self.convert_batch_async(db_agent_states, self.db_to_api)
    
    def validate_agent_state_data(self, db_agent_state: DBAgentState) -> bool:
        """验证智能体状态数据"""
        try:
            if not db_agent_state.agent_type or len(db_agent_state.agent_type.strip()) == 0:
                raise ValidationError("智能体类型不能为空", "agent_type", db_agent_state.agent_type)
            
            return True
        except ValidationError as e:
            if self.config.mode == ConversionMode.STRICT:
                raise
            elif self.config.mode == ConversionMode.LENIENT:
                self._log_warning(f"智能体状态数据验证警告: {e.message}")
            return False


class WorkflowConverter(ModelConverter):
    """工作流模型转换器"""
    
    def __init__(self, config: ConversionConfig = None):
        super().__init__(config)
    
    def db_to_dict(self, db_workflow: DBWorkflow) -> Dict[str, Any]:
        """数据库工作流模型转换为字典格式"""
        try:
            return {
                "id": db_workflow.id,
                "name": db_workflow.name,
                "description": db_workflow.description,
                "category": db_workflow.category,
                "version": f"{db_workflow.major_version}.{db_workflow.minor_version}.{db_workflow.patch_version}",
                "status": db_workflow.status,
                "is_active": db_workflow.is_active,
                "is_featured": db_workflow.is_featured,
                "is_template": db_workflow.is_template,
                "complexity_score": db_workflow.complexity_score,
                "estimated_duration": db_workflow.estimated_duration,
                "max_concurrent_executions": db_workflow.max_concurrent_executions,
                "tags": self._safe_get_dict(db_workflow.tags, {}).get("list", []),
                "workflow_definition": self._safe_get_dict(db_workflow.workflow_definition),
                "workflow_metadata": self._safe_get_dict(db_workflow.workflow_metadata),
                "permissions": self._safe_get_dict(db_workflow.permissions),
                "execution_stats": {
                    "execution_count": db_workflow.execution_count,
                    "success_count": db_workflow.success_count,
                    "failure_count": db_workflow.failure_count,
                    "average_duration": db_workflow.average_duration,
                    "last_execution_at": self._safe_convert_datetime(db_workflow.last_execution_at)
                },
                "rating": {
                    "rating": db_workflow.rating,
                    "rating_count": db_workflow.rating_count
                },
                "usage_count": db_workflow.usage_count,
                "created_at": self._safe_convert_datetime(db_workflow.created_at),
                "updated_at": self._safe_convert_datetime(db_workflow.updated_at),
                "published_at": self._safe_convert_datetime(db_workflow.published_at),
                "archived_at": self._safe_convert_datetime(db_workflow.archived_at)
            }
        except Exception as e:
            raise ConversionError(f"工作流转换失败: {str(e)}", "Workflow", str(db_workflow.id), e)
    
    def db_list_to_dict(self, db_workflows: List[DBWorkflow]) -> List[Dict[str, Any]]:
        """批量转换工作流列表"""
        result = self.convert_batch(db_workflows, self.db_to_dict)
        return [r.data for r in result.results if r.success]
    
    async def db_list_to_dict_async(self, db_workflows: List[DBWorkflow]) -> BatchConversionResult:
        """异步批量转换工作流列表"""
        return await self.convert_batch_async(db_workflows, self.db_to_dict)
    
    def validate_workflow_data(self, db_workflow: DBWorkflow) -> bool:
        """验证工作流数据"""
        try:
            if not db_workflow.name or len(db_workflow.name.strip()) == 0:
                raise ValidationError("工作流名称不能为空", "name", db_workflow.name)
            
            definition = self._safe_get_dict(db_workflow.workflow_definition)
            if not definition or not definition.get("steps"):
                raise ValidationError("工作流定义必须包含步骤", "workflow_definition", definition)
            
            return True
        except ValidationError as e:
            if self.config.mode == ConversionMode.STRICT:
                raise
            elif self.config.mode == ConversionMode.LENIENT:
                self._log_warning(f"工作流数据验证警告: {e.message}")
            return False


class MemoryConverter(ModelConverter):
    """记忆模型转换器"""
    
    def __init__(self, config: ConversionConfig = None):
        super().__init__(config)
    
    def db_to_dict(self, db_memory: DBMemory) -> Dict[str, Any]:
        """数据库记忆模型转换为字典格式"""
        try:
            return {
                "id": db_memory.id,
                "user_id": db_memory.user_id,
                "session_id": db_memory.session_id,
                "parent_memory_id": db_memory.parent_memory_id,
                "memory_type": db_memory.memory_type,
                "memory_subtype": db_memory.memory_subtype,
                "content": db_memory.content,
                "content_hash": db_memory.content_hash,
                "vector_info": {
                    "vector_model": db_memory.vector_model,
                    "vector_dimension": db_memory.vector_dimension,
                    "content_vector": db_memory.content_vector  # 注意：向量数据可能很大
                },
                "scores": {
                    "importance_score": db_memory.importance_score,
                    "confidence_score": db_memory.confidence_score,
                    "relevance_score": db_memory.relevance_score,
                    "quality_score": db_memory.quality_score
                },
                "usage_stats": {
                    "access_count": db_memory.access_count,
                    "retrieval_count": db_memory.retrieval_count,
                    "consolidation_count": db_memory.consolidation_count
                },
                "lifecycle": {
                    "decay_rate": db_memory.decay_rate,
                    "strength": db_memory.strength,
                    "is_consolidated": db_memory.is_consolidated,
                    "is_active": db_memory.is_active,
                    "expires_at": self._safe_convert_datetime(db_memory.expires_at),
                    "archived_at": self._safe_convert_datetime(db_memory.archived_at)
                },
                "metadata": self._safe_get_dict(db_memory.memory_metadata),
                "tags": self._safe_get_dict(db_memory.tags, {}).get("list", []),
                "associations": self._safe_get_dict(db_memory.associations),
                "context_data": self._safe_get_dict(db_memory.context_data),
                "timestamps": {
                    "created_at": self._safe_convert_datetime(db_memory.created_at),
                    "updated_at": self._safe_convert_datetime(db_memory.updated_at),
                    "last_accessed_at": self._safe_convert_datetime(db_memory.last_accessed_at),
                    "last_retrieved_at": self._safe_convert_datetime(db_memory.last_retrieved_at),
                    "last_consolidated_at": self._safe_convert_datetime(db_memory.last_consolidated_at)
                }
            }
        except Exception as e:
            raise ConversionError(f"记忆转换失败: {str(e)}", "Memory", str(db_memory.id), e)
    
    def db_list_to_dict(self, db_memories: List[DBMemory]) -> List[Dict[str, Any]]:
        """批量转换记忆列表"""
        result = self.convert_batch(db_memories, self.db_to_dict)
        return [r.data for r in result.results if r.success]
    
    async def db_list_to_dict_async(self, db_memories: List[DBMemory]) -> BatchConversionResult:
        """异步批量转换记忆列表"""
        return await self.convert_batch_async(db_memories, self.db_to_dict)
    
    def validate_memory_data(self, db_memory: DBMemory) -> bool:
        """验证记忆数据"""
        try:
            if not db_memory.content or len(db_memory.content.strip()) == 0:
                raise ValidationError("记忆内容不能为空", "content", db_memory.content)
            
            if not db_memory.memory_type or len(db_memory.memory_type.strip()) == 0:
                raise ValidationError("记忆类型不能为空", "memory_type", db_memory.memory_type)
            
            if db_memory.importance_score and (db_memory.importance_score < 0 or db_memory.importance_score > 1):
                raise ValidationError("重要性分数必须在0-1之间", "importance_score", db_memory.importance_score)
            
            if db_memory.access_count and db_memory.access_count < 0:
                raise ValidationError("访问次数不能为负数", "access_count", db_memory.access_count)
            
            return True
        except ValidationError as e:
            if self.config.mode == ConversionMode.STRICT:
                raise
            elif self.config.mode == ConversionMode.LENIENT:
                self._log_warning(f"记忆数据验证警告: {e.message}")
            return False


class ResponseBuilder:
    """响应构建器 - 提供统一的API响应格式"""
    
    def __init__(self, config: ConversionConfig = None):
        self.config = config or ConversionConfig()
        self._response_cache = {}
        self._cache_timestamps = {}
    
    @performance_tracker
    def success(self, data: Any = None, message: str = "操作成功", 
                metadata: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """构建成功响应"""
        try:
            response = {
                "success": True,
                "message": message,
                "data": data,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": kwargs.get("request_id"),
                "version": kwargs.get("api_version", "1.0")
            }
            
            # 添加性能信息（如果启用）
            if self.config.enable_performance_tracking:
                response["performance"] = {
                    "processing_time_ms": kwargs.get("processing_time", 0),
                    "cache_hit": kwargs.get("cache_hit", False)
                }
            
            response.update({k: v for k, v in kwargs.items() 
                           if k not in ["request_id", "api_version", "processing_time", "cache_hit"]})
            return response
        except Exception as e:
            logger.error(f"构建成功响应失败: {str(e)}")
            return self._fallback_response("success", data, message)
    
    @performance_tracker
    def error(self, message: str = "操作失败", error_code: str = None, 
              details: Any = None, status_code: int = 400, 
              suggestions: List[str] = None, **kwargs) -> Dict[str, Any]:
        """构建错误响应"""
        try:
            response = {
                "success": False,
                "message": message,
                "error": {
                    "code": error_code or "UNKNOWN_ERROR",
                    "details": details,
                    "suggestions": suggestions or [],
                    "type": kwargs.get("error_type", "application_error")
                },
                "status_code": status_code,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": kwargs.get("request_id"),
                "version": kwargs.get("api_version", "1.0")
            }
            
            # 添加调试信息（仅在开发模式）
            if hasattr(self.config, 'enable_debug') and self.config.enable_debug and kwargs.get("debug_info"):
                response["debug"] = kwargs["debug_info"]
            
            response.update({k: v for k, v in kwargs.items() 
                           if k not in ["request_id", "api_version", "error_type", "debug_info"]})
            return response
        except Exception as e:
            logger.error(f"构建错误响应失败: {str(e)}")
            return self._fallback_response("error", message, error_code)
    
    @performance_tracker
    def paginated(self, data: List[Any], page: int, page_size: int, total: int,
                  filters: Dict[str, Any] = None, sort: Dict[str, Any] = None,
                  **kwargs) -> Dict[str, Any]:
        """构建分页响应"""
        try:
            total_pages = max(1, (total + page_size - 1) // page_size)
            current_page = max(1, min(page, total_pages))
            
            response = {
                "success": True,
                "data": data,
                "pagination": {
                    "page": current_page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages,
                    "has_next": current_page < total_pages,
                    "has_prev": current_page > 1,
                    "next_page": current_page + 1 if current_page < total_pages else None,
                    "prev_page": current_page - 1 if current_page > 1 else None
                },
                "filters": filters or {},
                "sort": sort or {},
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": kwargs.get("request_id"),
                "version": kwargs.get("api_version", "1.0")
            }
            
            response.update({k: v for k, v in kwargs.items() 
                           if k not in ["request_id", "api_version"]})
            return response
        except Exception as e:
            logger.error(f"构建分页响应失败: {str(e)}")
            return self._fallback_response("paginated", data, page, page_size, total)
    
    def validation_error(self, errors: List[Dict[str, Any]], 
                        message: str = "数据验证失败", **kwargs) -> Dict[str, Any]:
        """构建验证错误响应"""
        return self.error(
            message=message,
            error_code="VALIDATION_ERROR",
            details={
                "validation_errors": errors,
                "error_count": len(errors)
            },
            status_code=422,
            error_type="validation_error",
            suggestions=[
                "请检查输入数据格式",
                "确保所有必填字段都已提供",
                "验证数据类型和长度限制"
            ],
            **kwargs
        )
    
    def batch_result(self, results: BatchConversionResult, 
                    message: str = "批量操作完成", **kwargs) -> Dict[str, Any]:
        """构建批量操作结果响应"""
        successful_data = [r.data for r in results.results if r.success]
        failed_results = [r for r in results.results if not r.success]
        
        response_data = {
            "success_count": results.success_count,
            "error_count": results.error_count,
            "total_count": results.total_count,
            "success_rate": results.success_rate,
            "data": successful_data
        }
        
        if failed_results:
            response_data["failures"] = [
                {
                    "index": i,
                    "error": r.error,
                    "warnings": r.warnings
                } for i, r in enumerate(failed_results)
            ]
        
        if hasattr(results, 'warnings') and results.warnings:
            response_data["warnings"] = results.warnings
        
        return self.success(
            data=response_data,
            message=message,
            metadata={
                "total_processing_time": results.total_processing_time,
                "batch_size": results.total_count
            },
            **kwargs
        )
    
    def _fallback_response(self, response_type: str, *args, **kwargs) -> Dict[str, Any]:
        """备用响应构建器（当主要构建器失败时使用）"""
        return {
            "success": response_type == "success",
            "message": "响应构建失败，使用备用格式",
            "data": args[0] if args else None,
            "error": {
                "code": "RESPONSE_BUILD_ERROR",
                "details": "主要响应构建器失败"
            } if response_type == "error" else None,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def clear_cache(self):
        """清除响应缓存"""
        self._response_cache.clear()
        self._cache_timestamps.clear()
    
    @staticmethod
    def build_success_response(data: Any, message: str = "操作成功") -> BaseResponse:
        """构建成功响应（向后兼容）"""
        return BaseResponse.success(data=data, message=message)
    
    @staticmethod
    def build_error_response(message: str, error_code: str = "INTERNAL_ERROR") -> BaseResponse:
        """构建错误响应（向后兼容）"""
        return BaseResponse.error(message=message, error_code=error_code)
    
    @staticmethod
    def build_paginated_response(
        items: List[Any],
        total: int,
        page: int,
        page_size: int,
        message: str = "查询成功"
    ) -> PaginatedResponse:
        """构建分页响应（向后兼容）"""
        from .api_models import PaginationInfo
        
        pagination = PaginationInfo.create(
            total=total,
            page=page,
            page_size=page_size
        )
        
        return PaginatedResponse.success(
            data=items,
            pagination=pagination,
            message=message
        )


# 缓存装饰器
def cached_model_conversion(cache_ttl: int = 300, max_cache_size: int = 1000):
    """模型转换缓存装饰器
    
    Args:
        cache_ttl: 缓存生存时间（秒）
        max_cache_size: 最大缓存条目数
    """
    def decorator(func):
        cache = {}
        cache_timestamps = {}
        access_counts = defaultdict(int)
        
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = str(hash((str(args), str(sorted(kwargs.items())))))
            current_time = time.time()
            
            # 检查缓存是否有效
            if (cache_key in cache and 
                cache_key in cache_timestamps and 
                current_time - cache_timestamps[cache_key] < cache_ttl):
                access_counts[cache_key] += 1
                return cache[cache_key]
            
            # 执行函数并缓存结果
            try:
                result = func(*args, **kwargs)
                
                # 如果缓存已满，移除最少使用的条目
                if len(cache) >= max_cache_size:
                    # 找到访问次数最少且最旧的条目
                    lru_key = min(
                        cache_timestamps.keys(),
                        key=lambda k: (access_counts[k], cache_timestamps[k])
                    )
                    cache.pop(lru_key, None)
                    cache_timestamps.pop(lru_key, None)
                    access_counts.pop(lru_key, None)
                
                cache[cache_key] = result
                cache_timestamps[cache_key] = current_time
                access_counts[cache_key] = 1
                
                # 定期清理过期缓存
                if len(cache) % 100 == 0:  # 每100次调用清理一次
                    expired_keys = [
                        key for key, timestamp in cache_timestamps.items()
                        if current_time - timestamp >= cache_ttl
                    ]
                    for key in expired_keys:
                        cache.pop(key, None)
                        cache_timestamps.pop(key, None)
                        access_counts.pop(key, None)
                
                return result
            except Exception as e:
                logger.error(f"缓存装饰器执行函数失败: {str(e)}")
                raise
        
        # 添加缓存管理方法
        def clear_cache():
            cache.clear()
            cache_timestamps.clear()
            access_counts.clear()
        
        def get_cache_stats():
            current_time = time.time()
            valid_entries = sum(
                1 for timestamp in cache_timestamps.values()
                if current_time - timestamp < cache_ttl
            )
            return {
                "total_entries": len(cache),
                "valid_entries": valid_entries,
                "expired_entries": len(cache) - valid_entries,
                "cache_hit_rate": sum(access_counts.values()) / max(len(cache), 1),
                "max_size": max_cache_size,
                "ttl": cache_ttl
            }
        
        wrapper.clear_cache = clear_cache
        wrapper.get_cache_stats = get_cache_stats
        
        return wrapper
    return decorator


# 全局转换器实例（使用默认配置）
default_config = ConversionConfig()
default_user_converter = UserConverter(default_config)
default_session_converter = SessionConverter(default_config)
default_message_converter = MessageConverter(default_config)
default_tool_call_converter = ToolCallConverter(default_config)
default_agent_state_converter = AgentStateConverter(default_config)
default_workflow_converter = WorkflowConverter(default_config)
default_memory_converter = MemoryConverter(default_config)
default_response_builder = ResponseBuilder(default_config)


# 便捷函数
def get_converter(converter_type: str, config: ConversionConfig = None):
    """获取指定类型的转换器实例
    
    Args:
        converter_type: 转换器类型 ('user', 'session', 'message', 'tool_call', 'agent_state', 'workflow', 'memory')
        config: 转换配置，如果为None则使用默认配置
    
    Returns:
        对应的转换器实例
    """
    config = config or default_config
    
    converters = {
        'user': UserConverter,
        'session': SessionConverter,
        'message': MessageConverter,
        'tool_call': ToolCallConverter,
        'agent_state': AgentStateConverter,
        'workflow': WorkflowConverter,
        'memory': MemoryConverter
    }
    
    if converter_type not in converters:
        raise ValueError(f"不支持的转换器类型: {converter_type}")
    
    return converters[converter_type](config)


def create_response_builder(config: ConversionConfig = None) -> ResponseBuilder:
    """创建响应构建器实例
    
    Args:
        config: 转换配置，如果为None则使用默认配置
    
    Returns:
        ResponseBuilder实例
    """
    return ResponseBuilder(config or default_config)


# 导出所有转换器
__all__ = [
    # 异常类
    'ConversionError',
    'ValidationError',
    
    # 配置和结果类
    'ConversionMode',
    'ConversionConfig',
    'ConversionResult',
    'BatchConversionResult',
    
    # 转换器类
    'ModelConverter',
    'UserConverter',
    'SessionConverter', 
    'MessageConverter',
    'ToolCallConverter',
    'AgentStateConverter',
    'WorkflowConverter',
    'MemoryConverter',
    'ResponseBuilder',
    
    # 装饰器和工具函数
    'performance_tracker',
    'cached_model_conversion',
    
    # 便捷函数
    'get_converter',
    'create_response_builder',
    
    # 全局实例
    'default_config',
    'default_user_converter',
    'default_session_converter',
    'default_message_converter',
    'default_tool_call_converter',
    'default_agent_state_converter',
    'default_workflow_converter',
    'default_memory_converter',
    'default_response_builder'
]