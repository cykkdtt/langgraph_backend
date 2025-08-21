"""模型工厂系统模块

本模块提供模型实例创建、配置和管理功能。
"""

from typing import (
    Dict, Any, Optional, List, Union, Callable, Type, 
    TypeVar, Generic, Tuple, Set, ClassVar
)
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps
import logging
import threading
import uuid
import inspect
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON

# 导入项目模型
from .models import (
    User, Session as ChatSession, Thread, Message, Workflow, 
    WorkflowExecution, WorkflowStep, Memory, MemoryVector, 
    TimeTravel, Attachment, UserPreference, SystemConfig
)
from .events import EventType, emit_business_event
from .validation import validate_data
from .security import get_current_user


logger = logging.getLogger(__name__)


T = TypeVar('T')
M = TypeVar('M')  # Model type


class CreationStrategy(Enum):
    """创建策略枚举"""
    SIMPLE = "simple"                  # 简单创建
    VALIDATED = "validated"            # 验证创建
    SECURED = "secured"                # 安全创建
    TEMPLATED = "templated"            # 模板创建
    BULK = "bulk"                      # 批量创建
    LAZY = "lazy"                      # 懒创建
    CACHED = "cached"                  # 缓存创建


class InitializationMode(Enum):
    """初始化模式枚举"""
    MINIMAL = "minimal"                # 最小初始化
    STANDARD = "standard"              # 标准初始化
    FULL = "full"                      # 完整初始化
    CUSTOM = "custom"                  # 自定义初始化


class FactoryScope(Enum):
    """工厂作用域枚举"""
    SINGLETON = "singleton"            # 单例
    PROTOTYPE = "prototype"            # 原型
    SESSION = "session"                # 会话
    REQUEST = "request"                # 请求
    THREAD = "thread"                  # 线程


@dataclass
class CreationContext:
    """创建上下文"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    thread_id: Optional[str] = None
    creation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strategy: CreationStrategy = CreationStrategy.SIMPLE
    initialization_mode: InitializationMode = InitializationMode.STANDARD
    scope: FactoryScope = FactoryScope.PROTOTYPE
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_context: Optional['CreationContext'] = None
    
    def copy(self, **overrides) -> 'CreationContext':
        """复制上下文"""
        data = {
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'thread_id': self.thread_id,
            'creation_time': self.creation_time,
            'strategy': self.strategy,
            'initialization_mode': self.initialization_mode,
            'scope': self.scope,
            'metadata': self.metadata.copy(),
            'parent_context': self
        }
        data.update(overrides)
        return CreationContext(**data)
    
    def get_context_id(self) -> str:
        """获取上下文ID"""
        if self.scope == FactoryScope.SESSION and self.session_id:
            return f"session:{self.session_id}"
        elif self.scope == FactoryScope.REQUEST and self.request_id:
            return f"request:{self.request_id}"
        elif self.scope == FactoryScope.THREAD and self.thread_id:
            return f"thread:{self.thread_id}"
        else:
            return f"prototype:{uuid.uuid4()}"


@dataclass
class ModelTemplate:
    """模型模板"""
    name: str
    model_class: Type
    default_values: Dict[str, Any] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    computed_fields: Dict[str, Callable] = field(default_factory=dict)
    validators: List[Callable] = field(default_factory=list)
    post_processors: List[Callable] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def apply_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """应用默认值"""
        result = self.default_values.copy()
        result.update(data)
        return result
    
    def compute_fields(self, data: Dict[str, Any], context: CreationContext) -> Dict[str, Any]:
        """计算字段"""
        result = data.copy()
        
        for field_name, computer in self.computed_fields.items():
            if field_name not in result:
                try:
                    if inspect.signature(computer).parameters:
                        # 计算函数需要参数
                        result[field_name] = computer(data, context)
                    else:
                        # 计算函数不需要参数
                        result[field_name] = computer()
                except Exception as e:
                    logger.warning(f"Failed to compute field '{field_name}': {e}")
        
        return result
    
    def validate(self, data: Dict[str, Any], context: CreationContext) -> List[str]:
        """验证数据"""
        errors = []
        
        # 检查必填字段
        for field in self.required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Required field '{field}' is missing")
        
        # 运行自定义验证器
        for validator in self.validators:
            try:
                if inspect.signature(validator).parameters:
                    # 验证器需要参数
                    result = validator(data, context)
                else:
                    # 验证器不需要参数
                    result = validator(data)
                
                if isinstance(result, str):
                    errors.append(result)
                elif isinstance(result, list):
                    errors.extend(result)
                elif result is False:
                    errors.append(f"Validation failed for {validator.__name__}")
            except Exception as e:
                errors.append(f"Validation error in {validator.__name__}: {e}")
        
        return errors
    
    def post_process(self, instance: Any, context: CreationContext) -> Any:
        """后处理"""
        for processor in self.post_processors:
            try:
                if inspect.signature(processor).parameters:
                    # 处理器需要参数
                    instance = processor(instance, context) or instance
                else:
                    # 处理器不需要参数
                    instance = processor(instance) or instance
            except Exception as e:
                logger.warning(f"Post-processing error in {processor.__name__}: {e}")
        
        return instance


class BaseFactory(ABC, Generic[T]):
    """基础工厂抽象类"""
    
    def __init__(self, model_class: Type[T]):
        self.model_class = model_class
        self._templates: Dict[str, ModelTemplate] = {}
        self._instances: Dict[str, T] = {}  # 用于单例和缓存
        self._lock = threading.RLock()
    
    @abstractmethod
    def create(self, context: CreationContext, **kwargs) -> T:
        """创建模型实例"""
        pass
    
    def register_template(self, template: ModelTemplate) -> None:
        """注册模板"""
        with self._lock:
            self._templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[ModelTemplate]:
        """获取模板"""
        return self._templates.get(name)
    
    def create_from_template(self, template_name: str, context: CreationContext, **kwargs) -> T:
        """从模板创建"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # 应用模板
        data = template.apply_defaults(kwargs)
        data = template.compute_fields(data, context)
        
        # 验证数据
        if context.strategy in [CreationStrategy.VALIDATED, CreationStrategy.SECURED]:
            errors = template.validate(data, context)
            if errors:
                raise ValueError(f"Validation errors: {', '.join(errors)}")
        
        # 创建实例
        instance = self.create(context, **data)
        
        # 后处理
        instance = template.post_process(instance, context)
        
        return instance
    
    def clear_cache(self) -> None:
        """清除缓存"""
        with self._lock:
            self._instances.clear()


class SimpleFactory(BaseFactory[T]):
    """简单工厂"""
    
    def create(self, context: CreationContext, **kwargs) -> T:
        """创建模型实例"""
        # 检查是否需要从缓存获取
        if context.scope in [FactoryScope.SINGLETON, FactoryScope.SESSION]:
            context_id = context.get_context_id()
            
            with self._lock:
                if context_id in self._instances:
                    return self._instances[context_id]
        
        # 创建新实例
        instance = self.model_class(**kwargs)
        
        # 设置通用字段
        if hasattr(instance, 'created_at') and not getattr(instance, 'created_at', None):
            instance.created_at = context.creation_time
        
        if hasattr(instance, 'updated_at'):
            instance.updated_at = context.creation_time
        
        if hasattr(instance, 'created_by') and context.user_id:
            instance.created_by = context.user_id
        
        # 缓存实例
        if context.scope in [FactoryScope.SINGLETON, FactoryScope.SESSION]:
            context_id = context.get_context_id()
            with self._lock:
                self._instances[context_id] = instance
        
        # 发布事件
        emit_business_event(
            EventType.MODEL_AFTER_INSERT,
            "model_created",
            data={
                'model': self.model_class.__name__,
                'instance_id': getattr(instance, 'id', None),
                'context': context.get_context_id(),
                'strategy': context.strategy.value
            }
        )
        
        return instance


class ValidatedFactory(BaseFactory[T]):
    """验证工厂"""
    
    def create(self, context: CreationContext, **kwargs) -> T:
        """创建验证的模型实例"""
        # 数据验证
        try:
            validate_data(kwargs, self.model_class.__name__.lower())
        except Exception as e:
            logger.error(f"Validation failed for {self.model_class.__name__}: {e}")
            raise
        
        # 创建实例
        instance = self.model_class(**kwargs)
        
        # 设置通用字段
        if hasattr(instance, 'created_at') and not getattr(instance, 'created_at', None):
            instance.created_at = context.creation_time
        
        if hasattr(instance, 'updated_at'):
            instance.updated_at = context.creation_time
        
        if hasattr(instance, 'created_by') and context.user_id:
            instance.created_by = context.user_id
        
        # 发布事件
        emit_business_event(
            EventType.MODEL_AFTER_INSERT,
            "validated_model_created",
            data={
                'model': self.model_class.__name__,
                'instance_id': getattr(instance, 'id', None),
                'context': context.get_context_id()
            }
        )
        
        return instance


class SecuredFactory(BaseFactory[T]):
    """安全工厂"""
    
    def create(self, context: CreationContext, **kwargs) -> T:
        """创建安全的模型实例"""
        # 安全检查
        current_user = get_current_user()
        if not current_user and context.user_id:
            logger.warning(f"Creating {self.model_class.__name__} without authenticated user")
        
        # 数据验证
        try:
            validate_data(kwargs, self.model_class.__name__.lower())
        except Exception as e:
            logger.error(f"Validation failed for {self.model_class.__name__}: {e}")
            raise
        
        # 创建实例
        instance = self.model_class(**kwargs)
        
        # 设置安全字段
        if hasattr(instance, 'created_at') and not getattr(instance, 'created_at', None):
            instance.created_at = context.creation_time
        
        if hasattr(instance, 'updated_at'):
            instance.updated_at = context.creation_time
        
        if hasattr(instance, 'created_by'):
            instance.created_by = context.user_id or (current_user.id if current_user else None)
        
        if hasattr(instance, 'is_active') and not hasattr(kwargs, 'is_active'):
            instance.is_active = True
        
        # 发布事件
        emit_business_event(
            EventType.MODEL_AFTER_INSERT,
            "secured_model_created",
            data={
                'model': self.model_class.__name__,
                'instance_id': getattr(instance, 'id', None),
                'context': context.get_context_id(),
                'user_id': context.user_id
            }
        )
        
        return instance


class BulkFactory(BaseFactory[T]):
    """批量工厂"""
    
    def create_bulk(self, context: CreationContext, data_list: List[Dict[str, Any]]) -> List[T]:
        """批量创建模型实例"""
        instances = []
        
        for data in data_list:
            try:
                # 数据验证（如果需要）
                if context.strategy in [CreationStrategy.VALIDATED, CreationStrategy.SECURED]:
                    validate_data(data, self.model_class.__name__.lower())
                
                # 创建实例
                instance = self.model_class(**data)
                
                # 设置通用字段
                if hasattr(instance, 'created_at') and not getattr(instance, 'created_at', None):
                    instance.created_at = context.creation_time
                
                if hasattr(instance, 'updated_at'):
                    instance.updated_at = context.creation_time
                
                if hasattr(instance, 'created_by') and context.user_id:
                    instance.created_by = context.user_id
                
                instances.append(instance)
                
            except Exception as e:
                logger.error(f"Failed to create {self.model_class.__name__} instance: {e}")
                if context.strategy == CreationStrategy.SECURED:
                    raise  # 安全模式下抛出异常
                # 否则跳过错误的数据
        
        # 发布事件
        emit_business_event(
            EventType.MODEL_AFTER_INSERT,
            "bulk_models_created",
            data={
                'model': self.model_class.__name__,
                'count': len(instances),
                'context': context.get_context_id()
            }
        )
        
        return instances
    
    def create(self, context: CreationContext, **kwargs) -> T:
        """创建单个实例（委托给批量创建）"""
        instances = self.create_bulk(context, [kwargs])
        return instances[0] if instances else None


class FactoryRegistry:
    """工厂注册表"""
    
    def __init__(self):
        self._factories: Dict[Type, BaseFactory] = {}
        self._default_strategies: Dict[Type, CreationStrategy] = {}
        self._lock = threading.RLock()
    
    def register_factory(self, model_class: Type, factory: BaseFactory, 
                        default_strategy: CreationStrategy = CreationStrategy.SIMPLE) -> None:
        """注册工厂"""
        with self._lock:
            self._factories[model_class] = factory
            self._default_strategies[model_class] = default_strategy
    
    def get_factory(self, model_class: Type) -> Optional[BaseFactory]:
        """获取工厂"""
        return self._factories.get(model_class)
    
    def get_default_strategy(self, model_class: Type) -> CreationStrategy:
        """获取默认策略"""
        return self._default_strategies.get(model_class, CreationStrategy.SIMPLE)
    
    def create_instance(self, model_class: Type, context: Optional[CreationContext] = None, 
                       **kwargs) -> Any:
        """创建实例"""
        factory = self.get_factory(model_class)
        if not factory:
            # 使用默认工厂
            factory = SimpleFactory(model_class)
        
        if context is None:
            context = CreationContext(
                strategy=self.get_default_strategy(model_class)
            )
        
        return factory.create(context, **kwargs)
    
    def create_from_template(self, model_class: Type, template_name: str, 
                           context: Optional[CreationContext] = None, **kwargs) -> Any:
        """从模板创建实例"""
        factory = self.get_factory(model_class)
        if not factory:
            raise ValueError(f"No factory registered for {model_class.__name__}")
        
        if context is None:
            context = CreationContext(
                strategy=CreationStrategy.TEMPLATED
            )
        
        return factory.create_from_template(template_name, context, **kwargs)
    
    def register_template(self, model_class: Type, template: ModelTemplate) -> None:
        """注册模板"""
        factory = self.get_factory(model_class)
        if factory:
            factory.register_template(template)
        else:
            logger.warning(f"No factory registered for {model_class.__name__}")
    
    def clear_all_caches(self) -> None:
        """清除所有缓存"""
        with self._lock:
            for factory in self._factories.values():
                factory.clear_cache()


class ModelFactory:
    """模型工厂管理器"""
    
    def __init__(self):
        self.registry = FactoryRegistry()
        self._initialize_default_factories()
        self._initialize_default_templates()
    
    def _initialize_default_factories(self) -> None:
        """初始化默认工厂"""
        # 注册简单工厂
        models_simple = [SystemConfig, UserPreference, Attachment]
        for model_class in models_simple:
            self.registry.register_factory(
                model_class, 
                SimpleFactory(model_class), 
                CreationStrategy.SIMPLE
            )
        
        # 注册验证工厂
        models_validated = [Thread, Message, Memory, MemoryVector]
        for model_class in models_validated:
            self.registry.register_factory(
                model_class, 
                ValidatedFactory(model_class), 
                CreationStrategy.VALIDATED
            )
        
        # 注册安全工厂
        models_secured = [User, ChatSession, Workflow, WorkflowExecution, WorkflowStep, TimeTravel]
        for model_class in models_secured:
            self.registry.register_factory(
                model_class, 
                SecuredFactory(model_class), 
                CreationStrategy.SECURED
            )
    
    def _initialize_default_templates(self) -> None:
        """初始化默认模板"""
        # 用户模板
        user_template = ModelTemplate(
            name="default_user",
            model_class=User,
            default_values={
                'is_active': True,
                'is_verified': False
            },
            required_fields=['username', 'email'],
            computed_fields={
                'id': lambda: str(uuid.uuid4()),
                'created_at': lambda: datetime.now(timezone.utc)
            },
            validators=[
                lambda data: "Invalid email format" if '@' not in data.get('email', '') else None
            ]
        )
        self.registry.register_template(User, user_template)
        
        # 会话模板
        session_template = ModelTemplate(
            name="default_session",
            model_class=ChatSession,
            default_values={
                'is_active': True
            },
            required_fields=['user_id'],
            computed_fields={
                'id': lambda: str(uuid.uuid4()),
                'created_at': lambda: datetime.now(timezone.utc)
            }
        )
        self.registry.register_template(ChatSession, session_template)
        
        # 线程模板
        thread_template = ModelTemplate(
            name="default_thread",
            model_class=Thread,
            default_values={
                'is_active': True
            },
            required_fields=['session_id', 'title'],
            computed_fields={
                'id': lambda: str(uuid.uuid4()),
                'created_at': lambda: datetime.now(timezone.utc)
            }
        )
        self.registry.register_template(Thread, thread_template)
        
        # 消息模板
        message_template = ModelTemplate(
            name="default_message",
            model_class=Message,
            required_fields=['thread_id', 'content', 'role'],
            computed_fields={
                'id': lambda: str(uuid.uuid4()),
                'created_at': lambda: datetime.now(timezone.utc)
            }
        )
        self.registry.register_template(Message, message_template)
        
        # 工作流模板
        workflow_template = ModelTemplate(
            name="default_workflow",
            model_class=Workflow,
            default_values={
                'is_active': True,
                'version': '1.0.0'
            },
            required_fields=['name', 'definition'],
            computed_fields={
                'id': lambda: str(uuid.uuid4()),
                'created_at': lambda: datetime.now(timezone.utc)
            }
        )
        self.registry.register_template(Workflow, workflow_template)
    
    def create(self, model_class: Type, context: Optional[CreationContext] = None, **kwargs) -> Any:
        """创建模型实例"""
        return self.registry.create_instance(model_class, context, **kwargs)
    
    def create_user(self, username: str, email: str, **kwargs) -> User:
        """创建用户"""
        context = CreationContext(strategy=CreationStrategy.SECURED)
        return self.create(User, context, username=username, email=email, **kwargs)
    
    def create_session(self, user_id: str, **kwargs) -> ChatSession:
        """创建会话"""
        context = CreationContext(user_id=user_id, strategy=CreationStrategy.SECURED)
        return self.create(ChatSession, context, user_id=user_id, **kwargs)
    
    def create_thread(self, session_id: str, title: str, **kwargs) -> Thread:
        """创建线程"""
        context = CreationContext(session_id=session_id, strategy=CreationStrategy.VALIDATED)
        return self.create(Thread, context, session_id=session_id, title=title, **kwargs)
    
    def create_message(self, thread_id: str, content: str, role: str, **kwargs) -> Message:
        """创建消息"""
        context = CreationContext(thread_id=thread_id, strategy=CreationStrategy.VALIDATED)
        return self.create(Message, context, thread_id=thread_id, content=content, role=role, **kwargs)
    
    def create_workflow(self, name: str, definition: Dict[str, Any], **kwargs) -> Workflow:
        """创建工作流"""
        context = CreationContext(strategy=CreationStrategy.SECURED)
        return self.create(Workflow, context, name=name, definition=definition, **kwargs)
    
    def create_from_template(self, model_class: Type, template_name: str, 
                           context: Optional[CreationContext] = None, **kwargs) -> Any:
        """从模板创建"""
        return self.registry.create_from_template(model_class, template_name, context, **kwargs)
    
    def register_factory(self, model_class: Type, factory: BaseFactory, 
                        default_strategy: CreationStrategy = CreationStrategy.SIMPLE) -> None:
        """注册工厂"""
        self.registry.register_factory(model_class, factory, default_strategy)
    
    def register_template(self, model_class: Type, template: ModelTemplate) -> None:
        """注册模板"""
        self.registry.register_template(model_class, template)
    
    def clear_caches(self) -> None:
        """清除缓存"""
        self.registry.clear_all_caches()


# 工厂装饰器
def with_factory(model_class: Type, strategy: CreationStrategy = CreationStrategy.SIMPLE):
    """工厂装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 创建上下文
            context = CreationContext(strategy=strategy)
            
            # 执行原函数
            result = func(*args, **kwargs)
            
            # 如果返回字典，则创建模型实例
            if isinstance(result, dict):
                return model_factory.create(model_class, context, **result)
            
            return result
        
        return wrapper
    return decorator


def create_with_context(context: CreationContext):
    """上下文创建装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 将上下文添加到关键字参数
            kwargs['context'] = context
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def track_creation(model_class: Type):
    """跟踪创建装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now(timezone.utc)
            
            try:
                result = func(*args, **kwargs)
                
                # 发布成功事件
                emit_business_event(
                    EventType.MODEL_AFTER_INSERT,
                    "model_creation_tracked",
                    data={
                        'model': model_class.__name__,
                        'function': func.__name__,
                        'duration': (datetime.now(timezone.utc) - start_time).total_seconds(),
                        'success': True
                    }
                )
                
                return result
                
            except Exception as e:
                # 发布失败事件
                emit_business_event(
                    EventType.MODEL_AFTER_INSERT,
                    "model_creation_failed",
                    data={
                        'model': model_class.__name__,
                        'function': func.__name__,
                        'duration': (datetime.now(timezone.utc) - start_time).total_seconds(),
                        'error': str(e),
                        'success': False
                    }
                )
                
                raise
        
        return wrapper
    return decorator


# 全局模型工厂
model_factory = ModelFactory()


# 便捷函数
def create_model(model_class: Type, **kwargs) -> Any:
    """创建模型实例"""
    return model_factory.create(model_class, **kwargs)


def create_user(username: str, email: str, **kwargs) -> User:
    """创建用户"""
    return model_factory.create_user(username, email, **kwargs)


def create_session(user_id: str, **kwargs) -> ChatSession:
    """创建会话"""
    return model_factory.create_session(user_id, **kwargs)


def create_thread(session_id: str, title: str, **kwargs) -> Thread:
    """创建线程"""
    return model_factory.create_thread(session_id, title, **kwargs)


def create_message(thread_id: str, content: str, role: str, **kwargs) -> Message:
    """创建消息"""
    return model_factory.create_message(thread_id, content, role, **kwargs)


def create_workflow(name: str, definition: Dict[str, Any], **kwargs) -> Workflow:
    """创建工作流"""
    return model_factory.create_workflow(name, definition, **kwargs)


# 导出所有类和函数
__all__ = [
    "CreationStrategy",
    "InitializationMode",
    "FactoryScope",
    "CreationContext",
    "ModelTemplate",
    "BaseFactory",
    "SimpleFactory",
    "ValidatedFactory",
    "SecuredFactory",
    "BulkFactory",
    "FactoryRegistry",
    "ModelFactory",
    "with_factory",
    "create_with_context",
    "track_creation",
    "create_model",
    "create_user",
    "create_session",
    "create_thread",
    "create_message",
    "create_workflow",
    "model_factory"
]