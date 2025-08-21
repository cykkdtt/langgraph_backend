"""模型验证系统模块

本模块提供数据验证、业务规则检查和约束验证功能。
"""

from typing import (
    Dict, Any, Optional, List, Union, Callable, Type, 
    TypeVar, Generic, Tuple, Set, Pattern
)
from datetime import datetime, date, time
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps
import re
import json
import logging
import threading
from decimal import Decimal
from uuid import UUID
from email_validator import validate_email, EmailNotValidError
from sqlalchemy.orm import Session
from sqlalchemy import inspect

# 导入项目模型
from .models import (
    User, Session as ChatSession, Thread, Message, Workflow, 
    WorkflowExecution, WorkflowStep, Memory, MemoryVector, 
    TimeTravel, Attachment, UserPreference, SystemConfig
)
from .events import EventType, emit_business_event


logger = logging.getLogger(__name__)


T = TypeVar('T')


class ValidationType(Enum):
    """验证类型枚举"""
    REQUIRED = "required"              # 必填
    TYPE = "type"                      # 类型
    FORMAT = "format"                  # 格式
    RANGE = "range"                    # 范围
    LENGTH = "length"                  # 长度
    PATTERN = "pattern"                # 模式
    CUSTOM = "custom"                  # 自定义
    BUSINESS = "business"              # 业务规则
    CONSTRAINT = "constraint"          # 约束
    DEPENDENCY = "dependency"          # 依赖


class ValidationLevel(Enum):
    """验证级别枚举"""
    STRICT = "strict"                  # 严格
    NORMAL = "normal"                  # 正常
    LOOSE = "loose"                    # 宽松
    DISABLED = "disabled"              # 禁用


class ValidationScope(Enum):
    """验证范围枚举"""
    FIELD = "field"                    # 字段
    OBJECT = "object"                  # 对象
    COLLECTION = "collection"          # 集合
    CROSS_OBJECT = "cross_object"      # 跨对象
    GLOBAL = "global"                  # 全局


@dataclass
class ValidationError:
    """验证错误"""
    field: str
    message: str
    code: str
    value: Any = None
    validation_type: ValidationType = ValidationType.CUSTOM
    severity: str = "error"  # error, warning, info
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'field': self.field,
            'message': self.message,
            'code': self.code,
            'value': self.value,
            'validation_type': self.validation_type.value,
            'severity': self.severity,
            'context': self.context
        }


@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, field: str, message: str, code: str, 
                 value: Any = None, validation_type: ValidationType = ValidationType.CUSTOM,
                 context: Optional[Dict[str, Any]] = None) -> None:
        """添加错误"""
        error = ValidationError(
            field=field,
            message=message,
            code=code,
            value=value,
            validation_type=validation_type,
            severity="error",
            context=context or {}
        )
        self.errors.append(error)
        self.valid = False
    
    def add_warning(self, field: str, message: str, code: str,
                   value: Any = None, validation_type: ValidationType = ValidationType.CUSTOM,
                   context: Optional[Dict[str, Any]] = None) -> None:
        """添加警告"""
        warning = ValidationError(
            field=field,
            message=message,
            code=code,
            value=value,
            validation_type=validation_type,
            severity="warning",
            context=context or {}
        )
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """是否有错误"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """是否有警告"""
        return len(self.warnings) > 0
    
    def get_error_messages(self) -> List[str]:
        """获取错误消息"""
        return [error.message for error in self.errors]
    
    def get_warning_messages(self) -> List[str]:
        """获取警告消息"""
        return [warning.message for warning in self.warnings]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'valid': self.valid,
            'errors': [error.to_dict() for error in self.errors],
            'warnings': [warning.to_dict() for warning in self.warnings],
            'metadata': self.metadata
        }


class BaseValidator(ABC):
    """验证器基类"""
    
    def __init__(self, field: str, message: Optional[str] = None):
        self.field = field
        self.message = message
    
    @abstractmethod
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """验证值"""
        pass
    
    def get_error_message(self, default_message: str) -> str:
        """获取错误消息"""
        return self.message or default_message


class RequiredValidator(BaseValidator):
    """必填验证器"""
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        if value is None or (isinstance(value, str) and value.strip() == ""):
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} is required"),
                "required",
                value,
                ValidationType.REQUIRED
            )
        
        return result


class TypeValidator(BaseValidator):
    """类型验证器"""
    
    def __init__(self, field: str, expected_type: Type, message: Optional[str] = None):
        super().__init__(field, message)
        self.expected_type = expected_type
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        if value is not None and not isinstance(value, self.expected_type):
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} must be of type {self.expected_type.__name__}"),
                "invalid_type",
                value,
                ValidationType.TYPE
            )
        
        return result


class LengthValidator(BaseValidator):
    """长度验证器"""
    
    def __init__(self, field: str, min_length: Optional[int] = None, 
                 max_length: Optional[int] = None, message: Optional[str] = None):
        super().__init__(field, message)
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        if value is None:
            return result
        
        try:
            length = len(value)
            
            if self.min_length is not None and length < self.min_length:
                result.add_error(
                    self.field,
                    self.get_error_message(f"{self.field} must be at least {self.min_length} characters long"),
                    "min_length",
                    value,
                    ValidationType.LENGTH
                )
            
            if self.max_length is not None and length > self.max_length:
                result.add_error(
                    self.field,
                    self.get_error_message(f"{self.field} must be at most {self.max_length} characters long"),
                    "max_length",
                    value,
                    ValidationType.LENGTH
                )
        
        except TypeError:
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} does not support length validation"),
                "no_length",
                value,
                ValidationType.LENGTH
            )
        
        return result


class RangeValidator(BaseValidator):
    """范围验证器"""
    
    def __init__(self, field: str, min_value: Optional[Union[int, float, Decimal]] = None,
                 max_value: Optional[Union[int, float, Decimal]] = None, 
                 message: Optional[str] = None):
        super().__init__(field, message)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        if value is None:
            return result
        
        try:
            if self.min_value is not None and value < self.min_value:
                result.add_error(
                    self.field,
                    self.get_error_message(f"{self.field} must be at least {self.min_value}"),
                    "min_value",
                    value,
                    ValidationType.RANGE
                )
            
            if self.max_value is not None and value > self.max_value:
                result.add_error(
                    self.field,
                    self.get_error_message(f"{self.field} must be at most {self.max_value}"),
                    "max_value",
                    value,
                    ValidationType.RANGE
                )
        
        except TypeError:
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} does not support range validation"),
                "no_comparison",
                value,
                ValidationType.RANGE
            )
        
        return result


class PatternValidator(BaseValidator):
    """模式验证器"""
    
    def __init__(self, field: str, pattern: Union[str, Pattern], 
                 message: Optional[str] = None):
        super().__init__(field, message)
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        if value is None:
            return result
        
        if not isinstance(value, str):
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} must be a string for pattern validation"),
                "not_string",
                value,
                ValidationType.PATTERN
            )
            return result
        
        if not self.pattern.match(value):
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} does not match the required pattern"),
                "pattern_mismatch",
                value,
                ValidationType.PATTERN
            )
        
        return result


class EmailValidator(BaseValidator):
    """邮箱验证器"""
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        if value is None:
            return result
        
        if not isinstance(value, str):
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} must be a string"),
                "not_string",
                value,
                ValidationType.FORMAT
            )
            return result
        
        try:
            validate_email(value)
        except EmailNotValidError:
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} is not a valid email address"),
                "invalid_email",
                value,
                ValidationType.FORMAT
            )
        
        return result


class URLValidator(BaseValidator):
    """URL验证器"""
    
    def __init__(self, field: str, schemes: Optional[List[str]] = None, 
                 message: Optional[str] = None):
        super().__init__(field, message)
        self.schemes = schemes or ['http', 'https']
        
        # 构建URL正则表达式
        scheme_pattern = '|'.join(self.schemes)
        self.url_pattern = re.compile(
            rf'^({scheme_pattern})://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # 域名
            r'[A-Z]{2,6}\.?|'  # 顶级域名
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP地址
            r'(?::\d+)?'  # 端口
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        if value is None:
            return result
        
        if not isinstance(value, str):
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} must be a string"),
                "not_string",
                value,
                ValidationType.FORMAT
            )
            return result
        
        if not self.url_pattern.match(value):
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} is not a valid URL"),
                "invalid_url",
                value,
                ValidationType.FORMAT
            )
        
        return result


class UUIDValidator(BaseValidator):
    """UUID验证器"""
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        if value is None:
            return result
        
        if isinstance(value, UUID):
            return result
        
        if not isinstance(value, str):
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} must be a string or UUID"),
                "not_string_or_uuid",
                value,
                ValidationType.FORMAT
            )
            return result
        
        try:
            UUID(value)
        except ValueError:
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} is not a valid UUID"),
                "invalid_uuid",
                value,
                ValidationType.FORMAT
            )
        
        return result


class DateTimeValidator(BaseValidator):
    """日期时间验证器"""
    
    def __init__(self, field: str, min_date: Optional[datetime] = None,
                 max_date: Optional[datetime] = None, message: Optional[str] = None):
        super().__init__(field, message)
        self.min_date = min_date
        self.max_date = max_date
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        if value is None:
            return result
        
        if not isinstance(value, (datetime, date)):
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} must be a datetime or date"),
                "not_datetime",
                value,
                ValidationType.TYPE
            )
            return result
        
        # 转换为datetime进行比较
        if isinstance(value, date) and not isinstance(value, datetime):
            value = datetime.combine(value, time.min)
        
        if self.min_date and value < self.min_date:
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} must be after {self.min_date}"),
                "min_date",
                value,
                ValidationType.RANGE
            )
        
        if self.max_date and value > self.max_date:
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} must be before {self.max_date}"),
                "max_date",
                value,
                ValidationType.RANGE
            )
        
        return result


class ChoiceValidator(BaseValidator):
    """选择验证器"""
    
    def __init__(self, field: str, choices: List[Any], message: Optional[str] = None):
        super().__init__(field, message)
        self.choices = choices
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        if value is None:
            return result
        
        if value not in self.choices:
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} must be one of {self.choices}"),
                "invalid_choice",
                value,
                ValidationType.CONSTRAINT
            )
        
        return result


class CustomValidator(BaseValidator):
    """自定义验证器"""
    
    def __init__(self, field: str, validator_func: Callable[[Any], bool],
                 message: Optional[str] = None, error_code: str = "custom_validation"):
        super().__init__(field, message)
        self.validator_func = validator_func
        self.error_code = error_code
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        try:
            if not self.validator_func(value):
                result.add_error(
                    self.field,
                    self.get_error_message(f"{self.field} failed custom validation"),
                    self.error_code,
                    value,
                    ValidationType.CUSTOM
                )
        except Exception as e:
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} validation error: {str(e)}"),
                "validation_exception",
                value,
                ValidationType.CUSTOM
            )
        
        return result


class BusinessRuleValidator(BaseValidator):
    """业务规则验证器"""
    
    def __init__(self, field: str, rule_func: Callable[[Any, Dict[str, Any]], bool],
                 message: Optional[str] = None, error_code: str = "business_rule"):
        super().__init__(field, message)
        self.rule_func = rule_func
        self.error_code = error_code
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(valid=True)
        
        try:
            if not self.rule_func(value, context or {}):
                result.add_error(
                    self.field,
                    self.get_error_message(f"{self.field} violates business rule"),
                    self.error_code,
                    value,
                    ValidationType.BUSINESS
                )
        except Exception as e:
            result.add_error(
                self.field,
                self.get_error_message(f"{self.field} business rule error: {str(e)}"),
                "business_rule_exception",
                value,
                ValidationType.BUSINESS
            )
        
        return result


class ValidationSchema:
    """验证模式"""
    
    def __init__(self, name: str):
        self.name = name
        self.field_validators: Dict[str, List[BaseValidator]] = {}
        self.object_validators: List[Callable[[Dict[str, Any]], ValidationResult]] = []
        self.level = ValidationLevel.NORMAL
        self.scope = ValidationScope.OBJECT
    
    def add_field_validator(self, field: str, validator: BaseValidator) -> 'ValidationSchema':
        """添加字段验证器"""
        if field not in self.field_validators:
            self.field_validators[field] = []
        self.field_validators[field].append(validator)
        return self
    
    def add_object_validator(self, validator: Callable[[Dict[str, Any]], ValidationResult]) -> 'ValidationSchema':
        """添加对象验证器"""
        self.object_validators.append(validator)
        return self
    
    def required(self, field: str, message: Optional[str] = None) -> 'ValidationSchema':
        """添加必填验证"""
        return self.add_field_validator(field, RequiredValidator(field, message))
    
    def type_check(self, field: str, expected_type: Type, message: Optional[str] = None) -> 'ValidationSchema':
        """添加类型验证"""
        return self.add_field_validator(field, TypeValidator(field, expected_type, message))
    
    def length(self, field: str, min_length: Optional[int] = None, 
              max_length: Optional[int] = None, message: Optional[str] = None) -> 'ValidationSchema':
        """添加长度验证"""
        return self.add_field_validator(field, LengthValidator(field, min_length, max_length, message))
    
    def range_check(self, field: str, min_value: Optional[Union[int, float, Decimal]] = None,
                   max_value: Optional[Union[int, float, Decimal]] = None, 
                   message: Optional[str] = None) -> 'ValidationSchema':
        """添加范围验证"""
        return self.add_field_validator(field, RangeValidator(field, min_value, max_value, message))
    
    def pattern(self, field: str, pattern: Union[str, Pattern], 
               message: Optional[str] = None) -> 'ValidationSchema':
        """添加模式验证"""
        return self.add_field_validator(field, PatternValidator(field, pattern, message))
    
    def email(self, field: str, message: Optional[str] = None) -> 'ValidationSchema':
        """添加邮箱验证"""
        return self.add_field_validator(field, EmailValidator(field, message))
    
    def url(self, field: str, schemes: Optional[List[str]] = None, 
           message: Optional[str] = None) -> 'ValidationSchema':
        """添加URL验证"""
        return self.add_field_validator(field, URLValidator(field, schemes, message))
    
    def uuid(self, field: str, message: Optional[str] = None) -> 'ValidationSchema':
        """添加UUID验证"""
        return self.add_field_validator(field, UUIDValidator(field, message))
    
    def datetime_check(self, field: str, min_date: Optional[datetime] = None,
                      max_date: Optional[datetime] = None, 
                      message: Optional[str] = None) -> 'ValidationSchema':
        """添加日期时间验证"""
        return self.add_field_validator(field, DateTimeValidator(field, min_date, max_date, message))
    
    def choice(self, field: str, choices: List[Any], 
              message: Optional[str] = None) -> 'ValidationSchema':
        """添加选择验证"""
        return self.add_field_validator(field, ChoiceValidator(field, choices, message))
    
    def custom(self, field: str, validator_func: Callable[[Any], bool],
              message: Optional[str] = None, error_code: str = "custom_validation") -> 'ValidationSchema':
        """添加自定义验证"""
        return self.add_field_validator(field, CustomValidator(field, validator_func, message, error_code))
    
    def business_rule(self, field: str, rule_func: Callable[[Any, Dict[str, Any]], bool],
                     message: Optional[str] = None, error_code: str = "business_rule") -> 'ValidationSchema':
        """添加业务规则验证"""
        return self.add_field_validator(field, BusinessRuleValidator(field, rule_func, message, error_code))
    
    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """验证数据"""
        result = ValidationResult(valid=True)
        context = context or {}
        
        # 字段级验证
        for field, validators in self.field_validators.items():
            value = data.get(field)
            
            for validator in validators:
                field_result = validator.validate(value, context)
                
                # 合并结果
                result.errors.extend(field_result.errors)
                result.warnings.extend(field_result.warnings)
                
                if field_result.has_errors():
                    result.valid = False
        
        # 对象级验证
        for validator in self.object_validators:
            try:
                object_result = validator(data)
                
                # 合并结果
                result.errors.extend(object_result.errors)
                result.warnings.extend(object_result.warnings)
                
                if object_result.has_errors():
                    result.valid = False
            
            except Exception as e:
                result.add_error(
                    "object",
                    f"Object validation error: {str(e)}",
                    "object_validation_exception",
                    data,
                    ValidationType.CUSTOM
                )
        
        return result


class ValidationRegistry:
    """验证注册表"""
    
    def __init__(self):
        self._schemas: Dict[str, ValidationSchema] = {}
        self._model_schemas: Dict[Type, ValidationSchema] = {}
        self._lock = threading.RLock()
    
    def register_schema(self, name: str, schema: ValidationSchema) -> None:
        """注册验证模式"""
        with self._lock:
            self._schemas[name] = schema
    
    def register_model_schema(self, model_class: Type, schema: ValidationSchema) -> None:
        """注册模型验证模式"""
        with self._lock:
            self._model_schemas[model_class] = schema
    
    def get_schema(self, name: str) -> Optional[ValidationSchema]:
        """获取验证模式"""
        return self._schemas.get(name)
    
    def get_model_schema(self, model_class: Type) -> Optional[ValidationSchema]:
        """获取模型验证模式"""
        return self._model_schemas.get(model_class)
    
    def list_schemas(self) -> List[str]:
        """列出所有模式名称"""
        return list(self._schemas.keys())
    
    def remove_schema(self, name: str) -> bool:
        """移除验证模式"""
        with self._lock:
            if name in self._schemas:
                del self._schemas[name]
                return True
            return False


class ValidationManager:
    """验证管理器"""
    
    def __init__(self):
        self.registry = ValidationRegistry()
        self.level = ValidationLevel.NORMAL
        self._initialize_default_schemas()
    
    def _initialize_default_schemas(self) -> None:
        """初始化默认验证模式"""
        # 用户验证模式
        user_schema = ValidationSchema("user")
        user_schema.required("username").length("username", 3, 50)
        user_schema.required("email").email("email")
        user_schema.required("password").length("password", 8, 128)
        user_schema.type_check("is_active", bool)
        user_schema.datetime_check("created_at")
        
        self.registry.register_schema("user", user_schema)
        self.registry.register_model_schema(User, user_schema)
        
        # 会话验证模式
        session_schema = ValidationSchema("session")
        session_schema.required("user_id").uuid("user_id")
        session_schema.length("title", 1, 200)
        session_schema.datetime_check("created_at")
        session_schema.datetime_check("updated_at")
        
        self.registry.register_schema("session", session_schema)
        self.registry.register_model_schema(ChatSession, session_schema)
        
        # 线程验证模式
        thread_schema = ValidationSchema("thread")
        thread_schema.required("session_id").uuid("session_id")
        thread_schema.length("title", 1, 200)
        thread_schema.datetime_check("created_at")
        thread_schema.datetime_check("updated_at")
        
        self.registry.register_schema("thread", thread_schema)
        self.registry.register_model_schema(Thread, thread_schema)
        
        # 消息验证模式
        message_schema = ValidationSchema("message")
        message_schema.required("thread_id").uuid("thread_id")
        message_schema.required("content").length("content", 1, 10000)
        message_schema.required("role").choice("role", ["user", "assistant", "system"])
        message_schema.datetime_check("created_at")
        
        self.registry.register_schema("message", message_schema)
        self.registry.register_model_schema(Message, message_schema)
        
        # 工作流验证模式
        workflow_schema = ValidationSchema("workflow")
        workflow_schema.required("name").length("name", 1, 100)
        workflow_schema.required("user_id").uuid("user_id")
        workflow_schema.type_check("is_active", bool)
        workflow_schema.datetime_check("created_at")
        
        self.registry.register_schema("workflow", workflow_schema)
        self.registry.register_model_schema(Workflow, workflow_schema)
    
    def validate(self, data: Dict[str, Any], schema_name: str, 
                context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """使用指定模式验证数据"""
        schema = self.registry.get_schema(schema_name)
        if not schema:
            result = ValidationResult(valid=False)
            result.add_error(
                "schema",
                f"Validation schema '{schema_name}' not found",
                "schema_not_found"
            )
            return result
        
        return schema.validate(data, context)
    
    def validate_model(self, model_instance: Any, 
                      context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """验证模型实例"""
        model_class = type(model_instance)
        schema = self.registry.get_model_schema(model_class)
        
        if not schema:
            result = ValidationResult(valid=False)
            result.add_error(
                "schema",
                f"No validation schema found for model {model_class.__name__}",
                "model_schema_not_found"
            )
            return result
        
        # 将模型实例转换为字典
        if hasattr(model_instance, '__dict__'):
            data = {k: v for k, v in model_instance.__dict__.items() if not k.startswith('_')}
        else:
            data = {}
        
        return schema.validate(data, context)
    
    def validate_dict(self, data: Dict[str, Any], model_class: Type,
                     context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """使用模型类验证字典数据"""
        schema = self.registry.get_model_schema(model_class)
        
        if not schema:
            result = ValidationResult(valid=False)
            result.add_error(
                "schema",
                f"No validation schema found for model {model_class.__name__}",
                "model_schema_not_found"
            )
            return result
        
        return schema.validate(data, context)
    
    def create_schema(self, name: str) -> ValidationSchema:
        """创建新的验证模式"""
        schema = ValidationSchema(name)
        self.registry.register_schema(name, schema)
        return schema
    
    def get_schema(self, name: str) -> Optional[ValidationSchema]:
        """获取验证模式"""
        return self.registry.get_schema(name)
    
    def set_validation_level(self, level: ValidationLevel) -> None:
        """设置验证级别"""
        self.level = level


# 验证装饰器
def validate_input(schema_name: str, context_func: Optional[Callable] = None):
    """输入验证装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取验证数据（通常是第一个参数或kwargs中的data）
            if args and isinstance(args[0], dict):
                data = args[0]
            elif 'data' in kwargs:
                data = kwargs['data']
            else:
                # 如果没有找到数据，跳过验证
                return func(*args, **kwargs)
            
            # 获取上下文
            context = {}
            if context_func:
                context = context_func(*args, **kwargs)
            
            # 执行验证
            result = validation_manager.validate(data, schema_name, context)
            
            if not result.valid:
                # 发布验证失败事件
                emit_business_event(
                    EventType.SYSTEM_STARTUP,  # 使用系统事件类型
                    "validation_failed",
                    data={
                        'schema': schema_name,
                        'errors': [error.to_dict() for error in result.errors],
                        'data': data
                    }
                )
                
                # 抛出验证异常
                raise ValueError(f"Validation failed: {', '.join(result.get_error_messages())}")
            
            # 如果有警告，记录日志
            if result.has_warnings():
                logger.warning(f"Validation warnings for {schema_name}: {', '.join(result.get_warning_messages())}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_model_input(model_class: Type, context_func: Optional[Callable] = None):
    """模型输入验证装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取验证数据
            if args and isinstance(args[0], dict):
                data = args[0]
            elif 'data' in kwargs:
                data = kwargs['data']
            else:
                return func(*args, **kwargs)
            
            # 获取上下文
            context = {}
            if context_func:
                context = context_func(*args, **kwargs)
            
            # 执行验证
            result = validation_manager.validate_dict(data, model_class, context)
            
            if not result.valid:
                emit_business_event(
                    EventType.SYSTEM_STARTUP,
                    "model_validation_failed",
                    data={
                        'model': model_class.__name__,
                        'errors': [error.to_dict() for error in result.errors],
                        'data': data
                    }
                )
                
                raise ValueError(f"Model validation failed: {', '.join(result.get_error_messages())}")
            
            if result.has_warnings():
                logger.warning(f"Model validation warnings for {model_class.__name__}: {', '.join(result.get_warning_messages())}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# 全局验证管理器
validation_manager = ValidationManager()


# 便捷函数
def validate_data(data: Dict[str, Any], schema_name: str, 
                 context: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """验证数据"""
    return validation_manager.validate(data, schema_name, context)


def validate_model(model_instance: Any, 
                  context: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """验证模型实例"""
    return validation_manager.validate_model(model_instance, context)


def create_validation_schema(name: str) -> ValidationSchema:
    """创建验证模式"""
    return validation_manager.create_schema(name)


def get_validation_schema(name: str) -> Optional[ValidationSchema]:
    """获取验证模式"""
    return validation_manager.get_schema(name)


# 导出所有类和函数
__all__ = [
    "ValidationType",
    "ValidationLevel",
    "ValidationScope",
    "ValidationError",
    "ValidationResult",
    "BaseValidator",
    "RequiredValidator",
    "TypeValidator",
    "LengthValidator",
    "RangeValidator",
    "PatternValidator",
    "EmailValidator",
    "URLValidator",
    "UUIDValidator",
    "DateTimeValidator",
    "ChoiceValidator",
    "CustomValidator",
    "BusinessRuleValidator",
    "ValidationSchema",
    "ValidationRegistry",
    "ValidationManager",
    "validate_input",
    "validate_model_input",
    "validate_data",
    "validate_model",
    "create_validation_schema",
    "get_validation_schema",
    "validation_manager"
]