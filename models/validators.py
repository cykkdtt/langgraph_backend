"""数据验证工具模块

本模块提供统一的数据验证功能，包括字段验证、业务规则验证、数据完整性检查等。
"""

from typing import List, Optional, Dict, Any, Type, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import re
import json
from functools import wraps
from sqlalchemy.orm import Session
from sqlalchemy import func

from .database import (
    User, Session as DBSession, Thread, Message, Workflow, WorkflowExecution,
    WorkflowStep, Memory, MemoryVector, TimeTravel, Attachment,
    UserPreference, SystemConfig
)


class ValidationLevel(Enum):
    """验证级别枚举"""
    BASIC = "basic"        # 基础验证
    STANDARD = "standard"  # 标准验证
    STRICT = "strict"      # 严格验证
    CUSTOM = "custom"      # 自定义验证


class ValidationErrorType(Enum):
    """验证错误类型枚举"""
    REQUIRED = "required"              # 必填字段
    FORMAT = "format"                  # 格式错误
    LENGTH = "length"                  # 长度错误
    RANGE = "range"                    # 范围错误
    PATTERN = "pattern"                # 模式匹配错误
    UNIQUE = "unique"                  # 唯一性错误
    REFERENCE = "reference"            # 引用错误
    BUSINESS_RULE = "business_rule"    # 业务规则错误
    PERMISSION = "permission"          # 权限错误
    DEPENDENCY = "dependency"          # 依赖错误


@dataclass
class ValidationError:
    """验证错误"""
    field: str
    error_type: ValidationErrorType
    message: str
    value: Any = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'field': self.field,
            'error_type': self.error_type.value,
            'message': self.message,
            'value': self.value,
            'context': self.context
        }


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def add_error(self, field: str, error_type: ValidationErrorType, message: str, 
                  value: Any = None, context: Dict[str, Any] = None) -> None:
        """添加错误"""
        error = ValidationError(field, error_type, message, value, context)
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, field: str, error_type: ValidationErrorType, message: str,
                   value: Any = None, context: Dict[str, Any] = None) -> None:
        """添加警告"""
        warning = ValidationError(field, error_type, message, value, context)
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'is_valid': self.is_valid,
            'errors': [error.to_dict() for error in self.errors],
            'warnings': [warning.to_dict() for warning in self.warnings]
        }


class BaseValidator:
    """基础验证器"""
    
    def __init__(self, session: Optional[Session] = None, level: ValidationLevel = ValidationLevel.STANDARD):
        self.session = session
        self.level = level
        self.result = ValidationResult(is_valid=True, errors=[])
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """执行验证"""
        self.result = ValidationResult(is_valid=True, errors=[])
        
        # 基础验证
        if self.level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            self._validate_required_fields(data)
            self._validate_field_formats(data)
            self._validate_field_lengths(data)
            self._validate_field_ranges(data)
        
        # 标准验证
        if self.level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            self._validate_patterns(data)
            self._validate_references(data, context)
        
        # 严格验证
        if self.level == ValidationLevel.STRICT:
            self._validate_uniqueness(data, context)
            self._validate_business_rules(data, context)
        
        # 自定义验证
        if self.level == ValidationLevel.CUSTOM:
            self._validate_custom_rules(data, context)
        
        return self.result
    
    def _validate_required_fields(self, data: Dict[str, Any]) -> None:
        """验证必填字段"""
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                self.result.add_error(
                    field, ValidationErrorType.REQUIRED, 
                    f"Field '{field}' is required"
                )
    
    def _validate_field_formats(self, data: Dict[str, Any]) -> None:
        """验证字段格式"""
        format_rules = self.get_format_rules()
        for field, rule in format_rules.items():
            if field in data and data[field] is not None:
                if not self._check_format(data[field], rule):
                    self.result.add_error(
                        field, ValidationErrorType.FORMAT,
                        f"Field '{field}' has invalid format",
                        data[field]
                    )
    
    def _validate_field_lengths(self, data: Dict[str, Any]) -> None:
        """验证字段长度"""
        length_rules = self.get_length_rules()
        for field, rule in length_rules.items():
            if field in data and data[field] is not None:
                value = str(data[field])
                if not self._check_length(value, rule):
                    self.result.add_error(
                        field, ValidationErrorType.LENGTH,
                        f"Field '{field}' length is invalid",
                        data[field], {'rule': rule}
                    )
    
    def _validate_field_ranges(self, data: Dict[str, Any]) -> None:
        """验证字段范围"""
        range_rules = self.get_range_rules()
        for field, rule in range_rules.items():
            if field in data and data[field] is not None:
                if not self._check_range(data[field], rule):
                    self.result.add_error(
                        field, ValidationErrorType.RANGE,
                        f"Field '{field}' value is out of range",
                        data[field], {'rule': rule}
                    )
    
    def _validate_patterns(self, data: Dict[str, Any]) -> None:
        """验证模式匹配"""
        pattern_rules = self.get_pattern_rules()
        for field, pattern in pattern_rules.items():
            if field in data and data[field] is not None:
                if not re.match(pattern, str(data[field])):
                    self.result.add_error(
                        field, ValidationErrorType.PATTERN,
                        f"Field '{field}' does not match required pattern",
                        data[field], {'pattern': pattern}
                    )
    
    def _validate_references(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> None:
        """验证引用完整性"""
        if not self.session:
            return
        
        reference_rules = self.get_reference_rules()
        for field, rule in reference_rules.items():
            if field in data and data[field] is not None:
                if not self._check_reference(data[field], rule):
                    self.result.add_error(
                        field, ValidationErrorType.REFERENCE,
                        f"Referenced {rule['model'].__name__} not found",
                        data[field]
                    )
    
    def _validate_uniqueness(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> None:
        """验证唯一性"""
        if not self.session:
            return
        
        unique_rules = self.get_unique_rules()
        for field, rule in unique_rules.items():
            if field in data and data[field] is not None:
                if not self._check_uniqueness(data[field], rule, context):
                    self.result.add_error(
                        field, ValidationErrorType.UNIQUE,
                        f"Field '{field}' value already exists",
                        data[field]
                    )
    
    def _validate_business_rules(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> None:
        """验证业务规则"""
        business_rules = self.get_business_rules()
        for rule_name, rule_func in business_rules.items():
            try:
                if not rule_func(data, context, self.session):
                    self.result.add_error(
                        rule_name, ValidationErrorType.BUSINESS_RULE,
                        f"Business rule '{rule_name}' validation failed"
                    )
            except Exception as e:
                self.result.add_error(
                    rule_name, ValidationErrorType.BUSINESS_RULE,
                    f"Business rule '{rule_name}' validation error: {str(e)}"
                )
    
    def _validate_custom_rules(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> None:
        """验证自定义规则"""
        custom_rules = self.get_custom_rules()
        for rule_name, rule_func in custom_rules.items():
            try:
                rule_result = rule_func(data, context, self.session)
                if isinstance(rule_result, ValidationResult):
                    self.result.errors.extend(rule_result.errors)
                    self.result.warnings.extend(rule_result.warnings)
                    if not rule_result.is_valid:
                        self.result.is_valid = False
                elif not rule_result:
                    self.result.add_error(
                        rule_name, ValidationErrorType.BUSINESS_RULE,
                        f"Custom rule '{rule_name}' validation failed"
                    )
            except Exception as e:
                self.result.add_error(
                    rule_name, ValidationErrorType.BUSINESS_RULE,
                    f"Custom rule '{rule_name}' validation error: {str(e)}"
                )
    
    def _check_format(self, value: Any, rule: Dict[str, Any]) -> bool:
        """检查格式"""
        format_type = rule.get('type')
        
        if format_type == 'email':
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return re.match(pattern, str(value)) is not None
        elif format_type == 'url':
            pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            return re.match(pattern, str(value)) is not None
        elif format_type == 'phone':
            pattern = r'^\+?[1-9]\d{1,14}$'
            return re.match(pattern, str(value)) is not None
        elif format_type == 'datetime':
            try:
                datetime.fromisoformat(str(value).replace('Z', '+00:00'))
                return True
            except ValueError:
                return False
        elif format_type == 'json':
            try:
                json.loads(str(value))
                return True
            except (json.JSONDecodeError, TypeError):
                return False
        
        return True
    
    def _check_length(self, value: str, rule: Dict[str, Any]) -> bool:
        """检查长度"""
        length = len(value)
        min_length = rule.get('min', 0)
        max_length = rule.get('max', float('inf'))
        
        return min_length <= length <= max_length
    
    def _check_range(self, value: Any, rule: Dict[str, Any]) -> bool:
        """检查范围"""
        try:
            num_value = float(value)
            min_value = rule.get('min', float('-inf'))
            max_value = rule.get('max', float('inf'))
            
            return min_value <= num_value <= max_value
        except (ValueError, TypeError):
            return False
    
    def _check_reference(self, value: Any, rule: Dict[str, Any]) -> bool:
        """检查引用"""
        model_class = rule['model']
        field = rule.get('field', 'id')
        
        query = self.session.query(model_class).filter(
            getattr(model_class, field) == value
        )
        
        return query.first() is not None
    
    def _check_uniqueness(self, value: Any, rule: Dict[str, Any], context: Dict[str, Any] = None) -> bool:
        """检查唯一性"""
        model_class = rule['model']
        field = rule['field']
        
        query = self.session.query(model_class).filter(
            getattr(model_class, field) == value
        )
        
        # 如果是更新操作，排除当前记录
        if context and 'id' in context:
            query = query.filter(model_class.id != context['id'])
        
        return query.first() is None
    
    # 抽象方法，子类需要实现
    def get_required_fields(self) -> List[str]:
        """获取必填字段"""
        return []
    
    def get_format_rules(self) -> Dict[str, Dict[str, Any]]:
        """获取格式规则"""
        return {}
    
    def get_length_rules(self) -> Dict[str, Dict[str, Any]]:
        """获取长度规则"""
        return {}
    
    def get_range_rules(self) -> Dict[str, Dict[str, Any]]:
        """获取范围规则"""
        return {}
    
    def get_pattern_rules(self) -> Dict[str, str]:
        """获取模式规则"""
        return {}
    
    def get_reference_rules(self) -> Dict[str, Dict[str, Any]]:
        """获取引用规则"""
        return {}
    
    def get_unique_rules(self) -> Dict[str, Dict[str, Any]]:
        """获取唯一性规则"""
        return {}
    
    def get_business_rules(self) -> Dict[str, Callable]:
        """获取业务规则"""
        return {}
    
    def get_custom_rules(self) -> Dict[str, Callable]:
        """获取自定义规则"""
        return {}


class UserValidator(BaseValidator):
    """用户验证器"""
    
    def get_required_fields(self) -> List[str]:
        return ['username', 'email', 'password']
    
    def get_format_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            'email': {'type': 'email'},
            'website': {'type': 'url'},
        }
    
    def get_length_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            'username': {'min': 3, 'max': 50},
            'password': {'min': 8, 'max': 128},
            'full_name': {'min': 1, 'max': 100},
            'bio': {'max': 500}
        }
    
    def get_pattern_rules(self) -> Dict[str, str]:
        return {
            'username': r'^[a-zA-Z0-9_-]+$',
            'role': r'^(admin|user|moderator)$',
            'status': r'^(active|inactive|suspended|pending)$'
        }
    
    def get_unique_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            'username': {'model': User, 'field': 'username'},
            'email': {'model': User, 'field': 'email'}
        }
    
    def get_business_rules(self) -> Dict[str, Callable]:
        return {
            'password_strength': self._validate_password_strength,
            'email_domain': self._validate_email_domain
        }
    
    def _validate_password_strength(self, data: Dict[str, Any], context: Dict[str, Any], session: Session) -> bool:
        """验证密码强度"""
        password = data.get('password')
        if not password:
            return True  # 密码为空时由必填验证处理
        
        # 检查密码复杂性
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
        
        complexity_count = sum([has_upper, has_lower, has_digit, has_special])
        
        if len(password) >= 12 and complexity_count >= 3:
            return True
        elif len(password) >= 8 and complexity_count >= 4:
            return True
        
        return False
    
    def _validate_email_domain(self, data: Dict[str, Any], context: Dict[str, Any], session: Session) -> bool:
        """验证邮箱域名"""
        email = data.get('email')
        if not email:
            return True
        
        # 检查是否为临时邮箱域名
        temp_domains = ['10minutemail.com', 'tempmail.org', 'guerrillamail.com']
        domain = email.split('@')[-1].lower()
        
        return domain not in temp_domains


class MessageValidator(BaseValidator):
    """消息验证器"""
    
    def get_required_fields(self) -> List[str]:
        return ['thread_id', 'user_id', 'role', 'content']
    
    def get_length_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            'content': {'min': 1, 'max': 10000}
        }
    
    def get_pattern_rules(self) -> Dict[str, str]:
        return {
            'role': r'^(user|assistant|system|function)$',
            'message_type': r'^(text|image|file|audio|video|code|markdown)$'
        }
    
    def get_reference_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            'thread_id': {'model': Thread, 'field': 'id'},
            'user_id': {'model': User, 'field': 'id'},
            'parent_id': {'model': Message, 'field': 'id'},
            'reply_to_id': {'model': Message, 'field': 'id'}
        }
    
    def get_business_rules(self) -> Dict[str, Callable]:
        return {
            'thread_access': self._validate_thread_access,
            'content_safety': self._validate_content_safety
        }
    
    def _validate_thread_access(self, data: Dict[str, Any], context: Dict[str, Any], session: Session) -> bool:
        """验证线程访问权限"""
        thread_id = data.get('thread_id')
        user_id = data.get('user_id')
        
        if not thread_id or not user_id:
            return True
        
        thread = session.query(Thread).filter(Thread.id == thread_id).first()
        if not thread:
            return False
        
        # 检查用户是否有权限访问该线程
        return thread.user_id == user_id or thread.is_public
    
    def _validate_content_safety(self, data: Dict[str, Any], context: Dict[str, Any], session: Session) -> bool:
        """验证内容安全性"""
        content = data.get('content', '')
        
        # 简单的内容安全检查
        unsafe_patterns = [
            r'<script[^>]*>.*?</script>',  # 脚本标签
            r'javascript:',               # JavaScript协议
            r'on\w+\s*=',                # 事件处理器
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False
        
        return True


class WorkflowValidator(BaseValidator):
    """工作流验证器"""
    
    def get_required_fields(self) -> List[str]:
        return ['user_id', 'name', 'definition']
    
    def get_length_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            'name': {'min': 1, 'max': 100},
            'description': {'max': 500}
        }
    
    def get_pattern_rules(self) -> Dict[str, str]:
        return {
            'status': r'^(draft|active|inactive|archived)$'
        }
    
    def get_reference_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            'user_id': {'model': User, 'field': 'id'}
        }
    
    def get_business_rules(self) -> Dict[str, Callable]:
        return {
            'definition_structure': self._validate_definition_structure,
            'workflow_complexity': self._validate_workflow_complexity
        }
    
    def _validate_definition_structure(self, data: Dict[str, Any], context: Dict[str, Any], session: Session) -> bool:
        """验证工作流定义结构"""
        definition = data.get('definition')
        if not definition:
            return True
        
        try:
            if isinstance(definition, str):
                definition = json.loads(definition)
            
            # 检查必要的字段
            required_fields = ['nodes', 'edges', 'start_node']
            for field in required_fields:
                if field not in definition:
                    return False
            
            # 检查节点和边的有效性
            nodes = definition.get('nodes', [])
            edges = definition.get('edges', [])
            start_node = definition.get('start_node')
            
            # 验证开始节点存在
            node_ids = [node.get('id') for node in nodes]
            if start_node not in node_ids:
                return False
            
            # 验证边的连接有效性
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                if source not in node_ids or target not in node_ids:
                    return False
            
            return True
        except (json.JSONDecodeError, TypeError, AttributeError):
            return False
    
    def _validate_workflow_complexity(self, data: Dict[str, Any], context: Dict[str, Any], session: Session) -> bool:
        """验证工作流复杂度"""
        definition = data.get('definition')
        if not definition:
            return True
        
        try:
            if isinstance(definition, str):
                definition = json.loads(definition)
            
            nodes = definition.get('nodes', [])
            edges = definition.get('edges', [])
            
            # 限制节点和边的数量
            max_nodes = 100
            max_edges = 200
            
            if len(nodes) > max_nodes or len(edges) > max_edges:
                return False
            
            return True
        except (json.JSONDecodeError, TypeError):
            return False


class MemoryValidator(BaseValidator):
    """记忆验证器"""
    
    def get_required_fields(self) -> List[str]:
        return ['user_id', 'memory_type', 'content']
    
    def get_length_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            'title': {'min': 1, 'max': 200},
            'content': {'min': 1, 'max': 5000},
            'summary': {'max': 500}
        }
    
    def get_range_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            'importance': {'min': 0.0, 'max': 1.0},
            'weight': {'min': 0.0, 'max': 1.0}
        }
    
    def get_pattern_rules(self) -> Dict[str, str]:
        return {
            'memory_type': r'^(semantic|episodic|procedural|working)$'
        }
    
    def get_reference_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            'user_id': {'model': User, 'field': 'id'},
            'thread_id': {'model': Thread, 'field': 'id'}
        }


class ValidatorFactory:
    """验证器工厂"""
    
    _validators = {
        'user': UserValidator,
        'message': MessageValidator,
        'workflow': WorkflowValidator,
        'memory': MemoryValidator,
    }
    
    @classmethod
    def create(cls, validator_type: str, session: Optional[Session] = None, 
              level: ValidationLevel = ValidationLevel.STANDARD) -> BaseValidator:
        """创建验证器"""
        validator_class = cls._validators.get(validator_type, BaseValidator)
        return validator_class(session, level)
    
    @classmethod
    def register(cls, validator_type: str, validator_class: Type[BaseValidator]) -> None:
        """注册验证器"""
        cls._validators[validator_type] = validator_class


def validate_data(data: Dict[str, Any], validator_type: str, session: Optional[Session] = None,
                 level: ValidationLevel = ValidationLevel.STANDARD, 
                 context: Dict[str, Any] = None) -> ValidationResult:
    """验证数据的便捷函数"""
    validator = ValidatorFactory.create(validator_type, session, level)
    return validator.validate(data, context)


def validation_required(validator_type: str, level: ValidationLevel = ValidationLevel.STANDARD):
    """验证装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 假设第一个参数是数据，第二个参数是session
            if len(args) >= 2:
                data = args[0] if isinstance(args[0], dict) else kwargs.get('data')
                session = args[1] if hasattr(args[1], 'query') else kwargs.get('session')
                
                if data:
                    result = validate_data(data, validator_type, session, level)
                    if not result.is_valid:
                        raise ValueError(f"Validation failed: {result.errors}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 导出所有类和函数
__all__ = [
    "ValidationLevel",
    "ValidationErrorType",
    "ValidationError",
    "ValidationResult",
    "BaseValidator",
    "UserValidator",
    "MessageValidator",
    "WorkflowValidator",
    "MemoryValidator",
    "ValidatorFactory",
    "validate_data",
    "validation_required"
]