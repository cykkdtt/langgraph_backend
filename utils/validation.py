"""数据验证工具

提供各种数据验证功能和自定义异常类。
"""

import re
import json
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from datetime import datetime
from email_validator import validate_email as email_validate, EmailNotValidError
from jsonschema import validate, ValidationError as JsonValidationError
from pydantic import BaseModel, ValidationError as PydanticValidationError


class ValidationException(Exception):
    """数据验证异常"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error": "validation_error",
            "message": self.message,
            "field": self.field,
            "value": str(self.value) if self.value is not None else None
        }


class BusinessRuleException(Exception):
    """业务规则异常"""
    
    def __init__(self, message: str, rule: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.rule = rule
        self.context = context or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error": "business_rule_error",
            "message": self.message,
            "rule": self.rule,
            "context": self.context
        }


class PermissionDeniedException(Exception):
    """权限拒绝异常"""
    
    def __init__(self, message: str, resource: Optional[str] = None, action: Optional[str] = None):
        self.message = message
        self.resource = resource
        self.action = action
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error": "permission_denied",
            "message": self.message,
            "resource": self.resource,
            "action": self.action
        }


class DataValidator:
    """数据验证器"""
    
    # 常用正则表达式
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PASSWORD_PATTERN = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]{3,30}$')
    PHONE_PATTERN = re.compile(r'^\+?1?\d{9,15}$')
    URL_PATTERN = re.compile(r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$')
    
    @staticmethod
    def validate_required(value: Any, field_name: str) -> Any:
        """验证必填字段"""
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValidationException(f"{field_name} is required", field_name, value)
        return value
    
    @staticmethod
    def validate_string_length(value: str, field_name: str, min_length: int = 0, max_length: int = None) -> str:
        """验证字符串长度"""
        if not isinstance(value, str):
            raise ValidationException(f"{field_name} must be a string", field_name, value)
        
        length = len(value)
        if length < min_length:
            raise ValidationException(
                f"{field_name} must be at least {min_length} characters long",
                field_name, value
            )
        
        if max_length is not None and length > max_length:
            raise ValidationException(
                f"{field_name} must be no more than {max_length} characters long",
                field_name, value
            )
        
        return value
    
    @staticmethod
    def validate_number_range(value: Union[int, float], field_name: str, 
                            min_value: Union[int, float] = None, 
                            max_value: Union[int, float] = None) -> Union[int, float]:
        """验证数字范围"""
        if not isinstance(value, (int, float)):
            raise ValidationException(f"{field_name} must be a number", field_name, value)
        
        if min_value is not None and value < min_value:
            raise ValidationException(
                f"{field_name} must be at least {min_value}",
                field_name, value
            )
        
        if max_value is not None and value > max_value:
            raise ValidationException(
                f"{field_name} must be no more than {max_value}",
                field_name, value
            )
        
        return value
    
    @staticmethod
    def validate_choice(value: Any, field_name: str, choices: List[Any]) -> Any:
        """验证选择值"""
        if value not in choices:
            raise ValidationException(
                f"{field_name} must be one of: {', '.join(map(str, choices))}",
                field_name, value
            )
        return value
    
    @staticmethod
    def validate_pattern(value: str, field_name: str, pattern: re.Pattern, 
                        error_message: str = None) -> str:
        """验证正则表达式模式"""
        if not isinstance(value, str):
            raise ValidationException(f"{field_name} must be a string", field_name, value)
        
        if not pattern.match(value):
            message = error_message or f"{field_name} format is invalid"
            raise ValidationException(message, field_name, value)
        
        return value
    
    @classmethod
    def validate_email(cls, email: str, field_name: str = "email") -> str:
        """验证邮箱地址"""
        if not isinstance(email, str):
            raise ValidationException(f"{field_name} must be a string", field_name, email)
        
        try:
            # 使用email-validator库进行验证
            validated_email = email_validate(email)
            return validated_email.email
        except EmailNotValidError as e:
            raise ValidationException(f"Invalid {field_name}: {str(e)}", field_name, email)
    
    @classmethod
    def validate_password(cls, password: str, field_name: str = "password") -> str:
        """验证密码强度"""
        if not isinstance(password, str):
            raise ValidationException(f"{field_name} must be a string", field_name, password)
        
        # 检查长度
        if len(password) < 8:
            raise ValidationException(
                f"{field_name} must be at least 8 characters long",
                field_name, password
            )
        
        if len(password) > 128:
            raise ValidationException(
                f"{field_name} must be no more than 128 characters long",
                field_name, password
            )
        
        # 检查复杂性
        if not re.search(r'[a-z]', password):
            raise ValidationException(
                f"{field_name} must contain at least one lowercase letter",
                field_name, password
            )
        
        if not re.search(r'[A-Z]', password):
            raise ValidationException(
                f"{field_name} must contain at least one uppercase letter",
                field_name, password
            )
        
        if not re.search(r'\d', password):
            raise ValidationException(
                f"{field_name} must contain at least one digit",
                field_name, password
            )
        
        if not re.search(r'[@$!%*?&]', password):
            raise ValidationException(
                f"{field_name} must contain at least one special character (@$!%*?&)",
                field_name, password
            )
        
        return password
    
    @classmethod
    def validate_username(cls, username: str, field_name: str = "username") -> str:
        """验证用户名"""
        return cls.validate_pattern(
            username, field_name, cls.USERNAME_PATTERN,
            f"{field_name} must be 3-30 characters long and contain only letters, numbers, and underscores"
        )
    
    @classmethod
    def validate_phone(cls, phone: str, field_name: str = "phone") -> str:
        """验证电话号码"""
        return cls.validate_pattern(
            phone, field_name, cls.PHONE_PATTERN,
            f"{field_name} format is invalid"
        )
    
    @classmethod
    def validate_url(cls, url: str, field_name: str = "url") -> str:
        """验证URL"""
        return cls.validate_pattern(
            url, field_name, cls.URL_PATTERN,
            f"{field_name} must be a valid URL"
        )
    
    @staticmethod
    def validate_uuid(value: Union[str, UUID], field_name: str = "id") -> UUID:
        """验证UUID"""
        if isinstance(value, UUID):
            return value
        
        if not isinstance(value, str):
            raise ValidationException(f"{field_name} must be a valid UUID string", field_name, value)
        
        try:
            return UUID(value)
        except ValueError:
            raise ValidationException(f"{field_name} must be a valid UUID", field_name, value)
    
    @staticmethod
    def validate_datetime(value: Union[str, datetime], field_name: str = "datetime") -> datetime:
        """验证日期时间"""
        if isinstance(value, datetime):
            return value
        
        if not isinstance(value, str):
            raise ValidationException(f"{field_name} must be a valid datetime string", field_name, value)
        
        try:
            # 尝试解析ISO格式
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            try:
                # 尝试解析其他常见格式
                return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                raise ValidationException(f"{field_name} must be a valid datetime", field_name, value)
    
    @staticmethod
    def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any], field_name: str = "data") -> Dict[str, Any]:
        """验证JSON Schema"""
        try:
            validate(instance=data, schema=schema)
            return data
        except JsonValidationError as e:
            raise ValidationException(
                f"{field_name} schema validation failed: {e.message}",
                field_name, data
            )
    
    @staticmethod
    def validate_pydantic_model(data: Dict[str, Any], model_class: type, field_name: str = "data") -> BaseModel:
        """验证Pydantic模型"""
        try:
            return model_class(**data)
        except PydanticValidationError as e:
            errors = []
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                errors.append(f"{field}: {error['msg']}")
            
            raise ValidationException(
                f"{field_name} validation failed: {'; '.join(errors)}",
                field_name, data
            )


class BusinessRuleValidator:
    """业务规则验证器"""
    
    @staticmethod
    def validate_user_permissions(user_id: str, resource: str, action: str, 
                                permissions: List[str]) -> None:
        """验证用户权限"""
        required_permission = f"{resource}:{action}"
        if required_permission not in permissions and "admin:all" not in permissions:
            raise PermissionDeniedException(
                f"User does not have permission to {action} {resource}",
                resource, action
            )
    
    @staticmethod
    def validate_rate_limit(user_id: str, action: str, current_count: int, limit: int) -> None:
        """验证速率限制"""
        if current_count >= limit:
            raise BusinessRuleException(
                f"Rate limit exceeded for {action}",
                "rate_limit",
                {"user_id": user_id, "action": action, "current_count": current_count, "limit": limit}
            )
    
    @staticmethod
    def validate_resource_ownership(user_id: str, resource_owner_id: str, resource_type: str) -> None:
        """验证资源所有权"""
        if user_id != resource_owner_id:
            raise PermissionDeniedException(
                f"User does not own this {resource_type}",
                resource_type, "access"
            )
    
    @staticmethod
    def validate_session_active(session_id: str, is_active: bool) -> None:
        """验证会话是否活跃"""
        if not is_active:
            raise BusinessRuleException(
                "Session is not active",
                "session_active",
                {"session_id": session_id}
            )
    
    @staticmethod
    def validate_workflow_execution_limit(user_id: str, current_executions: int, limit: int) -> None:
        """验证工作流执行限制"""
        if current_executions >= limit:
            raise BusinessRuleException(
                "Workflow execution limit exceeded",
                "execution_limit",
                {"user_id": user_id, "current_executions": current_executions, "limit": limit}
            )
    
    @staticmethod
    def validate_memory_storage_limit(user_id: str, current_size: int, limit: int) -> None:
        """验证记忆存储限制"""
        if current_size >= limit:
            raise BusinessRuleException(
                "Memory storage limit exceeded",
                "storage_limit",
                {"user_id": user_id, "current_size": current_size, "limit": limit}
            )


# 便捷函数
def validate_email(email: str, field_name: str = "email") -> str:
    """验证邮箱地址"""
    return DataValidator.validate_email(email, field_name)


def validate_password(password: str, field_name: str = "password") -> str:
    """验证密码强度"""
    return DataValidator.validate_password(password, field_name)


def validate_uuid(value: Union[str, UUID], field_name: str = "id") -> UUID:
    """验证UUID"""
    return DataValidator.validate_uuid(value, field_name)


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any], field_name: str = "data") -> Dict[str, Any]:
    """验证JSON Schema"""
    return DataValidator.validate_json_schema(data, schema, field_name)


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
    """验证必填字段"""
    for field in required_fields:
        if field not in data or data[field] is None:
            raise ValidationException(f"{field} is required", field, data.get(field))
    return data


def validate_model_data(data: Dict[str, Any], model_class: type, field_name: str = "data") -> BaseModel:
    """验证模型数据"""
    return DataValidator.validate_pydantic_model(data, model_class, field_name)