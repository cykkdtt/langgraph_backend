#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证模块

提供统一的数据验证功能，包括：
- 数据验证框架
- 模型验证器
- 验证装饰器
- 数据清洗和转换
- 验证规则配置
"""

import re
import json
import yaml
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Callable, Type, TypeVar, Generic,
    Set, Tuple, Pattern, ClassVar, get_type_hints, get_origin, get_args
)
from dataclasses import dataclass, field
from pydantic import BaseModel, ValidationError, validator
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')
ModelType = TypeVar('ModelType', bound=BaseModel)


class ValidationLevel(Enum):
    """验证级别枚举"""
    STRICT = "strict"  # 严格验证
    NORMAL = "normal"  # 正常验证
    LOOSE = "loose"    # 宽松验证
    SKIP = "skip"      # 跳过验证


class ValidationErrorType(Enum):
    """验证错误类型枚举"""
    REQUIRED = "required"          # 必填字段缺失
    TYPE_ERROR = "type_error"      # 类型错误
    VALUE_ERROR = "value_error"    # 值错误
    FORMAT_ERROR = "format_error"  # 格式错误
    LENGTH_ERROR = "length_error"  # 长度错误
    RANGE_ERROR = "range_error"    # 范围错误
    CUSTOM_ERROR = "custom_error"  # 自定义错误


@dataclass
class ValidationError:
    """验证错误信息"""
    field: str
    error_type: ValidationErrorType
    message: str
    value: Any = None
    expected: Any = None
    code: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'field': self.field,
            'error_type': self.error_type.value,
            'message': self.message,
            'value': self.value,
            'expected': self.expected,
            'code': self.code,
            'context': self.context
        }


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: ValidationError) -> None:
        """添加错误"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """添加警告"""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'is_valid': self.is_valid,
            'errors': [error.to_dict() for error in self.errors],
            'warnings': self.warnings,
            'cleaned_data': self.cleaned_data,
            'metadata': self.metadata
        }


class BaseValidator(ABC):
    """验证器基类"""
    
    def __init__(self, name: str, message: Optional[str] = None, 
                 level: ValidationLevel = ValidationLevel.NORMAL):
        self.name = name
        self.message = message
        self.level = level
    
    @abstractmethod
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """执行验证"""
        pass
    
    def __call__(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """使验证器可调用"""
        return self.validate(value, context)


class RequiredValidator(BaseValidator):
    """必填验证器"""
    
    def __init__(self, message: Optional[str] = None):
        super().__init__("required", message or "字段为必填项")
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None or (isinstance(value, str) and not value.strip()):
            result.add_error(ValidationError(
                field=context.get('field', 'unknown') if context else 'unknown',
                error_type=ValidationErrorType.REQUIRED,
                message=self.message,
                value=value
            ))
        
        return result


class TypeValidator(BaseValidator):
    """类型验证器"""
    
    def __init__(self, expected_type: Type, message: Optional[str] = None):
        self.expected_type = expected_type
        super().__init__("type", message or f"字段类型必须为 {expected_type.__name__}")
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is not None and not isinstance(value, self.expected_type):
            result.add_error(ValidationError(
                field=context.get('field', 'unknown') if context else 'unknown',
                error_type=ValidationErrorType.TYPE_ERROR,
                message=self.message,
                value=value,
                expected=self.expected_type.__name__
            ))
        
        return result


class LengthValidator(BaseValidator):
    """长度验证器"""
    
    def __init__(self, min_length: Optional[int] = None, max_length: Optional[int] = None,
                 message: Optional[str] = None):
        self.min_length = min_length
        self.max_length = max_length
        super().__init__("length", message or self._build_message())
    
    def _build_message(self) -> str:
        """构建错误消息"""
        if self.min_length is not None and self.max_length is not None:
            return f"字段长度必须在 {self.min_length} 到 {self.max_length} 之间"
        elif self.min_length is not None:
            return f"字段长度不能少于 {self.min_length}"
        elif self.max_length is not None:
            return f"字段长度不能超过 {self.max_length}"
        return "字段长度无效"
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None:
            return result
        
        try:
            length = len(value)
            
            if self.min_length is not None and length < self.min_length:
                result.add_error(ValidationError(
                    field=context.get('field', 'unknown') if context else 'unknown',
                    error_type=ValidationErrorType.LENGTH_ERROR,
                    message=self.message,
                    value=value,
                    expected=f"min_length: {self.min_length}"
                ))
            
            if self.max_length is not None and length > self.max_length:
                result.add_error(ValidationError(
                    field=context.get('field', 'unknown') if context else 'unknown',
                    error_type=ValidationErrorType.LENGTH_ERROR,
                    message=self.message,
                    value=value,
                    expected=f"max_length: {self.max_length}"
                ))
        
        except TypeError:
            result.add_error(ValidationError(
                field=context.get('field', 'unknown') if context else 'unknown',
                error_type=ValidationErrorType.TYPE_ERROR,
                message="字段不支持长度检查",
                value=value
            ))
        
        return result


class RangeValidator(BaseValidator):
    """范围验证器"""
    
    def __init__(self, min_value: Optional[Union[int, float]] = None,
                 max_value: Optional[Union[int, float]] = None,
                 message: Optional[str] = None):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__("range", message or self._build_message())
    
    def _build_message(self) -> str:
        """构建错误消息"""
        if self.min_value is not None and self.max_value is not None:
            return f"字段值必须在 {self.min_value} 到 {self.max_value} 之间"
        elif self.min_value is not None:
            return f"字段值不能小于 {self.min_value}"
        elif self.max_value is not None:
            return f"字段值不能大于 {self.max_value}"
        return "字段值超出范围"
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None:
            return result
        
        try:
            numeric_value = float(value)
            
            if self.min_value is not None and numeric_value < self.min_value:
                result.add_error(ValidationError(
                    field=context.get('field', 'unknown') if context else 'unknown',
                    error_type=ValidationErrorType.RANGE_ERROR,
                    message=self.message,
                    value=value,
                    expected=f"min_value: {self.min_value}"
                ))
            
            if self.max_value is not None and numeric_value > self.max_value:
                result.add_error(ValidationError(
                    field=context.get('field', 'unknown') if context else 'unknown',
                    error_type=ValidationErrorType.RANGE_ERROR,
                    message=self.message,
                    value=value,
                    expected=f"max_value: {self.max_value}"
                ))
        
        except (ValueError, TypeError):
            result.add_error(ValidationError(
                field=context.get('field', 'unknown') if context else 'unknown',
                error_type=ValidationErrorType.TYPE_ERROR,
                message="字段值必须为数字类型",
                value=value
            ))
        
        return result


class RegexValidator(BaseValidator):
    """正则表达式验证器"""
    
    def __init__(self, pattern: Union[str, Pattern], message: Optional[str] = None):
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        super().__init__("regex", message or f"字段格式不匹配模式: {self.pattern.pattern}")
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None:
            return result
        
        str_value = str(value)
        if not self.pattern.match(str_value):
            result.add_error(ValidationError(
                field=context.get('field', 'unknown') if context else 'unknown',
                error_type=ValidationErrorType.FORMAT_ERROR,
                message=self.message,
                value=value,
                expected=self.pattern.pattern
            ))
        
        return result


class EmailValidator(RegexValidator):
    """邮箱验证器"""
    
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    def __init__(self, message: Optional[str] = None):
        super().__init__(self.EMAIL_PATTERN, message or "邮箱格式无效")


class URLValidator(RegexValidator):
    """URL验证器"""
    
    URL_PATTERN = re.compile(
        r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    )
    
    def __init__(self, message: Optional[str] = None):
        super().__init__(self.URL_PATTERN, message or "URL格式无效")


class CustomValidator(BaseValidator):
    """自定义验证器"""
    
    def __init__(self, validator_func: Callable[[Any], bool], name: str,
                 message: Optional[str] = None):
        self.validator_func = validator_func
        super().__init__(name, message or f"自定义验证失败: {name}")
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        try:
            if not self.validator_func(value):
                result.add_error(ValidationError(
                    field=context.get('field', 'unknown') if context else 'unknown',
                    error_type=ValidationErrorType.CUSTOM_ERROR,
                    message=self.message,
                    value=value
                ))
        except Exception as e:
            result.add_error(ValidationError(
                field=context.get('field', 'unknown') if context else 'unknown',
                error_type=ValidationErrorType.CUSTOM_ERROR,
                message=f"验证器执行错误: {str(e)}",
                value=value
            ))
        
        return result


class ValidationSchema:
    """验证模式"""
    
    def __init__(self, fields: Optional[Dict[str, List[BaseValidator]]] = None):
        self.fields = fields or {}
        self.global_validators: List[BaseValidator] = []
    
    def add_field(self, field_name: str, validators: List[BaseValidator]) -> None:
        """添加字段验证器"""
        self.fields[field_name] = validators
    
    def add_global_validator(self, validator: BaseValidator) -> None:
        """添加全局验证器"""
        self.global_validators.append(validator)
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """验证数据"""
        result = ValidationResult(is_valid=True, cleaned_data={})
        
        # 验证各个字段
        for field_name, validators in self.fields.items():
            field_value = data.get(field_name)
            context = {'field': field_name, 'data': data}
            
            for validator in validators:
                field_result = validator.validate(field_value, context)
                
                if not field_result.is_valid:
                    result.errors.extend(field_result.errors)
                    result.is_valid = False
                
                result.warnings.extend(field_result.warnings)
            
            # 如果验证通过，添加到清洗后的数据中
            if field_name not in [error.field for error in result.errors]:
                result.cleaned_data[field_name] = field_value
        
        # 执行全局验证器
        for validator in self.global_validators:
            global_result = validator.validate(data)
            
            if not global_result.is_valid:
                result.errors.extend(global_result.errors)
                result.is_valid = False
            
            result.warnings.extend(global_result.warnings)
        
        return result


class ModelValidator(Generic[ModelType]):
    """模型验证器"""
    
    def __init__(self, model_class: Type[ModelType], schema: Optional[ValidationSchema] = None):
        self.model_class = model_class
        self.schema = schema or self._build_schema_from_model()
    
    def _build_schema_from_model(self) -> ValidationSchema:
        """从模型构建验证模式"""
        schema = ValidationSchema()
        
        # 获取模型的类型提示
        type_hints = get_type_hints(self.model_class)
        
        for field_name, field_type in type_hints.items():
            validators = self._build_validators_for_type(field_type)
            if validators:
                schema.add_field(field_name, validators)
        
        return schema
    
    def _build_validators_for_type(self, field_type: Type) -> List[BaseValidator]:
        """根据类型构建验证器"""
        validators = []
        
        # 处理Optional类型
        origin = get_origin(field_type)
        args = get_args(field_type)
        
        if origin is Union and len(args) == 2 and type(None) in args:
            # Optional类型，不添加必填验证器
            actual_type = args[0] if args[1] is type(None) else args[1]
            validators.extend(self._build_validators_for_basic_type(actual_type))
        else:
            # 非Optional类型，添加必填验证器
            validators.append(RequiredValidator())
            validators.extend(self._build_validators_for_basic_type(field_type))
        
        return validators
    
    def _build_validators_for_basic_type(self, field_type: Type) -> List[BaseValidator]:
        """为基本类型构建验证器"""
        validators = []
        
        # 添加类型验证器
        if field_type in (str, int, float, bool, list, dict):
            validators.append(TypeValidator(field_type))
        
        # 为字符串类型添加长度验证器
        if field_type is str:
            validators.append(LengthValidator(max_length=1000))  # 默认最大长度
        
        return validators
    
    def validate(self, data: Union[Dict[str, Any], ModelType]) -> ValidationResult:
        """验证模型数据"""
        if isinstance(data, self.model_class):
            # 如果是模型实例，转换为字典
            data = data.dict() if hasattr(data, 'dict') else data.__dict__
        
        return self.schema.validate(data)
    
    def validate_batch(self, data_list: List[Union[Dict[str, Any], ModelType]]) -> List[ValidationResult]:
        """批量验证"""
        return [self.validate(data) for data in data_list]


class DataCleaner:
    """数据清洗器"""
    
    @staticmethod
    def clean_string(value: str, strip: bool = True, lower: bool = False,
                    upper: bool = False) -> str:
        """清洗字符串"""
        if not isinstance(value, str):
            value = str(value)
        
        if strip:
            value = value.strip()
        
        if lower:
            value = value.lower()
        elif upper:
            value = value.upper()
        
        return value
    
    @staticmethod
    def clean_phone(phone: str) -> str:
        """清洗电话号码"""
        # 移除所有非数字字符
        cleaned = re.sub(r'\D', '', phone)
        
        # 如果是中国手机号，确保格式正确
        if len(cleaned) == 11 and cleaned.startswith('1'):
            return cleaned
        
        return phone  # 返回原始值如果不符合预期格式
    
    @staticmethod
    def clean_email(email: str) -> str:
        """清洗邮箱地址"""
        return email.strip().lower()
    
    @staticmethod
    def mask_sensitive_data(value: str, mask_char: str = '*',
                          show_first: int = 2, show_last: int = 2) -> str:
        """脱敏敏感数据"""
        if len(value) <= show_first + show_last:
            return mask_char * len(value)
        
        return (value[:show_first] + 
                mask_char * (len(value) - show_first - show_last) + 
                value[-show_last:])
    
    @staticmethod
    def convert_type(value: Any, target_type: Type) -> Any:
        """类型转换"""
        if value is None:
            return None
        
        if isinstance(value, target_type):
            return value
        
        try:
            if target_type is str:
                return str(value)
            elif target_type is int:
                return int(float(value))  # 先转float再转int，处理"1.0"这种情况
            elif target_type is float:
                return float(value)
            elif target_type is bool:
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            elif target_type is Decimal:
                return Decimal(str(value))
            elif target_type is datetime:
                if isinstance(value, str):
                    # 尝试多种日期格式
                    formats = [
                        '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%d',
                        '%Y/%m/%d %H:%M:%S',
                        '%Y/%m/%d',
                        '%d/%m/%Y',
                        '%d-%m-%Y'
                    ]
                    for fmt in formats:
                        try:
                            return datetime.strptime(value, fmt)
                        except ValueError:
                            continue
                    raise ValueError(f"无法解析日期格式: {value}")
                return value
            else:
                return target_type(value)
        
        except (ValueError, TypeError) as e:
            raise ValueError(f"无法将 {value} 转换为 {target_type.__name__}: {str(e)}")


def validate_params(*validators: BaseValidator):
    """参数验证装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数参数名
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # 验证位置参数
            for i, (arg, validator) in enumerate(zip(args, validators)):
                field_name = param_names[i] if i < len(param_names) else f'arg_{i}'
                context = {'field': field_name}
                result = validator.validate(arg, context)
                
                if not result.is_valid:
                    error_messages = [error.message for error in result.errors]
                    raise ValueError(f"参数 {field_name} 验证失败: {'; '.join(error_messages)}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_request(schema: ValidationSchema, strict: bool = True):
    """API请求验证装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 假设第一个参数是请求数据
            if args:
                request_data = args[0]
                if hasattr(request_data, 'dict'):
                    data = request_data.dict()
                elif hasattr(request_data, '__dict__'):
                    data = request_data.__dict__
                else:
                    data = request_data
                
                result = schema.validate(data)
                
                if not result.is_valid and strict:
                    error_messages = [error.message for error in result.errors]
                    raise ValueError(f"请求验证失败: {'; '.join(error_messages)}")
                
                # 将清洗后的数据传递给函数
                if result.cleaned_data:
                    args = (result.cleaned_data,) + args[1:]
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


async def validate_async(validator: BaseValidator, value: Any, 
                        context: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """异步验证"""
    # 在实际应用中，这里可以执行异步操作，如数据库查询
    return validator.validate(value, context)


class ValidationRuleLoader:
    """验证规则加载器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.rules_cache: Dict[str, ValidationSchema] = {}
    
    def load_from_json(self, file_path: Path) -> ValidationSchema:
        """从JSON文件加载验证规则"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return self._build_schema_from_config(config)
    
    def load_from_yaml(self, file_path: Path) -> ValidationSchema:
        """从YAML文件加载验证规则"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return self._build_schema_from_config(config)
    
    def _build_schema_from_config(self, config: Dict[str, Any]) -> ValidationSchema:
        """从配置构建验证模式"""
        schema = ValidationSchema()
        
        fields_config = config.get('fields', {})
        for field_name, field_config in fields_config.items():
            validators = self._build_validators_from_config(field_config)
            schema.add_field(field_name, validators)
        
        return schema
    
    def _build_validators_from_config(self, field_config: Dict[str, Any]) -> List[BaseValidator]:
        """从配置构建验证器"""
        validators = []
        
        # 必填验证
        if field_config.get('required', False):
            validators.append(RequiredValidator(field_config.get('required_message')))
        
        # 类型验证
        field_type = field_config.get('type')
        if field_type:
            type_map = {
                'string': str,
                'integer': int,
                'float': float,
                'boolean': bool,
                'list': list,
                'dict': dict
            }
            if field_type in type_map:
                validators.append(TypeValidator(type_map[field_type], 
                                              field_config.get('type_message')))
        
        # 长度验证
        min_length = field_config.get('min_length')
        max_length = field_config.get('max_length')
        if min_length is not None or max_length is not None:
            validators.append(LengthValidator(min_length, max_length, 
                                            field_config.get('length_message')))
        
        # 范围验证
        min_value = field_config.get('min_value')
        max_value = field_config.get('max_value')
        if min_value is not None or max_value is not None:
            validators.append(RangeValidator(min_value, max_value, 
                                           field_config.get('range_message')))
        
        # 正则验证
        pattern = field_config.get('pattern')
        if pattern:
            validators.append(RegexValidator(pattern, field_config.get('pattern_message')))
        
        # 邮箱验证
        if field_config.get('email', False):
            validators.append(EmailValidator(field_config.get('email_message')))
        
        # URL验证
        if field_config.get('url', False):
            validators.append(URLValidator(field_config.get('url_message')))
        
        return validators
    
    def get_schema(self, schema_name: str) -> Optional[ValidationSchema]:
        """获取缓存的验证模式"""
        return self.rules_cache.get(schema_name)
    
    def cache_schema(self, schema_name: str, schema: ValidationSchema) -> None:
        """缓存验证模式"""
        self.rules_cache[schema_name] = schema


class ValidationManager:
    """验证管理器"""
    
    def __init__(self):
        self.schemas: Dict[str, ValidationSchema] = {}
        self.rule_loader = ValidationRuleLoader()
        self.data_cleaner = DataCleaner()
    
    def register_schema(self, name: str, schema: ValidationSchema) -> None:
        """注册验证模式"""
        self.schemas[name] = schema
    
    def get_schema(self, name: str) -> Optional[ValidationSchema]:
        """获取验证模式"""
        return self.schemas.get(name)
    
    def validate_data(self, schema_name: str, data: Dict[str, Any]) -> ValidationResult:
        """验证数据"""
        schema = self.get_schema(schema_name)
        if not schema:
            raise ValueError(f"未找到验证模式: {schema_name}")
        
        return schema.validate(data)
    
    def clean_and_validate(self, schema_name: str, data: Dict[str, Any]) -> ValidationResult:
        """清洗并验证数据"""
        # 先进行数据清洗
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                cleaned_data[key] = self.data_cleaner.clean_string(value)
            else:
                cleaned_data[key] = value
        
        # 再进行验证
        return self.validate_data(schema_name, cleaned_data)


# 预定义的常用验证器实例
COMMON_VALIDATORS = {
    'required': RequiredValidator(),
    'email': EmailValidator(),
    'url': URLValidator(),
    'phone': RegexValidator(r'^1[3-9]\d{9}$', '手机号格式无效'),
    'id_card': RegexValidator(r'^\d{17}[\dXx]$', '身份证号格式无效'),
    'positive_int': CustomValidator(lambda x: isinstance(x, int) and x > 0, 
                                   'positive_int', '必须为正整数'),
    'non_empty_string': CustomValidator(lambda x: isinstance(x, str) and x.strip(), 
                                       'non_empty_string', '不能为空字符串')
}


# 导出主要类和函数
__all__ = [
    'ValidationLevel', 'ValidationErrorType', 'ValidationError', 'ValidationResult',
    'BaseValidator', 'RequiredValidator', 'TypeValidator', 'LengthValidator',
    'RangeValidator', 'RegexValidator', 'EmailValidator', 'URLValidator',
    'CustomValidator', 'ValidationSchema', 'ModelValidator', 'DataCleaner',
    'validate_params', 'validate_request', 'validate_async',
    'ValidationRuleLoader', 'ValidationManager', 'COMMON_VALIDATORS'
]