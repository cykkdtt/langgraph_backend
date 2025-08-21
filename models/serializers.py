"""模型序列化系统模块

本模块提供数据序列化、反序列化和格式转换功能。
"""

from typing import (
    Dict, Any, Optional, List, Union, Callable, Type, 
    TypeVar, Generic, Tuple, Set, get_type_hints
)
from datetime import datetime, date, time
from decimal import Decimal
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from abc import ABC, abstractmethod
import json
import pickle
import base64
import uuid
import logging
from collections import defaultdict
from functools import wraps
import xml.etree.ElementTree as ET
import yaml
import csv
import io
from sqlalchemy.orm import Session
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy.ext.declarative import DeclarativeMeta

# 导入项目模型
from .models import (
    User, Session as ChatSession, Thread, Message, Workflow, 
    WorkflowExecution, WorkflowStep, Memory, MemoryVector, 
    TimeTravel, Attachment, UserPreference, SystemConfig
)
from .events import EventType, emit_business_event


logger = logging.getLogger(__name__)


T = TypeVar('T')
S = TypeVar('S')


class SerializationFormat(Enum):
    """序列化格式枚举"""
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    PICKLE = "pickle"
    BASE64 = "base64"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"


class SerializationMode(Enum):
    """序列化模式枚举"""
    FULL = "full"                  # 完整序列化
    PARTIAL = "partial"            # 部分序列化
    REFERENCE = "reference"        # 引用序列化
    LAZY = "lazy"                  # 懒加载序列化
    COMPRESSED = "compressed"      # 压缩序列化


class SerializationLevel(Enum):
    """序列化级别枚举"""
    SHALLOW = "shallow"            # 浅层序列化
    DEEP = "deep"                  # 深层序列化
    RECURSIVE = "recursive"        # 递归序列化


@dataclass
class SerializationContext:
    """序列化上下文"""
    format: SerializationFormat
    mode: SerializationMode = SerializationMode.FULL
    level: SerializationLevel = SerializationLevel.SHALLOW
    include_fields: Optional[Set[str]] = None
    exclude_fields: Optional[Set[str]] = None
    max_depth: int = 10
    current_depth: int = 0
    visited_objects: Set[int] = field(default_factory=set)
    custom_serializers: Dict[Type, Callable] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_include_field(self, field_name: str) -> bool:
        """检查是否应该包含字段"""
        if self.include_fields and field_name not in self.include_fields:
            return False
        if self.exclude_fields and field_name in self.exclude_fields:
            return False
        return True
    
    def can_go_deeper(self) -> bool:
        """检查是否可以继续深入"""
        return self.current_depth < self.max_depth
    
    def enter_object(self, obj_id: int) -> bool:
        """进入对象（检查循环引用）"""
        if obj_id in self.visited_objects:
            return False
        self.visited_objects.add(obj_id)
        self.current_depth += 1
        return True
    
    def exit_object(self, obj_id: int) -> None:
        """退出对象"""
        self.visited_objects.discard(obj_id)
        self.current_depth -= 1
    
    def copy(self) -> 'SerializationContext':
        """复制上下文"""
        return SerializationContext(
            format=self.format,
            mode=self.mode,
            level=self.level,
            include_fields=self.include_fields.copy() if self.include_fields else None,
            exclude_fields=self.exclude_fields.copy() if self.exclude_fields else None,
            max_depth=self.max_depth,
            current_depth=self.current_depth,
            visited_objects=self.visited_objects.copy(),
            custom_serializers=self.custom_serializers.copy(),
            metadata=self.metadata.copy()
        )


@dataclass
class SerializationResult:
    """序列化结果"""
    data: Any
    format: SerializationFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    size: int = 0
    compression_ratio: float = 1.0
    serialization_time: float = 0.0
    
    def to_bytes(self) -> bytes:
        """转换为字节"""
        if isinstance(self.data, bytes):
            return self.data
        elif isinstance(self.data, str):
            return self.data.encode('utf-8')
        else:
            return str(self.data).encode('utf-8')
    
    def to_string(self) -> str:
        """转换为字符串"""
        if isinstance(self.data, str):
            return self.data
        elif isinstance(self.data, bytes):
            return self.data.decode('utf-8')
        else:
            return str(self.data)


class BaseSerializer(ABC, Generic[T]):
    """序列化器基类"""
    
    def __init__(self, target_type: Type[T]):
        self.target_type = target_type
        self._type_handlers: Dict[Type, Callable] = {}
        self._register_default_handlers()
    
    @abstractmethod
    def serialize(self, obj: T, context: SerializationContext) -> SerializationResult:
        """序列化对象"""
        pass
    
    @abstractmethod
    def deserialize(self, data: Any, context: SerializationContext) -> T:
        """反序列化对象"""
        pass
    
    def register_type_handler(self, type_class: Type, handler: Callable) -> None:
        """注册类型处理器"""
        self._type_handlers[type_class] = handler
    
    def _register_default_handlers(self) -> None:
        """注册默认类型处理器"""
        self._type_handlers.update({
            datetime: lambda x: x.isoformat(),
            date: lambda x: x.isoformat(),
            time: lambda x: x.isoformat(),
            Decimal: lambda x: str(x),
            uuid.UUID: lambda x: str(x),
            bytes: lambda x: base64.b64encode(x).decode('ascii'),
            set: lambda x: list(x),
            frozenset: lambda x: list(x),
        })
    
    def _handle_special_types(self, obj: Any) -> Any:
        """处理特殊类型"""
        obj_type = type(obj)
        
        # 检查是否有自定义处理器
        for type_class, handler in self._type_handlers.items():
            if isinstance(obj, type_class):
                return handler(obj)
        
        # 处理枚举
        if isinstance(obj, Enum):
            return obj.value
        
        return obj
    
    def _is_sqlalchemy_model(self, obj: Any) -> bool:
        """检查是否是SQLAlchemy模型"""
        return hasattr(obj, '__table__') and hasattr(obj, '__mapper__')
    
    def _get_sqlalchemy_columns(self, obj: Any) -> List[str]:
        """获取SQLAlchemy模型的列"""
        if self._is_sqlalchemy_model(obj):
            return [column.name for column in obj.__table__.columns]
        return []
    
    def _get_sqlalchemy_relationships(self, obj: Any) -> List[str]:
        """获取SQLAlchemy模型的关系"""
        if self._is_sqlalchemy_model(obj):
            return [rel.key for rel in obj.__mapper__.relationships]
        return []


class JSONSerializer(BaseSerializer[Any]):
    """JSON序列化器"""
    
    def __init__(self):
        super().__init__(Any)
        self.json_encoder = self._create_json_encoder()
    
    def serialize(self, obj: Any, context: SerializationContext) -> SerializationResult:
        """序列化为JSON"""
        import time
        start_time = time.time()
        
        try:
            serialized_obj = self._serialize_object(obj, context)
            json_data = json.dumps(serialized_obj, cls=self.json_encoder, 
                                 ensure_ascii=False, indent=2)
            
            result = SerializationResult(
                data=json_data,
                format=SerializationFormat.JSON,
                size=len(json_data.encode('utf-8')),
                serialization_time=time.time() - start_time
            )
            
            return result
        
        except Exception as e:
            logger.error(f"JSON serialization error: {e}")
            raise
    
    def deserialize(self, data: Any, context: SerializationContext) -> Any:
        """从JSON反序列化"""
        try:
            if isinstance(data, str):
                return json.loads(data)
            elif isinstance(data, bytes):
                return json.loads(data.decode('utf-8'))
            else:
                return data
        
        except Exception as e:
            logger.error(f"JSON deserialization error: {e}")
            raise
    
    def _serialize_object(self, obj: Any, context: SerializationContext) -> Any:
        """序列化对象"""
        if obj is None:
            return None
        
        obj_id = id(obj)
        
        # 检查循环引用
        if not context.enter_object(obj_id):
            return {"__ref__": obj_id}
        
        try:
            # 处理基本类型
            if isinstance(obj, (str, int, float, bool)):
                return obj
            
            # 处理特殊类型
            handled = self._handle_special_types(obj)
            if handled != obj:
                return handled
            
            # 处理列表和元组
            if isinstance(obj, (list, tuple)):
                if context.can_go_deeper():
                    return [self._serialize_object(item, context) for item in obj]
                else:
                    return f"<{type(obj).__name__} with {len(obj)} items>"
            
            # 处理字典
            if isinstance(obj, dict):
                if context.can_go_deeper():
                    return {
                        str(k): self._serialize_object(v, context) 
                        for k, v in obj.items()
                    }
                else:
                    return f"<dict with {len(obj)} items>"
            
            # 处理SQLAlchemy模型
            if self._is_sqlalchemy_model(obj):
                return self._serialize_sqlalchemy_model(obj, context)
            
            # 处理数据类
            if is_dataclass(obj):
                return self._serialize_dataclass(obj, context)
            
            # 处理普通对象
            return self._serialize_regular_object(obj, context)
        
        finally:
            context.exit_object(obj_id)
    
    def _serialize_sqlalchemy_model(self, obj: Any, context: SerializationContext) -> Dict[str, Any]:
        """序列化SQLAlchemy模型"""
        result = {"__type__": type(obj).__name__}
        
        # 序列化列
        for column_name in self._get_sqlalchemy_columns(obj):
            if context.should_include_field(column_name):
                value = getattr(obj, column_name, None)
                result[column_name] = self._serialize_object(value, context)
        
        # 序列化关系（根据模式和级别）
        if context.level in [SerializationLevel.DEEP, SerializationLevel.RECURSIVE]:
            for rel_name in self._get_sqlalchemy_relationships(obj):
                if context.should_include_field(rel_name) and context.can_go_deeper():
                    value = getattr(obj, rel_name, None)
                    if value is not None:
                        if context.mode == SerializationMode.REFERENCE:
                            # 只序列化引用
                            if hasattr(value, 'id'):
                                result[rel_name] = {"__ref__": value.id}
                        else:
                            result[rel_name] = self._serialize_object(value, context)
        
        return result
    
    def _serialize_dataclass(self, obj: Any, context: SerializationContext) -> Dict[str, Any]:
        """序列化数据类"""
        result = {"__type__": type(obj).__name__}
        
        for field_info in fields(obj):
            field_name = field_info.name
            if context.should_include_field(field_name):
                value = getattr(obj, field_name)
                result[field_name] = self._serialize_object(value, context)
        
        return result
    
    def _serialize_regular_object(self, obj: Any, context: SerializationContext) -> Dict[str, Any]:
        """序列化普通对象"""
        result = {"__type__": type(obj).__name__}
        
        # 获取对象属性
        for attr_name in dir(obj):
            if (not attr_name.startswith('_') and 
                not callable(getattr(obj, attr_name)) and
                context.should_include_field(attr_name)):
                
                try:
                    value = getattr(obj, attr_name)
                    result[attr_name] = self._serialize_object(value, context)
                except Exception:
                    # 忽略无法访问的属性
                    pass
        
        return result
    
    def _create_json_encoder(self):
        """创建JSON编码器"""
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                # 处理特殊类型
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, date):
                    return obj.isoformat()
                elif isinstance(obj, time):
                    return obj.isoformat()
                elif isinstance(obj, Decimal):
                    return str(obj)
                elif isinstance(obj, uuid.UUID):
                    return str(obj)
                elif isinstance(obj, bytes):
                    return base64.b64encode(obj).decode('ascii')
                elif isinstance(obj, (set, frozenset)):
                    return list(obj)
                elif isinstance(obj, Enum):
                    return obj.value
                
                return super().default(obj)
        
        return CustomJSONEncoder


class XMLSerializer(BaseSerializer[Any]):
    """XML序列化器"""
    
    def __init__(self):
        super().__init__(Any)
    
    def serialize(self, obj: Any, context: SerializationContext) -> SerializationResult:
        """序列化为XML"""
        import time
        start_time = time.time()
        
        try:
            root = ET.Element("root")
            self._serialize_to_element(obj, root, context)
            
            xml_data = ET.tostring(root, encoding='unicode')
            
            result = SerializationResult(
                data=xml_data,
                format=SerializationFormat.XML,
                size=len(xml_data.encode('utf-8')),
                serialization_time=time.time() - start_time
            )
            
            return result
        
        except Exception as e:
            logger.error(f"XML serialization error: {e}")
            raise
    
    def deserialize(self, data: Any, context: SerializationContext) -> Any:
        """从XML反序列化"""
        try:
            if isinstance(data, str):
                root = ET.fromstring(data)
            elif isinstance(data, bytes):
                root = ET.fromstring(data.decode('utf-8'))
            else:
                raise ValueError("Invalid XML data")
            
            return self._deserialize_from_element(root)
        
        except Exception as e:
            logger.error(f"XML deserialization error: {e}")
            raise
    
    def _serialize_to_element(self, obj: Any, parent: ET.Element, context: SerializationContext) -> None:
        """序列化到XML元素"""
        if obj is None:
            parent.set("type", "null")
            return
        
        obj_type = type(obj).__name__
        parent.set("type", obj_type)
        
        if isinstance(obj, (str, int, float, bool)):
            parent.text = str(obj)
        
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                item_elem = ET.SubElement(parent, "item")
                item_elem.set("index", str(i))
                self._serialize_to_element(item, item_elem, context)
        
        elif isinstance(obj, dict):
            for key, value in obj.items():
                item_elem = ET.SubElement(parent, "item")
                item_elem.set("key", str(key))
                self._serialize_to_element(value, item_elem, context)
        
        else:
            # 处理复杂对象
            handled = self._handle_special_types(obj)
            if handled != obj:
                parent.text = str(handled)
            else:
                parent.text = str(obj)
    
    def _deserialize_from_element(self, element: ET.Element) -> Any:
        """从XML元素反序列化"""
        obj_type = element.get("type", "str")
        
        if obj_type == "null":
            return None
        elif obj_type == "str":
            return element.text or ""
        elif obj_type == "int":
            return int(element.text or "0")
        elif obj_type == "float":
            return float(element.text or "0.0")
        elif obj_type == "bool":
            return element.text.lower() == "true"
        elif obj_type in ["list", "tuple"]:
            items = []
            for item_elem in element.findall("item"):
                items.append(self._deserialize_from_element(item_elem))
            return tuple(items) if obj_type == "tuple" else items
        elif obj_type == "dict":
            result = {}
            for item_elem in element.findall("item"):
                key = item_elem.get("key")
                value = self._deserialize_from_element(item_elem)
                result[key] = value
            return result
        else:
            return element.text


class YAMLSerializer(BaseSerializer[Any]):
    """YAML序列化器"""
    
    def __init__(self):
        super().__init__(Any)
    
    def serialize(self, obj: Any, context: SerializationContext) -> SerializationResult:
        """序列化为YAML"""
        import time
        start_time = time.time()
        
        try:
            # 使用JSON序列化器预处理对象
            json_serializer = JSONSerializer()
            processed_obj = json_serializer._serialize_object(obj, context)
            
            yaml_data = yaml.dump(processed_obj, default_flow_style=False, 
                                allow_unicode=True, indent=2)
            
            result = SerializationResult(
                data=yaml_data,
                format=SerializationFormat.YAML,
                size=len(yaml_data.encode('utf-8')),
                serialization_time=time.time() - start_time
            )
            
            return result
        
        except Exception as e:
            logger.error(f"YAML serialization error: {e}")
            raise
    
    def deserialize(self, data: Any, context: SerializationContext) -> Any:
        """从YAML反序列化"""
        try:
            if isinstance(data, str):
                return yaml.safe_load(data)
            elif isinstance(data, bytes):
                return yaml.safe_load(data.decode('utf-8'))
            else:
                return data
        
        except Exception as e:
            logger.error(f"YAML deserialization error: {e}")
            raise


class CSVSerializer(BaseSerializer[List[Dict[str, Any]]]):
    """CSV序列化器"""
    
    def __init__(self):
        super().__init__(List[Dict[str, Any]])
    
    def serialize(self, obj: List[Dict[str, Any]], context: SerializationContext) -> SerializationResult:
        """序列化为CSV"""
        import time
        start_time = time.time()
        
        try:
            if not obj:
                return SerializationResult(
                    data="",
                    format=SerializationFormat.CSV,
                    serialization_time=time.time() - start_time
                )
            
            # 获取所有字段名
            fieldnames = set()
            for item in obj:
                if isinstance(item, dict):
                    fieldnames.update(item.keys())
            
            fieldnames = sorted(fieldnames)
            
            # 写入CSV
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in obj:
                if isinstance(item, dict):
                    # 处理特殊类型
                    processed_item = {}
                    for key, value in item.items():
                        processed_item[key] = self._handle_special_types(value)
                    writer.writerow(processed_item)
            
            csv_data = output.getvalue()
            
            result = SerializationResult(
                data=csv_data,
                format=SerializationFormat.CSV,
                size=len(csv_data.encode('utf-8')),
                serialization_time=time.time() - start_time
            )
            
            return result
        
        except Exception as e:
            logger.error(f"CSV serialization error: {e}")
            raise
    
    def deserialize(self, data: Any, context: SerializationContext) -> List[Dict[str, Any]]:
        """从CSV反序列化"""
        try:
            if isinstance(data, str):
                input_data = io.StringIO(data)
            elif isinstance(data, bytes):
                input_data = io.StringIO(data.decode('utf-8'))
            else:
                raise ValueError("Invalid CSV data")
            
            reader = csv.DictReader(input_data)
            return list(reader)
        
        except Exception as e:
            logger.error(f"CSV deserialization error: {e}")
            raise


class PickleSerializer(BaseSerializer[Any]):
    """Pickle序列化器"""
    
    def __init__(self):
        super().__init__(Any)
    
    def serialize(self, obj: Any, context: SerializationContext) -> SerializationResult:
        """序列化为Pickle"""
        import time
        start_time = time.time()
        
        try:
            pickle_data = pickle.dumps(obj)
            
            result = SerializationResult(
                data=pickle_data,
                format=SerializationFormat.PICKLE,
                size=len(pickle_data),
                serialization_time=time.time() - start_time
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Pickle serialization error: {e}")
            raise
    
    def deserialize(self, data: Any, context: SerializationContext) -> Any:
        """从Pickle反序列化"""
        try:
            if isinstance(data, bytes):
                return pickle.loads(data)
            else:
                raise ValueError("Pickle data must be bytes")
        
        except Exception as e:
            logger.error(f"Pickle deserialization error: {e}")
            raise


class SerializerRegistry:
    """序列化器注册表"""
    
    def __init__(self):
        self._serializers: Dict[SerializationFormat, BaseSerializer] = {}
        self._register_default_serializers()
    
    def _register_default_serializers(self) -> None:
        """注册默认序列化器"""
        self._serializers[SerializationFormat.JSON] = JSONSerializer()
        self._serializers[SerializationFormat.XML] = XMLSerializer()
        self._serializers[SerializationFormat.YAML] = YAMLSerializer()
        self._serializers[SerializationFormat.CSV] = CSVSerializer()
        self._serializers[SerializationFormat.PICKLE] = PickleSerializer()
    
    def register(self, format: SerializationFormat, serializer: BaseSerializer) -> None:
        """注册序列化器"""
        self._serializers[format] = serializer
    
    def get(self, format: SerializationFormat) -> Optional[BaseSerializer]:
        """获取序列化器"""
        return self._serializers.get(format)
    
    def get_supported_formats(self) -> List[SerializationFormat]:
        """获取支持的格式"""
        return list(self._serializers.keys())


class SerializationManager:
    """序列化管理器"""
    
    def __init__(self):
        self.registry = SerializerRegistry()
        self._default_context = SerializationContext(
            format=SerializationFormat.JSON
        )
    
    def serialize(self, obj: Any, format: SerializationFormat = SerializationFormat.JSON,
                 context: Optional[SerializationContext] = None) -> SerializationResult:
        """序列化对象"""
        if context is None:
            context = self._default_context.copy()
            context.format = format
        
        serializer = self.registry.get(format)
        if not serializer:
            raise ValueError(f"Unsupported serialization format: {format}")
        
        try:
            result = serializer.serialize(obj, context)
            
            # 发布事件
            emit_business_event(
                EventType.SYSTEM_STARTUP,  # 使用系统事件类型
                "serialization",
                data={
                    'action': 'serialize',
                    'format': format.value,
                    'size': result.size,
                    'time': result.serialization_time
                }
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
    
    def deserialize(self, data: Any, format: SerializationFormat = SerializationFormat.JSON,
                   context: Optional[SerializationContext] = None) -> Any:
        """反序列化对象"""
        if context is None:
            context = self._default_context.copy()
            context.format = format
        
        serializer = self.registry.get(format)
        if not serializer:
            raise ValueError(f"Unsupported serialization format: {format}")
        
        try:
            result = serializer.deserialize(data, context)
            
            # 发布事件
            emit_business_event(
                EventType.SYSTEM_STARTUP,  # 使用系统事件类型
                "serialization",
                data={
                    'action': 'deserialize',
                    'format': format.value
                }
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise
    
    def convert_format(self, data: Any, from_format: SerializationFormat,
                      to_format: SerializationFormat,
                      context: Optional[SerializationContext] = None) -> SerializationResult:
        """转换格式"""
        # 先反序列化
        obj = self.deserialize(data, from_format, context)
        
        # 再序列化为目标格式
        return self.serialize(obj, to_format, context)
    
    def register_serializer(self, format: SerializationFormat, serializer: BaseSerializer) -> None:
        """注册自定义序列化器"""
        self.registry.register(format, serializer)
    
    def get_supported_formats(self) -> List[SerializationFormat]:
        """获取支持的格式"""
        return self.registry.get_supported_formats()


# 序列化装饰器
def serializable(format: SerializationFormat = SerializationFormat.JSON,
                include_fields: Optional[Set[str]] = None,
                exclude_fields: Optional[Set[str]] = None):
    """序列化装饰器"""
    def decorator(cls):
        def to_dict(self, **kwargs):
            context = SerializationContext(
                format=format,
                include_fields=include_fields,
                exclude_fields=exclude_fields
            )
            context.metadata.update(kwargs)
            
            result = serialization_manager.serialize(self, format, context)
            if format == SerializationFormat.JSON:
                return json.loads(result.data)
            return result.data
        
        def from_dict(cls, data: Dict[str, Any]):
            # 简单的从字典创建对象
            if hasattr(cls, '__init__'):
                try:
                    return cls(**data)
                except Exception:
                    # 如果直接创建失败，尝试设置属性
                    obj = cls()
                    for key, value in data.items():
                        if hasattr(obj, key):
                            setattr(obj, key, value)
                    return obj
            return data
        
        cls.to_dict = to_dict
        cls.from_dict = classmethod(from_dict)
        
        return cls
    
    return decorator


# 全局序列化管理器
serialization_manager = SerializationManager()


# 便捷函数
def serialize(obj: Any, format: SerializationFormat = SerializationFormat.JSON,
             **kwargs) -> SerializationResult:
    """序列化对象"""
    context = SerializationContext(format=format)
    for key, value in kwargs.items():
        if hasattr(context, key):
            setattr(context, key, value)
    
    return serialization_manager.serialize(obj, format, context)


def deserialize(data: Any, format: SerializationFormat = SerializationFormat.JSON,
               **kwargs) -> Any:
    """反序列化对象"""
    context = SerializationContext(format=format)
    for key, value in kwargs.items():
        if hasattr(context, key):
            setattr(context, key, value)
    
    return serialization_manager.deserialize(data, format, context)


def convert_format(data: Any, from_format: SerializationFormat,
                  to_format: SerializationFormat, **kwargs) -> SerializationResult:
    """转换格式"""
    context = SerializationContext(format=from_format)
    for key, value in kwargs.items():
        if hasattr(context, key):
            setattr(context, key, value)
    
    return serialization_manager.convert_format(data, from_format, to_format, context)


# 导出所有类和函数
__all__ = [
    "SerializationFormat",
    "SerializationMode",
    "SerializationLevel",
    "SerializationContext",
    "SerializationResult",
    "BaseSerializer",
    "JSONSerializer",
    "XMLSerializer",
    "YAMLSerializer",
    "CSVSerializer",
    "PickleSerializer",
    "SerializerRegistry",
    "SerializationManager",
    "serialization_manager",
    "serializable",
    "serialize",
    "deserialize",
    "convert_format"
]