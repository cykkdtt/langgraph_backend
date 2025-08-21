"""模型序列化系统模块

本模块提供数据序列化、反序列化和格式转换功能。
"""

from typing import (
    Dict, Any, Optional, List, Union, Callable, Type, 
    Tuple, Set, ClassVar, Protocol, TypeVar, Generic,
    NamedTuple, AsyncGenerator, Awaitable, IO
)
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps, singledispatch
import logging
import json
import pickle
import base64
import gzip
import bz2
import lzma
import uuid
import re
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager

# XML imports (optional)
try:
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    XML_AVAILABLE = True
except ImportError:
    ET = None
    minidom = None
    XML_AVAILABLE = False

# YAML imports (optional)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

# MessagePack imports (optional)
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    msgpack = None
    MSGPACK_AVAILABLE = False

# Protobuf imports (optional)
try:
    from google.protobuf import message
    from google.protobuf.json_format import MessageToJson, Parse
    PROTOBUF_AVAILABLE = True
except ImportError:
    message = None
    MessageToJson = None
    Parse = None
    PROTOBUF_AVAILABLE = False

# 导入项目模块
from .events import EventType, emit_business_event
from .validation import validate_data


logger = logging.getLogger(__name__)


T = TypeVar('T')
Serializable = TypeVar('Serializable')


class SerializationFormat(Enum):
    """序列化格式枚举"""
    JSON = "json"                          # JSON格式
    PICKLE = "pickle"                      # Python Pickle格式
    XML = "xml"                            # XML格式
    YAML = "yaml"                          # YAML格式
    MSGPACK = "msgpack"                    # MessagePack格式
    PROTOBUF = "protobuf"                  # Protocol Buffers格式
    CSV = "csv"                            # CSV格式
    BINARY = "binary"                      # 二进制格式
    BASE64 = "base64"                      # Base64编码
    HEX = "hex"                            # 十六进制编码


class CompressionType(Enum):
    """压缩类型枚举"""
    NONE = "none"                          # 无压缩
    GZIP = "gzip"                          # GZIP压缩
    BZIP2 = "bzip2"                        # BZIP2压缩
    LZMA = "lzma"                          # LZMA压缩
    ZLIB = "zlib"                          # ZLIB压缩


class EncodingType(Enum):
    """编码类型枚举"""
    UTF8 = "utf-8"                         # UTF-8编码
    UTF16 = "utf-16"                       # UTF-16编码
    UTF32 = "utf-32"                       # UTF-32编码
    ASCII = "ascii"                        # ASCII编码
    LATIN1 = "latin-1"                     # Latin-1编码
    GBK = "gbk"                            # GBK编码
    GB2312 = "gb2312"                      # GB2312编码


class SerializationMode(Enum):
    """序列化模式枚举"""
    STRICT = "strict"                      # 严格模式
    LENIENT = "lenient"                    # 宽松模式
    SAFE = "safe"                          # 安全模式
    FAST = "fast"                          # 快速模式


@dataclass
class SerializationOptions:
    """序列化选项"""
    # 基本选项
    format: SerializationFormat = SerializationFormat.JSON
    encoding: EncodingType = EncodingType.UTF8
    compression: CompressionType = CompressionType.NONE
    mode: SerializationMode = SerializationMode.STRICT
    
    # JSON选项
    json_indent: Optional[int] = None      # JSON缩进
    json_sort_keys: bool = False           # JSON键排序
    json_ensure_ascii: bool = False        # JSON确保ASCII
    json_separators: Optional[Tuple[str, str]] = None  # JSON分隔符
    
    # XML选项
    xml_root_tag: str = "root"             # XML根标签
    xml_pretty: bool = True                # XML美化
    xml_encoding: str = "utf-8"            # XML编码
    
    # YAML选项
    yaml_default_flow_style: bool = False  # YAML流式风格
    yaml_allow_unicode: bool = True        # YAML允许Unicode
    
    # 压缩选项
    compression_level: int = 6             # 压缩级别
    
    # 安全选项
    allow_pickle: bool = False             # 允许Pickle
    max_depth: int = 100                   # 最大深度
    max_size_mb: float = 100.0             # 最大大小（MB）
    
    # 性能选项
    use_cache: bool = True                 # 使用缓存
    parallel_processing: bool = False      # 并行处理
    
    # 自定义选项
    custom_encoders: Dict[Type, Callable] = field(default_factory=dict)
    custom_decoders: Dict[Type, Callable] = field(default_factory=dict)
    type_hints: Dict[str, Type] = field(default_factory=dict)


@dataclass
class SerializationResult:
    """序列化结果"""
    data: Union[str, bytes]                # 序列化数据
    format: SerializationFormat           # 格式
    encoding: EncodingType                # 编码
    compression: CompressionType          # 压缩
    
    # 元数据
    original_size: int = 0                # 原始大小
    compressed_size: int = 0              # 压缩后大小
    serialization_time_ms: float = 0.0   # 序列化时间
    
    # 校验信息
    checksum: Optional[str] = None        # 校验和
    version: str = "1.0"                  # 版本
    
    @property
    def compression_ratio(self) -> float:
        """压缩比"""
        if self.original_size == 0:
            return 0.0
        return self.compressed_size / self.original_size
    
    @property
    def size_reduction_percent(self) -> float:
        """大小减少百分比"""
        if self.original_size == 0:
            return 0.0
        return (1 - self.compression_ratio) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'format': self.format.value,
            'encoding': self.encoding.value,
            'compression': self.compression.value,
            'original_size': self.original_size,
            'compressed_size': self.compressed_size,
            'serialization_time_ms': self.serialization_time_ms,
            'compression_ratio': self.compression_ratio,
            'size_reduction_percent': self.size_reduction_percent,
            'checksum': self.checksum,
            'version': self.version
        }


class SerializationError(Exception):
    """序列化错误"""
    pass


class DeserializationError(Exception):
    """反序列化错误"""
    pass


class CompressionError(Exception):
    """压缩错误"""
    pass


class EncodingError(Exception):
    """编码错误"""
    pass


class SerializerProtocol(Protocol):
    """序列化器协议"""
    
    def serialize(self, obj: Any, options: SerializationOptions) -> SerializationResult:
        """序列化对象"""
        ...
    
    def deserialize(self, data: Union[str, bytes], target_type: Type[T], 
                   options: SerializationOptions) -> T:
        """反序列化数据"""
        ...


class JSONSerializer:
    """JSON序列化器"""
    
    def __init__(self):
        self._custom_encoders = {}
        self._custom_decoders = {}
    
    def serialize(self, obj: Any, options: SerializationOptions) -> SerializationResult:
        """序列化为JSON"""
        import time
        start_time = time.time()
        
        try:
            # 预处理对象
            processed_obj = self._preprocess_object(obj, options)
            
            # JSON序列化选项
            json_options = {
                'ensure_ascii': options.json_ensure_ascii,
                'sort_keys': options.json_sort_keys,
                'indent': options.json_indent,
                'separators': options.json_separators,
                'default': self._json_default
            }
            
            # 序列化
            json_str = json.dumps(processed_obj, **json_options)
            
            # 编码
            data = json_str.encode(options.encoding.value)
            
            # 压缩
            original_size = len(data)
            compressed_data = self._compress(data, options.compression, options.compression_level)
            
            # 计算校验和
            checksum = self._calculate_checksum(compressed_data)
            
            return SerializationResult(
                data=compressed_data,
                format=SerializationFormat.JSON,
                encoding=options.encoding,
                compression=options.compression,
                original_size=original_size,
                compressed_size=len(compressed_data),
                serialization_time_ms=(time.time() - start_time) * 1000,
                checksum=checksum
            )
            
        except Exception as e:
            raise SerializationError(f"JSON serialization failed: {e}")
    
    def deserialize(self, data: Union[str, bytes], target_type: Type[T], 
                   options: SerializationOptions) -> T:
        """从JSON反序列化"""
        try:
            # 解压缩
            if isinstance(data, bytes):
                decompressed_data = self._decompress(data, options.compression)
                json_str = decompressed_data.decode(options.encoding.value)
            else:
                json_str = data
            
            # JSON反序列化
            obj = json.loads(json_str, object_hook=self._json_object_hook)
            
            # 后处理对象
            return self._postprocess_object(obj, target_type, options)
            
        except Exception as e:
            raise DeserializationError(f"JSON deserialization failed: {e}")
    
    def _preprocess_object(self, obj: Any, options: SerializationOptions) -> Any:
        """预处理对象"""
        return self._convert_for_json(obj, options, set())
    
    def _convert_for_json(self, obj: Any, options: SerializationOptions, seen: Set[int]) -> Any:
        """转换对象为JSON兼容格式"""
        # 防止循环引用
        obj_id = id(obj)
        if obj_id in seen:
            return f"<circular_reference:{obj_id}>"
        
        # 基本类型
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        # 日期时间类型
        if isinstance(obj, datetime):
            return {'__type__': 'datetime', 'value': obj.isoformat()}
        elif isinstance(obj, date):
            return {'__type__': 'date', 'value': obj.isoformat()}
        elif isinstance(obj, time):
            return {'__type__': 'time', 'value': obj.isoformat()}
        elif isinstance(obj, timedelta):
            return {'__type__': 'timedelta', 'value': obj.total_seconds()}
        
        # 数值类型
        elif isinstance(obj, Decimal):
            return {'__type__': 'decimal', 'value': str(obj)}
        elif isinstance(obj, complex):
            return {'__type__': 'complex', 'real': obj.real, 'imag': obj.imag}
        
        # UUID
        elif isinstance(obj, uuid.UUID):
            return {'__type__': 'uuid', 'value': str(obj)}
        
        # 字节类型
        elif isinstance(obj, bytes):
            return {'__type__': 'bytes', 'value': base64.b64encode(obj).decode()}
        elif isinstance(obj, bytearray):
            return {'__type__': 'bytearray', 'value': base64.b64encode(obj).decode()}
        
        # 集合类型
        elif isinstance(obj, set):
            seen.add(obj_id)
            result = {'__type__': 'set', 'value': [self._convert_for_json(item, options, seen) for item in obj]}
            seen.remove(obj_id)
            return result
        elif isinstance(obj, frozenset):
            seen.add(obj_id)
            result = {'__type__': 'frozenset', 'value': [self._convert_for_json(item, options, seen) for item in obj]}
            seen.remove(obj_id)
            return result
        elif isinstance(obj, tuple):
            seen.add(obj_id)
            result = {'__type__': 'tuple', 'value': [self._convert_for_json(item, options, seen) for item in obj]}
            seen.remove(obj_id)
            return result
        
        # 列表和字典
        elif isinstance(obj, list):
            seen.add(obj_id)
            result = [self._convert_for_json(item, options, seen) for item in obj]
            seen.remove(obj_id)
            return result
        elif isinstance(obj, dict):
            seen.add(obj_id)
            result = {str(k): self._convert_for_json(v, options, seen) for k, v in obj.items()}
            seen.remove(obj_id)
            return result
        
        # 枚举
        elif isinstance(obj, Enum):
            return {'__type__': 'enum', 'class': f"{obj.__class__.__module__}.{obj.__class__.__name__}", 'value': obj.value}
        
        # 数据类
        elif is_dataclass(obj):
            seen.add(obj_id)
            field_dict = {}
            for field in fields(obj):
                field_dict[field.name] = self._convert_for_json(getattr(obj, field.name), options, seen)
            result = {
                '__type__': 'dataclass',
                'class': f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                'fields': field_dict
            }
            seen.remove(obj_id)
            return result
        
        # 自定义编码器
        obj_type = type(obj)
        if obj_type in options.custom_encoders:
            return options.custom_encoders[obj_type](obj)
        
        # 通用对象
        if hasattr(obj, '__dict__'):
            seen.add(obj_id)
            result = {
                '__type__': 'object',
                'class': f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                'attrs': {k: self._convert_for_json(v, options, seen) for k, v in obj.__dict__.items()}
            }
            seen.remove(obj_id)
            return result
        
        # 无法序列化的对象
        return {'__type__': 'unserializable', 'repr': repr(obj)}
    
    def _postprocess_object(self, obj: Any, target_type: Type[T], options: SerializationOptions) -> T:
        """后处理对象"""
        return self._convert_from_json(obj, options)
    
    def _convert_from_json(self, obj: Any, options: SerializationOptions) -> Any:
        """从JSON格式转换对象"""
        if not isinstance(obj, dict) or '__type__' not in obj:
            if isinstance(obj, list):
                return [self._convert_from_json(item, options) for item in obj]
            elif isinstance(obj, dict):
                return {k: self._convert_from_json(v, options) for k, v in obj.items()}
            else:
                return obj
        
        obj_type = obj['__type__']
        
        # 日期时间类型
        if obj_type == 'datetime':
            return datetime.fromisoformat(obj['value'])
        elif obj_type == 'date':
            return date.fromisoformat(obj['value'])
        elif obj_type == 'time':
            return time.fromisoformat(obj['value'])
        elif obj_type == 'timedelta':
            return timedelta(seconds=obj['value'])
        
        # 数值类型
        elif obj_type == 'decimal':
            return Decimal(obj['value'])
        elif obj_type == 'complex':
            return complex(obj['real'], obj['imag'])
        
        # UUID
        elif obj_type == 'uuid':
            return uuid.UUID(obj['value'])
        
        # 字节类型
        elif obj_type == 'bytes':
            return base64.b64decode(obj['value'])
        elif obj_type == 'bytearray':
            return bytearray(base64.b64decode(obj['value']))
        
        # 集合类型
        elif obj_type == 'set':
            return set(self._convert_from_json(item, options) for item in obj['value'])
        elif obj_type == 'frozenset':
            return frozenset(self._convert_from_json(item, options) for item in obj['value'])
        elif obj_type == 'tuple':
            return tuple(self._convert_from_json(item, options) for item in obj['value'])
        
        # 枚举
        elif obj_type == 'enum':
            # 动态导入枚举类
            class_path = obj['class']
            module_name, class_name = class_path.rsplit('.', 1)
            try:
                import importlib
                module = importlib.import_module(module_name)
                enum_class = getattr(module, class_name)
                return enum_class(obj['value'])
            except Exception:
                return obj['value']  # 回退到原始值
        
        # 其他类型
        else:
            return obj  # 保持原样
    
    def _json_default(self, obj: Any) -> Any:
        """JSON默认编码器"""
        # 这个方法不应该被调用，因为我们在预处理中处理了所有类型
        return str(obj)
    
    def _json_object_hook(self, obj: Dict[str, Any]) -> Any:
        """JSON对象钩子"""
        # 在反序列化时处理特殊对象
        return obj
    
    def _compress(self, data: bytes, compression: CompressionType, level: int) -> bytes:
        """压缩数据"""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=level)
        elif compression == CompressionType.BZIP2:
            return bz2.compress(data, compresslevel=level)
        elif compression == CompressionType.LZMA:
            return lzma.compress(data, preset=level)
        else:
            raise CompressionError(f"Unsupported compression type: {compression}")
    
    def _decompress(self, data: bytes, compression: CompressionType) -> bytes:
        """解压缩数据"""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.BZIP2:
            return bz2.decompress(data)
        elif compression == CompressionType.LZMA:
            return lzma.decompress(data)
        else:
            raise CompressionError(f"Unsupported compression type: {compression}")
    
    def _calculate_checksum(self, data: bytes) -> str:
        """计算校验和"""
        import hashlib
        return hashlib.md5(data).hexdigest()


class PickleSerializer:
    """Pickle序列化器"""
    
    def serialize(self, obj: Any, options: SerializationOptions) -> SerializationResult:
        """序列化为Pickle"""
        import time
        start_time = time.time()
        
        try:
            if not options.allow_pickle:
                raise SerializationError("Pickle serialization is disabled")
            
            # Pickle序列化
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 压缩
            original_size = len(data)
            compressed_data = self._compress(data, options.compression, options.compression_level)
            
            # 计算校验和
            checksum = self._calculate_checksum(compressed_data)
            
            return SerializationResult(
                data=compressed_data,
                format=SerializationFormat.PICKLE,
                encoding=options.encoding,
                compression=options.compression,
                original_size=original_size,
                compressed_size=len(compressed_data),
                serialization_time_ms=(time.time() - start_time) * 1000,
                checksum=checksum
            )
            
        except Exception as e:
            raise SerializationError(f"Pickle serialization failed: {e}")
    
    def deserialize(self, data: Union[str, bytes], target_type: Type[T], 
                   options: SerializationOptions) -> T:
        """从Pickle反序列化"""
        try:
            if not options.allow_pickle:
                raise DeserializationError("Pickle deserialization is disabled")
            
            # 解压缩
            if isinstance(data, str):
                data = data.encode(options.encoding.value)
            
            decompressed_data = self._decompress(data, options.compression)
            
            # Pickle反序列化
            return pickle.loads(decompressed_data)
            
        except Exception as e:
            raise DeserializationError(f"Pickle deserialization failed: {e}")
    
    def _compress(self, data: bytes, compression: CompressionType, level: int) -> bytes:
        """压缩数据"""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=level)
        elif compression == CompressionType.BZIP2:
            return bz2.compress(data, compresslevel=level)
        elif compression == CompressionType.LZMA:
            return lzma.compress(data, preset=level)
        else:
            raise CompressionError(f"Unsupported compression type: {compression}")
    
    def _decompress(self, data: bytes, compression: CompressionType) -> bytes:
        """解压缩数据"""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.BZIP2:
            return bz2.decompress(data)
        elif compression == CompressionType.LZMA:
            return lzma.decompress(data)
        else:
            raise CompressionError(f"Unsupported compression type: {compression}")
    
    def _calculate_checksum(self, data: bytes) -> str:
        """计算校验和"""
        import hashlib
        return hashlib.md5(data).hexdigest()


class XMLSerializer:
    """XML序列化器"""
    
    def __init__(self):
        if not XML_AVAILABLE:
            raise SerializationError("XML support is not available")
    
    def serialize(self, obj: Any, options: SerializationOptions) -> SerializationResult:
        """序列化为XML"""
        import time
        start_time = time.time()
        
        try:
            # 创建XML元素
            root = ET.Element(options.xml_root_tag)
            self._object_to_xml(obj, root)
            
            # 生成XML字符串
            if options.xml_pretty:
                xml_str = self._prettify_xml(root, options.xml_encoding)
            else:
                xml_str = ET.tostring(root, encoding=options.xml_encoding).decode(options.xml_encoding)
            
            # 编码
            data = xml_str.encode(options.encoding.value)
            
            # 压缩
            original_size = len(data)
            compressed_data = self._compress(data, options.compression, options.compression_level)
            
            # 计算校验和
            checksum = self._calculate_checksum(compressed_data)
            
            return SerializationResult(
                data=compressed_data,
                format=SerializationFormat.XML,
                encoding=options.encoding,
                compression=options.compression,
                original_size=original_size,
                compressed_size=len(compressed_data),
                serialization_time_ms=(time.time() - start_time) * 1000,
                checksum=checksum
            )
            
        except Exception as e:
            raise SerializationError(f"XML serialization failed: {e}")
    
    def deserialize(self, data: Union[str, bytes], target_type: Type[T], 
                   options: SerializationOptions) -> T:
        """从XML反序列化"""
        try:
            # 解压缩
            if isinstance(data, bytes):
                decompressed_data = self._decompress(data, options.compression)
                xml_str = decompressed_data.decode(options.encoding.value)
            else:
                xml_str = data
            
            # 解析XML
            root = ET.fromstring(xml_str)
            
            # 转换为对象
            return self._xml_to_object(root)
            
        except Exception as e:
            raise DeserializationError(f"XML deserialization failed: {e}")
    
    def _object_to_xml(self, obj: Any, parent: ET.Element) -> None:
        """将对象转换为XML"""
        if obj is None:
            elem = ET.SubElement(parent, "null")
        elif isinstance(obj, bool):
            elem = ET.SubElement(parent, "boolean")
            elem.text = str(obj).lower()
        elif isinstance(obj, int):
            elem = ET.SubElement(parent, "integer")
            elem.text = str(obj)
        elif isinstance(obj, float):
            elem = ET.SubElement(parent, "float")
            elem.text = str(obj)
        elif isinstance(obj, str):
            elem = ET.SubElement(parent, "string")
            elem.text = obj
        elif isinstance(obj, (list, tuple)):
            elem = ET.SubElement(parent, "array")
            elem.set("type", "list" if isinstance(obj, list) else "tuple")
            for item in obj:
                item_elem = ET.SubElement(elem, "item")
                self._object_to_xml(item, item_elem)
        elif isinstance(obj, dict):
            elem = ET.SubElement(parent, "object")
            for key, value in obj.items():
                item_elem = ET.SubElement(elem, "property")
                item_elem.set("name", str(key))
                self._object_to_xml(value, item_elem)
        else:
            # 复杂对象
            elem = ET.SubElement(parent, "complex")
            elem.set("type", type(obj).__name__)
            elem.text = str(obj)
    
    def _xml_to_object(self, elem: ET.Element) -> Any:
        """将XML转换为对象"""
        if elem.tag == "null":
            return None
        elif elem.tag == "boolean":
            return elem.text.lower() == "true"
        elif elem.tag == "integer":
            return int(elem.text)
        elif elem.tag == "float":
            return float(elem.text)
        elif elem.tag == "string":
            return elem.text or ""
        elif elem.tag == "array":
            items = [self._xml_to_object(child) for child in elem]
            if elem.get("type") == "tuple":
                return tuple(items)
            else:
                return items
        elif elem.tag == "object":
            obj = {}
            for child in elem:
                if child.tag == "property":
                    name = child.get("name")
                    value = self._xml_to_object(child[0]) if len(child) > 0 else None
                    obj[name] = value
            return obj
        elif elem.tag == "complex":
            return elem.text
        else:
            # 递归处理子元素
            if len(elem) == 0:
                return elem.text
            elif len(elem) == 1:
                return self._xml_to_object(elem[0])
            else:
                return [self._xml_to_object(child) for child in elem]
    
    def _prettify_xml(self, elem: ET.Element, encoding: str) -> str:
        """美化XML"""
        rough_string = ET.tostring(elem, encoding)
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ", encoding=encoding).decode(encoding)
    
    def _compress(self, data: bytes, compression: CompressionType, level: int) -> bytes:
        """压缩数据"""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=level)
        elif compression == CompressionType.BZIP2:
            return bz2.compress(data, compresslevel=level)
        elif compression == CompressionType.LZMA:
            return lzma.compress(data, preset=level)
        else:
            raise CompressionError(f"Unsupported compression type: {compression}")
    
    def _decompress(self, data: bytes, compression: CompressionType) -> bytes:
        """解压缩数据"""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.BZIP2:
            return bz2.decompress(data)
        elif compression == CompressionType.LZMA:
            return lzma.decompress(data)
        else:
            raise CompressionError(f"Unsupported compression type: {compression}")
    
    def _calculate_checksum(self, data: bytes) -> str:
        """计算校验和"""
        import hashlib
        return hashlib.md5(data).hexdigest()


class SerializationManager:
    """序列化管理器"""
    
    def __init__(self):
        self._serializers: Dict[SerializationFormat, SerializerProtocol] = {
            SerializationFormat.JSON: JSONSerializer(),
            SerializationFormat.PICKLE: PickleSerializer(),
        }
        
        # 可选序列化器
        if XML_AVAILABLE:
            self._serializers[SerializationFormat.XML] = XMLSerializer()
        
        # 统计信息
        self._stats = {
            'serializations': 0,
            'deserializations': 0,
            'errors': 0,
            'total_time_ms': 0.0
        }
    
    def register_serializer(self, format: SerializationFormat, serializer: SerializerProtocol) -> None:
        """注册序列化器"""
        self._serializers[format] = serializer
    
    def serialize(self, obj: Any, options: Optional[SerializationOptions] = None) -> SerializationResult:
        """序列化对象"""
        if options is None:
            options = SerializationOptions()
        
        import time
        start_time = time.time()
        
        try:
            # 获取序列化器
            serializer = self._serializers.get(options.format)
            if not serializer:
                raise SerializationError(f"No serializer found for format: {options.format}")
            
            # 验证大小限制
            self._validate_size_limits(obj, options)
            
            # 序列化
            result = serializer.serialize(obj, options)
            
            # 更新统计
            self._stats['serializations'] += 1
            self._stats['total_time_ms'] += result.serialization_time_ms
            
            # 发布事件
            emit_business_event(
                EventType.SERIALIZATION_COMPLETED,
                "serialization_management",
                data={
                    'format': options.format.value,
                    'original_size': result.original_size,
                    'compressed_size': result.compressed_size,
                    'compression_ratio': result.compression_ratio,
                    'time_ms': result.serialization_time_ms
                }
            )
            
            return result
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Serialization failed: {e}")
            raise
        
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self._stats['total_time_ms'] += elapsed_ms
    
    def deserialize(self, data: Union[str, bytes], target_type: Type[T], 
                   options: Optional[SerializationOptions] = None) -> T:
        """反序列化数据"""
        if options is None:
            options = SerializationOptions()
        
        import time
        start_time = time.time()
        
        try:
            # 获取序列化器
            serializer = self._serializers.get(options.format)
            if not serializer:
                raise DeserializationError(f"No serializer found for format: {options.format}")
            
            # 反序列化
            result = serializer.deserialize(data, target_type, options)
            
            # 更新统计
            self._stats['deserializations'] += 1
            
            # 发布事件
            emit_business_event(
                EventType.DESERIALIZATION_COMPLETED,
                "serialization_management",
                data={
                    'format': options.format.value,
                    'target_type': target_type.__name__ if hasattr(target_type, '__name__') else str(target_type)
                }
            )
            
            return result
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Deserialization failed: {e}")
            raise
        
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self._stats['total_time_ms'] += elapsed_ms
    
    def serialize_to_file(self, obj: Any, file_path: Union[str, Path], 
                         options: Optional[SerializationOptions] = None) -> SerializationResult:
        """序列化到文件"""
        result = self.serialize(obj, options)
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(result.data, str):
            with open(file_path, 'w', encoding=options.encoding.value if options else 'utf-8') as f:
                f.write(result.data)
        else:
            with open(file_path, 'wb') as f:
                f.write(result.data)
        
        return result
    
    def deserialize_from_file(self, file_path: Union[str, Path], target_type: Type[T], 
                             options: Optional[SerializationOptions] = None) -> T:
        """从文件反序列化"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DeserializationError(f"File not found: {file_path}")
        
        # 根据文件扩展名推断格式
        if options is None:
            options = SerializationOptions()
            if file_path.suffix.lower() == '.json':
                options.format = SerializationFormat.JSON
            elif file_path.suffix.lower() == '.xml':
                options.format = SerializationFormat.XML
            elif file_path.suffix.lower() in ['.pkl', '.pickle']:
                options.format = SerializationFormat.PICKLE
                options.allow_pickle = True
        
        # 读取文件
        if options.format == SerializationFormat.PICKLE or file_path.suffix.lower() in ['.pkl', '.pickle']:
            with open(file_path, 'rb') as f:
                data = f.read()
        else:
            with open(file_path, 'r', encoding=options.encoding.value) as f:
                data = f.read()
        
        return self.deserialize(data, target_type, options)
    
    def _validate_size_limits(self, obj: Any, options: SerializationOptions) -> None:
        """验证大小限制"""
        try:
            # 估算对象大小
            import sys
            size_bytes = sys.getsizeof(obj)
            max_size_bytes = options.max_size_mb * 1024 * 1024
            
            if size_bytes > max_size_bytes:
                raise SerializationError(f"Object size ({size_bytes} bytes) exceeds limit ({max_size_bytes} bytes)")
        except Exception:
            # 如果无法计算大小，跳过验证
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()
    
    def get_supported_formats(self) -> List[SerializationFormat]:
        """获取支持的格式"""
        return list(self._serializers.keys())


# 序列化装饰器
def serializable(format: SerializationFormat = SerializationFormat.JSON, 
                options: Optional[SerializationOptions] = None):
    """序列化装饰器"""
    def decorator(cls: Type) -> Type:
        # 添加序列化方法
        def serialize(self, **kwargs) -> SerializationResult:
            manager = get_default_serialization_manager()
            if not manager:
                raise SerializationError("No serialization manager available")
            
            serialize_options = options or SerializationOptions(format=format)
            # 合并kwargs到选项
            for key, value in kwargs.items():
                if hasattr(serialize_options, key):
                    setattr(serialize_options, key, value)
            
            return manager.serialize(self, serialize_options)
        
        def deserialize(cls, data: Union[str, bytes], **kwargs) -> 'cls':
            manager = get_default_serialization_manager()
            if not manager:
                raise DeserializationError("No serialization manager available")
            
            deserialize_options = options or SerializationOptions(format=format)
            # 合并kwargs到选项
            for key, value in kwargs.items():
                if hasattr(deserialize_options, key):
                    setattr(deserialize_options, key, value)
            
            return manager.deserialize(data, cls, deserialize_options)
        
        # 添加方法到类
        cls.serialize = serialize
        cls.deserialize = classmethod(deserialize)
        
        return cls
    
    return decorator


# 全局序列化管理器
_default_serialization_manager: Optional[SerializationManager] = None


def initialize_serialization() -> SerializationManager:
    """初始化序列化管理器"""
    global _default_serialization_manager
    _default_serialization_manager = SerializationManager()
    return _default_serialization_manager


def get_default_serialization_manager() -> Optional[SerializationManager]:
    """获取默认序列化管理器"""
    return _default_serialization_manager


# 便捷函数
def serialize(obj: Any, format: SerializationFormat = SerializationFormat.JSON, 
             **kwargs) -> SerializationResult:
    """序列化对象"""
    manager = get_default_serialization_manager()
    if not manager:
        manager = initialize_serialization()
    
    options = SerializationOptions(format=format)
    for key, value in kwargs.items():
        if hasattr(options, key):
            setattr(options, key, value)
    
    return manager.serialize(obj, options)


def deserialize(data: Union[str, bytes], target_type: Type[T], 
               format: SerializationFormat = SerializationFormat.JSON, 
               **kwargs) -> T:
    """反序列化数据"""
    manager = get_default_serialization_manager()
    if not manager:
        manager = initialize_serialization()
    
    options = SerializationOptions(format=format)
    for key, value in kwargs.items():
        if hasattr(options, key):
            setattr(options, key, value)
    
    return manager.deserialize(data, target_type, options)


def serialize_to_file(obj: Any, file_path: Union[str, Path], 
                     format: Optional[SerializationFormat] = None, 
                     **kwargs) -> SerializationResult:
    """序列化到文件"""
    manager = get_default_serialization_manager()
    if not manager:
        manager = initialize_serialization()
    
    # 根据文件扩展名推断格式
    if format is None:
        file_path_obj = Path(file_path)
        if file_path_obj.suffix.lower() == '.json':
            format = SerializationFormat.JSON
        elif file_path_obj.suffix.lower() == '.xml':
            format = SerializationFormat.XML
        elif file_path_obj.suffix.lower() in ['.pkl', '.pickle']:
            format = SerializationFormat.PICKLE
            kwargs['allow_pickle'] = True
        else:
            format = SerializationFormat.JSON
    
    options = SerializationOptions(format=format)
    for key, value in kwargs.items():
        if hasattr(options, key):
            setattr(options, key, value)
    
    return manager.serialize_to_file(obj, file_path, options)


def deserialize_from_file(file_path: Union[str, Path], target_type: Type[T], 
                         format: Optional[SerializationFormat] = None, 
                         **kwargs) -> T:
    """从文件反序列化"""
    manager = get_default_serialization_manager()
    if not manager:
        manager = initialize_serialization()
    
    options = None
    if format is not None:
        options = SerializationOptions(format=format)
        for key, value in kwargs.items():
            if hasattr(options, key):
                setattr(options, key, value)
    
    return manager.deserialize_from_file(file_path, target_type, options)


def get_serialization_statistics() -> Dict[str, Any]:
    """获取序列化统计"""
    manager = get_default_serialization_manager()
    if manager:
        return manager.get_statistics()
    return {}


def get_supported_formats() -> List[SerializationFormat]:
    """获取支持的格式"""
    manager = get_default_serialization_manager()
    if manager:
        return manager.get_supported_formats()
    return []


# 导出所有类和函数
__all__ = [
    "SerializationFormat",
    "CompressionType",
    "EncodingType",
    "SerializationMode",
    "SerializationOptions",
    "SerializationResult",
    "SerializationError",
    "DeserializationError",
    "CompressionError",
    "EncodingError",
    "SerializerProtocol",
    "JSONSerializer",
    "PickleSerializer",
    "XMLSerializer",
    "SerializationManager",
    "serializable",
    "initialize_serialization",
    "get_default_serialization_manager",
    "serialize",
    "deserialize",
    "serialize_to_file",
    "deserialize_from_file",
    "get_serialization_statistics",
    "get_supported_formats"
]