"""辅助工具模块

提供UUID生成、日期时间处理、字符串处理、哈希计算、字典合并等通用功能。
"""

import uuid
import hashlib
import json
import re
import secrets
import string
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from urllib.parse import urlparse, parse_qs
import base64
import gzip
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DateFormat(Enum):
    """日期格式枚举"""
    ISO_8601 = "%Y-%m-%dT%H:%M:%S.%fZ"
    ISO_DATE = "%Y-%m-%d"
    ISO_TIME = "%H:%M:%S"
    READABLE = "%Y年%m月%d日 %H:%M:%S"
    FILENAME_SAFE = "%Y%m%d_%H%M%S"
    LOG_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class PaginationInfo:
    """分页信息"""
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_prev: bool
    next_page: Optional[int] = None
    prev_page: Optional[int] = None


class UUIDGenerator:
    """UUID生成器"""
    
    @staticmethod
    def generate_uuid4() -> str:
        """生成UUID4字符串"""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_short_id(length: int = 8) -> str:
        """生成短ID"""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def generate_nanoid(length: int = 21) -> str:
        """生成NanoID"""
        alphabet = string.ascii_letters + string.digits + '-_'
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def is_valid_uuid(uuid_string: str) -> bool:
        """验证UUID格式"""
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def generate_time_based_id() -> str:
        """生成基于时间的ID"""
        timestamp = int(datetime.now().timestamp() * 1000000)
        random_part = secrets.token_hex(4)
        return f"{timestamp:016x}{random_part}"


class DateTimeHelper:
    """日期时间辅助工具"""
    
    @staticmethod
    def now_utc() -> datetime:
        """获取当前UTC时间"""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def now_local() -> datetime:
        """获取当前本地时间"""
        return datetime.now()
    
    @staticmethod
    def format_datetime(dt: datetime, format_type: DateFormat = DateFormat.ISO_8601) -> str:
        """格式化日期时间"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime(format_type.value)
    
    @staticmethod
    def parse_datetime(date_string: str, format_type: DateFormat = DateFormat.ISO_8601) -> Optional[datetime]:
        """解析日期时间字符串"""
        try:
            return datetime.strptime(date_string, format_type.value)
        except ValueError as e:
            logger.warning(f"Failed to parse datetime '{date_string}': {e}")
            return None
    
    @staticmethod
    def parse_iso_datetime(iso_string: str) -> Optional[datetime]:
        """解析ISO格式日期时间"""
        try:
            # 处理不同的ISO格式
            if iso_string.endswith('Z'):
                iso_string = iso_string[:-1] + '+00:00'
            return datetime.fromisoformat(iso_string)
        except ValueError as e:
            logger.warning(f"Failed to parse ISO datetime '{iso_string}': {e}")
            return None
    
    @staticmethod
    def add_timezone(dt: datetime, tz: timezone = timezone.utc) -> datetime:
        """为naive datetime添加时区信息"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=tz)
        return dt
    
    @staticmethod
    def to_timestamp(dt: datetime) -> int:
        """转换为时间戳（毫秒）"""
        return int(dt.timestamp() * 1000)
    
    @staticmethod
    def from_timestamp(timestamp: int) -> datetime:
        """从时间戳创建datetime（毫秒）"""
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
    
    @staticmethod
    def time_ago(dt: datetime) -> str:
        """计算时间差描述"""
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        diff = now - dt
        
        if diff.days > 365:
            years = diff.days // 365
            return f"{years}年前"
        elif diff.days > 30:
            months = diff.days // 30
            return f"{months}个月前"
        elif diff.days > 0:
            return f"{diff.days}天前"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}小时前"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes}分钟前"
        else:
            return "刚刚"
    
    @staticmethod
    def is_business_day(dt: datetime) -> bool:
        """检查是否为工作日"""
        return dt.weekday() < 5  # 0-4 为周一到周五
    
    @staticmethod
    def next_business_day(dt: datetime) -> datetime:
        """获取下一个工作日"""
        next_day = dt + timedelta(days=1)
        while not DateTimeHelper.is_business_day(next_day):
            next_day += timedelta(days=1)
        return next_day


class StringHelper:
    """字符串处理辅助工具"""
    
    @staticmethod
    def slugify(text: str) -> str:
        """转换为URL友好的slug"""
        # 转换为小写
        text = text.lower()
        # 替换空格和特殊字符为连字符
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        # 去除首尾连字符
        return text.strip('-')
    
    @staticmethod
    def truncate(text: str, length: int, suffix: str = "...") -> str:
        """截断字符串"""
        if len(text) <= length:
            return text
        return text[:length - len(suffix)] + suffix
    
    @staticmethod
    def camel_to_snake(name: str) -> str:
        """驼峰命名转下划线命名"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    @staticmethod
    def snake_to_camel(name: str) -> str:
        """下划线命名转驼峰命名"""
        components = name.split('_')
        return components[0] + ''.join(x.capitalize() for x in components[1:])
    
    @staticmethod
    def pascal_case(name: str) -> str:
        """转换为帕斯卡命名（首字母大写的驼峰）"""
        return ''.join(x.capitalize() for x in name.split('_'))
    
    @staticmethod
    def extract_numbers(text: str) -> List[int]:
        """提取字符串中的数字"""
        return [int(x) for x in re.findall(r'\d+', text)]
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """提取字符串中的邮箱地址"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """提取字符串中的URL"""
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(pattern, text)
    
    @staticmethod
    def mask_sensitive_data(text: str, mask_char: str = "*", 
                          visible_chars: int = 4) -> str:
        """遮蔽敏感数据"""
        if len(text) <= visible_chars * 2:
            return mask_char * len(text)
        
        start = text[:visible_chars]
        end = text[-visible_chars:]
        middle = mask_char * (len(text) - visible_chars * 2)
        
        return start + middle + end
    
    @staticmethod
    def generate_random_string(length: int, 
                             include_uppercase: bool = True,
                             include_lowercase: bool = True,
                             include_digits: bool = True,
                             include_symbols: bool = False) -> str:
        """生成随机字符串"""
        chars = ""
        if include_uppercase:
            chars += string.ascii_uppercase
        if include_lowercase:
            chars += string.ascii_lowercase
        if include_digits:
            chars += string.digits
        if include_symbols:
            chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        if not chars:
            raise ValueError("At least one character type must be included")
        
        return ''.join(secrets.choice(chars) for _ in range(length))


class HashHelper:
    """哈希计算辅助工具"""
    
    @staticmethod
    def md5_hash(data: Union[str, bytes]) -> str:
        """计算MD5哈希"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.md5(data).hexdigest()
    
    @staticmethod
    def sha256_hash(data: Union[str, bytes]) -> str:
        """计算SHA256哈希"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def sha512_hash(data: Union[str, bytes]) -> str:
        """计算SHA512哈希"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha512(data).hexdigest()
    
    @staticmethod
    def file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
        """计算文件哈希"""
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def generate_checksum(data: Dict[str, Any]) -> str:
        """生成数据校验和"""
        # 将字典转换为排序的JSON字符串
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return HashHelper.sha256_hash(json_str)


class DictHelper:
    """字典处理辅助工具"""
    
    @staticmethod
    def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = DictHelper.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """扁平化嵌套字典"""
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(DictHelper.flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    @staticmethod
    def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
        """反扁平化字典"""
        result = {}
        
        for key, value in d.items():
            keys = key.split(sep)
            current = result
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        return result
    
    @staticmethod
    def filter_dict(d: Dict[str, Any], 
                   predicate: Callable[[str, Any], bool]) -> Dict[str, Any]:
        """过滤字典"""
        return {k: v for k, v in d.items() if predicate(k, v)}
    
    @staticmethod
    def map_dict_values(d: Dict[str, Any], 
                       mapper: Callable[[Any], Any]) -> Dict[str, Any]:
        """映射字典值"""
        return {k: mapper(v) for k, v in d.items()}
    
    @staticmethod
    def get_nested_value(d: Dict[str, Any], key_path: str, 
                        default: Any = None, sep: str = '.') -> Any:
        """获取嵌套字典值"""
        keys = key_path.split(sep)
        current = d
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    @staticmethod
    def set_nested_value(d: Dict[str, Any], key_path: str, 
                        value: Any, sep: str = '.') -> None:
        """设置嵌套字典值"""
        keys = key_path.split(sep)
        current = d
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value


class DataConverter:
    """数据转换工具"""
    
    @staticmethod
    def to_json(obj: Any, ensure_ascii: bool = False, indent: int = None) -> str:
        """转换为JSON字符串"""
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Decimal):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            elif hasattr(obj, '_asdict'):
                return obj._asdict()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(obj, default=json_serializer, 
                         ensure_ascii=ensure_ascii, indent=indent)
    
    @staticmethod
    def from_json(json_str: str) -> Any:
        """从JSON字符串解析"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
    
    @staticmethod
    def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
        """数据类转字典"""
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise TypeError(f"Object of type {type(obj)} cannot be converted to dict")
    
    @staticmethod
    def compress_data(data: bytes) -> bytes:
        """压缩数据"""
        return gzip.compress(data)
    
    @staticmethod
    def decompress_data(compressed_data: bytes) -> bytes:
        """解压数据"""
        return gzip.decompress(compressed_data)
    
    @staticmethod
    def serialize_object(obj: Any) -> bytes:
        """序列化对象"""
        return pickle.dumps(obj)
    
    @staticmethod
    def deserialize_object(data: bytes) -> Any:
        """反序列化对象"""
        return pickle.loads(data)
    
    @staticmethod
    def base64_encode(data: Union[str, bytes]) -> str:
        """Base64编码"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return base64.b64encode(data).decode('utf-8')
    
    @staticmethod
    def base64_decode(encoded_data: str) -> bytes:
        """Base64解码"""
        return base64.b64decode(encoded_data)


class PaginationHelper:
    """分页辅助工具"""
    
    @staticmethod
    def calculate_pagination(page: int, page_size: int, total_items: int) -> PaginationInfo:
        """计算分页信息"""
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 10
        
        total_pages = (total_items + page_size - 1) // page_size
        
        if page > total_pages and total_pages > 0:
            page = total_pages
        
        has_next = page < total_pages
        has_prev = page > 1
        
        return PaginationInfo(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev,
            next_page=page + 1 if has_next else None,
            prev_page=page - 1 if has_prev else None
        )
    
    @staticmethod
    def get_offset_limit(page: int, page_size: int) -> tuple[int, int]:
        """获取偏移量和限制数"""
        offset = (page - 1) * page_size
        return offset, page_size


class URLHelper:
    """URL处理辅助工具"""
    
    @staticmethod
    def parse_url(url: str) -> Dict[str, Any]:
        """解析URL"""
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        return {
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "path": parsed.path,
            "params": parsed.params,
            "query": parsed.query,
            "fragment": parsed.fragment,
            "query_params": query_params,
            "hostname": parsed.hostname,
            "port": parsed.port,
            "username": parsed.username,
            "password": parsed.password
        }
    
    @staticmethod
    def build_url(base_url: str, path: str = None, 
                 query_params: Dict[str, Any] = None) -> str:
        """构建URL"""
        url = base_url.rstrip('/')
        
        if path:
            url += '/' + path.lstrip('/')
        
        if query_params:
            query_string = '&'.join(
                f"{k}={v}" for k, v in query_params.items() if v is not None
            )
            if query_string:
                url += '?' + query_string
        
        return url
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """验证URL格式"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


# 便捷函数
def generate_id() -> str:
    """生成ID"""
    return UUIDGenerator.generate_uuid4()


def generate_short_id(length: int = 8) -> str:
    """生成短ID"""
    return UUIDGenerator.generate_short_id(length)


def now_utc() -> datetime:
    """获取当前UTC时间"""
    return DateTimeHelper.now_utc()


def format_datetime(dt: datetime, format_type: DateFormat = DateFormat.ISO_8601) -> str:
    """格式化日期时间"""
    return DateTimeHelper.format_datetime(dt, format_type)


def slugify(text: str) -> str:
    """转换为URL友好的slug"""
    return StringHelper.slugify(text)


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并字典"""
    return DictHelper.deep_merge(dict1, dict2)


def calculate_pagination(page: int, page_size: int, total_items: int) -> PaginationInfo:
    """计算分页信息"""
    return PaginationHelper.calculate_pagination(page, page_size, total_items)