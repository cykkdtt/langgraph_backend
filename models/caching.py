"""模型缓存系统模块

本模块提供数据缓存、查询缓存和分布式缓存功能。
"""

from typing import (
    Dict, Any, Optional, List, Union, Callable, Type, 
    Tuple, Set, ClassVar, Protocol, TypeVar, Generic,
    NamedTuple, AsyncGenerator, Awaitable, Iterator
)
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps, cached_property
import logging
import time
import threading
import hashlib
import pickle
import json
import weakref
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Redis imports (optional)
try:
    import redis
    from redis.sentinel import Sentinel
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    Sentinel = None
    REDIS_AVAILABLE = False

# Memcached imports (optional)
try:
    import pymemcache
    from pymemcache.client.base import Client as MemcachedClient
    MEMCACHED_AVAILABLE = True
except ImportError:
    pymemcache = None
    MemcachedClient = None
    MEMCACHED_AVAILABLE = False

# SQLAlchemy imports
try:
    from sqlalchemy import (
        text, inspect, MetaData, Table, Column,
        Integer, String, DateTime, Boolean, Float, Text,
        create_engine, Engine, event
    )
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.sql import select, func
    from sqlalchemy.engine import Connection
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    text = None
    inspect = None
    MetaData = None
    Table = None
    Column = None
    create_engine = None
    Engine = None
    event = None
    Session = None
    sessionmaker = None
    select = None
    func = None
    Connection = None
    SQLALCHEMY_AVAILABLE = False

# 导入项目模块
from .events import EventType, emit_business_event
from .validation import validate_data


logger = logging.getLogger(__name__)


T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class CacheType(Enum):
    """缓存类型枚举"""
    MEMORY = "memory"                      # 内存缓存
    REDIS = "redis"                        # Redis缓存
    MEMCACHED = "memcached"                # Memcached缓存
    DATABASE = "database"                  # 数据库缓存
    HYBRID = "hybrid"                      # 混合缓存


class EvictionPolicy(Enum):
    """淘汰策略枚举"""
    LRU = "lru"                            # 最近最少使用
    LFU = "lfu"                            # 最少使用频率
    FIFO = "fifo"                          # 先进先出
    TTL = "ttl"                            # 生存时间
    RANDOM = "random"                      # 随机淘汰


class CacheLevel(Enum):
    """缓存级别枚举"""
    L1 = "l1"                              # 一级缓存（内存）
    L2 = "l2"                              # 二级缓存（Redis等）
    L3 = "l3"                              # 三级缓存（数据库等）


class SerializationFormat(Enum):
    """序列化格式枚举"""
    PICKLE = "pickle"                      # Pickle序列化
    JSON = "json"                          # JSON序列化
    MSGPACK = "msgpack"                    # MessagePack序列化
    PROTOBUF = "protobuf"                  # Protocol Buffers


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str                               # 缓存键
    value: Any                             # 缓存值
    
    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    accessed_at: datetime = field(default_factory=datetime.now)  # 访问时间
    updated_at: datetime = field(default_factory=datetime.now)   # 更新时间
    expires_at: Optional[datetime] = None  # 过期时间
    
    # 统计信息
    access_count: int = 0                  # 访问次数
    hit_count: int = 0                     # 命中次数
    miss_count: int = 0                    # 未命中次数
    
    # 元数据
    size: Optional[int] = None             # 数据大小（字节）
    tags: Set[str] = field(default_factory=set)  # 标签
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    @property
    def ttl(self) -> Optional[int]:
        """剩余生存时间（秒）"""
        if self.expires_at is None:
            return None
        
        remaining = self.expires_at - datetime.now()
        return max(0, int(remaining.total_seconds()))
    
    def touch(self) -> None:
        """更新访问时间"""
        self.accessed_at = datetime.now()
        self.access_count += 1
    
    def hit(self) -> None:
        """记录命中"""
        self.hit_count += 1
        self.touch()
    
    def miss(self) -> None:
        """记录未命中"""
        self.miss_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'key': self.key,
            'created_at': self.created_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'access_count': self.access_count,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'size': self.size,
            'tags': list(self.tags),
            'metadata': self.metadata,
            'ttl': self.ttl
        }


@dataclass
class CacheStats:
    """缓存统计"""
    # 基本统计
    total_requests: int = 0                # 总请求数
    cache_hits: int = 0                    # 缓存命中数
    cache_misses: int = 0                  # 缓存未命中数
    
    # 操作统计
    gets: int = 0                          # 获取操作数
    sets: int = 0                          # 设置操作数
    deletes: int = 0                       # 删除操作数
    evictions: int = 0                     # 淘汰数
    
    # 大小统计
    current_size: int = 0                  # 当前大小
    max_size: int = 0                      # 最大大小
    total_size: int = 0                    # 总大小
    
    # 时间统计
    total_get_time: float = 0.0            # 总获取时间
    total_set_time: float = 0.0            # 总设置时间
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """未命中率"""
        return 1.0 - self.hit_rate
    
    @property
    def average_get_time(self) -> float:
        """平均获取时间"""
        if self.gets == 0:
            return 0.0
        return self.total_get_time / self.gets
    
    @property
    def average_set_time(self) -> float:
        """平均设置时间"""
        if self.sets == 0:
            return 0.0
        return self.total_set_time / self.sets
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'gets': self.gets,
            'sets': self.sets,
            'deletes': self.deletes,
            'evictions': self.evictions,
            'current_size': self.current_size,
            'max_size': self.max_size,
            'total_size': self.total_size,
            'average_get_time': self.average_get_time,
            'average_set_time': self.average_set_time
        }


@dataclass
class CacheConfig:
    """缓存配置"""
    # 基本配置
    cache_type: CacheType = CacheType.MEMORY  # 缓存类型
    max_size: int = 1000                   # 最大条目数
    max_memory: int = 100 * 1024 * 1024    # 最大内存（字节）
    
    # 过期配置
    default_ttl: Optional[int] = None      # 默认TTL（秒）
    max_ttl: Optional[int] = None          # 最大TTL（秒）
    
    # 淘汰配置
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU  # 淘汰策略
    eviction_batch_size: int = 10          # 批量淘汰大小
    
    # 序列化配置
    serialization_format: SerializationFormat = SerializationFormat.PICKLE  # 序列化格式
    compress: bool = False                 # 是否压缩
    
    # Redis配置
    redis_host: str = "localhost"          # Redis主机
    redis_port: int = 6379                 # Redis端口
    redis_db: int = 0                      # Redis数据库
    redis_password: Optional[str] = None   # Redis密码
    redis_sentinel_hosts: Optional[List[Tuple[str, int]]] = None  # Sentinel主机
    redis_sentinel_service: Optional[str] = None  # Sentinel服务名
    
    # Memcached配置
    memcached_hosts: List[str] = field(default_factory=lambda: ["localhost:11211"])  # Memcached主机
    
    # 性能配置
    enable_stats: bool = True              # 启用统计
    enable_monitoring: bool = True         # 启用监控
    background_cleanup: bool = True        # 后台清理
    cleanup_interval: int = 60             # 清理间隔（秒）
    
    # 分布式配置
    enable_replication: bool = False       # 启用复制
    replication_factor: int = 2            # 复制因子
    consistency_level: str = "eventual"    # 一致性级别


class CacheError(Exception):
    """缓存错误"""
    pass


class CacheKeyError(CacheError):
    """缓存键错误"""
    pass


class CacheSerializationError(CacheError):
    """缓存序列化错误"""
    pass


class CacheConnectionError(CacheError):
    """缓存连接错误"""
    pass


class CacheBackend(ABC):
    """缓存后端抽象基类"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass
    
    @abstractmethod
    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """获取所有键"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """获取缓存大小"""
        pass


class MemoryCache(CacheBackend):
    """内存缓存后端"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
        # 启动后台清理
        if config.background_cleanup:
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True
            )
            self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()
        
        with self._lock:
            self._stats.gets += 1
            self._stats.total_requests += 1
            
            entry = self._cache.get(key)
            if entry is None:
                self._stats.cache_misses += 1
                return None
            
            # 检查过期
            if entry.is_expired:
                del self._cache[key]
                self._stats.cache_misses += 1
                return None
            
            # 更新访问信息
            entry.hit()
            self._stats.cache_hits += 1
            
            # LRU更新
            if self.config.eviction_policy == EvictionPolicy.LRU:
                self._cache.move_to_end(key)
            
            self._stats.total_get_time += time.time() - start_time
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        start_time = time.time()
        
        with self._lock:
            self._stats.sets += 1
            
            # 计算过期时间
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif self.config.default_ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=self.config.default_ttl)
            
            # 创建缓存条目
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at
            )
            
            # 计算大小
            try:
                entry.size = len(pickle.dumps(value))
            except Exception:
                entry.size = 0
            
            # 检查是否需要淘汰
            self._evict_if_needed()
            
            # 添加到缓存
            self._cache[key] = entry
            
            # 更新统计
            self._stats.current_size = len(self._cache)
            self._stats.max_size = max(self._stats.max_size, self._stats.current_size)
            self._stats.total_size += entry.size or 0
            self._stats.total_set_time += time.time() - start_time
            
            return True
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            self._stats.deletes += 1
            
            if key in self._cache:
                entry = self._cache.pop(key)
                self._stats.current_size = len(self._cache)
                self._stats.total_size -= entry.size or 0
                return True
            
            return False
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            
            if entry.is_expired:
                del self._cache[key]
                return False
            
            return True
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._stats.current_size = 0
            self._stats.total_size = 0
    
    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """获取所有键"""
        with self._lock:
            keys = list(self._cache.keys())
            
            if pattern:
                import re
                regex = re.compile(pattern)
                keys = [k for k in keys if regex.match(k)]
            
            return keys
    
    def size(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self._cache)
    
    def _evict_if_needed(self) -> None:
        """如果需要则淘汰条目"""
        # 检查大小限制
        while len(self._cache) >= self.config.max_size:
            self._evict_one()
        
        # 检查内存限制
        current_memory = sum(entry.size or 0 for entry in self._cache.values())
        while current_memory > self.config.max_memory and self._cache:
            self._evict_one()
            current_memory = sum(entry.size or 0 for entry in self._cache.values())
    
    def _evict_one(self) -> None:
        """淘汰一个条目"""
        if not self._cache:
            return
        
        if self.config.eviction_policy == EvictionPolicy.LRU:
            # 淘汰最近最少使用的
            key = next(iter(self._cache))
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            # 淘汰使用频率最低的
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            # 淘汰最早的
            key = next(iter(self._cache))
        elif self.config.eviction_policy == EvictionPolicy.TTL:
            # 淘汰最早过期的
            key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].expires_at or datetime.max
            )
        else:  # RANDOM
            import random
            key = random.choice(list(self._cache.keys()))
        
        del self._cache[key]
        self._stats.evictions += 1
    
    def _cleanup_loop(self) -> None:
        """清理循环"""
        while True:
            try:
                self._cleanup_expired()
                time.sleep(self.config.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def _cleanup_expired(self) -> None:
        """清理过期条目"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1
    
    def get_stats(self) -> CacheStats:
        """获取统计信息"""
        with self._lock:
            stats = CacheStats(
                total_requests=self._stats.total_requests,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                gets=self._stats.gets,
                sets=self._stats.sets,
                deletes=self._stats.deletes,
                evictions=self._stats.evictions,
                current_size=len(self._cache),
                max_size=self._stats.max_size,
                total_size=sum(entry.size or 0 for entry in self._cache.values()),
                total_get_time=self._stats.total_get_time,
                total_set_time=self._stats.total_set_time
            )
            return stats


class RedisCache(CacheBackend):
    """Redis缓存后端"""
    
    def __init__(self, config: CacheConfig):
        if not REDIS_AVAILABLE:
            raise CacheError("Redis is not available. Please install redis-py.")
        
        self.config = config
        self._stats = CacheStats()
        
        # 创建Redis连接
        try:
            if config.redis_sentinel_hosts:
                # 使用Sentinel
                sentinel = Sentinel(config.redis_sentinel_hosts)
                self._redis = sentinel.master_for(
                    config.redis_sentinel_service,
                    password=config.redis_password,
                    db=config.redis_db
                )
            else:
                # 直接连接
                self._redis = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    password=config.redis_password,
                    decode_responses=False
                )
            
            # 测试连接
            self._redis.ping()
            
        except Exception as e:
            raise CacheConnectionError(f"Failed to connect to Redis: {e}")
    
    def _serialize(self, value: Any) -> bytes:
        """序列化值"""
        try:
            if self.config.serialization_format == SerializationFormat.PICKLE:
                return pickle.dumps(value)
            elif self.config.serialization_format == SerializationFormat.JSON:
                return json.dumps(value).encode('utf-8')
            else:
                return pickle.dumps(value)  # 默认使用pickle
        except Exception as e:
            raise CacheSerializationError(f"Failed to serialize value: {e}")
    
    def _deserialize(self, data: bytes) -> Any:
        """反序列化值"""
        try:
            if self.config.serialization_format == SerializationFormat.PICKLE:
                return pickle.loads(data)
            elif self.config.serialization_format == SerializationFormat.JSON:
                return json.loads(data.decode('utf-8'))
            else:
                return pickle.loads(data)  # 默认使用pickle
        except Exception as e:
            raise CacheSerializationError(f"Failed to deserialize value: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()
        
        try:
            self._stats.gets += 1
            self._stats.total_requests += 1
            
            data = self._redis.get(key)
            if data is None:
                self._stats.cache_misses += 1
                return None
            
            value = self._deserialize(data)
            self._stats.cache_hits += 1
            self._stats.total_get_time += time.time() - start_time
            
            return value
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats.cache_misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        start_time = time.time()
        
        try:
            self._stats.sets += 1
            
            data = self._serialize(value)
            
            # 设置TTL
            if ttl is not None:
                result = self._redis.setex(key, ttl, data)
            elif self.config.default_ttl is not None:
                result = self._redis.setex(key, self.config.default_ttl, data)
            else:
                result = self._redis.set(key, data)
            
            self._stats.total_set_time += time.time() - start_time
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            self._stats.deletes += 1
            result = self._redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            return bool(self._redis.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        try:
            self._redis.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """获取所有键"""
        try:
            pattern = pattern or "*"
            keys = self._redis.keys(pattern)
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []
    
    def size(self) -> int:
        """获取缓存大小"""
        try:
            return self._redis.dbsize()
        except Exception as e:
            logger.error(f"Redis size error: {e}")
            return 0
    
    def get_stats(self) -> CacheStats:
        """获取统计信息"""
        try:
            info = self._redis.info()
            
            stats = CacheStats(
                total_requests=self._stats.total_requests,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                gets=self._stats.gets,
                sets=self._stats.sets,
                deletes=self._stats.deletes,
                current_size=info.get('db0', {}).get('keys', 0),
                total_get_time=self._stats.total_get_time,
                total_set_time=self._stats.total_set_time
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return self._stats


class MultiLevelCache:
    """多级缓存"""
    
    def __init__(self, levels: List[Tuple[CacheLevel, CacheBackend]]):
        self.levels = sorted(levels, key=lambda x: x[0].value)  # 按级别排序
        self._stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        self._stats.gets += 1
        self._stats.total_requests += 1
        
        # 从低级别到高级别查找
        for i, (level, backend) in enumerate(self.levels):
            value = backend.get(key)
            if value is not None:
                self._stats.cache_hits += 1
                
                # 回填到更低级别的缓存
                for j in range(i):
                    _, lower_backend = self.levels[j]
                    lower_backend.set(key, value)
                
                return value
        
        self._stats.cache_misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        self._stats.sets += 1
        
        # 设置到所有级别
        success = True
        for level, backend in self.levels:
            if not backend.set(key, value, ttl):
                success = False
        
        return success
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        self._stats.deletes += 1
        
        # 从所有级别删除
        success = True
        for level, backend in self.levels:
            if not backend.delete(key):
                success = False
        
        return success
    
    def clear(self) -> None:
        """清空所有级别的缓存"""
        for level, backend in self.levels:
            backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'overall': self._stats.to_dict(),
            'levels': {}
        }
        
        for level, backend in self.levels:
            if hasattr(backend, 'get_stats'):
                stats['levels'][level.value] = backend.get_stats().to_dict()
        
        return stats


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._backends: Dict[str, CacheBackend] = {}
        self._default_backend: Optional[CacheBackend] = None
        self._lock = threading.RLock()
        
        # 初始化默认后端
        self._initialize_default_backend()
    
    def _initialize_default_backend(self) -> None:
        """初始化默认后端"""
        if self.config.cache_type == CacheType.MEMORY:
            self._default_backend = MemoryCache(self.config)
        elif self.config.cache_type == CacheType.REDIS:
            self._default_backend = RedisCache(self.config)
        elif self.config.cache_type == CacheType.HYBRID:
            # 创建多级缓存
            memory_config = CacheConfig(
                cache_type=CacheType.MEMORY,
                max_size=self.config.max_size // 2
            )
            redis_config = CacheConfig(
                cache_type=CacheType.REDIS,
                redis_host=self.config.redis_host,
                redis_port=self.config.redis_port,
                redis_db=self.config.redis_db,
                redis_password=self.config.redis_password
            )
            
            levels = [
                (CacheLevel.L1, MemoryCache(memory_config)),
                (CacheLevel.L2, RedisCache(redis_config))
            ]
            
            self._default_backend = MultiLevelCache(levels)
        else:
            raise CacheError(f"Unsupported cache type: {self.config.cache_type}")
        
        self._backends['default'] = self._default_backend
    
    def register_backend(self, name: str, backend: CacheBackend) -> None:
        """注册缓存后端"""
        with self._lock:
            self._backends[name] = backend
    
    def get_backend(self, name: str = "default") -> Optional[CacheBackend]:
        """获取缓存后端"""
        return self._backends.get(name)
    
    def get(self, key: str, backend: str = "default") -> Optional[Any]:
        """获取缓存值"""
        cache_backend = self.get_backend(backend)
        if cache_backend:
            return cache_backend.get(key)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
           backend: str = "default") -> bool:
        """设置缓存值"""
        cache_backend = self.get_backend(backend)
        if cache_backend:
            return cache_backend.set(key, value, ttl)
        return False
    
    def delete(self, key: str, backend: str = "default") -> bool:
        """删除缓存值"""
        cache_backend = self.get_backend(backend)
        if cache_backend:
            return cache_backend.delete(key)
        return False
    
    def exists(self, key: str, backend: str = "default") -> bool:
        """检查键是否存在"""
        cache_backend = self.get_backend(backend)
        if cache_backend:
            return cache_backend.exists(key)
        return False
    
    def clear(self, backend: str = "default") -> None:
        """清空缓存"""
        cache_backend = self.get_backend(backend)
        if cache_backend:
            cache_backend.clear()
    
    def clear_all(self) -> None:
        """清空所有缓存"""
        with self._lock:
            for backend in self._backends.values():
                backend.clear()
    
    def keys(self, pattern: Optional[str] = None, backend: str = "default") -> List[str]:
        """获取所有键"""
        cache_backend = self.get_backend(backend)
        if cache_backend:
            return cache_backend.keys(pattern)
        return []
    
    def size(self, backend: str = "default") -> int:
        """获取缓存大小"""
        cache_backend = self.get_backend(backend)
        if cache_backend:
            return cache_backend.size()
        return 0
    
    def get_stats(self, backend: str = "default") -> Dict[str, Any]:
        """获取统计信息"""
        cache_backend = self.get_backend(backend)
        if cache_backend and hasattr(cache_backend, 'get_stats'):
            return cache_backend.get_stats().to_dict()
        return {}
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有后端的统计信息"""
        stats = {}
        with self._lock:
            for name, backend in self._backends.items():
                if hasattr(backend, 'get_stats'):
                    stats[name] = backend.get_stats().to_dict()
        return stats
    
    def invalidate_by_tags(self, tags: Set[str], backend: str = "default") -> int:
        """根据标签失效缓存"""
        # 这是一个简化实现，实际需要存储键-标签映射
        cache_backend = self.get_backend(backend)
        if not cache_backend:
            return 0
        
        # 获取所有键并检查标签（需要扩展CacheEntry支持）
        keys = cache_backend.keys()
        invalidated = 0
        
        for key in keys:
            # 这里需要实际的标签检查逻辑
            # 简化实现：假设键包含标签信息
            if any(tag in key for tag in tags):
                if cache_backend.delete(key):
                    invalidated += 1
        
        return invalidated


# 缓存装饰器
def cached(key_func: Optional[Callable] = None, ttl: Optional[int] = None, 
          backend: str = "default"):
    """缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 默认键生成策略
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
                
                # 生成哈希以避免键过长
                if len(cache_key) > 200:
                    cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            
            # 尝试从缓存获取
            manager = get_default_cache_manager()
            if manager:
                cached_result = manager.get(cache_key, backend)
                if cached_result is not None:
                    return cached_result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            
            if manager:
                manager.set(cache_key, result, ttl, backend)
            
            return result
        
        return wrapper
    return decorator


def cache_invalidate(key_pattern: str, backend: str = "default") -> int:
    """失效缓存"""
    manager = get_default_cache_manager()
    if manager:
        cache_backend = manager.get_backend(backend)
        if cache_backend:
            keys = cache_backend.keys(key_pattern)
            invalidated = 0
            for key in keys:
                if cache_backend.delete(key):
                    invalidated += 1
            return invalidated
    return 0


# 全局缓存管理器
_default_cache_manager: Optional[CacheManager] = None


def initialize_cache(config: Optional[CacheConfig] = None) -> CacheManager:
    """初始化缓存管理器"""
    global _default_cache_manager
    _default_cache_manager = CacheManager(config)
    
    # 发布事件
    emit_business_event(
        EventType.CACHE_INITIALIZED,
        "cache",
        data={'config': config.__dict__ if config else {}}
    )
    
    return _default_cache_manager


def get_default_cache_manager() -> Optional[CacheManager]:
    """获取默认缓存管理器"""
    return _default_cache_manager


# 便捷函数
def cache_get(key: str, backend: str = "default") -> Optional[Any]:
    """获取缓存值"""
    manager = get_default_cache_manager()
    if manager:
        return manager.get(key, backend)
    return None


def cache_set(key: str, value: Any, ttl: Optional[int] = None, 
             backend: str = "default") -> bool:
    """设置缓存值"""
    manager = get_default_cache_manager()
    if manager:
        return manager.set(key, value, ttl, backend)
    return False


def cache_delete(key: str, backend: str = "default") -> bool:
    """删除缓存值"""
    manager = get_default_cache_manager()
    if manager:
        return manager.delete(key, backend)
    return False


def cache_exists(key: str, backend: str = "default") -> bool:
    """检查键是否存在"""
    manager = get_default_cache_manager()
    if manager:
        return manager.exists(key, backend)
    return False


def cache_clear(backend: str = "default") -> None:
    """清空缓存"""
    manager = get_default_cache_manager()
    if manager:
        manager.clear(backend)


def cache_clear_all() -> None:
    """清空所有缓存"""
    manager = get_default_cache_manager()
    if manager:
        manager.clear_all()


def cache_keys(pattern: Optional[str] = None, backend: str = "default") -> List[str]:
    """获取所有键"""
    manager = get_default_cache_manager()
    if manager:
        return manager.keys(pattern, backend)
    return []


def cache_size(backend: str = "default") -> int:
    """获取缓存大小"""
    manager = get_default_cache_manager()
    if manager:
        return manager.size(backend)
    return 0


def cache_stats(backend: str = "default") -> Dict[str, Any]:
    """获取缓存统计"""
    manager = get_default_cache_manager()
    if manager:
        return manager.get_stats(backend)
    return {}


def cache_all_stats() -> Dict[str, Any]:
    """获取所有缓存统计"""
    manager = get_default_cache_manager()
    if manager:
        return manager.get_all_stats()
    return {}


# 导出所有类和函数
__all__ = [
    "CacheType",