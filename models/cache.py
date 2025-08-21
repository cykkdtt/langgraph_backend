"""模型缓存系统模块

本模块提供多级缓存、缓存策略和缓存管理功能。
"""

from typing import (
    Dict, Any, Optional, List, Union, Callable, Type, 
    TypeVar, Generic, Tuple, Set
)
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps, lru_cache
import threading
import time
import hashlib
import pickle
import json
import weakref
import logging
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import redis
from sqlalchemy.orm import Session
from sqlalchemy import inspect as sqlalchemy_inspect

# 导入项目模型
from .models import (
    User, Session as ChatSession, Thread, Message, Workflow, 
    WorkflowExecution, WorkflowStep, Memory, MemoryVector, 
    TimeTravel, Attachment, UserPreference, SystemConfig
)
from .events import EventType, emit_business_event


logger = logging.getLogger(__name__)


T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class CacheLevel(Enum):
    """缓存级别枚举"""
    L1_MEMORY = "l1_memory"        # 内存缓存
    L2_REDIS = "l2_redis"          # Redis缓存
    L3_DATABASE = "l3_database"    # 数据库缓存


class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"                    # 最近最少使用
    LFU = "lfu"                    # 最少使用频率
    FIFO = "fifo"                  # 先进先出
    LIFO = "lifo"                  # 后进先出
    TTL = "ttl"                    # 生存时间
    WRITE_THROUGH = "write_through"  # 写穿透
    WRITE_BACK = "write_back"      # 写回
    WRITE_AROUND = "write_around"  # 写绕过


class CacheStatus(Enum):
    """缓存状态枚举"""
    HIT = "hit"                    # 命中
    MISS = "miss"                  # 未命中
    EXPIRED = "expired"            # 过期
    INVALID = "invalid"            # 无效
    EVICTED = "evicted"            # 被驱逐


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def touch(self) -> None:
        """更新访问时间"""
        self.accessed_at = datetime.now()
        self.access_count += 1
    
    def update_value(self, value: Any) -> None:
        """更新值"""
        self.value = value
        self.updated_at = datetime.now()
        self.size = self._calculate_size(value)
    
    def _calculate_size(self, value: Any) -> int:
        """计算值的大小"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return len(str(value))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'key': self.key,
            'created_at': self.created_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'access_count': self.access_count,
            'size': self.size,
            'metadata': self.metadata,
            'tags': list(self.tags)
        }


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_size: int = 0
    entry_count: int = 0
    average_access_time: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit(self, access_time: float = 0.0) -> None:
        """更新命中统计"""
        self.hits += 1
        self._update_hit_rate()
        self._update_average_access_time(access_time)
    
    def update_miss(self, access_time: float = 0.0) -> None:
        """更新未命中统计"""
        self.misses += 1
        self._update_hit_rate()
        self._update_average_access_time(access_time)
    
    def update_eviction(self) -> None:
        """更新驱逐统计"""
        self.evictions += 1
    
    def update_expiration(self) -> None:
        """更新过期统计"""
        self.expirations += 1
    
    def _update_hit_rate(self) -> None:
        """更新命中率"""
        total = self.hits + self.misses
        if total > 0:
            self.hit_rate = self.hits / total
    
    def _update_average_access_time(self, access_time: float) -> None:
        """更新平均访问时间"""
        total_requests = self.hits + self.misses
        if total_requests > 1:
            self.average_access_time = (
                (self.average_access_time * (total_requests - 1) + access_time) / total_requests
            )
        else:
            self.average_access_time = access_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'expirations': self.expirations,
            'total_size': self.total_size,
            'entry_count': self.entry_count,
            'average_access_time': self.average_access_time,
            'hit_rate': self.hit_rate
        }


class CacheBackend(ABC, Generic[K, V]):
    """缓存后端基类"""
    
    def __init__(self, name: str, max_size: int = 1000):
        self.name = name
        self.max_size = max_size
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    @abstractmethod
    def get(self, key: K) -> Optional[V]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set(self, key: K, value: V, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete(self, key: K) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass
    
    @abstractmethod
    def exists(self, key: K) -> bool:
        """检查键是否存在"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """获取缓存大小"""
        pass
    
    @abstractmethod
    def keys(self) -> List[K]:
        """获取所有键"""
        pass
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计"""
        return self.stats
    
    def reset_stats(self) -> None:
        """重置统计"""
        self.stats = CacheStats()


class MemoryCache(CacheBackend[str, Any]):
    """内存缓存"""
    
    def __init__(self, name: str, max_size: int = 1000, 
                 strategy: CacheStrategy = CacheStrategy.LRU):
        super().__init__(name, max_size)
        self.strategy = strategy
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict = OrderedDict()
        self._access_frequency: Dict[str, int] = defaultdict(int)
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()
        
        with self._lock:
            if key not in self._cache:
                self.stats.update_miss(time.time() - start_time)
                return None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if entry.is_expired():
                del self._cache[key]
                self._access_order.pop(key, None)
                self._access_frequency.pop(key, None)
                self.stats.update_expiration()
                self.stats.update_miss(time.time() - start_time)
                return None
            
            # 更新访问信息
            entry.touch()
            self._update_access_order(key)
            self._access_frequency[key] += 1
            
            self.stats.update_hit(time.time() - start_time)
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        with self._lock:
            now = datetime.now()
            expires_at = now + timedelta(seconds=ttl) if ttl else None
            
            if key in self._cache:
                # 更新现有条目
                entry = self._cache[key]
                entry.update_value(value)
                entry.expires_at = expires_at
            else:
                # 创建新条目
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=now,
                    accessed_at=now,
                    updated_at=now,
                    expires_at=expires_at
                )
                
                # 检查是否需要驱逐
                if len(self._cache) >= self.max_size:
                    self._evict()
                
                self._cache[key] = entry
            
            self._update_access_order(key)
            self._access_frequency[key] += 1
            self._update_stats()
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.pop(key, None)
                self._access_frequency.pop(key, None)
                self._update_stats()
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_frequency.clear()
            self._update_stats()
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired():
                self.delete(key)
                return False
            
            return True
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)
    
    def keys(self) -> List[str]:
        """获取所有键"""
        with self._lock:
            # 清理过期键
            expired_keys = [
                key for key, entry in self._cache.items() 
                if entry.is_expired()
            ]
            for key in expired_keys:
                self.delete(key)
            
            return list(self._cache.keys())
    
    def _update_access_order(self, key: str) -> None:
        """更新访问顺序"""
        if self.strategy == CacheStrategy.LRU:
            self._access_order.pop(key, None)
            self._access_order[key] = True
    
    def _evict(self) -> None:
        """驱逐缓存条目"""
        if not self._cache:
            return
        
        evict_key = None
        
        if self.strategy == CacheStrategy.LRU:
            # 驱逐最近最少使用的
            evict_key = next(iter(self._access_order))
        
        elif self.strategy == CacheStrategy.LFU:
            # 驱逐使用频率最低的
            evict_key = min(self._access_frequency, key=self._access_frequency.get)
        
        elif self.strategy == CacheStrategy.FIFO:
            # 驱逐最早添加的
            evict_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].created_at)
        
        elif self.strategy == CacheStrategy.LIFO:
            # 驱逐最晚添加的
            evict_key = max(self._cache.keys(), 
                           key=lambda k: self._cache[k].created_at)
        
        if evict_key:
            self.delete(evict_key)
            self.stats.update_eviction()
    
    def _update_stats(self) -> None:
        """更新统计信息"""
        self.stats.entry_count = len(self._cache)
        self.stats.total_size = sum(entry.size for entry in self._cache.values())
    
    def get_entries_by_tag(self, tag: str) -> List[CacheEntry]:
        """根据标签获取条目"""
        with self._lock:
            return [
                entry for entry in self._cache.values() 
                if tag in entry.tags
            ]
    
    def invalidate_by_tag(self, tag: str) -> int:
        """根据标签失效缓存"""
        with self._lock:
            keys_to_delete = [
                key for key, entry in self._cache.items() 
                if tag in entry.tags
            ]
            
            for key in keys_to_delete:
                self.delete(key)
            
            return len(keys_to_delete)


class RedisCache(CacheBackend[str, Any]):
    """Redis缓存"""
    
    def __init__(self, name: str, redis_client: redis.Redis, 
                 max_size: int = 10000, key_prefix: str = "cache:"):
        super().__init__(name, max_size)
        self.redis = redis_client
        self.key_prefix = key_prefix
    
    def _make_key(self, key: str) -> str:
        """生成Redis键"""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()
        
        try:
            redis_key = self._make_key(key)
            data = self.redis.get(redis_key)
            
            if data is None:
                self.stats.update_miss(time.time() - start_time)
                return None
            
            value = pickle.loads(data)
            self.stats.update_hit(time.time() - start_time)
            return value
        
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self.stats.update_miss(time.time() - start_time)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        try:
            redis_key = self._make_key(key)
            data = pickle.dumps(value)
            
            if ttl:
                self.redis.setex(redis_key, ttl, data)
            else:
                self.redis.set(redis_key, data)
        
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            redis_key = self._make_key(key)
            result = self.redis.delete(redis_key)
            return result > 0
        
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
        
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            redis_key = self._make_key(key)
            return self.redis.exists(redis_key) > 0
        
        except Exception as e:
            logger.error(f"Redis cache exists error: {e}")
            return False
    
    def size(self) -> int:
        """获取缓存大小"""
        try:
            pattern = f"{self.key_prefix}*"
            return len(self.redis.keys(pattern))
        
        except Exception as e:
            logger.error(f"Redis cache size error: {e}")
            return 0
    
    def keys(self) -> List[str]:
        """获取所有键"""
        try:
            pattern = f"{self.key_prefix}*"
            redis_keys = self.redis.keys(pattern)
            return [key.decode().replace(self.key_prefix, '') for key in redis_keys]
        
        except Exception as e:
            logger.error(f"Redis cache keys error: {e}")
            return []


class MultiLevelCache:
    """多级缓存"""
    
    def __init__(self, name: str):
        self.name = name
        self._backends: Dict[CacheLevel, CacheBackend] = {}
        self._lock = threading.RLock()
        self.stats = CacheStats()
    
    def add_backend(self, level: CacheLevel, backend: CacheBackend) -> None:
        """添加缓存后端"""
        with self._lock:
            self._backends[level] = backend
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()
        
        with self._lock:
            # 按级别顺序查找
            for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE]:
                if level not in self._backends:
                    continue
                
                backend = self._backends[level]
                value = backend.get(key)
                
                if value is not None:
                    # 回填到更高级别的缓存
                    self._backfill(key, value, level)
                    self.stats.update_hit(time.time() - start_time)
                    return value
            
            self.stats.update_miss(time.time() - start_time)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        with self._lock:
            # 写入所有级别的缓存
            for backend in self._backends.values():
                backend.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            success = False
            for backend in self._backends.values():
                if backend.delete(key):
                    success = True
            return success
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            for backend in self._backends.values():
                backend.clear()
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            for backend in self._backends.values():
                if backend.exists(key):
                    return True
            return False
    
    def _backfill(self, key: str, value: Any, found_level: CacheLevel) -> None:
        """回填到更高级别的缓存"""
        levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE]
        found_index = levels.index(found_level)
        
        # 回填到更高级别
        for i in range(found_index):
            level = levels[i]
            if level in self._backends:
                self._backends[level].set(key, value)
    
    def get_combined_stats(self) -> Dict[str, CacheStats]:
        """获取组合统计"""
        stats = {'combined': self.stats}
        for level, backend in self._backends.items():
            stats[level.value] = backend.get_stats()
        return stats


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self._caches: Dict[str, Union[CacheBackend, MultiLevelCache]] = {}
        self._lock = threading.RLock()
        self._default_ttl = 3600  # 1小时
        self._key_generators: Dict[Type, Callable] = {}
        self._invalidation_rules: Dict[str, List[str]] = defaultdict(list)
    
    def create_memory_cache(self, name: str, max_size: int = 1000, 
                           strategy: CacheStrategy = CacheStrategy.LRU) -> MemoryCache:
        """创建内存缓存"""
        cache = MemoryCache(name, max_size, strategy)
        self._caches[name] = cache
        return cache
    
    def create_redis_cache(self, name: str, redis_client: redis.Redis, 
                          max_size: int = 10000, key_prefix: str = "cache:") -> RedisCache:
        """创建Redis缓存"""
        cache = RedisCache(name, redis_client, max_size, key_prefix)
        self._caches[name] = cache
        return cache
    
    def create_multi_level_cache(self, name: str, 
                                backends: Dict[CacheLevel, CacheBackend]) -> MultiLevelCache:
        """创建多级缓存"""
        cache = MultiLevelCache(name)
        for level, backend in backends.items():
            cache.add_backend(level, backend)
        self._caches[name] = cache
        return cache
    
    def get_cache(self, name: str) -> Optional[Union[CacheBackend, MultiLevelCache]]:
        """获取缓存"""
        return self._caches.get(name)
    
    def register_key_generator(self, model_class: Type, generator: Callable) -> None:
        """注册键生成器"""
        self._key_generators[model_class] = generator
    
    def generate_key(self, model_class: Type, *args, **kwargs) -> str:
        """生成缓存键"""
        if model_class in self._key_generators:
            return self._key_generators[model_class](*args, **kwargs)
        
        # 默认键生成策略
        key_parts = [model_class.__name__]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def add_invalidation_rule(self, trigger_pattern: str, target_pattern: str) -> None:
        """添加失效规则"""
        self._invalidation_rules[trigger_pattern].append(target_pattern)
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """根据模式失效缓存"""
        total_invalidated = 0
        
        # 直接失效匹配的键
        for cache in self._caches.values():
            if hasattr(cache, 'keys'):
                keys_to_delete = [
                    key for key in cache.keys() 
                    if self._match_pattern(key, pattern)
                ]
                for key in keys_to_delete:
                    cache.delete(key)
                total_invalidated += len(keys_to_delete)
        
        # 应用失效规则
        if pattern in self._invalidation_rules:
            for target_pattern in self._invalidation_rules[pattern]:
                total_invalidated += self.invalidate_by_pattern(target_pattern)
        
        return total_invalidated
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """匹配模式"""
        # 简单的通配符匹配
        if '*' in pattern:
            import fnmatch
            return fnmatch.fnmatch(key, pattern)
        return key == pattern
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有缓存统计"""
        stats = {}
        for name, cache in self._caches.items():
            if hasattr(cache, 'get_combined_stats'):
                stats[name] = cache.get_combined_stats()
            elif hasattr(cache, 'get_stats'):
                stats[name] = cache.get_stats().to_dict()
        return stats
    
    def clear_all(self) -> None:
        """清空所有缓存"""
        for cache in self._caches.values():
            cache.clear()


# 缓存装饰器
def cached(cache_name: str = "default", ttl: Optional[int] = None, 
          key_func: Optional[Callable] = None):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager.generate_key(func, *args, **kwargs)
            
            # 获取缓存
            cache = cache_manager.get_cache(cache_name)
            if not cache:
                return func(*args, **kwargs)
            
            # 尝试从缓存获取
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                emit_business_event(
                    EventType.SYSTEM_STARTUP,  # 使用系统事件类型
                    "cache",
                    data={'action': 'hit', 'key': cache_key, 'cache': cache_name}
                )
                return cached_result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            emit_business_event(
                EventType.SYSTEM_STARTUP,  # 使用系统事件类型
                "cache",
                data={'action': 'set', 'key': cache_key, 'cache': cache_name}
            )
            
            return result
        
        return wrapper
    return decorator


# 全局缓存管理器
cache_manager = CacheManager()


# 便捷函数
def get_cache(name: str) -> Optional[Union[CacheBackend, MultiLevelCache]]:
    """获取缓存"""
    return cache_manager.get_cache(name)


def invalidate_cache(pattern: str) -> int:
    """失效缓存"""
    return cache_manager.invalidate_by_pattern(pattern)


def clear_all_caches() -> None:
    """清空所有缓存"""
    cache_manager.clear_all()


# 导出所有类和函数
__all__ = [
    "CacheLevel",
    "CacheStrategy",
    "CacheStatus",
    "CacheEntry",
    "CacheStats",
    "CacheBackend",
    "MemoryCache",
    "RedisCache",
    "MultiLevelCache",
    "CacheManager",
    "cache_manager",
    "cached",
    "get_cache",
    "invalidate_cache",
    "clear_all_caches"
]