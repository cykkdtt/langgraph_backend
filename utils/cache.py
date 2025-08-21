"""缓存管理工具

提供多种缓存策略、缓存装饰器和缓存统计功能。
"""

import time
import json
import hashlib
import threading
import pickle
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: Optional[int] = None  # 生存时间（秒）
    size: int = 0  # 条目大小（字节）
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def touch(self) -> None:
        """更新访问时间和计数"""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0
    total_size: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """未命中率"""
        return 1.0 - self.hit_rate


class CacheStrategy(ABC):
    """缓存策略抽象基类"""
    
    @abstractmethod
    def should_evict(self, entries: Dict[str, CacheEntry], 
                    new_entry_size: int, max_size: int) -> List[str]:
        """确定应该驱逐哪些条目"""
        pass


class LRUStrategy(CacheStrategy):
    """最近最少使用策略"""
    
    def should_evict(self, entries: Dict[str, CacheEntry], 
                    new_entry_size: int, max_size: int) -> List[str]:
        current_size = sum(entry.size for entry in entries.values())
        
        if current_size + new_entry_size <= max_size:
            return []
        
        # 按访问时间排序，最久未访问的在前
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].accessed_at
        )
        
        to_evict = []
        size_to_free = current_size + new_entry_size - max_size
        freed_size = 0
        
        for key, entry in sorted_entries:
            to_evict.append(key)
            freed_size += entry.size
            if freed_size >= size_to_free:
                break
        
        return to_evict


class LFUStrategy(CacheStrategy):
    """最少使用频率策略"""
    
    def should_evict(self, entries: Dict[str, CacheEntry], 
                    new_entry_size: int, max_size: int) -> List[str]:
        current_size = sum(entry.size for entry in entries.values())
        
        if current_size + new_entry_size <= max_size:
            return []
        
        # 按访问次数排序，访问次数少的在前
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: (x[1].access_count, x[1].accessed_at)
        )
        
        to_evict = []
        size_to_free = current_size + new_entry_size - max_size
        freed_size = 0
        
        for key, entry in sorted_entries:
            to_evict.append(key)
            freed_size += entry.size
            if freed_size >= size_to_free:
                break
        
        return to_evict


class TTLStrategy(CacheStrategy):
    """基于TTL的策略"""
    
    def should_evict(self, entries: Dict[str, CacheEntry], 
                    new_entry_size: int, max_size: int) -> List[str]:
        # 首先移除过期条目
        expired_keys = [key for key, entry in entries.items() if entry.is_expired()]
        
        current_size = sum(
            entry.size for key, entry in entries.items() 
            if key not in expired_keys
        )
        
        if current_size + new_entry_size <= max_size:
            return expired_keys
        
        # 如果还需要更多空间，按剩余TTL排序
        non_expired = {
            key: entry for key, entry in entries.items() 
            if key not in expired_keys
        }
        
        sorted_entries = sorted(
            non_expired.items(),
            key=lambda x: x[1].created_at.timestamp() + (x[1].ttl or float('inf'))
        )
        
        to_evict = expired_keys.copy()
        size_to_free = current_size + new_entry_size - max_size
        freed_size = 0
        
        for key, entry in sorted_entries:
            to_evict.append(key)
            freed_size += entry.size
            if freed_size >= size_to_free:
                break
        
        return to_evict


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_size: int = 1024 * 1024 * 100,  # 100MB
                 strategy: CacheStrategy = None,
                 default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.strategy = strategy or LRUStrategy()
        self.default_ttl = default_ttl
        self.entries: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self.lock = threading.RLock()
        self._cleanup_thread = None
        self._cleanup_interval = 300  # 5分钟
        self._running = False
    
    def start_cleanup(self, interval: int = 300) -> None:
        """启动清理线程"""
        if self._running:
            return
        
        self._cleanup_interval = interval
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired)
        self._cleanup_thread.daemon = True
        self._cleanup_thread.start()
        logger.info(f"Started cache cleanup with {interval}s interval")
    
    def stop_cleanup(self) -> None:
        """停止清理线程"""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        logger.info("Stopped cache cleanup")
    
    def _cleanup_expired(self) -> None:
        """清理过期条目"""
        while self._running:
            try:
                with self.lock:
                    expired_keys = [
                        key for key, entry in self.entries.items() 
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        del self.entries[key]
                        self.stats.expired += 1
                    
                    if expired_keys:
                        self._update_stats()
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
            
            time.sleep(self._cleanup_interval)
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            entry = self.entries.get(key)
            
            if entry is None:
                self.stats.misses += 1
                return None
            
            if entry.is_expired():
                del self.entries[key]
                self.stats.misses += 1
                self.stats.expired += 1
                self._update_stats()
                return None
            
            entry.touch()
            self.stats.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None,
           tags: Optional[List[str]] = None) -> bool:
        """设置缓存值"""
        try:
            # 计算值的大小
            serialized = pickle.dumps(value)
            size = len(serialized)
            
            with self.lock:
                # 检查是否需要驱逐条目
                to_evict = self.strategy.should_evict(self.entries, size, self.max_size)
                
                for evict_key in to_evict:
                    if evict_key in self.entries:
                        del self.entries[evict_key]
                        self.stats.evictions += 1
                
                # 创建新条目
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    accessed_at=datetime.now(),
                    ttl=ttl or self.default_ttl,
                    size=size,
                    tags=tags or []
                )
                
                self.entries[key] = entry
                self._update_stats()
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache entry {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存条目"""
        with self.lock:
            if key in self.entries:
                del self.entries[key]
                self._update_stats()
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.entries.clear()
            self._update_stats()
    
    def clear_by_tags(self, tags: List[str]) -> int:
        """根据标签清除缓存"""
        with self.lock:
            to_delete = []
            for key, entry in self.entries.items():
                if any(tag in entry.tags for tag in tags):
                    to_delete.append(key)
            
            for key in to_delete:
                del self.entries[key]
            
            self._update_stats()
            return len(to_delete)
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        with self.lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                expired=self.stats.expired,
                total_size=self.stats.total_size,
                entry_count=self.stats.entry_count
            )
    
    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """获取缓存键列表"""
        with self.lock:
            keys = list(self.entries.keys())
            
            if pattern:
                import re
                regex = re.compile(pattern)
                keys = [key for key in keys if regex.search(key)]
            
            return keys
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """获取缓存条目信息"""
        with self.lock:
            entry = self.entries.get(key)
            if not entry:
                return None
            
            return {
                "key": entry.key,
                "created_at": entry.created_at.isoformat(),
                "accessed_at": entry.accessed_at.isoformat(),
                "access_count": entry.access_count,
                "ttl": entry.ttl,
                "size": entry.size,
                "tags": entry.tags,
                "is_expired": entry.is_expired(),
                "metadata": entry.metadata
            }
    
    def _update_stats(self) -> None:
        """更新统计信息"""
        self.stats.entry_count = len(self.entries)
        self.stats.total_size = sum(entry.size for entry in self.entries.values())
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        # 创建参数的哈希
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()


# 全局缓存管理器
_global_cache = CacheManager()


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器"""
    return _global_cache


def cache_result(ttl: Optional[int] = None, 
                tags: Optional[List[str]] = None,
                key_func: Optional[Callable] = None):
    """缓存函数结果的装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _global_cache._generate_key(
                    f"{func.__module__}.{func.__name__}", args, kwargs
                )
            
            # 尝试从缓存获取
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            _global_cache.set(cache_key, result, ttl=ttl, tags=tags)
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _global_cache._generate_key(
                    f"{func.__module__}.{func.__name__}", args, kwargs
                )
            
            # 尝试从缓存获取
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            _global_cache.set(cache_key, result, ttl=ttl, tags=tags)
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def invalidate_cache(pattern: Optional[str] = None, 
                    tags: Optional[List[str]] = None):
    """缓存失效装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # 清除相关缓存
            if tags:
                _global_cache.clear_by_tags(tags)
            elif pattern:
                keys_to_delete = _global_cache.get_keys(pattern)
                for key in keys_to_delete:
                    _global_cache.delete(key)
            
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # 清除相关缓存
            if tags:
                _global_cache.clear_by_tags(tags)
            elif pattern:
                keys_to_delete = _global_cache.get_keys(pattern)
                for key in keys_to_delete:
                    _global_cache.delete(key)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class DistributedCache:
    """分布式缓存（Redis实现）"""
    
    def __init__(self, redis_client=None, prefix: str = "langgraph:"):
        self.redis = redis_client
        self.prefix = prefix
        self.local_cache = CacheManager(max_size=1024 * 1024 * 10)  # 10MB本地缓存
    
    def _make_key(self, key: str) -> str:
        """生成Redis键"""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        # 首先尝试本地缓存
        local_result = self.local_cache.get(key)
        if local_result is not None:
            return local_result
        
        # 尝试Redis缓存
        if self.redis:
            try:
                redis_key = self._make_key(key)
                data = self.redis.get(redis_key)
                if data:
                    result = pickle.loads(data)
                    # 同步到本地缓存
                    self.local_cache.set(key, result, ttl=300)  # 5分钟本地TTL
                    return result
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        # 设置本地缓存
        self.local_cache.set(key, value, ttl=min(ttl or 3600, 300))
        
        # 设置Redis缓存
        if self.redis:
            try:
                redis_key = self._make_key(key)
                data = pickle.dumps(value)
                if ttl:
                    self.redis.setex(redis_key, ttl, data)
                else:
                    self.redis.set(redis_key, data)
                return True
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                return False
        
        return True
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        # 删除本地缓存
        self.local_cache.delete(key)
        
        # 删除Redis缓存
        if self.redis:
            try:
                redis_key = self._make_key(key)
                return bool(self.redis.delete(redis_key))
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
                return False
        
        return True


def start_cache_monitoring(interval: int = 300) -> None:
    """启动缓存监控"""
    _global_cache.start_cleanup(interval)
    logger.info("Cache monitoring started")


def stop_cache_monitoring() -> None:
    """停止缓存监控"""
    _global_cache.stop_cleanup()
    logger.info("Cache monitoring stopped")


def get_cache_stats() -> Dict[str, Any]:
    """获取缓存统计信息"""
    stats = _global_cache.get_stats()
    return {
        "hits": stats.hits,
        "misses": stats.misses,
        "hit_rate": stats.hit_rate,
        "miss_rate": stats.miss_rate,
        "evictions": stats.evictions,
        "expired": stats.expired,
        "total_size": stats.total_size,
        "entry_count": stats.entry_count,
        "max_size": _global_cache.max_size
    }