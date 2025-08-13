"""
MCP缓存管理器

提供MCP工具调用和资源的智能缓存机制，包括：
- 多级缓存策略（内存、Redis）
- 缓存失效和更新策略
- 性能指标收集
- 缓存预热和清理
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pickle
from contextlib import asynccontextmanager

import redis.asyncio as redis
from pydantic import BaseModel, Field


class CacheLevel(Enum):
    """缓存级别"""
    MEMORY = "memory"
    REDIS = "redis"
    BOTH = "both"


class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    TTL = "ttl"  # 基于时间
    ADAPTIVE = "adaptive"  # 自适应


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age(self) -> float:
        """缓存年龄（秒）"""
        return time.time() - self.created_at


class MCPCacheConfig(BaseModel):
    """MCP缓存配置"""
    # 基本配置
    enabled: bool = Field(default=True, description="是否启用缓存")
    default_ttl: float = Field(default=3600.0, description="默认TTL（秒）")
    max_memory_size: int = Field(default=100 * 1024 * 1024, description="最大内存缓存大小（字节）")
    max_memory_entries: int = Field(default=10000, description="最大内存缓存条目数")
    
    # Redis配置
    redis_enabled: bool = Field(default=True, description="是否启用Redis缓存")
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis连接URL")
    redis_key_prefix: str = Field(default="mcp:cache:", description="Redis键前缀")
    redis_ttl: float = Field(default=7200.0, description="Redis TTL（秒）")
    
    # 缓存策略
    strategy: CacheStrategy = Field(default=CacheStrategy.ADAPTIVE, description="缓存策略")
    eviction_threshold: float = Field(default=0.8, description="驱逐阈值")
    
    # 性能配置
    compression_enabled: bool = Field(default=True, description="是否启用压缩")
    compression_threshold: int = Field(default=1024, description="压缩阈值（字节）")
    batch_size: int = Field(default=100, description="批处理大小")
    
    # 预热配置
    preload_enabled: bool = Field(default=False, description="是否启用预加载")
    preload_patterns: List[str] = Field(default_factory=list, description="预加载模式")


class MCPCacheMetrics:
    """缓存指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        
        self.total_latency = 0.0
        self.redis_latency = 0.0
        self.memory_latency = 0.0
        
        self.memory_size = 0
        self.memory_entries = 0
        self.redis_entries = 0
        
        self.start_time = time.time()
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def average_latency(self) -> float:
        """平均延迟"""
        total_ops = self.hits + self.misses + self.sets
        return self.total_latency / total_ops if total_ops > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "average_latency": self.average_latency,
            "memory_size": self.memory_size,
            "memory_entries": self.memory_entries,
            "redis_entries": self.redis_entries,
            "uptime": time.time() - self.start_time
        }


class MCPMemoryCache:
    """内存缓存实现"""
    
    def __init__(self, config: MCPCacheConfig):
        self.config = config
        self.logger = logging.getLogger("mcp.cache.memory")
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # LRU顺序
        self._lock = asyncio.Lock()
        
        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """启动清理任务"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                await self._cleanup_expired()
                await self._enforce_limits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"缓存清理失败: {e}")
    
    async def _cleanup_expired(self):
        """清理过期条目"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            
            if expired_keys:
                self.logger.debug(f"清理了 {len(expired_keys)} 个过期缓存条目")
    
    async def _enforce_limits(self):
        """强制执行限制"""
        async with self._lock:
            # 检查条目数限制
            if len(self._cache) > self.config.max_memory_entries:
                await self._evict_entries(len(self._cache) - self.config.max_memory_entries)
            
            # 检查内存大小限制
            total_size = sum(entry.size for entry in self._cache.values())
            if total_size > self.config.max_memory_size:
                await self._evict_by_size(total_size - self.config.max_memory_size)
    
    async def _evict_entries(self, count: int):
        """驱逐指定数量的条目"""
        if self.config.strategy == CacheStrategy.LRU:
            # 驱逐最近最少使用的条目
            to_evict = self._access_order[:count]
        elif self.config.strategy == CacheStrategy.LFU:
            # 驱逐使用频率最低的条目
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].access_count
            )
            to_evict = [key for key, _ in sorted_entries[:count]]
        else:
            # 默认使用LRU
            to_evict = self._access_order[:count]
        
        for key in to_evict:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
    
    async def _evict_by_size(self, target_size: int):
        """按大小驱逐条目"""
        evicted_size = 0
        to_evict = []
        
        # 按访问顺序驱逐
        for key in self._access_order:
            if key in self._cache:
                entry = self._cache[key]
                to_evict.append(key)
                evicted_size += entry.size
                
                if evicted_size >= target_size:
                    break
        
        for key in to_evict:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
    
    def _update_access_order(self, key: str):
        """更新访问顺序"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _calculate_size(self, value: Any) -> int:
        """计算值的大小"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return len(str(value).encode('utf-8'))
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None
            
            # 更新访问信息
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._update_access_order(key)
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        try:
            size = self._calculate_size(value)
            
            async with self._lock:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    ttl=ttl or self.config.default_ttl,
                    size=size
                )
                
                self._cache[key] = entry
                self._update_access_order(key)
                
                return True
                
        except Exception as e:
            self.logger.error(f"设置内存缓存失败: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
    
    async def clear(self):
        """清空缓存"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_size = sum(entry.size for entry in self._cache.values())
        return {
            "entries": len(self._cache),
            "size": total_size,
            "max_entries": self.config.max_memory_entries,
            "max_size": self.config.max_memory_size,
            "utilization": len(self._cache) / self.config.max_memory_entries,
            "size_utilization": total_size / self.config.max_memory_size
        }


class MCPRedisCache:
    """Redis缓存实现"""
    
    def __init__(self, config: MCPCacheConfig):
        self.config = config
        self.logger = logging.getLogger("mcp.cache.redis")
        
        self._redis: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """连接Redis"""
        try:
            self._redis = redis.from_url(self.config.redis_url)
            await self._redis.ping()
            self._connected = True
            self.logger.info("Redis缓存连接成功")
            return True
        except Exception as e:
            self.logger.error(f"Redis缓存连接失败: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """断开Redis连接"""
        if self._redis:
            await self._redis.close()
            self._connected = False
    
    def _make_key(self, key: str) -> str:
        """生成Redis键"""
        return f"{self.config.redis_key_prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """序列化值"""
        try:
            data = pickle.dumps(value)
            if self.config.compression_enabled and len(data) > self.config.compression_threshold:
                import gzip
                data = gzip.compress(data)
                return b"compressed:" + data
            return b"raw:" + data
        except Exception as e:
            self.logger.error(f"序列化失败: {e}")
            raise
    
    def _deserialize(self, data: bytes) -> Any:
        """反序列化值"""
        try:
            if data.startswith(b"compressed:"):
                import gzip
                data = gzip.decompress(data[11:])
            elif data.startswith(b"raw:"):
                data = data[4:]
            else:
                # 兼容旧格式
                pass
            
            return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"反序列化失败: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self._connected or not self._redis:
            return None
        
        try:
            redis_key = self._make_key(key)
            data = await self._redis.get(redis_key)
            
            if data is None:
                return None
            
            return self._deserialize(data)
            
        except Exception as e:
            self.logger.error(f"Redis获取失败: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        if not self._connected or not self._redis:
            return False
        
        try:
            redis_key = self._make_key(key)
            data = self._serialize(value)
            
            ttl_seconds = int(ttl or self.config.redis_ttl)
            await self._redis.setex(redis_key, ttl_seconds, data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Redis设置失败: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        if not self._connected or not self._redis:
            return False
        
        try:
            redis_key = self._make_key(key)
            await self._redis.delete(redis_key)
            return True
            
        except Exception as e:
            self.logger.error(f"Redis删除失败: {e}")
            return False
    
    async def clear(self, pattern: str = "*"):
        """清空缓存"""
        if not self._connected or not self._redis:
            return
        
        try:
            pattern_key = self._make_key(pattern)
            keys = await self._redis.keys(pattern_key)
            if keys:
                await self._redis.delete(*keys)
                
        except Exception as e:
            self.logger.error(f"Redis清空失败: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._connected or not self._redis:
            return {"connected": False}
        
        try:
            info = await self._redis.info("memory")
            pattern_key = self._make_key("*")
            keys = await self._redis.keys(pattern_key)
            
            return {
                "connected": True,
                "entries": len(keys),
                "memory_usage": info.get("used_memory", 0),
                "max_memory": info.get("maxmemory", 0)
            }
            
        except Exception as e:
            self.logger.error(f"获取Redis统计失败: {e}")
            return {"connected": False, "error": str(e)}


class MCPCacheManager:
    """MCP缓存管理器"""
    
    def __init__(self, config: Optional[MCPCacheConfig] = None):
        self.config = config or MCPCacheConfig()
        self.logger = logging.getLogger("mcp.cache.manager")
        
        # 缓存实例
        self.memory_cache = MCPMemoryCache(self.config)
        self.redis_cache = MCPRedisCache(self.config) if self.config.redis_enabled else None
        
        # 指标
        self.metrics = MCPCacheMetrics()
        
        # 初始化
        self._initialized = False
    
    async def initialize(self) -> bool:
        """初始化缓存管理器"""
        try:
            if self.redis_cache:
                await self.redis_cache.connect()
            
            self._initialized = True
            self.logger.info("缓存管理器初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"缓存管理器初始化失败: {e}")
            return False
    
    async def cleanup(self):
        """清理资源"""
        if self.redis_cache:
            await self.redis_cache.disconnect()
        
        await self.memory_cache.clear()
        self._initialized = False
    
    def _make_cache_key(self, server_name: str, operation: str, **kwargs) -> str:
        """生成缓存键"""
        # 创建一个确定性的键
        key_parts = [server_name, operation]
        
        # 添加参数
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (dict, list)):
                v = json.dumps(v, sort_keys=True)
            key_parts.append(f"{k}:{v}")
        
        key_string = "|".join(key_parts)
        
        # 使用哈希避免键过长
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{server_name}:{operation}:{key_hash}"
    
    async def get(self, server_name: str, operation: str, **kwargs) -> Optional[Any]:
        """获取缓存值"""
        if not self.config.enabled or not self._initialized:
            return None
        
        start_time = time.time()
        cache_key = self._make_cache_key(server_name, operation, **kwargs)
        
        try:
            # 先尝试内存缓存
            value = await self.memory_cache.get(cache_key)
            if value is not None:
                self.metrics.hits += 1
                self.metrics.memory_latency += time.time() - start_time
                return value
            
            # 再尝试Redis缓存
            if self.redis_cache:
                value = await self.redis_cache.get(cache_key)
                if value is not None:
                    # 回写到内存缓存
                    await self.memory_cache.set(cache_key, value)
                    
                    self.metrics.hits += 1
                    self.metrics.redis_latency += time.time() - start_time
                    return value
            
            # 缓存未命中
            self.metrics.misses += 1
            return None
            
        except Exception as e:
            self.logger.error(f"缓存获取失败: {e}")
            self.metrics.errors += 1
            return None
        finally:
            self.metrics.total_latency += time.time() - start_time
    
    async def set(self, server_name: str, operation: str, value: Any, ttl: Optional[float] = None, **kwargs) -> bool:
        """设置缓存值"""
        if not self.config.enabled or not self._initialized:
            return False
        
        start_time = time.time()
        cache_key = self._make_cache_key(server_name, operation, **kwargs)
        
        try:
            success = True
            
            # 设置内存缓存
            if not await self.memory_cache.set(cache_key, value, ttl):
                success = False
            
            # 设置Redis缓存
            if self.redis_cache:
                if not await self.redis_cache.set(cache_key, value, ttl):
                    success = False
            
            if success:
                self.metrics.sets += 1
            else:
                self.metrics.errors += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"缓存设置失败: {e}")
            self.metrics.errors += 1
            return False
        finally:
            self.metrics.total_latency += time.time() - start_time
    
    async def delete(self, server_name: str, operation: str, **kwargs) -> bool:
        """删除缓存值"""
        if not self.config.enabled or not self._initialized:
            return False
        
        cache_key = self._make_cache_key(server_name, operation, **kwargs)
        
        try:
            success = True
            
            # 删除内存缓存
            if not await self.memory_cache.delete(cache_key):
                success = False
            
            # 删除Redis缓存
            if self.redis_cache:
                if not await self.redis_cache.delete(cache_key):
                    success = False
            
            if success:
                self.metrics.deletes += 1
            else:
                self.metrics.errors += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"缓存删除失败: {e}")
            self.metrics.errors += 1
            return False
    
    async def clear_server_cache(self, server_name: str):
        """清空指定服务器的缓存"""
        try:
            # 清空Redis缓存
            if self.redis_cache:
                await self.redis_cache.clear(f"{server_name}:*")
            
            # 内存缓存需要逐个检查和删除
            # 这里简化处理，直接清空所有内存缓存
            await self.memory_cache.clear()
            
            self.logger.info(f"已清空服务器 {server_name} 的缓存")
            
        except Exception as e:
            self.logger.error(f"清空服务器缓存失败: {e}")
    
    async def clear_all(self):
        """清空所有缓存"""
        try:
            await self.memory_cache.clear()
            
            if self.redis_cache:
                await self.redis_cache.clear()
            
            self.logger.info("已清空所有缓存")
            
        except Exception as e:
            self.logger.error(f"清空所有缓存失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            "enabled": self.config.enabled,
            "initialized": self._initialized,
            "metrics": self.metrics.to_dict(),
            "memory_cache": self.memory_cache.get_stats()
        }
        
        if self.redis_cache:
            # Redis统计需要异步获取，这里返回占位符
            stats["redis_cache"] = {"enabled": True}
        else:
            stats["redis_cache"] = {"enabled": False}
        
        return stats
    
    async def get_detailed_stats(self) -> Dict[str, Any]:
        """获取详细统计信息（包括异步数据）"""
        stats = self.get_stats()
        
        if self.redis_cache:
            stats["redis_cache"] = await self.redis_cache.get_stats()
        
        return stats
    
    @asynccontextmanager
    async def cached_call(self, server_name: str, operation: str, ttl: Optional[float] = None, **kwargs):
        """缓存调用上下文管理器"""
        # 尝试从缓存获取
        cached_result = await self.get(server_name, operation, **kwargs)
        if cached_result is not None:
            yield cached_result
            return
        
        # 缓存未命中，执行实际调用
        result = yield None
        
        # 缓存结果
        if result is not None:
            await self.set(server_name, operation, result, ttl, **kwargs)


# 全局缓存管理器实例
_cache_manager: Optional[MCPCacheManager] = None


def get_cache_manager() -> MCPCacheManager:
    """获取全局缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = MCPCacheManager()
    return _cache_manager


async def initialize_cache_manager() -> bool:
    """初始化全局缓存管理器"""
    manager = get_cache_manager()
    return await manager.initialize()