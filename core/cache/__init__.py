"""
缓存管理模块

本模块提供缓存管理功能，包括：
- Redis缓存管理
- 会话缓存
- 统一缓存接口
"""

from .redis_manager import (
    RedisManager,
    SessionCache,
    CacheManager,
    get_cache_manager,
    get_redis_manager,
    get_session_cache
)

__all__ = [
    "RedisManager",
    "SessionCache", 
    "CacheManager",
    "get_cache_manager",
    "get_redis_manager",
    "get_session_cache"
]