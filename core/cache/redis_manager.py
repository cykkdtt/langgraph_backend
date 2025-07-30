"""
Redis缓存管理器

本模块提供Redis缓存功能，包括：
- Redis连接管理
- 缓存操作封装
- 会话存储
- 性能优化
"""

import redis.asyncio as redis
import json
import logging
from typing import Any, Optional, Dict, List, Union
from datetime import timedelta
from config.settings import Settings

logger = logging.getLogger(__name__)

class RedisManager:
    """Redis缓存管理器"""
    
    def __init__(self, settings: Optional[Settings] = None):
        """初始化Redis管理器"""
        self.settings = settings or Settings()
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """连接Redis服务器"""
        try:
            redis_url = self.settings.database.redis_url
            logger.info(f"正在连接Redis: {redis_url.replace(':' + self.settings.database.redis_password or '', ':***')}")
            
            self.redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # 测试连接
            await self.redis_client.ping()
            self._connected = True
            logger.info("✅ Redis连接成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ Redis连接失败: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """断开Redis连接"""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False
            logger.info("Redis连接已断开")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            if not self._connected or not self.redis_client:
                return {"status": "disconnected", "error": "Redis未连接"}
            
            # 测试ping
            pong = await self.redis_client.ping()
            
            # 获取服务器信息
            info = await self.redis_client.info()
            
            return {
                "status": "healthy",
                "ping": pong,
                "version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime": info.get("uptime_in_seconds")
            }
        except Exception as e:
            logger.error(f"Redis健康检查失败: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            if not self._connected or not self.redis_client:
                logger.warning("Redis未连接，跳过缓存设置")
                return False
            
            # 序列化值
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif not isinstance(value, str):
                value = str(value)
            
            # 设置值
            if expire:
                await self.redis_client.setex(key, expire, value)
            else:
                await self.redis_client.set(key, value)
            
            return True
        except Exception as e:
            logger.error(f"Redis设置失败 {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            if not self._connected or not self.redis_client:
                return None
            
            value = await self.redis_client.get(key)
            if value is None:
                return None
            
            # 尝试反序列化JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Redis获取失败 {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            if not self._connected or not self.redis_client:
                return False
            
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis删除失败 {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            if not self._connected or not self.redis_client:
                return False
            
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis检查存在失败 {key}: {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """设置键过期时间"""
        try:
            if not self._connected or not self.redis_client:
                return False
            
            return await self.redis_client.expire(key, seconds)
        except Exception as e:
            logger.error(f"Redis设置过期失败 {key}: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配的键列表"""
        try:
            if not self._connected or not self.redis_client:
                return []
            
            return await self.redis_client.keys(pattern)
        except Exception as e:
            logger.error(f"Redis获取键列表失败 {pattern}: {e}")
            return []
    
    async def flushdb(self) -> bool:
        """清空当前数据库"""
        try:
            if not self._connected or not self.redis_client:
                return False
            
            await self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis清空数据库失败: {e}")
            return False

class SessionCache:
    """会话缓存管理器"""
    
    def __init__(self, redis_manager: RedisManager, prefix: str = "session:"):
        self.redis_manager = redis_manager
        self.prefix = prefix
        self.default_expire = 3600  # 1小时
    
    async def set_session(self, session_id: str, data: Dict[str, Any], expire: Optional[int] = None) -> bool:
        """设置会话数据"""
        key = f"{self.prefix}{session_id}"
        expire = expire or self.default_expire
        return await self.redis_manager.set(key, data, expire)
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话数据"""
        key = f"{self.prefix}{session_id}"
        return await self.redis_manager.get(key)
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        key = f"{self.prefix}{session_id}"
        return await self.redis_manager.delete(key)
    
    async def extend_session(self, session_id: str, expire: Optional[int] = None) -> bool:
        """延长会话过期时间"""
        key = f"{self.prefix}{session_id}"
        expire = expire or self.default_expire
        return await self.redis_manager.expire(key, expire)

class CacheManager:
    """统一缓存管理器"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.redis_manager = RedisManager(self.settings)
        self.session_cache = SessionCache(self.redis_manager)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """初始化缓存管理器"""
        if self._initialized:
            return True
        
        success = await self.redis_manager.connect()
        if success:
            self._initialized = True
            logger.info("缓存管理器初始化成功")
        else:
            logger.warning("缓存管理器初始化失败，将使用内存缓存")
        
        return success
    
    async def cleanup(self):
        """清理资源"""
        await self.redis_manager.disconnect()
        self._initialized = False
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return await self.redis_manager.health_check()

# 全局缓存管理器实例
_cache_manager: Optional[CacheManager] = None

async def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()
    return _cache_manager

async def get_redis_manager() -> RedisManager:
    """获取Redis管理器实例"""
    cache_manager = await get_cache_manager()
    return cache_manager.redis_manager

async def get_session_cache() -> SessionCache:
    """获取会话缓存实例"""
    cache_manager = await get_cache_manager()
    return cache_manager.session_cache