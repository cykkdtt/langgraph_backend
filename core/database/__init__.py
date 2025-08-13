"""
多智能体LangGraph项目 - 数据库初始化

本模块提供数据库的初始化和管理功能，包括：
- PostgreSQL数据库初始化
- 表结构创建
- 数据库连接池管理
- 数据库健康检查
- 数据库迁移
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool
import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from alembic.config import Config
from alembic import command

from config.settings import get_settings
from core.logging import get_logger

logger = get_logger("database.manager")

# SQLAlchemy基类
Base = declarative_base()


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.postgres_pool: Optional[Pool] = None
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.async_engine = None
        self.async_session_factory = None
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """初始化数据库连接"""
        if self.is_initialized:
            logger.info("数据库管理器已经初始化")
            return True
        
        try:
            logger.info("开始初始化数据库连接...")
            
            # 初始化PostgreSQL连接池
            await self._initialize_postgres()
            
            # 初始化Redis连接池
            await self._initialize_redis()
            
            # 初始化SQLAlchemy异步引擎
            await self._initialize_sqlalchemy()
            
            # 创建数据库表结构
            await self._create_tables()
            
            self.is_initialized = True
            logger.info("数据库管理器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"数据库管理器初始化失败: {e}")
            return False
    
    async def _initialize_postgres(self) -> None:
        """初始化PostgreSQL连接池"""
        try:
            # 解析连接URL
            postgres_url = self.settings.database.postgres_url
            
            # 创建连接池
            self.postgres_pool = await asyncpg.create_pool(
                postgres_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'jit': 'off'  # 关闭JIT以提高连接速度
                }
            )
            
            # 测试连接
            async with self.postgres_pool.acquire() as conn:
                result = await conn.fetchval("SELECT version()")
                logger.info(f"PostgreSQL连接成功: {result}")
                
        except Exception as e:
            logger.error(f"PostgreSQL连接初始化失败: {e}")
            raise
    
    async def _initialize_redis(self) -> None:
        """初始化Redis连接池"""
        try:
            redis_url = self.settings.database.redis_url
            
            # 创建连接池
            self.redis_pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=20,
                retry_on_timeout=True,
                decode_responses=True
            )
            
            # 测试连接
            redis_client = redis.Redis(connection_pool=self.redis_pool)
            await redis_client.ping()
            logger.info("Redis连接成功")
            
        except Exception as e:
            logger.error(f"Redis连接初始化失败: {e}")
            raise
    
    async def _initialize_sqlalchemy(self) -> None:
        """初始化SQLAlchemy异步引擎"""
        try:
            # 创建异步引擎
            postgres_url = self.settings.database.postgres_url
            # 将postgresql://转换为postgresql+asyncpg://
            async_url = postgres_url.replace("postgresql://", "postgresql+asyncpg://")
            
            self.async_engine = create_async_engine(
                async_url,
                echo=self.settings.app.debug,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # 创建会话工厂
            self.async_session_factory = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("SQLAlchemy异步引擎初始化成功")
            
        except Exception as e:
            logger.error(f"SQLAlchemy引擎初始化失败: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """创建数据库表结构"""
        try:
            # 这里可以添加自定义表的创建逻辑
            # LangGraph的表会自动创建，这里主要是为了扩展表
            
            async with self.async_engine.begin() as conn:
                # 创建扩展（如果需要）
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                
                # 创建自定义表（如果有的话）
                # await conn.run_sync(Base.metadata.create_all)
                
            logger.info("数据库表结构检查完成")
            
        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """数据库健康检查"""
        health_status = {
            "postgres": {"status": "unknown", "details": {}},
            "redis": {"status": "unknown", "details": {}},
            "overall": "unknown"
        }
        
        # PostgreSQL健康检查
        try:
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    # 检查连接
                    version = await conn.fetchval("SELECT version()")
                    pool_size = self.postgres_pool.get_size()
                    
                    health_status["postgres"] = {
                        "status": "healthy",
                        "details": {
                            "version": version,
                            "pool_size": pool_size,
                            "pool_max_size": self.postgres_pool.get_max_size()
                        }
                    }
            else:
                health_status["postgres"]["status"] = "not_initialized"
                
        except Exception as e:
            health_status["postgres"] = {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
        
        # Redis健康检查
        try:
            if self.redis_pool:
                redis_client = redis.Redis(connection_pool=self.redis_pool)
                await redis_client.ping()
                info = await redis_client.info()
                
                health_status["redis"] = {
                    "status": "healthy",
                    "details": {
                        "version": info.get("redis_version"),
                        "connected_clients": info.get("connected_clients"),
                        "used_memory": info.get("used_memory_human")
                    }
                }
            else:
                health_status["redis"]["status"] = "not_initialized"
                
        except Exception as e:
            health_status["redis"] = {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
        
        # 整体状态
        postgres_ok = health_status["postgres"]["status"] == "healthy"
        redis_ok = health_status["redis"]["status"] == "healthy"
        
        if postgres_ok and redis_ok:
            health_status["overall"] = "healthy"
        elif postgres_ok or redis_ok:
            health_status["overall"] = "degraded"
        else:
            health_status["overall"] = "unhealthy"
        
        return health_status
    
    @asynccontextmanager
    async def get_postgres_connection(self):
        """获取PostgreSQL连接的上下文管理器"""
        if not self.postgres_pool:
            raise RuntimeError("PostgreSQL连接池未初始化")
        
        async with self.postgres_pool.acquire() as conn:
            yield conn
    
    @asynccontextmanager
    async def get_redis_connection(self):
        """获取Redis连接的上下文管理器"""
        if not self.redis_pool:
            raise RuntimeError("Redis连接池未初始化")
        
        redis_client = redis.Redis(connection_pool=self.redis_pool)
        try:
            yield redis_client
        finally:
            await redis_client.close()
    
    @asynccontextmanager
    async def get_async_session(self):
        """获取SQLAlchemy异步会话的上下文管理器"""
        if not self.async_session_factory:
            raise RuntimeError("SQLAlchemy会话工厂未初始化")
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def execute_sql(self, sql: str, params: Optional[Dict] = None) -> List[Dict]:
        """执行SQL查询"""
        async with self.get_postgres_connection() as conn:
            if params:
                result = await conn.fetch(sql, *params.values())
            else:
                result = await conn.fetch(sql)
            
            return [dict(row) for row in result]
    
    async def cleanup(self) -> None:
        """清理数据库连接"""
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
                logger.info("PostgreSQL连接池已关闭")
            
            if self.redis_pool:
                await self.redis_pool.disconnect()
                logger.info("Redis连接池已关闭")
            
            if self.async_engine:
                await self.async_engine.dispose()
                logger.info("SQLAlchemy引擎已关闭")
                
        except Exception as e:
            logger.error(f"数据库连接清理失败: {e}")
        
        self.is_initialized = False


# 全局数据库管理器实例
database_manager = DatabaseManager()


async def get_database_manager() -> DatabaseManager:
    """获取数据库管理器实例"""
    if not database_manager.is_initialized:
        await database_manager.initialize()
    return database_manager


async def initialize_database() -> bool:
    """初始化数据库的便捷函数"""
    return await database_manager.initialize()


@asynccontextmanager
async def get_postgres_connection():
    """获取PostgreSQL连接的便捷函数"""
    async with database_manager.get_postgres_connection() as conn:
        yield conn


@asynccontextmanager
async def get_redis_connection():
    """获取Redis连接的便捷函数"""
    async with database_manager.get_redis_connection() as conn:
        yield conn


@asynccontextmanager
async def get_async_session():
    """获取SQLAlchemy会话的便捷函数"""
    async with database_manager.get_async_session() as session:
        yield session


# 导出列表
__all__ = [
    "Base",
    "DatabaseManager",
    "database_manager",
    "get_database_manager",
    "initialize_database",
    "get_postgres_connection",
    "get_redis_connection",
    "get_async_session"
]