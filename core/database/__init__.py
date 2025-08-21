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
            
            # 尝试初始化Redis连接池（可选）
            try:
                await self._initialize_redis()
            except Exception as e:
                logger.warning(f"Redis连接初始化失败，将跳过Redis功能: {e}")
                self.redis_pool = None
            
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
            
            # 优化的连接池配置
            pool_config = {
                'dsn': postgres_url,
                'min_size': 2,  # 减少最小连接数
                'max_size': 10,  # 减少最大连接数
                'command_timeout': 30,  # 减少命令超时时间
                'statement_cache_size': 0,  # 禁用prepared statements以避免冲突
                'server_settings': {
                    'application_name': 'langgraph_backend',
                    'jit': 'off',  # 关闭JIT以提高连接速度
                    'statement_timeout': '30s',  # SQL语句超时
                    'idle_in_transaction_session_timeout': '60s',  # 事务空闲超时
                },
                # 连接初始化回调
                'init': self._init_postgres_connection
            }
            
            # 创建连接池
            logger.info("创建PostgreSQL连接池...")
            self.postgres_pool = await asyncpg.create_pool(**pool_config)
            
            # 清理可能存在的prepared statements
            await self._cleanup_prepared_statements()
            
            # 测试连接
            async with self.postgres_pool.acquire() as conn:
                result = await conn.fetchval("SELECT version()")
                logger.info(f"PostgreSQL连接成功: {result}")
                
        except Exception as e:
            logger.error(f"PostgreSQL连接初始化失败: {e}")
            raise
    
    async def _init_postgres_connection(self, conn: asyncpg.Connection):
        """PostgreSQL连接初始化回调"""
        try:
            # 设置连接级别的配置
            await conn.execute("SET statement_timeout = '30s'")
            await conn.execute("SET idle_in_transaction_session_timeout = '60s'")
            logger.debug(f"PostgreSQL连接 {id(conn)} 初始化完成")
        except Exception as e:
            logger.warning(f"PostgreSQL连接初始化回调失败: {e}")
    
    async def _cleanup_prepared_statements(self):
        """清理残留的prepared statements"""
        try:
            async with self.postgres_pool.acquire() as conn:
                # 查询所有prepared statements
                statements = await conn.fetch(
                    "SELECT name FROM pg_prepared_statements WHERE name LIKE '_pg%'"
                )
                
                if statements:
                    logger.info(f"清理 {len(statements)} 个残留的prepared statements")
                    
                    # 清理所有以_pg开头的prepared statements
                    for stmt in statements:
                        try:
                            await conn.execute(f"DEALLOCATE {stmt['name']}")
                            logger.debug(f"已清理prepared statement: {stmt['name']}")
                        except Exception as e:
                            logger.debug(f"清理prepared statement {stmt['name']} 失败: {e}")
                    
                    logger.info("Prepared statements清理完成")
                
        except Exception as e:
            logger.warning(f"清理prepared statements失败: {e}")
    
    async def _initialize_redis(self) -> None:
        """初始化Redis连接池"""
        try:
            redis_url = self.settings.redis.redis_url
            
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
                echo=self.settings.debug,
                pool_size=5,  # 减少连接池大小
                max_overflow=10,  # 减少最大溢出连接数
                pool_pre_ping=True,
                pool_recycle=1800,  # 减少连接回收时间
                connect_args={
                    "statement_cache_size": 0,  # 禁用预编译语句以兼容pgbouncer
                    "prepared_statement_cache_size": 0,
                    "server_settings": {
                        "application_name": "langgraph_sqlalchemy",
                        "jit": "off"
                    }
                }
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
            
            # 使用原生asyncpg连接来避免prepared statement冲突
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    # 创建扩展（如果需要）
                    try:
                        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                        logger.info("vector扩展检查完成")
                    except Exception as e:
                        logger.warning(f"vector扩展创建跳过: {e}")
                    
                    try:
                        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                        logger.info("pg_trgm扩展检查完成")
                    except Exception as e:
                        logger.warning(f"pg_trgm扩展创建跳过: {e}")
                        
            logger.info("数据库表结构检查完成")
            
        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            # 不要抛出异常，让系统继续运行
            logger.warning("数据库表创建失败，但系统将继续运行")
    
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
        logger.info("开始清理数据库连接...")
        
        # PostgreSQL连接池清理
        if self.postgres_pool:
            try:
                logger.info("关闭PostgreSQL连接池...")
                # 使用超时控制避免长时间等待
                await asyncio.wait_for(self.postgres_pool.close(), timeout=30.0)
                logger.info("PostgreSQL连接池已正常关闭")
            except asyncio.TimeoutError:
                logger.warning("PostgreSQL连接池关闭超时，强制终止连接")
                try:
                    # 强制终止连接池
                    if hasattr(self.postgres_pool, 'terminate'):
                        self.postgres_pool.terminate()
                    logger.info("PostgreSQL连接池已强制关闭")
                except Exception as e:
                    logger.error(f"强制关闭PostgreSQL连接池失败: {e}")
            except Exception as e:
                logger.error(f"PostgreSQL连接池关闭失败: {e}")
        
        # Redis连接池清理
        if self.redis_pool:
            try:
                logger.info("关闭Redis连接池...")
                await asyncio.wait_for(self.redis_pool.disconnect(), timeout=10.0)
                logger.info("Redis连接池已关闭")
            except asyncio.TimeoutError:
                logger.warning("Redis连接池关闭超时")
            except Exception as e:
                logger.error(f"Redis连接池关闭失败: {e}")
        
        # SQLAlchemy引擎清理
        if self.async_engine:
            try:
                logger.info("关闭SQLAlchemy引擎...")
                await asyncio.wait_for(self.async_engine.dispose(), timeout=15.0)
                logger.info("SQLAlchemy引擎已关闭")
            except asyncio.TimeoutError:
                logger.warning("SQLAlchemy引擎关闭超时")
            except Exception as e:
                logger.error(f"SQLAlchemy引擎关闭失败: {e}")
        
        self.is_initialized = False
        logger.info("数据库连接清理完成")


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