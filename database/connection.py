"""数据库连接管理模块

提供数据库配置、连接池管理和表操作功能。
"""

import os
import logging
from typing import Optional, Dict, Any, AsyncGenerator, Generator
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass

from sqlalchemy import (
    create_engine, 
    Engine, 
    MetaData,
    event,
    pool
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.orm import (
    sessionmaker,
    Session,
    declarative_base
)
from sqlalchemy.pool import QueuePool, NullPool

logger = logging.getLogger(__name__)

# 创建基础模型类
Base = declarative_base()

@dataclass
class DatabaseConfig:
    """数据库配置类"""
    
    # 基础连接配置
    url: str
    async_url: Optional[str] = None
    
    # 连接池配置
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    
    # 会话配置
    autocommit: bool = False
    autoflush: bool = True
    expire_on_commit: bool = True
    
    # 引擎配置
    echo: bool = False
    echo_pool: bool = False
    future: bool = True
    
    # 其他配置
    connect_args: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """从环境变量创建配置"""
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:password@localhost:5432/langgraph_study"
        )
        
        # 生成异步URL
        async_url = None
        if database_url.startswith("postgresql://"):
            async_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        elif database_url.startswith("sqlite://"):
            async_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
        
        return cls(
            url=database_url,
            async_url=async_url,
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
            echo_pool=os.getenv("DB_ECHO_POOL", "false").lower() == "true"
        )
    
    def get_engine_kwargs(self) -> Dict[str, Any]:
        """获取引擎创建参数"""
        kwargs = {
            "echo": self.echo,
            "echo_pool": self.echo_pool,
            "future": self.future,
            "pool_pre_ping": self.pool_pre_ping,
            "pool_recycle": self.pool_recycle
        }
        
        # 添加连接池配置
        if not self.url.startswith("sqlite://"):
            kwargs.update({
                "poolclass": QueuePool,
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "pool_timeout": self.pool_timeout
            })
        else:
            # SQLite 使用 NullPool
            kwargs["poolclass"] = NullPool
        
        # 添加连接参数
        if self.connect_args:
            kwargs["connect_args"] = self.connect_args
        
        return kwargs
    
    def get_session_kwargs(self) -> Dict[str, Any]:
        """获取会话创建参数"""
        return {
            "autocommit": self.autocommit,
            "autoflush": self.autoflush,
            "expire_on_commit": self.expire_on_commit
        }


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig.from_env()
        self._engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._async_session_factory: Optional[async_sessionmaker] = None
        self._metadata: Optional[MetaData] = None
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置数据库日志"""
        if self.config.echo:
            logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
        if self.config.echo_pool:
            logging.getLogger("sqlalchemy.pool").setLevel(logging.DEBUG)
    
    @property
    def engine(self) -> Engine:
        """获取同步引擎"""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine
    
    @property
    def async_engine(self) -> AsyncEngine:
        """获取异步引擎"""
        if self._async_engine is None:
            self._async_engine = self._create_async_engine()
        return self._async_engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """获取同步会话工厂"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                **self.config.get_session_kwargs()
            )
        return self._session_factory
    
    @property
    def async_session_factory(self) -> async_sessionmaker:
        """获取异步会话工厂"""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                **self.config.get_session_kwargs()
            )
        return self._async_session_factory
    
    @property
    def metadata(self) -> MetaData:
        """获取元数据"""
        if self._metadata is None:
            self._metadata = Base.metadata
        return self._metadata
    
    def _create_engine(self) -> Engine:
        """创建同步引擎"""
        engine = create_engine(
            self.config.url,
            **self.config.get_engine_kwargs()
        )
        
        # 添加事件监听器
        self._setup_engine_events(engine)
        
        logger.info(f"Created database engine: {self.config.url}")
        return engine
    
    def _create_async_engine(self) -> AsyncEngine:
        """创建异步引擎"""
        if not self.config.async_url:
            raise ValueError("Async URL not configured")
        
        engine = create_async_engine(
            self.config.async_url,
            **self.config.get_engine_kwargs()
        )
        
        logger.info(f"Created async database engine: {self.config.async_url}")
        return engine
    
    def _setup_engine_events(self, engine: Engine):
        """设置引擎事件监听器"""
        
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """为SQLite设置PRAGMA"""
            if "sqlite" in str(engine.url):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
        
        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """连接检出事件"""
            logger.debug("Database connection checked out")
        
        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """连接检入事件"""
            logger.debug("Database connection checked in")
    
    def get_session(self) -> Session:
        """获取同步会话"""
        return self.session_factory()
    
    def get_async_session(self) -> AsyncSession:
        """获取异步会话"""
        return self.async_session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """同步会话上下文管理器"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def async_session_scope(self) -> AsyncGenerator[AsyncSession, None]:
        """异步会话上下文管理器"""
        session = self.get_async_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    def create_tables(self, checkfirst: bool = True):
        """创建所有表"""
        try:
            self.metadata.create_all(self.engine, checkfirst=checkfirst)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def async_create_tables(self, checkfirst: bool = True):
        """异步创建所有表"""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(
                    lambda sync_conn: self.metadata.create_all(
                        sync_conn, checkfirst=checkfirst
                    )
                )
            logger.info("Database tables created successfully (async)")
        except Exception as e:
            logger.error(f"Failed to create tables (async): {e}")
            raise
    
    def drop_tables(self, checkfirst: bool = True):
        """删除所有表"""
        try:
            self.metadata.drop_all(self.engine, checkfirst=checkfirst)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    async def async_drop_tables(self, checkfirst: bool = True):
        """异步删除所有表"""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(
                    lambda sync_conn: self.metadata.drop_all(
                        sync_conn, checkfirst=checkfirst
                    )
                )
            logger.info("Database tables dropped successfully (async)")
        except Exception as e:
            logger.error(f"Failed to drop tables (async): {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self._engine:
            self._engine.dispose()
            logger.info("Database engine disposed")
        
        if self._async_engine:
            # 异步引擎的关闭需要在异步上下文中进行
            logger.info("Async database engine marked for disposal")
    
    async def async_close(self):
        """异步关闭数据库连接"""
        if self._async_engine:
            await self._async_engine.dispose()
            logger.info("Async database engine disposed")
        
        if self._engine:
            self._engine.dispose()
            logger.info("Database engine disposed")
    
    def health_check(self) -> bool:
        """数据库健康检查"""
        try:
            with self.session_scope() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def async_health_check(self) -> bool:
        """异步数据库健康检查"""
        try:
            async with self.async_session_scope() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Async database health check failed: {e}")
            return False


# 全局数据库管理器实例
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """获取全局数据库管理器实例"""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager


def set_database_manager(manager: DatabaseManager):
    """设置全局数据库管理器实例"""
    global _database_manager
    _database_manager = manager


# 便捷函数
def get_session() -> Session:
    """获取数据库会话"""
    return get_database_manager().get_session()


def get_async_session() -> AsyncSession:
    """获取异步数据库会话"""
    return get_database_manager().get_async_session()


def create_tables(checkfirst: bool = True):
    """创建数据库表"""
    get_database_manager().create_tables(checkfirst=checkfirst)


async def async_create_tables(checkfirst: bool = True):
    """异步创建数据库表"""
    await get_database_manager().async_create_tables(checkfirst=checkfirst)


def drop_tables(checkfirst: bool = True):
    """删除数据库表"""
    get_database_manager().drop_tables(checkfirst=checkfirst)


async def async_drop_tables(checkfirst: bool = True):
    """异步删除数据库表"""
    await get_database_manager().async_drop_tables(checkfirst=checkfirst)