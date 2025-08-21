"""数据库会话管理模块

提供会话生命周期管理、事务处理和依赖注入功能。
"""

import logging
from typing import Optional, AsyncGenerator, Generator, Callable, Any
from contextlib import contextmanager, asynccontextmanager
from functools import wraps

from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from .connection import get_database_manager, DatabaseManager

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or get_database_manager()
        self._session_count = 0
        self._async_session_count = 0
    
    @property
    def session_count(self) -> int:
        """当前活跃会话数量"""
        return self._session_count
    
    @property
    def async_session_count(self) -> int:
        """当前活跃异步会话数量"""
        return self._async_session_count
    
    def create_session(self) -> Session:
        """创建新的数据库会话"""
        session = self.db_manager.get_session()
        self._session_count += 1
        logger.debug(f"Created session, active sessions: {self._session_count}")
        return session
    
    def create_async_session(self) -> AsyncSession:
        """创建新的异步数据库会话"""
        session = self.db_manager.get_async_session()
        self._async_session_count += 1
        logger.debug(f"Created async session, active sessions: {self._async_session_count}")
        return session
    
    def close_session(self, session: Session):
        """关闭数据库会话"""
        try:
            session.close()
            self._session_count = max(0, self._session_count - 1)
            logger.debug(f"Closed session, active sessions: {self._session_count}")
        except Exception as e:
            logger.error(f"Error closing session: {e}")
    
    async def close_async_session(self, session: AsyncSession):
        """关闭异步数据库会话"""
        try:
            await session.close()
            self._async_session_count = max(0, self._async_session_count - 1)
            logger.debug(f"Closed async session, active sessions: {self._async_session_count}")
        except Exception as e:
            logger.error(f"Error closing async session: {e}")
    
    @contextmanager
    def session_scope(
        self, 
        autocommit: bool = True,
        rollback_on_error: bool = True
    ) -> Generator[Session, None, None]:
        """会话上下文管理器
        
        Args:
            autocommit: 是否自动提交事务
            rollback_on_error: 出错时是否自动回滚
        """
        session = self.create_session()
        try:
            yield session
            if autocommit:
                session.commit()
                logger.debug("Session committed successfully")
        except SQLAlchemyError as e:
            if rollback_on_error:
                session.rollback()
                logger.warning(f"Session rolled back due to SQLAlchemy error: {e}")
            raise
        except Exception as e:
            if rollback_on_error:
                session.rollback()
                logger.error(f"Session rolled back due to error: {e}")
            raise
        finally:
            self.close_session(session)
    
    @asynccontextmanager
    async def async_session_scope(
        self,
        autocommit: bool = True,
        rollback_on_error: bool = True
    ) -> AsyncGenerator[AsyncSession, None]:
        """异步会话上下文管理器
        
        Args:
            autocommit: 是否自动提交事务
            rollback_on_error: 出错时是否自动回滚
        """
        session = self.create_async_session()
        try:
            yield session
            if autocommit:
                await session.commit()
                logger.debug("Async session committed successfully")
        except SQLAlchemyError as e:
            if rollback_on_error:
                await session.rollback()
                logger.warning(f"Async session rolled back due to SQLAlchemy error: {e}")
            raise
        except Exception as e:
            if rollback_on_error:
                await session.rollback()
                logger.error(f"Async session rolled back due to error: {e}")
            raise
        finally:
            await self.close_async_session(session)
    
    @contextmanager
    def transaction_scope(self, session: Session) -> Generator[Session, None, None]:
        """事务上下文管理器"""
        trans = session.begin()
        try:
            yield session
            trans.commit()
            logger.debug("Transaction committed successfully")
        except Exception as e:
            trans.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
    
    @asynccontextmanager
    async def async_transaction_scope(
        self, 
        session: AsyncSession
    ) -> AsyncGenerator[AsyncSession, None]:
        """异步事务上下文管理器"""
        trans = await session.begin()
        try:
            yield session
            await trans.commit()
            logger.debug("Async transaction committed successfully")
        except Exception as e:
            await trans.rollback()
            logger.error(f"Async transaction rolled back due to error: {e}")
            raise
    
    def with_session(self, func: Callable) -> Callable:
        """会话装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.session_scope() as session:
                return func(session, *args, **kwargs)
        return wrapper
    
    def with_async_session(self, func: Callable) -> Callable:
        """异步会话装饰器"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self.async_session_scope() as session:
                return await func(session, *args, **kwargs)
        return wrapper
    
    def with_transaction(self, func: Callable) -> Callable:
        """事务装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.session_scope() as session:
                with self.transaction_scope(session):
                    return func(session, *args, **kwargs)
        return wrapper
    
    def with_async_transaction(self, func: Callable) -> Callable:
        """异步事务装饰器"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self.async_session_scope() as session:
                async with self.async_transaction_scope(session):
                    return await func(session, *args, **kwargs)
        return wrapper
    
    def bulk_insert(self, session: Session, objects: list, batch_size: int = 1000):
        """批量插入对象"""
        try:
            for i in range(0, len(objects), batch_size):
                batch = objects[i:i + batch_size]
                session.bulk_save_objects(batch)
                session.flush()
                logger.debug(f"Bulk inserted batch {i//batch_size + 1}, size: {len(batch)}")
            session.commit()
            logger.info(f"Bulk inserted {len(objects)} objects successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Bulk insert failed: {e}")
            raise
    
    async def async_bulk_insert(
        self, 
        session: AsyncSession, 
        objects: list, 
        batch_size: int = 1000
    ):
        """异步批量插入对象"""
        try:
            for i in range(0, len(objects), batch_size):
                batch = objects[i:i + batch_size]
                session.add_all(batch)
                await session.flush()
                logger.debug(f"Async bulk inserted batch {i//batch_size + 1}, size: {len(batch)}")
            await session.commit()
            logger.info(f"Async bulk inserted {len(objects)} objects successfully")
        except Exception as e:
            await session.rollback()
            logger.error(f"Async bulk insert failed: {e}")
            raise
    
    def get_stats(self) -> dict:
        """获取会话统计信息"""
        return {
            "active_sessions": self._session_count,
            "active_async_sessions": self._async_session_count,
            "total_active": self._session_count + self._async_session_count
        }


# 全局会话管理器实例
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """获取全局会话管理器实例"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def set_session_manager(manager: SessionManager):
    """设置全局会话管理器实例"""
    global _session_manager
    _session_manager = manager


# 便捷函数和装饰器
@contextmanager
def session_scope(
    autocommit: bool = True,
    rollback_on_error: bool = True
) -> Generator[Session, None, None]:
    """全局会话上下文管理器"""
    with get_session_manager().session_scope(
        autocommit=autocommit,
        rollback_on_error=rollback_on_error
    ) as session:
        yield session


@asynccontextmanager
async def async_session_scope(
    autocommit: bool = True,
    rollback_on_error: bool = True
) -> AsyncGenerator[AsyncSession, None]:
    """全局异步会话上下文管理器"""
    async with get_session_manager().async_session_scope(
        autocommit=autocommit,
        rollback_on_error=rollback_on_error
    ) as session:
        yield session


def with_session(func: Callable) -> Callable:
    """全局会话装饰器"""
    return get_session_manager().with_session(func)


def with_async_session(func: Callable) -> Callable:
    """全局异步会话装饰器"""
    return get_session_manager().with_async_session(func)


def with_transaction(func: Callable) -> Callable:
    """全局事务装饰器"""
    return get_session_manager().with_transaction(func)


def with_async_transaction(func: Callable) -> Callable:
    """全局异步事务装饰器"""
    return get_session_manager().with_async_transaction(func)


# 依赖注入函数（用于FastAPI等框架）
def get_db_session() -> Generator[Session, None, None]:
    """获取数据库会话（用于依赖注入）"""
    with session_scope() as session:
        yield session


async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """获取异步数据库会话（用于依赖注入）"""
    async with async_session_scope() as session:
        yield session