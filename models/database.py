"""数据库连接池和事务管理工具模块

本模块提供数据库连接池管理、事务控制、连接监控、健康检查等功能。
"""

from typing import Dict, Any, Optional, List, Callable, Union, AsyncGenerator, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import time
import threading
import asyncio
from abc import ABC, abstractmethod
from sqlalchemy import (
    create_engine, event, text, inspect, MetaData
)
from sqlalchemy.orm import (
    sessionmaker, Session, scoped_session
)
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.pool import (
    QueuePool, NullPool, StaticPool, AssertionPool
)
from sqlalchemy.exc import (
    SQLAlchemyError, DisconnectionError, TimeoutError as SQLTimeoutError
)
from sqlalchemy.sql import text as sql_text
import logging


class ConnectionStatus(Enum):
    """连接状态枚举"""
    ACTIVE = "active"        # 活跃
    IDLE = "idle"            # 空闲
    CLOSED = "closed"        # 已关闭
    ERROR = "error"          # 错误
    TIMEOUT = "timeout"      # 超时


class TransactionStatus(Enum):
    """事务状态枚举"""
    ACTIVE = "active"        # 活跃
    COMMITTED = "committed"  # 已提交
    ROLLED_BACK = "rolled_back"  # 已回滚
    FAILED = "failed"        # 失败


class PoolType(Enum):
    """连接池类型枚举"""
    QUEUE = "queue"          # 队列池
    NULL = "null"            # 空池
    STATIC = "static"        # 静态池
    ASSERTION = "assertion"  # 断言池


@dataclass
class ConnectionMetrics:
    """连接指标"""
    connection_id: str
    status: ConnectionStatus
    created_at: datetime
    last_used_at: datetime
    query_count: int = 0
    total_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    
    @property
    def avg_query_time(self) -> float:
        """平均查询时间"""
        return self.total_time / self.query_count if self.query_count > 0 else 0.0
    
    @property
    def idle_time(self) -> float:
        """空闲时间（秒）"""
        return (datetime.now() - self.last_used_at).total_seconds()


@dataclass
class PoolMetrics:
    """连接池指标"""
    pool_size: int = 0
    checked_out: int = 0
    overflow: int = 0
    checked_in: int = 0
    total_connections: int = 0
    failed_connections: int = 0
    avg_connection_time: float = 0.0
    peak_connections: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def utilization_rate(self) -> float:
        """连接池利用率"""
        return (self.checked_out / self.pool_size) if self.pool_size > 0 else 0.0


@dataclass
class TransactionMetrics:
    """事务指标"""
    transaction_id: str
    status: TransactionStatus
    started_at: datetime
    ended_at: Optional[datetime] = None
    query_count: int = 0
    affected_rows: int = 0
    isolation_level: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """事务持续时间（秒）"""
        end_time = self.ended_at or datetime.now()
        return (end_time - self.started_at).total_seconds()


class DatabaseConfig:
    """数据库配置"""
    
    def __init__(self,
                 url: str,
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: int = 30,
                 pool_recycle: int = 3600,
                 pool_pre_ping: bool = True,
                 pool_type: PoolType = PoolType.QUEUE,
                 echo: bool = False,
                 echo_pool: bool = False,
                 connect_args: Dict[str, Any] = None,
                 execution_options: Dict[str, Any] = None):
        self.url = url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.pool_type = pool_type
        self.echo = echo
        self.echo_pool = echo_pool
        self.connect_args = connect_args or {}
        self.execution_options = execution_options or {}
    
    def get_pool_class(self):
        """获取连接池类"""
        pool_classes = {
            PoolType.QUEUE: QueuePool,
            PoolType.NULL: NullPool,
            PoolType.STATIC: StaticPool,
            PoolType.ASSERTION: AssertionPool
        }
        return pool_classes.get(self.pool_type, QueuePool)
    
    def to_engine_kwargs(self) -> Dict[str, Any]:
        """转换为引擎参数"""
        kwargs = {
            'echo': self.echo,
            'echo_pool': self.echo_pool,
            'connect_args': self.connect_args,
            'execution_options': self.execution_options
        }
        
        if self.pool_type != PoolType.NULL:
            kwargs.update({
                'poolclass': self.get_pool_class(),
                'pool_size': self.pool_size,
                'max_overflow': self.max_overflow,
                'pool_timeout': self.pool_timeout,
                'pool_recycle': self.pool_recycle,
                'pool_pre_ping': self.pool_pre_ping
            })
        
        return kwargs


class ConnectionMonitor:
    """连接监控器"""
    
    def __init__(self):
        self.connections: Dict[str, ConnectionMetrics] = {}
        self.pool_metrics = PoolMetrics()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def register_connection(self, connection_id: str) -> None:
        """注册连接"""
        with self.lock:
            self.connections[connection_id] = ConnectionMetrics(
                connection_id=connection_id,
                status=ConnectionStatus.ACTIVE,
                created_at=datetime.now(),
                last_used_at=datetime.now()
            )
            self.pool_metrics.total_connections += 1
    
    def update_connection_status(self, connection_id: str, status: ConnectionStatus) -> None:
        """更新连接状态"""
        with self.lock:
            if connection_id in self.connections:
                self.connections[connection_id].status = status
                self.connections[connection_id].last_used_at = datetime.now()
    
    def record_query(self, connection_id: str, execution_time: float, error: str = None) -> None:
        """记录查询"""
        with self.lock:
            if connection_id in self.connections:
                metrics = self.connections[connection_id]
                metrics.query_count += 1
                metrics.total_time += execution_time
                metrics.last_used_at = datetime.now()
                
                if error:
                    metrics.error_count += 1
                    metrics.last_error = error
    
    def remove_connection(self, connection_id: str) -> None:
        """移除连接"""
        with self.lock:
            if connection_id in self.connections:
                del self.connections[connection_id]
    
    def get_connection_metrics(self, connection_id: str) -> Optional[ConnectionMetrics]:
        """获取连接指标"""
        with self.lock:
            return self.connections.get(connection_id)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self.lock:
            active_connections = sum(1 for c in self.connections.values() 
                                   if c.status == ConnectionStatus.ACTIVE)
            idle_connections = sum(1 for c in self.connections.values() 
                                 if c.status == ConnectionStatus.IDLE)
            error_connections = sum(1 for c in self.connections.values() 
                                  if c.status == ConnectionStatus.ERROR)
            
            total_queries = sum(c.query_count for c in self.connections.values())
            total_errors = sum(c.error_count for c in self.connections.values())
            avg_query_time = sum(c.avg_query_time for c in self.connections.values()) / len(self.connections) if self.connections else 0
            
            return {
                'pool_metrics': {
                    'total_connections': len(self.connections),
                    'active_connections': active_connections,
                    'idle_connections': idle_connections,
                    'error_connections': error_connections,
                    'utilization_rate': active_connections / len(self.connections) if self.connections else 0
                },
                'query_metrics': {
                    'total_queries': total_queries,
                    'total_errors': total_errors,
                    'error_rate': total_errors / total_queries if total_queries > 0 else 0,
                    'avg_query_time': avg_query_time
                },
                'connections': [c.__dict__ for c in self.connections.values()]
            }
    
    def cleanup_idle_connections(self, max_idle_time: int = 300) -> int:
        """清理空闲连接"""
        with self.lock:
            now = datetime.now()
            to_remove = []
            
            for conn_id, metrics in self.connections.items():
                if (metrics.status == ConnectionStatus.IDLE and 
                    (now - metrics.last_used_at).total_seconds() > max_idle_time):
                    to_remove.append(conn_id)
            
            for conn_id in to_remove:
                del self.connections[conn_id]
            
            return len(to_remove)


class TransactionManager:
    """事务管理器"""
    
    def __init__(self):
        self.transactions: Dict[str, TransactionMetrics] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def start_transaction(self, transaction_id: str, isolation_level: str = None) -> TransactionMetrics:
        """开始事务"""
        with self.lock:
            metrics = TransactionMetrics(
                transaction_id=transaction_id,
                status=TransactionStatus.ACTIVE,
                started_at=datetime.now(),
                isolation_level=isolation_level
            )
            self.transactions[transaction_id] = metrics
            return metrics
    
    def end_transaction(self, transaction_id: str, status: TransactionStatus) -> Optional[TransactionMetrics]:
        """结束事务"""
        with self.lock:
            if transaction_id in self.transactions:
                metrics = self.transactions[transaction_id]
                metrics.status = status
                metrics.ended_at = datetime.now()
                return metrics
            return None
    
    def record_query(self, transaction_id: str, affected_rows: int = 0) -> None:
        """记录查询"""
        with self.lock:
            if transaction_id in self.transactions:
                metrics = self.transactions[transaction_id]
                metrics.query_count += 1
                metrics.affected_rows += affected_rows
    
    def get_transaction_metrics(self, transaction_id: str) -> Optional[TransactionMetrics]:
        """获取事务指标"""
        with self.lock:
            return self.transactions.get(transaction_id)
    
    def get_active_transactions(self) -> List[TransactionMetrics]:
        """获取活跃事务"""
        with self.lock:
            return [t for t in self.transactions.values() 
                   if t.status == TransactionStatus.ACTIVE]
    
    def cleanup_completed_transactions(self, max_age_hours: int = 24) -> int:
        """清理已完成的事务"""
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(hours=max_age_hours)
            to_remove = []
            
            for trans_id, metrics in self.transactions.items():
                if (metrics.status in [TransactionStatus.COMMITTED, TransactionStatus.ROLLED_BACK] and
                    metrics.ended_at and metrics.ended_at < cutoff):
                    to_remove.append(trans_id)
            
            for trans_id in to_remove:
                del self.transactions[trans_id]
            
            return len(to_remove)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        self.scoped_session_factory: Optional[scoped_session] = None
        self.connection_monitor = ConnectionMonitor()
        self.transaction_manager = TransactionManager()
        self.logger = logging.getLogger(__name__)
        self._setup_engine()
        self._setup_event_listeners()
    
    def _setup_engine(self) -> None:
        """设置数据库引擎"""
        try:
            self.engine = create_engine(
                self.config.url,
                **self.config.to_engine_kwargs()
            )
            
            self.session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False
            )
            
            self.scoped_session_factory = scoped_session(self.session_factory)
            
            self.logger.info(f"Database engine created successfully: {self.config.url}")
        
        except Exception as e:
            self.logger.error(f"Failed to create database engine: {e}")
            raise
    
    def _setup_event_listeners(self) -> None:
        """设置事件监听器"""
        if not self.engine:
            return
        
        @event.listens_for(self.engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            connection_id = str(id(dbapi_connection))
            self.connection_monitor.register_connection(connection_id)
            self.logger.debug(f"Connection established: {connection_id}")
        
        @event.listens_for(self.engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            connection_id = str(id(dbapi_connection))
            self.connection_monitor.update_connection_status(
                connection_id, ConnectionStatus.ACTIVE
            )
        
        @event.listens_for(self.engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            connection_id = str(id(dbapi_connection))
            self.connection_monitor.update_connection_status(
                connection_id, ConnectionStatus.IDLE
            )
        
        @event.listens_for(self.engine, "close")
        def on_close(dbapi_connection, connection_record):
            connection_id = str(id(dbapi_connection))
            self.connection_monitor.update_connection_status(
                connection_id, ConnectionStatus.CLOSED
            )
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            if hasattr(context, '_query_start_time'):
                execution_time = time.time() - context._query_start_time
                connection_id = str(id(conn.connection))
                self.connection_monitor.record_query(connection_id, execution_time)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """获取数据库会话"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        """获取数据库连接"""
        connection = self.engine.connect()
        try:
            yield connection
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            raise
        finally:
            connection.close()
    
    @contextmanager
    def transaction(self, isolation_level: str = None) -> Generator[Session, None, None]:
        """事务上下文管理器"""
        session = self.session_factory()
        transaction_id = str(id(session))
        
        # 开始事务监控
        self.transaction_manager.start_transaction(transaction_id, isolation_level)
        
        try:
            if isolation_level:
                session.connection(execution_options={'isolation_level': isolation_level})
            
            yield session
            session.commit()
            
            # 记录事务成功
            self.transaction_manager.end_transaction(
                transaction_id, TransactionStatus.COMMITTED
            )
        
        except Exception as e:
            session.rollback()
            
            # 记录事务失败
            self.transaction_manager.end_transaction(
                transaction_id, TransactionStatus.ROLLED_BACK
            )
            
            self.logger.error(f"Transaction error: {e}")
            raise
        
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """执行查询"""
        with self.get_session() as session:
            return session.execute(text(query), params or {})
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            start_time = time.time()
            
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            response_time = time.time() - start_time
            
            # 获取连接池状态
            pool = self.engine.pool
            pool_status = {
                'size': pool.size(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'checked_in': pool.checkedin()
            }
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'pool_status': pool_status,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取数据库指标"""
        connection_metrics = self.connection_monitor.get_all_metrics()
        active_transactions = self.transaction_manager.get_active_transactions()
        
        return {
            'connection_metrics': connection_metrics,
            'transaction_metrics': {
                'active_transactions': len(active_transactions),
                'transactions': [t.__dict__ for t in active_transactions]
            },
            'health_status': self.health_check(),
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self) -> Dict[str, int]:
        """清理资源"""
        cleaned_connections = self.connection_monitor.cleanup_idle_connections()
        cleaned_transactions = self.transaction_manager.cleanup_completed_transactions()
        
        return {
            'cleaned_connections': cleaned_connections,
            'cleaned_transactions': cleaned_transactions
        }
    
    def close(self) -> None:
        """关闭数据库管理器"""
        if self.scoped_session_factory:
            self.scoped_session_factory.remove()
        
        if self.engine:
            self.engine.dispose()
        
        self.logger.info("Database manager closed")


def with_database(db_manager: DatabaseManager):
    """数据库装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with db_manager.get_session() as session:
                return func(session, *args, **kwargs)
        return wrapper
    return decorator


def with_transaction(db_manager: DatabaseManager, isolation_level: str = None):
    """事务装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with db_manager.transaction(isolation_level) as session:
                return func(session, *args, **kwargs)
        return wrapper
    return decorator


# 导出所有类和函数
__all__ = [
    "ConnectionStatus",
    "TransactionStatus",
    "PoolType",
    "ConnectionMetrics",
    "PoolMetrics",
    "TransactionMetrics",
    "DatabaseConfig",
    "ConnectionMonitor",
    "TransactionManager",
    "DatabaseManager",
    "with_database",
    "with_transaction"
]