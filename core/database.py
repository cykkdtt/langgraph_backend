"""数据库配置和连接管理

本模块提供数据库连接、会话管理和基础操作功能。
"""

import os
import logging
from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine import Engine
from models.database_models import Base

# 配置日志
logger = logging.getLogger(__name__)

# 数据库配置
class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self):
        # 从环境变量获取数据库配置
        self.database_url = os.getenv(
            "DATABASE_URL",
            "sqlite:///./langgraph_system.db"
        )
        self.echo = os.getenv("DB_ECHO", "false").lower() == "true"
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        
    def get_engine_kwargs(self) -> dict:
        """获取数据库引擎参数"""
        kwargs = {
            "echo": self.echo,
            "future": True,
        }
        
        # SQLite 特殊配置
        if self.database_url.startswith("sqlite"):
            kwargs.update({
                "poolclass": StaticPool,
                "connect_args": {
                    "check_same_thread": False,
                    "timeout": 20,
                },
            })
        else:
            # PostgreSQL/MySQL 配置
            kwargs.update({
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "pool_timeout": self.pool_timeout,
                "pool_recycle": self.pool_recycle,
                "pool_pre_ping": True,
            })
        
        return kwargs


# 全局配置实例
config = DatabaseConfig()

# 创建数据库引擎
engine = create_engine(config.database_url, **config.get_engine_kwargs())

# 创建会话工厂
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)


# 数据库事件监听器
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """为 SQLite 设置 pragma"""
    if "sqlite" in str(dbapi_connection):
        cursor = dbapi_connection.cursor()
        # 启用外键约束
        cursor.execute("PRAGMA foreign_keys=ON")
        # 设置 WAL 模式以提高并发性能
        cursor.execute("PRAGMA journal_mode=WAL")
        # 设置同步模式
        cursor.execute("PRAGMA synchronous=NORMAL")
        # 设置缓存大小
        cursor.execute("PRAGMA cache_size=10000")
        # 设置临时存储
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()


@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """SQL 执行前的日志记录"""
    if config.echo:
        logger.debug(f"SQL: {statement}")
        if parameters:
            logger.debug(f"Parameters: {parameters}")


def create_tables():
    """创建所有数据库表"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表创建成功")
    except Exception as e:
        logger.error(f"创建数据库表失败: {e}")
        raise


def drop_tables():
    """删除所有数据库表"""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("数据库表删除成功")
    except Exception as e:
        logger.error(f"删除数据库表失败: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """获取数据库会话（依赖注入用）"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"数据库会话错误: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """获取数据库会话（上下文管理器）"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"数据库操作错误: {e}")
        db.rollback()
        raise
    finally:
        db.close()


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def init_db(self):
        """初始化数据库"""
        create_tables()
    
    def reset_db(self):
        """重置数据库"""
        drop_tables()
        create_tables()
    
    def get_session(self) -> Session:
        """获取新的数据库会话"""
        return self.SessionLocal()
    
    def execute_raw_sql(self, sql: str, params: Optional[dict] = None):
        """执行原生 SQL"""
        with get_db_session() as db:
            result = db.execute(sql, params or {})
            return result.fetchall()
    
    def check_connection(self) -> bool:
        """检查数据库连接"""
        try:
            with get_db_session() as db:
                db.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"数据库连接检查失败: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> dict:
        """获取表信息"""
        try:
            from sqlalchemy import inspect
            inspector = inspect(self.engine)
            
            # 获取表的列信息
            columns = inspector.get_columns(table_name)
            
            # 获取索引信息
            indexes = inspector.get_indexes(table_name)
            
            # 获取外键信息
            foreign_keys = inspector.get_foreign_keys(table_name)
            
            return {
                "columns": columns,
                "indexes": indexes,
                "foreign_keys": foreign_keys
            }
        except Exception as e:
            logger.error(f"获取表信息失败: {e}")
            return {}
    
    def get_database_stats(self) -> dict:
        """获取数据库统计信息"""
        try:
            with get_db_session() as db:
                # 获取所有表的行数
                table_stats = {}
                
                # 用户统计
                user_count = db.execute("SELECT COUNT(*) FROM users").scalar()
                table_stats["users"] = user_count
                
                # 会话统计
                session_count = db.execute("SELECT COUNT(*) FROM sessions").scalar()
                table_stats["sessions"] = session_count
                
                # 消息统计
                message_count = db.execute("SELECT COUNT(*) FROM messages").scalar()
                table_stats["messages"] = message_count
                
                # 工具调用统计
                tool_call_count = db.execute("SELECT COUNT(*) FROM tool_calls").scalar()
                table_stats["tool_calls"] = tool_call_count
                
                # 智能体状态统计
                agent_state_count = db.execute("SELECT COUNT(*) FROM agent_states").scalar()
                table_stats["agent_states"] = agent_state_count
                
                # 工作流统计
                workflow_count = db.execute("SELECT COUNT(*) FROM workflows").scalar()
                table_stats["workflows"] = workflow_count
                
                # 工作流执行统计
                execution_count = db.execute("SELECT COUNT(*) FROM workflow_executions").scalar()
                table_stats["workflow_executions"] = execution_count
                
                # 记忆统计
                memory_count = db.execute("SELECT COUNT(*) FROM memories").scalar()
                table_stats["memories"] = memory_count
                
                # 系统日志统计
                log_count = db.execute("SELECT COUNT(*) FROM system_logs").scalar()
                table_stats["system_logs"] = log_count
                
                return {
                    "table_stats": table_stats,
                    "total_records": sum(table_stats.values()),
                    "database_url": config.database_url,
                    "engine_info": str(self.engine.url)
                }
        except Exception as e:
            logger.error(f"获取数据库统计信息失败: {e}")
            return {}


# 全局数据库管理器实例
db_manager = DatabaseManager()


# 数据库初始化函数
def init_database():
    """初始化数据库"""
    logger.info("正在初始化数据库...")
    db_manager.init_db()
    logger.info("数据库初始化完成")


def reset_database():
    """重置数据库"""
    logger.warning("正在重置数据库...")
    db_manager.reset_db()
    logger.info("数据库重置完成")


# 导出主要组件
__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_session",
    "DatabaseManager",
    "db_manager",
    "create_tables",
    "drop_tables",
    "init_database",
    "reset_database",
    "DatabaseConfig",
    "config"
]