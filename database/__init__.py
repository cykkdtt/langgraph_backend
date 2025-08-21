"""数据库模块

提供数据库连接、会话管理和配置功能。
"""

from .connection import (
    DatabaseConfig,
    DatabaseManager,
    get_database_manager,
    get_session,
    get_async_session,
    create_tables,
    drop_tables
)

from .session import (
    SessionManager,
    get_session_manager,
    session_scope,
    async_session_scope
)

from .migrations import (
    MigrationManager,
    get_migration_manager,
    run_migrations,
    create_migration,
    rollback_migration
)

__all__ = [
    # 数据库连接
    "DatabaseConfig",
    "DatabaseManager",
    "get_database_manager",
    "get_session",
    "get_async_session",
    "create_tables",
    "drop_tables",
    
    # 会话管理
    "SessionManager",
    "get_session_manager",
    "session_scope",
    "async_session_scope",
    
    # 迁移管理
    "MigrationManager",
    "get_migration_manager",
    "run_migrations",
    "create_migration",
    "rollback_migration"
]