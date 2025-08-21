"""数据库迁移管理模块

提供数据库版本控制、迁移脚本管理和自动迁移功能。
"""

import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from alembic.runtime.environment import EnvironmentContext
from sqlalchemy import text, MetaData

from .connection import get_database_manager, DatabaseManager

logger = logging.getLogger(__name__)


class MigrationManager:
    """数据库迁移管理器"""
    
    def __init__(
        self, 
        db_manager: Optional[DatabaseManager] = None,
        migrations_dir: Optional[str] = None
    ):
        self.db_manager = db_manager or get_database_manager()
        self.migrations_dir = Path(migrations_dir or "migrations")
        self.alembic_cfg: Optional[Config] = None
        
        # 确保迁移目录存在
        self.migrations_dir.mkdir(exist_ok=True)
        
        # 初始化Alembic配置
        self._init_alembic_config()
    
    def _init_alembic_config(self):
        """初始化Alembic配置"""
        try:
            # 创建alembic.ini配置文件
            alembic_ini_path = self.migrations_dir / "alembic.ini"
            if not alembic_ini_path.exists():
                self._create_alembic_ini(alembic_ini_path)
            
            # 创建Alembic配置对象
            self.alembic_cfg = Config(str(alembic_ini_path))
            self.alembic_cfg.set_main_option("script_location", str(self.migrations_dir))
            self.alembic_cfg.set_main_option("sqlalchemy.url", self.db_manager.config.url)
            
            # 创建版本目录
            versions_dir = self.migrations_dir / "versions"
            versions_dir.mkdir(exist_ok=True)
            
            # 创建env.py文件
            env_py_path = self.migrations_dir / "env.py"
            if not env_py_path.exists():
                self._create_env_py(env_py_path)
            
            logger.info("Alembic configuration initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alembic configuration: {e}")
            raise
    
    def _create_alembic_ini(self, path: Path):
        """创建alembic.ini配置文件"""
        content = f"""# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = {self.migrations_dir}

# template used to generate migration file names; The default value is %%(rev)s_%%(slug)s
# Uncomment the line below if you want the files to be prepended with date and time
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s

# sys.path path, will be prepended to sys.path if present.
# defaults to the current working directory.
prepend_sys_path = .

# timezone to use when rendering the date within the migration file
# as well as the filename.
# If specified, requires the python-dateutil library that can be
# installed by adding `alembic[tz]` to the pip requirements
# string value is passed to dateutil.tz.gettz()
# leave blank for localtime
# timezone =

# max length of characters to apply to the
# "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version number format
# version_num_format = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s

# version path separator; As mentioned above, this is the character used to split
# version_locations. The default within new alembic.ini files is "os", which uses
# os.pathsep. If this key is omitted entirely, it falls back to the legacy
# behavior of splitting on spaces and/or commas.
# Valid values for version_path_separator are:
#
# version_path_separator = :
# version_path_separator = ;
# version_path_separator = space
version_path_separator = os

# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = {self.db_manager.config.url}


[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" entrypoint
# hooks = black
# black.type = console_scripts
# black.entrypoint = black
# black.options = -l 79 REVISION_SCRIPT_FILENAME

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %%(levelname)-5.5s [%%(name)s] %%(message)s
datefmt = %%H:%%M:%%S
"""
        path.write_text(content)
    
    def _create_env_py(self, path: Path):
        """创建env.py环境文件"""
        content = '''"""Alembic环境配置文件"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# 导入模型以便Alembic能够检测到它们
from langgraph_study.models.database import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''
        path.write_text(content)
    
    def create_migration(
        self, 
        message: str, 
        autogenerate: bool = True,
        sql: bool = False
    ) -> str:
        """创建新的迁移脚本
        
        Args:
            message: 迁移描述信息
            autogenerate: 是否自动生成迁移内容
            sql: 是否生成SQL脚本
        
        Returns:
            生成的迁移文件路径
        """
        try:
            if autogenerate:
                command.revision(
                    self.alembic_cfg,
                    message=message,
                    autogenerate=True,
                    sql=sql
                )
            else:
                command.revision(
                    self.alembic_cfg,
                    message=message,
                    sql=sql
                )
            
            logger.info(f"Created migration: {message}")
            return message
            
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            raise
    
    def run_migrations(self, revision: str = "head") -> None:
        """运行数据库迁移
        
        Args:
            revision: 目标版本，默认为最新版本
        """
        try:
            command.upgrade(self.alembic_cfg, revision)
            logger.info(f"Migrations upgraded to: {revision}")
        except Exception as e:
            logger.error(f"Failed to run migrations: {e}")
            raise
    
    def rollback_migration(self, revision: str) -> None:
        """回滚数据库迁移
        
        Args:
            revision: 目标版本
        """
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info(f"Migrations downgraded to: {revision}")
        except Exception as e:
            logger.error(f"Failed to rollback migration: {e}")
            raise
    
    def get_current_revision(self) -> Optional[str]:
        """获取当前数据库版本"""
        try:
            with self.db_manager.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return None
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """获取迁移历史"""
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            history = []
            
            for revision in script_dir.walk_revisions():
                history.append({
                    "revision": revision.revision,
                    "down_revision": revision.down_revision,
                    "branch_labels": revision.branch_labels,
                    "depends_on": revision.depends_on,
                    "doc": revision.doc,
                    "create_date": getattr(revision, 'create_date', None)
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get migration history: {e}")
            return []
    
    def get_pending_migrations(self) -> List[str]:
        """获取待执行的迁移"""
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            current_rev = self.get_current_revision()
            
            if current_rev is None:
                # 如果没有当前版本，返回所有迁移
                return [rev.revision for rev in script_dir.walk_revisions()]
            
            pending = []
            for revision in script_dir.walk_revisions("head", current_rev):
                if revision.revision != current_rev:
                    pending.append(revision.revision)
            
            return pending
            
        except Exception as e:
            logger.error(f"Failed to get pending migrations: {e}")
            return []
    
    def show_migration(self, revision: str) -> Optional[str]:
        """显示迁移内容"""
        try:
            command.show(self.alembic_cfg, revision)
            return revision
        except Exception as e:
            logger.error(f"Failed to show migration {revision}: {e}")
            return None
    
    def stamp_database(self, revision: str = "head") -> None:
        """标记数据库版本（不执行迁移）"""
        try:
            command.stamp(self.alembic_cfg, revision)
            logger.info(f"Database stamped with revision: {revision}")
        except Exception as e:
            logger.error(f"Failed to stamp database: {e}")
            raise
    
    def init_database(self) -> None:
        """初始化数据库（创建表并标记为最新版本）"""
        try:
            # 创建所有表
            self.db_manager.create_tables()
            
            # 标记为最新版本
            self.stamp_database("head")
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def reset_database(self) -> None:
        """重置数据库（删除所有表并重新创建）"""
        try:
            # 删除所有表
            self.db_manager.drop_tables()
            
            # 重新初始化
            self.init_database()
            
            logger.info("Database reset successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise
    
    def check_migration_status(self) -> Dict[str, Any]:
        """检查迁移状态"""
        try:
            current_rev = self.get_current_revision()
            pending_migrations = self.get_pending_migrations()
            migration_history = self.get_migration_history()
            
            return {
                "current_revision": current_rev,
                "pending_migrations": pending_migrations,
                "pending_count": len(pending_migrations),
                "total_migrations": len(migration_history),
                "is_up_to_date": len(pending_migrations) == 0
            }
            
        except Exception as e:
            logger.error(f"Failed to check migration status: {e}")
            return {
                "error": str(e),
                "current_revision": None,
                "pending_migrations": [],
                "pending_count": 0,
                "total_migrations": 0,
                "is_up_to_date": False
            }


# 全局迁移管理器实例
_migration_manager: Optional[MigrationManager] = None


def get_migration_manager() -> MigrationManager:
    """获取全局迁移管理器实例"""
    global _migration_manager
    if _migration_manager is None:
        _migration_manager = MigrationManager()
    return _migration_manager


def set_migration_manager(manager: MigrationManager):
    """设置全局迁移管理器实例"""
    global _migration_manager
    _migration_manager = manager


# 便捷函数
def create_migration(message: str, autogenerate: bool = True) -> str:
    """创建迁移脚本"""
    return get_migration_manager().create_migration(message, autogenerate)


def run_migrations(revision: str = "head") -> None:
    """运行迁移"""
    get_migration_manager().run_migrations(revision)


def rollback_migration(revision: str) -> None:
    """回滚迁移"""
    get_migration_manager().rollback_migration(revision)


def init_database() -> None:
    """初始化数据库"""
    get_migration_manager().init_database()


def reset_database() -> None:
    """重置数据库"""
    get_migration_manager().reset_database()


def check_migration_status() -> Dict[str, Any]:
    """检查迁移状态"""
    return get_migration_manager().check_migration_status()