"""模型迁移系统模块

本模块提供数据库迁移管理、版本控制和自动化迁移功能。
"""

from typing import (
    Dict, Any, Optional, List, Union, Callable, Type, 
    Tuple, Set, ClassVar, Protocol, TypeVar, Generic,
    NamedTuple, AsyncGenerator, Awaitable
)
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps, lru_cache
import logging
import os
import re
import hashlib
import json
from pathlib import Path
from collections import defaultdict, OrderedDict
import threading
import asyncio
from contextlib import contextmanager

# SQLAlchemy imports
try:
    from sqlalchemy import (
        create_engine, MetaData, Table, Column, Integer, String, 
        DateTime, Text, Boolean, text, inspect, event
    )
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.engine import Engine
    from sqlalchemy.schema import CreateTable, DropTable
    from sqlalchemy.sql import Select
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    create_engine = MetaData = Table = Column = None
    Session = sessionmaker = Engine = None

# 导入项目模块
from .events import EventType, emit_business_event
from .validation import validate_data


logger = logging.getLogger(__name__)


T = TypeVar('T')


class MigrationStatus(Enum):
    """迁移状态枚举"""
    PENDING = "pending"                    # 待执行
    RUNNING = "running"                    # 执行中
    COMPLETED = "completed"                # 已完成
    FAILED = "failed"                      # 失败
    ROLLED_BACK = "rolled_back"            # 已回滚
    SKIPPED = "skipped"                    # 已跳过


class MigrationType(Enum):
    """迁移类型枚举"""
    SCHEMA = "schema"                      # 模式迁移
    DATA = "data"                          # 数据迁移
    INDEX = "index"                        # 索引迁移
    CONSTRAINT = "constraint"              # 约束迁移
    FUNCTION = "function"                  # 函数迁移
    TRIGGER = "trigger"                    # 触发器迁移
    VIEW = "view"                          # 视图迁移
    CUSTOM = "custom"                      # 自定义迁移


class MigrationDirection(Enum):
    """迁移方向枚举"""
    UP = "up"                              # 向上迁移
    DOWN = "down"                          # 向下迁移


class DatabaseDialect(Enum):
    """数据库方言枚举"""
    POSTGRESQL = "postgresql"              # PostgreSQL
    MYSQL = "mysql"                        # MySQL
    SQLITE = "sqlite"                      # SQLite
    ORACLE = "oracle"                      # Oracle
    MSSQL = "mssql"                        # SQL Server


@dataclass
class MigrationFile:
    """迁移文件"""
    version: str                           # 版本号
    name: str                              # 迁移名称
    file_path: Path                        # 文件路径
    
    # 内容
    up_sql: str                            # 向上迁移SQL
    down_sql: str                          # 向下迁移SQL
    
    # 元数据
    description: str = ""                  # 描述
    migration_type: MigrationType = MigrationType.SCHEMA
    dependencies: List[str] = field(default_factory=list)  # 依赖的迁移
    tags: List[str] = field(default_factory=list)          # 标签
    
    # 时间信息
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # 校验信息
    checksum: Optional[str] = None         # 文件校验和
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self.calculate_checksum()
    
    def calculate_checksum(self) -> str:
        """计算文件校验和"""
        content = f"{self.up_sql}\n{self.down_sql}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def validate_checksum(self) -> bool:
        """验证校验和"""
        return self.checksum == self.calculate_checksum()
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'MigrationFile':
        """从文件创建迁移"""
        # 解析文件名获取版本和名称
        filename = file_path.stem
        parts = filename.split('_', 1)
        
        if len(parts) != 2:
            raise ValueError(f"Invalid migration filename format: {filename}")
        
        version, name = parts
        
        # 读取文件内容
        content = file_path.read_text(encoding='utf-8')
        
        # 解析SQL内容
        up_sql, down_sql = cls._parse_sql_content(content)
        
        return cls(
            version=version,
            name=name,
            file_path=file_path,
            up_sql=up_sql,
            down_sql=down_sql
        )
    
    @staticmethod
    def _parse_sql_content(content: str) -> Tuple[str, str]:
        """解析SQL内容"""
        # 查找 -- +migrate Up 和 -- +migrate Down 标记
        up_pattern = r'--\s*\+migrate\s+Up\s*\n(.*?)(?=--\s*\+migrate\s+Down|$)'
        down_pattern = r'--\s*\+migrate\s+Down\s*\n(.*?)$'
        
        up_match = re.search(up_pattern, content, re.DOTALL | re.IGNORECASE)
        down_match = re.search(down_pattern, content, re.DOTALL | re.IGNORECASE)
        
        up_sql = up_match.group(1).strip() if up_match else ""
        down_sql = down_match.group(1).strip() if down_match else ""
        
        return up_sql, down_sql
    
    def to_file(self, directory: Path) -> None:
        """保存到文件"""
        filename = f"{self.version}_{self.name}.sql"
        file_path = directory / filename
        
        content = f"""-- Migration: {self.name}
-- Version: {self.version}
-- Description: {self.description}
-- Type: {self.migration_type.value}
-- Created: {self.created_at.isoformat()}

-- +migrate Up
{self.up_sql}

-- +migrate Down
{self.down_sql}
"""
        
        file_path.write_text(content, encoding='utf-8')
        self.file_path = file_path


@dataclass
class MigrationRecord:
    """迁移记录"""
    version: str                           # 版本号
    name: str                              # 迁移名称
    status: MigrationStatus                # 状态
    
    # 执行信息
    applied_at: Optional[datetime] = None  # 应用时间
    rolled_back_at: Optional[datetime] = None  # 回滚时间
    execution_time_ms: Optional[float] = None  # 执行时间
    
    # 错误信息
    error_message: Optional[str] = None    # 错误消息
    error_details: Optional[str] = None    # 错误详情
    
    # 校验信息
    checksum: Optional[str] = None         # 校验和
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_applied(self) -> bool:
        """是否已应用"""
        return self.status == MigrationStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """是否失败"""
        return self.status == MigrationStatus.FAILED
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version': self.version,
            'name': self.name,
            'status': self.status.value,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'rolled_back_at': self.rolled_back_at.isoformat() if self.rolled_back_at else None,
            'execution_time_ms': self.execution_time_ms,
            'error_message': self.error_message,
            'error_details': self.error_details,
            'checksum': self.checksum,
            'metadata': self.metadata
        }


@dataclass
class MigrationConfig:
    """迁移配置"""
    # 基本配置
    migrations_directory: Path             # 迁移文件目录
    table_name: str = "schema_migrations"  # 迁移记录表名
    
    # 数据库配置
    database_url: Optional[str] = None     # 数据库连接URL
    dialect: DatabaseDialect = DatabaseDialect.POSTGRESQL
    
    # 执行配置
    auto_create_table: bool = True         # 自动创建迁移表
    validate_checksums: bool = True        # 验证校验和
    allow_missing_migrations: bool = False # 允许缺失迁移
    
    # 安全配置
    require_confirmation: bool = False     # 需要确认
    dry_run: bool = False                  # 干运行
    
    # 性能配置
    transaction_per_migration: bool = True # 每个迁移一个事务
    timeout_seconds: int = 300             # 超时时间
    
    # 备份配置
    create_backup: bool = False            # 创建备份
    backup_directory: Optional[Path] = None # 备份目录


class MigrationError(Exception):
    """迁移错误"""
    pass


class MigrationValidationError(MigrationError):
    """迁移验证错误"""
    pass


class MigrationExecutionError(MigrationError):
    """迁移执行错误"""
    pass


class MigrationFileManager:
    """迁移文件管理器"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self._ensure_migrations_directory()
    
    def _ensure_migrations_directory(self) -> None:
        """确保迁移目录存在"""
        self.config.migrations_directory.mkdir(parents=True, exist_ok=True)
    
    def scan_migrations(self) -> List[MigrationFile]:
        """扫描迁移文件"""
        migrations = []
        
        for file_path in self.config.migrations_directory.glob('*.sql'):
            try:
                migration = MigrationFile.from_file(file_path)
                migrations.append(migration)
            except Exception as e:
                logger.warning(f"Failed to parse migration file {file_path}: {e}")
        
        # 按版本排序
        migrations.sort(key=lambda m: m.version)
        return migrations
    
    def get_migration(self, version: str) -> Optional[MigrationFile]:
        """获取指定版本的迁移"""
        migrations = self.scan_migrations()
        for migration in migrations:
            if migration.version == version:
                return migration
        return None
    
    def create_migration(self, name: str, migration_type: MigrationType = MigrationType.SCHEMA,
                        description: str = "", up_sql: str = "", down_sql: str = "") -> MigrationFile:
        """创建新迁移"""
        # 生成版本号（时间戳格式）
        version = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        
        # 确保版本号唯一
        existing_versions = {m.version for m in self.scan_migrations()}
        counter = 1
        original_version = version
        while version in existing_versions:
            version = f"{original_version}_{counter:02d}"
            counter += 1
        
        # 创建迁移文件
        migration = MigrationFile(
            version=version,
            name=name,
            file_path=Path(),  # 临时路径
            up_sql=up_sql,
            down_sql=down_sql,
            description=description,
            migration_type=migration_type
        )
        
        # 保存到文件
        migration.to_file(self.config.migrations_directory)
        
        logger.info(f"Created migration: {version}_{name}")
        return migration
    
    def validate_migrations(self, migrations: List[MigrationFile]) -> List[str]:
        """验证迁移文件"""
        errors = []
        
        # 检查版本号重复
        versions = [m.version for m in migrations]
        duplicates = [v for v in set(versions) if versions.count(v) > 1]
        if duplicates:
            errors.append(f"Duplicate migration versions: {duplicates}")
        
        # 检查文件校验和
        if self.config.validate_checksums:
            for migration in migrations:
                if not migration.validate_checksum():
                    errors.append(f"Checksum mismatch for migration {migration.version}")
        
        # 检查依赖关系
        for migration in migrations:
            for dep in migration.dependencies:
                if not any(m.version == dep for m in migrations):
                    errors.append(f"Missing dependency {dep} for migration {migration.version}")
        
        return errors


class MigrationRecordManager:
    """迁移记录管理器"""
    
    def __init__(self, config: MigrationConfig, engine: Engine):
        self.config = config
        self.engine = engine
        self._metadata = MetaData()
        self._migration_table = self._create_migration_table()
        
        if config.auto_create_table:
            self._ensure_migration_table()
    
    def _create_migration_table(self) -> Table:
        """创建迁移记录表"""
        return Table(
            self.config.table_name,
            self._metadata,
            Column('version', String(50), primary_key=True),
            Column('name', String(255), nullable=False),
            Column('status', String(20), nullable=False),
            Column('applied_at', DateTime),
            Column('rolled_back_at', DateTime),
            Column('execution_time_ms', Integer),
            Column('error_message', Text),
            Column('error_details', Text),
            Column('checksum', String(32)),
            Column('metadata', Text)  # JSON格式的元数据
        )
    
    def _ensure_migration_table(self) -> None:
        """确保迁移表存在"""
        try:
            self._metadata.create_all(self.engine, tables=[self._migration_table])
        except Exception as e:
            logger.error(f"Failed to create migration table: {e}")
            raise MigrationError(f"Failed to create migration table: {e}")
    
    def get_applied_migrations(self) -> List[MigrationRecord]:
        """获取已应用的迁移"""
        with Session(self.engine) as session:
            try:
                result = session.execute(
                    self._migration_table.select().order_by(self._migration_table.c.version)
                )
                
                records = []
                for row in result:
                    metadata = json.loads(row.metadata) if row.metadata else {}
                    
                    record = MigrationRecord(
                        version=row.version,
                        name=row.name,
                        status=MigrationStatus(row.status),
                        applied_at=row.applied_at,
                        rolled_back_at=row.rolled_back_at,
                        execution_time_ms=row.execution_time_ms,
                        error_message=row.error_message,
                        error_details=row.error_details,
                        checksum=row.checksum,
                        metadata=metadata
                    )
                    records.append(record)
                
                return records
                
            except Exception as e:
                logger.error(f"Failed to get applied migrations: {e}")
                return []
    
    def record_migration(self, record: MigrationRecord) -> None:
        """记录迁移"""
        with Session(self.engine) as session:
            try:
                # 检查记录是否已存在
                existing = session.execute(
                    self._migration_table.select().where(
                        self._migration_table.c.version == record.version
                    )
                ).first()
                
                metadata_json = json.dumps(record.metadata) if record.metadata else None
                
                if existing:
                    # 更新现有记录
                    session.execute(
                        self._migration_table.update().where(
                            self._migration_table.c.version == record.version
                        ).values(
                            name=record.name,
                            status=record.status.value,
                            applied_at=record.applied_at,
                            rolled_back_at=record.rolled_back_at,
                            execution_time_ms=record.execution_time_ms,
                            error_message=record.error_message,
                            error_details=record.error_details,
                            checksum=record.checksum,
                            metadata=metadata_json
                        )
                    )
                else:
                    # 插入新记录
                    session.execute(
                        self._migration_table.insert().values(
                            version=record.version,
                            name=record.name,
                            status=record.status.value,
                            applied_at=record.applied_at,
                            rolled_back_at=record.rolled_back_at,
                            execution_time_ms=record.execution_time_ms,
                            error_message=record.error_message,
                            error_details=record.error_details,
                            checksum=record.checksum,
                            metadata=metadata_json
                        )
                    )
                
                session.commit()
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to record migration: {e}")
                raise MigrationError(f"Failed to record migration: {e}")
    
    def remove_migration_record(self, version: str) -> None:
        """移除迁移记录"""
        with Session(self.engine) as session:
            try:
                session.execute(
                    self._migration_table.delete().where(
                        self._migration_table.c.version == version
                    )
                )
                session.commit()
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to remove migration record: {e}")
                raise MigrationError(f"Failed to remove migration record: {e}")


class MigrationExecutor:
    """迁移执行器"""
    
    def __init__(self, config: MigrationConfig, engine: Engine):
        self.config = config
        self.engine = engine
        self.record_manager = MigrationRecordManager(config, engine)
    
    def execute_migration(self, migration: MigrationFile, direction: MigrationDirection) -> MigrationRecord:
        """执行单个迁移"""
        start_time = datetime.utcnow()
        
        # 创建迁移记录
        record = MigrationRecord(
            version=migration.version,
            name=migration.name,
            status=MigrationStatus.RUNNING,
            checksum=migration.checksum
        )
        
        try:
            # 记录开始状态
            self.record_manager.record_migration(record)
            
            # 获取要执行的SQL
            sql = migration.up_sql if direction == MigrationDirection.UP else migration.down_sql
            
            if not sql.strip():
                logger.warning(f"No SQL to execute for migration {migration.version} ({direction.value})")
                record.status = MigrationStatus.SKIPPED
                self.record_manager.record_migration(record)
                return record
            
            # 执行SQL
            execution_start = datetime.utcnow()
            
            if self.config.dry_run:
                logger.info(f"DRY RUN: Would execute migration {migration.version} ({direction.value})")
                logger.info(f"SQL: {sql}")
            else:
                self._execute_sql(sql)
            
            execution_end = datetime.utcnow()
            execution_time_ms = (execution_end - execution_start).total_seconds() * 1000
            
            # 更新记录
            record.status = MigrationStatus.COMPLETED
            record.execution_time_ms = execution_time_ms
            
            if direction == MigrationDirection.UP:
                record.applied_at = execution_end
            else:
                record.rolled_back_at = execution_end
            
            self.record_manager.record_migration(record)
            
            # 发布事件
            emit_business_event(
                EventType.MIGRATION_EXECUTED,
                f"migration_{direction.value}",
                data={
                    'version': migration.version,
                    'name': migration.name,
                    'direction': direction.value,
                    'execution_time_ms': execution_time_ms
                }
            )
            
            logger.info(f"Successfully executed migration {migration.version} ({direction.value}) "
                       f"in {execution_time_ms:.2f}ms")
            
            return record
            
        except Exception as e:
            # 记录错误
            record.status = MigrationStatus.FAILED
            record.error_message = str(e)
            record.error_details = self._get_error_details(e)
            
            execution_end = datetime.utcnow()
            record.execution_time_ms = (execution_end - execution_start).total_seconds() * 1000
            
            self.record_manager.record_migration(record)
            
            logger.error(f"Failed to execute migration {migration.version} ({direction.value}): {e}")
            raise MigrationExecutionError(f"Migration {migration.version} failed: {e}")
    
    def _execute_sql(self, sql: str) -> None:
        """执行SQL"""
        with Session(self.engine) as session:
            try:
                if self.config.transaction_per_migration:
                    # 在事务中执行
                    session.execute(text(sql))
                    session.commit()
                else:
                    # 不使用事务（自动提交）
                    session.execute(text(sql))
                    
            except Exception as e:
                if self.config.transaction_per_migration:
                    session.rollback()
                raise
    
    def _get_error_details(self, error: Exception) -> str:
        """获取错误详情"""
        import traceback
        return traceback.format_exc()


class MigrationPlanner:
    """迁移规划器"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
    
    def plan_migrations(self, available_migrations: List[MigrationFile],
                       applied_migrations: List[MigrationRecord],
                       target_version: Optional[str] = None) -> List[Tuple[MigrationFile, MigrationDirection]]:
        """规划迁移执行计划"""
        applied_versions = {r.version for r in applied_migrations if r.is_applied}
        
        if target_version is None:
            # 迁移到最新版本
            return self._plan_up_migrations(available_migrations, applied_versions)
        else:
            # 迁移到指定版本
            return self._plan_to_version(available_migrations, applied_versions, target_version)
    
    def _plan_up_migrations(self, available_migrations: List[MigrationFile],
                           applied_versions: Set[str]) -> List[Tuple[MigrationFile, MigrationDirection]]:
        """规划向上迁移"""
        plan = []
        
        for migration in available_migrations:
            if migration.version not in applied_versions:
                plan.append((migration, MigrationDirection.UP))
        
        return plan
    
    def _plan_to_version(self, available_migrations: List[MigrationFile],
                        applied_versions: Set[str], target_version: str) -> List[Tuple[MigrationFile, MigrationDirection]]:
        """规划到指定版本的迁移"""
        plan = []
        
        # 找到目标版本的索引
        target_index = None
        for i, migration in enumerate(available_migrations):
            if migration.version == target_version:
                target_index = i
                break
        
        if target_index is None:
            raise MigrationError(f"Target version {target_version} not found")
        
        # 确定当前版本的索引
        current_index = -1
        for i, migration in enumerate(available_migrations):
            if migration.version in applied_versions:
                current_index = i
        
        if target_index > current_index:
            # 向上迁移
            for i in range(current_index + 1, target_index + 1):
                migration = available_migrations[i]
                if migration.version not in applied_versions:
                    plan.append((migration, MigrationDirection.UP))
        else:
            # 向下迁移
            for i in range(current_index, target_index, -1):
                migration = available_migrations[i]
                if migration.version in applied_versions:
                    plan.append((migration, MigrationDirection.DOWN))
        
        return plan
    
    def validate_plan(self, plan: List[Tuple[MigrationFile, MigrationDirection]]) -> List[str]:
        """验证迁移计划"""
        errors = []
        
        # 检查依赖关系
        for migration, direction in plan:
            if direction == MigrationDirection.UP:
                for dep in migration.dependencies:
                    # 检查依赖是否在计划中或已应用
                    dep_in_plan = any(m.version == dep and d == MigrationDirection.UP 
                                    for m, d in plan)
                    if not dep_in_plan:
                        errors.append(f"Dependency {dep} not satisfied for migration {migration.version}")
        
        return errors


class MigrationManager:
    """迁移管理器"""
    
    def __init__(self, config: MigrationConfig, engine: Engine = None):
        self.config = config
        self.engine = engine or self._create_engine()
        
        self.file_manager = MigrationFileManager(config)
        self.record_manager = MigrationRecordManager(config, self.engine)
        self.executor = MigrationExecutor(config, self.engine)
        self.planner = MigrationPlanner(config)
    
    def _create_engine(self) -> Engine:
        """创建数据库引擎"""
        if not self.config.database_url:
            raise MigrationError("Database URL is required")
        
        return create_engine(self.config.database_url)
    
    def scan_migrations(self) -> List[MigrationFile]:
        """扫描迁移文件"""
        return self.file_manager.scan_migrations()
    
    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        available_migrations = self.scan_migrations()
        applied_migrations = self.record_manager.get_applied_migrations()
        
        applied_versions = {r.version for r in applied_migrations if r.is_applied}
        pending_migrations = [m for m in available_migrations if m.version not in applied_versions]
        
        return {
            'total_migrations': len(available_migrations),
            'applied_migrations': len(applied_versions),
            'pending_migrations': len(pending_migrations),
            'failed_migrations': len([r for r in applied_migrations if r.is_failed]),
            'latest_version': available_migrations[-1].version if available_migrations else None,
            'current_version': max(applied_versions) if applied_versions else None,
            'pending_versions': [m.version for m in pending_migrations]
        }
    
    def create_migration(self, name: str, migration_type: MigrationType = MigrationType.SCHEMA,
                        description: str = "", up_sql: str = "", down_sql: str = "") -> MigrationFile:
        """创建新迁移"""
        return self.file_manager.create_migration(name, migration_type, description, up_sql, down_sql)
    
    def migrate(self, target_version: Optional[str] = None, dry_run: bool = None) -> List[MigrationRecord]:
        """执行迁移"""
        if dry_run is not None:
            self.config.dry_run = dry_run
        
        # 扫描可用迁移
        available_migrations = self.scan_migrations()
        applied_migrations = self.record_manager.get_applied_migrations()
        
        # 验证迁移文件
        validation_errors = self.file_manager.validate_migrations(available_migrations)
        if validation_errors:
            raise MigrationValidationError(f"Migration validation failed: {validation_errors}")
        
        # 规划迁移
        plan = self.planner.plan_migrations(available_migrations, applied_migrations, target_version)
        
        if not plan:
            logger.info("No migrations to execute")
            return []
        
        # 验证计划
        plan_errors = self.planner.validate_plan(plan)
        if plan_errors:
            raise MigrationValidationError(f"Migration plan validation failed: {plan_errors}")
        
        # 执行迁移
        results = []
        
        logger.info(f"Executing {len(plan)} migrations")
        
        for migration, direction in plan:
            try:
                result = self.executor.execute_migration(migration, direction)
                results.append(result)
                
                if result.is_failed:
                    logger.error(f"Migration {migration.version} failed, stopping execution")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to execute migration {migration.version}: {e}")
                break
        
        return results
    
    def rollback(self, target_version: Optional[str] = None, steps: int = 1) -> List[MigrationRecord]:
        """回滚迁移"""
        applied_migrations = self.record_manager.get_applied_migrations()
        applied_versions = [r.version for r in applied_migrations if r.is_applied]
        
        if not applied_versions:
            logger.info("No migrations to rollback")
            return []
        
        if target_version:
            # 回滚到指定版本
            available_migrations = self.scan_migrations()
            plan = self.planner.plan_migrations(available_migrations, applied_migrations, target_version)
        else:
            # 回滚指定步数
            available_migrations = self.scan_migrations()
            migrations_to_rollback = applied_versions[-steps:]
            
            plan = []
            for version in reversed(migrations_to_rollback):
                migration = next((m for m in available_migrations if m.version == version), None)
                if migration:
                    plan.append((migration, MigrationDirection.DOWN))
        
        # 执行回滚
        results = []
        
        for migration, direction in plan:
            try:
                result = self.executor.execute_migration(migration, direction)
                results.append(result)
                
                if result.is_failed:
                    logger.error(f"Rollback of migration {migration.version} failed, stopping execution")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to rollback migration {migration.version}: {e}")
                break
        
        return results
    
    def reset(self) -> None:
        """重置所有迁移"""
        applied_migrations = self.record_manager.get_applied_migrations()
        applied_versions = [r.version for r in applied_migrations if r.is_applied]
        
        if applied_versions:
            self.rollback(target_version=None)
    
    def force_version(self, version: str) -> None:
        """强制设置版本（不执行SQL）"""
        available_migrations = self.scan_migrations()
        migration = next((m for m in available_migrations if m.version == version), None)
        
        if not migration:
            raise MigrationError(f"Migration version {version} not found")
        
        record = MigrationRecord(
            version=version,
            name=migration.name,
            status=MigrationStatus.COMPLETED,
            applied_at=datetime.utcnow(),
            checksum=migration.checksum
        )
        
        self.record_manager.record_migration(record)
        logger.info(f"Forced migration version to {version}")


# 迁移装饰器
def migration(name: str, migration_type: MigrationType = MigrationType.SCHEMA,
             description: str = ""):
    """迁移装饰器"""
    def decorator(func: Callable) -> Callable:
        # 这里可以添加迁移注册逻辑
        func._migration_name = name
        func._migration_type = migration_type
        func._migration_description = description
        return func
    
    return decorator


# 全局迁移管理器
_default_migration_manager: Optional[MigrationManager] = None


def initialize_migrations(config: MigrationConfig, engine: Engine = None) -> MigrationManager:
    """初始化迁移管理器"""
    global _default_migration_manager
    _default_migration_manager = MigrationManager(config, engine)
    return _default_migration_manager


def get_default_migration_manager() -> Optional[MigrationManager]:
    """获取默认迁移管理器"""
    return _default_migration_manager


# 便捷函数
def migrate(target_version: Optional[str] = None, dry_run: bool = False) -> List[MigrationRecord]:
    """执行迁移"""
    manager = get_default_migration_manager()
    if manager:
        return manager.migrate(target_version, dry_run)
    return []


def rollback(target_version: Optional[str] = None, steps: int = 1) -> List[MigrationRecord]:
    """回滚迁移"""
    manager = get_default_migration_manager()
    if manager:
        return manager.rollback(target_version, steps)
    return []


def create_migration(name: str, migration_type: MigrationType = MigrationType.SCHEMA,
                    description: str = "", up_sql: str = "", down_sql: str = "") -> Optional[MigrationFile]:
    """创建迁移"""
    manager = get_default_migration_manager()
    if manager:
        return manager.create_migration(name, migration_type, description, up_sql, down_sql)
    return None


def get_migration_status() -> Dict[str, Any]:
    """获取迁移状态"""
    manager = get_default_migration_manager()
    if manager:
        return manager.get_migration_status()
    return {}


# 导出所有类和函数
__all__ = [
    "MigrationStatus",
    "MigrationType",
    "MigrationDirection",
    "DatabaseDialect",
    "MigrationFile",
    "MigrationRecord",
    "MigrationConfig",
    "MigrationError",
    "MigrationValidationError",
    "MigrationExecutionError",
    "MigrationFileManager",
    "MigrationRecordManager",
    "MigrationExecutor",
    "MigrationPlanner",
    "MigrationManager",
    "migration",
    "initialize_migrations",
    "get_default_migration_manager",
    "migrate",
    "rollback",
    "create_migration",
    "get_migration_status"
]