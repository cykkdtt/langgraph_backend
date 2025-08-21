"""模型迁移系统模块

本模块提供数据库迁移、版本控制和模式管理功能。
"""

from typing import (
    Dict, Any, Optional, List, Union, Callable, Type, 
    Tuple, Set, ClassVar, Protocol, TypeVar, Generic,
    NamedTuple, AsyncGenerator, Awaitable, Iterator
)
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps, cached_property
import logging
import os
import re
import hashlib
import json
from pathlib import Path
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import threading

# SQLAlchemy imports
try:
    from sqlalchemy import (
        text, inspect, MetaData, Table, Column,
        Integer, String, DateTime, Boolean, Float, Text,
        create_engine, Engine, event
    )
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.sql import select, func
    from sqlalchemy.engine import Connection
    from sqlalchemy.schema import CreateTable, DropTable
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    text = None
    inspect = None
    MetaData = None
    Table = None
    Column = None
    create_engine = None
    Engine = None
    event = None
    Session = None
    sessionmaker = None
    select = None
    func = None
    Connection = None
    CreateTable = None
    DropTable = None
    SQLALCHEMY_AVAILABLE = False

# 导入项目模块
from .events import EventType, emit_business_event
from .validation import validate_data


logger = logging.getLogger(__name__)


T = TypeVar('T')


class MigrationType(Enum):
    """迁移类型枚举"""
    CREATE_TABLE = "create_table"          # 创建表
    DROP_TABLE = "drop_table"              # 删除表
    ALTER_TABLE = "alter_table"            # 修改表
    ADD_COLUMN = "add_column"              # 添加列
    DROP_COLUMN = "drop_column"            # 删除列
    MODIFY_COLUMN = "modify_column"        # 修改列
    ADD_INDEX = "add_index"                # 添加索引
    DROP_INDEX = "drop_index"              # 删除索引
    ADD_CONSTRAINT = "add_constraint"      # 添加约束
    DROP_CONSTRAINT = "drop_constraint"    # 删除约束
    INSERT_DATA = "insert_data"            # 插入数据
    UPDATE_DATA = "update_data"            # 更新数据
    DELETE_DATA = "delete_data"            # 删除数据
    CUSTOM = "custom"                      # 自定义


class MigrationStatus(Enum):
    """迁移状态枚举"""
    PENDING = "pending"                    # 待执行
    RUNNING = "running"                    # 执行中
    COMPLETED = "completed"                # 已完成
    FAILED = "failed"                      # 失败
    ROLLED_BACK = "rolled_back"            # 已回滚
    SKIPPED = "skipped"                    # 已跳过


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
class MigrationOperation:
    """迁移操作"""
    operation_type: MigrationType          # 操作类型
    sql: str                               # SQL语句
    rollback_sql: Optional[str] = None     # 回滚SQL
    description: Optional[str] = None      # 描述
    
    # 条件执行
    condition: Optional[str] = None        # 执行条件
    dialect_specific: Optional[DatabaseDialect] = None  # 特定数据库方言
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'operation_type': self.operation_type.value,
            'sql': self.sql,
            'rollback_sql': self.rollback_sql,
            'description': self.description,
            'condition': self.condition,
            'dialect_specific': self.dialect_specific.value if self.dialect_specific else None,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MigrationOperation':
        """从字典创建"""
        return cls(
            operation_type=MigrationType(data['operation_type']),
            sql=data['sql'],
            rollback_sql=data.get('rollback_sql'),
            description=data.get('description'),
            condition=data.get('condition'),
            dialect_specific=DatabaseDialect(data['dialect_specific']) if data.get('dialect_specific') else None,
            created_at=datetime.fromisoformat(data['created_at'])
        )


@dataclass
class Migration:
    """迁移"""
    version: str                           # 版本号
    name: str                              # 名称
    description: str                       # 描述
    operations: List[MigrationOperation] = field(default_factory=list)  # 操作列表
    
    # 依赖关系
    dependencies: List[str] = field(default_factory=list)  # 依赖的迁移版本
    
    # 状态信息
    status: MigrationStatus = MigrationStatus.PENDING
    applied_at: Optional[datetime] = None  # 应用时间
    rolled_back_at: Optional[datetime] = None  # 回滚时间
    
    # 执行信息
    execution_time: Optional[float] = None # 执行时间（秒）
    error_message: Optional[str] = None    # 错误信息
    
    # 元数据
    author: Optional[str] = None           # 作者
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def checksum(self) -> str:
        """计算校验和"""
        content = f"{self.version}:{self.name}:{self.description}"
        for op in self.operations:
            content += f":{op.sql}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @property
    def is_applied(self) -> bool:
        """是否已应用"""
        return self.status == MigrationStatus.COMPLETED
    
    @property
    def can_rollback(self) -> bool:
        """是否可以回滚"""
        return (
            self.status == MigrationStatus.COMPLETED and
            all(op.rollback_sql for op in self.operations)
        )
    
    def add_operation(self, operation: MigrationOperation) -> None:
        """添加操作"""
        self.operations.append(operation)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version': self.version,
            'name': self.name,
            'description': self.description,
            'operations': [op.to_dict() for op in self.operations],
            'dependencies': self.dependencies,
            'status': self.status.value,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'rolled_back_at': self.rolled_back_at.isoformat() if self.rolled_back_at else None,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Migration':
        """从字典创建"""
        migration = cls(
            version=data['version'],
            name=data['name'],
            description=data['description'],
            operations=[MigrationOperation.from_dict(op) for op in data['operations']],
            dependencies=data.get('dependencies', []),
            status=MigrationStatus(data['status']),
            applied_at=datetime.fromisoformat(data['applied_at']) if data.get('applied_at') else None,
            rolled_back_at=datetime.fromisoformat(data['rolled_back_at']) if data.get('rolled_back_at') else None,
            execution_time=data.get('execution_time'),
            error_message=data.get('error_message'),
            author=data.get('author'),
            created_at=datetime.fromisoformat(data['created_at'])
        )
        return migration


@dataclass
class MigrationConfig:
    """迁移配置"""
    # 路径配置
    migrations_dir: str = "migrations"     # 迁移文件目录
    schema_file: str = "schema.sql"        # 模式文件
    
    # 表配置
    migration_table: str = "schema_migrations"  # 迁移记录表
    
    # 执行配置
    auto_create_migration_table: bool = True  # 自动创建迁移表
    transaction_per_migration: bool = True    # 每个迁移使用事务
    validate_checksums: bool = True           # 验证校验和
    
    # 备份配置
    backup_before_migration: bool = False     # 迁移前备份
    backup_dir: str = "backups"               # 备份目录
    
    # 安全配置
    allow_destructive_operations: bool = False  # 允许破坏性操作
    require_confirmation: bool = True           # 需要确认
    
    # 并发配置
    max_concurrent_migrations: int = 1        # 最大并发迁移数
    
    # 日志配置
    log_migrations: bool = True               # 记录迁移日志
    log_level: str = "INFO"                   # 日志级别


class MigrationError(Exception):
    """迁移错误"""
    pass


class MigrationValidationError(MigrationError):
    """迁移验证错误"""
    pass


class MigrationExecutionError(MigrationError):
    """迁移执行错误"""
    pass


class MigrationDependencyError(MigrationError):
    """迁移依赖错误"""
    pass


class MigrationFileParser:
    """迁移文件解析器"""
    
    def __init__(self):
        self._version_pattern = re.compile(r'^(\d{14})_(.+)\.sql$')
        self._operation_pattern = re.compile(r'^--\s*(\w+):\s*(.*)$', re.MULTILINE)
    
    def parse_migration_file(self, file_path: Path) -> Migration:
        """解析迁移文件"""
        if not file_path.exists():
            raise MigrationError(f"Migration file not found: {file_path}")
        
        # 解析文件名
        match = self._version_pattern.match(file_path.name)
        if not match:
            raise MigrationError(f"Invalid migration file name: {file_path.name}")
        
        version = match.group(1)
        name = match.group(2).replace('_', ' ').title()
        
        # 读取文件内容
        content = file_path.read_text(encoding='utf-8')
        
        # 解析内容
        migration = Migration(
            version=version,
            name=name,
            description=f"Migration {name}"
        )
        
        # 分割UP和DOWN部分
        parts = content.split('-- DOWN')
        up_content = parts[0].replace('-- UP', '').strip()
        down_content = parts[1].strip() if len(parts) > 1 else None
        
        # 解析UP操作
        if up_content:
            up_operations = self._parse_sql_operations(up_content)
            for op_sql in up_operations:
                operation = MigrationOperation(
                    operation_type=self._detect_operation_type(op_sql),
                    sql=op_sql
                )
                migration.add_operation(operation)
        
        # 解析DOWN操作（回滚）
        if down_content:
            down_operations = self._parse_sql_operations(down_content)
            for i, op_sql in enumerate(down_operations):
                if i < len(migration.operations):
                    migration.operations[i].rollback_sql = op_sql
        
        return migration
    
    def _parse_sql_operations(self, content: str) -> List[str]:
        """解析SQL操作"""
        # 按分号分割SQL语句
        statements = [stmt.strip() for stmt in content.split(';') if stmt.strip()]
        return statements
    
    def _detect_operation_type(self, sql: str) -> MigrationType:
        """检测操作类型"""
        sql_upper = sql.upper().strip()
        
        if sql_upper.startswith('CREATE TABLE'):
            return MigrationType.CREATE_TABLE
        elif sql_upper.startswith('DROP TABLE'):
            return MigrationType.DROP_TABLE
        elif sql_upper.startswith('ALTER TABLE'):
            if 'ADD COLUMN' in sql_upper or 'ADD ' in sql_upper:
                return MigrationType.ADD_COLUMN
            elif 'DROP COLUMN' in sql_upper or 'DROP ' in sql_upper:
                return MigrationType.DROP_COLUMN
            else:
                return MigrationType.ALTER_TABLE
        elif sql_upper.startswith('CREATE INDEX'):
            return MigrationType.ADD_INDEX
        elif sql_upper.startswith('DROP INDEX'):
            return MigrationType.DROP_INDEX
        elif sql_upper.startswith('INSERT'):
            return MigrationType.INSERT_DATA
        elif sql_upper.startswith('UPDATE'):
            return MigrationType.UPDATE_DATA
        elif sql_upper.startswith('DELETE'):
            return MigrationType.DELETE_DATA
        else:
            return MigrationType.CUSTOM
    
    def generate_migration_file(self, migration: Migration, file_path: Path) -> None:
        """生成迁移文件"""
        content = f"-- Migration: {migration.name}\n"
        content += f"-- Version: {migration.version}\n"
        content += f"-- Description: {migration.description}\n"
        content += f"-- Author: {migration.author or 'Unknown'}\n"
        content += f"-- Created: {migration.created_at.isoformat()}\n\n"
        
        # UP部分
        content += "-- UP\n"
        for operation in migration.operations:
            if operation.description:
                content += f"-- {operation.description}\n"
            content += f"{operation.sql};\n\n"
        
        # DOWN部分
        content += "-- DOWN\n"
        for operation in reversed(migration.operations):
            if operation.rollback_sql:
                content += f"{operation.rollback_sql};\n\n"
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        file_path.write_text(content, encoding='utf-8')


class MigrationValidator:
    """迁移验证器"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
    
    def validate_migration(self, migration: Migration) -> List[str]:
        """验证迁移"""
        errors = []
        
        # 验证版本格式
        if not re.match(r'^\d{14}$', migration.version):
            errors.append(f"Invalid version format: {migration.version}")
        
        # 验证名称
        if not migration.name or not migration.name.strip():
            errors.append("Migration name is required")
        
        # 验证操作
        if not migration.operations:
            errors.append("Migration must have at least one operation")
        
        for i, operation in enumerate(migration.operations):
            op_errors = self._validate_operation(operation, i)
            errors.extend(op_errors)
        
        # 验证依赖
        for dep in migration.dependencies:
            if not re.match(r'^\d{14}$', dep):
                errors.append(f"Invalid dependency version format: {dep}")
        
        return errors
    
    def _validate_operation(self, operation: MigrationOperation, index: int) -> List[str]:
        """验证操作"""
        errors = []
        
        # 验证SQL
        if not operation.sql or not operation.sql.strip():
            errors.append(f"Operation {index}: SQL is required")
        
        # 验证破坏性操作
        if not self.config.allow_destructive_operations:
            if self._is_destructive_operation(operation):
                errors.append(f"Operation {index}: Destructive operation not allowed")
        
        return errors
    
    def _is_destructive_operation(self, operation: MigrationOperation) -> bool:
        """检查是否为破坏性操作"""
        destructive_types = {
            MigrationType.DROP_TABLE,
            MigrationType.DROP_COLUMN,
            MigrationType.DROP_INDEX,
            MigrationType.DROP_CONSTRAINT,
            MigrationType.DELETE_DATA
        }
        
        return operation.operation_type in destructive_types
    
    def validate_dependencies(self, migrations: List[Migration]) -> List[str]:
        """验证依赖关系"""
        errors = []
        migration_versions = {m.version for m in migrations}
        
        for migration in migrations:
            for dep in migration.dependencies:
                if dep not in migration_versions:
                    errors.append(f"Migration {migration.version} depends on non-existent migration {dep}")
        
        # 检查循环依赖
        cycle_errors = self._check_circular_dependencies(migrations)
        errors.extend(cycle_errors)
        
        return errors
    
    def _check_circular_dependencies(self, migrations: List[Migration]) -> List[str]:
        """检查循环依赖"""
        errors = []
        
        # 构建依赖图
        graph = {m.version: set(m.dependencies) for m in migrations}
        
        # 使用DFS检测循环
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for version in graph:
            if version not in visited:
                if has_cycle(version):
                    errors.append(f"Circular dependency detected involving migration {version}")
        
        return errors


class MigrationExecutor:
    """迁移执行器"""
    
    def __init__(self, engine: Engine, config: MigrationConfig):
        self.engine = engine
        self.config = config
        self._lock = threading.Lock()
    
    def execute_migration(self, migration: Migration, direction: MigrationDirection = MigrationDirection.UP) -> bool:
        """执行迁移"""
        with self._lock:
            try:
                logger.info(f"Executing migration {migration.version} ({direction.value})")
                
                start_time = datetime.now()
                
                if self.config.transaction_per_migration:
                    with self.engine.begin() as conn:
                        success = self._execute_migration_operations(conn, migration, direction)
                else:
                    with self.engine.connect() as conn:
                        success = self._execute_migration_operations(conn, migration, direction)
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # 更新迁移状态
                if success:
                    migration.status = MigrationStatus.COMPLETED
                    if direction == MigrationDirection.UP:
                        migration.applied_at = end_time
                    else:
                        migration.rolled_back_at = end_time
                    migration.execution_time = execution_time
                    
                    # 更新迁移记录
                    self._update_migration_record(migration, direction)
                    
                    logger.info(f"Migration {migration.version} completed in {execution_time:.2f}s")
                else:
                    migration.status = MigrationStatus.FAILED
                    logger.error(f"Migration {migration.version} failed")
                
                return success
                
            except Exception as e:
                migration.status = MigrationStatus.FAILED
                migration.error_message = str(e)
                logger.error(f"Migration {migration.version} failed: {e}")
                return False
    
    def _execute_migration_operations(self, conn: Connection, migration: Migration, 
                                    direction: MigrationDirection) -> bool:
        """执行迁移操作"""
        try:
            operations = migration.operations
            if direction == MigrationDirection.DOWN:
                operations = reversed(operations)
            
            for operation in operations:
                sql = operation.sql if direction == MigrationDirection.UP else operation.rollback_sql
                
                if not sql:
                    if direction == MigrationDirection.DOWN:
                        logger.warning(f"No rollback SQL for operation: {operation.operation_type.value}")
                        continue
                    else:
                        raise MigrationExecutionError(f"No SQL for operation: {operation.operation_type.value}")
                
                # 检查执行条件
                if operation.condition and not self._evaluate_condition(conn, operation.condition):
                    logger.info(f"Skipping operation due to condition: {operation.condition}")
                    continue
                
                # 检查数据库方言
                if operation.dialect_specific and not self._is_dialect_compatible(operation.dialect_specific):
                    logger.info(f"Skipping operation for dialect: {operation.dialect_specific.value}")
                    continue
                
                logger.debug(f"Executing SQL: {sql}")
                conn.execute(text(sql))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute migration operations: {e}")
            raise MigrationExecutionError(f"Migration execution failed: {e}")
    
    def _evaluate_condition(self, conn: Connection, condition: str) -> bool:
        """评估执行条件"""
        try:
            result = conn.execute(text(condition))
            return bool(result.scalar())
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False
    
    def _is_dialect_compatible(self, dialect: DatabaseDialect) -> bool:
        """检查数据库方言兼容性"""
        engine_dialect = self.engine.dialect.name.lower()
        return dialect.value.lower() == engine_dialect
    
    def _update_migration_record(self, migration: Migration, direction: MigrationDirection) -> None:
        """更新迁移记录"""
        try:
            with self.engine.begin() as conn:
                if direction == MigrationDirection.UP:
                    # 插入或更新迁移记录
                    sql = f"""
                    INSERT INTO {self.config.migration_table} 
                    (version, name, checksum, applied_at, execution_time)
                    VALUES (:version, :name, :checksum, :applied_at, :execution_time)
                    ON CONFLICT (version) DO UPDATE SET
                        applied_at = :applied_at,
                        execution_time = :execution_time
                    """
                    
                    conn.execute(text(sql), {
                        'version': migration.version,
                        'name': migration.name,
                        'checksum': migration.checksum,
                        'applied_at': migration.applied_at,
                        'execution_time': migration.execution_time
                    })
                else:
                    # 删除迁移记录
                    sql = f"DELETE FROM {self.config.migration_table} WHERE version = :version"
                    conn.execute(text(sql), {'version': migration.version})
                    
        except Exception as e:
            logger.error(f"Failed to update migration record: {e}")
    
    def create_migration_table(self) -> None:
        """创建迁移表"""
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.config.migration_table} (
            version VARCHAR(14) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            checksum VARCHAR(32) NOT NULL,
            applied_at TIMESTAMP NOT NULL,
            execution_time FLOAT
        )
        """
        
        with self.engine.begin() as conn:
            conn.execute(text(sql))
        
        logger.info(f"Migration table '{self.config.migration_table}' created")
    
    def get_applied_migrations(self) -> List[str]:
        """获取已应用的迁移"""
        try:
            sql = f"SELECT version FROM {self.config.migration_table} ORDER BY version"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                return [row[0] for row in result]
                
        except Exception as e:
            logger.warning(f"Failed to get applied migrations: {e}")
            return []


class MigrationManager:
    """迁移管理器"""
    
    def __init__(self, engine: Engine, config: Optional[MigrationConfig] = None):
        self.engine = engine
        self.config = config or MigrationConfig()
        self.parser = MigrationFileParser()
        self.validator = MigrationValidator(self.config)
        self.executor = MigrationExecutor(engine, self.config)
        
        # 迁移存储
        self._migrations: Dict[str, Migration] = {}
        self._applied_migrations: Set[str] = set()
        
        # 统计信息
        self._stats = {
            'migrations_applied': 0,
            'migrations_rolled_back': 0,
            'total_execution_time': 0.0,
            'errors': 0
        }
        
        # 初始化
        self._initialize()
    
    def _initialize(self) -> None:
        """初始化迁移管理器"""
        try:
            # 创建迁移表
            if self.config.auto_create_migration_table:
                self.executor.create_migration_table()
            
            # 加载已应用的迁移
            self._applied_migrations = set(self.executor.get_applied_migrations())
            
            # 加载迁移文件
            self.load_migrations()
            
            logger.info("Migration manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize migration manager: {e}")
            raise MigrationError(f"Initialization failed: {e}")
    
    def load_migrations(self) -> None:
        """加载迁移文件"""
        migrations_path = Path(self.config.migrations_dir)
        if not migrations_path.exists():
            migrations_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created migrations directory: {migrations_path}")
            return
        
        # 扫描迁移文件
        migration_files = sorted(migrations_path.glob("*.sql"))
        
        for file_path in migration_files:
            try:
                migration = self.parser.parse_migration_file(file_path)
                
                # 验证迁移
                errors = self.validator.validate_migration(migration)
                if errors:
                    logger.error(f"Invalid migration {file_path.name}: {'; '.join(errors)}")
                    continue
                
                # 检查是否已应用
                if migration.version in self._applied_migrations:
                    migration.status = MigrationStatus.COMPLETED
                    migration.applied_at = datetime.now()  # 实际时间需要从数据库获取
                
                self._migrations[migration.version] = migration
                logger.debug(f"Loaded migration: {migration.version} - {migration.name}")
                
            except Exception as e:
                logger.error(f"Failed to load migration {file_path.name}: {e}")
        
        logger.info(f"Loaded {len(self._migrations)} migrations")
    
    def create_migration(self, name: str, operations: List[MigrationOperation], 
                        author: Optional[str] = None, 
                        dependencies: Optional[List[str]] = None) -> Migration:
        """创建新迁移"""
        # 生成版本号（时间戳）
        version = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 确保版本唯一
        while version in self._migrations:
            version = str(int(version) + 1)
        
        # 创建迁移
        migration = Migration(
            version=version,
            name=name,
            description=f"Migration: {name}",
            operations=operations,
            dependencies=dependencies or [],
            author=author
        )
        
        # 验证迁移
        errors = self.validator.validate_migration(migration)
        if errors:
            raise MigrationValidationError(f"Invalid migration: {'; '.join(errors)}")
        
        # 生成迁移文件
        file_name = f"{version}_{name.lower().replace(' ', '_')}.sql"
        file_path = Path(self.config.migrations_dir) / file_name
        self.parser.generate_migration_file(migration, file_path)
        
        # 添加到管理器
        self._migrations[version] = migration
        
        logger.info(f"Created migration: {version} - {name}")
        
        # 发布事件
        emit_business_event(
            EventType.MIGRATION_CREATED,
            "migration_management",
            data=migration.to_dict()
        )
        
        return migration
    
    def apply_migrations(self, target_version: Optional[str] = None) -> List[Migration]:
        """应用迁移"""
        try:
            # 获取待应用的迁移
            pending_migrations = self.get_pending_migrations(target_version)
            
            if not pending_migrations:
                logger.info("No pending migrations to apply")
                return []
            
            # 验证依赖关系
            dep_errors = self.validator.validate_dependencies(list(self._migrations.values()))
            if dep_errors:
                raise MigrationDependencyError(f"Dependency errors: {'; '.join(dep_errors)}")
            
            applied_migrations = []
            
            for migration in pending_migrations:
                logger.info(f"Applying migration: {migration.version} - {migration.name}")
                
                # 检查依赖
                if not self._check_dependencies(migration):
                    logger.error(f"Dependencies not met for migration {migration.version}")
                    continue
                
                # 执行迁移
                success = self.executor.execute_migration(migration, MigrationDirection.UP)
                
                if success:
                    self._applied_migrations.add(migration.version)
                    applied_migrations.append(migration)
                    self._stats['migrations_applied'] += 1
                    if migration.execution_time:
                        self._stats['total_execution_time'] += migration.execution_time
                    
                    # 发布事件
                    emit_business_event(
                        EventType.MIGRATION_APPLIED,
                        "migration_management",
                        data=migration.to_dict()
                    )
                else:
                    self._stats['errors'] += 1
                    logger.error(f"Failed to apply migration {migration.version}")
                    break
            
            logger.info(f"Applied {len(applied_migrations)} migrations")
            return applied_migrations
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to apply migrations: {e}")
            raise MigrationError(f"Migration application failed: {e}")
    
    def rollback_migration(self, version: str) -> bool:
        """回滚迁移"""
        try:
            migration = self._migrations.get(version)
            if not migration:
                raise MigrationError(f"Migration not found: {version}")
            
            if not migration.is_applied:
                logger.warning(f"Migration {version} is not applied")
                return False
            
            if not migration.can_rollback:
                raise MigrationError(f"Migration {version} cannot be rolled back")
            
            logger.info(f"Rolling back migration: {version} - {migration.name}")
            
            # 执行回滚
            success = self.executor.execute_migration(migration, MigrationDirection.DOWN)
            
            if success:
                self._applied_migrations.discard(version)
                migration.status = MigrationStatus.ROLLED_BACK
                self._stats['migrations_rolled_back'] += 1
                
                # 发布事件
                emit_business_event(
                    EventType.MIGRATION_ROLLED_BACK,
                    "migration_management",
                    data=migration.to_dict()
                )
                
                logger.info(f"Successfully rolled back migration {version}")
            else:
                self._stats['errors'] += 1
                logger.error(f"Failed to rollback migration {version}")
            
            return success
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to rollback migration {version}: {e}")
            return False
    
    def get_pending_migrations(self, target_version: Optional[str] = None) -> List[Migration]:
        """获取待应用的迁移"""
        pending = []
        
        for version in sorted(self._migrations.keys()):
            migration = self._migrations[version]
            
            # 检查目标版本
            if target_version and version > target_version:
                break
            
            # 检查是否已应用
            if version not in self._applied_migrations:
                pending.append(migration)
        
        return pending
    
    def get_applied_migrations(self) -> List[Migration]:
        """获取已应用的迁移"""
        applied = []
        
        for version in sorted(self._applied_migrations):
            if version in self._migrations:
                applied.append(self._migrations[version])
        
        return applied
    
    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        total_migrations = len(self._migrations)
        applied_migrations = len(self._applied_migrations)
        pending_migrations = total_migrations - applied_migrations
        
        return {
            'total_migrations': total_migrations,
            'applied_migrations': applied_migrations,
            'pending_migrations': pending_migrations,
            'latest_version': max(self._migrations.keys()) if self._migrations else None,
            'latest_applied_version': max(self._applied_migrations) if self._applied_migrations else None,
            'statistics': self._stats.copy()
        }
    
    def _check_dependencies(self, migration: Migration) -> bool:
        """检查迁移依赖"""
        for dep_version in migration.dependencies:
            if dep_version not in self._applied_migrations:
                logger.error(f"Dependency {dep_version} not applied for migration {migration.version}")
                return False
        return True
    
    def validate_all_migrations(self) -> Dict[str, List[str]]:
        """验证所有迁移"""
        validation_results = {}
        
        for version, migration in self._migrations.items():
            errors = self.validator.validate_migration(migration)
            if errors:
                validation_results[version] = errors
        
        # 验证依赖关系
        dep_errors = self.validator.validate_dependencies(list(self._migrations.values()))
        if dep_errors:
            validation_results['dependencies'] = dep_errors
        
        return validation_results
    
    def export_schema(self, file_path: Optional[Path] = None) -> str:
        """导出数据库模式"""
        if not file_path:
            file_path = Path(self.config.schema_file)
        
        try:
            # 获取数据库元数据
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            
            schema_sql = ""
            
            # 生成建表语句
            for table in metadata.sorted_tables:
                create_stmt = CreateTable(table)
                schema_sql += str(create_stmt.compile(self.engine)) + ";\n\n"
            
            # 写入文件
            if file_path:
                file_path.write_text(schema_sql, encoding='utf-8')
                logger.info(f"Schema exported to {file_path}")
            
            return schema_sql
            
        except Exception as e:
            logger.error(f"Failed to export schema: {e}")
            raise MigrationError(f"Schema export failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()


# 迁移装饰器
def migration(name: str, dependencies: Optional[List[str]] = None):
    """迁移装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 执行函数获取操作列表
            operations = func(*args, **kwargs)
            
            if not isinstance(operations, list):
                operations = [operations]
            
            # 获取默认迁移管理器
            manager = get_default_migration_manager()
            if manager:
                return manager.create_migration(name, operations, dependencies=dependencies)
            
            return None
        
        return wrapper
    return decorator


# 全局迁移管理器
_default_migration_manager: Optional[MigrationManager] = None


def initialize_migration(engine: Engine, config: Optional[MigrationConfig] = None) -> MigrationManager:
    """初始化迁移管理器"""
    global _default_migration_manager
    _default_migration_manager = MigrationManager(engine, config)
    return _default_migration_manager


def get_default_migration_manager() -> Optional[MigrationManager]:
    """获取默认迁移管理器"""
    return _default_migration_manager


# 便捷函数
def create_migration(name: str, operations: List[MigrationOperation], 
                    author: Optional[str] = None, 
                    dependencies: Optional[List[str]] = None) -> Optional[Migration]:
    """创建迁移"""
    manager = get_default_migration_manager()
    if manager:
        return manager.create_migration(name, operations, author, dependencies)
    return None


def apply_migrations(target_version: Optional[str] = None) -> List[Migration]:
    """应用迁移"""
    manager = get_default_migration_manager()
    if manager:
        return manager.apply_migrations(target_version)
    return []


def rollback_migration(version: str) -> bool:
    """回滚迁移"""
    manager = get_default_migration_manager()
    if manager:
        return manager.rollback_migration(version)
    return False


def get_migration_status() -> Dict[str, Any]:
    """获取迁移状态"""
    manager = get_default_migration_manager()
    if manager:
        return manager.get_migration_status()
    return {}


def get_migration_statistics() -> Dict[str, Any]:
    """获取迁移统计"""
    manager = get_default_migration_manager()
    if manager:
        return manager.get_statistics()
    return {}


def validate_migrations() -> Dict[str, List[str]]:
    """验证迁移"""
    manager = get_default_migration_manager()
    if manager:
        return manager.validate_all_migrations()
    return {}


def export_schema(file_path: Optional[Path] = None) -> str:
    """导出模式"""
    manager = get_default_migration_manager()
    if manager:
        return manager.export_schema(file_path)
    return ""


# 导出所有类和函数
__all__ = [
    "MigrationType",
    "MigrationStatus",
    "MigrationDirection",
    "DatabaseDialect",
    "MigrationOperation",
    "Migration",
    "MigrationConfig",
    "MigrationError",
    "MigrationValidationError",
    "MigrationExecutionError",
    "MigrationDependencyError",
    "MigrationFileParser",
    "MigrationValidator",
    "MigrationExecutor",
    "MigrationManager",
    "migration",
    "initialize_migration",
    "get_default_migration_manager",
    "create_migration",
    "apply_migrations",
    "rollback_migration",
    "get_migration_status",
    "get_migration_statistics",
    "validate_migrations",
    "export_schema"
]