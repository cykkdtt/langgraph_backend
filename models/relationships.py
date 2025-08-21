"""模型关系系统模块

本模块提供数据库关系管理、外键约束和关联查询功能。
"""

from typing import (
    Dict, Any, Optional, List, Union, Callable, Type, 
    Tuple, Set, ClassVar, Protocol, TypeVar, Generic,
    NamedTuple, AsyncGenerator, Awaitable, ForwardRef
)
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps, cached_property
import logging
import inspect
from collections import defaultdict, deque
from contextlib import contextmanager
from weakref import WeakSet, WeakKeyDictionary

# SQLAlchemy imports
try:
    from sqlalchemy import (
        Column, ForeignKey, Table, MetaData, 
        Integer, String, DateTime, Boolean,
        inspect as sa_inspect, event
    )
    from sqlalchemy.orm import (
        relationship, backref, Session, sessionmaker,
        declarative_base, declared_attr, validates
    )
    from sqlalchemy.ext.declarative import DeclarativeMeta
    from sqlalchemy.sql import select, join, and_, or_
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    Column = None
    ForeignKey = None
    Table = None
    MetaData = None
    relationship = None
    backref = None
    Session = None
    sessionmaker = None
    declarative_base = None
    declared_attr = None
    validates = None
    DeclarativeMeta = None
    sa_inspect = None
    event = None
    select = None
    join = None
    and_ = None
    or_ = None
    SQLALCHEMY_AVAILABLE = False

# 导入项目模块
from .events import EventType, emit_business_event
from .validation import validate_data


logger = logging.getLogger(__name__)


T = TypeVar('T')
Model = TypeVar('Model')
Related = TypeVar('Related')


class RelationshipType(Enum):
    """关系类型枚举"""
    ONE_TO_ONE = "one_to_one"              # 一对一关系
    ONE_TO_MANY = "one_to_many"            # 一对多关系
    MANY_TO_ONE = "many_to_one"            # 多对一关系
    MANY_TO_MANY = "many_to_many"          # 多对多关系
    SELF_REFERENTIAL = "self_referential"  # 自引用关系
    POLYMORPHIC = "polymorphic"            # 多态关系


class CascadeType(Enum):
    """级联类型枚举"""
    NONE = "none"                          # 无级联
    SAVE_UPDATE = "save-update"            # 保存更新级联
    MERGE = "merge"                        # 合并级联
    DELETE = "delete"                      # 删除级联
    DELETE_ORPHAN = "delete-orphan"        # 删除孤儿级联
    REFRESH = "refresh"                    # 刷新级联
    EXPUNGE = "expunge"                    # 清除级联
    ALL = "all"                            # 所有级联


class LoadingStrategy(Enum):
    """加载策略枚举"""
    LAZY = "lazy"                          # 懒加载
    EAGER = "eager"                        # 急加载
    SELECT = "select"                      # 选择加载
    JOINED = "joined"                      # 连接加载
    SUBQUERY = "subquery"                  # 子查询加载
    DYNAMIC = "dynamic"                    # 动态加载
    NOLOAD = "noload"                      # 不加载


class ConstraintType(Enum):
    """约束类型枚举"""
    FOREIGN_KEY = "foreign_key"            # 外键约束
    UNIQUE = "unique"                      # 唯一约束
    CHECK = "check"                        # 检查约束
    NOT_NULL = "not_null"                  # 非空约束
    PRIMARY_KEY = "primary_key"            # 主键约束
    INDEX = "index"                        # 索引约束


@dataclass
class RelationshipDefinition:
    """关系定义"""
    # 基本信息
    name: str                              # 关系名称
    source_model: Type                     # 源模型
    target_model: Union[Type, str]         # 目标模型
    relationship_type: RelationshipType    # 关系类型
    
    # 外键信息
    foreign_key: Optional[str] = None      # 外键字段
    back_populates: Optional[str] = None   # 反向关系
    backref_name: Optional[str] = None     # 反向引用名称
    
    # 加载策略
    loading_strategy: LoadingStrategy = LoadingStrategy.LAZY
    cascade: List[CascadeType] = field(default_factory=list)
    
    # 多对多关系
    association_table: Optional[str] = None  # 关联表名
    secondary_table: Optional[Table] = None  # 中间表
    
    # 条件和排序
    condition: Optional[str] = None        # 关系条件
    order_by: Optional[str] = None         # 排序字段
    
    # 其他选项
    nullable: bool = True                  # 是否可为空
    unique: bool = False                   # 是否唯一
    passive_deletes: bool = False          # 被动删除
    passive_updates: bool = True           # 被动更新
    
    # 元数据
    description: Optional[str] = None      # 描述
    tags: List[str] = field(default_factory=list)  # 标签
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """初始化后处理"""
        # 验证关系类型和配置的一致性
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """验证配置"""
        # 验证多对多关系必须有关联表
        if self.relationship_type == RelationshipType.MANY_TO_MANY:
            if not self.association_table and not self.secondary_table:
                raise ValueError("Many-to-many relationship requires association_table or secondary_table")
        
        # 验证一对一关系的唯一性
        if self.relationship_type == RelationshipType.ONE_TO_ONE:
            self.unique = True
    
    def get_sqlalchemy_kwargs(self) -> Dict[str, Any]:
        """获取SQLAlchemy关系参数"""
        kwargs = {}
        
        # 基本参数
        if self.back_populates:
            kwargs['back_populates'] = self.back_populates
        
        if self.backref_name:
            kwargs['backref'] = self.backref_name
        
        # 加载策略
        if self.loading_strategy == LoadingStrategy.LAZY:
            kwargs['lazy'] = 'select'
        elif self.loading_strategy == LoadingStrategy.EAGER:
            kwargs['lazy'] = False
        elif self.loading_strategy == LoadingStrategy.JOINED:
            kwargs['lazy'] = 'joined'
        elif self.loading_strategy == LoadingStrategy.SUBQUERY:
            kwargs['lazy'] = 'subquery'
        elif self.loading_strategy == LoadingStrategy.DYNAMIC:
            kwargs['lazy'] = 'dynamic'
        elif self.loading_strategy == LoadingStrategy.NOLOAD:
            kwargs['lazy'] = 'noload'
        
        # 级联
        if self.cascade:
            cascade_str = ', '.join([c.value for c in self.cascade])
            kwargs['cascade'] = cascade_str
        
        # 多对多关系
        if self.secondary_table:
            kwargs['secondary'] = self.secondary_table
        
        # 条件和排序
        if self.condition:
            kwargs['primaryjoin'] = self.condition
        
        if self.order_by:
            kwargs['order_by'] = self.order_by
        
        # 其他选项
        kwargs['passive_deletes'] = self.passive_deletes
        kwargs['passive_updates'] = self.passive_updates
        
        return kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'source_model': self.source_model.__name__ if hasattr(self.source_model, '__name__') else str(self.source_model),
            'target_model': self.target_model.__name__ if hasattr(self.target_model, '__name__') else str(self.target_model),
            'relationship_type': self.relationship_type.value,
            'foreign_key': self.foreign_key,
            'back_populates': self.back_populates,
            'backref_name': self.backref_name,
            'loading_strategy': self.loading_strategy.value,
            'cascade': [c.value for c in self.cascade],
            'association_table': self.association_table,
            'condition': self.condition,
            'order_by': self.order_by,
            'nullable': self.nullable,
            'unique': self.unique,
            'passive_deletes': self.passive_deletes,
            'passive_updates': self.passive_updates,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ConstraintDefinition:
    """约束定义"""
    name: str                              # 约束名称
    constraint_type: ConstraintType        # 约束类型
    table_name: str                        # 表名
    columns: List[str]                     # 列名列表
    
    # 外键约束
    referenced_table: Optional[str] = None # 引用表
    referenced_columns: Optional[List[str]] = None  # 引用列
    on_delete: Optional[str] = None        # 删除时动作
    on_update: Optional[str] = None        # 更新时动作
    
    # 检查约束
    check_condition: Optional[str] = None  # 检查条件
    
    # 索引约束
    index_type: Optional[str] = None       # 索引类型
    unique: bool = False                   # 是否唯一
    
    # 元数据
    description: Optional[str] = None      # 描述
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'constraint_type': self.constraint_type.value,
            'table_name': self.table_name,
            'columns': self.columns,
            'referenced_table': self.referenced_table,
            'referenced_columns': self.referenced_columns,
            'on_delete': self.on_delete,
            'on_update': self.on_update,
            'check_condition': self.check_condition,
            'index_type': self.index_type,
            'unique': self.unique,
            'description': self.description,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class RelationshipQuery:
    """关系查询"""
    source_model: Type                     # 源模型
    target_model: Type                     # 目标模型
    relationship_name: str                 # 关系名称
    
    # 查询条件
    filters: Dict[str, Any] = field(default_factory=dict)
    joins: List[str] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    
    # 分页
    limit: Optional[int] = None
    offset: Optional[int] = None
    
    # 加载选项
    eager_load: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'source_model': self.source_model.__name__,
            'target_model': self.target_model.__name__,
            'relationship_name': self.relationship_name,
            'filters': self.filters,
            'joins': self.joins,
            'order_by': self.order_by,
            'limit': self.limit,
            'offset': self.offset,
            'eager_load': self.eager_load
        }


class RelationshipError(Exception):
    """关系错误"""
    pass


class ConstraintError(Exception):
    """约束错误"""
    pass


class CircularReferenceError(RelationshipError):
    """循环引用错误"""
    pass


class RelationshipRegistry:
    """关系注册表"""
    
    def __init__(self):
        self._relationships: Dict[str, RelationshipDefinition] = {}
        self._constraints: Dict[str, ConstraintDefinition] = {}
        self._model_relationships: Dict[Type, List[RelationshipDefinition]] = defaultdict(list)
        self._dependency_graph: Dict[Type, Set[Type]] = defaultdict(set)
        self._reverse_dependency_graph: Dict[Type, Set[Type]] = defaultdict(set)
    
    def register_relationship(self, definition: RelationshipDefinition) -> None:
        """注册关系"""
        key = f"{definition.source_model.__name__}.{definition.name}"
        
        if key in self._relationships:
            logger.warning(f"Relationship {key} already registered, overwriting")
        
        self._relationships[key] = definition
        self._model_relationships[definition.source_model].append(definition)
        
        # 更新依赖图
        target_model = definition.target_model
        if isinstance(target_model, str):
            # 延迟解析
            pass
        else:
            self._dependency_graph[definition.source_model].add(target_model)
            self._reverse_dependency_graph[target_model].add(definition.source_model)
        
        logger.info(f"Registered relationship: {key}")
    
    def register_constraint(self, definition: ConstraintDefinition) -> None:
        """注册约束"""
        if definition.name in self._constraints:
            logger.warning(f"Constraint {definition.name} already registered, overwriting")
        
        self._constraints[definition.name] = definition
        logger.info(f"Registered constraint: {definition.name}")
    
    def get_relationship(self, source_model: Type, name: str) -> Optional[RelationshipDefinition]:
        """获取关系"""
        key = f"{source_model.__name__}.{name}"
        return self._relationships.get(key)
    
    def get_model_relationships(self, model: Type) -> List[RelationshipDefinition]:
        """获取模型的所有关系"""
        return self._model_relationships.get(model, [])
    
    def get_constraint(self, name: str) -> Optional[ConstraintDefinition]:
        """获取约束"""
        return self._constraints.get(name)
    
    def get_all_relationships(self) -> Dict[str, RelationshipDefinition]:
        """获取所有关系"""
        return self._relationships.copy()
    
    def get_all_constraints(self) -> Dict[str, ConstraintDefinition]:
        """获取所有约束"""
        return self._constraints.copy()
    
    def check_circular_dependencies(self) -> List[List[Type]]:
        """检查循环依赖"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: Type, path: List[Type]) -> None:
            if node in rec_stack:
                # 找到循环
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self._dependency_graph.get(node, set()):
                dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for model in self._dependency_graph:
            if model not in visited:
                dfs(model, [])
        
        return cycles
    
    def get_dependency_order(self) -> List[Type]:
        """获取依赖顺序（拓扑排序）"""
        # Kahn算法
        in_degree = defaultdict(int)
        
        # 计算入度
        for model in self._dependency_graph:
            for dependent in self._dependency_graph[model]:
                in_degree[dependent] += 1
        
        # 初始化队列
        queue = deque([model for model in self._dependency_graph if in_degree[model] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in self._dependency_graph.get(current, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否有循环依赖
        if len(result) != len(self._dependency_graph):
            cycles = self.check_circular_dependencies()
            raise CircularReferenceError(f"Circular dependencies detected: {cycles}")
        
        return result
    
    def clear(self) -> None:
        """清空注册表"""
        self._relationships.clear()
        self._constraints.clear()
        self._model_relationships.clear()
        self._dependency_graph.clear()
        self._reverse_dependency_graph.clear()


class RelationshipBuilder:
    """关系构建器"""
    
    def __init__(self, registry: RelationshipRegistry):
        self.registry = registry
        self._association_tables: Dict[str, Table] = {}
    
    def one_to_one(self, source_model: Type, target_model: Type, 
                   name: str, foreign_key: str, **kwargs) -> RelationshipDefinition:
        """创建一对一关系"""
        definition = RelationshipDefinition(
            name=name,
            source_model=source_model,
            target_model=target_model,
            relationship_type=RelationshipType.ONE_TO_ONE,
            foreign_key=foreign_key,
            unique=True,
            **kwargs
        )
        
        self.registry.register_relationship(definition)
        return definition
    
    def one_to_many(self, source_model: Type, target_model: Type, 
                    name: str, foreign_key: str, **kwargs) -> RelationshipDefinition:
        """创建一对多关系"""
        definition = RelationshipDefinition(
            name=name,
            source_model=source_model,
            target_model=target_model,
            relationship_type=RelationshipType.ONE_TO_MANY,
            foreign_key=foreign_key,
            **kwargs
        )
        
        self.registry.register_relationship(definition)
        return definition
    
    def many_to_one(self, source_model: Type, target_model: Type, 
                    name: str, foreign_key: str, **kwargs) -> RelationshipDefinition:
        """创建多对一关系"""
        definition = RelationshipDefinition(
            name=name,
            source_model=source_model,
            target_model=target_model,
            relationship_type=RelationshipType.MANY_TO_ONE,
            foreign_key=foreign_key,
            **kwargs
        )
        
        self.registry.register_relationship(definition)
        return definition
    
    def many_to_many(self, source_model: Type, target_model: Type, 
                     name: str, association_table: str, **kwargs) -> RelationshipDefinition:
        """创建多对多关系"""
        definition = RelationshipDefinition(
            name=name,
            source_model=source_model,
            target_model=target_model,
            relationship_type=RelationshipType.MANY_TO_MANY,
            association_table=association_table,
            **kwargs
        )
        
        self.registry.register_relationship(definition)
        return definition
    
    def self_referential(self, model: Type, name: str, foreign_key: str, 
                        **kwargs) -> RelationshipDefinition:
        """创建自引用关系"""
        definition = RelationshipDefinition(
            name=name,
            source_model=model,
            target_model=model,
            relationship_type=RelationshipType.SELF_REFERENTIAL,
            foreign_key=foreign_key,
            **kwargs
        )
        
        self.registry.register_relationship(definition)
        return definition
    
    def create_association_table(self, name: str, metadata: MetaData, 
                                left_table: str, right_table: str,
                                left_column: str = "id", right_column: str = "id") -> Table:
        """创建关联表"""
        if not SQLALCHEMY_AVAILABLE:
            raise RelationshipError("SQLAlchemy is not available")
        
        table = Table(
            name, metadata,
            Column(f"{left_table}_id", Integer, ForeignKey(f"{left_table}.{left_column}"), primary_key=True),
            Column(f"{right_table}_id", Integer, ForeignKey(f"{right_table}.{right_column}"), primary_key=True)
        )
        
        self._association_tables[name] = table
        return table
    
    def get_association_table(self, name: str) -> Optional[Table]:
        """获取关联表"""
        return self._association_tables.get(name)


class RelationshipQueryBuilder:
    """关系查询构建器"""
    
    def __init__(self, registry: RelationshipRegistry):
        self.registry = registry
    
    def build_query(self, query_def: RelationshipQuery, session: Session) -> Any:
        """构建查询"""
        if not SQLALCHEMY_AVAILABLE:
            raise RelationshipError("SQLAlchemy is not available")
        
        # 获取关系定义
        relationship = self.registry.get_relationship(
            query_def.source_model, 
            query_def.relationship_name
        )
        
        if not relationship:
            raise RelationshipError(
                f"Relationship {query_def.relationship_name} not found for {query_def.source_model.__name__}"
            )
        
        # 构建基础查询
        query = session.query(query_def.target_model)
        
        # 添加连接
        for join_rel in query_def.joins:
            join_relationship = self.registry.get_relationship(query_def.target_model, join_rel)
            if join_relationship:
                query = query.join(join_relationship.target_model)
        
        # 添加过滤条件
        for field, value in query_def.filters.items():
            if hasattr(query_def.target_model, field):
                attr = getattr(query_def.target_model, field)
                if isinstance(value, (list, tuple)):
                    query = query.filter(attr.in_(value))
                else:
                    query = query.filter(attr == value)
        
        # 添加排序
        for order_field in query_def.order_by:
            if order_field.startswith('-'):
                field_name = order_field[1:]
                if hasattr(query_def.target_model, field_name):
                    attr = getattr(query_def.target_model, field_name)
                    query = query.order_by(attr.desc())
            else:
                if hasattr(query_def.target_model, order_field):
                    attr = getattr(query_def.target_model, order_field)
                    query = query.order_by(attr)
        
        # 添加急加载
        for eager_rel in query_def.eager_load:
            if hasattr(query_def.target_model, eager_rel):
                attr = getattr(query_def.target_model, eager_rel)
                query = query.options(attr.load())
        
        # 添加分页
        if query_def.offset:
            query = query.offset(query_def.offset)
        
        if query_def.limit:
            query = query.limit(query_def.limit)
        
        return query
    
    def execute_query(self, query_def: RelationshipQuery, session: Session) -> List[Any]:
        """执行查询"""
        query = self.build_query(query_def, session)
        return query.all()
    
    def count_query(self, query_def: RelationshipQuery, session: Session) -> int:
        """计算查询结果数量"""
        query = self.build_query(query_def, session)
        return query.count()


class RelationshipManager:
    """关系管理器"""
    
    def __init__(self):
        self.registry = RelationshipRegistry()
        self.builder = RelationshipBuilder(self.registry)
        self.query_builder = RelationshipQueryBuilder(self.registry)
        
        # 统计信息
        self._stats = {
            'relationships_registered': 0,
            'constraints_registered': 0,
            'queries_executed': 0,
            'errors': 0
        }
    
    def register_relationship(self, definition: RelationshipDefinition) -> None:
        """注册关系"""
        try:
            self.registry.register_relationship(definition)
            self._stats['relationships_registered'] += 1
            
            # 发布事件
            emit_business_event(
                EventType.RELATIONSHIP_REGISTERED,
                "relationship_management",
                data=definition.to_dict()
            )
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to register relationship: {e}")
            raise
    
    def register_constraint(self, definition: ConstraintDefinition) -> None:
        """注册约束"""
        try:
            self.registry.register_constraint(definition)
            self._stats['constraints_registered'] += 1
            
            # 发布事件
            emit_business_event(
                EventType.CONSTRAINT_REGISTERED,
                "relationship_management",
                data=definition.to_dict()
            )
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to register constraint: {e}")
            raise
    
    def create_one_to_one(self, source_model: Type, target_model: Type, 
                         name: str, foreign_key: str, **kwargs) -> RelationshipDefinition:
        """创建一对一关系"""
        return self.builder.one_to_one(source_model, target_model, name, foreign_key, **kwargs)
    
    def create_one_to_many(self, source_model: Type, target_model: Type, 
                          name: str, foreign_key: str, **kwargs) -> RelationshipDefinition:
        """创建一对多关系"""
        return self.builder.one_to_many(source_model, target_model, name, foreign_key, **kwargs)
    
    def create_many_to_one(self, source_model: Type, target_model: Type, 
                          name: str, foreign_key: str, **kwargs) -> RelationshipDefinition:
        """创建多对一关系"""
        return self.builder.many_to_one(source_model, target_model, name, foreign_key, **kwargs)
    
    def create_many_to_many(self, source_model: Type, target_model: Type, 
                           name: str, association_table: str, **kwargs) -> RelationshipDefinition:
        """创建多对多关系"""
        return self.builder.many_to_many(source_model, target_model, name, association_table, **kwargs)
    
    def create_self_referential(self, model: Type, name: str, foreign_key: str, 
                               **kwargs) -> RelationshipDefinition:
        """创建自引用关系"""
        return self.builder.self_referential(model, name, foreign_key, **kwargs)
    
    def query_relationship(self, query_def: RelationshipQuery, session: Session) -> List[Any]:
        """查询关系"""
        try:
            result = self.query_builder.execute_query(query_def, session)
            self._stats['queries_executed'] += 1
            
            # 发布事件
            emit_business_event(
                EventType.RELATIONSHIP_QUERIED,
                "relationship_management",
                data={
                    'query': query_def.to_dict(),
                    'result_count': len(result)
                }
            )
            
            return result
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to query relationship: {e}")
            raise
    
    def get_model_relationships(self, model: Type) -> List[RelationshipDefinition]:
        """获取模型关系"""
        return self.registry.get_model_relationships(model)
    
    def get_dependency_order(self) -> List[Type]:
        """获取依赖顺序"""
        return self.registry.get_dependency_order()
    
    def check_circular_dependencies(self) -> List[List[Type]]:
        """检查循环依赖"""
        return self.registry.check_circular_dependencies()
    
    def validate_relationships(self) -> Dict[str, List[str]]:
        """验证关系"""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # 检查循环依赖
        try:
            cycles = self.check_circular_dependencies()
            if cycles:
                for cycle in cycles:
                    cycle_str = ' -> '.join([model.__name__ for model in cycle])
                    issues['errors'].append(f"Circular dependency: {cycle_str}")
        except Exception as e:
            issues['errors'].append(f"Failed to check circular dependencies: {e}")
        
        # 检查关系一致性
        for relationship in self.registry.get_all_relationships().values():
            # 检查反向关系
            if relationship.back_populates:
                target_model = relationship.target_model
                if not isinstance(target_model, str):
                    reverse_rel = self.registry.get_relationship(target_model, relationship.back_populates)
                    if not reverse_rel:
                        issues['warnings'].append(
                            f"Reverse relationship {relationship.back_populates} not found for {target_model.__name__}"
                        )
        
        return issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats.update({
            'total_relationships': len(self.registry.get_all_relationships()),
            'total_constraints': len(self.registry.get_all_constraints()),
            'models_with_relationships': len(self.registry._model_relationships)
        })
        return stats
    
    def export_schema(self) -> Dict[str, Any]:
        """导出关系模式"""
        return {
            'relationships': {
                key: rel.to_dict() 
                for key, rel in self.registry.get_all_relationships().items()
            },
            'constraints': {
                key: constraint.to_dict() 
                for key, constraint in self.registry.get_all_constraints().items()
            },
            'dependency_order': [
                model.__name__ for model in self.get_dependency_order()
            ],
            'statistics': self.get_statistics()
        }


# 关系装饰器
def relationship_field(relationship_type: RelationshipType, target_model: Union[Type, str], 
                      **kwargs):
    """关系字段装饰器"""
    def decorator(func_or_attr):
        # 存储关系元数据
        if not hasattr(func_or_attr, '_relationship_metadata'):
            func_or_attr._relationship_metadata = []
        
        func_or_attr._relationship_metadata.append({
            'type': relationship_type,
            'target': target_model,
            'options': kwargs
        })
        
        return func_or_attr
    
    return decorator


def one_to_one(target_model: Union[Type, str], **kwargs):
    """一对一关系装饰器"""
    return relationship_field(RelationshipType.ONE_TO_ONE, target_model, **kwargs)


def one_to_many(target_model: Union[Type, str], **kwargs):
    """一对多关系装饰器"""
    return relationship_field(RelationshipType.ONE_TO_MANY, target_model, **kwargs)


def many_to_one(target_model: Union[Type, str], **kwargs):
    """多对一关系装饰器"""
    return relationship_field(RelationshipType.MANY_TO_ONE, target_model, **kwargs)


def many_to_many(target_model: Union[Type, str], **kwargs):
    """多对多关系装饰器"""
    return relationship_field(RelationshipType.MANY_TO_MANY, target_model, **kwargs)


def constraint(constraint_type: ConstraintType, **kwargs):
    """约束装饰器"""
    def decorator(cls: Type) -> Type:
        if not hasattr(cls, '_constraint_metadata'):
            cls._constraint_metadata = []
        
        cls._constraint_metadata.append({
            'type': constraint_type,
            'options': kwargs
        })
        
        return cls
    
    return decorator


# 全局关系管理器
_default_relationship_manager: Optional[RelationshipManager] = None


def initialize_relationships() -> RelationshipManager:
    """初始化关系管理器"""
    global _default_relationship_manager
    _default_relationship_manager = RelationshipManager()
    return _default_relationship_manager


def get_default_relationship_manager() -> Optional[RelationshipManager]:
    """获取默认关系管理器"""
    return _default_relationship_manager


# 便捷函数
def register_relationship(definition: RelationshipDefinition) -> None:
    """注册关系"""
    manager = get_default_relationship_manager()
    if not manager:
        manager = initialize_relationships()
    
    manager.register_relationship(definition)


def register_constraint(definition: ConstraintDefinition) -> None:
    """注册约束"""
    manager = get_default_relationship_manager()
    if not manager:
        manager = initialize_relationships()
    
    manager.register_constraint(definition)


def create_one_to_one(source_model: Type, target_model: Type, 
                     name: str, foreign_key: str, **kwargs) -> RelationshipDefinition:
    """创建一对一关系"""
    manager = get_default_relationship_manager()
    if not manager:
        manager = initialize_relationships()
    
    return manager.create_one_to_one(source_model, target_model, name, foreign_key, **kwargs)


def create_one_to_many(source_model: Type, target_model: Type, 
                      name: str, foreign_key: str, **kwargs) -> RelationshipDefinition:
    """创建一对多关系"""
    manager = get_default_relationship_manager()
    if not manager:
        manager = initialize_relationships()
    
    return manager.create_one_to_many(source_model, target_model, name, foreign_key, **kwargs)


def create_many_to_one(source_model: Type, target_model: Type, 
                      name: str, foreign_key: str, **kwargs) -> RelationshipDefinition:
    """创建多对一关系"""
    manager = get_default_relationship_manager()
    if not manager:
        manager = initialize_relationships()
    
    return manager.create_many_to_one(source_model, target_model, name, foreign_key, **kwargs)


def create_many_to_many(source_model: Type, target_model: Type, 
                       name: str, association_table: str, **kwargs) -> RelationshipDefinition:
    """创建多对多关系"""
    manager = get_default_relationship_manager()
    if not manager:
        manager = initialize_relationships()
    
    return manager.create_many_to_many(source_model, target_model, name, association_table, **kwargs)


def query_relationship(query_def: RelationshipQuery, session: Session) -> List[Any]:
    """查询关系"""
    manager = get_default_relationship_manager()
    if not manager:
        manager = initialize_relationships()
    
    return manager.query_relationship(query_def, session)


def get_model_relationships(model: Type) -> List[RelationshipDefinition]:
    """获取模型关系"""
    manager = get_default_relationship_manager()
    if manager:
        return manager.get_model_relationships(model)
    return []


def validate_relationships() -> Dict[str, List[str]]:
    """验证关系"""
    manager = get_default_relationship_manager()
    if manager:
        return manager.validate_relationships()
    return {'errors': [], 'warnings': []}


def get_relationship_statistics() -> Dict[str, Any]:
    """获取关系统计"""
    manager = get_default_relationship_manager()
    if manager:
        return manager.get_statistics()
    return {}


def export_relationship_schema() -> Dict[str, Any]:
    """导出关系模式"""
    manager = get_default_relationship_manager()
    if manager:
        return manager.export_schema()
    return {}


# 导出所有类和函数
__all__ = [
    "RelationshipType",
    "CascadeType",
    "LoadingStrategy",
    "ConstraintType",
    "RelationshipDefinition",
    "ConstraintDefinition",
    "RelationshipQuery",
    "RelationshipError",
    "ConstraintError",
    "CircularReferenceError",
    "RelationshipRegistry",
    "RelationshipBuilder",
    "RelationshipQueryBuilder",
    "RelationshipManager",
    "relationship_field",
    "one_to_one",
    "one_to_many",
    "many_to_one",
    "many_to_many",
    "constraint",
    "initialize_relationships",
    "get_default_relationship_manager",
    "register_relationship",
    "register_constraint",
    "create_one_to_one",
    "create_one_to_many",
    "create_many_to_one",
    "create_many_to_many",
    "query_relationship",
    "get_model_relationships",
    "validate_relationships",
    "get_relationship_statistics",
    "export_relationship_schema"
]