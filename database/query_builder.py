"""数据库查询构建器模块

提供类型安全的查询构建、查询优化和复杂查询支持。
"""

import logging
from typing import (
    Type, TypeVar, Generic, Optional, List, Dict, Any, Union, 
    Callable, Tuple, Set
)
from datetime import datetime, date
from enum import Enum

from sqlalchemy import (
    select, insert, update, delete, func, and_, or_, not_, 
    text, case, cast, literal_column, desc, asc
)
from sqlalchemy.orm import (
    Query, Session, joinedload, selectinload, subqueryload,
    contains_eager, Load
)
from sqlalchemy.sql import Select, Insert, Update, Delete
from sqlalchemy.sql.elements import BinaryExpression, ColumnElement
from sqlalchemy.sql.operators import ColumnOperators
from sqlalchemy.dialects import postgresql, mysql, sqlite

from ..models.database import Base
from ..models.api import SortOrder, FilterParams, PaginationParams

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T', bound=Base)


class ComparisonOperator(str, Enum):
    """比较操作符枚举"""
    EQ = "eq"          # 等于
    NE = "ne"          # 不等于
    LT = "lt"          # 小于
    LE = "le"          # 小于等于
    GT = "gt"          # 大于
    GE = "ge"          # 大于等于
    IN = "in"          # 包含
    NOT_IN = "not_in"  # 不包含
    LIKE = "like"      # 模糊匹配
    ILIKE = "ilike"    # 忽略大小写模糊匹配
    IS_NULL = "is_null"        # 为空
    IS_NOT_NULL = "is_not_null" # 不为空
    BETWEEN = "between"         # 范围
    CONTAINS = "contains"       # 包含（数组）
    OVERLAPS = "overlaps"       # 重叠（数组）


class LogicalOperator(str, Enum):
    """逻辑操作符枚举"""
    AND = "and"
    OR = "or"
    NOT = "not"


class JoinType(str, Enum):
    """连接类型枚举"""
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"


class FilterCondition:
    """过滤条件类"""
    
    def __init__(
        self,
        field: str,
        operator: ComparisonOperator,
        value: Any,
        table_alias: Optional[str] = None
    ):
        self.field = field
        self.operator = operator
        self.value = value
        self.table_alias = table_alias
    
    def __repr__(self) -> str:
        alias_prefix = f"{self.table_alias}." if self.table_alias else ""
        return f"FilterCondition({alias_prefix}{self.field} {self.operator} {self.value})"


class QueryFilter:
    """查询过滤器类"""
    
    def __init__(self, logical_op: LogicalOperator = LogicalOperator.AND):
        self.conditions: List[Union[FilterCondition, 'QueryFilter']] = []
        self.logical_op = logical_op
    
    def add_condition(
        self, 
        field: str, 
        operator: ComparisonOperator, 
        value: Any,
        table_alias: Optional[str] = None
    ) -> 'QueryFilter':
        """添加过滤条件"""
        condition = FilterCondition(field, operator, value, table_alias)
        self.conditions.append(condition)
        return self
    
    def add_filter(self, filter_obj: 'QueryFilter') -> 'QueryFilter':
        """添加子过滤器"""
        self.conditions.append(filter_obj)
        return self
    
    def eq(self, field: str, value: Any, table_alias: Optional[str] = None) -> 'QueryFilter':
        """等于条件"""
        return self.add_condition(field, ComparisonOperator.EQ, value, table_alias)
    
    def ne(self, field: str, value: Any, table_alias: Optional[str] = None) -> 'QueryFilter':
        """不等于条件"""
        return self.add_condition(field, ComparisonOperator.NE, value, table_alias)
    
    def lt(self, field: str, value: Any, table_alias: Optional[str] = None) -> 'QueryFilter':
        """小于条件"""
        return self.add_condition(field, ComparisonOperator.LT, value, table_alias)
    
    def le(self, field: str, value: Any, table_alias: Optional[str] = None) -> 'QueryFilter':
        """小于等于条件"""
        return self.add_condition(field, ComparisonOperator.LE, value, table_alias)
    
    def gt(self, field: str, value: Any, table_alias: Optional[str] = None) -> 'QueryFilter':
        """大于条件"""
        return self.add_condition(field, ComparisonOperator.GT, value, table_alias)
    
    def ge(self, field: str, value: Any, table_alias: Optional[str] = None) -> 'QueryFilter':
        """大于等于条件"""
        return self.add_condition(field, ComparisonOperator.GE, value, table_alias)
    
    def in_(self, field: str, values: List[Any], table_alias: Optional[str] = None) -> 'QueryFilter':
        """包含条件"""
        return self.add_condition(field, ComparisonOperator.IN, values, table_alias)
    
    def not_in(self, field: str, values: List[Any], table_alias: Optional[str] = None) -> 'QueryFilter':
        """不包含条件"""
        return self.add_condition(field, ComparisonOperator.NOT_IN, values, table_alias)
    
    def like(self, field: str, pattern: str, table_alias: Optional[str] = None) -> 'QueryFilter':
        """模糊匹配条件"""
        return self.add_condition(field, ComparisonOperator.LIKE, pattern, table_alias)
    
    def ilike(self, field: str, pattern: str, table_alias: Optional[str] = None) -> 'QueryFilter':
        """忽略大小写模糊匹配条件"""
        return self.add_condition(field, ComparisonOperator.ILIKE, pattern, table_alias)
    
    def is_null(self, field: str, table_alias: Optional[str] = None) -> 'QueryFilter':
        """为空条件"""
        return self.add_condition(field, ComparisonOperator.IS_NULL, None, table_alias)
    
    def is_not_null(self, field: str, table_alias: Optional[str] = None) -> 'QueryFilter':
        """不为空条件"""
        return self.add_condition(field, ComparisonOperator.IS_NOT_NULL, None, table_alias)
    
    def between(self, field: str, start: Any, end: Any, table_alias: Optional[str] = None) -> 'QueryFilter':
        """范围条件"""
        return self.add_condition(field, ComparisonOperator.BETWEEN, (start, end), table_alias)
    
    def __repr__(self) -> str:
        return f"QueryFilter({self.logical_op}, {len(self.conditions)} conditions)"


class QueryBuilder(Generic[T]):
    """查询构建器类"""
    
    def __init__(self, model: Type[T], session: Optional[Session] = None):
        self.model = model
        self.session = session
        self._query = select(model)
        self._joins: List[Tuple[Any, str, Optional[str]]] = []
        self._filters: List[QueryFilter] = []
        self._order_by: List[Tuple[str, SortOrder]] = []
        self._group_by: List[str] = []
        self._having: List[QueryFilter] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._distinct: bool = False
        self._load_options: List[Any] = []
    
    def filter(self, filter_obj: QueryFilter) -> 'QueryBuilder[T]':
        """添加过滤条件"""
        self._filters.append(filter_obj)
        return self
    
    def filter_by(self, **kwargs) -> 'QueryBuilder[T]':
        """通过关键字参数添加过滤条件"""
        filter_obj = QueryFilter()
        for field, value in kwargs.items():
            filter_obj.eq(field, value)
        return self.filter(filter_obj)
    
    def join(
        self, 
        target, 
        join_type: JoinType = JoinType.INNER,
        on_condition: Optional[str] = None
    ) -> 'QueryBuilder[T]':
        """添加连接"""
        self._joins.append((target, join_type, on_condition))
        return self
    
    def left_join(self, target, on_condition: Optional[str] = None) -> 'QueryBuilder[T]':
        """左连接"""
        return self.join(target, JoinType.LEFT, on_condition)
    
    def inner_join(self, target, on_condition: Optional[str] = None) -> 'QueryBuilder[T]':
        """内连接"""
        return self.join(target, JoinType.INNER, on_condition)
    
    def order_by(self, field: str, order: SortOrder = SortOrder.ASC) -> 'QueryBuilder[T]':
        """添加排序"""
        self._order_by.append((field, order))
        return self
    
    def group_by(self, *fields: str) -> 'QueryBuilder[T]':
        """添加分组"""
        self._group_by.extend(fields)
        return self
    
    def having(self, filter_obj: QueryFilter) -> 'QueryBuilder[T]':
        """添加HAVING条件"""
        self._having.append(filter_obj)
        return self
    
    def limit(self, count: int) -> 'QueryBuilder[T]':
        """设置限制数量"""
        self._limit = count
        return self
    
    def offset(self, count: int) -> 'QueryBuilder[T]':
        """设置偏移量"""
        self._offset = count
        return self
    
    def paginate(self, page: int, per_page: int) -> 'QueryBuilder[T]':
        """分页"""
        self._offset = (page - 1) * per_page
        self._limit = per_page
        return self
    
    def distinct(self) -> 'QueryBuilder[T]':
        """去重"""
        self._distinct = True
        return self
    
    def options(self, *load_options) -> 'QueryBuilder[T]':
        """添加加载选项"""
        self._load_options.extend(load_options)
        return self
    
    def eager_load(self, *relationships) -> 'QueryBuilder[T]':
        """预加载关系"""
        for rel in relationships:
            self._load_options.append(joinedload(rel))
        return self
    
    def select_in_load(self, *relationships) -> 'QueryBuilder[T]':
        """使用selectin加载关系"""
        for rel in relationships:
            self._load_options.append(selectinload(rel))
        return self
    
    def _build_filter_expression(self, filter_obj: QueryFilter, model_class: Type[T]):
        """构建过滤表达式"""
        if not filter_obj.conditions:
            return None
        
        expressions = []
        
        for condition in filter_obj.conditions:
            if isinstance(condition, QueryFilter):
                # 递归处理子过滤器
                sub_expr = self._build_filter_expression(condition, model_class)
                if sub_expr is not None:
                    expressions.append(sub_expr)
            else:
                # 处理单个条件
                expr = self._build_condition_expression(condition, model_class)
                if expr is not None:
                    expressions.append(expr)
        
        if not expressions:
            return None
        
        # 根据逻辑操作符组合表达式
        if filter_obj.logical_op == LogicalOperator.AND:
            return and_(*expressions)
        elif filter_obj.logical_op == LogicalOperator.OR:
            return or_(*expressions)
        elif filter_obj.logical_op == LogicalOperator.NOT:
            return not_(and_(*expressions))
        else:
            return and_(*expressions)
    
    def _build_condition_expression(self, condition: FilterCondition, model_class: Type[T]):
        """构建单个条件表达式"""
        try:
            # 获取字段属性
            if condition.table_alias:
                # TODO: 处理表别名的情况
                field_attr = getattr(model_class, condition.field)
            else:
                field_attr = getattr(model_class, condition.field)
            
            # 根据操作符构建表达式
            if condition.operator == ComparisonOperator.EQ:
                return field_attr == condition.value
            elif condition.operator == ComparisonOperator.NE:
                return field_attr != condition.value
            elif condition.operator == ComparisonOperator.LT:
                return field_attr < condition.value
            elif condition.operator == ComparisonOperator.LE:
                return field_attr <= condition.value
            elif condition.operator == ComparisonOperator.GT:
                return field_attr > condition.value
            elif condition.operator == ComparisonOperator.GE:
                return field_attr >= condition.value
            elif condition.operator == ComparisonOperator.IN:
                return field_attr.in_(condition.value)
            elif condition.operator == ComparisonOperator.NOT_IN:
                return ~field_attr.in_(condition.value)
            elif condition.operator == ComparisonOperator.LIKE:
                return field_attr.like(condition.value)
            elif condition.operator == ComparisonOperator.ILIKE:
                return field_attr.ilike(condition.value)
            elif condition.operator == ComparisonOperator.IS_NULL:
                return field_attr.is_(None)
            elif condition.operator == ComparisonOperator.IS_NOT_NULL:
                return field_attr.is_not(None)
            elif condition.operator == ComparisonOperator.BETWEEN:
                start, end = condition.value
                return field_attr.between(start, end)
            elif condition.operator == ComparisonOperator.CONTAINS:
                return field_attr.contains(condition.value)
            elif condition.operator == ComparisonOperator.OVERLAPS:
                return field_attr.overlaps(condition.value)
            else:
                logger.warning(f"Unsupported operator: {condition.operator}")
                return None
                
        except AttributeError as e:
            logger.error(f"Field '{condition.field}' not found in model {model_class.__name__}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error building condition expression: {e}")
            return None
    
    def build(self) -> Select:
        """构建查询"""
        query = self._query
        
        # 应用去重
        if self._distinct:
            query = query.distinct()
        
        # 应用连接
        for target, join_type, on_condition in self._joins:
            if join_type == JoinType.INNER:
                query = query.join(target)
            elif join_type == JoinType.LEFT:
                query = query.outerjoin(target)
            # TODO: 支持其他连接类型
        
        # 应用过滤条件
        for filter_obj in self._filters:
            filter_expr = self._build_filter_expression(filter_obj, self.model)
            if filter_expr is not None:
                query = query.where(filter_expr)
        
        # 应用分组
        if self._group_by:
            group_columns = []
            for field in self._group_by:
                try:
                    group_columns.append(getattr(self.model, field))
                except AttributeError:
                    logger.warning(f"Group by field '{field}' not found")
            if group_columns:
                query = query.group_by(*group_columns)
        
        # 应用HAVING条件
        for having_filter in self._having:
            having_expr = self._build_filter_expression(having_filter, self.model)
            if having_expr is not None:
                query = query.having(having_expr)
        
        # 应用排序
        if self._order_by:
            order_columns = []
            for field, order in self._order_by:
                try:
                    column = getattr(self.model, field)
                    if order == SortOrder.DESC:
                        order_columns.append(desc(column))
                    else:
                        order_columns.append(asc(column))
                except AttributeError:
                    logger.warning(f"Order by field '{field}' not found")
            if order_columns:
                query = query.order_by(*order_columns)
        
        # 应用限制和偏移
        if self._offset is not None:
            query = query.offset(self._offset)
        if self._limit is not None:
            query = query.limit(self._limit)
        
        return query
    
    def count(self) -> int:
        """获取记录数量"""
        if not self.session:
            raise ValueError("Session is required for count operation")
        
        # 构建计数查询
        count_query = select(func.count()).select_from(self.model)
        
        # 应用过滤条件
        for filter_obj in self._filters:
            filter_expr = self._build_filter_expression(filter_obj, self.model)
            if filter_expr is not None:
                count_query = count_query.where(filter_expr)
        
        return self.session.scalar(count_query) or 0
    
    def all(self) -> List[T]:
        """获取所有记录"""
        if not self.session:
            raise ValueError("Session is required for query execution")
        
        query = self.build()
        
        # 应用加载选项
        if self._load_options:
            query = query.options(*self._load_options)
        
        return list(self.session.scalars(query).all())
    
    def first(self) -> Optional[T]:
        """获取第一条记录"""
        if not self.session:
            raise ValueError("Session is required for query execution")
        
        query = self.build()
        
        # 应用加载选项
        if self._load_options:
            query = query.options(*self._load_options)
        
        return self.session.scalars(query).first()
    
    def one(self) -> T:
        """获取唯一记录"""
        if not self.session:
            raise ValueError("Session is required for query execution")
        
        query = self.build()
        
        # 应用加载选项
        if self._load_options:
            query = query.options(*self._load_options)
        
        return self.session.scalars(query).one()
    
    def one_or_none(self) -> Optional[T]:
        """获取唯一记录或None"""
        if not self.session:
            raise ValueError("Session is required for query execution")
        
        query = self.build()
        
        # 应用加载选项
        if self._load_options:
            query = query.options(*self._load_options)
        
        return self.session.scalars(query).one_or_none()
    
    def paginate_result(
        self, 
        page: int, 
        per_page: int
    ) -> Dict[str, Any]:
        """分页查询结果"""
        if not self.session:
            raise ValueError("Session is required for pagination")
        
        # 获取总数
        total = self.count()
        
        # 应用分页
        self.paginate(page, per_page)
        items = self.all()
        
        # 计算分页信息
        total_pages = (total + per_page - 1) // per_page
        has_prev = page > 1
        has_next = page < total_pages
        
        return {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "has_prev": has_prev,
            "has_next": has_next,
            "prev_page": page - 1 if has_prev else None,
            "next_page": page + 1 if has_next else None
        }


def create_query_builder(model: Type[T], session: Optional[Session] = None) -> QueryBuilder[T]:
    """创建查询构建器"""
    return QueryBuilder(model, session)


def create_filter(logical_op: LogicalOperator = LogicalOperator.AND) -> QueryFilter:
    """创建查询过滤器"""
    return QueryFilter(logical_op)


# 便捷函数
def and_filter() -> QueryFilter:
    """创建AND过滤器"""
    return QueryFilter(LogicalOperator.AND)


def or_filter() -> QueryFilter:
    """创建OR过滤器"""
    return QueryFilter(LogicalOperator.OR)


def not_filter() -> QueryFilter:
    """创建NOT过滤器"""
    return QueryFilter(LogicalOperator.NOT)