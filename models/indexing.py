"""模型索引系统模块

本模块提供数据库索引管理、性能优化和查询分析功能。
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
import re
import hashlib
import json
from collections import defaultdict, OrderedDict, Counter
import threading
import asyncio
from contextlib import contextmanager

# SQLAlchemy imports
try:
    from sqlalchemy import (
        create_engine, MetaData, Table, Column, Integer, String, 
        DateTime, Text, Boolean, Index, text, inspect, event
    )
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.engine import Engine
    from sqlalchemy.schema import CreateIndex, DropIndex
    from sqlalchemy.sql import Select
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    create_engine = MetaData = Table = Column = None
    Session = sessionmaker = Engine = Index = None

# 导入项目模块
from .events import EventType, emit_business_event
from .validation import validate_data


logger = logging.getLogger(__name__)


T = TypeVar('T')


class IndexType(Enum):
    """索引类型枚举"""
    BTREE = "btree"                        # B树索引
    HASH = "hash"                          # 哈希索引
    GIN = "gin"                            # GIN索引（PostgreSQL）
    GIST = "gist"                          # GiST索引（PostgreSQL）
    SPGIST = "spgist"                      # SP-GiST索引（PostgreSQL）
    BRIN = "brin"                          # BRIN索引（PostgreSQL）
    FULLTEXT = "fulltext"                  # 全文索引
    SPATIAL = "spatial"                    # 空间索引
    PARTIAL = "partial"                    # 部分索引
    EXPRESSION = "expression"              # 表达式索引
    COMPOSITE = "composite"                # 复合索引
    COVERING = "covering"                  # 覆盖索引


class IndexStatus(Enum):
    """索引状态枚举"""
    ACTIVE = "active"                      # 活跃
    INACTIVE = "inactive"                  # 非活跃
    BUILDING = "building"                  # 构建中
    INVALID = "invalid"                    # 无效
    DUPLICATE = "duplicate"                # 重复
    UNUSED = "unused"                      # 未使用
    REDUNDANT = "redundant"                # 冗余


class IndexUsageLevel(Enum):
    """索引使用级别枚举"""
    HIGH = "high"                          # 高使用率
    MEDIUM = "medium"                      # 中等使用率
    LOW = "low"                            # 低使用率
    NEVER = "never"                        # 从未使用


class QueryPattern(Enum):
    """查询模式枚举"""
    EQUALITY = "equality"                  # 等值查询
    RANGE = "range"                        # 范围查询
    LIKE = "like"                          # 模糊查询
    IN = "in"                              # IN查询
    JOIN = "join"                          # 连接查询
    ORDER_BY = "order_by"                  # 排序查询
    GROUP_BY = "group_by"                  # 分组查询
    AGGREGATE = "aggregate"                # 聚合查询


@dataclass
class IndexColumn:
    """索引列"""
    name: str                              # 列名
    order: str = "ASC"                     # 排序方向（ASC/DESC）
    length: Optional[int] = None           # 索引长度（部分索引）
    expression: Optional[str] = None       # 表达式（表达式索引）
    
    def __str__(self) -> str:
        if self.expression:
            return f"({self.expression})"
        
        result = self.name
        if self.length:
            result += f"({self.length})"
        if self.order != "ASC":
            result += f" {self.order}"
        
        return result


@dataclass
class IndexDefinition:
    """索引定义"""
    name: str                              # 索引名称
    table_name: str                        # 表名
    columns: List[IndexColumn]             # 索引列
    
    # 索引属性
    index_type: IndexType = IndexType.BTREE
    unique: bool = False                   # 是否唯一索引
    partial: bool = False                  # 是否部分索引
    where_clause: Optional[str] = None     # WHERE条件（部分索引）
    
    # 元数据
    description: str = ""                  # 描述
    tags: List[str] = field(default_factory=list)  # 标签
    
    # 时间信息
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def column_names(self) -> List[str]:
        """获取列名列表"""
        return [col.name for col in self.columns if not col.expression]
    
    @property
    def is_composite(self) -> bool:
        """是否为复合索引"""
        return len(self.columns) > 1
    
    @property
    def is_expression(self) -> bool:
        """是否为表达式索引"""
        return any(col.expression for col in self.columns)
    
    def to_sql(self, dialect: str = "postgresql") -> str:
        """生成创建索引的SQL"""
        # 构建列定义
        column_defs = []
        for col in self.columns:
            if col.expression:
                column_defs.append(f"({col.expression})")
            else:
                col_def = col.name
                if col.length and dialect in ["mysql"]:
                    col_def += f"({col.length})"
                if col.order != "ASC":
                    col_def += f" {col.order}"
                column_defs.append(col_def)
        
        columns_str = ", ".join(column_defs)
        
        # 构建SQL
        sql_parts = ["CREATE"]
        
        if self.unique:
            sql_parts.append("UNIQUE")
        
        sql_parts.append("INDEX")
        sql_parts.append(self.name)
        sql_parts.append("ON")
        sql_parts.append(self.table_name)
        
        # 添加索引类型
        if self.index_type != IndexType.BTREE and dialect == "postgresql":
            sql_parts.append(f"USING {self.index_type.value.upper()}")
        
        sql_parts.append(f"({columns_str})")
        
        # 添加WHERE条件
        if self.where_clause:
            sql_parts.append(f"WHERE {self.where_clause}")
        
        return " ".join(sql_parts) + ";"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'table_name': self.table_name,
            'columns': [{
                'name': col.name,
                'order': col.order,
                'length': col.length,
                'expression': col.expression
            } for col in self.columns],
            'index_type': self.index_type.value,
            'unique': self.unique,
            'partial': self.partial,
            'where_clause': self.where_clause,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class IndexStatistics:
    """索引统计信息"""
    name: str                              # 索引名称
    table_name: str                        # 表名
    
    # 大小信息
    size_bytes: int = 0                    # 索引大小（字节）
    size_mb: float = 0.0                   # 索引大小（MB）
    
    # 使用统计
    scans: int = 0                         # 扫描次数
    tuples_read: int = 0                   # 读取元组数
    tuples_fetched: int = 0                # 获取元组数
    
    # 性能指标
    selectivity: float = 0.0               # 选择性
    cardinality: int = 0                   # 基数
    null_frac: float = 0.0                 # 空值比例
    
    # 时间信息
    last_used: Optional[datetime] = None   # 最后使用时间
    last_analyzed: Optional[datetime] = None  # 最后分析时间
    
    # 状态
    status: IndexStatus = IndexStatus.ACTIVE
    usage_level: IndexUsageLevel = IndexUsageLevel.MEDIUM
    
    @property
    def usage_ratio(self) -> float:
        """使用率"""
        if self.tuples_read == 0:
            return 0.0
        return self.tuples_fetched / self.tuples_read
    
    @property
    def efficiency(self) -> float:
        """效率分数（0-100）"""
        score = 0.0
        
        # 使用频率权重（40%）
        if self.scans > 1000:
            score += 40
        elif self.scans > 100:
            score += 30
        elif self.scans > 10:
            score += 20
        elif self.scans > 0:
            score += 10
        
        # 选择性权重（30%）
        if self.selectivity > 0.9:
            score += 30
        elif self.selectivity > 0.7:
            score += 25
        elif self.selectivity > 0.5:
            score += 20
        elif self.selectivity > 0.3:
            score += 15
        elif self.selectivity > 0.1:
            score += 10
        
        # 使用率权重（20%）
        usage_ratio = self.usage_ratio
        if usage_ratio > 0.8:
            score += 20
        elif usage_ratio > 0.6:
            score += 15
        elif usage_ratio > 0.4:
            score += 10
        elif usage_ratio > 0.2:
            score += 5
        
        # 大小权重（10%）
        if self.size_mb < 10:
            score += 10
        elif self.size_mb < 50:
            score += 8
        elif self.size_mb < 100:
            score += 6
        elif self.size_mb < 500:
            score += 4
        elif self.size_mb < 1000:
            score += 2
        
        return min(score, 100.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'table_name': self.table_name,
            'size_bytes': self.size_bytes,
            'size_mb': self.size_mb,
            'scans': self.scans,
            'tuples_read': self.tuples_read,
            'tuples_fetched': self.tuples_fetched,
            'selectivity': self.selectivity,
            'cardinality': self.cardinality,
            'null_frac': self.null_frac,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'last_analyzed': self.last_analyzed.isoformat() if self.last_analyzed else None,
            'status': self.status.value,
            'usage_level': self.usage_level.value,
            'usage_ratio': self.usage_ratio,
            'efficiency': self.efficiency
        }


@dataclass
class QueryAnalysis:
    """查询分析"""
    sql: str                               # SQL语句
    table_name: str                        # 主表名
    
    # 查询模式
    patterns: List[QueryPattern] = field(default_factory=list)
    
    # 使用的列
    where_columns: List[str] = field(default_factory=list)  # WHERE条件列
    order_columns: List[str] = field(default_factory=list)  # ORDER BY列
    group_columns: List[str] = field(default_factory=list)  # GROUP BY列
    join_columns: List[str] = field(default_factory=list)   # JOIN列
    
    # 性能指标
    execution_time_ms: float = 0.0         # 执行时间
    rows_examined: int = 0                 # 检查行数
    rows_returned: int = 0                 # 返回行数
    
    # 索引使用
    used_indexes: List[str] = field(default_factory=list)   # 使用的索引
    missing_indexes: List[str] = field(default_factory=list)  # 缺失的索引
    
    # 时间信息
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def selectivity(self) -> float:
        """查询选择性"""
        if self.rows_examined == 0:
            return 0.0
        return self.rows_returned / self.rows_examined
    
    @property
    def is_slow_query(self) -> bool:
        """是否为慢查询"""
        return self.execution_time_ms > 1000  # 超过1秒
    
    @property
    def needs_index(self) -> bool:
        """是否需要索引"""
        return (
            self.is_slow_query or 
            self.selectivity < 0.1 or 
            len(self.missing_indexes) > 0
        )


@dataclass
class IndexRecommendation:
    """索引推荐"""
    table_name: str                        # 表名
    columns: List[str]                     # 推荐的列
    index_type: IndexType = IndexType.BTREE
    
    # 推荐原因
    reason: str = ""                       # 推荐原因
    query_patterns: List[QueryPattern] = field(default_factory=list)
    
    # 预期收益
    expected_improvement: float = 0.0      # 预期性能提升（%）
    estimated_size_mb: float = 0.0         # 预估大小
    
    # 优先级
    priority: int = 1                      # 优先级（1-10）
    confidence: float = 0.0                # 置信度（0-1）
    
    # 时间信息
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def index_name(self) -> str:
        """生成索引名称"""
        column_str = "_".join(self.columns)
        return f"idx_{self.table_name}_{column_str}"
    
    def to_index_definition(self) -> IndexDefinition:
        """转换为索引定义"""
        index_columns = [IndexColumn(name=col) for col in self.columns]
        
        return IndexDefinition(
            name=self.index_name,
            table_name=self.table_name,
            columns=index_columns,
            index_type=self.index_type,
            description=self.reason
        )


@dataclass
class IndexConfig:
    """索引配置"""
    # 分析配置
    enable_analysis: bool = True           # 启用分析
    analysis_interval_hours: int = 24      # 分析间隔（小时）
    
    # 推荐配置
    enable_recommendations: bool = True    # 启用推荐
    min_query_count: int = 10              # 最小查询次数
    min_execution_time_ms: float = 100.0   # 最小执行时间
    
    # 清理配置
    enable_cleanup: bool = False           # 启用清理
    unused_threshold_days: int = 30        # 未使用阈值（天）
    low_usage_threshold: int = 10          # 低使用阈值
    
    # 性能配置
    max_index_size_mb: float = 1000.0      # 最大索引大小
    max_indexes_per_table: int = 20        # 每表最大索引数
    
    # 数据库配置
    database_url: Optional[str] = None     # 数据库连接URL
    schema_name: str = "public"            # 模式名


class IndexError(Exception):
    """索引错误"""
    pass


class IndexAnalysisError(IndexError):
    """索引分析错误"""
    pass


class IndexCreationError(IndexError):
    """索引创建错误"""
    pass


class IndexStatisticsCollector:
    """索引统计信息收集器"""
    
    def __init__(self, engine: Engine, config: IndexConfig):
        self.engine = engine
        self.config = config
    
    def collect_index_statistics(self, table_name: Optional[str] = None) -> List[IndexStatistics]:
        """收集索引统计信息"""
        with Session(self.engine) as session:
            try:
                # PostgreSQL查询
                if self.engine.dialect.name == 'postgresql':
                    return self._collect_postgresql_stats(session, table_name)
                # MySQL查询
                elif self.engine.dialect.name == 'mysql':
                    return self._collect_mysql_stats(session, table_name)
                # SQLite查询
                elif self.engine.dialect.name == 'sqlite':
                    return self._collect_sqlite_stats(session, table_name)
                else:
                    logger.warning(f"Unsupported database dialect: {self.engine.dialect.name}")
                    return []
                    
            except Exception as e:
                logger.error(f"Failed to collect index statistics: {e}")
                raise IndexAnalysisError(f"Failed to collect index statistics: {e}")
    
    def _collect_postgresql_stats(self, session: Session, table_name: Optional[str]) -> List[IndexStatistics]:
        """收集PostgreSQL索引统计"""
        sql = """
        SELECT 
            schemaname,
            tablename,
            indexname,
            pg_size_pretty(pg_relation_size(indexrelid)) as size,
            pg_relation_size(indexrelid) as size_bytes,
            idx_scan,
            idx_tup_read,
            idx_tup_fetch,
            n_distinct,
            null_frac,
            last_analyze
        FROM pg_stat_user_indexes 
        JOIN pg_stats ON pg_stats.tablename = pg_stat_user_indexes.relname 
            AND pg_stats.attname = pg_get_indexdef(indexrelid)
        WHERE schemaname = :schema
        """
        
        if table_name:
            sql += " AND tablename = :table_name"
        
        params = {'schema': self.config.schema_name}
        if table_name:
            params['table_name'] = table_name
        
        result = session.execute(text(sql), params)
        
        statistics = []
        for row in result:
            stats = IndexStatistics(
                name=row.indexname,
                table_name=row.tablename,
                size_bytes=row.size_bytes or 0,
                size_mb=(row.size_bytes or 0) / (1024 * 1024),
                scans=row.idx_scan or 0,
                tuples_read=row.idx_tup_read or 0,
                tuples_fetched=row.idx_tup_fetch or 0,
                cardinality=row.n_distinct or 0,
                null_frac=row.null_frac or 0.0,
                last_analyzed=row.last_analyze
            )
            
            # 计算使用级别
            stats.usage_level = self._calculate_usage_level(stats)
            statistics.append(stats)
        
        return statistics
    
    def _collect_mysql_stats(self, session: Session, table_name: Optional[str]) -> List[IndexStatistics]:
        """收集MySQL索引统计"""
        sql = """
        SELECT 
            TABLE_NAME,
            INDEX_NAME,
            CARDINALITY,
            INDEX_LENGTH
        FROM information_schema.STATISTICS 
        WHERE TABLE_SCHEMA = DATABASE()
        """
        
        if table_name:
            sql += " AND TABLE_NAME = :table_name"
        
        params = {}
        if table_name:
            params['table_name'] = table_name
        
        result = session.execute(text(sql), params)
        
        statistics = []
        for row in result:
            stats = IndexStatistics(
                name=row.INDEX_NAME,
                table_name=row.TABLE_NAME,
                size_bytes=row.INDEX_LENGTH or 0,
                size_mb=(row.INDEX_LENGTH or 0) / (1024 * 1024),
                cardinality=row.CARDINALITY or 0
            )
            
            stats.usage_level = self._calculate_usage_level(stats)
            statistics.append(stats)
        
        return statistics
    
    def _collect_sqlite_stats(self, session: Session, table_name: Optional[str]) -> List[IndexStatistics]:
        """收集SQLite索引统计"""
        # SQLite的统计信息有限
        sql = "SELECT name, tbl_name FROM sqlite_master WHERE type = 'index'"
        
        if table_name:
            sql += " AND tbl_name = :table_name"
        
        params = {}
        if table_name:
            params['table_name'] = table_name
        
        result = session.execute(text(sql), params)
        
        statistics = []
        for row in result:
            stats = IndexStatistics(
                name=row.name,
                table_name=row.tbl_name
            )
            
            stats.usage_level = IndexUsageLevel.MEDIUM  # SQLite无法获取详细统计
            statistics.append(stats)
        
        return statistics
    
    def _calculate_usage_level(self, stats: IndexStatistics) -> IndexUsageLevel:
        """计算使用级别"""
        if stats.scans == 0:
            return IndexUsageLevel.NEVER
        elif stats.scans < self.config.low_usage_threshold:
            return IndexUsageLevel.LOW
        elif stats.scans < 100:
            return IndexUsageLevel.MEDIUM
        else:
            return IndexUsageLevel.HIGH


class QueryAnalyzer:
    """查询分析器"""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self._query_cache: Dict[str, QueryAnalysis] = {}
    
    def analyze_query(self, sql: str, execution_time_ms: float = 0.0,
                     rows_examined: int = 0, rows_returned: int = 0) -> QueryAnalysis:
        """分析查询"""
        # 生成查询哈希
        query_hash = hashlib.md5(sql.encode()).hexdigest()
        
        # 检查缓存
        if query_hash in self._query_cache:
            cached = self._query_cache[query_hash]
            # 更新性能指标
            cached.execution_time_ms = max(cached.execution_time_ms, execution_time_ms)
            cached.rows_examined = max(cached.rows_examined, rows_examined)
            cached.rows_returned = max(cached.rows_returned, rows_returned)
            return cached
        
        # 解析SQL
        analysis = self._parse_sql(sql)
        analysis.execution_time_ms = execution_time_ms
        analysis.rows_examined = rows_examined
        analysis.rows_returned = rows_returned
        
        # 缓存结果
        self._query_cache[query_hash] = analysis
        
        return analysis
    
    def _parse_sql(self, sql: str) -> QueryAnalysis:
        """解析SQL语句"""
        sql_lower = sql.lower().strip()
        
        # 提取表名
        table_name = self._extract_table_name(sql_lower)
        
        analysis = QueryAnalysis(
            sql=sql,
            table_name=table_name
        )
        
        # 分析查询模式
        analysis.patterns = self._identify_patterns(sql_lower)
        
        # 提取列信息
        analysis.where_columns = self._extract_where_columns(sql_lower)
        analysis.order_columns = self._extract_order_columns(sql_lower)
        analysis.group_columns = self._extract_group_columns(sql_lower)
        analysis.join_columns = self._extract_join_columns(sql_lower)
        
        return analysis
    
    def _extract_table_name(self, sql: str) -> str:
        """提取表名"""
        # 简单的表名提取（可以改进）
        patterns = [
            r'from\s+([\w_]+)',
            r'update\s+([\w_]+)',
            r'insert\s+into\s+([\w_]+)',
            r'delete\s+from\s+([\w_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sql)
            if match:
                return match.group(1)
        
        return "unknown"
    
    def _identify_patterns(self, sql: str) -> List[QueryPattern]:
        """识别查询模式"""
        patterns = []
        
        if 'where' in sql:
            if '=' in sql:
                patterns.append(QueryPattern.EQUALITY)
            if 'between' in sql or '>' in sql or '<' in sql:
                patterns.append(QueryPattern.RANGE)
            if 'like' in sql or 'ilike' in sql:
                patterns.append(QueryPattern.LIKE)
            if ' in (' in sql:
                patterns.append(QueryPattern.IN)
        
        if 'join' in sql:
            patterns.append(QueryPattern.JOIN)
        
        if 'order by' in sql:
            patterns.append(QueryPattern.ORDER_BY)
        
        if 'group by' in sql:
            patterns.append(QueryPattern.GROUP_BY)
        
        if any(func in sql for func in ['count(', 'sum(', 'avg(', 'max(', 'min(']):
            patterns.append(QueryPattern.AGGREGATE)
        
        return patterns
    
    def _extract_where_columns(self, sql: str) -> List[str]:
        """提取WHERE条件中的列"""
        columns = []
        
        # 查找WHERE子句
        where_match = re.search(r'where\s+(.+?)(?:\s+order\s+by|\s+group\s+by|\s+limit|$)', sql)
        if where_match:
            where_clause = where_match.group(1)
            
            # 提取列名（简单实现）
            column_pattern = r'([\w_]+)\s*[=<>!]'
            matches = re.findall(column_pattern, where_clause)
            columns.extend(matches)
        
        return list(set(columns))
    
    def _extract_order_columns(self, sql: str) -> List[str]:
        """提取ORDER BY中的列"""
        columns = []
        
        order_match = re.search(r'order\s+by\s+(.+?)(?:\s+limit|$)', sql)
        if order_match:
            order_clause = order_match.group(1)
            
            # 提取列名
            column_pattern = r'([\w_]+)'
            matches = re.findall(column_pattern, order_clause)
            columns.extend(matches)
        
        return list(set(columns))
    
    def _extract_group_columns(self, sql: str) -> List[str]:
        """提取GROUP BY中的列"""
        columns = []
        
        group_match = re.search(r'group\s+by\s+(.+?)(?:\s+order\s+by|\s+limit|$)', sql)
        if group_match:
            group_clause = group_match.group(1)
            
            # 提取列名
            column_pattern = r'([\w_]+)'
            matches = re.findall(column_pattern, group_clause)
            columns.extend(matches)
        
        return list(set(columns))
    
    def _extract_join_columns(self, sql: str) -> List[str]:
        """提取JOIN中的列"""
        columns = []
        
        # 查找JOIN条件
        join_pattern = r'join\s+[\w_]+\s+on\s+([\w_.]+)\s*=\s*([\w_.]+)'
        matches = re.findall(join_pattern, sql)
        
        for left, right in matches:
            # 提取列名（去掉表前缀）
            left_col = left.split('.')[-1] if '.' in left else left
            right_col = right.split('.')[-1] if '.' in right else right
            columns.extend([left_col, right_col])
        
        return list(set(columns))


class IndexRecommendationEngine:
    """索引推荐引擎"""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self._query_history: List[QueryAnalysis] = []
    
    def add_query_analysis(self, analysis: QueryAnalysis) -> None:
        """添加查询分析"""
        self._query_history.append(analysis)
        
        # 限制历史记录大小
        if len(self._query_history) > 10000:
            self._query_history = self._query_history[-5000:]
    
    def generate_recommendations(self, table_name: Optional[str] = None) -> List[IndexRecommendation]:
        """生成索引推荐"""
        recommendations = []
        
        # 按表分组查询
        table_queries = defaultdict(list)
        for analysis in self._query_history:
            if not table_name or analysis.table_name == table_name:
                table_queries[analysis.table_name].append(analysis)
        
        # 为每个表生成推荐
        for table, queries in table_queries.items():
            table_recommendations = self._generate_table_recommendations(table, queries)
            recommendations.extend(table_recommendations)
        
        # 按优先级排序
        recommendations.sort(key=lambda r: (r.priority, r.confidence), reverse=True)
        
        return recommendations
    
    def _generate_table_recommendations(self, table_name: str, queries: List[QueryAnalysis]) -> List[IndexRecommendation]:
        """为表生成推荐"""
        recommendations = []
        
        # 分析列使用频率
        column_usage = self._analyze_column_usage(queries)
        
        # 生成单列索引推荐
        single_column_recs = self._recommend_single_column_indexes(table_name, column_usage, queries)
        recommendations.extend(single_column_recs)
        
        # 生成复合索引推荐
        composite_recs = self._recommend_composite_indexes(table_name, queries)
        recommendations.extend(composite_recs)
        
        # 生成特殊索引推荐
        special_recs = self._recommend_special_indexes(table_name, queries)
        recommendations.extend(special_recs)
        
        return recommendations
    
    def _analyze_column_usage(self, queries: List[QueryAnalysis]) -> Dict[str, Dict[str, int]]:
        """分析列使用情况"""
        usage = defaultdict(lambda: defaultdict(int))
        
        for query in queries:
            # WHERE条件列
            for col in query.where_columns:
                usage[col]['where'] += 1
            
            # ORDER BY列
            for col in query.order_columns:
                usage[col]['order'] += 1
            
            # GROUP BY列
            for col in query.group_columns:
                usage[col]['group'] += 1
            
            # JOIN列
            for col in query.join_columns:
                usage[col]['join'] += 1
        
        return dict(usage)
    
    def _recommend_single_column_indexes(self, table_name: str, column_usage: Dict[str, Dict[str, int]],
                                       queries: List[QueryAnalysis]) -> List[IndexRecommendation]:
        """推荐单列索引"""
        recommendations = []
        
        for column, usage in column_usage.items():
            total_usage = sum(usage.values())
            
            if total_usage >= self.config.min_query_count:
                # 计算优先级和置信度
                priority = min(10, total_usage // 10)
                confidence = min(1.0, total_usage / 100.0)
                
                # 确定索引类型
                index_type = IndexType.BTREE
                if usage.get('where', 0) > usage.get('order', 0):
                    index_type = IndexType.HASH if self._is_equality_only(column, queries) else IndexType.BTREE
                
                recommendation = IndexRecommendation(
                    table_name=table_name,
                    columns=[column],
                    index_type=index_type,
                    reason=f"High usage in queries ({total_usage} times)",
                    priority=priority,
                    confidence=confidence,
                    expected_improvement=self._estimate_improvement(column, queries)
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_composite_indexes(self, table_name: str, queries: List[QueryAnalysis]) -> List[IndexRecommendation]:
        """推荐复合索引"""
        recommendations = []
        
        # 分析常见的列组合
        column_combinations = defaultdict(int)
        
        for query in queries:
            # WHERE + ORDER BY组合
            if query.where_columns and query.order_columns:
                combo = tuple(sorted(query.where_columns + query.order_columns))
                column_combinations[combo] += 1
            
            # WHERE列组合
            if len(query.where_columns) > 1:
                combo = tuple(sorted(query.where_columns))
                column_combinations[combo] += 1
        
        # 生成推荐
        for columns, count in column_combinations.items():
            if count >= self.config.min_query_count and len(columns) <= 5:
                priority = min(10, count // 5)
                confidence = min(1.0, count / 50.0)
                
                recommendation = IndexRecommendation(
                    table_name=table_name,
                    columns=list(columns),
                    index_type=IndexType.BTREE,
                    reason=f"Common column combination in queries ({count} times)",
                    priority=priority,
                    confidence=confidence,
                    expected_improvement=self._estimate_composite_improvement(columns, queries)
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_special_indexes(self, table_name: str, queries: List[QueryAnalysis]) -> List[IndexRecommendation]:
        """推荐特殊索引"""
        recommendations = []
        
        # 分析查询模式
        pattern_counts = defaultdict(int)
        for query in queries:
            for pattern in query.patterns:
                pattern_counts[pattern] += 1
        
        # 全文搜索索引
        if pattern_counts.get(QueryPattern.LIKE, 0) >= self.config.min_query_count:
            # 这里需要更复杂的逻辑来识别文本列
            pass
        
        return recommendations
    
    def _is_equality_only(self, column: str, queries: List[QueryAnalysis]) -> bool:
        """检查列是否只用于等值查询"""
        for query in queries:
            if column in query.where_columns:
                if QueryPattern.RANGE in query.patterns:
                    return False
        return True
    
    def _estimate_improvement(self, column: str, queries: List[QueryAnalysis]) -> float:
        """估算性能提升"""
        # 简单的估算逻辑
        total_time = sum(q.execution_time_ms for q in queries if column in q.where_columns)
        if total_time > 0:
            return min(80.0, total_time / 100.0)  # 最多80%提升
        return 20.0  # 默认20%提升
    
    def _estimate_composite_improvement(self, columns: Tuple[str, ...], queries: List[QueryAnalysis]) -> float:
        """估算复合索引性能提升"""
        # 复合索引通常有更高的提升
        relevant_queries = [
            q for q in queries 
            if any(col in q.where_columns + q.order_columns for col in columns)
        ]
        
        if relevant_queries:
            avg_time = sum(q.execution_time_ms for q in relevant_queries) / len(relevant_queries)
            return min(90.0, avg_time / 50.0)  # 最多90%提升
        
        return 30.0  # 默认30%提升


class IndexManager:
    """索引管理器"""
    
    def __init__(self, config: IndexConfig, engine: Engine = None):
        self.config = config
        self.engine = engine or self._create_engine()
        
        self.statistics_collector = IndexStatisticsCollector(self.engine, config)
        self.query_analyzer = QueryAnalyzer(config)
        self.recommendation_engine = IndexRecommendationEngine(config)
        
        self._indexes: Dict[str, IndexDefinition] = {}
        self._statistics: Dict[str, IndexStatistics] = {}
    
    def _create_engine(self) -> Engine:
        """创建数据库引擎"""
        if not self.config.database_url:
            raise IndexError("Database URL is required")
        
        return create_engine(self.config.database_url)
    
    def register_index(self, index_def: IndexDefinition) -> None:
        """注册索引定义"""
        self._indexes[index_def.name] = index_def
        
        # 发布事件
        emit_business_event(
            EventType.INDEX_REGISTERED,
            "index_management",
            data={
                'index_name': index_def.name,
                'table_name': index_def.table_name,
                'columns': index_def.column_names
            }
        )
    
    def create_index(self, index_def: IndexDefinition, if_not_exists: bool = True) -> bool:
        """创建索引"""
        try:
            # 生成SQL
            sql = index_def.to_sql(self.engine.dialect.name)
            
            if if_not_exists:
                # 检查索引是否已存在
                if self._index_exists(index_def.name):
                    logger.info(f"Index {index_def.name} already exists")
                    return True
            
            # 执行创建
            with Session(self.engine) as session:
                session.execute(text(sql))
                session.commit()
            
            # 注册索引
            self.register_index(index_def)
            
            logger.info(f"Created index: {index_def.name}")
            
            # 发布事件
            emit_business_event(
                EventType.INDEX_CREATED,
                "index_management",
                data={
                    'index_name': index_def.name,
                    'table_name': index_def.table_name,
                    'sql': sql
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index {index_def.name}: {e}")
            raise IndexCreationError(f"Failed to create index {index_def.name}: {e}")
    
    def drop_index(self, index_name: str, if_exists: bool = True) -> bool:
        """删除索引"""
        try:
            if if_exists and not self._index_exists(index_name):
                logger.info(f"Index {index_name} does not exist")
                return True
            
            # 生成SQL
            sql = f"DROP INDEX {index_name};"
            
            # 执行删除
            with Session(self.engine) as session:
                session.execute(text(sql))
                session.commit()
            
            # 移除注册
            if index_name in self._indexes:
                del self._indexes[index_name]
            
            logger.info(f"Dropped index: {index_name}")
            
            # 发布事件
            emit_business_event(
                EventType.INDEX_DROPPED,
                "index_management",
                data={'index_name': index_name}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop index {index_name}: {e}")
            raise IndexError(f"Failed to drop index {index_name}: {e}")
    
    def _index_exists(self, index_name: str) -> bool:
        """检查索引是否存在"""
        with Session(self.engine) as session:
            try:
                if self.engine.dialect.name == 'postgresql':
                    sql = """
                    SELECT 1 FROM pg_indexes 
                    WHERE schemaname = :schema AND indexname = :index_name
                    """
                    result = session.execute(text(sql), {
                        'schema': self.config.schema_name,
                        'index_name': index_name
                    })
                elif self.engine.dialect.name == 'mysql':
                    sql = """
                    SELECT 1 FROM information_schema.STATISTICS 
                    WHERE TABLE_SCHEMA = DATABASE() AND INDEX_NAME = :index_name
                    """
                    result = session.execute(text(sql), {'index_name': index_name})
                elif self.engine.dialect.name == 'sqlite':
                    sql = "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = :index_name"
                    result = session.execute(text(sql), {'index_name': index_name})
                else:
                    return False
                
                return result.first() is not None
                
            except Exception:
                return False
    
    def analyze_query(self, sql: str, execution_time_ms: float = 0.0,
                     rows_examined: int = 0, rows_returned: int = 0) -> QueryAnalysis:
        """分析查询"""
        analysis = self.query_analyzer.analyze_query(sql, execution_time_ms, rows_examined, rows_returned)
        
        # 添加到推荐引擎
        self.recommendation_engine.add_query_analysis(analysis)
        
        return analysis
    
    def get_recommendations(self, table_name: Optional[str] = None) -> List[IndexRecommendation]:
        """获取索引推荐"""
        return self.recommendation_engine.generate_recommendations(table_name)
    
    def get_statistics(self, table_name: Optional[str] = None, refresh: bool = False) -> List[IndexStatistics]:
        """获取索引统计"""
        if refresh or not self._statistics:
            stats = self.statistics_collector.collect_index_statistics(table_name)
            for stat in stats:
                self._statistics[stat.name] = stat
        
        if table_name:
            return [stat for stat in self._statistics.values() if stat.table_name == table_name]
        
        return list(self._statistics.values())
    
    def get_unused_indexes(self, days: int = None) -> List[IndexStatistics]:
        """获取未使用的索引"""
        days = days or self.config.unused_threshold_days
        threshold_date = datetime.utcnow() - timedelta(days=days)
        
        unused = []
        for stat in self._statistics.values():
            if (stat.usage_level == IndexUsageLevel.NEVER or 
                (stat.last_used and stat.last_used < threshold_date)):
                unused.append(stat)
        
        return unused
    
    def optimize_indexes(self, table_name: Optional[str] = None, dry_run: bool = True) -> Dict[str, Any]:
        """优化索引"""
        results = {
            'recommendations': [],
            'unused_indexes': [],
            'actions_taken': [],
            'estimated_savings_mb': 0.0
        }
        
        # 获取推荐
        recommendations = self.get_recommendations(table_name)
        results['recommendations'] = [rec.to_dict() for rec in recommendations]
        
        # 获取未使用的索引
        unused = self.get_unused_indexes()
        if table_name:
            unused = [idx for idx in unused if idx.table_name == table_name]
        
        results['unused_indexes'] = [idx.to_dict() for idx in unused]
        results['estimated_savings_mb'] = sum(idx.size_mb for idx in unused)
        
        if not dry_run:
            # 创建推荐的索引（高优先级）
            for rec in recommendations:
                if rec.priority >= 7:  # 只创建高优先级索引
                    try:
                        index_def = rec.to_index_definition()
                        if self.create_index(index_def):
                            results['actions_taken'].append(f"Created index: {index_def.name}")
                    except Exception as e:
                        logger.error(f"Failed to create recommended index: {e}")
            
            # 删除未使用的索引（如果启用清理）
            if self.config.enable_cleanup:
                for idx in unused:
                    if idx.usage_level == IndexUsageLevel.NEVER:
                        try:
                            if self.drop_index(idx.name):
                                results['actions_taken'].append(f"Dropped unused index: {idx.name}")
                        except Exception as e:
                            logger.error(f"Failed to drop unused index: {e}")
        
        return results
    
    def get_index_summary(self) -> Dict[str, Any]:
        """获取索引摘要"""
        stats = self.get_statistics(refresh=True)
        
        total_size_mb = sum(stat.size_mb for stat in stats)
        usage_distribution = Counter(stat.usage_level.value for stat in stats)
        
        return {
            'total_indexes': len(stats),
            'total_size_mb': total_size_mb,
            'usage_distribution': dict(usage_distribution),
            'average_efficiency': sum(stat.efficiency for stat in stats) / len(stats) if stats else 0,
            'recommendations_count': len(self.get_recommendations()),
            'unused_indexes_count': len(self.get_unused_indexes())
        }


# 索引装饰器
def indexed(*columns, index_type: IndexType = IndexType.BTREE, unique: bool = False):
    """索引装饰器"""
    def decorator(cls):
        # 这里可以添加索引注册逻辑
        if not hasattr(cls, '_indexes'):
            cls._indexes = []
        
        index_columns = [IndexColumn(name=col) for col in columns]
        index_def = IndexDefinition(
            name=f"idx_{cls.__name__.lower()}_{'_'.join(columns)}",
            table_name=cls.__tablename__ if hasattr(cls, '__tablename__') else cls.__name__.lower(),
            columns=index_columns,
            index_type=index_type,
            unique=unique
        )
        
        cls._indexes.append(index_def)
        return cls
    
    return decorator


def monitor_query(func: Callable) -> Callable:
    """查询监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.utcnow()
        
        try:
            result = func(*args, **kwargs)
            
            # 计算执行时间
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # 如果有默认管理器，分析查询
            manager = get_default_index_manager()
            if manager and hasattr(result, 'statement'):
                sql = str(result.statement)
                manager.analyze_query(sql, execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    return wrapper


# 全局索引管理器
_default_index_manager: Optional[IndexManager] = None


def initialize_indexing(config: IndexConfig, engine: Engine = None) -> IndexManager:
    """初始化索引管理器"""
    global _default_index_manager
    _default_index_manager = IndexManager(config, engine)
    return _default_index_manager


def get_default_index_manager() -> Optional[IndexManager]:
    """获取默认索引管理器"""
    return _default_index_manager


# 便捷函数
def create_index(index_def: IndexDefinition, if_not_exists: bool = True) -> bool:
    """创建索引"""
    manager = get_default_index_manager()
    if manager:
        return manager.create_index(index_def, if_not_exists)
    return False


def drop_index(index_name: str, if_exists: bool = True) -> bool:
    """删除索引"""
    manager = get_default_index_manager()
    if manager:
        return manager.drop_index(index_name, if_exists)
    return False


def analyze_query(sql: str, execution_time_ms: float = 0.0) -> Optional[QueryAnalysis]:
    """分析查询"""
    manager = get_default_index_manager()
    if manager:
        return manager.analyze_query(sql, execution_time_ms)
    return None


def get_index_recommendations(table_name: Optional[str] = None) -> List[IndexRecommendation]:
    """获取索引推荐"""
    manager = get_default_index_manager()
    if manager:
        return manager.get_recommendations(table_name)
    return []


def optimize_indexes(table_name: Optional[str] = None, dry_run: bool = True) -> Dict[str, Any]:
    """优化索引"""
    manager = get_default_index_manager()
    if manager:
        return manager.optimize_indexes(table_name, dry_run)
    return {}


def get_index_statistics(table_name: Optional[str] = None) -> List[IndexStatistics]:
    """获取索引统计"""
    manager = get_default_index_manager()
    if manager:
        return manager.get_statistics(table_name)
    return []


# 导出所有类和函数
__all__ = [
    "IndexType",
    "IndexStatus",
    "IndexUsageLevel",
    "QueryPattern",
    "IndexColumn",
    "IndexDefinition",
    "IndexStatistics",
    "QueryAnalysis",
    "IndexRecommendation",
    "IndexConfig",
    "IndexError",
    "IndexAnalysisError",
    "IndexCreationError",
    "IndexStatisticsCollector",
    "QueryAnalyzer",
    "IndexRecommendationEngine",
    "IndexManager",
    "indexed",
    "monitor_query",
    "initialize_indexing",
    "get_default_index_manager",
    "create_index",
    "drop_index",
    "analyze_query",
    "get_index_recommendations",
    "optimize_indexes",
    "get_index_statistics"
]