"""数据库性能监控和优化工具模块

本模块提供数据库性能监控、查询优化、索引管理、缓存策略等功能。
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import hashlib
import threading
from collections import defaultdict, deque
from functools import wraps
import logging
from sqlalchemy import text, event, inspect
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from sqlalchemy.sql import Select
from sqlalchemy.dialects import postgresql


class QueryType(Enum):
    """查询类型枚举"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"
    INDEX = "index"


class PerformanceLevel(Enum):
    """性能级别枚举"""
    EXCELLENT = "excellent"  # < 10ms
    GOOD = "good"           # 10-50ms
    AVERAGE = "average"     # 50-200ms
    POOR = "poor"           # 200-1000ms
    CRITICAL = "critical"   # > 1000ms


@dataclass
class QueryMetrics:
    """查询指标"""
    query_id: str
    sql: str
    query_type: QueryType
    execution_time: float
    rows_affected: int
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    explain_plan: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    @property
    def performance_level(self) -> PerformanceLevel:
        """获取性能级别"""
        if self.execution_time < 0.01:  # < 10ms
            return PerformanceLevel.EXCELLENT
        elif self.execution_time < 0.05:  # < 50ms
            return PerformanceLevel.GOOD
        elif self.execution_time < 0.2:  # < 200ms
            return PerformanceLevel.AVERAGE
        elif self.execution_time < 1.0:  # < 1000ms
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'query_id': self.query_id,
            'sql': self.sql,
            'query_type': self.query_type.value,
            'execution_time': self.execution_time,
            'rows_affected': self.rows_affected,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'parameters': self.parameters,
            'explain_plan': self.explain_plan,
            'error': self.error,
            'performance_level': self.performance_level.value
        }


@dataclass
class IndexSuggestion:
    """索引建议"""
    table_name: str
    columns: List[str]
    index_type: str  # btree, hash, gin, gist等
    reason: str
    estimated_benefit: float  # 0-1之间，表示预期性能提升
    query_count: int  # 受益的查询数量
    create_sql: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'table_name': self.table_name,
            'columns': self.columns,
            'index_type': self.index_type,
            'reason': self.reason,
            'estimated_benefit': self.estimated_benefit,
            'query_count': self.query_count,
            'create_sql': self.create_sql
        }


@dataclass
class PerformanceReport:
    """性能报告"""
    start_time: datetime
    end_time: datetime
    total_queries: int
    avg_execution_time: float
    slow_queries: List[QueryMetrics]
    query_distribution: Dict[QueryType, int]
    performance_distribution: Dict[PerformanceLevel, int]
    index_suggestions: List[IndexSuggestion]
    cache_stats: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_queries': self.total_queries,
            'avg_execution_time': self.avg_execution_time,
            'slow_queries': [q.to_dict() for q in self.slow_queries],
            'query_distribution': {k.value: v for k, v in self.query_distribution.items()},
            'performance_distribution': {k.value: v for k, v in self.performance_distribution.items()},
            'index_suggestions': [s.to_dict() for s in self.index_suggestions],
            'cache_stats': self.cache_stats,
            'recommendations': self.recommendations
        }


class QueryCache:
    """查询缓存"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_order = deque()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
    
    def _generate_key(self, sql: str, params: Dict[str, Any] = None) -> str:
        """生成缓存键"""
        key_data = {'sql': sql, 'params': params or {}}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, sql: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """获取缓存"""
        key = self._generate_key(sql, params)
        
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # 检查是否过期
                if datetime.now() - timestamp > timedelta(seconds=self.ttl):
                    del self.cache[key]
                    self.access_order.remove(key)
                    self.stats['misses'] += 1
                    self.stats['size'] -= 1
                    return None
                
                # 更新访问顺序
                self.access_order.remove(key)
                self.access_order.append(key)
                self.stats['hits'] += 1
                return value
            
            self.stats['misses'] += 1
            return None
    
    def set(self, sql: str, value: Any, params: Dict[str, Any] = None) -> None:
        """设置缓存"""
        key = self._generate_key(sql, params)
        
        with self.lock:
            # 如果缓存已满，删除最旧的项
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = self.access_order.popleft()
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
                self.stats['size'] -= 1
            
            # 添加或更新缓存
            if key in self.cache:
                self.access_order.remove(key)
            else:
                self.stats['size'] += 1
            
            self.cache[key] = (value, datetime.now())
            self.access_order.append(key)
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'size': 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, engine: Engine, max_metrics: int = 10000):
        self.engine = engine
        self.max_metrics = max_metrics
        self.metrics: deque[QueryMetrics] = deque(maxlen=max_metrics)
        self.slow_query_threshold = 0.1  # 100ms
        self.cache = QueryCache()
        self.logger = logging.getLogger(__name__)
        self._setup_event_listeners()
    
    def _setup_event_listeners(self) -> None:
        """设置事件监听器"""
        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
            context._query_id = hashlib.md5(
                f"{statement}{parameters}{time.time()}".encode()
            ).hexdigest()[:16]
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            execution_time = time.time() - context._query_start_time
            
            # 创建查询指标
            metrics = QueryMetrics(
                query_id=context._query_id,
                sql=statement,
                query_type=self._detect_query_type(statement),
                execution_time=execution_time,
                rows_affected=cursor.rowcount if cursor.rowcount >= 0 else 0,
                timestamp=datetime.now(),
                parameters=dict(parameters) if parameters else {}
            )
            
            self.metrics.append(metrics)
            
            # 记录慢查询
            if execution_time > self.slow_query_threshold:
                self.logger.warning(
                    f"Slow query detected: {execution_time:.3f}s - {statement[:100]}..."
                )
    
    def _detect_query_type(self, sql: str) -> QueryType:
        """检测查询类型"""
        sql_lower = sql.strip().lower()
        
        if sql_lower.startswith('select'):
            return QueryType.SELECT
        elif sql_lower.startswith('insert'):
            return QueryType.INSERT
        elif sql_lower.startswith('update'):
            return QueryType.UPDATE
        elif sql_lower.startswith('delete'):
            return QueryType.DELETE
        elif sql_lower.startswith('create'):
            return QueryType.CREATE
        elif sql_lower.startswith('drop'):
            return QueryType.DROP
        elif sql_lower.startswith('alter'):
            return QueryType.ALTER
        elif 'index' in sql_lower:
            return QueryType.INDEX
        else:
            return QueryType.SELECT  # 默认
    
    def get_slow_queries(self, threshold: float = None, limit: int = 10) -> List[QueryMetrics]:
        """获取慢查询"""
        if threshold is None:
            threshold = self.slow_query_threshold
        
        slow_queries = [
            metric for metric in self.metrics
            if metric.execution_time > threshold
        ]
        
        # 按执行时间降序排序
        slow_queries.sort(key=lambda x: x.execution_time, reverse=True)
        
        return slow_queries[:limit]
    
    def get_query_distribution(self) -> Dict[QueryType, int]:
        """获取查询类型分布"""
        distribution = defaultdict(int)
        for metric in self.metrics:
            distribution[metric.query_type] += 1
        return dict(distribution)
    
    def get_performance_distribution(self) -> Dict[PerformanceLevel, int]:
        """获取性能级别分布"""
        distribution = defaultdict(int)
        for metric in self.metrics:
            distribution[metric.performance_level] += 1
        return dict(distribution)
    
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """分析查询模式"""
        if not self.metrics:
            return {}
        
        # 统计查询频率
        query_patterns = defaultdict(int)
        table_access = defaultdict(int)
        
        for metric in self.metrics:
            # 简化SQL以识别模式
            normalized_sql = self._normalize_sql(metric.sql)
            query_patterns[normalized_sql] += 1
            
            # 提取表名
            tables = self._extract_tables(metric.sql)
            for table in tables:
                table_access[table] += 1
        
        return {
            'most_frequent_queries': dict(sorted(
                query_patterns.items(), key=lambda x: x[1], reverse=True
            )[:10]),
            'most_accessed_tables': dict(sorted(
                table_access.items(), key=lambda x: x[1], reverse=True
            )[:10])
        }
    
    def _normalize_sql(self, sql: str) -> str:
        """标准化SQL以识别模式"""
        # 移除参数值，保留结构
        import re
        
        # 替换数字和字符串字面量
        sql = re.sub(r"'[^']*'", "'?'", sql)
        sql = re.sub(r'"[^"]*"', '"?"', sql)
        sql = re.sub(r'\b\d+\b', '?', sql)
        
        # 移除多余空格
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        return sql
    
    def _extract_tables(self, sql: str) -> List[str]:
        """从SQL中提取表名"""
        import re
        
        # 简单的表名提取（可以改进）
        patterns = [
            r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'UPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'INSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'DELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        tables = set()
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.update(matches)
        
        return list(tables)
    
    def suggest_indexes(self, session: Session) -> List[IndexSuggestion]:
        """建议索引"""
        suggestions = []
        
        # 分析慢查询中的WHERE条件
        slow_queries = self.get_slow_queries()
        table_columns = defaultdict(set)
        
        for query in slow_queries:
            # 提取WHERE条件中的列
            columns = self._extract_where_columns(query.sql)
            for table, cols in columns.items():
                table_columns[table].update(cols)
        
        # 检查现有索引
        existing_indexes = self._get_existing_indexes(session)
        
        # 生成索引建议
        for table, columns in table_columns.items():
            for column in columns:
                if not self._has_index(existing_indexes, table, column):
                    suggestion = IndexSuggestion(
                        table_name=table,
                        columns=[column],
                        index_type='btree',
                        reason=f'Frequent filtering on {column} in slow queries',
                        estimated_benefit=0.7,  # 估算值
                        query_count=len([q for q in slow_queries if table in q.sql]),
                        create_sql=f'CREATE INDEX idx_{table}_{column} ON {table} ({column});'
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _extract_where_columns(self, sql: str) -> Dict[str, List[str]]:
        """从WHERE条件中提取列名"""
        import re
        
        # 简化的WHERE条件解析
        where_match = re.search(r'WHERE\s+(.+?)(?:ORDER|GROUP|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return {}
        
        where_clause = where_match.group(1)
        
        # 提取列名（简化版本）
        column_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>!]'
        matches = re.findall(column_pattern, where_clause)
        
        result = defaultdict(list)
        for table, column in matches:
            result[table].append(column)
        
        return dict(result)
    
    def _get_existing_indexes(self, session: Session) -> Dict[str, List[Dict[str, Any]]]:
        """获取现有索引"""
        try:
            # PostgreSQL查询索引信息
            sql = """
            SELECT 
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes 
            WHERE schemaname = 'public'
            ORDER BY tablename, indexname;
            """
            
            result = session.execute(text(sql))
            indexes = defaultdict(list)
            
            for row in result:
                indexes[row.tablename].append({
                    'name': row.indexname,
                    'definition': row.indexdef
                })
            
            return dict(indexes)
        except Exception as e:
            self.logger.error(f"Error getting existing indexes: {e}")
            return {}
    
    def _has_index(self, existing_indexes: Dict[str, List[Dict[str, Any]]], 
                   table: str, column: str) -> bool:
        """检查是否已有索引"""
        if table not in existing_indexes:
            return False
        
        for index in existing_indexes[table]:
            if column in index['definition']:
                return True
        
        return False
    
    def generate_report(self, session: Session, hours: int = 24) -> PerformanceReport:
        """生成性能报告"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # 过滤时间范围内的指标
        filtered_metrics = [
            metric for metric in self.metrics
            if start_time <= metric.timestamp <= end_time
        ]
        
        if not filtered_metrics:
            return PerformanceReport(
                start_time=start_time,
                end_time=end_time,
                total_queries=0,
                avg_execution_time=0.0,
                slow_queries=[],
                query_distribution={},
                performance_distribution={},
                index_suggestions=[],
                cache_stats=self.cache.get_stats(),
                recommendations=[]
            )
        
        # 计算统计信息
        total_queries = len(filtered_metrics)
        avg_execution_time = sum(m.execution_time for m in filtered_metrics) / total_queries
        slow_queries = [m for m in filtered_metrics if m.execution_time > self.slow_query_threshold]
        
        # 查询分布
        query_distribution = defaultdict(int)
        performance_distribution = defaultdict(int)
        
        for metric in filtered_metrics:
            query_distribution[metric.query_type] += 1
            performance_distribution[metric.performance_level] += 1
        
        # 索引建议
        index_suggestions = self.suggest_indexes(session)
        
        # 生成建议
        recommendations = self._generate_recommendations(
            filtered_metrics, slow_queries, avg_execution_time
        )
        
        return PerformanceReport(
            start_time=start_time,
            end_time=end_time,
            total_queries=total_queries,
            avg_execution_time=avg_execution_time,
            slow_queries=slow_queries[:10],  # 只返回前10个慢查询
            query_distribution=dict(query_distribution),
            performance_distribution=dict(performance_distribution),
            index_suggestions=index_suggestions,
            cache_stats=self.cache.get_stats(),
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, all_metrics: List[QueryMetrics], 
                                slow_queries: List[QueryMetrics], 
                                avg_execution_time: float) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 慢查询建议
        if len(slow_queries) > len(all_metrics) * 0.1:  # 超过10%的查询是慢查询
            recommendations.append(
                f"发现 {len(slow_queries)} 个慢查询，建议优化查询语句或添加索引"
            )
        
        # 平均执行时间建议
        if avg_execution_time > 0.05:  # 平均执行时间超过50ms
            recommendations.append(
                f"平均查询时间为 {avg_execution_time:.3f}s，建议优化数据库性能"
            )
        
        # 缓存建议
        cache_stats = self.cache.get_stats()
        if cache_stats['hit_rate'] < 0.8:  # 缓存命中率低于80%
            recommendations.append(
                f"缓存命中率为 {cache_stats['hit_rate']:.2%}，建议优化缓存策略"
            )
        
        # 查询类型建议
        query_dist = defaultdict(int)
        for metric in all_metrics:
            query_dist[metric.query_type] += 1
        
        if query_dist[QueryType.SELECT] > len(all_metrics) * 0.8:
            recommendations.append("查询以SELECT为主，建议考虑读写分离")
        
        return recommendations


def cached_query(cache: QueryCache, ttl: int = None):
    """查询缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # 尝试从缓存获取
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # 执行查询
            result = func(*args, **kwargs)
            
            # 存入缓存
            cache.set(cache_key, result)
            
            return result
        return wrapper
    return decorator


def monitor_performance(monitor: PerformanceMonitor):
    """性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # 记录性能指标
                metrics = QueryMetrics(
                    query_id=f"{func.__name__}_{int(time.time())}",
                    sql=f"Function: {func.__name__}",
                    query_type=QueryType.SELECT,  # 默认类型
                    execution_time=execution_time,
                    rows_affected=0,
                    timestamp=datetime.now()
                )
                
                monitor.metrics.append(metrics)
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                
                # 记录错误
                metrics = QueryMetrics(
                    query_id=f"{func.__name__}_{int(time.time())}",
                    sql=f"Function: {func.__name__}",
                    query_type=QueryType.SELECT,
                    execution_time=execution_time,
                    rows_affected=0,
                    timestamp=datetime.now(),
                    error=str(e)
                )
                
                monitor.metrics.append(metrics)
                raise
        
        return wrapper
    return decorator


# 导出所有类和函数
__all__ = [
    "QueryType",
    "PerformanceLevel",
    "QueryMetrics",
    "IndexSuggestion",
    "PerformanceReport",
    "QueryCache",
    "PerformanceMonitor",
    "cached_query",
    "monitor_performance"
]