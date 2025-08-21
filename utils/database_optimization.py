"""数据库性能优化模块

提供数据库性能优化功能，包括索引管理、查询优化、连接池配置、
缓存策略、分页优化等功能。
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from functools import wraps
from contextlib import contextmanager
from sqlalchemy import (
    create_engine, text, Index, inspect, MetaData, Table,
    event, pool, select, func, and_, or_
)
from sqlalchemy.orm import Session, sessionmaker, Query
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.sql import Select
from redis import Redis
import json
import hashlib


class DatabaseOptimizer:
    """数据库优化器
    
    提供数据库性能优化功能，包括连接池管理、查询优化、索引建议等。
    """
    
    def __init__(self, database_url: str, redis_client: Redis = None):
        self.database_url = database_url
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        self._query_stats = {}
        self._slow_queries = []
        
        # 创建优化的数据库引擎
        self.engine = self._create_optimized_engine()
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def _create_optimized_engine(self) -> Engine:
        """创建优化的数据库引擎"""
        engine_kwargs = {
            'poolclass': QueuePool,
            'pool_size': 20,  # 连接池大小
            'max_overflow': 30,  # 最大溢出连接数
            'pool_pre_ping': True,  # 连接前ping检查
            'pool_recycle': 3600,  # 连接回收时间（秒）
            'pool_timeout': 30,  # 获取连接超时时间
            'echo': False,  # 生产环境关闭SQL日志
            'echo_pool': False,  # 关闭连接池日志
            'connect_args': {
                'connect_timeout': 10,
                'application_name': 'langgraph_backend'
            }
        }
        
        # PostgreSQL特定优化
        if 'postgresql' in self.database_url:
            engine_kwargs['connect_args'].update({
                'server_side_cursors': True,
                'options': '-c default_transaction_isolation=read_committed'
            })
        
        engine = create_engine(self.database_url, **engine_kwargs)
        
        # 注册事件监听器
        self._register_event_listeners(engine)
        
        return engine
    
    def _register_event_listeners(self, engine: Engine):
        """注册数据库事件监听器"""
        
        @event.listens_for(engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """查询执行前事件"""
            context._query_start_time = time.time()
        
        @event.listens_for(engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """查询执行后事件"""
            total_time = time.time() - context._query_start_time
            
            # 记录查询统计
            self._record_query_stats(statement, total_time, parameters)
            
            # 记录慢查询
            if total_time > 1.0:  # 超过1秒的查询
                self._record_slow_query(statement, total_time, parameters)
    
    def _record_query_stats(self, statement: str, execution_time: float, parameters: Any):
        """记录查询统计信息"""
        # 简化SQL语句用作键
        query_key = self._normalize_query(statement)
        
        if query_key not in self._query_stats:
            self._query_stats[query_key] = {
                'count': 0,
                'total_time': 0,
                'avg_time': 0,
                'max_time': 0,
                'min_time': float('inf')
            }
        
        stats = self._query_stats[query_key]
        stats['count'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['max_time'] = max(stats['max_time'], execution_time)
        stats['min_time'] = min(stats['min_time'], execution_time)
    
    def _record_slow_query(self, statement: str, execution_time: float, parameters: Any):
        """记录慢查询"""
        slow_query = {
            'statement': statement,
            'execution_time': execution_time,
            'parameters': str(parameters)[:500],  # 限制参数长度
            'timestamp': datetime.utcnow()
        }
        
        self._slow_queries.append(slow_query)
        
        # 保持最近100条慢查询记录
        if len(self._slow_queries) > 100:
            self._slow_queries = self._slow_queries[-100:]
        
        # 记录到日志
        self.logger.warning(
            f"慢查询检测: {execution_time:.2f}s - {statement[:200]}..."
        )
    
    def _normalize_query(self, statement: str) -> str:
        """标准化查询语句"""
        # 移除参数占位符，保留查询结构
        import re
        normalized = re.sub(r'\$\d+|\?|%\([^)]+\)s', '?', statement)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized[:200]  # 限制长度
    
    @contextmanager
    def get_db_session(self):
        """获取数据库会话上下文管理器"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_query_stats(self) -> Dict[str, Any]:
        """获取查询统计信息"""
        return {
            'total_queries': sum(stats['count'] for stats in self._query_stats.values()),
            'unique_queries': len(self._query_stats),
            'slow_queries_count': len(self._slow_queries),
            'top_queries': sorted(
                self._query_stats.items(),
                key=lambda x: x[1]['total_time'],
                reverse=True
            )[:10],
            'recent_slow_queries': self._slow_queries[-10:]
        }
    
    def analyze_table_indexes(self, table_name: str) -> Dict[str, Any]:
        """分析表索引使用情况"""
        with self.get_db_session() as session:
            # 获取表的索引信息
            inspector = inspect(self.engine)
            indexes = inspector.get_indexes(table_name)
            
            # 获取索引使用统计（PostgreSQL）
            if 'postgresql' in self.database_url:
                index_stats_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes 
                    WHERE tablename = :table_name
                """)
                
                result = session.execute(index_stats_query, {'table_name': table_name})
                index_usage = {row.indexname: dict(row) for row in result}
            else:
                index_usage = {}
            
            return {
                'table_name': table_name,
                'indexes': indexes,
                'index_usage': index_usage,
                'recommendations': self._generate_index_recommendations(table_name, indexes, index_usage)
            }
    
    def _generate_index_recommendations(self, table_name: str, indexes: List[Dict], 
                                       index_usage: Dict[str, Any]) -> List[str]:
        """生成索引建议"""
        recommendations = []
        
        # 检查未使用的索引
        for index in indexes:
            index_name = index['name']
            if index_name in index_usage:
                usage = index_usage[index_name]
                if usage.get('idx_scan', 0) == 0:
                    recommendations.append(f"考虑删除未使用的索引: {index_name}")
        
        # 基于慢查询分析建议新索引
        for slow_query in self._slow_queries[-20:]:  # 分析最近20条慢查询
            statement = slow_query['statement'].lower()
            if f'from {table_name}' in statement or f'join {table_name}' in statement:
                # 简单的WHERE子句分析
                if 'where' in statement:
                    # 提取可能需要索引的字段
                    import re
                    where_clause = statement.split('where')[1].split('order by')[0].split('group by')[0]
                    fields = re.findall(r'(\w+)\s*[=<>!]', where_clause)
                    if fields:
                        recommendations.append(f"考虑为字段创建索引: {', '.join(set(fields))}")
        
        return recommendations
    
    def optimize_query(self, query: Query) -> Query:
        """优化查询"""
        # 添加查询提示和优化
        optimized_query = query
        
        # 如果是PostgreSQL，添加查询提示
        if 'postgresql' in self.database_url:
            # 启用并行查询
            optimized_query = optimized_query.execution_options(
                postgresql_readonly=True,
                postgresql_isolation_level='READ_COMMITTED'
            )
        
        return optimized_query
    
    def create_recommended_indexes(self, table_name: str, fields: List[str]):
        """创建推荐的索引"""
        with self.get_db_session() as session:
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=self.engine)
            
            for field in fields:
                if hasattr(table.c, field):
                    index_name = f"idx_{table_name}_{field}"
                    index = Index(index_name, getattr(table.c, field))
                    
                    try:
                        index.create(self.engine)
                        self.logger.info(f"创建索引成功: {index_name}")
                    except Exception as e:
                        self.logger.error(f"创建索引失败: {index_name}, 错误: {str(e)}")


class QueryCache:
    """查询缓存管理器"""
    
    def __init__(self, redis_client: Redis, default_ttl: int = 300):
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.cache_prefix = "query_cache:"
    
    def _generate_cache_key(self, query: str, params: Dict[str, Any] = None) -> str:
        """生成缓存键"""
        cache_data = {
            'query': query,
            'params': params or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
        return f"{self.cache_prefix}{cache_hash}"
    
    def get(self, query: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """获取缓存结果"""
        if not self.redis_client:
            return None
        
        cache_key = self._generate_cache_key(query, params)
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logging.warning(f"缓存读取失败: {str(e)}")
        
        return None
    
    def set(self, query: str, result: Any, params: Dict[str, Any] = None, 
            ttl: int = None) -> bool:
        """设置缓存结果"""
        if not self.redis_client:
            return False
        
        cache_key = self._generate_cache_key(query, params)
        ttl = ttl or self.default_ttl
        
        try:
            serialized_result = json.dumps(result, default=str)
            self.redis_client.setex(cache_key, ttl, serialized_result)
            return True
        except Exception as e:
            logging.warning(f"缓存写入失败: {str(e)}")
            return False
    
    def invalidate_pattern(self, pattern: str):
        """按模式失效缓存"""
        if not self.redis_client:
            return
        
        try:
            keys = self.redis_client.keys(f"{self.cache_prefix}*{pattern}*")
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logging.warning(f"缓存失效失败: {str(e)}")


def cached_query(ttl: int = 300, cache_key_func: callable = None):
    """查询缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取缓存实例（假设在全局或依赖注入中）
            cache = getattr(wrapper, '_cache', None)
            if not cache:
                return func(*args, **kwargs)
            
            # 生成缓存键
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行查询并缓存结果
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator


class PaginationOptimizer:
    """分页优化器"""
    
    @staticmethod
    def optimize_pagination(query: Query, page: int, page_size: int, 
                           use_cursor: bool = False, cursor_field: str = 'id') -> Tuple[List[Any], Dict[str, Any]]:
        """优化分页查询"""
        if use_cursor:
            return PaginationOptimizer._cursor_pagination(query, page_size, cursor_field)
        else:
            return PaginationOptimizer._offset_pagination(query, page, page_size)
    
    @staticmethod
    def _offset_pagination(query: Query, page: int, page_size: int) -> Tuple[List[Any], Dict[str, Any]]:
        """偏移分页（适用于小数据集）"""
        # 计算总数（优化：只在第一页或明确需要时计算）
        total_count = query.count()
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 获取数据
        items = query.offset(offset).limit(page_size).all()
        
        # 计算分页信息
        total_pages = (total_count + page_size - 1) // page_size
        
        pagination_info = {
            'page': page,
            'page_size': page_size,
            'total_items': total_count,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
        
        return items, pagination_info
    
    @staticmethod
    def _cursor_pagination(query: Query, page_size: int, cursor_field: str) -> Tuple[List[Any], Dict[str, Any]]:
        """游标分页（适用于大数据集）"""
        # 获取数据（多取一条用于判断是否有下一页）
        items = query.limit(page_size + 1).all()
        
        has_next = len(items) > page_size
        if has_next:
            items = items[:-1]  # 移除多取的一条
        
        # 生成游标
        next_cursor = None
        prev_cursor = None
        
        if items:
            if has_next:
                next_cursor = getattr(items[-1], cursor_field)
            prev_cursor = getattr(items[0], cursor_field)
        
        pagination_info = {
            'page_size': page_size,
            'has_next': has_next,
            'next_cursor': next_cursor,
            'prev_cursor': prev_cursor
        }
        
        return items, pagination_info


class DatabaseMonitor:
    """数据库监控器"""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
    
    def get_connection_pool_status(self) -> Dict[str, Any]:
        """获取连接池状态"""
        pool = self.engine.pool
        return {
            'pool_size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'invalid': pool.invalid()
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        with self.engine.connect() as conn:
            if 'postgresql' in str(self.engine.url):
                # PostgreSQL统计信息
                stats_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_tuples,
                        n_dead_tup as dead_tuples,
                        last_vacuum,
                        last_autovacuum,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_stat_user_tables
                    ORDER BY n_live_tup DESC
                    LIMIT 20
                """)
                
                result = conn.execute(stats_query)
                table_stats = [dict(row) for row in result]
                
                return {
                    'table_statistics': table_stats,
                    'connection_pool': self.get_connection_pool_status()
                }
            else:
                return {
                    'connection_pool': self.get_connection_pool_status()
                }
    
    def check_database_health(self) -> Dict[str, Any]:
        """检查数据库健康状态"""
        health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.utcnow()
        }
        
        try:
            # 连接测试
            start_time = time.time()
            with self.engine.connect() as conn:
                conn.execute(text('SELECT 1'))
            connection_time = time.time() - start_time
            
            health_status['checks']['connection'] = {
                'status': 'ok',
                'response_time': connection_time
            }
            
            # 连接池检查
            pool_status = self.get_connection_pool_status()
            pool_utilization = (pool_status['checked_out'] / pool_status['pool_size']) * 100
            
            health_status['checks']['connection_pool'] = {
                'status': 'ok' if pool_utilization < 80 else 'warning',
                'utilization': pool_utilization,
                'details': pool_status
            }
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['checks']['connection'] = {
                'status': 'error',
                'error': str(e)
            }
        
        return health_status


# 全局优化器实例（在应用启动时初始化）
_db_optimizer = None
_query_cache = None


def init_database_optimization(database_url: str, redis_client: Redis = None):
    """初始化数据库优化组件"""
    global _db_optimizer, _query_cache
    
    _db_optimizer = DatabaseOptimizer(database_url, redis_client)
    if redis_client:
        _query_cache = QueryCache(redis_client)


def get_db_optimizer() -> DatabaseOptimizer:
    """获取数据库优化器实例"""
    return _db_optimizer


def get_query_cache() -> QueryCache:
    """获取查询缓存实例"""
    return _query_cache