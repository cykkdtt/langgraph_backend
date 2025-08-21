"""模型优化系统模块

本模块提供数据库性能优化、查询优化和资源管理功能。
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
import time
import threading
import asyncio
from collections import defaultdict, deque
from contextlib import contextmanager
import statistics
import psutil
import gc

# SQLAlchemy imports
try:
    from sqlalchemy import (
        text, inspect, MetaData, Table, Column,
        Integer, String, DateTime, Boolean, Float,
        create_engine, Engine, event
    )
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.pool import QueuePool, StaticPool
    from sqlalchemy.sql import select, func
    from sqlalchemy.engine import Connection
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
    QueuePool = None
    StaticPool = None
    select = None
    func = None
    Connection = None
    SQLALCHEMY_AVAILABLE = False

# 导入项目模块
from .events import EventType, emit_business_event
from .validation import validate_data


logger = logging.getLogger(__name__)


T = TypeVar('T')


class OptimizationType(Enum):
    """优化类型枚举"""
    QUERY = "query"                        # 查询优化
    INDEX = "index"                        # 索引优化
    CONNECTION = "connection"              # 连接优化
    MEMORY = "memory"                      # 内存优化
    CACHE = "cache"                        # 缓存优化
    BATCH = "batch"                        # 批处理优化
    PARTITION = "partition"                # 分区优化
    VACUUM = "vacuum"                      # 清理优化


class PerformanceLevel(Enum):
    """性能级别枚举"""
    EXCELLENT = "excellent"                # 优秀
    GOOD = "good"                          # 良好
    AVERAGE = "average"                    # 一般
    POOR = "poor"                          # 较差
    CRITICAL = "critical"                  # 严重


class OptimizationStrategy(Enum):
    """优化策略枚举"""
    AGGRESSIVE = "aggressive"              # 激进优化
    BALANCED = "balanced"                  # 平衡优化
    CONSERVATIVE = "conservative"          # 保守优化
    CUSTOM = "custom"                      # 自定义优化


class ResourceType(Enum):
    """资源类型枚举"""
    CPU = "cpu"                            # CPU资源
    MEMORY = "memory"                      # 内存资源
    DISK = "disk"                          # 磁盘资源
    NETWORK = "network"                    # 网络资源
    DATABASE = "database"                  # 数据库资源


@dataclass
class QueryPerformance:
    """查询性能"""
    query_id: str                          # 查询ID
    sql: str                               # SQL语句
    execution_time: float                  # 执行时间（秒）
    rows_examined: Optional[int] = None    # 检查行数
    rows_returned: Optional[int] = None    # 返回行数
    index_usage: List[str] = field(default_factory=list)  # 使用的索引
    
    # 性能指标
    cpu_time: Optional[float] = None       # CPU时间
    io_cost: Optional[float] = None        # IO成本
    memory_usage: Optional[int] = None     # 内存使用
    
    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def performance_level(self) -> PerformanceLevel:
        """性能级别"""
        if self.execution_time < 0.1:
            return PerformanceLevel.EXCELLENT
        elif self.execution_time < 0.5:
            return PerformanceLevel.GOOD
        elif self.execution_time < 2.0:
            return PerformanceLevel.AVERAGE
        elif self.execution_time < 10.0:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    @property
    def efficiency_ratio(self) -> Optional[float]:
        """效率比率（返回行数/检查行数）"""
        if self.rows_examined and self.rows_examined > 0:
            return (self.rows_returned or 0) / self.rows_examined
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'query_id': self.query_id,
            'sql': self.sql,
            'execution_time': self.execution_time,
            'rows_examined': self.rows_examined,
            'rows_returned': self.rows_returned,
            'index_usage': self.index_usage,
            'cpu_time': self.cpu_time,
            'io_cost': self.io_cost,
            'memory_usage': self.memory_usage,
            'performance_level': self.performance_level.value,
            'efficiency_ratio': self.efficiency_ratio,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ResourceUsage:
    """资源使用情况"""
    resource_type: ResourceType            # 资源类型
    current_usage: float                   # 当前使用量
    max_usage: float                       # 最大使用量
    average_usage: float                   # 平均使用量
    peak_usage: float                      # 峰值使用量
    
    # 时间范围
    start_time: datetime                   # 开始时间
    end_time: datetime                     # 结束时间
    
    # 单位
    unit: str = "percent"                  # 单位
    
    @property
    def usage_percentage(self) -> float:
        """使用百分比"""
        if self.max_usage > 0:
            return (self.current_usage / self.max_usage) * 100
        return 0.0
    
    @property
    def is_critical(self) -> bool:
        """是否达到临界值"""
        return self.usage_percentage > 90
    
    @property
    def is_warning(self) -> bool:
        """是否达到警告值"""
        return self.usage_percentage > 75
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'resource_type': self.resource_type.value,
            'current_usage': self.current_usage,
            'max_usage': self.max_usage,
            'average_usage': self.average_usage,
            'peak_usage': self.peak_usage,
            'usage_percentage': self.usage_percentage,
            'is_critical': self.is_critical,
            'is_warning': self.is_warning,
            'unit': self.unit,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat()
        }


@dataclass
class OptimizationRecommendation:
    """优化建议"""
    optimization_type: OptimizationType    # 优化类型
    title: str                             # 标题
    description: str                       # 描述
    impact: str                            # 影响
    effort: str                            # 工作量
    priority: int                          # 优先级（1-10）
    
    # 具体建议
    action: str                            # 行动
    sql_commands: List[str] = field(default_factory=list)  # SQL命令
    config_changes: Dict[str, Any] = field(default_factory=dict)  # 配置变更
    
    # 预期收益
    expected_improvement: Optional[float] = None  # 预期改进百分比
    estimated_time_saving: Optional[float] = None  # 预计节省时间
    
    # 风险评估
    risk_level: str = "low"                # 风险级别
    rollback_plan: Optional[str] = None    # 回滚计划
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'optimization_type': self.optimization_type.value,
            'title': self.title,
            'description': self.description,
            'impact': self.impact,
            'effort': self.effort,
            'priority': self.priority,
            'action': self.action,
            'sql_commands': self.sql_commands,
            'config_changes': self.config_changes,
            'expected_improvement': self.expected_improvement,
            'estimated_time_saving': self.estimated_time_saving,
            'risk_level': self.risk_level,
            'rollback_plan': self.rollback_plan,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class OptimizationConfig:
    """优化配置"""
    # 策略配置
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    
    # 性能阈值
    slow_query_threshold: float = 1.0      # 慢查询阈值（秒）
    memory_warning_threshold: float = 75.0 # 内存警告阈值（%）
    cpu_warning_threshold: float = 80.0    # CPU警告阈值（%）
    
    # 监控配置
    enable_query_monitoring: bool = True   # 启用查询监控
    enable_resource_monitoring: bool = True # 启用资源监控
    monitoring_interval: int = 60          # 监控间隔（秒）
    
    # 优化配置
    auto_optimize: bool = False            # 自动优化
    max_optimization_time: int = 300       # 最大优化时间（秒）
    backup_before_optimize: bool = True    # 优化前备份
    
    # 连接池配置
    connection_pool_size: int = 20         # 连接池大小
    connection_pool_max_overflow: int = 10 # 连接池最大溢出
    connection_pool_timeout: int = 30      # 连接池超时
    
    # 缓存配置
    enable_query_cache: bool = True        # 启用查询缓存
    query_cache_size: int = 1000           # 查询缓存大小
    query_cache_ttl: int = 3600            # 查询缓存TTL
    
    # 批处理配置
    batch_size: int = 1000                 # 批处理大小
    enable_batch_optimization: bool = True # 启用批处理优化


class OptimizationError(Exception):
    """优化错误"""
    pass


class PerformanceMonitoringError(OptimizationError):
    """性能监控错误"""
    pass


class ResourceMonitoringError(OptimizationError):
    """资源监控错误"""
    pass


class QueryPerformanceMonitor:
    """查询性能监控器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._query_history: List[QueryPerformance] = []
        self._slow_queries: List[QueryPerformance] = []
        self._query_stats: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_query(self, query_performance: QueryPerformance) -> None:
        """记录查询性能"""
        with self._lock:
            self._query_history.append(query_performance)
            
            # 记录慢查询
            if query_performance.execution_time > self.config.slow_query_threshold:
                self._slow_queries.append(query_performance)
            
            # 更新统计信息
            query_hash = hash(query_performance.sql)
            self._query_stats[str(query_hash)].append(query_performance.execution_time)
            
            # 限制历史记录大小
            if len(self._query_history) > 10000:
                self._query_history = self._query_history[-5000:]
            
            if len(self._slow_queries) > 1000:
                self._slow_queries = self._slow_queries[-500:]
    
    def get_slow_queries(self, limit: int = 100) -> List[QueryPerformance]:
        """获取慢查询"""
        with self._lock:
            return sorted(
                self._slow_queries[-limit:], 
                key=lambda x: x.execution_time, 
                reverse=True
            )
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """获取查询统计"""
        with self._lock:
            if not self._query_history:
                return {}
            
            execution_times = [q.execution_time for q in self._query_history]
            
            return {
                'total_queries': len(self._query_history),
                'slow_queries': len(self._slow_queries),
                'average_execution_time': statistics.mean(execution_times),
                'median_execution_time': statistics.median(execution_times),
                'max_execution_time': max(execution_times),
                'min_execution_time': min(execution_times),
                'p95_execution_time': statistics.quantiles(execution_times, n=20)[18] if len(execution_times) > 20 else max(execution_times),
                'queries_per_second': len(self._query_history) / 3600 if self._query_history else 0
            }
    
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """分析查询模式"""
        with self._lock:
            patterns = {
                'most_frequent_queries': {},
                'slowest_query_types': {},
                'peak_hours': {},
                'index_usage_patterns': {}
            }
            
            # 分析最频繁的查询
            query_counts = defaultdict(int)
            for query in self._query_history:
                query_hash = hash(query.sql[:100])  # 使用前100个字符作为标识
                query_counts[query_hash] += 1
            
            patterns['most_frequent_queries'] = dict(
                sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            
            # 分析最慢的查询类型
            slow_query_types = defaultdict(list)
            for query in self._slow_queries:
                query_type = query.sql.strip().split()[0].upper()
                slow_query_types[query_type].append(query.execution_time)
            
            for query_type, times in slow_query_types.items():
                patterns['slowest_query_types'][query_type] = {
                    'count': len(times),
                    'average_time': statistics.mean(times),
                    'max_time': max(times)
                }
            
            return patterns


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._resource_history: Dict[ResourceType, List[ResourceUsage]] = defaultdict(list)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._monitoring:
            try:
                self._collect_resource_usage()
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _collect_resource_usage(self) -> None:
        """收集资源使用情况"""
        now = datetime.now()
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage = ResourceUsage(
                resource_type=ResourceType.CPU,
                current_usage=cpu_percent,
                max_usage=100.0,
                average_usage=cpu_percent,
                peak_usage=cpu_percent,
                start_time=now,
                end_time=now,
                unit="percent"
            )
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usage = ResourceUsage(
                resource_type=ResourceType.MEMORY,
                current_usage=memory.used,
                max_usage=memory.total,
                average_usage=memory.used,
                peak_usage=memory.used,
                start_time=now,
                end_time=now,
                unit="bytes"
            )
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_usage = ResourceUsage(
                resource_type=ResourceType.DISK,
                current_usage=disk.used,
                max_usage=disk.total,
                average_usage=disk.used,
                peak_usage=disk.used,
                start_time=now,
                end_time=now,
                unit="bytes"
            )
            
            with self._lock:
                self._resource_history[ResourceType.CPU].append(cpu_usage)
                self._resource_history[ResourceType.MEMORY].append(memory_usage)
                self._resource_history[ResourceType.DISK].append(disk_usage)
                
                # 限制历史记录大小
                for resource_type in self._resource_history:
                    if len(self._resource_history[resource_type]) > 1000:
                        self._resource_history[resource_type] = self._resource_history[resource_type][-500:]
        
        except Exception as e:
            logger.error(f"Failed to collect resource usage: {e}")
    
    def get_current_usage(self) -> Dict[ResourceType, ResourceUsage]:
        """获取当前资源使用情况"""
        with self._lock:
            current_usage = {}
            for resource_type, history in self._resource_history.items():
                if history:
                    current_usage[resource_type] = history[-1]
            return current_usage
    
    def get_resource_statistics(self, resource_type: ResourceType, 
                               hours: int = 24) -> Dict[str, Any]:
        """获取资源统计信息"""
        with self._lock:
            history = self._resource_history.get(resource_type, [])
            if not history:
                return {}
            
            # 过滤时间范围
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_history = [usage for usage in history if usage.start_time >= cutoff_time]
            
            if not recent_history:
                return {}
            
            usage_values = [usage.usage_percentage for usage in recent_history]
            
            return {
                'resource_type': resource_type.value,
                'sample_count': len(recent_history),
                'current_usage': recent_history[-1].usage_percentage,
                'average_usage': statistics.mean(usage_values),
                'max_usage': max(usage_values),
                'min_usage': min(usage_values),
                'p95_usage': statistics.quantiles(usage_values, n=20)[18] if len(usage_values) > 20 else max(usage_values),
                'critical_periods': len([u for u in usage_values if u > 90]),
                'warning_periods': len([u for u in usage_values if u > 75])
            }
    
    def check_resource_alerts(self) -> List[Dict[str, Any]]:
        """检查资源警报"""
        alerts = []
        current_usage = self.get_current_usage()
        
        for resource_type, usage in current_usage.items():
            if usage.is_critical:
                alerts.append({
                    'level': 'critical',
                    'resource_type': resource_type.value,
                    'usage_percentage': usage.usage_percentage,
                    'message': f"{resource_type.value.upper()} usage is critical: {usage.usage_percentage:.1f}%"
                })
            elif usage.is_warning:
                alerts.append({
                    'level': 'warning',
                    'resource_type': resource_type.value,
                    'usage_percentage': usage.usage_percentage,
                    'message': f"{resource_type.value.upper()} usage is high: {usage.usage_percentage:.1f}%"
                })
        
        return alerts


class OptimizationEngine:
    """优化引擎"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._recommendations: List[OptimizationRecommendation] = []
    
    def analyze_query_performance(self, query_monitor: QueryPerformanceMonitor) -> List[OptimizationRecommendation]:
        """分析查询性能并生成建议"""
        recommendations = []
        
        # 获取慢查询
        slow_queries = query_monitor.get_slow_queries(50)
        query_stats = query_monitor.get_query_statistics()
        
        # 分析慢查询
        if slow_queries:
            # 建议添加索引
            for query in slow_queries[:10]:  # 只分析前10个最慢的查询
                if query.efficiency_ratio and query.efficiency_ratio < 0.1:
                    recommendations.append(OptimizationRecommendation(
                        optimization_type=OptimizationType.INDEX,
                        title=f"Add index for slow query",
                        description=f"Query with {query.execution_time:.2f}s execution time has low efficiency ratio",
                        impact="High - Can significantly reduce query time",
                        effort="Medium - Requires index creation",
                        priority=8,
                        action="Create appropriate indexes for frequently queried columns",
                        sql_commands=[f"-- Analyze query: {query.sql[:100]}..."],
                        expected_improvement=70.0,
                        estimated_time_saving=query.execution_time * 0.7
                    ))
        
        # 分析查询缓存
        if query_stats.get('total_queries', 0) > 1000 and not self.config.enable_query_cache:
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.CACHE,
                title="Enable query result caching",
                description="High query volume detected, caching can improve performance",
                impact="Medium - Reduces database load for repeated queries",
                effort="Low - Configuration change",
                priority=6,
                action="Enable query result caching",
                config_changes={'enable_query_cache': True},
                expected_improvement=30.0
            ))
        
        # 分析连接池
        avg_time = query_stats.get('average_execution_time', 0)
        if avg_time > 0.5:
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.CONNECTION,
                title="Optimize connection pool settings",
                description="High average query time may indicate connection pool issues",
                impact="Medium - Improves connection management",
                effort="Low - Configuration change",
                priority=5,
                action="Increase connection pool size and optimize timeout settings",
                config_changes={
                    'connection_pool_size': self.config.connection_pool_size * 2,
                    'connection_pool_timeout': 60
                },
                expected_improvement=20.0
            ))
        
        return recommendations
    
    def analyze_resource_usage(self, resource_monitor: ResourceMonitor) -> List[OptimizationRecommendation]:
        """分析资源使用并生成建议"""
        recommendations = []
        
        # 检查内存使用
        memory_stats = resource_monitor.get_resource_statistics(ResourceType.MEMORY)
        if memory_stats and memory_stats.get('average_usage', 0) > 80:
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.MEMORY,
                title="Optimize memory usage",
                description=f"High memory usage detected: {memory_stats['average_usage']:.1f}%",
                impact="High - Prevents out of memory errors",
                effort="Medium - Requires code optimization",
                priority=9,
                action="Implement memory optimization strategies",
                expected_improvement=25.0,
                risk_level="medium"
            ))
        
        # 检查CPU使用
        cpu_stats = resource_monitor.get_resource_statistics(ResourceType.CPU)
        if cpu_stats and cpu_stats.get('average_usage', 0) > 75:
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.QUERY,
                title="Optimize CPU-intensive operations",
                description=f"High CPU usage detected: {cpu_stats['average_usage']:.1f}%",
                impact="High - Improves overall system performance",
                effort="High - Requires algorithm optimization",
                priority=8,
                action="Profile and optimize CPU-intensive database operations",
                expected_improvement=30.0
            ))
        
        # 检查磁盘使用
        disk_stats = resource_monitor.get_resource_statistics(ResourceType.DISK)
        if disk_stats and disk_stats.get('current_usage', 0) > 85:
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.VACUUM,
                title="Clean up disk space",
                description=f"High disk usage detected: {disk_stats['current_usage']:.1f}%",
                impact="Medium - Prevents disk space issues",
                effort="Low - Run cleanup commands",
                priority=7,
                action="Run database vacuum and cleanup operations",
                sql_commands=["VACUUM ANALYZE;", "REINDEX DATABASE;"],
                expected_improvement=15.0
            ))
        
        return recommendations
    
    def generate_batch_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """生成批处理优化建议"""
        recommendations = []
        
        if not self.config.enable_batch_optimization:
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.BATCH,
                title="Enable batch processing optimization",
                description="Batch processing can significantly improve performance for bulk operations",
                impact="High - Reduces database round trips",
                effort="Medium - Requires code changes",
                priority=7,
                action="Implement batch processing for bulk operations",
                config_changes={'enable_batch_optimization': True},
                expected_improvement=50.0
            ))
        
        return recommendations
    
    def get_all_recommendations(self) -> List[OptimizationRecommendation]:
        """获取所有建议"""
        return sorted(self._recommendations, key=lambda x: x.priority, reverse=True)
    
    def add_recommendation(self, recommendation: OptimizationRecommendation) -> None:
        """添加建议"""
        self._recommendations.append(recommendation)
    
    def clear_recommendations(self) -> None:
        """清空建议"""
        self._recommendations.clear()


class OptimizationManager:
    """优化管理器"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.query_monitor = QueryPerformanceMonitor(self.config)
        self.resource_monitor = ResourceMonitor(self.config)
        self.optimization_engine = OptimizationEngine(self.config)
        
        # 统计信息
        self._stats = {
            'optimizations_applied': 0,
            'recommendations_generated': 0,
            'performance_improvements': 0,
            'errors': 0
        }
        
        # 启动资源监控
        if self.config.enable_resource_monitoring:
            self.resource_monitor.start_monitoring()
    
    def record_query_performance(self, query_performance: QueryPerformance) -> None:
        """记录查询性能"""
        if self.config.enable_query_monitoring:
            self.query_monitor.record_query(query_performance)
    
    def analyze_performance(self) -> List[OptimizationRecommendation]:
        """分析性能并生成建议"""
        try:
            recommendations = []
            
            # 分析查询性能
            if self.config.enable_query_monitoring:
                query_recommendations = self.optimization_engine.analyze_query_performance(self.query_monitor)
                recommendations.extend(query_recommendations)
            
            # 分析资源使用
            if self.config.enable_resource_monitoring:
                resource_recommendations = self.optimization_engine.analyze_resource_usage(self.resource_monitor)
                recommendations.extend(resource_recommendations)
            
            # 生成批处理建议
            batch_recommendations = self.optimization_engine.generate_batch_optimization_recommendations()
            recommendations.extend(batch_recommendations)
            
            # 更新统计信息
            self._stats['recommendations_generated'] += len(recommendations)
            
            # 发布事件
            emit_business_event(
                EventType.OPTIMIZATION_ANALYSIS_COMPLETED,
                "optimization_management",
                data={
                    'recommendations_count': len(recommendations),
                    'analysis_time': datetime.now().isoformat()
                }
            )
            
            return recommendations
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to analyze performance: {e}")
            raise OptimizationError(f"Performance analysis failed: {e}")
    
    def apply_optimization(self, recommendation: OptimizationRecommendation, 
                          session: Optional[Session] = None) -> bool:
        """应用优化建议"""
        try:
            logger.info(f"Applying optimization: {recommendation.title}")
            
            # 执行SQL命令
            if recommendation.sql_commands and session:
                for sql_command in recommendation.sql_commands:
                    if not sql_command.strip().startswith('--'):  # 跳过注释
                        session.execute(text(sql_command))
                session.commit()
            
            # 应用配置变更
            if recommendation.config_changes:
                for key, value in recommendation.config_changes.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        logger.info(f"Updated config: {key} = {value}")
            
            # 更新统计信息
            self._stats['optimizations_applied'] += 1
            if recommendation.expected_improvement:
                self._stats['performance_improvements'] += recommendation.expected_improvement
            
            # 发布事件
            emit_business_event(
                EventType.OPTIMIZATION_APPLIED,
                "optimization_management",
                data=recommendation.to_dict()
            )
            
            logger.info(f"Successfully applied optimization: {recommendation.title}")
            return True
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to apply optimization {recommendation.title}: {e}")
            return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'query_statistics': self.query_monitor.get_query_statistics(),
            'slow_queries': [q.to_dict() for q in self.query_monitor.get_slow_queries(10)],
            'resource_usage': {},
            'resource_alerts': self.resource_monitor.check_resource_alerts(),
            'recommendations': [r.to_dict() for r in self.optimization_engine.get_all_recommendations()],
            'optimization_statistics': self._stats
        }
        
        # 添加资源使用统计
        for resource_type in ResourceType:
            stats = self.resource_monitor.get_resource_statistics(resource_type)
            if stats:
                report['resource_usage'][resource_type.value] = stats
        
        return report
    
    def cleanup_old_data(self, days: int = 7) -> None:
        """清理旧数据"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # 清理查询历史
        with self.query_monitor._lock:
            self.query_monitor._query_history = [
                q for q in self.query_monitor._query_history 
                if q.timestamp >= cutoff_time
            ]
            self.query_monitor._slow_queries = [
                q for q in self.query_monitor._slow_queries 
                if q.timestamp >= cutoff_time
            ]
        
        # 清理资源历史
        with self.resource_monitor._lock:
            for resource_type in self.resource_monitor._resource_history:
                self.resource_monitor._resource_history[resource_type] = [
                    usage for usage in self.resource_monitor._resource_history[resource_type]
                    if usage.start_time >= cutoff_time
                ]
        
        logger.info(f"Cleaned up data older than {days} days")
    
    def shutdown(self) -> None:
        """关闭优化管理器"""
        self.resource_monitor.stop_monitoring()
        logger.info("Optimization manager shutdown")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()


# 性能监控装饰器
def monitor_performance(query_id: Optional[str] = None):
    """性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # 记录性能
                performance = QueryPerformance(
                    query_id=query_id or func.__name__,
                    sql=f"Function: {func.__name__}",
                    execution_time=execution_time
                )
                
                # 获取默认优化管理器并记录性能
                manager = get_default_optimization_manager()
                if manager:
                    manager.record_query_performance(performance)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


# 全局优化管理器
_default_optimization_manager: Optional[OptimizationManager] = None


def initialize_optimization(config: Optional[OptimizationConfig] = None) -> OptimizationManager:
    """初始化优化管理器"""
    global _default_optimization_manager
    _default_optimization_manager = OptimizationManager(config)
    return _default_optimization_manager


def get_default_optimization_manager() -> Optional[OptimizationManager]:
    """获取默认优化管理器"""
    return _default_optimization_manager


# 便捷函数
def record_query_performance(query_performance: QueryPerformance) -> None:
    """记录查询性能"""
    manager = get_default_optimization_manager()
    if not manager:
        manager = initialize_optimization()
    
    manager.record_query_performance(query_performance)


def analyze_performance() -> List[OptimizationRecommendation]:
    """分析性能"""
    manager = get_default_optimization_manager()
    if not manager:
        manager = initialize_optimization()
    
    return manager.analyze_performance()


def apply_optimization(recommendation: OptimizationRecommendation, 
                     session: Optional[Session] = None) -> bool:
    """应用优化"""
    manager = get_default_optimization_manager()
    if not manager:
        manager = initialize_optimization()
    
    return manager.apply_optimization(recommendation, session)


def get_performance_report() -> Dict[str, Any]:
    """获取性能报告"""
    manager = get_default_optimization_manager()
    if manager:
        return manager.get_performance_report()
    return {}


def get_optimization_statistics() -> Dict[str, Any]:
    """获取优化统计"""
    manager = get_default_optimization_manager()
    if manager:
        return manager.get_statistics()
    return {}


def cleanup_optimization_data(days: int = 7) -> None:
    """清理优化数据"""
    manager = get_default_optimization_manager()
    if manager:
        manager.cleanup_old_data(days)


# 导出所有类和函数
__all__ = [
    "OptimizationType",
    "PerformanceLevel",
    "OptimizationStrategy",
    "ResourceType",
    "QueryPerformance",
    "ResourceUsage",
    "OptimizationRecommendation",
    "OptimizationConfig",
    "OptimizationError",
    "PerformanceMonitoringError",
    "ResourceMonitoringError",
    "QueryPerformanceMonitor",
    "ResourceMonitor",
    "OptimizationEngine",
    "OptimizationManager",
    "monitor_performance",
    "initialize_optimization",
    "get_default_optimization_manager",
    "record_query_performance",
    "analyze_performance",
    "apply_optimization",
    "get_performance_report",
    "get_optimization_statistics",
    "cleanup_optimization_data"
]