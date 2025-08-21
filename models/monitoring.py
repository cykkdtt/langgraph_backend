"""模型监控系统模块

本模块提供数据库性能监控、健康检查和告警功能。
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
import queue
import json
import psutil
from collections import defaultdict, deque
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
    from sqlalchemy.pool import Pool
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
    Pool = None
    SQLALCHEMY_AVAILABLE = False

# 导入项目模块
from .events import EventType, emit_business_event
from .validation import validate_data


logger = logging.getLogger(__name__)


T = TypeVar('T')


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"                    # 计数器
    GAUGE = "gauge"                        # 仪表盘
    HISTOGRAM = "histogram"                # 直方图
    SUMMARY = "summary"                    # 摘要
    TIMER = "timer"                        # 计时器


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"                          # 信息
    WARNING = "warning"                    # 警告
    ERROR = "error"                        # 错误
    CRITICAL = "critical"                  # 严重


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"                    # 健康
    DEGRADED = "degraded"                  # 降级
    UNHEALTHY = "unhealthy"                # 不健康
    UNKNOWN = "unknown"                    # 未知


class MonitoringMode(Enum):
    """监控模式枚举"""
    PASSIVE = "passive"                    # 被动监控
    ACTIVE = "active"                      # 主动监控
    HYBRID = "hybrid"                      # 混合监控


@dataclass
class Metric:
    """指标"""
    name: str                              # 指标名称
    metric_type: MetricType                # 指标类型
    value: Union[int, float]               # 指标值
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    
    # 标签和元数据
    labels: Dict[str, str] = field(default_factory=dict)  # 标签
    unit: Optional[str] = None             # 单位
    description: Optional[str] = None      # 描述
    
    # 统计信息（用于HISTOGRAM和SUMMARY）
    count: Optional[int] = None            # 计数
    sum: Optional[float] = None            # 总和
    min_value: Optional[float] = None      # 最小值
    max_value: Optional[float] = None      # 最大值
    percentiles: Optional[Dict[str, float]] = None  # 百分位数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'unit': self.unit,
            'description': self.description,
            'count': self.count,
            'sum': self.sum,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'percentiles': self.percentiles
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metric':
        """从字典创建"""
        return cls(
            name=data['name'],
            metric_type=MetricType(data['type']),
            value=data['value'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            labels=data.get('labels', {}),
            unit=data.get('unit'),
            description=data.get('description'),
            count=data.get('count'),
            sum=data.get('sum'),
            min_value=data.get('min_value'),
            max_value=data.get('max_value'),
            percentiles=data.get('percentiles')
        )


@dataclass
class Alert:
    """告警"""
    id: str                                # 告警ID
    name: str                              # 告警名称
    level: AlertLevel                      # 告警级别
    message: str                           # 告警消息
    
    # 触发信息
    metric_name: str                       # 触发指标
    threshold: Union[int, float]           # 阈值
    actual_value: Union[int, float]        # 实际值
    
    # 时间信息
    triggered_at: datetime = field(default_factory=datetime.now)  # 触发时间
    resolved_at: Optional[datetime] = None # 解决时间
    
    # 状态信息
    is_active: bool = True                 # 是否活跃
    acknowledgment: Optional[str] = None   # 确认信息
    
    # 元数据
    labels: Dict[str, str] = field(default_factory=dict)  # 标签
    annotations: Dict[str, str] = field(default_factory=dict)  # 注释
    
    @property
    def duration(self) -> Optional[timedelta]:
        """告警持续时间"""
        if self.resolved_at:
            return self.resolved_at - self.triggered_at
        return datetime.now() - self.triggered_at
    
    def resolve(self, message: Optional[str] = None) -> None:
        """解决告警"""
        self.is_active = False
        self.resolved_at = datetime.now()
        if message:
            self.annotations['resolution'] = message
    
    def acknowledge(self, message: str) -> None:
        """确认告警"""
        self.acknowledgment = message
        self.annotations['acknowledged_at'] = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level.value,
            'message': self.message,
            'metric_name': self.metric_name,
            'threshold': self.threshold,
            'actual_value': self.actual_value,
            'triggered_at': self.triggered_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'is_active': self.is_active,
            'acknowledgment': self.acknowledgment,
            'labels': self.labels,
            'annotations': self.annotations,
            'duration_seconds': self.duration.total_seconds() if self.duration else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """从字典创建"""
        alert = cls(
            id=data['id'],
            name=data['name'],
            level=AlertLevel(data['level']),
            message=data['message'],
            metric_name=data['metric_name'],
            threshold=data['threshold'],
            actual_value=data['actual_value'],
            triggered_at=datetime.fromisoformat(data['triggered_at']),
            resolved_at=datetime.fromisoformat(data['resolved_at']) if data.get('resolved_at') else None,
            is_active=data['is_active'],
            acknowledgment=data.get('acknowledgment'),
            labels=data.get('labels', {}),
            annotations=data.get('annotations', {})
        )
        return alert


@dataclass
class HealthCheck:
    """健康检查"""
    name: str                              # 检查名称
    status: HealthStatus                   # 健康状态
    message: str                           # 状态消息
    
    # 检查信息
    check_type: str                        # 检查类型
    endpoint: Optional[str] = None         # 检查端点
    
    # 时间信息
    checked_at: datetime = field(default_factory=datetime.now)  # 检查时间
    response_time: Optional[float] = None  # 响应时间（毫秒）
    
    # 详细信息
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息
    error: Optional[str] = None            # 错误信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'check_type': self.check_type,
            'endpoint': self.endpoint,
            'checked_at': self.checked_at.isoformat(),
            'response_time': self.response_time,
            'details': self.details,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HealthCheck':
        """从字典创建"""
        return cls(
            name=data['name'],
            status=HealthStatus(data['status']),
            message=data['message'],
            check_type=data['check_type'],
            endpoint=data.get('endpoint'),
            checked_at=datetime.fromisoformat(data['checked_at']),
            response_time=data.get('response_time'),
            details=data.get('details', {}),
            error=data.get('error')
        )


@dataclass
class MonitoringConfig:
    """监控配置"""
    # 基本配置
    enabled: bool = True                   # 是否启用监控
    mode: MonitoringMode = MonitoringMode.HYBRID  # 监控模式
    
    # 采集配置
    collection_interval: int = 60          # 采集间隔（秒）
    metric_retention_days: int = 30        # 指标保留天数
    
    # 数据库监控
    monitor_connections: bool = True       # 监控连接
    monitor_queries: bool = True           # 监控查询
    monitor_locks: bool = True             # 监控锁
    monitor_transactions: bool = True      # 监控事务
    
    # 系统监控
    monitor_cpu: bool = True               # 监控CPU
    monitor_memory: bool = True            # 监控内存
    monitor_disk: bool = True              # 监控磁盘
    monitor_network: bool = True           # 监控网络
    
    # 健康检查
    health_check_interval: int = 30        # 健康检查间隔（秒）
    health_check_timeout: int = 10         # 健康检查超时（秒）
    
    # 告警配置
    enable_alerts: bool = True             # 启用告警
    alert_cooldown: int = 300              # 告警冷却时间（秒）
    
    # 阈值配置
    cpu_warning_threshold: float = 80.0    # CPU警告阈值（%）
    cpu_critical_threshold: float = 95.0   # CPU严重阈值（%）
    memory_warning_threshold: float = 80.0 # 内存警告阈值（%）
    memory_critical_threshold: float = 95.0# 内存严重阈值（%）
    disk_warning_threshold: float = 80.0   # 磁盘警告阈值（%）
    disk_critical_threshold: float = 95.0  # 磁盘严重阈值（%）
    
    # 数据库阈值
    connection_warning_threshold: int = 80 # 连接警告阈值
    connection_critical_threshold: int = 95# 连接严重阈值
    slow_query_threshold: float = 1.0      # 慢查询阈值（秒）
    
    # 存储配置
    storage_backend: str = "memory"        # 存储后端
    storage_config: Dict[str, Any] = field(default_factory=dict)  # 存储配置


class MonitoringError(Exception):
    """监控错误"""
    pass


class MetricCollectionError(MonitoringError):
    """指标收集错误"""
    pass


class AlertingError(MonitoringError):
    """告警错误"""
    pass


class HealthCheckError(MonitoringError):
    """健康检查错误"""
    pass


class MetricCollector:
    """指标收集器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._metrics: deque = deque(maxlen=10000)  # 指标缓存
        self._lock = threading.Lock()
    
    def collect_metric(self, metric: Metric) -> None:
        """收集指标"""
        with self._lock:
            self._metrics.append(metric)
    
    def collect_database_metrics(self, engine: Engine) -> List[Metric]:
        """收集数据库指标"""
        metrics = []
        
        try:
            # 连接池指标
            if hasattr(engine.pool, 'size'):
                pool_size = engine.pool.size()
                checked_out = engine.pool.checkedout()
                
                metrics.append(Metric(
                    name="db_pool_size",
                    metric_type=MetricType.GAUGE,
                    value=pool_size,
                    labels={'engine': str(engine.url)}
                ))
                
                metrics.append(Metric(
                    name="db_pool_checked_out",
                    metric_type=MetricType.GAUGE,
                    value=checked_out,
                    labels={'engine': str(engine.url)}
                ))
                
                metrics.append(Metric(
                    name="db_pool_utilization",
                    metric_type=MetricType.GAUGE,
                    value=(checked_out / pool_size * 100) if pool_size > 0 else 0,
                    unit="percent",
                    labels={'engine': str(engine.url)}
                ))
            
            # 数据库统计信息
            with engine.connect() as conn:
                # 活跃连接数
                try:
                    result = conn.execute(text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"))
                    active_connections = result.scalar()
                    
                    metrics.append(Metric(
                        name="db_active_connections",
                        metric_type=MetricType.GAUGE,
                        value=active_connections,
                        labels={'engine': str(engine.url)}
                    ))
                except Exception:
                    pass  # 不同数据库的查询语法不同
                
                # 数据库大小
                try:
                    result = conn.execute(text("SELECT pg_database_size(current_database())"))
                    db_size = result.scalar()
                    
                    metrics.append(Metric(
                        name="db_size_bytes",
                        metric_type=MetricType.GAUGE,
                        value=db_size,
                        unit="bytes",
                        labels={'engine': str(engine.url)}
                    ))
                except Exception:
                    pass
        
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")
            raise MetricCollectionError(f"Database metric collection failed: {e}")
        
        return metrics
    
    def collect_system_metrics(self) -> List[Metric]:
        """收集系统指标"""
        metrics = []
        
        try:
            # CPU指标
            if self.config.monitor_cpu:
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_count = psutil.cpu_count()
                
                metrics.append(Metric(
                    name="system_cpu_usage",
                    metric_type=MetricType.GAUGE,
                    value=cpu_percent,
                    unit="percent"
                ))
                
                metrics.append(Metric(
                    name="system_cpu_count",
                    metric_type=MetricType.GAUGE,
                    value=cpu_count
                ))
            
            # 内存指标
            if self.config.monitor_memory:
                memory = psutil.virtual_memory()
                
                metrics.append(Metric(
                    name="system_memory_usage",
                    metric_type=MetricType.GAUGE,
                    value=memory.percent,
                    unit="percent"
                ))
                
                metrics.append(Metric(
                    name="system_memory_total",
                    metric_type=MetricType.GAUGE,
                    value=memory.total,
                    unit="bytes"
                ))
                
                metrics.append(Metric(
                    name="system_memory_available",
                    metric_type=MetricType.GAUGE,
                    value=memory.available,
                    unit="bytes"
                ))
            
            # 磁盘指标
            if self.config.monitor_disk:
                disk = psutil.disk_usage('/')
                
                metrics.append(Metric(
                    name="system_disk_usage",
                    metric_type=MetricType.GAUGE,
                    value=(disk.used / disk.total * 100),
                    unit="percent"
                ))
                
                metrics.append(Metric(
                    name="system_disk_total",
                    metric_type=MetricType.GAUGE,
                    value=disk.total,
                    unit="bytes"
                ))
                
                metrics.append(Metric(
                    name="system_disk_free",
                    metric_type=MetricType.GAUGE,
                    value=disk.free,
                    unit="bytes"
                ))
            
            # 网络指标
            if self.config.monitor_network:
                network = psutil.net_io_counters()
                
                metrics.append(Metric(
                    name="system_network_bytes_sent",
                    metric_type=MetricType.COUNTER,
                    value=network.bytes_sent,
                    unit="bytes"
                ))
                
                metrics.append(Metric(
                    name="system_network_bytes_recv",
                    metric_type=MetricType.COUNTER,
                    value=network.bytes_recv,
                    unit="bytes"
                ))
        
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            raise MetricCollectionError(f"System metric collection failed: {e}")
        
        return metrics
    
    def get_metrics(self, name_pattern: Optional[str] = None, 
                   since: Optional[datetime] = None) -> List[Metric]:
        """获取指标"""
        with self._lock:
            metrics = list(self._metrics)
        
        # 过滤指标
        if name_pattern:
            import re
            pattern = re.compile(name_pattern)
            metrics = [m for m in metrics if pattern.match(m.name)]
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        return metrics
    
    def clear_old_metrics(self) -> None:
        """清理旧指标"""
        cutoff_time = datetime.now() - timedelta(days=self.config.metric_retention_days)
        
        with self._lock:
            # 由于使用deque，这里只是示例，实际实现可能需要更复杂的存储
            self._metrics = deque(
                [m for m in self._metrics if m.timestamp >= cutoff_time],
                maxlen=self._metrics.maxlen
            )


class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._alerts: Dict[str, Alert] = {}  # 活跃告警
        self._alert_history: List[Alert] = []  # 告警历史
        self._last_alert_time: Dict[str, datetime] = {}  # 最后告警时间
        self._lock = threading.Lock()
    
    def check_thresholds(self, metrics: List[Metric]) -> List[Alert]:
        """检查阈值并生成告警"""
        new_alerts = []
        
        for metric in metrics:
            alerts = self._check_metric_thresholds(metric)
            new_alerts.extend(alerts)
        
        return new_alerts
    
    def _check_metric_thresholds(self, metric: Metric) -> List[Alert]:
        """检查单个指标的阈值"""
        alerts = []
        
        # CPU告警
        if metric.name == "system_cpu_usage":
            if metric.value >= self.config.cpu_critical_threshold:
                alert = self._create_alert(
                    f"cpu_critical_{metric.timestamp.timestamp()}",
                    "CPU Critical",
                    AlertLevel.CRITICAL,
                    f"CPU usage is {metric.value:.1f}%, exceeding critical threshold of {self.config.cpu_critical_threshold}%",
                    metric.name,
                    self.config.cpu_critical_threshold,
                    metric.value
                )
                alerts.append(alert)
            elif metric.value >= self.config.cpu_warning_threshold:
                alert = self._create_alert(
                    f"cpu_warning_{metric.timestamp.timestamp()}",
                    "CPU Warning",
                    AlertLevel.WARNING,
                    f"CPU usage is {metric.value:.1f}%, exceeding warning threshold of {self.config.cpu_warning_threshold}%",
                    metric.name,
                    self.config.cpu_warning_threshold,
                    metric.value
                )
                alerts.append(alert)
        
        # 内存告警
        elif metric.name == "system_memory_usage":
            if metric.value >= self.config.memory_critical_threshold:
                alert = self._create_alert(
                    f"memory_critical_{metric.timestamp.timestamp()}",
                    "Memory Critical",
                    AlertLevel.CRITICAL,
                    f"Memory usage is {metric.value:.1f}%, exceeding critical threshold of {self.config.memory_critical_threshold}%",
                    metric.name,
                    self.config.memory_critical_threshold,
                    metric.value
                )
                alerts.append(alert)
            elif metric.value >= self.config.memory_warning_threshold:
                alert = self._create_alert(
                    f"memory_warning_{metric.timestamp.timestamp()}",
                    "Memory Warning",
                    AlertLevel.WARNING,
                    f"Memory usage is {metric.value:.1f}%, exceeding warning threshold of {self.config.memory_warning_threshold}%",
                    metric.name,
                    self.config.memory_warning_threshold,
                    metric.value
                )
                alerts.append(alert)
        
        # 磁盘告警
        elif metric.name == "system_disk_usage":
            if metric.value >= self.config.disk_critical_threshold:
                alert = self._create_alert(
                    f"disk_critical_{metric.timestamp.timestamp()}",
                    "Disk Critical",
                    AlertLevel.CRITICAL,
                    f"Disk usage is {metric.value:.1f}%, exceeding critical threshold of {self.config.disk_critical_threshold}%",
                    metric.name,
                    self.config.disk_critical_threshold,
                    metric.value
                )
                alerts.append(alert)
            elif metric.value >= self.config.disk_warning_threshold:
                alert = self._create_alert(
                    f"disk_warning_{metric.timestamp.timestamp()}",
                    "Disk Warning",
                    AlertLevel.WARNING,
                    f"Disk usage is {metric.value:.1f}%, exceeding warning threshold of {self.config.disk_warning_threshold}%",
                    metric.name,
                    self.config.disk_warning_threshold,
                    metric.value
                )
                alerts.append(alert)
        
        # 数据库连接告警
        elif metric.name == "db_pool_utilization":
            if metric.value >= self.config.connection_critical_threshold:
                alert = self._create_alert(
                    f"db_connection_critical_{metric.timestamp.timestamp()}",
                    "Database Connection Critical",
                    AlertLevel.CRITICAL,
                    f"Database connection utilization is {metric.value:.1f}%, exceeding critical threshold of {self.config.connection_critical_threshold}%",
                    metric.name,
                    self.config.connection_critical_threshold,
                    metric.value
                )
                alerts.append(alert)
            elif metric.value >= self.config.connection_warning_threshold:
                alert = self._create_alert(
                    f"db_connection_warning_{metric.timestamp.timestamp()}",
                    "Database Connection Warning",
                    AlertLevel.WARNING,
                    f"Database connection utilization is {metric.value:.1f}%, exceeding warning threshold of {self.config.connection_warning_threshold}%",
                    metric.name,
                    self.config.connection_warning_threshold,
                    metric.value
                )
                alerts.append(alert)
        
        return alerts
    
    def _create_alert(self, alert_id: str, name: str, level: AlertLevel, 
                     message: str, metric_name: str, threshold: Union[int, float], 
                     actual_value: Union[int, float]) -> Alert:
        """创建告警"""
        # 检查冷却时间
        if self._is_in_cooldown(alert_id):
            return None
        
        alert = Alert(
            id=alert_id,
            name=name,
            level=level,
            message=message,
            metric_name=metric_name,
            threshold=threshold,
            actual_value=actual_value
        )
        
        with self._lock:
            self._alerts[alert_id] = alert
            self._alert_history.append(alert)
            self._last_alert_time[alert_id] = datetime.now()
        
        return alert
    
    def _is_in_cooldown(self, alert_id: str) -> bool:
        """检查是否在冷却时间内"""
        last_time = self._last_alert_time.get(alert_id)
        if not last_time:
            return False
        
        cooldown_period = timedelta(seconds=self.config.alert_cooldown)
        return datetime.now() - last_time < cooldown_period
    
    def resolve_alert(self, alert_id: str, message: Optional[str] = None) -> bool:
        """解决告警"""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and alert.is_active:
                alert.resolve(message)
                return True
        return False
    
    def acknowledge_alert(self, alert_id: str, message: str) -> bool:
        """确认告警"""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and alert.is_active:
                alert.acknowledge(message)
                return True
        return False
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """获取活跃告警"""
        with self._lock:
            alerts = [alert for alert in self._alerts.values() if alert.is_active]
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return alerts
    
    def get_alert_history(self, since: Optional[datetime] = None, 
                         level: Optional[AlertLevel] = None) -> List[Alert]:
        """获取告警历史"""
        with self._lock:
            alerts = list(self._alert_history)
        
        if since:
            alerts = [alert for alert in alerts if alert.triggered_at >= since]
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return alerts


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, HealthCheck] = {}
        self._lock = threading.Lock()
    
    def register_check(self, name: str, check_func: Callable, 
                      check_type: str = "custom") -> None:
        """注册健康检查"""
        self._checks[name] = (check_func, check_type)
    
    def run_checks(self) -> Dict[str, HealthCheck]:
        """运行所有健康检查"""
        results = {}
        
        for name, (check_func, check_type) in self._checks.items():
            try:
                start_time = time.time()
                result = check_func()
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # 转换为毫秒
                
                if isinstance(result, HealthCheck):
                    health_check = result
                    health_check.response_time = response_time
                elif isinstance(result, bool):
                    health_check = HealthCheck(
                        name=name,
                        status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                        message="Check passed" if result else "Check failed",
                        check_type=check_type,
                        response_time=response_time
                    )
                else:
                    health_check = HealthCheck(
                        name=name,
                        status=HealthStatus.HEALTHY,
                        message=str(result),
                        check_type=check_type,
                        response_time=response_time
                    )
                
                results[name] = health_check
                
            except Exception as e:
                health_check = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    check_type=check_type,
                    error=str(e)
                )
                results[name] = health_check
                logger.error(f"Health check '{name}' failed: {e}")
        
        with self._lock:
            self._results.update(results)
        
        return results
    
    def check_database_health(self, engine: Engine) -> HealthCheck:
        """检查数据库健康状态"""
        try:
            start_time = time.time()
            
            with engine.connect() as conn:
                # 执行简单查询
                result = conn.execute(text("SELECT 1"))
                result.scalar()
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database is accessible",
                check_type="database",
                response_time=response_time
            )
            
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
                check_type="database",
                error=str(e)
            )
    
    def get_overall_health(self) -> HealthStatus:
        """获取整体健康状态"""
        with self._lock:
            if not self._results:
                return HealthStatus.UNKNOWN
            
            statuses = [check.status for check in self._results.values()]
            
            if all(status == HealthStatus.HEALTHY for status in statuses):
                return HealthStatus.HEALTHY
            elif any(status == HealthStatus.UNHEALTHY for status in statuses):
                return HealthStatus.UNHEALTHY
            else:
                return HealthStatus.DEGRADED
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        with self._lock:
            results = dict(self._results)
        
        overall_status = self.get_overall_health()
        
        return {
            'overall_status': overall_status.value,
            'checks': {name: check.to_dict() for name, check in results.items()},
            'healthy_count': sum(1 for check in results.values() if check.status == HealthStatus.HEALTHY),
            'unhealthy_count': sum(1 for check in results.values() if check.status == HealthStatus.UNHEALTHY),
            'degraded_count': sum(1 for check in results.values() if check.status == HealthStatus.DEGRADED),
            'total_checks': len(results)
        }


class MonitoringManager:
    """监控管理器"""
    
    def __init__(self, engine: Engine, config: Optional[MonitoringConfig] = None):
        self.engine = engine
        self.config = config or MonitoringConfig()
        
        # 组件初始化
        self.metric_collector = MetricCollector(self.config)
        self.alert_manager = AlertManager(self.config)
        self.health_checker = HealthChecker(self.config)
        
        # 监控线程
        self._monitoring_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False
        
        # 统计信息
        self._stats = {
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'health_checks_performed': 0,
            'uptime_start': datetime.now()
        }
        
        # 注册默认健康检查
        self._register_default_health_checks()
    
    def _register_default_health_checks(self) -> None:
        """注册默认健康检查"""
        # 数据库健康检查
        self.health_checker.register_check(
            "database",
            lambda: self.health_checker.check_database_health(self.engine),
            "database"
        )
        
        # 系统健康检查
        def system_health_check():
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            if cpu_usage > 90 or memory_usage > 90:
                return HealthCheck(
                    name="system",
                    status=HealthStatus.UNHEALTHY,
                    message=f"High resource usage: CPU {cpu_usage}%, Memory {memory_usage}%",
                    check_type="system"
                )
            elif cpu_usage > 70 or memory_usage > 70:
                return HealthCheck(
                    name="system",
                    status=HealthStatus.DEGRADED,
                    message=f"Moderate resource usage: CPU {cpu_usage}%, Memory {memory_usage}%",
                    check_type="system"
                )
            else:
                return HealthCheck(
                    name="system",
                    status=HealthStatus.HEALTHY,
                    message=f"Normal resource usage: CPU {cpu_usage}%, Memory {memory_usage}%",
                    check_type="system"
                )
        
        self.health_checker.register_check("system", system_health_check, "system")
    
    def start_monitoring(self) -> None:
        """启动监控"""
        if self._running:
            logger.warning("Monitoring is already running")
            return
        
        self._running = True
        
        # 启动指标收集线程
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="MonitoringThread",
            daemon=True
        )
        self._monitoring_thread.start()
        
        # 启动健康检查线程
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name="HealthCheckThread",
            daemon=True
        )
        self._health_check_thread.start()
        
        logger.info("Monitoring started")
        
        # 发布事件
        emit_business_event(
            EventType.MONITORING_STARTED,
            "monitoring",
            data={'config': self.config.__dict__}
        )
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        if not self._running:
            logger.warning("Monitoring is not running")
            return
        
        self._running = False
        
        # 等待线程结束
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)
        
        logger.info("Monitoring stopped")
        
        # 发布事件
        emit_business_event(
            EventType.MONITORING_STOPPED,
            "monitoring",
            data={}
        )
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                # 收集指标
                metrics = []
                
                # 收集数据库指标
                if self.config.monitor_connections or self.config.monitor_queries:
                    db_metrics = self.metric_collector.collect_database_metrics(self.engine)
                    metrics.extend(db_metrics)
                
                # 收集系统指标
                system_metrics = self.metric_collector.collect_system_metrics()
                metrics.extend(system_metrics)
                
                # 存储指标
                for metric in metrics:
                    self.metric_collector.collect_metric(metric)
                    self._stats['metrics_collected'] += 1
                
                # 检查告警
                if self.config.enable_alerts:
                    alerts = self.alert_manager.check_thresholds(metrics)
                    for alert in alerts:
                        if alert:  # 可能因为冷却时间返回None
                            self._stats['alerts_triggered'] += 1
                            logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
                            
                            # 发布告警事件
                            emit_business_event(
                                EventType.ALERT_TRIGGERED,
                                "monitoring",
                                data=alert.to_dict()
                            )
                
                # 清理旧指标
                self.metric_collector.clear_old_metrics()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # 等待下一次收集
            time.sleep(self.config.collection_interval)
    
    def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self._running:
            try:
                # 运行健康检查
                self.health_checker.run_checks()
                self._stats['health_checks_performed'] += 1
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
            
            # 等待下一次检查
            time.sleep(self.config.health_check_interval)
    
    def get_metrics(self, name_pattern: Optional[str] = None, 
                   since: Optional[datetime] = None) -> List[Metric]:
        """获取指标"""
        return self.metric_collector.get_metrics(name_pattern, since)
    
    def get_alerts(self, level: Optional[AlertLevel] = None, 
                  active_only: bool = True) -> List[Alert]:
        """获取告警"""
        if active_only:
            return self.alert_manager.get_active_alerts(level)
        else:
            return self.alert_manager.get_alert_history(level=level)
    
    def resolve_alert(self, alert_id: str, message: Optional[str] = None) -> bool:
        """解决告警"""
        success = self.alert_manager.resolve_alert(alert_id, message)
        if success:
            # 发布事件
            emit_business_event(
                EventType.ALERT_RESOLVED,
                "monitoring",
                data={'alert_id': alert_id, 'message': message}
            )
        return success
    
    def acknowledge_alert(self, alert_id: str, message: str) -> bool:
        """确认告警"""
        return self.alert_manager.acknowledge_alert(alert_id, message)
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return self.health_checker.get_health_summary()
    
    def register_health_check(self, name: str, check_func: Callable, 
                             check_type: str = "custom") -> None:
        """注册健康检查"""
        self.health_checker.register_check(name, check_func, check_type)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        uptime = datetime.now() - self._stats['uptime_start']
        
        return {
            'running': self._running,
            'uptime_seconds': uptime.total_seconds(),
            'config': {
                'collection_interval': self.config.collection_interval,
                'health_check_interval': self.config.health_check_interval,
                'alerts_enabled': self.config.enable_alerts
            },
            'statistics': self._stats.copy(),
            'threads': {
                'monitoring_thread_alive': self._monitoring_thread.is_alive() if self._monitoring_thread else False,
                'health_check_thread_alive': self._health_check_thread.is_alive() if self._health_check_thread else False
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()


# 监控装饰器
def monitor_performance(metric_name: Optional[str] = None):
    """性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # 记录性能指标
                name = metric_name or f"function_{func.__name__}_duration"
                
                manager = get_default_monitoring_manager()
                if manager:
                    metric = Metric(
                        name=name,
                        metric_type=MetricType.TIMER,
                        value=execution_time,
                        unit="seconds",
                        labels={
                            'function': func.__name__,
                            'success': str(success)
                        }
                    )
                    
                    if error:
                        metric.labels['error'] = error
                    
                    manager.metric_collector.collect_metric(metric)
            
            return result
        
        return wrapper
    return decorator


# 全局监控管理器
_default_monitoring_manager: Optional[MonitoringManager] = None


def initialize_monitoring(engine: Engine, config: Optional[MonitoringConfig] = None) -> MonitoringManager:
    """初始化监控管理器"""
    global _default_monitoring_manager
    _default_monitoring_manager = MonitoringManager(engine, config)
    return _default_monitoring_manager


def get_default_monitoring_manager() -> Optional[MonitoringManager]:
    """获取默认监控管理器"""
    return _default_monitoring_manager


# 便捷函数
def start_monitoring() -> None:
    """启动监控"""
    manager = get_default_monitoring_manager()
    if manager:
        manager.start_monitoring()


def stop_monitoring() -> None:
    """停止监控"""
    manager = get_default_monitoring_manager()
    if manager:
        manager.stop_monitoring()


def get_metrics(name_pattern: Optional[str] = None, 
               since: Optional[datetime] = None) -> List[Metric]:
    """获取指标"""
    manager = get_default_monitoring_manager()
    if manager:
        return manager.get_metrics(name_pattern, since)
    return []


def get_alerts(level: Optional[AlertLevel] = None, 
              active_only: bool = True) -> List[Alert]:
    """获取告警"""
    manager = get_default_monitoring_manager()
    if manager:
        return manager.get_alerts(level, active_only)
    return []


def resolve_alert(alert_id: str, message: Optional[str] = None) -> bool:
    """解决告警"""
    manager = get_default_monitoring_manager()
    if manager:
        return manager.resolve_alert(alert_id, message)
    return False


def acknowledge_alert(alert_id: str, message: str) -> bool:
    """确认告警"""
    manager = get_default_monitoring_manager()
    if manager:
        return manager.acknowledge_alert(alert_id, message)
    return False


def get_health_status() -> Dict[str, Any]:
    """获取健康状态"""
    manager = get_default_monitoring_manager()
    if manager:
        return manager.get_health_status()
    return {}


def register_health_check(name: str, check_func: Callable, 
                         check_type: str = "custom") -> None:
    """注册健康检查"""
    manager = get_default_monitoring_manager()
    if manager:
        manager.register_health_check(name, check_func, check_type)


def get_monitoring_status() -> Dict[str, Any]:
    """获取监控状态"""
    manager = get_default_monitoring_manager()
    if manager:
        return manager.get_monitoring_status()
    return {}


def get_monitoring_statistics() -> Dict[str, Any]:
    """获取监控统计"""
    manager = get_default_monitoring_manager()
    if manager:
        return manager.get_statistics()
    return {}


# 导出所有类和函数
__all__ = [
    "MetricType",
    "AlertLevel",
    "HealthStatus",
    "MonitoringMode",
    "Metric",
    "Alert",
    "HealthCheck",
    "MonitoringConfig",
    "MonitoringError",
    "MetricCollectionError",
    "AlertingError",
    "HealthCheckError",
    "MetricCollector",
    "AlertManager",
    "HealthChecker",
    "MonitoringManager",
    "monitor_performance",
    "initialize_monitoring",
    "get_default_monitoring_manager",
    "start_monitoring",
    "stop_monitoring",
    "get_metrics",
    "get_alerts",
    "resolve_alert",
    "acknowledge_alert",
    "get_health_status",
    "register_health_check",
    "get_monitoring_status",
    "get_monitoring_statistics"
]