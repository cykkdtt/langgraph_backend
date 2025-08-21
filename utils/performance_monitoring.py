"""性能监控工具

提供API性能监控、指标收集和分析功能。
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestMetrics:
    """请求指标数据类"""
    endpoint: str
    method: str
    status_code: int
    duration: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """系统指标数据类"""
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_available: int
    disk_usage: float
    network_io: Dict[str, int]
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.request_metrics: deque = deque(maxlen=max_metrics)
        self.system_metrics: deque = deque(maxlen=max_metrics)
        self.custom_metrics: deque = deque(maxlen=max_metrics)
        self.lock = threading.RLock()
        self._collecting = False
        self._collection_thread = None
        self._collection_interval = 60  # 秒
    
    def add_request_metric(self, metric: RequestMetrics) -> None:
        """添加请求指标"""
        with self.lock:
            self.request_metrics.append(metric)
    
    def add_system_metric(self, metric: SystemMetrics) -> None:
        """添加系统指标"""
        with self.lock:
            self.system_metrics.append(metric)
    
    def add_custom_metric(self, metric: PerformanceMetric) -> None:
        """添加自定义指标"""
        with self.lock:
            self.custom_metrics.append(metric)
    
    def get_request_metrics(self, limit: Optional[int] = None, 
                          since: Optional[datetime] = None) -> List[RequestMetrics]:
        """获取请求指标"""
        with self.lock:
            metrics = list(self.request_metrics)
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        if limit:
            metrics = metrics[-limit:]
        
        return metrics
    
    def get_system_metrics(self, limit: Optional[int] = None,
                         since: Optional[datetime] = None) -> List[SystemMetrics]:
        """获取系统指标"""
        with self.lock:
            metrics = list(self.system_metrics)
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        if limit:
            metrics = metrics[-limit:]
        
        return metrics
    
    def get_custom_metrics(self, name: Optional[str] = None,
                         limit: Optional[int] = None,
                         since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """获取自定义指标"""
        with self.lock:
            metrics = list(self.custom_metrics)
        
        if name:
            metrics = [m for m in metrics if m.name == name]
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        if limit:
            metrics = metrics[-limit:]
        
        return metrics
    
    def start_collection(self, interval: int = 60) -> None:
        """开始系统指标收集"""
        if self._collecting:
            return
        
        self._collection_interval = interval
        self._collecting = True
        self._collection_thread = threading.Thread(target=self._collect_system_metrics)
        self._collection_thread.daemon = True
        self._collection_thread.start()
        logger.info(f"Started system metrics collection with {interval}s interval")
    
    def stop_collection(self) -> None:
        """停止系统指标收集"""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Stopped system metrics collection")
    
    def _collect_system_metrics(self) -> None:
        """收集系统指标"""
        while self._collecting:
            try:
                # 获取系统指标
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                metric = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used=memory.used,
                    memory_available=memory.available,
                    disk_usage=disk.percent,
                    network_io={
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv,
                        'packets_sent': network.packets_sent,
                        'packets_recv': network.packets_recv
                    }
                )
                
                self.add_system_metric(metric)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(self._collection_interval)
    
    def clear_metrics(self, metric_type: str = "all") -> None:
        """清除指标数据"""
        with self.lock:
            if metric_type in ["all", "request"]:
                self.request_metrics.clear()
            if metric_type in ["all", "system"]:
                self.system_metrics.clear()
            if metric_type in ["all", "custom"]:
                self.custom_metrics.clear()


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def analyze_request_performance(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """分析请求性能"""
        metrics = self.collector.get_request_metrics(since=since)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        # 按端点分组
        endpoint_metrics = defaultdict(list)
        for metric in metrics:
            endpoint_metrics[f"{metric.method} {metric.endpoint}"].append(metric)
        
        analysis = {
            "total_requests": len(metrics),
            "time_range": {
                "start": min(m.timestamp for m in metrics).isoformat(),
                "end": max(m.timestamp for m in metrics).isoformat()
            },
            "overall_stats": self._calculate_stats([m.duration for m in metrics]),
            "endpoints": {}
        }
        
        # 分析每个端点
        for endpoint, endpoint_metrics_list in endpoint_metrics.items():
            durations = [m.duration for m in endpoint_metrics_list]
            status_codes = defaultdict(int)
            errors = []
            
            for metric in endpoint_metrics_list:
                status_codes[metric.status_code] += 1
                if metric.error_message:
                    errors.append(metric.error_message)
            
            analysis["endpoints"][endpoint] = {
                "request_count": len(endpoint_metrics_list),
                "duration_stats": self._calculate_stats(durations),
                "status_codes": dict(status_codes),
                "error_rate": len(errors) / len(endpoint_metrics_list),
                "errors": errors[:10]  # 只显示前10个错误
            }
        
        return analysis
    
    def analyze_system_performance(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """分析系统性能"""
        metrics = self.collector.get_system_metrics(since=since)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]
        disk_values = [m.disk_usage for m in metrics]
        
        return {
            "time_range": {
                "start": min(m.timestamp for m in metrics).isoformat(),
                "end": max(m.timestamp for m in metrics).isoformat()
            },
            "cpu_stats": self._calculate_stats(cpu_values),
            "memory_stats": self._calculate_stats(memory_values),
            "disk_stats": self._calculate_stats(disk_values),
            "current_values": {
                "cpu_percent": metrics[-1].cpu_percent,
                "memory_percent": metrics[-1].memory_percent,
                "disk_usage": metrics[-1].disk_usage
            } if metrics else None
        }
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """获取性能警报"""
        alerts = []
        
        # 检查最近的系统指标
        recent_system = self.collector.get_system_metrics(limit=5)
        if recent_system:
            latest = recent_system[-1]
            
            if latest.cpu_percent > 80:
                alerts.append({
                    "type": "high_cpu",
                    "severity": "warning",
                    "message": f"High CPU usage: {latest.cpu_percent:.1f}%",
                    "timestamp": latest.timestamp.isoformat()
                })
            
            if latest.memory_percent > 85:
                alerts.append({
                    "type": "high_memory",
                    "severity": "warning",
                    "message": f"High memory usage: {latest.memory_percent:.1f}%",
                    "timestamp": latest.timestamp.isoformat()
                })
            
            if latest.disk_usage > 90:
                alerts.append({
                    "type": "high_disk",
                    "severity": "critical",
                    "message": f"High disk usage: {latest.disk_usage:.1f}%",
                    "timestamp": latest.timestamp.isoformat()
                })
        
        # 检查慢请求
        recent_requests = self.collector.get_request_metrics(limit=100)
        slow_requests = [r for r in recent_requests if r.duration > 5.0]  # 超过5秒的请求
        
        if slow_requests:
            alerts.append({
                "type": "slow_requests",
                "severity": "warning",
                "message": f"Found {len(slow_requests)} slow requests (>5s)",
                "timestamp": datetime.now().isoformat()
            })
        
        # 检查错误率
        if recent_requests:
            error_requests = [r for r in recent_requests if r.status_code >= 400]
            error_rate = len(error_requests) / len(recent_requests)
            
            if error_rate > 0.1:  # 错误率超过10%
                alerts.append({
                    "type": "high_error_rate",
                    "severity": "critical",
                    "message": f"High error rate: {error_rate:.1%}",
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """计算统计信息"""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / n,
            "median": sorted_values[n // 2] if n % 2 == 1 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2,
            "p95": sorted_values[int(n * 0.95)] if n > 0 else 0,
            "p99": sorted_values[int(n * 0.99)] if n > 0 else 0
        }


# 全局指标收集器
_global_collector = MetricsCollector()
_global_analyzer = PerformanceAnalyzer(_global_collector)


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    return _global_collector


def get_performance_analyzer() -> PerformanceAnalyzer:
    """获取全局性能分析器"""
    return _global_analyzer


@contextmanager
def measure_time(name: str, tags: Optional[Dict[str, str]] = None):
    """测量代码执行时间的上下文管理器"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        metric = PerformanceMetric(
            name=name,
            value=duration,
            unit="seconds",
            tags=tags or {},
            metadata={
                "memory_delta": memory_delta,
                "start_time": start_time,
                "end_time": end_time
            }
        )
        
        _global_collector.add_custom_metric(metric)


def monitor_performance(func: Optional[Callable] = None, *, 
                       name: Optional[str] = None,
                       tags: Optional[Dict[str, str]] = None):
    """性能监控装饰器"""
    def decorator(f: Callable) -> Callable:
        metric_name = name or f"{f.__module__}.{f.__name__}"
        
        @wraps(f)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            error_message = None
            
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                duration = end_time - start_time
                memory_delta = end_memory - start_memory
                
                metric = PerformanceMetric(
                    name=metric_name,
                    value=duration,
                    unit="seconds",
                    tags=tags or {},
                    metadata={
                        "memory_delta": memory_delta,
                        "error_message": error_message,
                        "function_args_count": len(args),
                        "function_kwargs_count": len(kwargs)
                    }
                )
                
                _global_collector.add_custom_metric(metric)
        
        @wraps(f)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            error_message = None
            
            try:
                result = await f(*args, **kwargs)
                return result
            except Exception as e:
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                duration = end_time - start_time
                memory_delta = end_memory - start_memory
                
                metric = PerformanceMetric(
                    name=metric_name,
                    value=duration,
                    unit="seconds",
                    tags=tags or {},
                    metadata={
                        "memory_delta": memory_delta,
                        "error_message": error_message,
                        "function_args_count": len(args),
                        "function_kwargs_count": len(kwargs)
                    }
                )
                
                _global_collector.add_custom_metric(metric)
        
        return async_wrapper if asyncio.iscoroutinefunction(f) else sync_wrapper
    
    return decorator if func is None else decorator(func)


def record_request_metric(endpoint: str, method: str, status_code: int,
                         duration: float, user_id: Optional[str] = None,
                         request_size: Optional[int] = None,
                         response_size: Optional[int] = None,
                         error_message: Optional[str] = None) -> None:
    """记录请求指标"""
    try:
        process = psutil.Process()
        memory_usage = process.memory_info().rss
        cpu_usage = process.cpu_percent()
    except:
        memory_usage = 0
        cpu_usage = 0
    
    metric = RequestMetrics(
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        duration=duration,
        memory_usage=memory_usage,
        cpu_usage=cpu_usage,
        user_id=user_id,
        request_size=request_size,
        response_size=response_size,
        error_message=error_message
    )
    
    _global_collector.add_request_metric(metric)


def get_performance_stats(since: Optional[datetime] = None) -> Dict[str, Any]:
    """获取性能统计信息"""
    analyzer = get_performance_analyzer()
    
    return {
        "request_performance": analyzer.analyze_request_performance(since),
        "system_performance": analyzer.analyze_system_performance(since),
        "alerts": analyzer.get_performance_alerts()
    }


def start_monitoring(collection_interval: int = 60) -> None:
    """启动性能监控"""
    _global_collector.start_collection(collection_interval)
    logger.info("Performance monitoring started")


def stop_monitoring() -> None:
    """停止性能监控"""
    _global_collector.stop_collection()
    logger.info("Performance monitoring stopped")


class PerformanceMiddleware:
    """FastAPI性能监控中间件"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        request_size = 0
        response_size = 0
        status_code = 500
        error_message = None
        
        # 获取请求信息
        method = scope["method"]
        path = scope["path"]
        user_id = None  # 可以从认证信息中获取
        
        # 包装receive以计算请求大小
        async def receive_wrapper():
            nonlocal request_size
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                request_size += len(body)
            return message
        
        # 包装send以计算响应大小和状态码
        async def send_wrapper(message):
            nonlocal response_size, status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                response_size += len(body)
            await send(message)
        
        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            duration = time.time() - start_time
            
            record_request_metric(
                endpoint=path,
                method=method,
                status_code=status_code,
                duration=duration,
                user_id=user_id,
                request_size=request_size,
                response_size=response_size,
                error_message=error_message
            )