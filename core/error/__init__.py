"""
多智能体LangGraph项目 - 错误处理和监控

本模块提供错误处理和监控功能，包括：
- 异常定义和分类
- 错误处理工具
- 性能监控
- 系统指标收集
- 告警机制
"""

import time
import traceback
import logging
import functools
import inspect
import asyncio
from typing import Dict, Any, Optional, List, Callable, Type, Union, TypeVar
from enum import Enum
from dataclasses import dataclass
import json
import uuid

from core.logging import get_logger

logger = get_logger("error.handler")

# 类型变量定义
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])


class ErrorSeverity(Enum):
    """错误严重程度"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误分类"""
    SYSTEM = "system"  # 系统错误
    DATABASE = "database"  # 数据库错误
    NETWORK = "network"  # 网络错误
    API = "api"  # API错误
    AUTHENTICATION = "authentication"  # 认证错误
    AUTHORIZATION = "authorization"  # 授权错误
    VALIDATION = "validation"  # 验证错误
    RESOURCE = "resource"  # 资源错误
    CONFIGURATION = "configuration"  # 配置错误
    LLM = "llm"  # LLM相关错误
    AGENT = "agent"  # 智能体错误
    TOOL = "tool"  # 工具错误
    MEMORY = "memory"  # 记忆错误
    UNKNOWN = "unknown"  # 未知错误


@dataclass
class ErrorContext:
    """错误上下文"""
    timestamp: float
    error_id: str
    component: str
    function: str
    args: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class ErrorInfo:
    """错误信息"""
    error_type: str
    error_message: str
    traceback: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext


class BaseError(Exception):
    """基础错误类"""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        component: str = "unknown",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.component = component
        self.user_id = user_id
        self.session_id = session_id
        self.additional_info = additional_info or {}
        self.timestamp = time.time()
        self.error_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "component": self.component,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "additional_info": self.additional_info
        }
    
    def to_error_info(self) -> ErrorInfo:
        """转换为错误信息"""
        tb = traceback.format_exc()
        
        context = ErrorContext(
            timestamp=self.timestamp,
            error_id=self.error_id,
            component=self.component,
            function=self.additional_info.get("function", "unknown"),
            args=self.additional_info.get("args"),
            user_id=self.user_id,
            session_id=self.session_id,
            additional_info=self.additional_info
        )
        
        return ErrorInfo(
            error_type=self.__class__.__name__,
            error_message=self.message,
            traceback=tb,
            severity=self.severity,
            category=self.category,
            context=context
        )


# 系统错误
class SystemError(BaseError):
    """系统错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("component", "system")
        super().__init__(message, **kwargs)


class ConfigurationError(BaseError):
    """配置错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault("component", "configuration")
        super().__init__(message, **kwargs)


# 数据库错误
class DatabaseError(BaseError):
    """数据库错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.DATABASE)
        kwargs.setdefault("component", "database")
        super().__init__(message, **kwargs)


class ConnectionError(BaseError):
    """连接错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.NETWORK)
        kwargs.setdefault("component", "network")
        super().__init__(message, **kwargs)


# API错误
class APIError(BaseError):
    """API错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.API)
        kwargs.setdefault("component", "api")
        super().__init__(message, **kwargs)


class AuthenticationError(BaseError):
    """认证错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.AUTHENTICATION)
        kwargs.setdefault("component", "auth")
        super().__init__(message, **kwargs)


class AuthorizationError(BaseError):
    """授权错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.AUTHORIZATION)
        kwargs.setdefault("component", "auth")
        super().__init__(message, **kwargs)


class ValidationError(BaseError):
    """验证错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        kwargs.setdefault("severity", ErrorSeverity.WARNING)
        super().__init__(message, **kwargs)


# LLM错误
class LLMError(BaseError):
    """LLM错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.LLM)
        kwargs.setdefault("component", "llm")
        super().__init__(message, **kwargs)


class LLMTimeoutError(LLMError):
    """LLM超时错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class LLMAPIError(LLMError):
    """LLM API错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


# 智能体错误
class AgentError(BaseError):
    """智能体错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.AGENT)
        kwargs.setdefault("component", "agent")
        super().__init__(message, **kwargs)


# 工具错误
class ToolError(BaseError):
    """工具错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.TOOL)
        kwargs.setdefault("component", "tool")
        super().__init__(message, **kwargs)


# 记忆错误
class MemoryError(BaseError):
    """记忆错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.MEMORY)
        kwargs.setdefault("component", "memory")
        super().__init__(message, **kwargs)


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.logger = get_logger("error.handler")
        self.error_history: List[ErrorInfo] = []
        self.max_history_size = 100
        self.error_callbacks: List[Callable[[ErrorInfo], None]] = []
    
    def handle_error(self, error: Union[BaseError, Exception], **context) -> ErrorInfo:
        """处理错误
        
        Args:
            error: 错误对象
            **context: 额外上下文
        
        Returns:
            ErrorInfo: 错误信息
        """
        # 如果是自定义错误，直接获取错误信息
        if isinstance(error, BaseError):
            # 更新上下文
            if context:
                error.additional_info.update(context)
            error_info = error.to_error_info()
        else:
            # 如果是标准异常，转换为错误信息
            tb = traceback.format_exc()
            
            # 创建错误上下文
            error_context = ErrorContext(
                timestamp=time.time(),
                error_id=str(uuid.uuid4()),
                component=context.get("component", "unknown"),
                function=context.get("function", "unknown"),
                args=context.get("args"),
                user_id=context.get("user_id"),
                session_id=context.get("session_id"),
                additional_info=context
            )
            
            # 创建错误信息
            error_info = ErrorInfo(
                error_type=error.__class__.__name__,
                error_message=str(error),
                traceback=tb,
                severity=context.get("severity", ErrorSeverity.ERROR),
                category=context.get("category", ErrorCategory.UNKNOWN),
                context=error_context
            )
        
        # 记录错误
        self._log_error(error_info)
        
        # 添加到历史
        self._add_to_history(error_info)
        
        # 触发回调
        self._trigger_callbacks(error_info)
        
        return error_info
    
    def _log_error(self, error_info: ErrorInfo) -> None:
        """记录错误日志"""
        # 根据严重程度选择日志级别
        log_message = f"[{error_info.error_type}] {error_info.error_message}"
        
        if error_info.severity == ErrorSeverity.DEBUG:
            self.logger.debug(log_message, extra={"error_info": error_info})
        elif error_info.severity == ErrorSeverity.INFO:
            self.logger.info(log_message, extra={"error_info": error_info})
        elif error_info.severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message, extra={"error_info": error_info})
        elif error_info.severity == ErrorSeverity.ERROR:
            self.logger.error(log_message, extra={"error_info": error_info})
            self.logger.error(f"Traceback: {error_info.traceback}")
        elif error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra={"error_info": error_info})
            self.logger.critical(f"Traceback: {error_info.traceback}")
    
    def _add_to_history(self, error_info: ErrorInfo) -> None:
        """添加到错误历史"""
        self.error_history.append(error_info)
        
        # 限制历史大小
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _trigger_callbacks(self, error_info: ErrorInfo) -> None:
        """触发错误回调"""
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                self.logger.error(f"错误回调执行失败: {e}")
    
    def register_callback(self, callback: Callable[[ErrorInfo], None]) -> None:
        """注册错误回调"""
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[ErrorInfo], None]) -> None:
        """注销错误回调"""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
    
    def get_recent_errors(self, limit: int = 10, severity: Optional[ErrorSeverity] = None,
                         category: Optional[ErrorCategory] = None) -> List[ErrorInfo]:
        """获取最近的错误"""
        filtered_errors = self.error_history
        
        if severity:
            filtered_errors = [e for e in filtered_errors if e.severity == severity]
        
        if category:
            filtered_errors = [e for e in filtered_errors if e.category == category]
        
        return filtered_errors[-limit:]
    
    def clear_history(self) -> None:
        """清空错误历史"""
        self.error_history.clear()


# 装饰器
def handle_errors(
    error_handler: Optional[ErrorHandler] = None,
    component: str = "unknown",
    reraise: bool = True
) -> Callable[[F], F]:
    """错误处理装饰器
    
    Args:
        error_handler: 错误处理器，如果为None则使用全局实例
        component: 组件名称
        reraise: 是否重新抛出异常
    
    Returns:
        装饰后的函数
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or global_error_handler
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取函数信息
                function_name = func.__name__
                
                # 获取调用参数
                call_args = {}
                try:
                    # 获取函数签名
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    
                    # 过滤掉self参数
                    if 'self' in bound_args.arguments:
                        bound_args.arguments.pop('self')
                    
                    call_args = dict(bound_args.arguments)
                except Exception:
                    # 如果无法获取参数，使用位置
                    call_args = {"args": args, "kwargs": kwargs}
                
                # 处理错误
                handler.handle_error(
                    e,
                    component=component,
                    function=function_name,
                    args=call_args
                )
                
                # 重新抛出异常
                if reraise:
                    raise
                
                return None
        
        return wrapper
    
    return decorator


def handle_async_errors(
    error_handler: Optional[ErrorHandler] = None,
    component: str = "unknown",
    reraise: bool = True
) -> Callable[[AsyncF], AsyncF]:
    """异步错误处理装饰器
    
    Args:
        error_handler: 错误处理器，如果为None则使用全局实例
        component: 组件名称
        reraise: 是否重新抛出异常
    
    Returns:
        装饰后的异步函数
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            handler = error_handler or global_error_handler
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # 获取函数信息
                function_name = func.__name__
                
                # 获取调用参数
                call_args = {}
                try:
                    # 获取函数签名
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    
                    # 过滤掉self参数
                    if 'self' in bound_args.arguments:
                        bound_args.arguments.pop('self')
                    
                    call_args = dict(bound_args.arguments)
                except Exception:
                    # 如果无法获取参数，使用位置
                    call_args = {"args": args, "kwargs": kwargs}
                
                # 处理错误
                handler.handle_error(
                    e,
                    component=component,
                    function=function_name,
                    args=call_args
                )
                
                # 重新抛出异常
                if reraise:
                    raise
                
                return None
        
        return wrapper
    
    return decorator


# 性能监控
class PerformanceMetric:
    """性能指标"""
    
    def __init__(self, name: str, component: str = "unknown"):
        self.name = name
        self.component = component
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.metadata = {}
    
    def start(self) -> 'PerformanceMetric':
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self) -> float:
        """停止计时并返回持续时间"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        return self.duration
    
    def add_metadata(self, key: str, value: Any) -> 'PerformanceMetric':
        """添加元数据"""
        self.metadata[key] = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "component": self.component,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "metadata": self.metadata
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.logger = get_logger("performance.monitor")
        self.metrics: List[PerformanceMetric] = []
        self.max_metrics = 1000
    
    def create_metric(self, name: str, component: str = "unknown") -> PerformanceMetric:
        """创建性能指标"""
        return PerformanceMetric(name, component)
    
    def record_metric(self, metric: PerformanceMetric) -> None:
        """记录性能指标"""
        if metric.duration is None:
            self.logger.warning(f"尝试记录未完成的性能指标: {metric.name}")
            return
        
        self.metrics.append(metric)
        
        # 限制指标数量
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
        
        # 记录日志
        self.logger.debug(
            f"性能指标: {metric.name} - {metric.duration:.4f}秒",
            extra={"metric": metric.to_dict()}
        )
    
    def get_metrics(self, component: Optional[str] = None, 
                   name: Optional[str] = None, limit: int = 100) -> List[PerformanceMetric]:
        """获取性能指标"""
        filtered_metrics = self.metrics
        
        if component:
            filtered_metrics = [m for m in filtered_metrics if m.component == component]
        
        if name:
            filtered_metrics = [m for m in filtered_metrics if m.name == name]
        
        return filtered_metrics[-limit:]
    
    def get_average_duration(self, component: Optional[str] = None, 
                            name: Optional[str] = None) -> Optional[float]:
        """获取平均持续时间"""
        metrics = self.get_metrics(component, name)
        
        if not metrics:
            return None
        
        return sum(m.duration for m in metrics) / len(metrics)
    
    def clear_metrics(self) -> None:
        """清空性能指标"""
        self.metrics.clear()


# 装饰器
def monitor_performance(
    name: Optional[str] = None,
    component: str = "unknown",
    monitor: Optional[PerformanceMonitor] = None
) -> Callable[[F], F]:
    """性能监控装饰器
    
    Args:
        name: 指标名称，如果为None则使用函数名
        component: 组件名称
        monitor: 性能监控器，如果为None则使用全局实例
    
    Returns:
        装饰后的函数
    """
    def decorator(func: F) -> F:
        metric_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            perf_monitor = monitor or global_performance_monitor
            metric = perf_monitor.create_metric(metric_name, component).start()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                metric.stop()
                perf_monitor.record_metric(metric)
        
        return wrapper
    
    return decorator


def monitor_async_performance(
    name: Optional[str] = None,
    component: str = "unknown",
    monitor: Optional[PerformanceMonitor] = None
) -> Callable[[AsyncF], AsyncF]:
    """异步性能监控装饰器
    
    Args:
        name: 指标名称，如果为None则使用函数名
        component: 组件名称
        monitor: 性能监控器，如果为None则使用全局实例
    
    Returns:
        装饰后的异步函数
    """
    def decorator(func: AsyncF) -> AsyncF:
        metric_name = name or func.__name__
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            perf_monitor = monitor or global_performance_monitor
            metric = perf_monitor.create_metric(metric_name, component).start()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                metric.stop()
                perf_monitor.record_metric(metric)
        
        return wrapper
    
    return decorator


# 全局实例
global_error_handler = ErrorHandler()
global_performance_monitor = PerformanceMonitor()


# 便捷函数
def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    return global_error_handler


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    return global_performance_monitor