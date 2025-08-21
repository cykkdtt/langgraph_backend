#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API性能优化模块

提供API性能优化相关的功能，包括：
- 响应时间优化
- 缓存策略
- 请求限流
- 压缩优化
- 批量处理优化
- 连接池管理
"""

import time
import asyncio
import hashlib
import gzip
import json
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timezone, timedelta
from functools import wraps, lru_cache
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class CacheStrategy(str, Enum):
    """缓存策略枚举"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    FIFO = "fifo"  # 先进先出
    TTL = "ttl"  # 基于时间的过期
    ADAPTIVE = "adaptive"  # 自适应缓存


class CompressionType(str, Enum):
    """压缩类型枚举"""
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "brotli"
    NONE = "none"


class OptimizationLevel(str, Enum):
    """优化级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AGGRESSIVE = "aggressive"


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    response_time: float = 0.0
    throughput: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    compression_ratio: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CacheEntry:
    """缓存条目数据类"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[int] = None  # 秒
    size: int = 0
    
    def is_expired(self) -> bool:
        """检查缓存是否过期"""
        if self.ttl is None:
            return False
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() > self.ttl
    
    def update_access(self):
        """更新访问信息"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


class RateLimiter:
    """请求限流器"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        """
        初始化限流器
        
        Args:
            max_requests: 时间窗口内最大请求数
            time_window: 时间窗口（秒）
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # 清理过期的请求记录
            while client_requests and client_requests[0] <= now - self.time_window:
                client_requests.popleft()
            
            # 检查是否超过限制
            if len(client_requests) >= self.max_requests:
                return False
            
            # 记录新请求
            client_requests.append(now)
            return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """获取剩余请求数"""
        with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # 清理过期的请求记录
            while client_requests and client_requests[0] <= now - self.time_window:
                client_requests.popleft()
            
            return max(0, self.max_requests - len(client_requests))


class AdaptiveCache:
    """自适应缓存系统"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        初始化自适应缓存
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认TTL（秒）
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # 用于LRU
        self.access_frequency = defaultdict(int)  # 用于LFU
        self.lock = threading.RLock()
        self.strategy = CacheStrategy.ADAPTIVE
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _evict_expired(self):
        """清理过期缓存"""
        expired_keys = []
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _remove_entry(self, key: str):
        """移除缓存条目"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            if key in self.access_frequency:
                del self.access_frequency[key]
    
    def _evict_by_strategy(self):
        """根据策略清理缓存"""
        if len(self.cache) < self.max_size:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # 移除最近最少使用的
            if self.access_order:
                oldest_key = self.access_order.popleft()
                self._remove_entry(oldest_key)
        
        elif self.strategy == CacheStrategy.LFU:
            # 移除使用频率最低的
            if self.access_frequency:
                min_freq_key = min(self.access_frequency.items(), key=lambda x: x[1])[0]
                self._remove_entry(min_freq_key)
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # 自适应策略：结合访问频率和时间
            if self.cache:
                now = datetime.now(timezone.utc)
                scores = {}
                for key, entry in self.cache.items():
                    time_score = (now - entry.last_accessed).total_seconds()
                    freq_score = 1.0 / (entry.access_count + 1)
                    scores[key] = time_score * freq_score
                
                worst_key = max(scores.items(), key=lambda x: x[1])[0]
                self._remove_entry(worst_key)
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            self._evict_expired()
            
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    entry.update_access()
                    self.access_frequency[key] += 1
                    
                    # 更新LRU顺序
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    
                    self.hit_count += 1
                    return entry.value
                else:
                    self._remove_entry(key)
            
            self.miss_count += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        with self.lock:
            self._evict_expired()
            self._evict_by_strategy()
            
            if ttl is None:
                ttl = self.default_ttl
            
            # 计算值的大小（简化版本）
            try:
                size = len(json.dumps(value, default=str))
            except:
                size = 1024  # 默认大小
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                ttl=ttl,
                size=size
            )
            
            self.cache[key] = entry
            self.access_order.append(key)
            self.access_frequency[key] = 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            total_size = sum(entry.size for entry in self.cache.values())
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "total_size": total_size,
                "strategy": self.strategy.value
            }


class ResponseCompressor:
    """响应压缩器"""
    
    def __init__(self, compression_type: CompressionType = CompressionType.GZIP,
                 compression_level: int = 6, min_size: int = 1024):
        """
        初始化压缩器
        
        Args:
            compression_type: 压缩类型
            compression_level: 压缩级别 (1-9)
            min_size: 最小压缩大小（字节）
        """
        self.compression_type = compression_type
        self.compression_level = compression_level
        self.min_size = min_size
    
    def compress(self, data: Union[str, bytes]) -> Tuple[bytes, float]:
        """压缩数据"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        original_size = len(data)
        
        if original_size < self.min_size:
            return data, 1.0  # 不压缩小数据
        
        if self.compression_type == CompressionType.GZIP:
            compressed = gzip.compress(data, compresslevel=self.compression_level)
        elif self.compression_type == CompressionType.DEFLATE:
            import zlib
            compressed = zlib.compress(data, level=self.compression_level)
        else:
            compressed = data  # 不支持的压缩类型
        
        compression_ratio = len(compressed) / original_size
        return compressed, compression_ratio
    
    def decompress(self, data: bytes) -> bytes:
        """解压缩数据"""
        if self.compression_type == CompressionType.GZIP:
            return gzip.decompress(data)
        elif self.compression_type == CompressionType.DEFLATE:
            import zlib
            return zlib.decompress(data)
        else:
            return data


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 1.0,
                 max_workers: int = 4):
        """
        初始化批量处理器
        
        Args:
            batch_size: 批量大小
            max_wait_time: 最大等待时间（秒）
            max_workers: 最大工作线程数
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_requests = []
        self.lock = threading.Lock()
        self.last_batch_time = time.time()
    
    def add_request(self, request_data: Any, callback: Callable) -> bool:
        """添加请求到批量处理队列"""
        with self.lock:
            self.pending_requests.append((request_data, callback))
            
            # 检查是否需要处理批量
            should_process = (
                len(self.pending_requests) >= self.batch_size or
                time.time() - self.last_batch_time >= self.max_wait_time
            )
            
            if should_process:
                self._process_batch()
            
            return True
    
    def _process_batch(self):
        """处理当前批量"""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        self.last_batch_time = time.time()
        
        # 提交批量处理任务
        future = self.executor.submit(self._execute_batch, batch)
        return future
    
    def _execute_batch(self, batch: List[Tuple[Any, Callable]]):
        """执行批量处理"""
        try:
            # 提取请求数据
            requests = [item[0] for item in batch]
            callbacks = [item[1] for item in batch]
            
            # 这里应该实现具体的批量处理逻辑
            # 示例：并行处理每个请求
            futures = []
            for request_data, callback in batch:
                future = self.executor.submit(callback, request_data)
                futures.append(future)
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # 处理结果
                except Exception as e:
                    # 处理异常
                    print(f"批量处理中的错误: {e}")
        
        except Exception as e:
            print(f"批量处理执行错误: {e}")
    
    def flush(self):
        """强制处理所有待处理的请求"""
        with self.lock:
            if self.pending_requests:
                self._process_batch()
    
    def shutdown(self):
        """关闭批量处理器"""
        self.flush()
        self.executor.shutdown(wait=True)


class APIOptimizer:
    """API优化器主类"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.MEDIUM):
        """
        初始化API优化器
        
        Args:
            optimization_level: 优化级别
        """
        self.optimization_level = optimization_level
        self.cache = AdaptiveCache()
        self.rate_limiter = RateLimiter(max_requests=1000, time_window=60)
        self.compressor = ResponseCompressor()
        self.batch_processor = BatchProcessor()
        self.metrics = PerformanceMetrics()
        self.start_time = time.time()
        
        # 根据优化级别调整参数
        self._configure_by_level()
    
    def _configure_by_level(self):
        """根据优化级别配置参数"""
        if self.optimization_level == OptimizationLevel.LOW:
            self.cache.max_size = 500
            self.rate_limiter.max_requests = 500
            self.compressor.compression_level = 3
            self.batch_processor.batch_size = 50
        
        elif self.optimization_level == OptimizationLevel.MEDIUM:
            self.cache.max_size = 1000
            self.rate_limiter.max_requests = 1000
            self.compressor.compression_level = 6
            self.batch_processor.batch_size = 100
        
        elif self.optimization_level == OptimizationLevel.HIGH:
            self.cache.max_size = 2000
            self.rate_limiter.max_requests = 2000
            self.compressor.compression_level = 9
            self.batch_processor.batch_size = 200
        
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            self.cache.max_size = 5000
            self.rate_limiter.max_requests = 5000
            self.compressor.compression_level = 9
            self.batch_processor.batch_size = 500
    
    def cache_response(self, cache_key: str = None, ttl: int = None):
        """缓存响应装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                if cache_key:
                    key = cache_key
                else:
                    key = self.cache._generate_key(func.__name__, *args, **kwargs)
                
                # 尝试从缓存获取
                cached_result = self.cache.get(key)
                if cached_result is not None:
                    return cached_result
                
                # 执行函数并缓存结果
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.cache.set(key, result, ttl)
                    
                    # 更新性能指标
                    response_time = time.time() - start_time
                    self.metrics.response_time = response_time
                    self.metrics.success_count += 1
                    
                    return result
                
                except Exception as e:
                    self.metrics.error_count += 1
                    raise e
            
            return wrapper
        return decorator
    
    def rate_limit(self, client_id_func: Callable = None):
        """请求限流装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 获取客户端ID
                if client_id_func:
                    client_id = client_id_func(*args, **kwargs)
                else:
                    client_id = "default"
                
                # 检查限流
                if not self.rate_limiter.is_allowed(client_id):
                    raise Exception(f"请求频率过高，客户端 {client_id} 被限流")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def compress_response(self, min_size: int = 1024):
        """响应压缩装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                # 如果结果是字符串或字节，尝试压缩
                if isinstance(result, (str, bytes)):
                    compressed, ratio = self.compressor.compress(result)
                    self.metrics.compression_ratio = ratio
                    return compressed
                
                return result
            
            return wrapper
        return decorator
    
    def monitor_performance(self):
        """性能监控装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # 更新性能指标
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    self.metrics.response_time = end_time - start_time
                    self.metrics.memory_usage = end_memory - start_memory
                    self.metrics.request_count += 1
                    self.metrics.success_count += 1
                    
                    # 计算吞吐量
                    elapsed = end_time - self.start_time
                    if elapsed > 0:
                        self.metrics.throughput = self.metrics.request_count / elapsed
                    
                    return result
                
                except Exception as e:
                    self.metrics.error_count += 1
                    self.metrics.error_rate = self.metrics.error_count / self.metrics.request_count
                    raise e
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（简化版本）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        cache_stats = self.cache.get_stats()
        
        return {
            "optimization_level": self.optimization_level.value,
            "metrics": {
                "response_time": self.metrics.response_time,
                "throughput": self.metrics.throughput,
                "error_rate": self.metrics.error_rate,
                "request_count": self.metrics.request_count,
                "success_count": self.metrics.success_count,
                "error_count": self.metrics.error_count,
                "memory_usage": self.metrics.memory_usage,
                "compression_ratio": self.metrics.compression_ratio
            },
            "cache": cache_stats,
            "rate_limiter": {
                "max_requests": self.rate_limiter.max_requests,
                "time_window": self.rate_limiter.time_window
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def optimize_batch_requests(self, requests: List[Any], 
                              processor_func: Callable) -> List[Any]:
        """优化批量请求处理"""
        results = []
        
        # 将请求添加到批量处理器
        for request in requests:
            self.batch_processor.add_request(request, processor_func)
        
        # 强制处理所有待处理的请求
        self.batch_processor.flush()
        
        return results
    
    def cleanup(self):
        """清理资源"""
        self.batch_processor.shutdown()
        self.cache.clear()


# 全局优化器实例
default_optimizer = APIOptimizer()


# 便捷装饰器函数
def cached(ttl: int = 3600, key: str = None):
    """缓存装饰器"""
    return default_optimizer.cache_response(cache_key=key, ttl=ttl)


def rate_limited(client_id_func: Callable = None):
    """限流装饰器"""
    return default_optimizer.rate_limit(client_id_func)


def compressed(min_size: int = 1024):
    """压缩装饰器"""
    return default_optimizer.compress_response(min_size)


def monitored():
    """性能监控装饰器"""
    return default_optimizer.monitor_performance()


# 使用示例
if __name__ == "__main__":
    # 创建优化器
    optimizer = APIOptimizer(OptimizationLevel.HIGH)
    
    # 使用装饰器优化API函数
    @optimizer.cache_response(ttl=1800)
    @optimizer.rate_limit()
    @optimizer.compress_response()
    @optimizer.monitor_performance()
    def example_api_function(data: str) -> str:
        """示例API函数"""
        # 模拟处理时间
        time.sleep(0.1)
        return f"处理结果: {data}"
    
    # 测试API函数
    try:
        result = example_api_function("测试数据")
        print(f"API结果: {result}")
        
        # 获取性能报告
        report = optimizer.get_performance_report()
        print(f"性能报告: {json.dumps(report, indent=2, ensure_ascii=False)}")
        
    finally:
        # 清理资源
        optimizer.cleanup()