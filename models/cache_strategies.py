#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存策略管理模块

提供多种缓存策略和管理功能，包括：
- 多级缓存系统
- 分布式缓存
- 缓存预热和预加载
- 缓存失效策略
- 缓存监控和统计
- 智能缓存策略
"""

import time
import json
import hashlib
import pickle
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod
import weakref
from concurrent.futures import ThreadPoolExecutor, Future
import logging


class CacheLevel(str, Enum):
    """缓存级别枚举"""
    L1_MEMORY = "l1_memory"  # 一级内存缓存
    L2_DISK = "l2_disk"      # 二级磁盘缓存
    L3_DISTRIBUTED = "l3_distributed"  # 三级分布式缓存
    L4_PERSISTENT = "l4_persistent"    # 四级持久化缓存


class EvictionPolicy(str, Enum):
    """缓存淘汰策略枚举"""
    LRU = "lru"          # 最近最少使用
    LFU = "lfu"          # 最少使用频率
    FIFO = "fifo"        # 先进先出
    LIFO = "lifo"        # 后进先出
    TTL = "ttl"          # 基于时间过期
    RANDOM = "random"    # 随机淘汰
    ADAPTIVE = "adaptive"  # 自适应策略
    COST_AWARE = "cost_aware"  # 成本感知策略


class CachePattern(str, Enum):
    """缓存模式枚举"""
    CACHE_ASIDE = "cache_aside"        # 旁路缓存
    WRITE_THROUGH = "write_through"    # 写穿透
    WRITE_BEHIND = "write_behind"      # 写回
    REFRESH_AHEAD = "refresh_ahead"    # 提前刷新
    READ_THROUGH = "read_through"      # 读穿透


class CacheConsistency(str, Enum):
    """缓存一致性级别枚举"""
    EVENTUAL = "eventual"      # 最终一致性
    STRONG = "strong"          # 强一致性
    WEAK = "weak"              # 弱一致性
    SESSION = "session"        # 会话一致性


@dataclass
class CacheMetrics:
    """缓存指标数据类"""
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    load_count: int = 0
    load_exception_count: int = 0
    total_load_time: float = 0.0
    average_load_penalty: float = 0.0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    request_count: int = 0
    memory_usage: int = 0
    disk_usage: int = 0
    network_usage: int = 0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_rates(self):
        """计算命中率和缺失率"""
        total = self.hit_count + self.miss_count
        if total > 0:
            self.hit_rate = self.hit_count / total
            self.miss_rate = self.miss_count / total
        else:
            self.hit_rate = 0.0
            self.miss_rate = 0.0
        
        if self.load_count > 0:
            self.average_load_penalty = self.total_load_time / self.load_count
    
    def reset(self):
        """重置指标"""
        self.__init__()


@dataclass
class CacheEntry:
    """缓存条目数据类"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    last_modified: datetime
    access_count: int = 0
    ttl: Optional[int] = None  # 秒
    size: int = 0
    cost: float = 1.0  # 缓存成本
    priority: int = 0  # 优先级
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    
    def is_expired(self) -> bool:
        """检查缓存是否过期"""
        if self.ttl is None:
            return False
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() > self.ttl
    
    def update_access(self):
        """更新访问信息"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
    
    def update_value(self, value: Any):
        """更新缓存值"""
        self.value = value
        self.last_modified = datetime.now(timezone.utc)
        self.version += 1
    
    def calculate_score(self, policy: EvictionPolicy) -> float:
        """根据淘汰策略计算分数"""
        now = datetime.now(timezone.utc)
        
        if policy == EvictionPolicy.LRU:
            return (now - self.last_accessed).total_seconds()
        elif policy == EvictionPolicy.LFU:
            return -self.access_count  # 负数，频率越高分数越低
        elif policy == EvictionPolicy.TTL:
            if self.ttl:
                return (now - self.created_at).total_seconds() / self.ttl
            return 0
        elif policy == EvictionPolicy.COST_AWARE:
            # 综合考虑成本、访问频率和时间
            time_factor = (now - self.last_accessed).total_seconds()
            freq_factor = 1.0 / (self.access_count + 1)
            return self.cost * time_factor * freq_factor
        else:
            return 0.0


class CacheStorage(ABC):
    """缓存存储抽象基类"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """获取缓存条目"""
        pass
    
    @abstractmethod
    def put(self, key: str, entry: CacheEntry) -> bool:
        """存储缓存条目"""
        pass
    
    @abstractmethod
    def remove(self, key: str) -> bool:
        """移除缓存条目"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """获取缓存大小"""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """获取所有键"""
        pass


class MemoryStorage(CacheStorage):
    """内存缓存存储"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.storage: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        with self.lock:
            entry = self.storage.get(key)
            if entry and not entry.is_expired():
                entry.update_access()
                return entry
            elif entry and entry.is_expired():
                del self.storage[key]
            return None
    
    def put(self, key: str, entry: CacheEntry) -> bool:
        with self.lock:
            if len(self.storage) >= self.max_size and key not in self.storage:
                return False  # 存储已满
            self.storage[key] = entry
            return True
    
    def remove(self, key: str) -> bool:
        with self.lock:
            if key in self.storage:
                del self.storage[key]
                return True
            return False
    
    def clear(self) -> bool:
        with self.lock:
            self.storage.clear()
            return True
    
    def size(self) -> int:
        with self.lock:
            return len(self.storage)
    
    def keys(self) -> List[str]:
        with self.lock:
            return list(self.storage.keys())


class DiskStorage(CacheStorage):
    """磁盘缓存存储"""
    
    def __init__(self, cache_dir: str = "/tmp/cache", max_size: int = 10000):
        import os
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.lock = threading.RLock()
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_file_path(self, key: str) -> str:
        """获取缓存文件路径"""
        import os
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.cache")
    
    def get(self, key: str) -> Optional[CacheEntry]:
        with self.lock:
            file_path = self._get_file_path(key)
            try:
                import os
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        entry = pickle.load(f)
                    
                    if not entry.is_expired():