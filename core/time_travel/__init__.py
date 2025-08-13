"""  
时间旅行和状态回溯模块

提供状态快照、检查点、回滚和分支管理功能。
"""

from .time_travel_types import (
    StateSnapshot,
    Checkpoint,
    RollbackPoint,
    StateVersion,
    TimeTravelConfig,
    HistoryQuery,
    BranchInfo,
    MergeRequest,
    SnapshotType,
    CheckpointType,
    RollbackStrategy
)

from .time_travel_manager import TimeTravelManager
from .state_history_manager import StateHistoryManager
from .checkpoint_manager import CheckpointManager
from .rollback_manager import RollbackManager

# 全局时间旅行管理器实例
_global_time_travel_manager = None


def get_time_travel_manager() -> TimeTravelManager:
    """获取全局时间旅行管理器实例"""
    global _global_time_travel_manager
    if _global_time_travel_manager is None:
        _global_time_travel_manager = TimeTravelManager()
    return _global_time_travel_manager


def set_time_travel_manager(manager: TimeTravelManager):
    """设置全局时间旅行管理器实例"""
    global _global_time_travel_manager
    _global_time_travel_manager = manager


__all__ = [
    # 管理器类
    "TimeTravelManager",
    "StateHistoryManager", 
    "CheckpointManager",
    "RollbackManager",
    
    # 全局管理器函数
    "get_time_travel_manager",
    "set_time_travel_manager",
    
    # 数据模型
    "StateSnapshot",
    "Checkpoint", 
    "RollbackPoint",
    "StateVersion",
    "BranchInfo",
    "MergeRequest",
    
    # 枚举类型
    "SnapshotType",
    "CheckpointType",
    "RollbackStrategy",
    
    # 配置和查询
    "TimeTravelConfig",
    "HistoryQuery"
]