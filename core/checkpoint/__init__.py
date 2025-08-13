"""
检查点管理模块

本模块提供检查点管理功能，包括：
- 检查点保存和加载
- 多种存储后端支持
- 检查点历史管理
"""

from .manager import (
    CheckpointMetadata,
    CheckpointInfo,
    CheckpointManager,
    get_checkpoint_manager,
    checkpoint_manager_context
)

__all__ = [
    "CheckpointMetadata",
    "CheckpointInfo", 
    "CheckpointManager",
    "get_checkpoint_manager",
    "checkpoint_manager_context"
]