"""
时间旅行类型定义

定义时间旅行相关的数据类型和枚举。
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class SnapshotType(str, Enum):
    """快照类型"""
    MANUAL = "manual"  # 手动快照
    AUTO = "auto"  # 自动快照
    CHECKPOINT = "checkpoint"  # 检查点快照
    BRANCH = "branch"  # 分支快照
    MERGE = "merge"  # 合并快照


class CheckpointType(str, Enum):
    """检查点类型"""
    STEP = "step"  # 步骤检查点
    MILESTONE = "milestone"  # 里程碑检查点
    ERROR = "error"  # 错误检查点
    USER = "user"  # 用户检查点
    SYSTEM = "system"  # 系统检查点


class RollbackStrategy(str, Enum):
    """回滚策略"""
    SOFT = "soft"  # 软回滚（保留历史）
    HARD = "hard"  # 硬回滚（删除历史）
    BRANCH = "branch"  # 分支回滚
    MERGE = "merge"  # 合并回滚


class StateVersion(BaseModel):
    """状态版本"""
    version_id: str = Field(description="版本ID")
    parent_version: Optional[str] = Field(None, description="父版本ID")
    branch_name: str = Field("main", description="分支名称")
    version_number: int = Field(description="版本号")
    created_at: datetime = Field(description="创建时间")
    created_by: Optional[str] = Field(None, description="创建者")
    description: Optional[str] = Field(None, description="版本描述")
    tags: List[str] = Field(default_factory=list, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class StateSnapshot(BaseModel):
    """状态快照"""
    id: str = Field(description="快照ID")
    version: StateVersion = Field(description="版本信息")
    type: SnapshotType = Field(description="快照类型")
    
    # 状态数据
    state_data: Dict[str, Any] = Field(description="状态数据")
    context_data: Optional[Dict[str, Any]] = Field(None, description="上下文数据")
    
    # 执行信息
    execution_id: Optional[str] = Field(None, description="执行ID")
    step_name: Optional[str] = Field(None, description="步骤名称")
    agent_id: Optional[str] = Field(None, description="智能体ID")
    
    # 时间信息
    timestamp: datetime = Field(description="时间戳")
    duration_from_start: Optional[float] = Field(None, description="从开始的时长")
    
    # 变更信息
    changes: Optional[Dict[str, Any]] = Field(None, description="变更内容")
    diff_from_parent: Optional[Dict[str, Any]] = Field(None, description="与父版本的差异")
    
    # 元数据
    size_bytes: Optional[int] = Field(None, description="快照大小")
    compression: Optional[str] = Field(None, description="压缩方式")
    checksum: Optional[str] = Field(None, description="校验和")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class Checkpoint(BaseModel):
    """检查点"""
    id: str = Field(description="检查点ID")
    name: str = Field(description="检查点名称")
    type: CheckpointType = Field(description="检查点类型")
    description: Optional[str] = Field(None, description="检查点描述")
    
    # 关联快照
    snapshot_id: str = Field(description="快照ID")
    
    # 执行信息
    execution_id: str = Field(description="执行ID")
    step_name: Optional[str] = Field(None, description="步骤名称")
    agent_id: str = Field(description="智能体ID")
    
    # 时间信息
    created_at: datetime = Field(description="创建时间")
    created_by: Optional[str] = Field(None, description="创建者")
    
    # 恢复配置
    restore_config: Optional[Dict[str, Any]] = Field(None, description="恢复配置")
    
    # 元数据
    tags: List[str] = Field(default_factory=list, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class RollbackPoint(BaseModel):
    """回滚点"""
    id: str = Field(description="回滚点ID")
    target_snapshot_id: str = Field(description="目标快照ID")
    target_checkpoint_id: Optional[str] = Field(None, description="目标检查点ID")
    
    # 回滚信息
    rollback_strategy: RollbackStrategy = Field(description="回滚策略")
    rollback_reason: Optional[str] = Field(None, description="回滚原因")
    
    # 执行信息
    execution_id: str = Field(description="执行ID")
    agent_id: str = Field(description="智能体ID")
    
    # 时间信息
    created_at: datetime = Field(description="创建时间")
    executed_at: Optional[datetime] = Field(None, description="执行时间")
    
    # 状态信息
    status: str = Field("pending", description="回滚状态")
    error: Optional[str] = Field(None, description="错误信息")
    
    # 影响范围
    affected_steps: List[str] = Field(default_factory=list, description="影响的步骤")
    affected_data: Optional[Dict[str, Any]] = Field(None, description="影响的数据")
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class TimeTravelConfig(BaseModel):
    """时间旅行配置"""
    # 快照配置
    auto_snapshot: bool = Field(True, description="自动快照")
    snapshot_interval: int = Field(10, description="快照间隔（步骤数）")
    max_snapshots: int = Field(100, description="最大快照数")
    
    # 检查点配置
    auto_checkpoint: bool = Field(True, description="自动检查点")
    checkpoint_on_error: bool = Field(True, description="错误时创建检查点")
    checkpoint_on_milestone: bool = Field(True, description="里程碑时创建检查点")
    
    # 存储配置
    compression_enabled: bool = Field(True, description="启用压缩")
    compression_algorithm: str = Field("gzip", description="压缩算法")
    storage_backend: str = Field("local", description="存储后端")
    
    # 清理配置
    auto_cleanup: bool = Field(True, description="自动清理")
    retention_days: int = Field(30, description="保留天数")
    max_storage_size: Optional[int] = Field(None, description="最大存储大小")
    
    # 分支配置
    enable_branching: bool = Field(False, description="启用分支")
    max_branches: int = Field(10, description="最大分支数")
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class HistoryQuery(BaseModel):
    """历史查询"""
    # 查询条件
    execution_id: Optional[str] = Field(None, description="执行ID")
    agent_id: Optional[str] = Field(None, description="智能体ID")
    step_name: Optional[str] = Field(None, description="步骤名称")
    
    # 时间范围
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    
    # 版本范围
    start_version: Optional[int] = Field(None, description="开始版本")
    end_version: Optional[int] = Field(None, description="结束版本")
    
    # 类型过滤
    snapshot_types: Optional[List[SnapshotType]] = Field(None, description="快照类型")
    checkpoint_types: Optional[List[CheckpointType]] = Field(None, description="检查点类型")
    
    # 分支过滤
    branch_name: Optional[str] = Field(None, description="分支名称")
    include_branches: bool = Field(False, description="包含分支")
    
    # 排序和分页
    sort_by: str = Field("timestamp", description="排序字段")
    sort_order: str = Field("desc", description="排序顺序")
    limit: Optional[int] = Field(None, description="限制数量")
    offset: int = Field(0, description="偏移量")
    
    # 包含内容
    include_state_data: bool = Field(False, description="包含状态数据")
    include_context_data: bool = Field(False, description="包含上下文数据")
    include_changes: bool = Field(False, description="包含变更信息")
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class BranchInfo(BaseModel):
    """分支信息"""
    name: str = Field(description="分支名称")
    description: Optional[str] = Field(None, description="分支描述")
    parent_branch: Optional[str] = Field(None, description="父分支")
    created_at: datetime = Field(description="创建时间")
    created_by: Optional[str] = Field(None, description="创建者")
    
    # 分支状态
    is_active: bool = Field(True, description="是否活跃")
    is_merged: bool = Field(False, description="是否已合并")
    merge_target: Optional[str] = Field(None, description="合并目标")
    
    # 统计信息
    snapshot_count: int = Field(0, description="快照数量")
    checkpoint_count: int = Field(0, description="检查点数量")
    last_activity: Optional[datetime] = Field(None, description="最后活动时间")
    
    # 元数据
    tags: List[str] = Field(default_factory=list, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class MergeRequest(BaseModel):
    """合并请求"""
    id: str = Field(description="合并请求ID")
    source_branch: str = Field(description="源分支")
    target_branch: str = Field(description="目标分支")
    title: str = Field(description="合并标题")
    description: Optional[str] = Field(None, description="合并描述")
    
    # 合并配置
    merge_strategy: str = Field("auto", description="合并策略")
    conflict_resolution: str = Field("manual", description="冲突解决方式")
    
    # 状态信息
    status: str = Field("pending", description="合并状态")
    created_at: datetime = Field(description="创建时间")
    created_by: Optional[str] = Field(None, description="创建者")
    
    # 冲突信息
    conflicts: List[Dict[str, Any]] = Field(default_factory=list, description="冲突列表")
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")