from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

# 快照类型枚举
class SnapshotType(str, Enum):
    MANUAL = "manual"  # 手动快照
    AUTO = "auto"  # 自动快照
    CHECKPOINT = "checkpoint"  # 检查点快照
    MILESTONE = "milestone"  # 里程碑快照
    BACKUP = "backup"  # 备份快照
    BRANCH = "branch"  # 分支快照

# 快照状态枚举
class SnapshotStatus(str, Enum):
    ACTIVE = "active"  # 活跃
    ARCHIVED = "archived"  # 归档
    DELETED = "deleted"  # 已删除
    CORRUPTED = "corrupted"  # 损坏
    RESTORING = "restoring"  # 恢复中
    CREATING = "creating"  # 创建中

# 回滚类型枚举
class RollbackType(str, Enum):
    FULL = "full"  # 完全回滚
    PARTIAL = "partial"  # 部分回滚
    SELECTIVE = "selective"  # 选择性回滚
    MERGE = "merge"  # 合并回滚

# 回滚状态枚举
class RollbackStatus(str, Enum):
    PENDING = "pending"  # 待处理
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消
    PARTIAL_SUCCESS = "partial_success"  # 部分成功

# 历史事件类型枚举
class HistoryEventType(str, Enum):
    SNAPSHOT_CREATED = "snapshot_created"  # 快照创建
    SNAPSHOT_DELETED = "snapshot_deleted"  # 快照删除
    ROLLBACK_INITIATED = "rollback_initiated"  # 回滚启动
    ROLLBACK_COMPLETED = "rollback_completed"  # 回滚完成
    STATE_CHANGED = "state_changed"  # 状态变更
    BRANCH_CREATED = "branch_created"  # 分支创建
    BRANCH_MERGED = "branch_merged"  # 分支合并
    CHECKPOINT_REACHED = "checkpoint_reached"  # 检查点到达

# 状态组件类型枚举
class StateComponentType(str, Enum):
    MEMORY = "memory"  # 记忆状态
    CONVERSATION = "conversation"  # 对话状态
    WORKFLOW = "workflow"  # 工作流状态
    AGENT = "agent"  # 智能体状态
    SESSION = "session"  # 会话状态
    USER_PREFERENCES = "user_preferences"  # 用户偏好
    SYSTEM_CONFIG = "system_config"  # 系统配置

# 状态组件模型
class StateComponent(BaseModel):
    component_type: StateComponentType = Field(..., description="组件类型")
    component_id: str = Field(..., description="组件ID")
    data: Dict[str, Any] = Field(..., description="组件数据")
    version: str = Field(..., description="版本号")
    checksum: str = Field(..., description="校验和")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

# 快照元数据模型
class SnapshotMetadata(BaseModel):
    user_id: str = Field(..., description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    thread_id: Optional[str] = Field(None, description="线程ID")
    workflow_id: Optional[str] = Field(None, description="工作流ID")
    agent_id: Optional[str] = Field(None, description="智能体ID")
    tags: List[str] = Field(default_factory=list, description="标签")
    description: Optional[str] = Field(None, description="描述")
    trigger_event: Optional[str] = Field(None, description="触发事件")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文")
    parent_snapshot_id: Optional[str] = Field(None, description="父快照ID")
    branch_name: Optional[str] = Field(None, description="分支名称")
    is_milestone: bool = Field(default=False, description="是否为里程碑")
    auto_cleanup: bool = Field(default=True, description="是否自动清理")
    retention_days: int = Field(default=30, description="保留天数")

# 快照创建请求
class SnapshotCreateRequest(BaseModel):
    type: SnapshotType = Field(..., description="快照类型")
    name: str = Field(..., description="快照名称")
    description: Optional[str] = Field(None, description="快照描述")
    metadata: SnapshotMetadata = Field(..., description="快照元数据")
    components: List[StateComponentType] = Field(..., description="要包含的状态组件")
    include_full_state: bool = Field(default=True, description="是否包含完整状态")
    compress: bool = Field(default=True, description="是否压缩")
    encrypt: bool = Field(default=False, description="是否加密")
    tags: List[str] = Field(default_factory=list, description="标签")
    auto_cleanup_after: Optional[int] = Field(None, description="自动清理天数")

# 快照更新请求
class SnapshotUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, description="快照名称")
    description: Optional[str] = Field(None, description="快照描述")
    status: Optional[SnapshotStatus] = Field(None, description="快照状态")
    tags: Optional[List[str]] = Field(None, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    retention_days: Optional[int] = Field(None, description="保留天数")
    is_milestone: Optional[bool] = Field(None, description="是否为里程碑")

# 快照搜索请求
class SnapshotSearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="搜索查询")
    type: Optional[SnapshotType] = Field(None, description="快照类型")
    status: Optional[SnapshotStatus] = Field(None, description="快照状态")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    thread_id: Optional[str] = Field(None, description="线程ID")
    workflow_id: Optional[str] = Field(None, description="工作流ID")
    agent_id: Optional[str] = Field(None, description="智能体ID")
    tags: Optional[List[str]] = Field(None, description="标签")
    created_after: Optional[datetime] = Field(None, description="创建时间起")
    created_before: Optional[datetime] = Field(None, description="创建时间止")
    is_milestone: Optional[bool] = Field(None, description="是否为里程碑")
    branch_name: Optional[str] = Field(None, description="分支名称")
    has_components: Optional[List[StateComponentType]] = Field(None, description="包含的组件")
    min_size: Optional[int] = Field(None, description="最小大小（字节）")
    max_size: Optional[int] = Field(None, description="最大大小（字节）")
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")
    sort_by: str = Field(default="created_at", description="排序字段")
    sort_order: str = Field(default="desc", description="排序方向")

# 回滚请求
class RollbackRequest(BaseModel):
    snapshot_id: str = Field(..., description="目标快照ID")
    rollback_type: RollbackType = Field(..., description="回滚类型")
    components: Optional[List[StateComponentType]] = Field(None, description="要回滚的组件")
    create_backup: bool = Field(default=True, description="是否创建备份")
    backup_name: Optional[str] = Field(None, description="备份名称")
    force: bool = Field(default=False, description="是否强制回滚")
    dry_run: bool = Field(default=False, description="是否为试运行")
    merge_strategy: Optional[str] = Field(None, description="合并策略")
    conflict_resolution: str = Field(default="abort", description="冲突解决策略")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="回滚元数据")
    notify_on_completion: bool = Field(default=True, description="完成时是否通知")

# 历史查看请求
class HistoryViewRequest(BaseModel):
    entity_type: str = Field(..., description="实体类型")
    entity_id: str = Field(..., description="实体ID")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    event_types: Optional[List[HistoryEventType]] = Field(None, description="事件类型")
    include_snapshots: bool = Field(default=True, description="是否包含快照")
    include_changes: bool = Field(default=True, description="是否包含变更")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    granularity: str = Field(default="minute", description="时间粒度")
    max_events: int = Field(default=1000, ge=1, le=10000, description="最大事件数")
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=50, ge=1, le=200, description="每页大小")

# 分支操作请求
class BranchRequest(BaseModel):
    source_snapshot_id: str = Field(..., description="源快照ID")
    branch_name: str = Field(..., description="分支名称")
    description: Optional[str] = Field(None, description="分支描述")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="分支元数据")
    auto_merge: bool = Field(default=False, description="是否自动合并")
    merge_strategy: Optional[str] = Field(None, description="合并策略")

# 快照信息模型
class SnapshotInfo(BaseModel):
    id: str = Field(..., description="快照ID")
    name: str = Field(..., description="快照名称")
    description: Optional[str] = Field(None, description="快照描述")
    type: SnapshotType = Field(..., description="快照类型")
    status: SnapshotStatus = Field(..., description="快照状态")
    metadata: SnapshotMetadata = Field(..., description="快照元数据")
    components: List[StateComponent] = Field(..., description="状态组件")
    size_bytes: int = Field(..., description="快照大小（字节）")
    compressed_size_bytes: Optional[int] = Field(None, description="压缩后大小")
    checksum: str = Field(..., description="校验和")
    version: str = Field(..., description="版本号")
    parent_snapshot_id: Optional[str] = Field(None, description="父快照ID")
    child_snapshot_ids: List[str] = Field(default_factory=list, description="子快照ID列表")
    branch_name: Optional[str] = Field(None, description="分支名称")
    is_milestone: bool = Field(default=False, description="是否为里程碑")
    tags: List[str] = Field(default_factory=list, description="标签")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    access_count: int = Field(default=0, description="访问次数")
    last_accessed: Optional[datetime] = Field(None, description="最后访问时间")
    rollback_count: int = Field(default=0, description="回滚次数")
    is_encrypted: bool = Field(default=False, description="是否加密")
    compression_ratio: Optional[float] = Field(None, description="压缩比")

# 回滚信息模型
class RollbackInfo(BaseModel):
    id: str = Field(..., description="回滚ID")
    snapshot_id: str = Field(..., description="目标快照ID")
    rollback_type: RollbackType = Field(..., description="回滚类型")
    status: RollbackStatus = Field(..., description="回滚状态")
    components: List[StateComponentType] = Field(..., description="回滚的组件")
    backup_snapshot_id: Optional[str] = Field(None, description="备份快照ID")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="进度")
    error_message: Optional[str] = Field(None, description="错误信息")
    conflicts: List[Dict[str, Any]] = Field(default_factory=list, description="冲突列表")
    changes_applied: List[Dict[str, Any]] = Field(default_factory=list, description="已应用的变更")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="回滚元数据")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    duration_seconds: Optional[float] = Field(None, description="持续时间（秒）")
    user_id: str = Field(..., description="用户ID")
    dry_run: bool = Field(default=False, description="是否为试运行")

# 历史事件模型
class HistoryEvent(BaseModel):
    id: str = Field(..., description="事件ID")
    event_type: HistoryEventType = Field(..., description="事件类型")
    entity_type: str = Field(..., description="实体类型")
    entity_id: str = Field(..., description="实体ID")
    snapshot_id: Optional[str] = Field(None, description="关联快照ID")
    user_id: str = Field(..., description="用户ID")
    timestamp: datetime = Field(..., description="时间戳")
    data: Dict[str, Any] = Field(..., description="事件数据")
    changes: List[Dict[str, Any]] = Field(default_factory=list, description="变更列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="事件元数据")
    context: Dict[str, Any] = Field(default_factory=dict, description="事件上下文")
    severity: str = Field(default="info", description="严重程度")
    source: str = Field(..., description="事件源")
    correlation_id: Optional[str] = Field(None, description="关联ID")

# 时间线模型
class Timeline(BaseModel):
    entity_type: str = Field(..., description="实体类型")
    entity_id: str = Field(..., description="实体ID")
    events: List[HistoryEvent] = Field(..., description="事件列表")
    snapshots: List[SnapshotInfo] = Field(default_factory=list, description="快照列表")
    start_time: datetime = Field(..., description="开始时间")
    end_time: datetime = Field(..., description="结束时间")
    total_events: int = Field(..., description="总事件数")
    total_snapshots: int = Field(..., description="总快照数")
    summary: Dict[str, Any] = Field(default_factory=dict, description="摘要信息")

# 分支信息模型
class BranchInfo(BaseModel):
    id: str = Field(..., description="分支ID")
    name: str = Field(..., description="分支名称")
    description: Optional[str] = Field(None, description="分支描述")
    source_snapshot_id: str = Field(..., description="源快照ID")
    current_snapshot_id: str = Field(..., description="当前快照ID")
    snapshots: List[str] = Field(default_factory=list, description="分支快照列表")
    is_active: bool = Field(default=True, description="是否活跃")
    is_merged: bool = Field(default=False, description="是否已合并")
    merged_to_branch: Optional[str] = Field(None, description="合并到的分支")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="分支元数据")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    merged_at: Optional[datetime] = Field(None, description="合并时间")
    user_id: str = Field(..., description="用户ID")

# 统计信息模型
class TimeTravelStatistics(BaseModel):
    total_snapshots: int = Field(default=0, description="总快照数")
    total_rollbacks: int = Field(default=0, description="总回滚数")
    total_branches: int = Field(default=0, description="总分支数")
    total_events: int = Field(default=0, description="总事件数")
    by_type: Dict[str, int] = Field(default_factory=dict, description="按类型统计")
    by_status: Dict[str, int] = Field(default_factory=dict, description="按状态统计")
    storage_used_mb: float = Field(default=0.0, description="存储使用量（MB）")
    average_snapshot_size_mb: float = Field(default=0.0, description="平均快照大小（MB）")
    most_active_user: Optional[str] = Field(None, description="最活跃用户")
    recent_activity: Dict[str, int] = Field(default_factory=dict, description="最近活动")
    rollback_success_rate: float = Field(default=0.0, description="回滚成功率")
    average_rollback_time_seconds: float = Field(default=0.0, description="平均回滚时间（秒）")
    milestone_count: int = Field(default=0, description="里程碑数量")
    expired_snapshots: int = Field(default=0, description="过期快照数")

# 响应模型
class SnapshotCreateResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    snapshot: Optional[SnapshotInfo] = Field(None, description="快照信息")

class SnapshotUpdateResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    snapshot: Optional[SnapshotInfo] = Field(None, description="快照信息")

class SnapshotDeleteResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")

class SnapshotListResponse(BaseModel):
    snapshots: List[SnapshotInfo] = Field(..., description="快照列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页")
    page_size: int = Field(..., description="每页大小")
    has_next: bool = Field(..., description="是否有下一页")
    statistics: Optional[TimeTravelStatistics] = Field(None, description="统计信息")

class RollbackResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    rollback: Optional[RollbackInfo] = Field(None, description="回滚信息")
    backup_snapshot_id: Optional[str] = Field(None, description="备份快照ID")

class HistoryViewResponse(BaseModel):
    timeline: Timeline = Field(..., description="时间线")
    total_events: int = Field(..., description="总事件数")
    page: int = Field(..., description="当前页")
    page_size: int = Field(..., description="每页大小")
    has_next: bool = Field(..., description="是否有下一页")

class BranchCreateResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    branch: Optional[BranchInfo] = Field(None, description="分支信息")

class BranchListResponse(BaseModel):
    branches: List[BranchInfo] = Field(..., description="分支列表")
    total: int = Field(..., description="总数")
    active_branches: int = Field(..., description="活跃分支数")
    merged_branches: int = Field(..., description="已合并分支数")

# 比较请求和响应
class CompareRequest(BaseModel):
    source_snapshot_id: str = Field(..., description="源快照ID")
    target_snapshot_id: str = Field(..., description="目标快照ID")
    components: Optional[List[StateComponentType]] = Field(None, description="要比较的组件")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    diff_format: str = Field(default="unified", description="差异格式")

class CompareResponse(BaseModel):
    source_snapshot: SnapshotInfo = Field(..., description="源快照")
    target_snapshot: SnapshotInfo = Field(..., description="目标快照")
    differences: List[Dict[str, Any]] = Field(..., description="差异列表")
    summary: Dict[str, Any] = Field(..., description="差异摘要")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="相似度分数")

# 导出请求和响应
class ExportRequest(BaseModel):
    snapshot_ids: Optional[List[str]] = Field(None, description="快照ID列表")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="日期范围")
    format: str = Field(default="json", description="导出格式")
    include_data: bool = Field(default=True, description="是否包含数据")
    compress: bool = Field(default=True, description="是否压缩")
    encrypt: bool = Field(default=False, description="是否加密")

class ExportResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    export_id: str = Field(..., description="导出ID")
    download_url: str = Field(..., description="下载链接")
    file_size: int = Field(..., description="文件大小")
    snapshot_count: int = Field(..., description="快照数量")
    expires_at: datetime = Field(..., description="过期时间")