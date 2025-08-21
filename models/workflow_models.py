from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

# 工作流状态枚举
class WorkflowStatus(str, Enum):
    DRAFT = "draft"  # 草稿
    ACTIVE = "active"  # 活跃
    PAUSED = "paused"  # 暂停
    ARCHIVED = "archived"  # 归档
    DELETED = "deleted"  # 已删除

# 工作流类型枚举
class WorkflowType(str, Enum):
    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"  # 并行执行
    CONDITIONAL = "conditional"  # 条件执行
    LOOP = "loop"  # 循环执行
    CUSTOM = "custom"  # 自定义

# 执行状态枚举
class ExecutionStatus(str, Enum):
    PENDING = "pending"  # 等待中
    RUNNING = "running"  # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消
    TIMEOUT = "timeout"  # 超时

# 节点类型枚举
class NodeType(str, Enum):
    START = "start"  # 开始节点
    END = "end"  # 结束节点
    AGENT = "agent"  # 智能体节点
    TOOL = "tool"  # 工具节点
    CONDITION = "condition"  # 条件节点
    LOOP = "loop"  # 循环节点
    PARALLEL = "parallel"  # 并行节点
    MERGE = "merge"  # 合并节点

# 触发器类型枚举
class TriggerType(str, Enum):
    MANUAL = "manual"  # 手动触发
    SCHEDULED = "scheduled"  # 定时触发
    EVENT = "event"  # 事件触发
    WEBHOOK = "webhook"  # Webhook触发
    API = "api"  # API触发

# 工作流节点模型
class WorkflowNode(BaseModel):
    id: str = Field(..., description="节点ID")
    type: NodeType = Field(..., description="节点类型")
    name: str = Field(..., description="节点名称")
    description: Optional[str] = Field(None, description="节点描述")
    config: Dict[str, Any] = Field(default_factory=dict, description="节点配置")
    position: Dict[str, float] = Field(default_factory=dict, description="节点位置")
    inputs: List[str] = Field(default_factory=list, description="输入连接")
    outputs: List[str] = Field(default_factory=list, description="输出连接")
    conditions: Optional[Dict[str, Any]] = Field(None, description="条件配置")
    timeout: Optional[int] = Field(None, description="超时时间（秒）")
    retry_count: int = Field(default=0, description="重试次数")
    is_optional: bool = Field(default=False, description="是否可选")

# 工作流边模型
class WorkflowEdge(BaseModel):
    id: str = Field(..., description="边ID")
    source: str = Field(..., description="源节点ID")
    target: str = Field(..., description="目标节点ID")
    condition: Optional[str] = Field(None, description="边条件")
    label: Optional[str] = Field(None, description="边标签")
    config: Dict[str, Any] = Field(default_factory=dict, description="边配置")

# 工作流触发器模型
class WorkflowTrigger(BaseModel):
    type: TriggerType = Field(..., description="触发器类型")
    config: Dict[str, Any] = Field(default_factory=dict, description="触发器配置")
    enabled: bool = Field(default=True, description="是否启用")
    schedule: Optional[str] = Field(None, description="定时表达式")
    event_type: Optional[str] = Field(None, description="事件类型")
    webhook_url: Optional[str] = Field(None, description="Webhook URL")

# 工作流创建请求
class WorkflowCreateRequest(BaseModel):
    name: str = Field(..., description="工作流名称")
    description: Optional[str] = Field(None, description="工作流描述")
    type: WorkflowType = Field(..., description="工作流类型")
    nodes: List[WorkflowNode] = Field(..., description="工作流节点")
    edges: List[WorkflowEdge] = Field(..., description="工作流边")
    triggers: List[WorkflowTrigger] = Field(default_factory=list, description="触发器")
    config: Dict[str, Any] = Field(default_factory=dict, description="工作流配置")
    tags: List[str] = Field(default_factory=list, description="标签")
    is_public: bool = Field(default=False, description="是否公开")
    timeout: Optional[int] = Field(None, description="超时时间（秒）")
    max_retries: int = Field(default=3, description="最大重试次数")

# 工作流更新请求
class WorkflowUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, description="工作流名称")
    description: Optional[str] = Field(None, description="工作流描述")
    type: Optional[WorkflowType] = Field(None, description="工作流类型")
    nodes: Optional[List[WorkflowNode]] = Field(None, description="工作流节点")
    edges: Optional[List[WorkflowEdge]] = Field(None, description="工作流边")
    triggers: Optional[List[WorkflowTrigger]] = Field(None, description="触发器")
    config: Optional[Dict[str, Any]] = Field(None, description="工作流配置")
    tags: Optional[List[str]] = Field(None, description="标签")
    is_public: Optional[bool] = Field(None, description="是否公开")
    timeout: Optional[int] = Field(None, description="超时时间（秒）")
    max_retries: Optional[int] = Field(None, description="最大重试次数")
    status: Optional[WorkflowStatus] = Field(None, description="工作流状态")

# 工作流执行请求
class WorkflowExecuteRequest(BaseModel):
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    config: Dict[str, Any] = Field(default_factory=dict, description="执行配置")
    timeout: Optional[int] = Field(None, description="超时时间（秒）")
    priority: int = Field(default=5, description="优先级（1-10）")
    callback_url: Optional[str] = Field(None, description="回调URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

# 工作流搜索请求
class WorkflowSearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="搜索关键词")
    type: Optional[WorkflowType] = Field(None, description="工作流类型")
    status: Optional[WorkflowStatus] = Field(None, description="工作流状态")
    tags: Optional[List[str]] = Field(None, description="标签")
    created_by: Optional[str] = Field(None, description="创建者")
    created_after: Optional[datetime] = Field(None, description="创建时间起")
    created_before: Optional[datetime] = Field(None, description="创建时间止")
    is_public: Optional[bool] = Field(None, description="是否公开")
    page: int = Field(default=1, description="页码")
    page_size: int = Field(default=20, description="每页大小")
    sort_by: str = Field(default="created_at", description="排序字段")
    sort_order: str = Field(default="desc", description="排序方向")

# 执行节点状态
class ExecutionNodeStatus(BaseModel):
    node_id: str = Field(..., description="节点ID")
    status: ExecutionStatus = Field(..., description="执行状态")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    duration: Optional[float] = Field(None, description="执行时长（秒）")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="输出数据")
    error_message: Optional[str] = Field(None, description="错误信息")
    retry_count: int = Field(default=0, description="重试次数")
    logs: List[str] = Field(default_factory=list, description="执行日志")

# 工作流执行信息
class WorkflowExecutionInfo(BaseModel):
    id: str = Field(..., description="执行ID")
    workflow_id: str = Field(..., description="工作流ID")
    workflow_name: str = Field(..., description="工作流名称")
    status: ExecutionStatus = Field(..., description="执行状态")
    start_time: datetime = Field(..., description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    duration: Optional[float] = Field(None, description="执行时长（秒）")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="输出数据")
    error_message: Optional[str] = Field(None, description="错误信息")
    progress: float = Field(default=0.0, description="执行进度（0-100）")
    node_statuses: List[ExecutionNodeStatus] = Field(default_factory=list, description="节点状态")
    created_by: str = Field(..., description="执行者")
    priority: int = Field(default=5, description="优先级")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

# 工作流信息
class WorkflowInfo(BaseModel):
    id: str = Field(..., description="工作流ID")
    name: str = Field(..., description="工作流名称")
    description: Optional[str] = Field(None, description="工作流描述")
    type: WorkflowType = Field(..., description="工作流类型")
    status: WorkflowStatus = Field(..., description="工作流状态")
    nodes: List[WorkflowNode] = Field(..., description="工作流节点")
    edges: List[WorkflowEdge] = Field(..., description="工作流边")
    triggers: List[WorkflowTrigger] = Field(default_factory=list, description="触发器")
    config: Dict[str, Any] = Field(default_factory=dict, description="工作流配置")
    tags: List[str] = Field(default_factory=list, description="标签")
    is_public: bool = Field(default=False, description="是否公开")
    timeout: Optional[int] = Field(None, description="超时时间（秒）")
    max_retries: int = Field(default=3, description="最大重试次数")
    created_by: str = Field(..., description="创建者")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    version: int = Field(default=1, description="版本号")
    execution_count: int = Field(default=0, description="执行次数")
    success_count: int = Field(default=0, description="成功次数")
    last_execution: Optional[datetime] = Field(None, description="最后执行时间")

# 工作流统计信息
class WorkflowStatistics(BaseModel):
    total_executions: int = Field(default=0, description="总执行次数")
    successful_executions: int = Field(default=0, description="成功执行次数")
    failed_executions: int = Field(default=0, description="失败执行次数")
    average_duration: float = Field(default=0.0, description="平均执行时长")
    success_rate: float = Field(default=0.0, description="成功率")
    last_24h_executions: int = Field(default=0, description="24小时内执行次数")
    peak_execution_time: Optional[str] = Field(None, description="执行高峰时间")
    most_failed_node: Optional[str] = Field(None, description="最常失败的节点")

# 响应模型
class WorkflowCreateResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    workflow: Optional[WorkflowInfo] = Field(None, description="工作流信息")

class WorkflowUpdateResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    workflow: Optional[WorkflowInfo] = Field(None, description="工作流信息")

class WorkflowDeleteResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")

class WorkflowListResponse(BaseModel):
    workflows: List[WorkflowInfo] = Field(..., description="工作流列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页")
    page_size: int = Field(..., description="每页大小")
    has_next: bool = Field(..., description="是否有下一页")

class WorkflowExecuteResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    execution: Optional[WorkflowExecutionInfo] = Field(None, description="执行信息")

class WorkflowExecutionListResponse(BaseModel):
    executions: List[WorkflowExecutionInfo] = Field(..., description="执行列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页")
    page_size: int = Field(..., description="每页大小")
    has_next: bool = Field(..., description="是否有下一页")

# 工作流模板模型
class WorkflowTemplate(BaseModel):
    id: str = Field(..., description="模板ID")
    name: str = Field(..., description="模板名称")
    description: str = Field(..., description="模板描述")
    category: str = Field(..., description="模板分类")
    type: WorkflowType = Field(..., description="工作流类型")
    nodes: List[WorkflowNode] = Field(..., description="节点模板")
    edges: List[WorkflowEdge] = Field(..., description="边模板")
    config: Dict[str, Any] = Field(default_factory=dict, description="配置模板")
    tags: List[str] = Field(default_factory=list, description="标签")
    is_featured: bool = Field(default=False, description="是否推荐")
    usage_count: int = Field(default=0, description="使用次数")
    rating: float = Field(default=0.0, description="评分")
    created_by: str = Field(..., description="创建者")
    created_at: datetime = Field(..., description="创建时间")

# 批量操作请求
class WorkflowBatchOperation(BaseModel):
    operation: str = Field(..., description="操作类型")
    workflow_ids: List[str] = Field(..., description="工作流ID列表")
    params: Dict[str, Any] = Field(default_factory=dict, description="操作参数")

class WorkflowBatchOperationResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="操作结果")
    failed_ids: List[str] = Field(default_factory=list, description="失败的ID")