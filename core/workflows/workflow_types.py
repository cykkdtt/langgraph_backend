"""
工作流类型定义

定义工作流相关的数据类型和枚举。
"""

from typing import Optional, List, Dict, Any, Union, Callable
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class WorkflowType(str, Enum):
    """工作流类型"""
    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"  # 并行执行
    CONDITIONAL = "conditional"  # 条件执行
    LOOP = "loop"  # 循环执行
    SUBGRAPH = "subgraph"  # 子图执行
    HYBRID = "hybrid"  # 混合执行


class ExecutionMode(str, Enum):
    """执行模式"""
    SYNC = "sync"  # 同步执行
    ASYNC = "async"  # 异步执行
    STREAM = "stream"  # 流式执行
    BATCH = "batch"  # 批量执行


class ConditionType(str, Enum):
    """条件类型"""
    SIMPLE = "simple"  # 简单条件
    COMPLEX = "complex"  # 复杂条件
    EXPRESSION = "expression"  # 表达式条件
    FUNCTION = "function"  # 函数条件


class StepStatus(str, Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    """工作流状态"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Condition(BaseModel):
    """条件定义"""
    type: ConditionType = Field(description="条件类型")
    expression: str = Field(description="条件表达式")
    variables: Dict[str, Any] = Field(default_factory=dict, description="变量")
    function: Optional[str] = Field(None, description="函数名称")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class WorkflowStep(BaseModel):
    """工作流步骤"""
    id: str = Field(description="步骤ID")
    name: str = Field(description="步骤名称")
    type: str = Field(description="步骤类型")
    description: Optional[str] = Field(None, description="步骤描述")
    
    # 执行配置
    function: Optional[str] = Field(None, description="执行函数")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="参数")
    timeout: Optional[int] = Field(None, description="超时时间（秒）")
    retry_count: int = Field(0, description="重试次数")
    
    # 条件配置
    condition: Optional[Condition] = Field(None, description="执行条件")
    skip_condition: Optional[Condition] = Field(None, description="跳过条件")
    
    # 依赖关系
    depends_on: List[str] = Field(default_factory=list, description="依赖步骤")
    next_steps: List[str] = Field(default_factory=list, description="下一步骤")
    
    # 状态信息
    status: StepStatus = Field(StepStatus.PENDING, description="步骤状态")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    error: Optional[str] = Field(None, description="错误信息")
    result: Optional[Any] = Field(None, description="执行结果")
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class SubgraphConfig(BaseModel):
    """子图配置"""
    name: str = Field(description="子图名称")
    description: Optional[str] = Field(None, description="子图描述")
    entry_point: str = Field(description="入口点")
    exit_points: List[str] = Field(description="出口点")
    
    # 输入输出配置
    input_schema: Optional[Dict[str, Any]] = Field(None, description="输入模式")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="输出模式")
    
    # 执行配置
    execution_mode: ExecutionMode = Field(ExecutionMode.ASYNC, description="执行模式")
    timeout: Optional[int] = Field(None, description="超时时间")
    max_iterations: Optional[int] = Field(None, description="最大迭代次数")
    
    # 资源配置
    resource_limits: Optional[Dict[str, Any]] = Field(None, description="资源限制")
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class WorkflowDefinition(BaseModel):
    """工作流定义"""
    id: str = Field(description="工作流ID")
    name: str = Field(description="工作流名称")
    description: Optional[str] = Field(None, description="工作流描述")
    version: str = Field(description="版本号")
    
    # 工作流配置
    type: WorkflowType = Field(description="工作流类型")
    execution_mode: ExecutionMode = Field(ExecutionMode.ASYNC, description="执行模式")
    
    # 步骤定义
    steps: List[WorkflowStep] = Field(description="工作流步骤")
    entry_point: str = Field(description="入口点")
    exit_points: List[str] = Field(description="出口点")
    
    # 子图配置
    subgraphs: Dict[str, SubgraphConfig] = Field(default_factory=dict, description="子图配置")
    
    # 全局配置
    global_timeout: Optional[int] = Field(None, description="全局超时时间")
    max_retries: int = Field(3, description="最大重试次数")
    error_handling: str = Field("stop", description="错误处理策略")
    
    # 变量和上下文
    variables: Dict[str, Any] = Field(default_factory=dict, description="全局变量")
    context_schema: Optional[Dict[str, Any]] = Field(None, description="上下文模式")
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    created_by: Optional[str] = Field(None, description="创建者")
    tags: List[str] = Field(default_factory=list, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class WorkflowExecution(BaseModel):
    """工作流执行实例"""
    id: str = Field(description="执行ID")
    workflow_id: str = Field(description="工作流ID")
    workflow_version: str = Field(description="工作流版本")
    
    # 执行状态
    status: WorkflowStatus = Field(WorkflowStatus.CREATED, description="执行状态")
    current_step: Optional[str] = Field(None, description="当前步骤")
    completed_steps: List[str] = Field(default_factory=list, description="已完成步骤")
    failed_steps: List[str] = Field(default_factory=list, description="失败步骤")
    
    # 时间信息
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    duration: Optional[float] = Field(None, description="执行时长（秒）")
    
    # 输入输出
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    output_data: Optional[Dict[str, Any]] = Field(None, description="输出数据")
    
    # 上下文和变量
    context: Dict[str, Any] = Field(default_factory=dict, description="执行上下文")
    variables: Dict[str, Any] = Field(default_factory=dict, description="变量值")
    
    # 错误信息
    error: Optional[str] = Field(None, description="错误信息")
    error_step: Optional[str] = Field(None, description="错误步骤")
    
    # 执行配置
    execution_config: Optional[Dict[str, Any]] = Field(None, description="执行配置")
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class ParallelTaskConfig(BaseModel):
    """并行任务配置"""
    task_id: str = Field(description="任务ID")
    function: str = Field(description="执行函数")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="参数")
    timeout: Optional[int] = Field(None, description="超时时间")
    retry_count: int = Field(0, description="重试次数")
    priority: int = Field(0, description="优先级")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class ConditionalBranch(BaseModel):
    """条件分支"""
    condition: Condition = Field(description="分支条件")
    target_step: str = Field(description="目标步骤")
    priority: int = Field(0, description="优先级")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class LoopConfig(BaseModel):
    """循环配置"""
    loop_type: str = Field(description="循环类型")  # for, while, until
    condition: Optional[Condition] = Field(None, description="循环条件")
    max_iterations: int = Field(100, description="最大迭代次数")
    break_condition: Optional[Condition] = Field(None, description="中断条件")
    continue_condition: Optional[Condition] = Field(None, description="继续条件")
    iteration_variable: Optional[str] = Field(None, description="迭代变量")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")