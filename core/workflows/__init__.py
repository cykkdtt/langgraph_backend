"""子图和复杂工作流模块

提供子图管理、工作流构建、条件路由和并行执行功能。"""

from .subgraph_manager import SubgraphManager
from .workflow_builder import WorkflowBuilder, WorkflowTemplate
from .conditional_router import ConditionalRouter, ConditionEvaluator
from .parallel_executor import ParallelExecutor, ParallelTask
from .workflow_types import (
    WorkflowType, ExecutionMode, ConditionType,
    SubgraphConfig, WorkflowStep, WorkflowDefinition, Condition
)

# 全局工作流管理器实例
_workflow_manager = None

def get_workflow_manager() -> SubgraphManager:
    """获取全局工作流管理器实例"""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = SubgraphManager()
    return _workflow_manager

__all__ = [
    # 管理器类
    "SubgraphManager",
    "WorkflowBuilder",
    "WorkflowTemplate", 
    "ConditionalRouter",
    "ConditionEvaluator",
    "ParallelExecutor",
    "ParallelTask",
    
    # 类型和模型
    "WorkflowType",
    "ExecutionMode", 
    "ConditionType",
    "SubgraphConfig",
    "WorkflowStep",
    "WorkflowDefinition",
    "Condition",
    
    # 全局实例获取函数
    "get_workflow_manager"
]