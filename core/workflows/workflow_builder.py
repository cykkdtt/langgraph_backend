"""
工作流构建器

提供复杂工作流的构建和管理功能。
"""

import asyncio
from typing import Optional, List, Dict, Any, Callable, Union
from datetime import datetime

from .workflow_types import (
    WorkflowDefinition, WorkflowStep, WorkflowExecution, SubgraphConfig,
    WorkflowType, ExecutionMode, StepStatus, WorkflowStatus,
    Condition, ConditionalBranch, ParallelTaskConfig, LoopConfig
)


class WorkflowBuilder:
    """工作流构建器"""
    
    def __init__(self):
        self._steps: List[WorkflowStep] = []
        self._conditions: List[Condition] = []
        self._parallel_configs: List[ParallelTaskConfig] = []
        self._loop_configs: List[LoopConfig] = []
        self._metadata: Dict[str, Any] = {}
        
        # 当前构建状态
        self._current_step_id: Optional[str] = None
        self._step_counter: int = 0
    
    def add_step(
        self,
        name: str,
        step_type: str = "action",
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "WorkflowBuilder":
        """添加工作流步骤"""
        self._step_counter += 1
        step_id = f"step_{self._step_counter}"
        
        step = WorkflowStep(
            id=step_id,
            name=name,
            type=step_type,
            config=config or {},
            dependencies=dependencies or [],
            conditions=conditions or [],
            timeout=timeout,
            retry_config=retry_config,
            metadata=metadata or {}
        )
        
        self._steps.append(step)
        self._current_step_id = step_id
        
        return self
    
    def add_condition(
        self,
        name: str,
        condition_type: str,
        expression: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "WorkflowBuilder":
        """添加条件"""
        condition_id = f"cond_{len(self._conditions) + 1}"
        
        condition = Condition(
            id=condition_id,
            name=name,
            type=condition_type,
            expression=expression,
            description=description,
            metadata=metadata or {}
        )
        
        self._conditions.append(condition)
        
        return self
    
    def add_parallel_task(
        self,
        name: str,
        tasks: List[str],
        max_concurrency: Optional[int] = None,
        wait_for_all: bool = True,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "WorkflowBuilder":
        """添加并行任务配置"""
        config_id = f"parallel_{len(self._parallel_configs) + 1}"
        
        parallel_config = ParallelTaskConfig(
            id=config_id,
            name=name,
            tasks=tasks,
            max_concurrency=max_concurrency,
            wait_for_all=wait_for_all,
            timeout=timeout,
            metadata=metadata or {}
        )
        
        self._parallel_configs.append(parallel_config)
        
        return self
    
    def add_loop(
        self,
        name: str,
        loop_type: str,
        condition: str,
        max_iterations: Optional[int] = None,
        break_condition: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "WorkflowBuilder":
        """添加循环配置"""
        loop_id = f"loop_{len(self._loop_configs) + 1}"
        
        loop_config = LoopConfig(
            id=loop_id,
            name=name,
            type=loop_type,
            condition=condition,
            max_iterations=max_iterations,
            break_condition=break_condition,
            metadata=metadata or {}
        )
        
        self._loop_configs.append(loop_config)
        
        return self
    
    def add_conditional_branch(
        self,
        condition_id: str,
        true_steps: List[str],
        false_steps: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "WorkflowBuilder":
        """添加条件分支"""
        branch_id = f"branch_{datetime.now().timestamp()}"
        
        branch = ConditionalBranch(
            id=branch_id,
            condition_id=condition_id,
            true_steps=true_steps,
            false_steps=false_steps or [],
            metadata=metadata or {}
        )
        
        # 将分支信息添加到元数据
        if "conditional_branches" not in self._metadata:
            self._metadata["conditional_branches"] = []
        self._metadata["conditional_branches"].append(branch.dict())
        
        return self
    
    def add_subgraph(
        self,
        name: str,
        subgraph_config: SubgraphConfig,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "WorkflowBuilder":
        """添加子图步骤"""
        return self.add_step(
            name=name,
            step_type="subgraph",
            config={
                "subgraph_config": subgraph_config.dict(),
                "input_mapping": input_mapping or {},
                "output_mapping": output_mapping or {}
            },
            metadata=metadata
        )
    
    def set_metadata(self, key: str, value: Any) -> "WorkflowBuilder":
        """设置元数据"""
        self._metadata[key] = value
        return self
    
    def build(
        self,
        name: str,
        description: Optional[str] = None,
        workflow_type: WorkflowType = WorkflowType.SEQUENTIAL,
        execution_mode: ExecutionMode = ExecutionMode.SYNC,
        version: str = "1.0.0"
    ) -> WorkflowDefinition:
        """构建工作流定义"""
        if not self._steps:
            raise ValueError("工作流必须包含至少一个步骤")
        
        # 验证依赖关系
        self._validate_dependencies()
        
        # 验证条件引用
        self._validate_conditions()
        
        workflow_id = f"workflow_{datetime.now().timestamp()}"
        
        workflow = WorkflowDefinition(
            id=workflow_id,
            name=name,
            description=description,
            type=workflow_type,
            execution_mode=execution_mode,
            version=version,
            steps=self._steps,
            conditions=self._conditions,
            parallel_configs=self._parallel_configs,
            loop_configs=self._loop_configs,
            created_at=datetime.now(),
            metadata=self._metadata
        )
        
        return workflow
    
    def reset(self) -> "WorkflowBuilder":
        """重置构建器"""
        self._steps.clear()
        self._conditions.clear()
        self._parallel_configs.clear()
        self._loop_configs.clear()
        self._metadata.clear()
        self._current_step_id = None
        self._step_counter = 0
        
        return self
    
    def _validate_dependencies(self) -> None:
        """验证步骤依赖关系"""
        step_ids = {step.id for step in self._steps}
        
        for step in self._steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise ValueError(f"步骤 {step.name} 依赖的步骤 {dep} 不存在")
        
        # 检查循环依赖
        self._check_circular_dependencies()
    
    def _validate_conditions(self) -> None:
        """验证条件引用"""
        condition_ids = {cond.id for cond in self._conditions}
        
        for step in self._steps:
            for cond_id in step.conditions:
                if cond_id not in condition_ids:
                    raise ValueError(f"步骤 {step.name} 引用的条件 {cond_id} 不存在")
    
    def _check_circular_dependencies(self) -> None:
        """检查循环依赖"""
        # 使用拓扑排序检查循环依赖
        in_degree = {step.id: 0 for step in self._steps}
        graph = {step.id: [] for step in self._steps}
        
        # 构建依赖图
        for step in self._steps:
            for dep in step.dependencies:
                graph[dep].append(step.id)
                in_degree[step.id] += 1
        
        # 拓扑排序
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        processed = 0
        
        while queue:
            current = queue.pop(0)
            processed += 1
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if processed != len(self._steps):
            raise ValueError("工作流中存在循环依赖")


class WorkflowTemplate:
    """工作流模板"""
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description
        self._templates: Dict[str, Callable[[WorkflowBuilder], WorkflowBuilder]] = {}
    
    def register_template(
        self,
        template_name: str,
        builder_func: Callable[[WorkflowBuilder], WorkflowBuilder]
    ) -> None:
        """注册工作流模板"""
        self._templates[template_name] = builder_func
    
    def create_workflow(
        self,
        template_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> WorkflowDefinition:
        """从模板创建工作流"""
        if template_name not in self._templates:
            raise ValueError(f"模板不存在: {template_name}")
        
        builder = WorkflowBuilder()
        
        # 应用模板
        template_func = self._templates[template_name]
        configured_builder = template_func(builder)
        
        # 应用参数
        if parameters:
            self._apply_parameters(configured_builder, parameters)
        
        return configured_builder.build(
            name=f"{self.name}_{template_name}",
            description=f"从模板 {template_name} 创建的工作流"
        )
    
    def list_templates(self) -> List[str]:
        """列出所有模板"""
        return list(self._templates.keys())
    
    def _apply_parameters(
        self,
        builder: WorkflowBuilder,
        parameters: Dict[str, Any]
    ) -> None:
        """应用参数到构建器"""
        # 简化实现，实际应用中可能需要更复杂的参数替换逻辑
        for key, value in parameters.items():
            builder.set_metadata(f"param_{key}", value)


# 预定义模板
def create_sequential_template() -> Callable[[WorkflowBuilder], WorkflowBuilder]:
    """创建顺序执行模板"""
    def template(builder: WorkflowBuilder) -> WorkflowBuilder:
        return (builder
                .add_step("初始化", "init")
                .add_step("处理", "process", dependencies=["step_1"])
                .add_step("完成", "finish", dependencies=["step_2"]))
    
    return template


def create_parallel_template() -> Callable[[WorkflowBuilder], WorkflowBuilder]:
    """创建并行执行模板"""
    def template(builder: WorkflowBuilder) -> WorkflowBuilder:
        return (builder
                .add_step("初始化", "init")
                .add_parallel_task("并行处理", ["task1", "task2", "task3"])
                .add_step("任务1", "task", dependencies=["step_1"])
                .add_step("任务2", "task", dependencies=["step_1"])
                .add_step("任务3", "task", dependencies=["step_1"])
                .add_step("汇总", "aggregate", dependencies=["step_3", "step_4", "step_5"]))
    
    return template


def create_conditional_template() -> Callable[[WorkflowBuilder], WorkflowBuilder]:
    """创建条件执行模板"""
    def template(builder: WorkflowBuilder) -> WorkflowBuilder:
        return (builder
                .add_step("检查条件", "check")
                .add_condition("主条件", "simple", "result == 'success'")
                .add_step("成功处理", "success_handler", conditions=["cond_1"])
                .add_step("失败处理", "failure_handler")
                .add_conditional_branch("cond_1", ["step_2"], ["step_3"])
                .add_step("完成", "finish", dependencies=["step_2", "step_3"]))
    
    return template


def create_loop_template() -> Callable[[WorkflowBuilder], WorkflowBuilder]:
    """创建循环执行模板"""
    def template(builder: WorkflowBuilder) -> WorkflowBuilder:
        return (builder
                .add_step("初始化", "init")
                .add_loop("主循环", "while", "counter < max_iterations", max_iterations=10)
                .add_step("循环体", "loop_body", dependencies=["step_1"])
                .add_step("完成", "finish", dependencies=["step_2"]))
    
    return template