"""
子图管理器

提供子图的创建、管理和执行功能。
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from .workflow_types import (
    SubgraphConfig, WorkflowExecution, WorkflowStatus,
    ExecutionMode, StepStatus
)


logger = logging.getLogger(__name__)


class SubgraphManager:
    """子图管理器"""
    
    def __init__(self):
        self.subgraphs: Dict[str, CompiledStateGraph] = {}
        self.subgraph_configs: Dict[str, SubgraphConfig] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_handlers: List[Callable] = []
        
    async def create_subgraph(
        self,
        config: SubgraphConfig,
        nodes: Dict[str, Callable],
        edges: List[tuple],
        conditional_edges: Optional[List[tuple]] = None
    ) -> str:
        """创建子图"""
        try:
            # 创建状态图
            graph = StateGraph(dict)
            
            # 添加节点
            for node_name, node_func in nodes.items():
                graph.add_node(node_name, node_func)
            
            # 添加边
            for edge in edges:
                if len(edge) == 2:
                    graph.add_edge(edge[0], edge[1])
                elif len(edge) == 3:
                    # 带条件的边
                    graph.add_conditional_edges(edge[0], edge[1], edge[2])
            
            # 添加条件边
            if conditional_edges:
                for cond_edge in conditional_edges:
                    graph.add_conditional_edges(cond_edge[0], cond_edge[1], cond_edge[2])
            
            # 设置入口点
            graph.set_entry_point(config.entry_point)
            
            # 设置结束点
            for exit_point in config.exit_points:
                graph.set_finish_point(exit_point)
            
            # 编译图
            compiled_graph = graph.compile()
            
            # 存储子图
            self.subgraphs[config.name] = compiled_graph
            self.subgraph_configs[config.name] = config
            
            logger.info(f"子图已创建: {config.name}")
            
            return config.name
            
        except Exception as e:
            logger.error(f"创建子图失败: {e}")
            raise
    
    async def execute_subgraph(
        self,
        subgraph_name: str,
        input_data: Dict[str, Any],
        execution_config: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """执行子图"""
        if subgraph_name not in self.subgraphs:
            raise ValueError(f"子图不存在: {subgraph_name}")
        
        execution_id = str(uuid4())
        config = self.subgraph_configs[subgraph_name]
        graph = self.subgraphs[subgraph_name]
        
        # 创建执行实例
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=subgraph_name,
            workflow_version="1.0",
            status=WorkflowStatus.RUNNING,
            input_data=input_data,
            started_at=datetime.utcnow(),
            execution_config=execution_config or {}
        )
        
        self.active_executions[execution_id] = execution
        
        try:
            # 触发执行处理器
            await self._trigger_execution_handlers("started", execution)
            
            # 执行子图
            if config.execution_mode == ExecutionMode.STREAM:
                result = await self._execute_streaming(graph, input_data, execution)
            else:
                result = await self._execute_standard(graph, input_data, execution)
            
            # 更新执行状态
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.output_data = result
            execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            
            # 触发完成处理器
            await self._trigger_execution_handlers("completed", execution)
            
            logger.info(f"子图执行完成: {subgraph_name}, 执行ID: {execution_id}")
            
        except Exception as e:
            # 更新执行状态
            execution.status = WorkflowStatus.FAILED
            execution.completed_at = datetime.utcnow()
            execution.error = str(e)
            execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            
            # 触发错误处理器
            await self._trigger_execution_handlers("failed", execution)
            
            logger.error(f"子图执行失败: {subgraph_name}, 错误: {e}")
            raise
        
        finally:
            # 清理活跃执行
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
        
        return execution
    
    async def stream_subgraph(
        self,
        subgraph_name: str,
        input_data: Dict[str, Any],
        execution_config: Optional[Dict[str, Any]] = None
    ):
        """流式执行子图"""
        if subgraph_name not in self.subgraphs:
            raise ValueError(f"子图不存在: {subgraph_name}")
        
        graph = self.subgraphs[subgraph_name]
        
        try:
            async for chunk in graph.astream(input_data):
                yield chunk
        except Exception as e:
            logger.error(f"子图流式执行失败: {subgraph_name}, 错误: {e}")
            raise
    
    async def pause_execution(self, execution_id: str) -> bool:
        """暂停执行"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        execution.status = WorkflowStatus.PAUSED
        
        # 触发暂停处理器
        await self._trigger_execution_handlers("paused", execution)
        
        logger.info(f"执行已暂停: {execution_id}")
        return True
    
    async def resume_execution(self, execution_id: str) -> bool:
        """恢复执行"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        execution.status = WorkflowStatus.RUNNING
        
        # 触发恢复处理器
        await self._trigger_execution_handlers("resumed", execution)
        
        logger.info(f"执行已恢复: {execution_id}")
        return True
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.utcnow()
        
        # 触发取消处理器
        await self._trigger_execution_handlers("cancelled", execution)
        
        # 清理活跃执行
        del self.active_executions[execution_id]
        
        logger.info(f"执行已取消: {execution_id}")
        return True
    
    def get_subgraph(self, name: str) -> Optional[CompiledStateGraph]:
        """获取子图"""
        return self.subgraphs.get(name)
    
    def get_subgraph_config(self, name: str) -> Optional[SubgraphConfig]:
        """获取子图配置"""
        return self.subgraph_configs.get(name)
    
    def list_subgraphs(self) -> List[str]:
        """列出所有子图"""
        return list(self.subgraphs.keys())
    
    def get_active_executions(self) -> List[WorkflowExecution]:
        """获取活跃执行"""
        return list(self.active_executions.values())
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """获取执行实例"""
        return self.active_executions.get(execution_id)
    
    async def remove_subgraph(self, name: str) -> bool:
        """移除子图"""
        if name not in self.subgraphs:
            return False
        
        # 检查是否有活跃执行
        active_executions = [
            exec for exec in self.active_executions.values()
            if exec.workflow_id == name
        ]
        
        if active_executions:
            logger.warning(f"子图有活跃执行，无法移除: {name}")
            return False
        
        # 移除子图
        del self.subgraphs[name]
        del self.subgraph_configs[name]
        
        logger.info(f"子图已移除: {name}")
        return True
    
    def register_execution_handler(self, handler: Callable):
        """注册执行处理器"""
        self.execution_handlers.append(handler)
    
    async def _execute_standard(
        self,
        graph: CompiledStateGraph,
        input_data: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """标准执行"""
        result = await graph.ainvoke(input_data)
        return result
    
    async def _execute_streaming(
        self,
        graph: CompiledStateGraph,
        input_data: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """流式执行"""
        result = {}
        async for chunk in graph.astream(input_data):
            result.update(chunk)
            
            # 更新执行上下文
            execution.context.update(chunk)
        
        return result
    
    async def _trigger_execution_handlers(self, event: str, execution: WorkflowExecution):
        """触发执行处理器"""
        for handler in self.execution_handlers:
            try:
                await handler(event, execution)
            except Exception as e:
                logger.error(f"执行处理器失败: {e}")
    
    async def create_nested_subgraph(
        self,
        parent_name: str,
        child_config: SubgraphConfig,
        nodes: Dict[str, Callable],
        edges: List[tuple],
        conditional_edges: Optional[List[tuple]] = None
    ) -> str:
        """创建嵌套子图"""
        if parent_name not in self.subgraphs:
            raise ValueError(f"父子图不存在: {parent_name}")
        
        # 创建子图名称
        child_name = f"{parent_name}.{child_config.name}"
        child_config.name = child_name
        
        # 创建子图
        return await self.create_subgraph(child_config, nodes, edges, conditional_edges)
    
    async def compose_subgraphs(
        self,
        composition_name: str,
        subgraph_names: List[str],
        composition_edges: List[tuple]
    ) -> str:
        """组合多个子图"""
        # 验证所有子图存在
        for name in subgraph_names:
            if name not in self.subgraphs:
                raise ValueError(f"子图不存在: {name}")
        
        # 创建组合图
        graph = StateGraph(dict)
        
        # 添加子图作为节点
        for subgraph_name in subgraph_names:
            subgraph = self.subgraphs[subgraph_name]
            graph.add_node(subgraph_name, subgraph.invoke)
        
        # 添加组合边
        for edge in composition_edges:
            graph.add_edge(edge[0], edge[1])
        
        # 设置入口点（第一个子图）
        graph.set_entry_point(subgraph_names[0])
        
        # 设置结束点（最后一个子图）
        graph.set_finish_point(subgraph_names[-1])
        
        # 编译组合图
        compiled_graph = graph.compile()
        
        # 创建组合配置
        composition_config = SubgraphConfig(
            name=composition_name,
            description=f"组合子图: {', '.join(subgraph_names)}",
            entry_point=subgraph_names[0],
            exit_points=[subgraph_names[-1]]
        )
        
        # 存储组合图
        self.subgraphs[composition_name] = compiled_graph
        self.subgraph_configs[composition_name] = composition_config
        
        logger.info(f"子图组合已创建: {composition_name}")
        
        return composition_name