"""
智能体协作优化

基于LangGraph最佳实践的多智能体协作模式实现。
参考: https://langchain-ai.github.io/langgraph/multi-agent/
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from ..agents.base import BaseAgent, AgentState, AgentType, AgentStatus


class CollaborationMode(str, Enum):
    """协作模式"""
    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"     # 并行执行
    HIERARCHICAL = "hierarchical"  # 层次化
    PEER_TO_PEER = "peer_to_peer"  # 点对点
    PIPELINE = "pipeline"     # 流水线
    CONSENSUS = "consensus"   # 共识决策


class MessageType(str, Enum):
    """消息类型"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    COLLABORATION_REQUEST = "collaboration_request"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    COMPLETION_NOTICE = "completion_notice"


@dataclass
class CollaborationMessage:
    """协作消息"""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: MessageType = MessageType.TASK_REQUEST
    sender_id: str = ""
    receiver_id: str = ""
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 0  # 0=低, 1=中, 2=高
    requires_response: bool = False
    correlation_id: Optional[str] = None


@dataclass
class CollaborationTask:
    """协作任务"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    assigned_agents: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationContext:
    """协作上下文"""
    session_id: str
    user_id: str
    collaboration_mode: CollaborationMode
    participating_agents: Set[str] = field(default_factory=set)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    message_history: List[CollaborationMessage] = field(default_factory=list)
    active_tasks: Dict[str, CollaborationTask] = field(default_factory=dict)
    completed_tasks: List[CollaborationTask] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class AgentCollaborationOrchestrator:
    """智能体协作编排器
    
    基于LangGraph最佳实践，实现：
    - 多种协作模式支持
    - 智能任务分配
    - 动态负载均衡
    - 故障恢复机制
    - 协作性能优化
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.collaboration_contexts: Dict[str, CollaborationContext] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.task_scheduler = TaskScheduler()
        self.load_balancer = LoadBalancer()
        self.logger = logging.getLogger("collaboration.orchestrator")
        
        # 协作策略
        self.collaboration_strategies: Dict[CollaborationMode, Callable] = {
            CollaborationMode.SEQUENTIAL: self._execute_sequential,
            CollaborationMode.PARALLEL: self._execute_parallel,
            CollaborationMode.HIERARCHICAL: self._execute_hierarchical,
            CollaborationMode.PEER_TO_PEER: self._execute_peer_to_peer,
            CollaborationMode.PIPELINE: self._execute_pipeline,
            CollaborationMode.CONSENSUS: self._execute_consensus,
        }
        
        # 消息处理器任务引用
        self._message_processor_task = None
        self._is_running = False
    
    async def start(self):
        """启动协作编排器"""
        if not self._is_running:
            try:
                self._message_processor_task = asyncio.create_task(self._message_processor())
                self._is_running = True
                self.logger.info("协作编排器已启动")
            except RuntimeError:
                # 如果没有运行的事件循环，则延迟启动
                self.logger.warning("没有运行的事件循环，消息处理器将在需要时启动")
    
    async def stop(self):
        """停止协作编排器"""
        if self._is_running and self._message_processor_task:
            self._message_processor_task.cancel()
            try:
                await self._message_processor_task
            except asyncio.CancelledError:
                pass
            self._is_running = False
            self.logger.info("协作编排器已停止")
    
    def _ensure_started(self):
        """确保消息处理器已启动"""
        if not self._is_running:
            try:
                loop = asyncio.get_running_loop()
                if loop and not loop.is_closed():
                    asyncio.create_task(self.start())
            except RuntimeError:
                # 没有运行的事件循环，稍后再启动
                pass
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """注册智能体"""
        try:
            self._ensure_started()  # 确保消息处理器已启动
            self.agents[agent.agent_id] = agent
            self.logger.info(f"智能体注册成功: {agent.agent_id}")
            return True
        except Exception as e:
            self.logger.error(f"智能体注册失败: {e}")
            return False
    
    async def create_collaboration_session(
        self,
        session_id: str,
        user_id: str,
        mode: CollaborationMode,
        participating_agents: List[str]
    ) -> CollaborationContext:
        """创建协作会话"""
        context = CollaborationContext(
            session_id=session_id,
            user_id=user_id,
            collaboration_mode=mode,
            participating_agents=set(participating_agents)
        )
        
        self.collaboration_contexts[session_id] = context
        self.logger.info(f"创建协作会话: {session_id}, 模式: {mode}, 智能体: {participating_agents}")
        
        return context
    
    async def execute_collaborative_task(
        self,
        session_id: str,
        task_description: str,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行协作任务"""
        if session_id not in self.collaboration_contexts:
            raise ValueError(f"协作会话不存在: {session_id}")
        
        context = self.collaboration_contexts[session_id]
        
        # 创建任务
        task = CollaborationTask(
            name=f"collaborative_task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            description=task_description,
            assigned_agents=list(context.participating_agents),
            metadata=task_data
        )
        
        context.active_tasks[task.id] = task
        
        try:
            # 根据协作模式执行任务
            strategy = self.collaboration_strategies[context.collaboration_mode]
            result = await strategy(context, task)
            
            # 标记任务完成
            task.status = "completed"
            task.result = result
            task.completed_at = datetime.utcnow()
            
            # 移动到已完成任务
            context.completed_tasks.append(task)
            del context.active_tasks[task.id]
            
            return {
                "success": True,
                "result": result,
                "task_id": task.id,
                "execution_time": (task.completed_at - task.created_at).total_seconds()
            }
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            
            self.logger.error(f"协作任务执行失败: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "task_id": task.id
            }
    
    async def _execute_sequential(
        self,
        context: CollaborationContext,
        task: CollaborationTask
    ) -> Any:
        """顺序执行模式"""
        results = []
        current_input = task.metadata.get("input", {})
        
        for agent_id in task.assigned_agents:
            if agent_id not in self.agents:
                continue
            
            agent = self.agents[agent_id]
            
            # 发送任务消息
            message = CollaborationMessage(
                type=MessageType.TASK_REQUEST,
                sender_id="orchestrator",
                receiver_id=agent_id,
                content={
                    "task_description": task.description,
                    "input_data": current_input,
                    "previous_results": results
                }
            )
            
            # 执行任务
            result = await self._send_message_and_wait_response(message, context)
            results.append({
                "agent_id": agent_id,
                "result": result
            })
            
            # 将结果作为下一个智能体的输入
            current_input = result
        
        return results
    
    async def _execute_parallel(
        self,
        context: CollaborationContext,
        task: CollaborationTask
    ) -> Any:
        """并行执行模式"""
        tasks = []
        input_data = task.metadata.get("input", {})
        
        for agent_id in task.assigned_agents:
            if agent_id not in self.agents:
                continue
            
            message = CollaborationMessage(
                type=MessageType.TASK_REQUEST,
                sender_id="orchestrator",
                receiver_id=agent_id,
                content={
                    "task_description": task.description,
                    "input_data": input_data
                }
            )
            
            task_coroutine = self._send_message_and_wait_response(message, context)
            tasks.append((agent_id, task_coroutine))
        
        # 并行执行所有任务
        results = []
        for agent_id, task_coroutine in tasks:
            try:
                result = await task_coroutine
                results.append({
                    "agent_id": agent_id,
                    "result": result,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "agent_id": agent_id,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    async def _execute_hierarchical(
        self,
        context: CollaborationContext,
        task: CollaborationTask
    ) -> Any:
        """层次化执行模式"""
        # 假设第一个智能体是主管理者
        supervisor_id = task.assigned_agents[0]
        worker_ids = task.assigned_agents[1:]
        
        # 主管理者分解任务
        supervisor_message = CollaborationMessage(
            type=MessageType.TASK_REQUEST,
            sender_id="orchestrator",
            receiver_id=supervisor_id,
            content={
                "task_description": task.description,
                "input_data": task.metadata.get("input", {}),
                "worker_agents": worker_ids,
                "mode": "supervisor"
            }
        )
        
        # 获取任务分解结果
        decomposition_result = await self._send_message_and_wait_response(
            supervisor_message, context
        )
        
        # 分配子任务给工作智能体
        subtask_results = []
        if isinstance(decomposition_result, dict) and "subtasks" in decomposition_result:
            for i, subtask in enumerate(decomposition_result["subtasks"]):
                if i < len(worker_ids):
                    worker_id = worker_ids[i]
                    worker_message = CollaborationMessage(
                        type=MessageType.TASK_REQUEST,
                        sender_id=supervisor_id,
                        receiver_id=worker_id,
                        content=subtask
                    )
                    
                    result = await self._send_message_and_wait_response(
                        worker_message, context
                    )
                    subtask_results.append({
                        "worker_id": worker_id,
                        "subtask": subtask,
                        "result": result
                    })
        
        # 主管理者整合结果
        integration_message = CollaborationMessage(
            type=MessageType.TASK_REQUEST,
            sender_id="orchestrator",
            receiver_id=supervisor_id,
            content={
                "mode": "integration",
                "subtask_results": subtask_results,
                "original_task": task.description
            }
        )
        
        final_result = await self._send_message_and_wait_response(
            integration_message, context
        )
        
        return {
            "supervisor_result": final_result,
            "subtask_results": subtask_results,
            "decomposition": decomposition_result
        }
    
    async def _execute_peer_to_peer(
        self,
        context: CollaborationContext,
        task: CollaborationTask
    ) -> Any:
        """点对点执行模式"""
        # 实现智能体间的直接通信和协作
        # 这里简化实现，实际可以更复杂
        return await self._execute_parallel(context, task)
    
    async def _execute_pipeline(
        self,
        context: CollaborationContext,
        task: CollaborationTask
    ) -> Any:
        """流水线执行模式"""
        # 类似顺序执行，但每个阶段可以有多个智能体并行处理
        pipeline_stages = task.metadata.get("pipeline_stages", [])
        if not pipeline_stages:
            # 默认将智能体分为阶段
            agents_per_stage = max(1, len(task.assigned_agents) // 3)
            pipeline_stages = [
                task.assigned_agents[i:i+agents_per_stage]
                for i in range(0, len(task.assigned_agents), agents_per_stage)
            ]
        
        current_input = task.metadata.get("input", {})
        stage_results = []
        
        for stage_index, stage_agents in enumerate(pipeline_stages):
            # 并行执行当前阶段的所有智能体
            stage_tasks = []
            for agent_id in stage_agents:
                if agent_id not in self.agents:
                    continue
                
                message = CollaborationMessage(
                    type=MessageType.TASK_REQUEST,
                    sender_id="orchestrator",
                    receiver_id=agent_id,
                    content={
                        "task_description": f"Pipeline Stage {stage_index + 1}: {task.description}",
                        "input_data": current_input,
                        "stage_index": stage_index
                    }
                )
                
                stage_tasks.append(
                    self._send_message_and_wait_response(message, context)
                )
            
            # 等待当前阶段完成
            stage_result = await asyncio.gather(*stage_tasks, return_exceptions=True)
            stage_results.append({
                "stage_index": stage_index,
                "agents": stage_agents,
                "results": stage_result
            })
            
            # 将阶段结果作为下一阶段的输入
            current_input = {
                "previous_stage_results": stage_result,
                "accumulated_results": stage_results
            }
        
        return {
            "pipeline_results": stage_results,
            "final_output": current_input
        }
    
    async def _execute_consensus(
        self,
        context: CollaborationContext,
        task: CollaborationTask
    ) -> Any:
        """共识决策模式"""
        # 所有智能体独立处理任务
        parallel_results = await self._execute_parallel(context, task)
        
        # 收集所有结果进行共识决策
        consensus_input = {
            "task_description": task.description,
            "all_results": parallel_results,
            "consensus_method": task.metadata.get("consensus_method", "majority_vote")
        }
        
        # 选择一个智能体作为共识协调者（通常是第一个）
        coordinator_id = task.assigned_agents[0]
        consensus_message = CollaborationMessage(
            type=MessageType.TASK_REQUEST,
            sender_id="orchestrator",
            receiver_id=coordinator_id,
            content={
                "mode": "consensus_coordination",
                **consensus_input
            }
        )
        
        consensus_result = await self._send_message_and_wait_response(
            consensus_message, context
        )
        
        return {
            "individual_results": parallel_results,
            "consensus_result": consensus_result,
            "consensus_method": task.metadata.get("consensus_method", "majority_vote")
        }
    
    async def _send_message_and_wait_response(
        self,
        message: CollaborationMessage,
        context: CollaborationContext,
        timeout: float = 30.0
    ) -> Any:
        """发送消息并等待响应"""
        # 添加到消息历史
        context.message_history.append(message)
        
        # 发送到消息队列
        await self.message_queue.put((message, context))
        
        # 等待响应（简化实现，实际需要更复杂的响应匹配机制）
        # 这里直接调用智能体
        if message.receiver_id in self.agents:
            agent = self.agents[message.receiver_id]
            
            # 构造智能体输入
            agent_input = {
                "messages": [HumanMessage(content=str(message.content))],
                "collaboration_context": context.shared_state
            }
            
            # 调用智能体
            result = await agent.chat(agent_input)
            return result.get("messages", [])[-1].content if result.get("messages") else None
        
        return None
    
    async def _message_processor(self):
        """消息处理器"""
        while True:
            try:
                message, context = await self.message_queue.get()
                # 处理消息的逻辑
                self.logger.debug(f"处理消息: {message.type} from {message.sender_id} to {message.receiver_id}")
                self.message_queue.task_done()
            except Exception as e:
                self.logger.error(f"消息处理错误: {e}")
    
    def get_collaboration_stats(self, session_id: str) -> Dict[str, Any]:
        """获取协作统计信息"""
        if session_id not in self.collaboration_contexts:
            return {}
        
        context = self.collaboration_contexts[session_id]
        
        return {
            "session_id": session_id,
            "collaboration_mode": context.collaboration_mode,
            "participating_agents": list(context.participating_agents),
            "total_messages": len(context.message_history),
            "active_tasks": len(context.active_tasks),
            "completed_tasks": len(context.completed_tasks),
            "session_duration": (datetime.utcnow() - context.created_at).total_seconds()
        }


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self):
        self.pending_tasks: List[CollaborationTask] = []
        self.running_tasks: Dict[str, CollaborationTask] = {}
        self.logger = logging.getLogger("collaboration.scheduler")
    
    async def schedule_task(self, task: CollaborationTask) -> bool:
        """调度任务"""
        # 检查依赖
        if await self._check_dependencies(task):
            self.running_tasks[task.id] = task
            task.status = "running"
            task.started_at = datetime.utcnow()
            return True
        else:
            self.pending_tasks.append(task)
            return False
    
    async def _check_dependencies(self, task: CollaborationTask) -> bool:
        """检查任务依赖"""
        for dep_id in task.dependencies:
            if dep_id in self.running_tasks:
                return False
        return True


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self.agent_loads: Dict[str, int] = {}
        self.logger = logging.getLogger("collaboration.load_balancer")
    
    def get_least_loaded_agent(self, available_agents: List[str]) -> str:
        """获取负载最轻的智能体"""
        if not available_agents:
            return ""
        
        min_load = float('inf')
        selected_agent = available_agents[0]
        
        for agent_id in available_agents:
            load = self.agent_loads.get(agent_id, 0)
            if load < min_load:
                min_load = load
                selected_agent = agent_id
        
        return selected_agent
    
    def update_agent_load(self, agent_id: str, load_delta: int):
        """更新智能体负载"""
        current_load = self.agent_loads.get(agent_id, 0)
        self.agent_loads[agent_id] = max(0, current_load + load_delta)


# 全局实例
_collaboration_orchestrator: Optional[AgentCollaborationOrchestrator] = None


def get_collaboration_orchestrator() -> AgentCollaborationOrchestrator:
    """获取协作编排器实例"""
    global _collaboration_orchestrator
    if _collaboration_orchestrator is None:
        _collaboration_orchestrator = AgentCollaborationOrchestrator()
    return _collaboration_orchestrator