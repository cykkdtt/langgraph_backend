"""
多智能体LangGraph项目 - 智能体管理器

本模块实现智能体的统一管理功能，包括：
- AgentManager: 智能体生命周期管理
- 智能体状态监控和健康检查
- 智能体性能统计和优化
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .base import BaseAgent, AgentType, AgentStatus, ChatResponse, StreamChunk, ChatRequest as BaseChatRequest
from models.chat_models import ChatRequest
from langchain_core.messages import HumanMessage
from .registry import AgentRegistry, AgentFactory, AgentInstance, get_agent_registry, get_agent_factory
from core.memory import get_memory_manager
from core.tools import get_tool_registry


@dataclass
class AgentPerformanceMetrics:
    """智能体性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    uptime_start: datetime = datetime.utcnow()
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        return 1.0 - self.success_rate
    
    @property
    def uptime_hours(self) -> float:
        """运行时间(小时)"""
        return (datetime.utcnow() - self.uptime_start).total_seconds() / 3600


class AgentManager:
    """智能体管理器
    
    提供智能体的统一管理接口，包括创建、获取、监控和清理。
    """
    
    def __init__(self):
        self.registry = get_agent_registry()
        self.factory = get_agent_factory()
        self.memory_manager = get_memory_manager()
        self.tool_registry = get_tool_registry()
        
        self._performance_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self._logger = logging.getLogger("agent.manager")
        
        # 监控任务
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # 配置
        self.max_idle_hours = 24
        self.monitoring_interval = 300  # 5分钟
        self.cleanup_interval = 3600    # 1小时
    
    async def start(self):
        """启动智能体管理器"""
        self._logger.info("启动智能体管理器")
        
        # 启动监控任务
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """停止智能体管理器"""
        self._logger.info("停止智能体管理器")
        
        # 停止监控任务
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 清理所有智能体实例
        await self._cleanup_all_instances()
    
    async def create_agent(
        self,
        agent_type: AgentType,
        user_id: str,
        session_id: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建智能体实例
        
        Args:
            agent_type: 智能体类型
            user_id: 用户ID
            session_id: 会话ID
            custom_config: 自定义配置
            
        Returns:
            实例ID
        """
        try:
            instance_id = await self.factory.create_agent(
                agent_type=agent_type,
                user_id=user_id,
                session_id=session_id,
                custom_config=custom_config
            )
            
            # 初始化性能指标
            self._performance_metrics[instance_id] = AgentPerformanceMetrics()
            
            self._logger.info(f"创建智能体实例: {agent_type.value} (ID: {instance_id})")
            
            return instance_id
            
        except Exception as e:
            self._logger.error(f"创建智能体实例失败: {e}")
            raise
    
    async def get_agent(self, instance_id: str) -> Optional[BaseAgent]:
        """获取智能体实例
        
        Args:
            instance_id: 实例ID
            
        Returns:
            智能体实例或None
        """
        return await self.factory.get_agent(instance_id)
    
    async def get_instance_info(self, instance_id: str) -> Optional[AgentInstance]:
        """获取智能体实例信息
        
        Args:
            instance_id: 实例ID
            
        Returns:
            实例信息或None
        """
        return await self.factory.get_instance_info(instance_id)
    
    async def list_instances(
        self,
        user_id: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
        status: Optional[AgentStatus] = None
    ) -> List[AgentInstance]:
        """列出智能体实例
        
        Args:
            user_id: 用户ID过滤
            agent_type: 智能体类型过滤
            status: 状态过滤
            
        Returns:
            实例列表
        """
        return await self.factory.list_instances(user_id, agent_type, status)
    
    async def process_message(
        self,
        instance_id: str,
        request  # 支持不同格式的ChatRequest
    ) -> ChatResponse:
        """处理消息
        
        Args:
            instance_id: 实例ID
            request: 聊天请求（支持models.chat_models.ChatRequest或core.agents.base.ChatRequest格式）
            
        Returns:
            聊天响应
        """
        start_time = datetime.utcnow()
        
        try:
            # 获取智能体实例
            agent = await self.get_agent(instance_id)
            if not agent:
                raise ValueError(f"智能体实例不存在: {instance_id}")
            
            # 转换请求格式为core.agents.base.ChatRequest
            if hasattr(request, 'message'):
                # models.chat_models.ChatRequest格式 -> core.agents.base.ChatRequest格式
                base_request = BaseChatRequest(
                    messages=[HumanMessage(content=request.message)],
                    user_id=request.user_id,
                    session_id=request.session_id,
                    stream=getattr(request, 'stream', False),
                    metadata=getattr(request, 'metadata', {})
                )
            elif hasattr(request, 'messages'):
                # 已经是core.agents.base.ChatRequest格式
                base_request = request
            else:
                raise ValueError("不支持的ChatRequest格式")
            
            # 处理消息
            response = await agent.chat(base_request)
            
            # 更新性能指标
            await self._update_performance_metrics(
                instance_id, 
                start_time, 
                success=True
            )
            
            return response
            
        except Exception as e:
            # 更新性能指标
            await self._update_performance_metrics(
                instance_id, 
                start_time, 
                success=False
            )
            
            self._logger.error(f"处理消息失败: {e}")
            raise
    
    async def stream_message(
        self,
        instance_id: str,
        request  # 支持不同格式的ChatRequest
    ):
        """流式处理消息
        
        Args:
            instance_id: 实例ID
            request: 聊天请求（支持models.chat_models.ChatRequest或core.agents.base.ChatRequest格式）
            
        Yields:
            流式响应块
        """
        start_time = datetime.utcnow()
        
        try:
            # 获取智能体实例
            agent = await self.get_agent(instance_id)
            if not agent:
                raise ValueError(f"智能体实例不存在: {instance_id}")
            
            # 转换请求格式为core.agents.base.ChatRequest
            if hasattr(request, 'message'):
                # models.chat_models.ChatRequest格式 -> core.agents.base.ChatRequest格式
                base_request = BaseChatRequest(
                    messages=[HumanMessage(content=request.message)],
                    user_id=request.user_id,
                    session_id=request.session_id,
                    stream=True,
                    metadata=getattr(request, 'metadata', {})
                )
            elif hasattr(request, 'messages'):
                # 已经是core.agents.base.ChatRequest格式
                base_request = request
            else:
                raise ValueError("不支持的ChatRequest格式")
            
            # 流式处理消息
            async for chunk in agent.astream(base_request):
                yield chunk
            
            # 更新性能指标
            await self._update_performance_metrics(
                instance_id, 
                start_time, 
                success=True
            )
            
        except Exception as e:
            # 更新性能指标
            await self._update_performance_metrics(
                instance_id, 
                start_time, 
                success=False
            )
            
            self._logger.error(f"流式处理消息失败: {e}")
            raise
    
    async def get_performance_metrics(self, instance_id: str) -> Optional[AgentPerformanceMetrics]:
        """获取智能体性能指标
        
        Args:
            instance_id: 实例ID
            
        Returns:
            性能指标或None
        """
        return self._performance_metrics.get(instance_id)
    
    async def get_all_performance_metrics(self) -> Dict[str, AgentPerformanceMetrics]:
        """获取所有智能体性能指标
        
        Returns:
            性能指标字典
        """
        return self._performance_metrics.copy()
    
    async def cleanup_agent(self, instance_id: str):
        """清理智能体实例
        
        Args:
            instance_id: 实例ID
        """
        try:
            await self.factory.cleanup_agent(instance_id)
            
            # 清理性能指标
            if instance_id in self._performance_metrics:
                del self._performance_metrics[instance_id]
            
            self._logger.info(f"清理智能体实例: {instance_id}")
            
        except Exception as e:
            self._logger.error(f"清理智能体实例失败: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            健康状态信息
        """
        try:
            # 获取工厂健康状态
            factory_health = await self.factory.health_check()
            
            # 计算总体性能指标
            total_requests = sum(m.total_requests for m in self._performance_metrics.values())
            total_successful = sum(m.successful_requests for m in self._performance_metrics.values())
            avg_success_rate = total_successful / total_requests if total_requests > 0 else 0.0
            
            # 获取实例状态分布
            instances = await self.list_instances()
            status_distribution = {}
            for status in AgentStatus:
                count = len([inst for inst in instances if inst.status == status])
                status_distribution[status.value] = count
            
            return {
                "manager_status": "healthy",
                "factory_health": factory_health,
                "total_requests": total_requests,
                "success_rate": avg_success_rate,
                "instance_status_distribution": status_distribution,
                "performance_metrics_count": len(self._performance_metrics),
                "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
                "cleanup_active": self._cleanup_task is not None and not self._cleanup_task.done()
            }
            
        except Exception as e:
            self._logger.error(f"健康检查失败: {e}")
            return {
                "manager_status": "unhealthy",
                "error": str(e)
            }
    
    async def _update_performance_metrics(
        self,
        instance_id: str,
        start_time: datetime,
        success: bool
    ):
        """更新性能指标
        
        Args:
            instance_id: 实例ID
            start_time: 开始时间
            success: 是否成功
        """
        if instance_id not in self._performance_metrics:
            self._performance_metrics[instance_id] = AgentPerformanceMetrics()
        
        metrics = self._performance_metrics[instance_id]
        
        # 计算响应时间
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 更新指标
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # 更新响应时间统计
        metrics.min_response_time = min(metrics.min_response_time, response_time)
        metrics.max_response_time = max(metrics.max_response_time, response_time)
        
        # 更新平均响应时间
        if metrics.total_requests == 1:
            metrics.avg_response_time = response_time
        else:
            metrics.avg_response_time = (
                (metrics.avg_response_time * (metrics.total_requests - 1) + response_time) 
                / metrics.total_requests
            )
        
        metrics.last_request_time = datetime.utcnow()
    
    async def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                await self._perform_health_checks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"监控循环错误: {e}")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.factory.cleanup_expired_instances(self.max_idle_hours)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"清理循环错误: {e}")
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        try:
            instances = await self.list_instances()
            
            for instance in instances:
                agent = await self.get_agent(instance.instance_id)
                if agent:
                    # 检查智能体健康状态
                    try:
                        health = await agent.health_check()
                        # 检查智能体是否已初始化和状态是否正常
                        is_healthy = (
                            health.get("is_initialized", False) and 
                            health.get("status") in ["ready", "idle"]
                        )
                        if not is_healthy:
                            self._logger.warning(f"智能体实例不健康: {instance.instance_id}, 状态: {health.get('status')}, 已初始化: {health.get('is_initialized')}")
                    except Exception as e:
                        self._logger.error(f"智能体健康检查失败: {instance.instance_id}, {e}")
            
        except Exception as e:
            self._logger.error(f"执行健康检查失败: {e}")
    
    async def _cleanup_all_instances(self):
        """清理所有智能体实例"""
        try:
            instances = await self.list_instances()
            
            for instance in instances:
                await self.cleanup_agent(instance.instance_id)
            
            self._logger.info("清理所有智能体实例完成")
            
        except Exception as e:
            self._logger.error(f"清理所有智能体实例失败: {e}")


# 全局实例
_agent_manager = None


def get_agent_manager() -> AgentManager:
    """获取全局智能体管理器"""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager()
    return _agent_manager


async def initialize_agent_manager():
    """初始化智能体管理器"""
    manager = get_agent_manager()
    await manager.start()
    
    logging.getLogger("agent.manager").info("智能体管理器初始化完成")
    
    return manager