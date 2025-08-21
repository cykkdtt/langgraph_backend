"""
多智能体LangGraph项目 - 智能体注册表和工厂

本模块实现智能体的注册、管理和创建功能，包括：
- AgentRegistry: 智能体类型注册表
- AgentFactory: 智能体实例工厂
- AgentManager: 智能体生命周期管理
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Type, Union
from datetime import datetime
from abc import ABC, abstractmethod
import uuid
import os

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi

from .base import BaseAgent, AgentType, AgentStatus, AgentMetadata
from .background_memory_manager import BackgroundMemoryManager
from config.settings import get_settings


class AgentConfig(BaseModel):
    """智能体配置模型"""
    agent_type: AgentType = Field(description="智能体类型")
    agent_id: Optional[str] = Field(default=None, description="智能体ID")
    name: str = Field(description="智能体名称")
    description: str = Field(description="智能体描述")
    version: str = Field(default="1.0.0", description="版本号")
    
    # 模型配置
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="LLM配置")
    llm: Optional[BaseLanguageModel] = Field(default=None, description="LLM实例")
    
    # 工具配置
    tools: List[Any] = Field(default_factory=list, description="工具列表")
    tool_permissions: Dict[str, List[str]] = Field(default_factory=dict, description="工具权限")
    
    # 能力配置
    capabilities: List[str] = Field(default_factory=list, description="能力列表")
    
    # 记忆配置
    memory_enabled: bool = Field(default=True, description="是否启用记忆")
    memory_namespace: Optional[str] = Field(default=None, description="记忆命名空间")
    
    # 协作配置
    collaboration_enabled: bool = Field(default=False, description="是否支持协作")
    handoff_targets: List[str] = Field(default_factory=list, description="可移交的目标智能体")
    
    # 性能配置
    max_iterations: int = Field(default=10, description="最大迭代次数")
    timeout: int = Field(default=300, description="超时时间(秒)")
    
    # 检查点配置
    checkpointer: Optional[Any] = Field(default=None, description="检查点管理器")
    
    # 自定义配置
    custom_config: Dict[str, Any] = Field(default_factory=dict, description="自定义配置")
    
    class Config:
        arbitrary_types_allowed = True  # 允许任意类型，用于 LLM 实例


class AgentInstance(BaseModel):
    """智能体实例模型"""
    instance_id: str = Field(description="实例ID")
    agent_type: AgentType = Field(description="智能体类型")
    user_id: str = Field(description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    
    # 实例状态
    status: AgentStatus = Field(default=AgentStatus.INITIALIZING, description="实例状态")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    last_used: datetime = Field(default_factory=datetime.utcnow, description="最后使用时间")
    
    # 配置和元数据
    config: AgentConfig = Field(description="智能体配置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="实例元数据")
    
    # 性能统计
    total_requests: int = Field(default=0, description="总请求数")
    total_errors: int = Field(default=0, description="总错误数")
    avg_response_time: float = Field(default=0.0, description="平均响应时间")


class AgentRegistry:
    """智能体注册表
    
    管理所有可用的智能体类型和配置。
    """
    
    def __init__(self):
        self._agent_classes: Dict[AgentType, Type[BaseAgent]] = {}
        self._agent_configs: Dict[AgentType, AgentConfig] = {}
        self._logger = logging.getLogger("agent.registry")
        
        # 初始化后台记忆管理器
        self._memory_manager = BackgroundMemoryManager()
        
        # 初始化默认配置
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """初始化默认智能体配置"""
        settings = get_settings()
        
        # 注册智能体类
        self._register_agent_classes()
        
        # Supervisor智能体配置 - 启用记忆功能
        self._agent_configs[AgentType.SUPERVISOR] = AgentConfig(
            agent_type=AgentType.SUPERVISOR,
            name="Supervisor Agent",
            description="主管智能体，负责任务协调和智能体间的协作",
            llm_config={
                "model_type": settings.llm.default_supervisor_model,
                "temperature": 0.7,
                "max_tokens": 2000
            },
            tools=[],  # 工具将在智能体初始化时动态添加
            capabilities=["task_coordination", "result_integration", "decision_making"],
            collaboration_enabled=True,
            handoff_targets=["research", "chart"],
            memory_enabled=True,  # Supervisor 需要记忆功能来协调工作流程
            memory_namespace="supervisor_agent"  # 独立的命名空间用于Supervisor智能体记忆
        )
        
        # Research智能体配置 - 不启用记忆功能
        self._agent_configs[AgentType.RESEARCH] = AgentConfig(
            agent_type=AgentType.RESEARCH,
            name="Research Agent",
            description="研究智能体，专门负责信息搜索和分析",
            llm_config={
                "model_type": settings.llm.default_research_model,
                "temperature": 0.3,
                "max_tokens": 3000
            },
            tools=["google_search", "web_scraper", "document_analyzer"],
            capabilities=["information_search", "data_analysis", "report_generation"],
            collaboration_enabled=True,
            handoff_targets=["supervisor", "chart"],
            memory_enabled=False  # Research智能体不需要记忆功能
        )
        
        # Chart智能体配置 - 不启用记忆功能
        self._agent_configs[AgentType.CHART] = AgentConfig(
            agent_type=AgentType.CHART,
            name="Chart Agent", 
            description="图表智能体，专门负责数据可视化和图表生成",
            llm_config={
                "model_type": settings.llm.default_chart_model,
                "temperature": 0.5,
                "max_tokens": 2000
            },
            tools=["mcp_chart_tools", "data_processor", "visualization_engine"],
            capabilities=["data_visualization", "chart_generation", "statistical_analysis"],
            collaboration_enabled=True,
            handoff_targets=["supervisor", "research"],
            memory_enabled=False  # Chart智能体不需要记忆功能
        )
        
        # RAG智能体配置 - 启用记忆功能
        self._agent_configs[AgentType.RAG] = AgentConfig(
            agent_type=AgentType.RAG,
            name="RAG Agent",
            description="检索增强生成智能体，结合知识库进行智能问答",
            llm_config={
                "model_type": settings.llm.default_chat_model,
                "temperature": 0.6,
                "max_tokens": 2500
            },
            tools=["vector_search", "document_retrieval", "knowledge_base"],
            capabilities=["knowledge_retrieval", "context_generation", "qa_generation"],
            memory_enabled=True,
            memory_namespace="rag_agent"  # 独立的命名空间用于RAG智能体记忆
        )
        
        # Code智能体配置 - 不启用记忆功能
        self._agent_configs[AgentType.CODE] = AgentConfig(
            agent_type=AgentType.CODE,
            name="Code Agent",
            description="代码智能体，专门负责代码生成、分析和优化",
            llm_config={
                "model_type": settings.llm.default_chat_model,
                "temperature": 0.2,
                "max_tokens": 4000
            },
            tools=["code_executor", "syntax_analyzer", "code_formatter"],
            capabilities=["code_generation", "code_analysis", "code_optimization"],
            memory_enabled=False  # 代码智能体不需要记忆功能
        )
        
        # 注册智能体类
        self._register_agent_classes()
    
    def _register_agent_classes(self):
        """注册智能体类
        
        根据 LangMem 最佳实践，只有需要记忆功能的智能体才使用 MemoryEnhancedAgent。
        其他智能体使用 BaseAgent 以避免不必要的性能开销。
        """
        try:
            # 导入智能体类
            from .base import BaseAgent
            from .memory_enhanced import MemoryEnhancedAgent
            
            # 定义需要记忆功能的智能体类型
            memory_enabled_agents = {AgentType.SUPERVISOR, AgentType.RAG}
            
            # 注册智能体类
            for agent_type in [AgentType.SUPERVISOR, AgentType.RESEARCH, AgentType.CHART, AgentType.RAG, AgentType.CODE]:
                if agent_type in memory_enabled_agents:
                    # 需要记忆功能的智能体使用 MemoryEnhancedAgent
                    self._agent_classes[agent_type] = MemoryEnhancedAgent
                    self._logger.info(f"注册记忆增强智能体: {agent_type.value} -> MemoryEnhancedAgent")
                else:
                    # 其他智能体使用 BaseAgent
                    self._agent_classes[agent_type] = BaseAgent
                    self._logger.info(f"注册基础智能体: {agent_type.value} -> BaseAgent")
            
        except ImportError as e:
            self._logger.warning(f"无法导入智能体类: {e}")
            # 回退到基础智能体
            from .base import BaseAgent
            for agent_type in [AgentType.SUPERVISOR, AgentType.RESEARCH, AgentType.CHART, AgentType.RAG, AgentType.CODE]:
                self._agent_classes[agent_type] = BaseAgent
                self._logger.info(f"回退注册智能体类: {agent_type.value} -> BaseAgent")
        except Exception as e:
            self._logger.error(f"注册智能体类失败: {e}")
            raise
    
    def register_agent_type(
        self, 
        agent_type: AgentType, 
        agent_class: Type[BaseAgent], 
        config: Optional[AgentConfig] = None
    ):
        """注册智能体类型
        
        Args:
            agent_type: 智能体类型
            agent_class: 智能体类
            config: 智能体配置
        """
        self._agent_classes[agent_type] = agent_class
        
        if config:
            self._agent_configs[agent_type] = config
        
        self._logger.info(f"注册智能体类型: {agent_type.value}")
    
    def get_agent_class(self, agent_type: AgentType) -> Optional[Type[BaseAgent]]:
        """获取智能体类
        
        Args:
            agent_type: 智能体类型
            
        Returns:
            智能体类或None
        """
        return self._agent_classes.get(agent_type)
    
    def get_agent_config(self, agent_type: AgentType) -> Optional[AgentConfig]:
        """获取智能体配置
        
        Args:
            agent_type: 智能体类型
            
        Returns:
            智能体配置或None
        """
        return self._agent_configs.get(agent_type)
    
    def get_memory_manager(self) -> BackgroundMemoryManager:
        """获取后台记忆管理器"""
        return self._memory_manager
    
    def list_agent_types(self) -> List[AgentType]:
        """列出所有注册的智能体类型
        
        Returns:
            智能体类型列表
        """
        return list(self._agent_classes.keys())
    
    def update_agent_config(self, agent_type: AgentType, config: AgentConfig):
        """更新智能体配置
        
        Args:
            agent_type: 智能体类型
            config: 新的配置
        """
        self._agent_configs[agent_type] = config
        self._logger.info(f"更新智能体配置: {agent_type.value}")
    
    def is_registered(self, agent_type: AgentType) -> bool:
        """检查智能体类型是否已注册
        
        Args:
            agent_type: 智能体类型
            
        Returns:
            是否已注册
        """
        return agent_type in self._agent_classes


class AgentFactory:
    """智能体工厂
    
    负责创建和管理智能体实例。
    """
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._instances: Dict[str, AgentInstance] = {}
        self._agents: Dict[str, BaseAgent] = {}
        self._logger = logging.getLogger("agent.factory")
    

    
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
            
        Raises:
            ValueError: 未知的智能体类型
            RuntimeError: 创建失败
        """
        try:
            # 检查智能体类型是否已注册
            agent_class = self.registry.get_agent_class(agent_type)
            if not agent_class:
                raise ValueError(f"未知的智能体类型: {agent_type.value}")
            
            # 获取基础配置
            base_config = self.registry.get_agent_config(agent_type)
            if not base_config:
                raise ValueError(f"未找到智能体配置: {agent_type.value}")
            
            # 合并自定义配置
            if custom_config:
                config_dict = base_config.dict()
                config_dict.update(custom_config)
                config = AgentConfig(**config_dict)
            else:
                config = base_config
            
            # 生成实例ID
            instance_id = str(uuid.uuid4())
            
            # 创建智能体实例记录
            instance = AgentInstance(
                instance_id=instance_id,
                agent_type=agent_type,
                user_id=user_id,
                session_id=session_id,
                config=config,
                metadata={
                    "created_by": "factory",
                    "factory_version": "1.0.0"
                }
            )
            
            # 创建实际的智能体对象
            agent = await self._instantiate_agent(agent_class, config)
            
            # 存储实例
            self._instances[instance_id] = instance
            self._agents[instance_id] = agent
            
            # 更新实例状态
            instance.status = AgentStatus.READY
            
            self._logger.info(f"创建智能体实例: {agent_type} (ID: {instance_id})")
            
            return instance_id
            
        except Exception as e:
            self._logger.error(f"创建智能体实例失败: {e}")
            raise RuntimeError(f"创建智能体实例失败: {str(e)}")
    
    async def _instantiate_agent(
        self, 
        agent_class: Type[BaseAgent], 
        config: AgentConfig
    ) -> BaseAgent:
        """实例化智能体对象
        
        Args:
            agent_class: 智能体类
            config: 智能体配置
            
        Returns:
            智能体实例
        """
        try:
            # 生成唯一的智能体ID
            agent_id = f"{config.agent_type.value}_{uuid.uuid4().hex[:8]}"
            
            # 根据llm_config创建LLM实例
            llm = self._create_llm_instance(config.llm_config)
            
            # 更新配置对象中的动态生成字段
            config.agent_id = agent_id
            config.llm = llm
            config.checkpointer = None  # 可以根据需要配置
            
            # 为启用记忆功能的智能体传递记忆管理器
            if config.memory_enabled:
                # 创建智能体实例，传递记忆管理器
                agent = agent_class(config=config, memory_manager=self.registry.get_memory_manager())
            else:
                # 创建智能体实例，不传递记忆管理器
                agent = agent_class(config=config)
            
            return agent
            
        except Exception as e:
            self._logger.error(f"实例化智能体失败: {e}")
            raise
    
    def _create_llm_instance(self, llm_config: Dict[str, Any]):
        """根据配置创建LLM实例
        
        Args:
            llm_config: LLM配置字典
            
        Returns:
            LLM实例
        """
        try:
            # 获取模型类型，默认使用deepseek
            model_type = llm_config.get("model_type", "deepseek-chat")
            temperature = llm_config.get("temperature", 0.7)
            max_tokens = llm_config.get("max_tokens", 2000)
            
            # 根据模型类型创建相应的LLM实例
            if "deepseek" in model_type.lower():
                # 使用DeepSeek模型
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    self._logger.warning("DEEPSEEK_API_KEY未设置，LLM将无法正常工作")
                    return None
                
                return ChatDeepSeek(
                    model="deepseek-chat",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key
                )
            
            elif "gpt" in model_type.lower() or "openai" in model_type.lower():
                # 使用OpenAI模型
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    self._logger.warning("OPENAI_API_KEY未设置，LLM将无法正常工作")
                    return None
                
                return ChatOpenAI(
                    model=model_type if "gpt" in model_type else "gpt-3.5-turbo",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key
                )
            
            elif "tongyi" in model_type.lower():
                # 使用通义千问模型
                api_key = os.getenv("TONGYI_API_KEY")
                if not api_key:
                    self._logger.warning("TONGYI_API_KEY未设置，LLM将无法正常工作")
                    return None
                
                return ChatTongyi(
                    model="qwen-turbo",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    dashscope_api_key=api_key
                )
            
            else:
                # 默认使用DeepSeek
                self._logger.info(f"未知模型类型 {model_type}，使用默认的DeepSeek模型")
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    self._logger.warning("DEEPSEEK_API_KEY未设置，LLM将无法正常工作")
                    return None
                
                return ChatDeepSeek(
                    model="deepseek-chat",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key
                )
                
        except Exception as e:
            self._logger.error(f"创建LLM实例失败: {e}")
            return None
    
    async def get_agent(self, instance_id: str) -> Optional[BaseAgent]:
        """获取智能体实例
        
        Args:
            instance_id: 实例ID
            
        Returns:
            智能体实例或None
        """
        agent = self._agents.get(instance_id)
        if agent:
            # 更新最后使用时间
            if instance_id in self._instances:
                self._instances[instance_id].last_used = datetime.utcnow()
        
        return agent
    
    async def get_instance_info(self, instance_id: str) -> Optional[AgentInstance]:
        """获取智能体实例信息
        
        Args:
            instance_id: 实例ID
            
        Returns:
            实例信息或None
        """
        return self._instances.get(instance_id)
    
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
        instances = list(self._instances.values())
        
        # 应用过滤条件
        if user_id:
            instances = [inst for inst in instances if inst.user_id == user_id]
        
        if agent_type:
            instances = [inst for inst in instances if inst.agent_type == agent_type]
        
        if status:
            instances = [inst for inst in instances if inst.status == status]
        
        return instances
    
    async def cleanup_agent(self, instance_id: str):
        """清理智能体实例
        
        Args:
            instance_id: 实例ID
        """
        try:
            # 清理智能体资源
            if instance_id in self._agents:
                agent = self._agents[instance_id]
                if hasattr(agent, 'cleanup'):
                    await agent.cleanup()
                del self._agents[instance_id]
            
            # 删除实例记录
            if instance_id in self._instances:
                del self._instances[instance_id]
            
            self._logger.info(f"清理智能体实例: {instance_id}")
            
        except Exception as e:
            self._logger.error(f"清理智能体实例失败: {e}")
    
    async def cleanup_expired_instances(self, max_idle_hours: int = 24):
        """清理过期的智能体实例
        
        Args:
            max_idle_hours: 最大空闲时间(小时)
        """
        current_time = datetime.utcnow()
        expired_instances = []
        
        for instance_id, instance in self._instances.items():
            idle_time = (current_time - instance.last_used).total_seconds() / 3600
            if idle_time > max_idle_hours:
                expired_instances.append(instance_id)
        
        for instance_id in expired_instances:
            await self.cleanup_agent(instance_id)
        
        if expired_instances:
            self._logger.info(f"清理了 {len(expired_instances)} 个过期实例")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            健康状态信息
        """
        total_instances = len(self._instances)
        active_instances = len([
            inst for inst in self._instances.values() 
            if inst.status == AgentStatus.READY
        ])
        
        return {
            "total_instances": total_instances,
            "active_instances": active_instances,
            "registered_types": len(self.registry.list_agent_types()),
            "factory_status": "healthy"
        }


# 全局实例
_agent_registry = None
_agent_factory = None


def get_agent_registry() -> AgentRegistry:
    """获取全局智能体注册表"""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry


def get_agent_factory() -> AgentFactory:
    """获取全局智能体工厂"""
    global _agent_factory
    if _agent_factory is None:
        registry = get_agent_registry()
        _agent_factory = AgentFactory(registry)
    return _agent_factory


async def initialize_agent_system():
    """初始化智能体系统"""
    registry = get_agent_registry()
    factory = get_agent_factory()
    
    # 这里可以注册具体的智能体类
    # registry.register_agent_type(AgentType.SUPERVISOR, SupervisorAgent)
    # registry.register_agent_type(AgentType.RESEARCH, ResearchAgent)
    # registry.register_agent_type(AgentType.CHART, ChartAgent)
    
    logging.getLogger("agent.system").info("智能体系统初始化完成")
    
    return registry, factory