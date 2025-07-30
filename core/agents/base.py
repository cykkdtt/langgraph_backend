"""
多智能体LangGraph项目 - 智能体基类

本模块定义了智能体的基类和相关数据模型，提供统一的接口和行为规范。
基于LangGraph 0.6.1的最新API设计。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator, List, Union, Annotated
from datetime import datetime
import uuid
import logging
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class AgentType(str, Enum):
    """智能体类型枚举"""
    SUPERVISOR = "supervisor"
    RESEARCH = "research"
    CHART = "chart"
    RAG = "rag"
    CODE = "code"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_CREATION = "content_creation"
    WORKFLOW = "workflow"


class AgentStatus(str, Enum):
    """智能体状态枚举"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


class AgentCapability(BaseModel):
    """智能体能力描述"""
    name: str = Field(description="能力名称")
    description: str = Field(description="能力描述")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="能力参数")


class AgentMetadata(BaseModel):
    """智能体元数据"""
    agent_id: str = Field(description="智能体唯一标识")
    agent_type: AgentType = Field(description="智能体类型")
    name: str = Field(description="智能体名称")
    description: str = Field(description="智能体描述")
    version: str = Field(default="1.0.0", description="智能体版本")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="智能体能力列表")
    tools: List[str] = Field(default_factory=list, description="可用工具列表")
    status: AgentStatus = Field(default=AgentStatus.INITIALIZING, description="智能体状态")


class ChatRequest(BaseModel):
    """对话请求模型"""
    messages: List[BaseMessage] = Field(description="消息列表")
    user_id: str = Field(description="用户ID")
    session_id: str = Field(description="会话ID")
    stream: bool = Field(default=False, description="是否流式响应")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class ChatResponse(BaseModel):
    """对话响应模型"""
    message: BaseMessage = Field(description="智能体回复消息")
    session_id: str = Field(description="会话ID")
    agent_id: str = Field(description="智能体ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="响应元数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间")


class StreamChunk(BaseModel):
    """流式响应块"""
    chunk_type: str = Field(description="块类型: message, tool_call, error, done")
    content: str = Field(description="块内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="块元数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")


class AgentState(TypedDict):
    """智能体状态模型 - 基于LangGraph的状态管理"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    session_id: str
    metadata: Dict[str, Any]


class BaseAgent:
    """智能体基类
    
    基于LangGraph 0.6.1的最新API设计，提供统一的智能体接口。
    不再使用抽象基类，而是通过组合StateGraph来实现智能体功能。
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm,
        tools: Optional[List] = None,
        checkpointer=None,
        **kwargs
    ):
        """初始化智能体
        
        Args:
            agent_id: 智能体ID
            name: 智能体名称
            description: 智能体描述
            llm: 语言模型
            tools: 工具列表
            checkpointer: 检查点保存器
            **kwargs: 其他参数
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = tools or []
        self.checkpointer = checkpointer
        self.config = kwargs
        
        # 状态和图
        self.graph: Optional[StateGraph] = None
        self.compiled_graph = None
        self.initialized = False
        
        # 元数据
        self.metadata = AgentMetadata(
            agent_id=agent_id,
            agent_type=kwargs.get("agent_type", AgentType.CODE),
            name=name,
            description=description,
            tools=[tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in self.tools]
        )
        
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        
    async def initialize(self) -> None:
        """初始化智能体"""
        if self.initialized:
            return
            
        try:
            self.metadata.status = AgentStatus.INITIALIZING
            self.logger.info(f"初始化智能体: {self.name}")
            
            # 构建图
            await self._build_graph()
            
            # 编译图
            self.compiled_graph = self.graph.compile(
                checkpointer=self.checkpointer,
                debug=self.config.get("debug", False)
            )
            
            self.metadata.status = AgentStatus.READY
            self.initialized = True
            self.logger.info(f"智能体初始化完成: {self.name}")
            
        except Exception as e:
            self.metadata.status = AgentStatus.ERROR
            self.logger.error(f"智能体初始化失败: {e}")
            raise
    
    async def _build_graph(self) -> None:
        """构建智能体的LangGraph图 - 子类应重写此方法"""
        # 创建基础图
        self.graph = StateGraph(AgentState)
        
        # 添加默认节点
        self.graph.add_node("chat", self._chat_node)
        
        # 设置入口和出口
        self.graph.add_edge(START, "chat")
        self.graph.add_edge("chat", END)
        
    async def _chat_node(self, state: AgentState) -> Dict[str, Any]:
        """默认对话节点"""
        messages = state["messages"]
        
        # 调用LLM
        if self.llm:
            response = await self.llm.ainvoke(messages)
            return {"messages": [response]}
        else:
            # 默认响应
            return {"messages": [AIMessage(content="我是一个基础智能体，请配置LLM以获得更好的响应。")]}
    
    async def chat(
        self,
        request: ChatRequest,
        config: Optional[RunnableConfig] = None
    ) -> ChatResponse:
        """处理对话请求"""
        if not self.initialized:
            await self.initialize()
        
        # 准备状态
        state = AgentState(
            messages=request.messages,
            user_id=request.user_id,
            session_id=request.session_id,
            metadata=request.metadata
        )
        
        # 执行图
        result = await self.compiled_graph.ainvoke(state, config=config)
        
        # 获取最后一条AI消息
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        last_message = ai_messages[-1] if ai_messages else AIMessage(content="无响应")
        
        return ChatResponse(
            message=last_message,
            session_id=request.session_id,
            agent_id=self.agent_id,
            metadata=result.get("metadata", {}),
        )
    
    async def stream_chat(
        self,
        request: ChatRequest,
        config: Optional[RunnableConfig] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """流式处理对话请求"""
        if not self.initialized:
            await self.initialize()
        
        # 准备状态
        state = AgentState(
            messages=request.messages,
            user_id=request.user_id,
            session_id=request.session_id,
            metadata=request.metadata
        )
        
        # 流式执行图
        async for chunk in self.compiled_graph.astream(state, config=config):
            # 处理不同类型的流式输出
            for node_name, node_output in chunk.items():
                if "messages" in node_output:
                    for message in node_output["messages"]:
                        if isinstance(message, AIMessage):
                            yield StreamChunk(
                                chunk_type="message",
                                content=message.content,
                                metadata={"node": node_name}
                            )
        
        # 发送完成信号
        yield StreamChunk(
            chunk_type="done",
            content="",
            metadata={"status": "completed"}
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            Dict: 健康状态信息
        """
        return {
            "agent_id": self.metadata.agent_id,
            "agent_type": self.metadata.agent_type.value,
            "status": self.metadata.status.value,
            "is_initialized": self._is_initialized,
            "uptime": (datetime.utcnow() - self.metadata.created_at).total_seconds(),
            "capabilities_count": len(self.metadata.capabilities),
            "tools_count": len(self.metadata.tools)
        }
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """获取智能体能力列表
        
        Returns:
            List[AgentCapability]: 能力列表
        """
        return self.metadata.capabilities
    
    async def get_tools(self) -> List[str]:
        """获取智能体工具列表
        
        Returns:
            List[str]: 工具列表
        """
        return self.metadata.tools
    
    async def update_status(self, status: AgentStatus):
        """更新智能体状态
        
        Args:
            status: 新状态
        """
        old_status = self.metadata.status
        self.metadata.status = status
        self.metadata.updated_at = datetime.utcnow()
        self.logger.info(f"智能体状态更新: {old_status.value} -> {status.value}")
    
    async def cleanup(self):
        """清理资源"""
        try:
            await self.update_status(AgentStatus.STOPPED)
            self.logger.info(f"智能体资源清理完成: {self.metadata.name}")
        except Exception as e:
            self.logger.error(f"智能体资源清理失败: {e}")
    
    def get_metadata(self) -> AgentMetadata:
        """获取智能体元数据
        
        Returns:
            AgentMetadata: 元数据
        """
        return self.metadata
    
    def __str__(self) -> str:
        return f"Agent({self.metadata.agent_type.value}, {self.metadata.agent_id})"
    
    def __repr__(self) -> str:
        return (
            f"BaseAgent("
            f"agent_type={self.metadata.agent_type.value}, "
            f"agent_id={self.metadata.agent_id}, "
            f"status={self.metadata.status.value}"
            f")"
        )


class AgentRegistry:
    """智能体注册表
    
    管理所有智能体类型的注册和实例化。
    """
    
    def __init__(self):
        self._agent_classes: Dict[AgentType, type] = {}
        self._agent_configs: Dict[AgentType, Dict[str, Any]] = {}
        self._instances: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("agent.registry")
    
    def register_agent_type(
        self, 
        agent_type: AgentType, 
        agent_class: type, 
        config: Dict[str, Any]
    ):
        """注册智能体类型
        
        Args:
            agent_type: 智能体类型
            agent_class: 智能体类
            config: 默认配置
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"智能体类必须继承BaseAgent: {agent_class}")
        
        self._agent_classes[agent_type] = agent_class
        self._agent_configs[agent_type] = config
        self.logger.info(f"注册智能体类型: {agent_type.value}")
    
    def get_agent_class(self, agent_type: AgentType) -> Optional[type]:
        """获取智能体类
        
        Args:
            agent_type: 智能体类型
            
        Returns:
            type: 智能体类
        """
        return self._agent_classes.get(agent_type)
    
    def get_agent_config(self, agent_type: AgentType) -> Optional[Dict[str, Any]]:
        """获取智能体配置
        
        Args:
            agent_type: 智能体类型
            
        Returns:
            Dict: 智能体配置
        """
        return self._agent_configs.get(agent_type)
    
    def list_agent_types(self) -> List[AgentType]:
        """列出所有注册的智能体类型
        
        Returns:
            List[AgentType]: 智能体类型列表
        """
        return list(self._agent_classes.keys())
    
    async def create_agent(
        self, 
        agent_type: AgentType, 
        config: Optional[Dict[str, Any]] = None
    ) -> BaseAgent:
        """创建智能体实例
        
        Args:
            agent_type: 智能体类型
            config: 智能体配置（可选，会与默认配置合并）
            
        Returns:
            BaseAgent: 智能体实例
        """
        agent_class = self.get_agent_class(agent_type)
        if not agent_class:
            raise ValueError(f"未注册的智能体类型: {agent_type}")
        
        # 合并配置
        default_config = self.get_agent_config(agent_type) or {}
        final_config = {**default_config, **(config or {})}
        
        # 创建实例
        agent = agent_class(agent_type, final_config)
        self._instances[agent.metadata.agent_id] = agent
        
        self.logger.info(f"创建智能体实例: {agent_type.value} ({agent.metadata.agent_id})")
        return agent
    
    def get_agent_instance(self, agent_id: str) -> Optional[BaseAgent]:
        """获取智能体实例
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            BaseAgent: 智能体实例
        """
        return self._instances.get(agent_id)
    
    def list_agent_instances(self) -> List[BaseAgent]:
        """列出所有智能体实例
        
        Returns:
            List[BaseAgent]: 智能体实例列表
        """
        return list(self._instances.values())
    
    async def remove_agent_instance(self, agent_id: str) -> bool:
        """移除智能体实例
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            bool: 是否成功移除
        """
        agent = self._instances.get(agent_id)
        if agent:
            await agent.cleanup()
            del self._instances[agent_id]
            self.logger.info(f"移除智能体实例: {agent_id}")
            return True
        return False
    
    async def cleanup_all(self):
        """清理所有智能体实例"""
        for agent_id in list(self._instances.keys()):
            await self.remove_agent_instance(agent_id)
        self.logger.info("所有智能体实例已清理")


# 全局智能体注册表
agent_registry = AgentRegistry()