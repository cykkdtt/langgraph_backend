"""
智能体使用核心模块完整指南

本文件展示了如何在创建的智能体中使用其他核心模块：
1. core/memory - 长期记忆管理
2. core/tools - 工具管理和MCP集成
3. core/streaming - 流式处理
4. core/time_travel - 时间旅行功能
5. core/optimization - 提示词优化
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 导入基础智能体类
from ..base import BaseAgent, ChatRequest, ChatResponse, StreamChunk
from ..memory_enhanced import MemoryEnhancedAgent

# ============================================================================
# 1. 使用 core/memory 模块
# ============================================================================

# 导入记忆相关模块
from ...memory import (
    LangMemManager,
    MemoryNamespace,
    MemoryScope,
    MemoryType,
    MemoryItem,
    MemoryQuery,
    get_memory_manager
)
from ...memory.tools import get_memory_tools, MemoryToolsFactory

# ============================================================================
# 2. 使用 core/tools 模块
# ============================================================================

# 导入工具管理模块
from ...tools import (
    ToolRegistry,
    ToolCategory,
    ToolPermission,
    ToolMetadata,
    ToolExecutionContext,
    ToolExecutionResult,
    BaseManagedTool
)

# 导入增强工具管理器
from ...tools.enhanced_tool_manager import (
    EnhancedToolManager,
    ToolExecutionMode,
    ToolValidationLevel,
    ToolValidator,
    get_enhanced_tool_manager
)

# 导入MCP工具管理器
from ...tools.mcp_manager import get_mcp_manager

# ============================================================================
# 3. 使用 core/streaming 模块
# ============================================================================

from ...streaming import (
    StreamManager,
    StreamType,
    StreamChunk as CoreStreamChunk,
    get_stream_manager
)

# ============================================================================
# 4. 使用 core/time_travel 模块
# ============================================================================

from ...time_travel import (
    TimeTravelManager,
    CheckpointManager,
    StateHistoryManager,
    get_time_travel_manager
)

# ============================================================================
# 5. 使用 core/optimization 模块
# ============================================================================

from ...optimization import (
    PromptOptimizer,
    get_prompt_optimizer
)

logger = logging.getLogger(__name__)


# ============================================================================
# 示例1：集成记忆功能的智能体
# ============================================================================

class MemoryIntegratedAgent(MemoryEnhancedAgent):
    """集成记忆功能的智能体示例
    
    展示如何使用 core/memory 模块的高级功能
    """
    
    def __init__(self, agent_id: str, llm, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name="记忆集成智能体",
            description="展示记忆模块集成的智能体",
            llm=llm,
            memory_config={
                "auto_store": True,
                "retrieval_limit": 10,
                "importance_threshold": 0.3
            },
            **kwargs
        )
    
    async def initialize(self):
        """初始化智能体和记忆系统"""
        await super().initialize()
        
        # 获取记忆管理器
        self.memory_manager = get_memory_manager()
        
        # 添加记忆工具
        memory_tools = await get_memory_tools(f"agent_{self.agent_id}")
        self.tools.extend(memory_tools)
        
        # 创建记忆工具工厂
        self.memory_tools_factory = MemoryToolsFactory()
        
        logger.info(f"记忆集成智能体初始化完成: {self.agent_id}")
    
    async def store_structured_knowledge(
        self,
        knowledge_data: Dict[str, Any],
        user_id: str,
        category: str = "general"
    ) -> str:
        """存储结构化知识"""
        namespace = MemoryNamespace(
            scope=MemoryScope.USER,
            identifier=user_id,
            sub_namespace=category
        )
        
        memory_item = MemoryItem(
            id=f"knowledge_{uuid.uuid4()}",
            content=str(knowledge_data),
            memory_type=MemoryType.SEMANTIC,
            metadata={
                "category": category,
                "structured": True,
                "agent_id": self.agent_id,
                **knowledge_data.get("metadata", {})
            },
            importance=knowledge_data.get("importance", 0.7)
        )
        
        return await self.memory_manager.store_memory(namespace, memory_item)
    
    async def search_memories_by_category(
        self,
        query: str,
        user_id: str,
        category: str,
        memory_type: MemoryType = MemoryType.SEMANTIC
    ) -> List[MemoryItem]:
        """按分类搜索记忆"""
        namespace = MemoryNamespace(
            scope=MemoryScope.USER,
            identifier=user_id,
            sub_namespace=category
        )
        
        memory_query = MemoryQuery(
            query=query,
            memory_type=memory_type,
            limit=5,
            min_importance=0.3
        )
        
        return await self.memory_manager.search_memories(namespace, memory_query)


# ============================================================================
# 示例2：集成工具管理的智能体
# ============================================================================

# 定义自定义工具
@tool
def advanced_search_tool(query: str, search_type: str = "web") -> str:
    """高级搜索工具"""
    return f"搜索结果: {query} (类型: {search_type})"

@tool
def data_analysis_tool(data: str, analysis_type: str = "basic") -> str:
    """数据分析工具"""
    return f"分析结果: {data} (类型: {analysis_type})"


class ToolIntegratedAgent(BaseAgent):
    """集成工具管理的智能体示例
    
    展示如何使用 core/tools 模块的功能
    """
    
    def __init__(self, agent_id: str, llm, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name="工具集成智能体",
            description="展示工具管理集成的智能体",
            llm=llm,
            tools=[advanced_search_tool, data_analysis_tool],
            **kwargs
        )
        
        # 工具管理器
        self.tool_registry = ToolRegistry()
        self.enhanced_tool_manager = None
        self.mcp_manager = None
    
    async def initialize(self):
        """初始化智能体和工具系统"""
        await super().initialize()
        
        # 获取增强工具管理器
        self.enhanced_tool_manager = get_enhanced_tool_manager()
        
        # 获取MCP管理器
        self.mcp_manager = get_mcp_manager()
        
        # 注册自定义工具到增强管理器
        for tool in self.tools:
            await self.enhanced_tool_manager.register_tool(
                tool,
                metadata={"category": "custom", "agent_id": self.agent_id},
                validation_level=ToolValidationLevel.BASIC
            )
        
        # 注册MCP工具
        mcp_count = await self.enhanced_tool_manager.register_mcp_tools()
        logger.info(f"注册了 {mcp_count} 个MCP工具")
        
        logger.info(f"工具集成智能体初始化完成: {self.agent_id}")
    
    async def execute_tool_with_context(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        user_id: str,
        session_id: str
    ) -> ToolExecutionResult:
        """使用上下文执行工具"""
        context = ToolExecutionContext(
            user_id=user_id,
            session_id=session_id,
            agent_id=self.agent_id,
            execution_id=str(uuid.uuid4()),
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        
        return await self.enhanced_tool_manager.execute_tool(
            tool_name,
            tool_input,
            context
        )
    
    async def execute_tools_parallel(
        self,
        tool_requests: List[Dict[str, Any]],
        user_id: str,
        session_id: str
    ) -> List[ToolExecutionResult]:
        """并行执行多个工具"""
        context = ToolExecutionContext(
            user_id=user_id,
            session_id=session_id,
            agent_id=self.agent_id,
            execution_id=str(uuid.uuid4())
        )
        
        return await self.enhanced_tool_manager.execute_tools_parallel(
            tool_requests,
            context
        )
    
    async def get_tool_statistics(self) -> Dict[str, Any]:
        """获取工具使用统计"""
        return await self.enhanced_tool_manager.get_execution_stats()


# ============================================================================
# 示例3：集成流式处理的智能体
# ============================================================================

class StreamingIntegratedAgent(BaseAgent):
    """集成流式处理的智能体示例
    
    展示如何使用 core/streaming 模块
    """
    
    def __init__(self, agent_id: str, llm, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name="流式处理智能体",
            description="展示流式处理集成的智能体",
            llm=llm,
            **kwargs
        )
        
        self.stream_manager = None
    
    async def initialize(self):
        """初始化智能体和流式处理系统"""
        await super().initialize()
        
        # 获取流式管理器
        self.stream_manager = get_stream_manager()
        
        logger.info(f"流式处理智能体初始化完成: {self.agent_id}")
    
    async def stream_chat_enhanced(
        self,
        request: ChatRequest,
        config: Optional[RunnableConfig] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """增强的流式对话处理"""
        try:
            # 创建流式会话
            stream_session = await self.stream_manager.create_stream_session(
                session_id=request.session_id,
                user_id=request.user_id,
                agent_id=self.agent_id
            )
            
            # 开始流式处理
            async for chunk in super().stream_chat(request, config):
                # 通过流式管理器处理
                enhanced_chunk = await self.stream_manager.process_chunk(
                    stream_session.session_id,
                    chunk
                )
                
                yield enhanced_chunk
            
            # 结束流式会话
            await self.stream_manager.end_stream_session(stream_session.session_id)
            
        except Exception as e:
            logger.error(f"增强流式处理失败: {e}")
            # 降级到基础流式处理
            async for chunk in super().stream_chat(request, config):
                yield chunk


# ============================================================================
# 示例4：集成时间旅行功能的智能体
# ============================================================================

class TimeTravelIntegratedAgent(BaseAgent):
    """集成时间旅行功能的智能体示例
    
    展示如何使用 core/time_travel 模块
    """
    
    def __init__(self, agent_id: str, llm, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name="时间旅行智能体",
            description="展示时间旅行功能集成的智能体",
            llm=llm,
            **kwargs
        )
        
        self.time_travel_manager = None
        self.checkpoint_manager = None
    
    async def initialize(self):
        """初始化智能体和时间旅行系统"""
        await super().initialize()
        
        # 获取时间旅行管理器
        self.time_travel_manager = get_time_travel_manager()
        self.checkpoint_manager = CheckpointManager()
        
        logger.info(f"时间旅行智能体初始化完成: {self.agent_id}")
    
    async def chat_with_checkpoints(
        self,
        request: ChatRequest,
        config: Optional[RunnableConfig] = None,
        create_checkpoint: bool = True
    ) -> ChatResponse:
        """带检查点的对话处理"""
        try:
            # 创建检查点（对话前）
            if create_checkpoint:
                checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                    session_id=request.session_id,
                    user_id=request.user_id,
                    agent_id=self.agent_id,
                    metadata={"type": "pre_chat", "timestamp": datetime.utcnow().isoformat()}
                )
                logger.info(f"创建对话前检查点: {checkpoint_id}")
            
            # 执行对话
            response = await super().chat(request, config)
            
            # 创建检查点（对话后）
            if create_checkpoint:
                post_checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                    session_id=request.session_id,
                    user_id=request.user_id,
                    agent_id=self.agent_id,
                    metadata={
                        "type": "post_chat",
                        "pre_checkpoint_id": checkpoint_id,
                        "response_length": len(response.message.content) if response.message else 0,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                logger.info(f"创建对话后检查点: {post_checkpoint_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"带检查点的对话处理失败: {e}")
            # 如果出错，可以选择回滚到之前的检查点
            if create_checkpoint and 'checkpoint_id' in locals():
                await self.rollback_to_checkpoint(checkpoint_id, request.session_id)
            raise
    
    async def rollback_to_checkpoint(
        self,
        checkpoint_id: str,
        session_id: str
    ) -> bool:
        """回滚到指定检查点"""
        try:
            success = await self.time_travel_manager.rollback_to_checkpoint(
                checkpoint_id,
                session_id
            )
            
            if success:
                logger.info(f"成功回滚到检查点: {checkpoint_id}")
            else:
                logger.error(f"回滚失败: {checkpoint_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"回滚操作异常: {e}")
            return False
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return await self.time_travel_manager.get_session_history(
            session_id,
            limit=limit
        )


# ============================================================================
# 示例5：集成提示词优化的智能体
# ============================================================================

class OptimizationIntegratedAgent(BaseAgent):
    """集成提示词优化的智能体示例
    
    展示如何使用 core/optimization 模块
    """
    
    def __init__(self, agent_id: str, llm, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name="优化集成智能体",
            description="展示提示词优化集成的智能体",
            llm=llm,
            **kwargs
        )
        
        self.prompt_optimizer = None
        self.optimization_enabled = True
    
    async def initialize(self):
        """初始化智能体和优化系统"""
        await super().initialize()
        
        # 获取提示词优化器
        self.prompt_optimizer = get_prompt_optimizer()
        
        logger.info(f"优化集成智能体初始化完成: {self.agent_id}")
    
    async def chat_with_optimization(
        self,
        request: ChatRequest,
        config: Optional[RunnableConfig] = None,
        optimize_prompt: bool = True
    ) -> ChatResponse:
        """带提示词优化的对话处理"""
        try:
            # 优化提示词
            if optimize_prompt and self.optimization_enabled:
                optimized_messages = await self._optimize_messages(request.messages)
                
                # 创建优化后的请求
                optimized_request = ChatRequest(
                    messages=optimized_messages,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    stream=request.stream,
                    metadata={
                        **request.metadata,
                        "prompt_optimized": True,
                        "optimization_timestamp": datetime.utcnow().isoformat()
                    }
                )
            else:
                optimized_request = request
            
            # 执行对话
            response = await super().chat(optimized_request, config)
            
            # 收集反馈用于进一步优化
            if optimize_prompt and self.optimization_enabled:
                await self._collect_optimization_feedback(
                    request.messages,
                    optimized_messages,
                    response,
                    request.user_id
                )
            
            return response
            
        except Exception as e:
            logger.error(f"带优化的对话处理失败: {e}")
            # 降级到基础对话处理
            return await super().chat(request, config)
    
    async def _optimize_messages(
        self,
        messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """优化消息列表"""
        try:
            # 提取系统提示和用户消息
            system_prompts = [msg for msg in messages if isinstance(msg, SystemMessage)]
            user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
            
            if system_prompts and user_messages:
                # 优化系统提示
                optimized_system_prompt = await self.prompt_optimizer.optimize_prompt(
                    prompt=system_prompts[0].content,
                    context={
                        "agent_id": self.agent_id,
                        "user_query": user_messages[-1].content
                    }
                )
                
                # 构建优化后的消息列表
                optimized_messages = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        optimized_messages.append(SystemMessage(content=optimized_system_prompt))
                    else:
                        optimized_messages.append(msg)
                
                return optimized_messages
            
            return messages
            
        except Exception as e:
            logger.error(f"消息优化失败: {e}")
            return messages
    
    async def _collect_optimization_feedback(
        self,
        original_messages: List[BaseMessage],
        optimized_messages: List[BaseMessage],
        response: ChatResponse,
        user_id: str
    ):
        """收集优化反馈"""
        try:
            feedback_data = {
                "agent_id": self.agent_id,
                "user_id": user_id,
                "original_prompt_length": sum(len(msg.content) for msg in original_messages),
                "optimized_prompt_length": sum(len(msg.content) for msg in optimized_messages),
                "response_length": len(response.message.content) if response.message else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.prompt_optimizer.collect_feedback(feedback_data)
            
        except Exception as e:
            logger.error(f"收集优化反馈失败: {e}")


# ============================================================================
# 示例6：全功能集成智能体
# ============================================================================

class FullyIntegratedAgent(MemoryEnhancedAgent):
    """全功能集成智能体
    
    集成所有核心模块功能的完整示例
    """
    
    def __init__(self, agent_id: str, llm, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name="全功能集成智能体",
            description="集成所有核心模块功能的智能体",
            llm=llm,
            tools=[advanced_search_tool, data_analysis_tool],
            memory_config={
                "auto_store": True,
                "retrieval_limit": 10,
                "importance_threshold": 0.3
            },
            **kwargs
        )
        
        # 核心模块管理器
        self.enhanced_tool_manager = None
        self.stream_manager = None
        self.time_travel_manager = None
        self.prompt_optimizer = None
    
    async def initialize(self):
        """初始化所有核心模块"""
        await super().initialize()
        
        # 初始化工具管理器
        self.enhanced_tool_manager = get_enhanced_tool_manager()
        for tool in self.tools:
            await self.enhanced_tool_manager.register_tool(tool)
        
        # 初始化流式管理器
        self.stream_manager = get_stream_manager()
        
        # 初始化时间旅行管理器
        self.time_travel_manager = get_time_travel_manager()
        
        # 初始化提示词优化器
        self.prompt_optimizer = get_prompt_optimizer()
        
        logger.info(f"全功能集成智能体初始化完成: {self.agent_id}")
    
    async def enhanced_chat(
        self,
        request: ChatRequest,
        config: Optional[RunnableConfig] = None,
        enable_optimization: bool = True,
        create_checkpoint: bool = True,
        use_enhanced_memory: bool = True
    ) -> ChatResponse:
        """增强的对话处理，集成所有功能"""
        try:
            # 1. 创建检查点
            checkpoint_id = None
            if create_checkpoint:
                checkpoint_id = await self.time_travel_manager.create_checkpoint(
                    session_id=request.session_id,
                    user_id=request.user_id,
                    agent_id=self.agent_id
                )
            
            # 2. 优化提示词
            if enable_optimization:
                request = await self._apply_prompt_optimization(request)
            
            # 3. 增强记忆检索
            if use_enhanced_memory:
                request = await self._enhance_with_advanced_memory(request)
            
            # 4. 执行对话
            response = await super().chat(request, config)
            
            # 5. 存储结构化记忆
            if use_enhanced_memory and response.message:
                await self._store_enhanced_memory(request, response)
            
            return response
            
        except Exception as e:
            logger.error(f"增强对话处理失败: {e}")
            
            # 错误恢复：回滚到检查点
            if checkpoint_id:
                await self.time_travel_manager.rollback_to_checkpoint(
                    checkpoint_id,
                    request.session_id
                )
            
            # 降级到基础处理
            return await super().chat(request, config)
    
    async def _apply_prompt_optimization(self, request: ChatRequest) -> ChatRequest:
        """应用提示词优化"""
        # 实现提示词优化逻辑
        return request
    
    async def _enhance_with_advanced_memory(self, request: ChatRequest) -> ChatRequest:
        """使用高级记忆增强"""
        # 实现高级记忆检索逻辑
        return request
    
    async def _store_enhanced_memory(self, request: ChatRequest, response: ChatResponse):
        """存储增强记忆"""
        # 实现增强记忆存储逻辑
        pass


# ============================================================================
# 使用示例和测试
# ============================================================================

async def test_integrated_agents():
    """测试集成智能体"""
    from config.settings import get_llm_by_name
    
    # 获取LLM
    llm = get_llm_by_name("qwen")
    
    # 测试记忆集成智能体
    print("=== 测试记忆集成智能体 ===")
    memory_agent = MemoryIntegratedAgent("memory_001", llm)
    await memory_agent.initialize()
    
    # 存储知识
    knowledge_id = await memory_agent.store_structured_knowledge(
        {
            "topic": "Python编程",
            "content": "Python是一种高级编程语言",
            "importance": 0.8
        },
        user_id="user_123",
        category="programming"
    )
    print(f"存储知识ID: {knowledge_id}")
    
    # 测试工具集成智能体
    print("\n=== 测试工具集成智能体 ===")
    tool_agent = ToolIntegratedAgent("tool_001", llm)
    await tool_agent.initialize()
    
    # 执行工具
    result = await tool_agent.execute_tool_with_context(
        "advanced_search_tool",
        {"query": "Python教程", "search_type": "web"},
        user_id="user_123",
        session_id="session_001"
    )
    print(f"工具执行结果: {result}")
    
    # 测试全功能集成智能体
    print("\n=== 测试全功能集成智能体 ===")
    full_agent = FullyIntegratedAgent("full_001", llm)
    await full_agent.initialize()
    
    request = ChatRequest(
        messages=[HumanMessage(content="请帮我搜索Python编程相关的资料")],
        user_id="user_123",
        session_id="session_001"
    )
    
    response = await full_agent.enhanced_chat(request)
    print(f"全功能智能体回复: {response.message.content}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_integrated_agents())