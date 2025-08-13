"""
智能体集成核心模块的实际示例

本文件展示了如何在实际项目中创建集成多个核心模块的智能体。
包含完整的导入、初始化、使用和错误处理代码。
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 导入基础智能体类
from ..base import BaseAgent, ChatRequest, ChatResponse, StreamChunk
from ..memory_enhanced import MemoryEnhancedAgent

# ============================================================================
# 导入核心模块
# ============================================================================

# 1. 记忆模块
try:
    from ...memory import (
        LangMemManager,
        MemoryNamespace,
        MemoryScope,
        MemoryType,
        MemoryItem,
        MemoryQuery
    )
    from ...memory.tools import get_memory_tools
    from ...memory.store_manager import get_memory_manager
    MEMORY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"记忆模块导入失败: {e}")
    MEMORY_AVAILABLE = False

# 2. 工具模块
try:
    from ...tools import (
        ToolRegistry,
        ToolCategory,
        ToolPermission,
        ToolMetadata,
        ToolExecutionContext,
        ToolExecutionResult,
        BaseManagedTool
    )
    from ...tools.enhanced_tool_manager import (
        EnhancedToolManager,
        ToolExecutionMode,
        ToolValidationLevel,
        get_enhanced_tool_manager
    )
    from ...tools.mcp_manager import get_mcp_manager
    TOOLS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"工具模块导入失败: {e}")
    TOOLS_AVAILABLE = False

# 3. 流式处理模块
try:
    from ...streaming import (
        StreamManager,
        StreamType,
        get_stream_manager
    )
    from ...streaming.stream_types import StreamChunk as CoreStreamChunk
    STREAMING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"流式处理模块导入失败: {e}")
    STREAMING_AVAILABLE = False

# 4. 时间旅行模块
try:
    from ...time_travel import (
        TimeTravelManager,
        get_time_travel_manager
    )
    from ...time_travel.checkpoint_manager import CheckpointManager
    from ...time_travel.state_history_manager import StateHistoryManager
    TIME_TRAVEL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"时间旅行模块导入失败: {e}")
    TIME_TRAVEL_AVAILABLE = False

# 5. 优化模块
try:
    from ...optimization import (
        PromptOptimizer,
        get_prompt_optimizer
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"优化模块导入失败: {e}")
    OPTIMIZATION_AVAILABLE = False

# 6. 工作流模块
try:
    from ...workflows import (
        WorkflowBuilder,
        ConditionalRouter,
        ParallelExecutor
    )
    WORKFLOWS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"工作流模块导入失败: {e}")
    WORKFLOWS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# 定义自定义工具
# ============================================================================

@tool
def search_knowledge_tool(query: str, category: str = "general") -> str:
    """搜索知识库工具"""
    return f"搜索结果: 找到关于'{query}'的{category}类别信息"

@tool
def analyze_data_tool(data: str, analysis_type: str = "basic") -> str:
    """数据分析工具"""
    return f"数据分析完成: {data[:50]}... (分析类型: {analysis_type})"

@tool
def generate_report_tool(content: str, format_type: str = "markdown") -> str:
    """生成报告工具"""
    return f"报告已生成 ({format_type}格式): {content[:100]}..."


# ============================================================================
# 智能体实现
# ============================================================================

class IntegratedAgent(MemoryEnhancedAgent):
    """集成多个核心模块的智能体示例
    
    这个智能体展示了如何安全地集成和使用各个核心模块，
    包括适当的错误处理和降级机制。
    """
    
    def __init__(self, agent_id: str, llm, **kwargs):
        # 基础配置
        super().__init__(
            agent_id=agent_id,
            name="集成智能体",
            description="展示核心模块集成的智能体",
            llm=llm,
            tools=[search_knowledge_tool, analyze_data_tool, generate_report_tool],
            memory_config={
                "auto_store": True,
                "retrieval_limit": 10,
                "importance_threshold": 0.3
            } if MEMORY_AVAILABLE else None,
            **kwargs
        )
        
        # 核心模块管理器
        self.enhanced_tool_manager = None
        self.stream_manager = None
        self.time_travel_manager = None
        self.prompt_optimizer = None
        self.workflow_builder = None
        
        # 功能开关
        self.features = {
            "memory": MEMORY_AVAILABLE,
            "tools": TOOLS_AVAILABLE,
            "streaming": STREAMING_AVAILABLE,
            "time_travel": TIME_TRAVEL_AVAILABLE,
            "optimization": OPTIMIZATION_AVAILABLE,
            "workflows": WORKFLOWS_AVAILABLE
        }
        
        logger.info(f"智能体功能状态: {self.features}")
    
    async def initialize(self):
        """初始化智能体和所有可用的核心模块"""
        await super().initialize()
        
        # 1. 初始化工具管理器
        if self.features["tools"]:
            try:
                self.enhanced_tool_manager = get_enhanced_tool_manager()
                
                # 注册自定义工具
                for tool in self.tools:
                    await self.enhanced_tool_manager.register_tool(
                        tool,
                        metadata={
                            "category": "custom",
                            "agent_id": self.agent_id,
                            "created_at": datetime.utcnow().isoformat()
                        },
                        validation_level=ToolValidationLevel.BASIC
                    )
                
                # 尝试注册MCP工具
                try:
                    mcp_count = await self.enhanced_tool_manager.register_mcp_tools()
                    logger.info(f"注册了 {mcp_count} 个MCP工具")
                except Exception as e:
                    logger.warning(f"MCP工具注册失败: {e}")
                
                logger.info("工具管理器初始化成功")
                
            except Exception as e:
                logger.error(f"工具管理器初始化失败: {e}")
                self.features["tools"] = False
        
        # 2. 初始化流式管理器
        if self.features["streaming"]:
            try:
                self.stream_manager = get_stream_manager()
                logger.info("流式管理器初始化成功")
            except Exception as e:
                logger.error(f"流式管理器初始化失败: {e}")
                self.features["streaming"] = False
        
        # 3. 初始化时间旅行管理器
        if self.features["time_travel"]:
            try:
                self.time_travel_manager = get_time_travel_manager()
                logger.info("时间旅行管理器初始化成功")
            except Exception as e:
                logger.error(f"时间旅行管理器初始化失败: {e}")
                self.features["time_travel"] = False
        
        # 4. 初始化提示词优化器
        if self.features["optimization"]:
            try:
                self.prompt_optimizer = get_prompt_optimizer()
                logger.info("提示词优化器初始化成功")
            except Exception as e:
                logger.error(f"提示词优化器初始化失败: {e}")
                self.features["optimization"] = False
        
        # 5. 初始化工作流构建器
        if self.features["workflows"]:
            try:
                self.workflow_builder = WorkflowBuilder()
                logger.info("工作流构建器初始化成功")
            except Exception as e:
                logger.error(f"工作流构建器初始化失败: {e}")
                self.features["workflows"] = False
        
        # 6. 添加记忆工具（如果记忆功能可用）
        if self.features["memory"]:
            try:
                memory_tools = await get_memory_tools(f"agent_{self.agent_id}")
                self.tools.extend(memory_tools)
                logger.info(f"添加了 {len(memory_tools)} 个记忆工具")
            except Exception as e:
                logger.error(f"记忆工具添加失败: {e}")
        
        logger.info(f"集成智能体初始化完成: {self.agent_id}")
        logger.info(f"最终功能状态: {self.features}")
    
    async def enhanced_chat(
        self,
        request: ChatRequest,
        config: Optional[RunnableConfig] = None,
        enable_optimization: bool = True,
        create_checkpoint: bool = True,
        use_enhanced_memory: bool = True
    ) -> ChatResponse:
        """增强的对话处理，集成所有可用功能"""
        checkpoint_id = None
        
        try:
            # 1. 创建检查点（如果时间旅行功能可用）
            if create_checkpoint and self.features["time_travel"]:
                try:
                    checkpoint_id = await self._create_checkpoint(request)
                    logger.debug(f"创建检查点: {checkpoint_id}")
                except Exception as e:
                    logger.warning(f"检查点创建失败: {e}")
            
            # 2. 优化提示词（如果优化功能可用）
            if enable_optimization and self.features["optimization"]:
                try:
                    request = await self._optimize_request(request)
                    logger.debug("提示词优化完成")
                except Exception as e:
                    logger.warning(f"提示词优化失败: {e}")
            
            # 3. 增强记忆检索（如果记忆功能可用）
            if use_enhanced_memory and self.features["memory"]:
                try:
                    request = await self._enhance_with_memory(request)
                    logger.debug("记忆增强完成")
                except Exception as e:
                    logger.warning(f"记忆增强失败: {e}")
            
            # 4. 执行对话
            response = await super().chat(request, config)
            
            # 5. 存储对话记忆（如果记忆功能可用）
            if use_enhanced_memory and self.features["memory"] and response.message:
                try:
                    await self._store_conversation_memory(request, response)
                    logger.debug("对话记忆存储完成")
                except Exception as e:
                    logger.warning(f"对话记忆存储失败: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"增强对话处理失败: {e}")
            
            # 错误恢复：回滚到检查点
            if checkpoint_id and self.features["time_travel"]:
                try:
                    await self._rollback_to_checkpoint(checkpoint_id, request.session_id)
                    logger.info(f"已回滚到检查点: {checkpoint_id}")
                except Exception as rollback_error:
                    logger.error(f"回滚失败: {rollback_error}")
            
            # 降级到基础处理
            return await super().chat(request, config)
    
    async def stream_chat_enhanced(
        self,
        request: ChatRequest,
        config: Optional[RunnableConfig] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """增强的流式对话处理"""
        stream_session = None
        
        try:
            # 创建流式会话（如果流式功能可用）
            if self.features["streaming"]:
                try:
                    stream_session = await self.stream_manager.create_stream_session(
                        session_id=request.session_id,
                        user_id=request.user_id,
                        agent_id=self.agent_id
                    )
                    logger.debug(f"创建流式会话: {stream_session.session_id}")
                except Exception as e:
                    logger.warning(f"流式会话创建失败: {e}")
            
            # 开始流式处理
            async for chunk in super().stream_chat(request, config):
                # 通过流式管理器处理（如果可用）
                if stream_session and self.features["streaming"]:
                    try:
                        enhanced_chunk = await self.stream_manager.process_chunk(
                            stream_session.session_id,
                            chunk
                        )
                        yield enhanced_chunk
                    except Exception as e:
                        logger.warning(f"流式块处理失败: {e}")
                        yield chunk
                else:
                    yield chunk
            
            # 结束流式会话
            if stream_session and self.features["streaming"]:
                try:
                    await self.stream_manager.end_stream_session(stream_session.session_id)
                    logger.debug("流式会话结束")
                except Exception as e:
                    logger.warning(f"流式会话结束失败: {e}")
                    
        except Exception as e:
            logger.error(f"增强流式处理失败: {e}")
            # 降级到基础流式处理
            async for chunk in super().stream_chat(request, config):
                yield chunk
    
    async def execute_tool_with_context(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        user_id: str,
        session_id: str
    ) -> Optional[ToolExecutionResult]:
        """使用上下文执行工具"""
        if not self.features["tools"]:
            logger.warning("工具管理功能不可用")
            return None
        
        try:
            context = ToolExecutionContext(
                user_id=user_id,
                session_id=session_id,
                agent_id=self.agent_id,
                execution_id=str(uuid.uuid4()),
                metadata={"timestamp": datetime.utcnow().isoformat()}
            )
            
            result = await self.enhanced_tool_manager.execute_tool(
                tool_name,
                tool_input,
                context
            )
            
            logger.info(f"工具执行成功: {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            return None
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取对话历史"""
        if not self.features["time_travel"]:
            logger.warning("时间旅行功能不可用")
            return []
        
        try:
            history = await self.time_travel_manager.get_session_history(
                session_id,
                limit=limit
            )
            return history
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        status = {
            "agent_id": self.agent_id,
            "name": self.name,
            "features": self.features,
            "tools_count": len(self.tools),
            "initialized": True
        }
        
        # 添加工具统计（如果可用）
        if self.features["tools"] and self.enhanced_tool_manager:
            try:
                tool_stats = await self.enhanced_tool_manager.get_execution_stats()
                status["tool_stats"] = tool_stats
            except Exception as e:
                logger.warning(f"获取工具统计失败: {e}")
        
        # 添加记忆统计（如果可用）
        if self.features["memory"]:
            try:
                memory_stats = await self.get_memory_stats(
                    user_id="system",
                    session_id="status_check"
                )
                status["memory_stats"] = memory_stats
            except Exception as e:
                logger.warning(f"获取记忆统计失败: {e}")
        
        return status
    
    # ========================================================================
    # 私有辅助方法
    # ========================================================================
    
    async def _create_checkpoint(self, request: ChatRequest) -> Optional[str]:
        """创建检查点"""
        if not self.time_travel_manager:
            return None
        
        checkpoint_manager = CheckpointManager()
        return await checkpoint_manager.create_checkpoint(
            session_id=request.session_id,
            user_id=request.user_id,
            agent_id=self.agent_id,
            metadata={
                "type": "pre_chat",
                "timestamp": datetime.utcnow().isoformat(),
                "message_count": len(request.messages)
            }
        )
    
    async def _rollback_to_checkpoint(self, checkpoint_id: str, session_id: str):
        """回滚到检查点"""
        if not self.time_travel_manager:
            return
        
        await self.time_travel_manager.rollback_to_checkpoint(
            checkpoint_id,
            session_id
        )
    
    async def _optimize_request(self, request: ChatRequest) -> ChatRequest:
        """优化请求"""
        if not self.prompt_optimizer:
            return request
        
        # 优化系统提示
        system_messages = [msg for msg in request.messages if isinstance(msg, SystemMessage)]
        if system_messages:
            optimized_prompt = await self.prompt_optimizer.optimize_prompt(
                prompt=system_messages[0].content,
                context={
                    "agent_id": self.agent_id,
                    "user_id": request.user_id,
                    "session_id": request.session_id
                }
            )
            
            # 替换优化后的系统提示
            optimized_messages = []
            for msg in request.messages:
                if isinstance(msg, SystemMessage):
                    optimized_messages.append(SystemMessage(content=optimized_prompt))
                else:
                    optimized_messages.append(msg)
            
            request.messages = optimized_messages
        
        return request
    
    async def _enhance_with_memory(self, request: ChatRequest) -> ChatRequest:
        """使用记忆增强请求"""
        # 这里可以添加额外的记忆增强逻辑
        # 基础的记忆增强已经在 MemoryEnhancedAgent 中实现
        return request
    
    async def _store_conversation_memory(self, request: ChatRequest, response: ChatResponse):
        """存储对话记忆"""
        if not MEMORY_AVAILABLE:
            return
        
        try:
            # 存储结构化对话记忆
            conversation_data = {
                "user_message": request.messages[-1].content if request.messages else "",
                "ai_response": response.message.content if response.message else "",
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": request.session_id,
                "agent_id": self.agent_id
            }
            
            await self.store_knowledge(
                content=str(conversation_data),
                user_id=request.user_id,
                memory_type=MemoryType.EPISODIC,
                metadata={
                    "category": "conversation",
                    "session_id": request.session_id,
                    "agent_id": self.agent_id
                },
                importance=0.6
            )
            
        except Exception as e:
            logger.error(f"对话记忆存储失败: {e}")


# ============================================================================
# 使用示例和测试
# ============================================================================

async def test_integrated_agent():
    """测试集成智能体"""
    try:
        # 这里需要根据实际项目配置获取LLM
        # from config.settings import get_llm_by_name
        # llm = get_llm_by_name("qwen")
        
        # 为了示例，我们使用一个模拟的LLM
        class MockLLM:
            async def ainvoke(self, messages):
                return AIMessage(content="这是一个模拟的AI回复")
        
        llm = MockLLM()
        
        # 创建集成智能体
        agent = IntegratedAgent("integrated_001", llm)
        await agent.initialize()
        
        # 获取智能体状态
        status = await agent.get_agent_status()
        print("=== 智能体状态 ===")
        print(f"智能体ID: {status['agent_id']}")
        print(f"名称: {status['name']}")
        print(f"可用功能: {status['features']}")
        print(f"工具数量: {status['tools_count']}")
        
        # 测试基础对话
        print("\n=== 测试基础对话 ===")
        request = ChatRequest(
            messages=[HumanMessage(content="你好，请介绍一下你的功能")],
            user_id="test_user",
            session_id="test_session_001"
        )
        
        response = await agent.enhanced_chat(request)
        print(f"AI回复: {response.message.content}")
        
        # 测试工具执行（如果可用）
        if agent.features["tools"]:
            print("\n=== 测试工具执行 ===")
            tool_result = await agent.execute_tool_with_context(
                "search_knowledge_tool",
                {"query": "Python编程", "category": "技术"},
                user_id="test_user",
                session_id="test_session_001"
            )
            if tool_result:
                print(f"工具执行结果: {tool_result}")
        
        # 测试对话历史（如果可用）
        if agent.features["time_travel"]:
            print("\n=== 测试对话历史 ===")
            history = await agent.get_conversation_history("test_session_001")
            print(f"对话历史条数: {len(history)}")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    asyncio.run(test_integrated_agent())