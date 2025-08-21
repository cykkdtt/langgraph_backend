"""
记忆增强的智能体基类

本模块提供集成LangMem长期记忆功能的智能体基类，支持：
- 自动记忆存储和检索
- 上下文感知的记忆管理
- 多种记忆类型（语义、情节、程序）
- 记忆驱动的对话增强
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, AsyncIterator, Union
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from .base import BaseAgent, AgentType, AgentStatus, StreamChunk, ChatRequest, ChatResponse
from ..memory import (
    LangMemManager, 
    MemoryNamespace, 
    MemoryScope, 
    MemoryType, 
    MemoryItem,
    MemoryQuery,
    get_memory_manager
)
from ..memory.tools import MemoryToolsFactory, get_memory_tools
from models.api_models import WebSocketMessage, WebSocketMessageType

logger = logging.getLogger(__name__)


class MemoryEnhancedAgent(BaseAgent):
    """记忆增强的智能体基类
    
    集成LangMem长期记忆功能，为智能体提供：
    - 自动记忆存储：重要对话和学习内容
    - 智能记忆检索：基于上下文的相关记忆召回
    - 记忆类型管理：语义、情节、程序记忆分类
    - 记忆工具集成：为智能体提供记忆管理工具
    """
    
    def __init__(
        self,
        config=None,
        memory_manager=None,
        **kwargs
    ):
        """初始化记忆增强智能体
        
        Args:
            config: 智能体配置对象
            memory_manager: 后台记忆管理器
            **kwargs: 其他参数
        """
        # 使用配置对象初始化基类
        super().__init__(config=config, **kwargs)
        
        # 后台记忆管理器
        self.background_memory_manager = memory_manager
        
        # 记忆配置
        memory_config = getattr(config, 'memory_config', {}) if config else {}
        self.auto_store_memories = memory_config.get("auto_store", True)
        self.memory_retrieval_limit = memory_config.get("retrieval_limit", 5)
        self.memory_importance_threshold = memory_config.get("importance_threshold", 0.4)
        
        # 记忆命名空间（从配置获取）
        self.memory_namespace = getattr(config, 'memory_namespace', None) if config else None
        
        # 记忆管理器（保持兼容性）
        self.memory_manager: Optional[LangMemManager] = None
        self.memory_tools_factory = MemoryToolsFactory()
        
        # 记忆命名空间缓存
        self.user_namespace_cache: Dict[str, MemoryNamespace] = {}
        self.agent_namespace_cache: Dict[str, MemoryNamespace] = {}
        
        logger.info(f"记忆增强智能体初始化: {self.agent_id}, 命名空间: {self.memory_namespace}")
    
    async def _build_graph(self) -> None:
        """构建支持工具调用的LangGraph图"""
        from langgraph.prebuilt import create_react_agent
        
        # 使用LangGraph的预构建ReAct智能体，支持工具调用
        if self.tools:
            # 创建支持工具的ReAct智能体
            self.compiled_graph = create_react_agent(
                model=self.llm,
                tools=self.tools,
                checkpointer=self.checkpointer
            )
            # 设置self.graph为None，表示使用预编译的图
            self.graph = None
            logger.info(f"创建支持工具调用的ReAct智能体，工具数量: {len(self.tools)}")
        else:
            # 如果没有工具，使用基础图
            await super()._build_graph()
            logger.info("创建基础对话智能体（无工具）")
    
    async def initialize(self) -> None:
        """初始化智能体和记忆系统"""
        # 初始化记忆管理器
        self.memory_manager = get_memory_manager()
        await self.memory_manager.initialize()
        
        # 添加协作工具到智能体工具集
        await self._add_collaboration_tools()
        
        # 添加记忆工具到智能体工具集
        if self.auto_store_memories:
            await self._add_memory_tools()
        
        # 构建图
        await self._build_graph()
        
        # 如果使用create_react_agent，它已经返回编译好的图
        if hasattr(self, 'compiled_graph') and self.compiled_graph is not None:
            # create_react_agent已经返回编译好的图，无需再次编译
            self.initialized = True
            logger.info(f"使用预编译的ReAct智能体: {self.agent_id}")
        else:
            # 使用父类的初始化逻辑来编译图
            if self.graph is not None:
                self.compiled_graph = self.graph.compile(
                    checkpointer=self.checkpointer,
                    interrupt_before=self.interrupt_before,
                    interrupt_after=self.interrupt_after
                )
                self.initialized = True
                logger.info(f"编译基础图完成: {self.agent_id}")
            else:
                raise ValueError("图构建失败：self.graph 和 self.compiled_graph 都为 None")
        
        logger.info(f"记忆增强智能体初始化完成: {self.agent_id}")
    
    async def _add_memory_tools(self) -> None:
        """添加LangMem官方记忆管理工具到智能体"""
        try:
            # 使用配置的命名空间或默认命名空间
            namespace = self.memory_namespace or f"agent_{self.agent_id}"
            
            # 创建包装好的记忆工具
            memory_tools = await self.memory_tools_factory.create_memory_tools(namespace)
            
            # 添加到智能体工具集
            self.tools.extend(memory_tools)
            
            logger.info(f"LangMem官方记忆工具添加成功: 2 个工具 (管理+搜索), 命名空间: {namespace}")
            
        except Exception as e:
            logger.error(f"添加LangMem记忆工具失败: {e}")
    
    async def _add_collaboration_tools(self) -> None:
        """添加协作工具到智能体"""
        try:
            from ..tools.collaboration_tools import get_all_collaboration_tools
            
            # 获取所有协作工具
            collaboration_tools = get_all_collaboration_tools()
            
            # 添加到智能体工具集
            self.tools.extend(collaboration_tools)
            
            logger.info(f"协作工具添加成功: {len(collaboration_tools)} 个工具")
            
        except Exception as e:
            logger.error(f"添加协作工具失败: {e}")
    
    def _get_user_namespace(self, user_id: str) -> MemoryNamespace:
        """获取用户记忆命名空间"""
        if user_id not in self.user_namespace_cache:
            self.user_namespace_cache[user_id] = MemoryNamespace(
                scope=MemoryScope.USER,
                identifier=user_id
            )
        return self.user_namespace_cache[user_id]
    
    def _get_agent_namespace(self, session_id: str) -> MemoryNamespace:
        """获取智能体记忆命名空间"""
        if session_id not in self.agent_namespace_cache:
            self.agent_namespace_cache[session_id] = MemoryNamespace(
                scope=MemoryScope.AGENT,
                identifier=self.agent_id,
                sub_namespace=session_id
            )
        return self.agent_namespace_cache[session_id]
    
    async def _store_conversation_memory(
        self,
        user_message: str,
        ai_response: str,
        user_id: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """存储对话记忆 - 使用LangMem官方智能体主动管理"""
        try:
            # 注意：现在记忆存储由智能体通过工具主动决定
            # 不再使用算法自动评估和存储
            # 智能体会根据对话内容主动调用记忆管理工具来保存重要信息
            
            logger.info(
                "记忆存储现在由智能体主动管理，通过LangMem工具决定保存什么内容"
            )
            
        except Exception as e:
            logger.error(f"记忆存储配置失败: {e}")
    
    # 移除了 _calculate_importance 方法，因为现在使用LangMem官方智能体主动管理记忆
    
    async def _retrieve_relevant_memories(
        self,
        query: str,
        user_id: str,
        session_id: str,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[MemoryItem]:
        """检索相关记忆 - 现在由智能体通过LangMem工具主动搜索"""
        try:
            # 注意：现在记忆检索由智能体通过搜索工具主动进行
            # 不再在后台自动检索记忆
            # 智能体会根据需要主动调用记忆搜索工具
            
            logger.info(
                "记忆检索现在由智能体主动管理，通过LangMem搜索工具获取相关记忆"
            )
            return []
            
        except Exception as e:
            logger.error(f"记忆检索配置失败: {e}")
            return []
    
    def _format_memories_for_context(self, memories: List[MemoryItem]) -> str:
        """格式化记忆为上下文字符串"""
        if not memories:
            return ""
        
        context_parts = ["相关记忆信息:"]
        
        for i, memory in enumerate(memories, 1):
            memory_type_name = {
                MemoryType.SEMANTIC: "知识",
                MemoryType.EPISODIC: "经历", 
                MemoryType.PROCEDURAL: "技能"
            }.get(memory.memory_type, "记忆")
            
            context_parts.append(
                f"{i}. [{memory_type_name}] {memory.content[:200]}..."
                if len(memory.content) > 200 else f"{i}. [{memory_type_name}] {memory.content}"
            )
        
        return "\n".join(context_parts)
    
    async def _enhance_messages_with_memory(
        self,
        messages: List[BaseMessage],
        user_id: str,
        session_id: str
    ) -> List[BaseMessage]:
        """使用记忆增强消息列表"""
        if not messages:
            return messages
        
        # 获取最后一条用户消息作为查询
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break
        
        if not last_user_message:
            return messages
        
        # 检索相关记忆
        relevant_memories = await self._retrieve_relevant_memories(
            last_user_message, user_id, session_id
        )
        
        if not relevant_memories:
            return messages
        
        # 格式化记忆上下文
        memory_context = self._format_memories_for_context(relevant_memories)
        
        # 在系统消息中添加记忆上下文
        enhanced_messages = []
        system_message_added = False
        
        for msg in messages:
            if isinstance(msg, SystemMessage) and not system_message_added:
                # 在现有系统消息中添加记忆上下文
                enhanced_content = f"{msg.content}\n\n{memory_context}"
                enhanced_messages.append(SystemMessage(content=enhanced_content))
                system_message_added = True
            else:
                enhanced_messages.append(msg)
        
        # 如果没有系统消息，添加一个包含记忆上下文的系统消息
        if not system_message_added:
            memory_system_message = SystemMessage(content=memory_context)
            enhanced_messages.insert(0, memory_system_message)
        
        return enhanced_messages
    
    async def chat(
        self,
        request,  # 使用通用类型，支持不同的ChatRequest格式
        config: Optional[RunnableConfig] = None
    ) -> ChatResponse:
        """LangMem智能体主动管理记忆的对话处理"""
        try:
            # 确保初始化
            if not self.initialized:
                await self.initialize()
            
            # 适配不同的ChatRequest格式
            if hasattr(request, 'messages'):
                # core.agents.base.ChatRequest格式
                messages = request.messages
            elif hasattr(request, 'message'):
                # models.chat_models.ChatRequest格式
                messages = [HumanMessage(content=request.message)]
            else:
                raise ValueError("不支持的ChatRequest格式")
            
            # 准备状态
            state = {
                "messages": messages
            }
            
            # 执行支持工具调用的图
            result = await self.compiled_graph.ainvoke(state, config=config)
            
            # 获取最后一条AI消息
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            last_message = ai_messages[-1] if ai_messages else AIMessage(content="无响应")
            
            logger.info("对话处理完成，智能体可通过LangMem工具主动管理记忆")
            
            return ChatResponse(
                message=last_message,
                session_id=request.session_id,
                agent_id=self.agent_id,
                metadata=result.get("metadata", {}),
            )
            
        except Exception as e:
            logger.error(f"对话处理失败: {e}")
            # 降级到基础对话处理
            return await super().chat(request, config)
    
    async def stream_chat(
        self,
        request,  # 使用通用类型，支持不同的ChatRequest格式
        config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[StreamChunk]:
        """LangMem智能体主动管理记忆的流式对话处理"""
        try:
            # 确保初始化
            if not self.initialized:
                await self.initialize()
            
            # 适配不同的ChatRequest格式
            if hasattr(request, 'messages'):
                # core.agents.base.ChatRequest格式
                messages = request.messages
            elif hasattr(request, 'message'):
                # models.chat_models.ChatRequest格式
                messages = [HumanMessage(content=request.message)]
            else:
                raise ValueError("不支持的ChatRequest格式")
            
            # 准备状态
            state = {
                "messages": messages
            }
            
            # 流式执行支持工具调用的图
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
            
            logger.info("流式对话处理完成，智能体可通过LangMem工具主动管理记忆")
            
        except Exception as e:
            logger.error(f"流式对话处理失败: {e}")
            # 降级到基础流式处理
            async for chunk in super().stream_chat(request, config):
                yield chunk
    
    async def store_knowledge(
        self,
        content: str,
        user_id: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.7
    ) -> str:
        """存储知识记忆
        
        Args:
            content: 知识内容
            user_id: 用户ID
            memory_type: 记忆类型
            metadata: 元数据
            importance: 重要性评分
            
        Returns:
            str: 记忆ID
        """
        if not self.memory_manager:
            await self.initialize()
        
        namespace = self._get_user_namespace(user_id)
        
        memory_item = MemoryItem(
            id=f"knowledge_{user_id}_{datetime.utcnow().timestamp()}",
            content=content,
            memory_type=memory_type,
            metadata={
                "user_id": user_id,
                "agent_id": self.agent_id,
                "stored_by_agent": True,
                **(metadata or {})
            },
            importance=importance
        )
        
        return await self.memory_manager.store_memory(namespace, memory_item)
    
    async def get_memory_stats(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """获取记忆统计信息"""
        if not self.memory_manager:
            return {}
        
        try:
            user_namespace = self._get_user_namespace(user_id)
            agent_namespace = self._get_agent_namespace(session_id)
            
            user_stats = await self.memory_manager.get_memory_stats(user_namespace)
            agent_stats = await self.memory_manager.get_memory_stats(agent_namespace)
            
            return {
                "user_memories": user_stats,
                "session_memories": agent_stats,
                "total_memories": user_stats.get("total_count", 0) + agent_stats.get("total_count", 0)
            }
            
        except Exception as e:
            logger.error(f"获取记忆统计失败: {e}")
            return {}
    
    async def cleanup_old_memories(
        self,
        user_id: str,
        session_id: str,
        days: int = 90,
        min_importance: float = 0.1
    ) -> Dict[str, int]:
        """清理旧记忆"""
        if not self.memory_manager:
            return {"user_deleted": 0, "session_deleted": 0}
        
        try:
            user_namespace = self._get_user_namespace(user_id)
            agent_namespace = self._get_agent_namespace(session_id)
            
            user_deleted = await self.memory_manager.cleanup_old_memories(
                user_namespace, days, min_importance
            )
            session_deleted = await self.memory_manager.cleanup_old_memories(
                agent_namespace, days, min_importance
            )
            
            return {
                "user_deleted": user_deleted,
                "session_deleted": session_deleted
            }
            
        except Exception as e:
            logger.error(f"清理旧记忆失败: {e}")
            return {"user_deleted": 0, "session_deleted": 0}
    
    async def _send_memory_saved_event(
        self,
        user_id: str,
        session_id: str,
        memory_item: MemoryItem,
        important_info: List[str]
    ) -> None:
        """发送记忆保存事件到前端"""
        try:
            # 使用绝对导入获取WebSocket连接管理器
            from api.websocket import connection_manager
            from models.chat_models import WebSocketMessage, WebSocketMessageType
            import uuid
            
            # 构建记忆保存事件数据
            # 优先显示重要信息摘要，而不是完整对话内容
            if important_info and len(important_info) > 0:
                # 如果有重要信息，显示重要信息列表
                content_summary = "保存了以下重要信息: " + ", ".join(important_info)
            else:
                # 如果没有重要信息，显示通用提示
                content_summary = f"保存了{memory_item.memory_type.value}记忆 (重要性: {memory_item.importance:.1f})"
            
            memory_data = {
                "memory_id": memory_item.id,
                "content": content_summary,
                "memory_type": memory_item.memory_type.value,
                "importance": memory_item.importance,
                "important_info": important_info,
                "timestamp": memory_item.created_at.isoformat() if hasattr(memory_item, 'created_at') else datetime.utcnow().isoformat(),
                "agent_id": self.agent_id,
                "agent_name": self.name
            }
            
            # 创建WebSocket消息
            ws_message = WebSocketMessage(
                type=WebSocketMessageType.MEMORY_SAVED,
                data=memory_data,
                user_id=user_id,
                session_id=session_id
            )
            
            # 发送到用户的所有连接
            await connection_manager.send_to_user(user_id, ws_message)
            
            logger.info(f"✅ 记忆保存事件已发送到前端 - user_id: {user_id}, memory_id: {memory_item.id}")
            
        except Exception as e:
            logger.error(f"❌ 发送记忆保存事件失败: {e}", exc_info=True)