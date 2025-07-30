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
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from .base import BaseAgent, AgentType, AgentStatus, ChatRequest, ChatResponse, StreamChunk
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
        agent_id: str,
        name: str,
        description: str,
        llm,
        tools: Optional[List] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """初始化记忆增强智能体
        
        Args:
            agent_id: 智能体ID
            name: 智能体名称
            description: 智能体描述
            llm: 语言模型
            tools: 工具列表
            checkpointer: 检查点保存器
            memory_config: 记忆配置
            **kwargs: 其他参数
        """
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            llm=llm,
            tools=tools or [],
            checkpointer=checkpointer,
            **kwargs
        )
        
        # 记忆配置
        self.memory_config = memory_config or {}
        self.auto_store_memories = self.memory_config.get("auto_store", True)
        self.memory_retrieval_limit = self.memory_config.get("retrieval_limit", 5)
        self.memory_importance_threshold = self.memory_config.get("importance_threshold", 0.3)
        
        # 记忆管理器
        self.memory_manager: Optional[LangMemManager] = None
        self.memory_tools_factory = MemoryToolsFactory()
        
        # 记忆命名空间
        self.user_namespace_cache: Dict[str, MemoryNamespace] = {}
        self.agent_namespace_cache: Dict[str, MemoryNamespace] = {}
        
        logger.info(f"记忆增强智能体初始化: {self.agent_id}")
    
    async def initialize(self) -> None:
        """初始化智能体和记忆系统"""
        await super().initialize()
        
        # 初始化记忆管理器
        self.memory_manager = get_memory_manager()
        await self.memory_manager.initialize()
        
        # 添加记忆工具到智能体工具集
        if self.auto_store_memories:
            await self._add_memory_tools()
        
        logger.info(f"记忆增强智能体初始化完成: {self.agent_id}")
    
    async def _add_memory_tools(self) -> None:
        """添加记忆管理工具到智能体"""
        try:
            # 创建通用记忆工具
            memory_tools = await get_memory_tools(f"agent_{self.agent_id}")
            
            # 添加到智能体工具集
            self.tools.extend(memory_tools)
            
            logger.info(f"记忆工具添加成功: {len(memory_tools)} 个工具")
            
        except Exception as e:
            logger.error(f"添加记忆工具失败: {e}")
    
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
        """存储对话记忆"""
        if not self.auto_store_memories or not self.memory_manager:
            return
        
        try:
            # 用户命名空间 - 存储用户相关记忆
            user_namespace = self._get_user_namespace(user_id)
            
            # 智能体命名空间 - 存储会话相关记忆
            agent_namespace = self._get_agent_namespace(session_id)
            
            # 构建对话记忆内容
            conversation_content = f"用户: {user_message}\n智能体: {ai_response}"
            
            # 计算重要性评分（简单实现，可以使用更复杂的算法）
            importance_score = self._calculate_importance(user_message, ai_response)
            
            # 准备元数据
            memory_metadata = {
                "user_id": user_id,
                "session_id": session_id,
                "agent_id": self.agent_id,
                "conversation_turn": True,
                **(metadata or {})
            }
            
            # 存储情节记忆（对话历史）
            if importance_score >= self.memory_importance_threshold:
                episodic_memory = MemoryItem(
                    id=f"conv_{session_id}_{datetime.utcnow().timestamp()}",
                    content=conversation_content,
                    memory_type=MemoryType.EPISODIC,
                    metadata=memory_metadata,
                    importance=importance_score
                )
                
                await self.memory_manager.store_memory(agent_namespace, episodic_memory)
                logger.debug(f"存储对话记忆: session={session_id}, importance={importance_score}")
            
        except Exception as e:
            logger.error(f"存储对话记忆失败: {e}")
    
    def _calculate_importance(self, user_message: str, ai_response: str) -> float:
        """计算记忆重要性评分
        
        简单的启发式算法，实际应用中可以使用更复杂的模型
        """
        importance = 0.5  # 基础分数
        
        # 消息长度因子
        total_length = len(user_message) + len(ai_response)
        if total_length > 200:
            importance += 0.1
        if total_length > 500:
            importance += 0.1
        
        # 关键词检测
        important_keywords = [
            "重要", "关键", "记住", "学习", "总结", "计划", "目标", 
            "问题", "解决", "方案", "建议", "决定", "选择"
        ]
        
        text = user_message + " " + ai_response
        keyword_count = sum(1 for keyword in important_keywords if keyword in text)
        importance += min(keyword_count * 0.05, 0.2)
        
        # 问号数量（表示用户的疑问）
        question_count = user_message.count("?") + user_message.count("？")
        importance += min(question_count * 0.05, 0.1)
        
        return min(importance, 1.0)
    
    async def _retrieve_relevant_memories(
        self,
        query: str,
        user_id: str,
        session_id: str,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[MemoryItem]:
        """检索相关记忆"""
        if not self.memory_manager:
            return []
        
        try:
            relevant_memories = []
            
            # 默认检索所有类型的记忆
            if not memory_types:
                memory_types = [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.PROCEDURAL]
            
            # 从用户命名空间检索
            user_namespace = self._get_user_namespace(user_id)
            for memory_type in memory_types:
                memory_query = MemoryQuery(
                    query=query,
                    memory_type=memory_type,
                    limit=self.memory_retrieval_limit // len(memory_types),
                    min_importance=self.memory_importance_threshold
                )
                
                memories = await self.memory_manager.search_memories(user_namespace, memory_query)
                relevant_memories.extend(memories)
            
            # 从智能体命名空间检索
            agent_namespace = self._get_agent_namespace(session_id)
            for memory_type in memory_types:
                memory_query = MemoryQuery(
                    query=query,
                    memory_type=memory_type,
                    limit=self.memory_retrieval_limit // len(memory_types),
                    min_importance=self.memory_importance_threshold
                )
                
                memories = await self.memory_manager.search_memories(agent_namespace, memory_query)
                relevant_memories.extend(memories)
            
            # 按重要性排序并去重
            relevant_memories.sort(key=lambda x: x.importance, reverse=True)
            unique_memories = []
            seen_content = set()
            
            for memory in relevant_memories:
                if memory.content not in seen_content:
                    unique_memories.append(memory)
                    seen_content.add(memory.content)
                    
                if len(unique_memories) >= self.memory_retrieval_limit:
                    break
            
            logger.debug(f"检索到相关记忆: {len(unique_memories)} 条")
            return unique_memories
            
        except Exception as e:
            logger.error(f"检索相关记忆失败: {e}")
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
        request: ChatRequest,
        config: Optional[RunnableConfig] = None
    ) -> ChatResponse:
        """记忆增强的对话处理"""
        try:
            # 确保初始化
            if not self.initialized:
                await self.initialize()
            
            # 使用记忆增强消息
            enhanced_messages = await self._enhance_messages_with_memory(
                request.messages,
                request.user_id,
                request.session_id
            )
            
            # 创建增强的请求
            enhanced_request = ChatRequest(
                messages=enhanced_messages,
                user_id=request.user_id,
                session_id=request.session_id,
                stream=request.stream,
                metadata=request.metadata
            )
            
            # 调用父类的对话处理
            response = await super().chat(enhanced_request, config)
            
            # 存储对话记忆
            if response.message and enhanced_messages:
                last_user_message = None
                for msg in reversed(request.messages):  # 使用原始消息
                    if isinstance(msg, HumanMessage):
                        last_user_message = msg.content
                        break
                
                if last_user_message:
                    await self._store_conversation_memory(
                        user_message=last_user_message,
                        ai_response=response.message.content,
                        user_id=request.user_id,
                        session_id=request.session_id,
                        metadata=request.metadata
                    )
            
            return response
            
        except Exception as e:
            logger.error(f"记忆增强对话处理失败: {e}")
            # 降级到基础对话处理
            return await super().chat(request, config)
    
    async def stream_chat(
        self,
        request: ChatRequest,
        config: Optional[RunnableConfig] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """记忆增强的流式对话处理"""
        try:
            # 确保初始化
            if not self.initialized:
                await self.initialize()
            
            # 使用记忆增强消息
            enhanced_messages = await self._enhance_messages_with_memory(
                request.messages,
                request.user_id,
                request.session_id
            )
            
            # 创建增强的请求
            enhanced_request = ChatRequest(
                messages=enhanced_messages,
                user_id=request.user_id,
                session_id=request.session_id,
                stream=True,
                metadata=request.metadata
            )
            
            # 收集完整响应用于记忆存储
            full_response = ""
            
            # 流式处理
            async for chunk in super().stream_chat(enhanced_request, config):
                if chunk.content:
                    full_response += chunk.content
                yield chunk
            
            # 存储对话记忆
            if full_response:
                last_user_message = None
                for msg in reversed(request.messages):  # 使用原始消息
                    if isinstance(msg, HumanMessage):
                        last_user_message = msg.content
                        break
                
                if last_user_message:
                    await self._store_conversation_memory(
                        user_message=last_user_message,
                        ai_response=full_response,
                        user_id=request.user_id,
                        session_id=request.session_id,
                        metadata=request.metadata
                    )
            
        except Exception as e:
            logger.error(f"记忆增强流式对话处理失败: {e}")
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