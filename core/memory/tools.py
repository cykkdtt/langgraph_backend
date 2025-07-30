"""
LangMem 工具工厂

提供记忆管理和搜索工具的创建功能，支持不同类型的记忆工具
（语义记忆、情节记忆、程序记忆）的动态创建。
"""

import logging
from typing import List, Dict, Any, Optional
from langmem import create_manage_memory_tool, create_search_memory_tool
from core.memory.store_manager import memory_store_manager

logger = logging.getLogger(__name__)


class MemoryToolsFactory:
    """记忆工具工厂类"""
    
    @staticmethod
    async def create_memory_tools(namespace: str) -> List[Any]:
        """为指定命名空间创建记忆工具"""
        try:
            # 确保存储管理器已初始化
            store_manager = await memory_store_manager.initialize()
            if not memory_store_manager.store:
                raise RuntimeError("记忆存储未初始化")
            
            # 创建记忆管理工具
            manage_tool = create_manage_memory_tool(
                store=memory_store_manager.store,
                namespace=namespace
            )
            
            # 创建记忆搜索工具
            search_tool = create_search_memory_tool(
                store=memory_store_manager.store,
                namespace=namespace
            )
            
            logger.info(f"为命名空间 '{namespace}' 创建记忆工具成功")
            return [manage_tool, search_tool]
            
        except Exception as e:
            logger.error(f"创建记忆工具失败: {e}")
            raise
    
    @staticmethod
    async def create_semantic_memory_tool(namespace: str):
        """创建语义记忆工具"""
        try:
            if not memory_store_manager.store:
                await memory_store_manager.initialize()
            
            semantic_namespace = f"{namespace}_semantic"
            return create_manage_memory_tool(
                store=memory_store_manager.store,
                namespace=semantic_namespace,
                memory_type="semantic"
            )
        except Exception as e:
            logger.error(f"创建语义记忆工具失败: {e}")
            raise
    
    @staticmethod
    async def create_episodic_memory_tool(namespace: str):
        """创建情节记忆工具"""
        try:
            if not memory_store_manager.store:
                await memory_store_manager.initialize()
            
            episodic_namespace = f"{namespace}_episodic"
            return create_manage_memory_tool(
                store=memory_store_manager.store,
                namespace=episodic_namespace,
                memory_type="episodic"
            )
        except Exception as e:
            logger.error(f"创建情节记忆工具失败: {e}")
            raise
    
    @staticmethod
    async def create_procedural_memory_tool(namespace: str):
        """创建程序记忆工具"""
        try:
            if not memory_store_manager.store:
                await memory_store_manager.initialize()
            
            procedural_namespace = f"{namespace}_procedural"
            return create_manage_memory_tool(
                store=memory_store_manager.store,
                namespace=procedural_namespace,
                memory_type="procedural"
            )
        except Exception as e:
            logger.error(f"创建程序记忆工具失败: {e}")
            raise
    
    @staticmethod
    async def create_search_tool_for_type(namespace: str, memory_type: str):
        """为特定记忆类型创建搜索工具"""
        try:
            if not memory_store_manager.store:
                await memory_store_manager.initialize()
            
            typed_namespace = f"{namespace}_{memory_type}"
            return create_search_memory_tool(
                store=memory_store_manager.store,
                namespace=typed_namespace
            )
        except Exception as e:
            logger.error(f"创建 {memory_type} 记忆搜索工具失败: {e}")
            raise
    
    @staticmethod
    async def create_user_memory_tools(user_id: str) -> Dict[str, Any]:
        """为用户创建完整的记忆工具集"""
        try:
            user_namespace = memory_store_manager.get_user_namespace(user_id)
            
            tools = {
                "manage": await MemoryToolsFactory.create_memory_tools(user_namespace),
                "semantic": await MemoryToolsFactory.create_semantic_memory_tool(user_namespace),
                "episodic": await MemoryToolsFactory.create_episodic_memory_tool(user_namespace),
                "procedural": await MemoryToolsFactory.create_procedural_memory_tool(user_namespace),
                "search_semantic": await MemoryToolsFactory.create_search_tool_for_type(
                    user_namespace, "semantic"
                ),
                "search_episodic": await MemoryToolsFactory.create_search_tool_for_type(
                    user_namespace, "episodic"
                ),
                "search_procedural": await MemoryToolsFactory.create_search_tool_for_type(
                    user_namespace, "procedural"
                )
            }
            
            logger.info(f"为用户 '{user_id}' 创建完整记忆工具集成功")
            return tools
            
        except Exception as e:
            logger.error(f"为用户 '{user_id}' 创建记忆工具集失败: {e}")
            raise
    
    @staticmethod
    async def create_agent_memory_tools(agent_id: str, session_id: str) -> Dict[str, Any]:
        """为智能体会话创建记忆工具集"""
        try:
            agent_namespace = memory_store_manager.get_namespace(agent_id, session_id)
            
            tools = {
                "manage": await MemoryToolsFactory.create_memory_tools(agent_namespace),
                "semantic": await MemoryToolsFactory.create_semantic_memory_tool(agent_namespace),
                "episodic": await MemoryToolsFactory.create_episodic_memory_tool(agent_namespace),
                "procedural": await MemoryToolsFactory.create_procedural_memory_tool(agent_namespace),
                "search_semantic": await MemoryToolsFactory.create_search_tool_for_type(
                    agent_namespace, "semantic"
                ),
                "search_episodic": await MemoryToolsFactory.create_search_tool_for_type(
                    agent_namespace, "episodic"
                ),
                "search_procedural": await MemoryToolsFactory.create_search_tool_for_type(
                    agent_namespace, "procedural"
                )
            }
            
            logger.info(f"为智能体 '{agent_id}' 会话 '{session_id}' 创建记忆工具集成功")
            return tools
            
        except Exception as e:
            logger.error(f"为智能体 '{agent_id}' 会话 '{session_id}' 创建记忆工具集失败: {e}")
            raise


# 便捷函数
async def get_memory_tools(namespace: str) -> List[Any]:
    """获取基础记忆工具"""
    return await MemoryToolsFactory.create_memory_tools(namespace)


async def get_user_memory_tools(user_id: str) -> Dict[str, Any]:
    """获取用户记忆工具集"""
    return await MemoryToolsFactory.create_user_memory_tools(user_id)


async def get_agent_memory_tools(agent_id: str, session_id: str) -> Dict[str, Any]:
    """获取智能体记忆工具集"""
    return await MemoryToolsFactory.create_agent_memory_tools(agent_id, session_id)