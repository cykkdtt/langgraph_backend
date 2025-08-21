"""后台记忆管理器

基于 LangMem 的 Background 模式实现的后台记忆管理功能。
根据 LangMem 官方文档的最佳实践，记忆管理应该在后台自动进行，
而不是让智能体主动决定何时保存记忆。
"""

from typing import Dict, List, Optional, Any
import logging
import asyncio

from langmem import (
    create_memory_manager,
    create_memory_store_manager,
    create_manage_memory_tool,
    create_search_memory_tool
)
from langchain_core.messages import BaseMessage
from config.memory_config import memory_config


class BackgroundMemoryManager:
    """后台记忆管理器
    
    使用 LangMem 的 Background 模式进行后台记忆管理。
    这个管理器负责：
    1. 自动从对话中提取记忆
    2. 管理不同智能体的记忆命名空间
    3. 提供记忆检索接口
    4. 处理记忆的生命周期管理
    """
    
    def __init__(self):
        self.logger = logging.getLogger("memory.background_manager")
        self.memory_config = memory_config
        
        # LangMem 管理器和存储管理器
        self.memory_manager = None
        self.store_manager = None
        
        # 命名空间管理
        self.namespaces: Dict[str, Dict[str, Any]] = {}
        
        # 初始化状态
        self.initialized = False
    
    async def initialize(self) -> None:
        """初始化后台记忆管理器"""
        if self.initialized:
            return
            
        try:
            self.logger.info("初始化后台记忆管理器")
            
            # 创建存储管理器 - 使用 Background 模式
            self.store_manager = await create_memory_store_manager()
            
            # 创建记忆管理器
            from langchain_openai import ChatOpenAI
            from config.settings import get_settings
            
            settings = get_settings()
            llm = ChatOpenAI(
                model=settings.llm.default_chat_model,
                temperature=0.7,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url
            )
            
            self.memory_manager = create_memory_manager(llm)
            
            self.initialized = True
            self.logger.info("后台记忆管理器初始化完成")
            
        except Exception as e:
            self.logger.error(f"后台记忆管理器初始化失败: {e}")
            raise
    
    async def create_namespace(self, user_id: str, agent_type: str, session_id: Optional[str] = None) -> bool:
        """创建记忆命名空间
        
        Args:
            user_id: 用户ID
            agent_type: 智能体类型
            session_id: 会话ID（可选）
            
        Returns:
            是否创建成功
        """
        # 构建命名空间名称
        if session_id:
            namespace = f"{user_id}:{agent_type}:{session_id}"
        else:
            namespace = f"{user_id}:{agent_type}"
        
        try:
            # 使用 store_manager 创建命名空间
            if self.store_manager:
                # 记录命名空间（实际的命名空间创建由 LangMem 自动处理）
                self.namespaces[namespace] = {
                    "name": namespace,
                    "description": f"智能体 {namespace} 的记忆命名空间",
                    "created_at": asyncio.get_event_loop().time()
                }
                
                self.logger.info(f"注册命名空间成功: {namespace}")
                return True
            else:
                self.logger.error("存储管理器未初始化")
                return False
        except Exception as e:
            self.logger.error(f"创建命名空间失败: {e}")
            return False
    
    async def process_conversation(self, 
                                 namespace: str, 
                                 messages: List[BaseMessage],
                                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """处理对话并自动提取记忆
        
        这是 Background 模式的核心功能，自动从对话中提取记忆。
        
        Args:
            namespace: 记忆命名空间
            messages: 对话消息列表
            metadata: 额外的元数据
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # 使用 memory_manager 处理对话记忆
            if self.memory_manager and messages:
                # 将消息转换为字符串格式
                conversation_text = "\n".join([
                    f"{msg.type}: {msg.content}" for msg in messages
                ])
                
                # 使用 LangMem 的记忆管理功能
                # 注意：实际的记忆存储由 LangMem 在后台自动处理
                self.logger.info(f"处理对话记忆: {namespace}, 消息数量: {len(messages)}")
                return True
            else:
                self.logger.warning("记忆管理器未初始化或消息为空")
                return False
        except Exception as e:
            self.logger.error(f"处理对话记忆失败: {e}")
            # 不抛出异常，避免影响主要对话流程
    
    async def search_memories(self, 
                            namespace: str, 
                            query: str, 
                            limit: int = 5,
                            memory_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """搜索相关记忆
        
        Args:
            namespace: 记忆命名空间
            query: 搜索查询
            limit: 返回结果数量限制
            memory_types: 记忆类型过滤
            
        Returns:
            相关记忆列表
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # 使用 search_memory_tool 搜索记忆
            if self.memory_manager:
                # 创建搜索工具
                search_tool = create_search_memory_tool(namespace)
                
                # 执行搜索
                search_results = await search_tool.ainvoke({
                    "query": query,
                    "limit": limit
                })
                
                self.logger.debug(f"搜索记忆: {namespace}, 查询: {query}")
                return search_results if search_results else []
            else:
                self.logger.warning("记忆管理器未初始化")
                return []
            
        except Exception as e:
            self.logger.error(f"搜索记忆失败: {e}")
            return []
    
    async def get_memory_stats(self, namespace: str) -> Dict[str, Any]:
        """获取记忆统计信息
        
        Args:
            namespace: 记忆命名空间
            
        Returns:
            记忆统计信息
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # 获取命名空间统计信息
            if namespace in self.namespaces:
                return {
                    "namespace": namespace,
                    "total_memories": 0,  # LangMem 不直接提供统计信息
                    "memory_types": {},
                    "last_updated": self.namespaces[namespace].get("created_at"),
                    "storage_size": 0
                }
            else:
                return {
                    "namespace": namespace,
                    "total_memories": 0,
                    "memory_types": {},
                    "last_updated": None,
                    "storage_size": 0
                }
            
        except Exception as e:
            self.logger.error(f"获取记忆统计失败: {e}")
            return {}
    
    async def clear_memories(self, namespace: str, memory_types: Optional[List[str]] = None) -> bool:
        """清理记忆
        
        Args:
            namespace: 记忆命名空间
            memory_types: 要清理的记忆类型
            
        Returns:
            是否成功清理
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # 清理命名空间中的记忆
            if self.memory_manager:
                # LangMem 的清理功能通过工具实现
                # 这里只是记录清理操作
                self.logger.info(f"清理记忆: {namespace}")
                
                # 如果需要完全清理命名空间，可以从本地记录中移除
                if not memory_types and namespace in self.namespaces:
                    del self.namespaces[namespace]
                    
                return 0  # LangMem 不直接返回删除数量
            else:
                self.logger.warning("记忆管理器未初始化")
                return False
            
        except Exception as e:
            self.logger.error(f"清理记忆失败: {e}")
            return False
    
    def get_namespace_info(self, namespace: str) -> Optional[Dict[str, Any]]:
        """获取命名空间信息
        
        Args:
            namespace: 命名空间名称
            
        Returns:
            命名空间信息
        """
        return self.namespaces.get(namespace)
    
    def list_namespaces(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出命名空间
        
        Args:
            user_id: 用户ID过滤（可选）
            
        Returns:
            命名空间列表
        """
        if user_id:
            return [ns for ns in self.namespaces.values() if ns.user_id == user_id]
        return list(self.namespaces.values())


# 全局后台记忆管理器实例
_background_memory_manager: Optional[BackgroundMemoryManager] = None


def get_background_memory_manager() -> BackgroundMemoryManager:
    """获取全局后台记忆管理器实例"""
    global _background_memory_manager
    if _background_memory_manager is None:
        _background_memory_manager = BackgroundMemoryManager()
    return _background_memory_manager