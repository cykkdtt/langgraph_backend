"""
多智能体LangGraph项目 - 长期记忆管理器

本模块提供LangMem长期记忆功能，支持：
- 语义记忆（事实和概念）
- 情节记忆（事件和经历）
- 程序记忆（技能和过程）
- 命名空间隔离
- 多种存储后端
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union, AsyncContextManager
from datetime import datetime
from contextlib import asynccontextmanager
from enum import Enum

from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from config.settings import get_settings


class MemoryType(str, Enum):
    """记忆类型"""
    SEMANTIC = "semantic"      # 语义记忆：事实、概念、知识
    EPISODIC = "episodic"      # 情节记忆：事件、经历、对话
    PROCEDURAL = "procedural"  # 程序记忆：技能、过程、方法


class MemoryScope(str, Enum):
    """记忆范围"""
    USER = "user"              # 用户级别
    AGENT = "agent"            # 智能体级别
    ORGANIZATION = "org"       # 组织级别
    GLOBAL = "global"          # 全局级别


class MemoryItem(BaseModel):
    """记忆项"""
    id: str = Field(description="记忆ID")
    content: str = Field(description="记忆内容")
    memory_type: MemoryType = Field(description="记忆类型")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性评分")
    access_count: int = Field(default=0, description="访问次数")
    last_accessed: Optional[datetime] = Field(default=None, description="最后访问时间")


class MemoryQuery(BaseModel):
    """记忆查询"""
    query: str = Field(description="查询内容")
    memory_type: Optional[MemoryType] = Field(default=None, description="记忆类型过滤")
    limit: int = Field(default=10, description="返回数量限制")
    min_importance: float = Field(default=0.0, description="最小重要性阈值")
    time_range: Optional[tuple] = Field(default=None, description="时间范围")


class MemoryNamespace:
    """记忆命名空间
    
    用于隔离不同用户、智能体、组织的记忆。
    """
    
    def __init__(
        self, 
        scope: MemoryScope, 
        identifier: str, 
        sub_namespace: Optional[str] = None
    ):
        """初始化命名空间
        
        Args:
            scope: 记忆范围
            identifier: 标识符（用户ID、智能体ID等）
            sub_namespace: 子命名空间（可选）
        """
        self.scope = scope
        self.identifier = identifier
        self.sub_namespace = sub_namespace
    
    def to_tuple(self) -> tuple:
        """转换为元组格式"""
        if self.sub_namespace:
            return (self.scope.value, self.identifier, self.sub_namespace)
        return (self.scope.value, self.identifier)
    
    def __str__(self) -> str:
        """字符串表示"""
        if self.sub_namespace:
            return f"{self.scope.value}:{self.identifier}:{self.sub_namespace}"
        return f"{self.scope.value}:{self.identifier}"


class LangMemManager:
    """LangMem长期记忆管理器
    
    提供统一的长期记忆管理接口，支持多种存储后端和记忆类型。
    """
    
    def __init__(self, storage_type: str = "postgres"):
        """初始化记忆管理器
        
        Args:
            storage_type: 存储类型 ("postgres", "memory")
        """
        self.storage_type = storage_type
        self.settings = get_settings()
        self.logger = logging.getLogger("memory.manager")
        
        self._store: Optional[BaseStore] = None
        self._context_manager: Optional[AsyncContextManager] = None
        self._is_initialized = False
    
    async def initialize(self) -> BaseStore:
        """初始化记忆存储器
        
        Returns:
            BaseStore: 记忆存储器实例
        """
        if self._is_initialized and self._store:
            return self._store
        
        try:
            self.logger.info(f"初始化记忆存储器: {self.storage_type}")
            
            if self.storage_type == "postgres":
                self._context_manager = AsyncPostgresStore.from_conn_string(
                    self.settings.database.postgres_url
                )
                self._store = await self._context_manager.__aenter__()
                
            elif self.storage_type == "memory":
                self._store = InMemoryStore()
                
            else:
                raise ValueError(f"不支持的存储类型: {self.storage_type}")
            
            # 设置表结构（如果需要）
            if hasattr(self._store, 'setup'):
                await self._store.setup()
                self.logger.info("记忆表结构初始化完成")
            
            self._is_initialized = True
            self.logger.info(f"记忆存储器初始化成功: {type(self._store).__name__}")
            
            return self._store
            
        except Exception as e:
            self.logger.error(f"记忆存储器初始化失败: {e}")
            # 降级到内存存储
            if self.storage_type != "memory":
                self.logger.info("降级使用内存存储")
                self.storage_type = "memory"
                self._store = InMemoryStore()
                self._is_initialized = True
                return self._store
            raise
    
    async def get_store(self) -> BaseStore:
        """获取记忆存储器实例
        
        Returns:
            BaseStore: 记忆存储器实例
        """
        if not self._is_initialized:
            await self.initialize()
        return self._store
    
    async def store_memory(
        self, 
        namespace: MemoryNamespace,
        memory_item: MemoryItem
    ) -> str:
        """存储记忆
        
        Args:
            namespace: 记忆命名空间
            memory_item: 记忆项
            
        Returns:
            str: 记忆ID
        """
        store = await self.get_store()
        
        # 准备存储数据
        memory_data = {
            "content": memory_item.content,
            "memory_type": memory_item.memory_type.value,
            "metadata": memory_item.metadata,
            "timestamp": memory_item.timestamp.isoformat(),
            "importance": memory_item.importance,
            "access_count": memory_item.access_count,
            "last_accessed": memory_item.last_accessed.isoformat() if memory_item.last_accessed else None
        }
        
        # 存储记忆
        await store.aput(namespace.to_tuple(), memory_item.id, memory_data)
        
        self.logger.info(f"存储记忆成功: namespace={namespace}, id={memory_item.id}")
        return memory_item.id
    
    async def retrieve_memory(
        self, 
        namespace: MemoryNamespace,
        memory_id: str
    ) -> Optional[MemoryItem]:
        """检索特定记忆
        
        Args:
            namespace: 记忆命名空间
            memory_id: 记忆ID
            
        Returns:
            MemoryItem: 记忆项，如果不存在则返回None
        """
        store = await self.get_store()
        
        memory_data = await store.aget(namespace.to_tuple(), memory_id)
        if not memory_data:
            return None
        
        # 更新访问统计
        memory_data["access_count"] = memory_data.get("access_count", 0) + 1
        memory_data["last_accessed"] = datetime.utcnow().isoformat()
        await store.aput(namespace.to_tuple(), memory_id, memory_data)
        
        # 构造记忆项
        return MemoryItem(
            id=memory_id,
            content=memory_data["content"],
            memory_type=MemoryType(memory_data["memory_type"]),
            metadata=memory_data.get("metadata", {}),
            timestamp=datetime.fromisoformat(memory_data["timestamp"]),
            importance=memory_data.get("importance", 0.5),
            access_count=memory_data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(memory_data["last_accessed"]) if memory_data.get("last_accessed") else None
        )
    
    async def search_memories(
        self, 
        namespace: MemoryNamespace,
        query: MemoryQuery
    ) -> List[MemoryItem]:
        """搜索记忆
        
        Args:
            namespace: 记忆命名空间
            query: 查询条件
            
        Returns:
            List[MemoryItem]: 匹配的记忆项列表
        """
        store = await self.get_store()
        
        memories = []
        async for key, memory_data in store.asearch(namespace.to_tuple()):
            # 构造记忆项
            memory_item = MemoryItem(
                id=key,
                content=memory_data["content"],
                memory_type=MemoryType(memory_data["memory_type"]),
                metadata=memory_data.get("metadata", {}),
                timestamp=datetime.fromisoformat(memory_data["timestamp"]),
                importance=memory_data.get("importance", 0.5),
                access_count=memory_data.get("access_count", 0),
                last_accessed=datetime.fromisoformat(memory_data["last_accessed"]) if memory_data.get("last_accessed") else None
            )
            
            # 应用过滤条件
            if query.memory_type and memory_item.memory_type != query.memory_type:
                continue
            
            if memory_item.importance < query.min_importance:
                continue
            
            # 简单的文本匹配（实际应用中可以使用更复杂的语义搜索）
            if query.query.lower() in memory_item.content.lower():
                memories.append(memory_item)
        
        # 按重要性和时间排序
        memories.sort(
            key=lambda x: (x.importance, x.timestamp), 
            reverse=True
        )
        
        return memories[:query.limit]
    
    async def update_memory(
        self, 
        namespace: MemoryNamespace,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """更新记忆
        
        Args:
            namespace: 记忆命名空间
            memory_id: 记忆ID
            updates: 更新数据
            
        Returns:
            bool: 是否更新成功
        """
        try:
            store = await self.get_store()
            
            # 获取现有记忆
            memory_data = await store.aget(namespace.to_tuple(), memory_id)
            if not memory_data:
                return False
            
            # 应用更新
            memory_data.update(updates)
            memory_data["last_accessed"] = datetime.utcnow().isoformat()
            
            # 保存更新
            await store.aput(namespace.to_tuple(), memory_id, memory_data)
            
            self.logger.info(f"更新记忆成功: namespace={namespace}, id={memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"更新记忆失败: {e}")
            return False
    
    async def delete_memory(
        self, 
        namespace: MemoryNamespace,
        memory_id: str
    ) -> bool:
        """删除记忆
        
        Args:
            namespace: 记忆命名空间
            memory_id: 记忆ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            store = await self.get_store()
            await store.adelete(namespace.to_tuple(), memory_id)
            
            self.logger.info(f"删除记忆成功: namespace={namespace}, id={memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"删除记忆失败: {e}")
            return False
    
    async def list_memories(
        self, 
        namespace: MemoryNamespace,
        limit: int = 50
    ) -> List[MemoryItem]:
        """列出命名空间下的所有记忆
        
        Args:
            namespace: 记忆命名空间
            limit: 返回数量限制
            
        Returns:
            List[MemoryItem]: 记忆项列表
        """
        store = await self.get_store()
        
        memories = []
        count = 0
        async for key, memory_data in store.asearch(namespace.to_tuple()):
            if count >= limit:
                break
            
            memory_item = MemoryItem(
                id=key,
                content=memory_data["content"],
                memory_type=MemoryType(memory_data["memory_type"]),
                metadata=memory_data.get("metadata", {}),
                timestamp=datetime.fromisoformat(memory_data["timestamp"]),
                importance=memory_data.get("importance", 0.5),
                access_count=memory_data.get("access_count", 0),
                last_accessed=datetime.fromisoformat(memory_data["last_accessed"]) if memory_data.get("last_accessed") else None
            )
            
            memories.append(memory_item)
            count += 1
        
        # 按时间排序
        memories.sort(key=lambda x: x.timestamp, reverse=True)
        return memories
    
    async def get_memory_stats(
        self, 
        namespace: MemoryNamespace
    ) -> Dict[str, Any]:
        """获取记忆统计信息
        
        Args:
            namespace: 记忆命名空间
            
        Returns:
            Dict: 统计信息
        """
        memories = await self.list_memories(namespace, limit=1000)
        
        stats = {
            "total_count": len(memories),
            "by_type": {},
            "avg_importance": 0.0,
            "total_access_count": 0,
            "most_accessed": None,
            "most_important": None
        }
        
        if not memories:
            return stats
        
        # 按类型统计
        for memory in memories:
            memory_type = memory.memory_type.value
            stats["by_type"][memory_type] = stats["by_type"].get(memory_type, 0) + 1
            stats["total_access_count"] += memory.access_count
        
        # 平均重要性
        stats["avg_importance"] = sum(m.importance for m in memories) / len(memories)
        
        # 最常访问的记忆
        most_accessed = max(memories, key=lambda x: x.access_count)
        stats["most_accessed"] = {
            "id": most_accessed.id,
            "content": most_accessed.content[:100] + "..." if len(most_accessed.content) > 100 else most_accessed.content,
            "access_count": most_accessed.access_count
        }
        
        # 最重要的记忆
        most_important = max(memories, key=lambda x: x.importance)
        stats["most_important"] = {
            "id": most_important.id,
            "content": most_important.content[:100] + "..." if len(most_important.content) > 100 else most_important.content,
            "importance": most_important.importance
        }
        
        return stats
    
    async def cleanup_old_memories(
        self, 
        namespace: MemoryNamespace,
        days: int = 90,
        min_importance: float = 0.1
    ) -> int:
        """清理旧的低重要性记忆
        
        Args:
            namespace: 记忆命名空间
            days: 保留天数
            min_importance: 最小重要性阈值
            
        Returns:
            int: 清理的记忆数量
        """
        try:
            cutoff_date = datetime.utcnow().timestamp() - (days * 24 * 3600)
            memories = await self.list_memories(namespace, limit=1000)
            
            deleted_count = 0
            for memory in memories:
                # 清理条件：时间久远且重要性低
                if (memory.timestamp.timestamp() < cutoff_date and 
                    memory.importance < min_importance):
                    
                    if await self.delete_memory(namespace, memory.id):
                        deleted_count += 1
            
            self.logger.info(f"清理记忆完成: namespace={namespace}, 删除数量={deleted_count}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"清理记忆失败: {e}")
            return 0
    
    async def cleanup(self):
        """清理资源"""
        try:
            if self._context_manager:
                await self._context_manager.__aexit__(None, None, None)
                self.logger.info("记忆存储器资源清理完成")
        except Exception as e:
            self.logger.error(f"记忆存储器资源清理失败: {e}")
        finally:
            self._store = None
            self._context_manager = None
            self._is_initialized = False


# 全局记忆管理器实例
_memory_manager: Optional[LangMemManager] = None


def get_memory_manager(storage_type: str = "postgres") -> LangMemManager:
    """获取全局记忆管理器实例
    
    Args:
        storage_type: 存储类型
        
    Returns:
        LangMemManager: 记忆管理器实例
    """
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = LangMemManager(storage_type)
    
    return _memory_manager


@asynccontextmanager
async def memory_manager_context(storage_type: str = "postgres"):
    """记忆管理器上下文管理器
    
    Args:
        storage_type: 存储类型
        
    Yields:
        LangMemManager: 记忆管理器实例
    """
    manager = LangMemManager(storage_type)
    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.cleanup()