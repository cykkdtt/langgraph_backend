"""
LangMem 存储管理器

提供记忆存储的统一管理接口，支持PostgreSQL和内存存储，
负责存储初始化、连接管理和命名空间生成。
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_community.embeddings import DashScopeEmbeddings
from langmem import create_memory_store_manager
from config.memory_config import memory_config

logger = logging.getLogger(__name__)


def create_embeddings():
    """创建嵌入模型实例"""
    embedding_model = memory_config.embedding_model
    
    if embedding_model.startswith("openai:text-embedding-v"):
        # 使用阿里云DashScope嵌入模型
        model_name = embedding_model.split(":", 1)[1]  # 提取模型名称
        return DashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
    else:
        # 其他嵌入模型的处理逻辑可以在这里添加
        raise ValueError(f"不支持的嵌入模型: {embedding_model}")


class MemoryStoreManager:
    """记忆存储管理器"""
    
    def __init__(self):
        self.store: Optional[BaseStore] = None
        self.memory_manager = None
        self._initialized = False
        self._store_context = None
    
    async def initialize(self) -> None:
        """初始化存储系统"""
        if self._initialized:
            return
        
        try:
            # 创建嵌入模型实例
            embeddings = create_embeddings()
            
            # 创建存储实例
            if memory_config.store_type == "postgres":
                # 使用from_conn_string创建AsyncPostgresStore上下文管理器
                self._store_context = AsyncPostgresStore.from_conn_string(
                    memory_config.postgres_url,
                    index={
                        "embed": embeddings,  # 使用DashScopeEmbeddings实例
                        "dims": memory_config.embedding_dims,
                        "fields": ["$"]  # 对所有字段进行向量化
                    }
                )
                self.store = await self._store_context.__aenter__()
                logger.info("使用 PostgreSQL 存储")
            else:
                self.store = InMemoryStore(
                    index={
                        "embed": embeddings,  # 使用DashScopeEmbeddings实例
                        "dims": memory_config.embedding_dims,
                        "fields": ["$"]  # 对所有字段进行向量化
                    }
                )
                self._store_context = None
                logger.info("使用内存存储")
            
            # 设置存储
            if hasattr(self.store, 'setup'):
                await self.store.setup()
            
            # 创建记忆存储管理器
            self.memory_manager = create_memory_store_manager(
                memory_config.memory_model,
                store=self.store,
                namespace=(memory_config.namespace_prefix, "{langgraph_user_id}")
            )
            
            self._initialized = True
            logger.info("记忆存储管理器初始化完成")
            
        except Exception as e:
            logger.error(f"记忆存储管理器初始化失败: {e}")
            raise
    
    async def cleanup(self) -> None:
        """清理资源"""
        try:
            # 如果是PostgresStore，需要退出上下文管理器
            if self._store_context:
                await self._store_context.__aexit__(None, None, None)
                logger.info("PostgreSQL存储上下文已关闭")
            elif self.store and hasattr(self.store, 'close'):
                await self.store.close()
                logger.info("记忆存储连接已关闭")
        except Exception as e:
            logger.error(f"关闭存储连接时出错: {e}")
        
        self._initialized = False
    
    def get_namespace(self, agent_id: str, session_id: str) -> str:
        """生成命名空间"""
        return f"agent_{agent_id}_session_{session_id}"
    
    def get_typed_namespace(self, agent_id: str, session_id: str, memory_type: str) -> str:
        """生成带类型的命名空间"""
        base_namespace = self.get_namespace(agent_id, session_id)
        return f"{base_namespace}_{memory_type}"
    
    def get_user_namespace(self, user_id: str) -> str:
        """生成用户命名空间"""
        return f"user_{user_id}"
    
    def get_agent_namespace(self, agent_type: str) -> str:
        """生成智能体类型命名空间"""
        return f"agent_type_{agent_type}"
    
    def get_organization_namespace(self, org_id: str, user_id: str) -> str:
        """生成组织命名空间"""
        return f"org_{org_id}_user_{user_id}"
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        if not self._initialized:
            return {
                "status": "unhealthy",
                "message": "存储管理器未初始化"
            }
        
        try:
            # 简单的存储测试
            test_namespace = "health_check"
            test_key = "test"
            test_value = {"test": "data"}
            
            # 写入测试数据
            await self.store.aput(test_namespace, test_key, test_value)
            
            # 读取测试数据
            result = await self.store.aget(test_namespace, test_key)
            
            # 清理测试数据
            await self.store.adelete(test_namespace, test_key)
            
            if result == test_value:
                return {
                    "status": "healthy",
                    "store_type": memory_config.store_type,
                    "initialized": self._initialized
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "存储读写测试失败"
                }
                
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "message": f"健康检查异常: {str(e)}"
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        if not self._initialized:
            return {"error": "存储管理器未初始化"}
        
        try:
            # 这里可以根据具体存储类型实现统计功能
            stats = {
                "store_type": memory_config.store_type,
                "initialized": self._initialized,
                "config": {
                    "max_memories_per_namespace": memory_config.max_memories_per_namespace,
                    "auto_consolidate": memory_config.auto_consolidate,
                    "consolidate_threshold": memory_config.consolidate_threshold
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": f"获取统计信息失败: {str(e)}"}


# 全局实例
memory_store_manager = MemoryStoreManager()


async def get_memory_store_manager() -> MemoryStoreManager:
    """获取记忆存储管理器实例"""
    if not memory_store_manager._initialized:
        await memory_store_manager.initialize()
    return memory_store_manager


# 别名函数，保持向后兼容
async def get_store_manager() -> MemoryStoreManager:
    """获取记忆存储管理器实例（别名）"""
    return await get_memory_store_manager()