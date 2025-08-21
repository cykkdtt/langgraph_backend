"""
多智能体LangGraph项目 - 检查点管理器

本模块提供统一的检查点管理功能，支持：
- PostgreSQL持久化存储
- SQLite本地存储
- 内存存储
- 检查点的保存、加载、列表和清理
"""

import os
import logging
from typing import Dict, Any, Optional, List, AsyncContextManager
from datetime import datetime
from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, Field
from psycopg_pool import AsyncConnectionPool

from config.settings import get_settings


class CheckpointMetadata(BaseModel):
    """检查点元数据"""
    checkpoint_id: str = Field(description="检查点ID")
    thread_id: str = Field(description="线程ID")
    timestamp: datetime = Field(description="创建时间")
    agent_type: Optional[str] = Field(default=None, description="智能体类型")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class CheckpointInfo(BaseModel):
    """检查点信息"""
    config: Dict[str, Any] = Field(description="检查点配置")
    state: Dict[str, Any] = Field(description="检查点状态")
    metadata: CheckpointMetadata = Field(description="检查点元数据")


class CheckpointManager:
    """检查点管理器
    
    提供统一的检查点管理接口，支持多种存储后端。
    """
    
    def __init__(self, storage_type: str = "postgres"):
        """初始化检查点管理器
        
        Args:
            storage_type: 存储类型 ("postgres", "memory")
        """
        self.storage_type = storage_type
        self.settings = get_settings()
        self.logger = logging.getLogger("checkpoint.manager")
        
        self._checkpointer: Optional[BaseCheckpointSaver] = None
        self._checkpointer_context = None
        self._context_manager: Optional[AsyncContextManager] = None
        self._is_initialized = False
    
    async def initialize(self) -> BaseCheckpointSaver:
        """初始化检查点存储器
        
        Returns:
            BaseCheckpointSaver: 检查点存储器实例
        """
        if self._is_initialized and self._checkpointer:
            return self._checkpointer
        
        try:
            self.logger.info(f"初始化检查点存储器: {self.storage_type}")
            
            if self.storage_type == "postgres":
                try:
                    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                    
                    postgres_uri = self.settings.database.async_url
                    self.logger.info(f"连接PostgreSQL数据库: {postgres_uri.split('@')[-1] if '@' in postgres_uri else 'localhost'}")
                    
                    # 使用 from_conn_string 方法创建上下文管理器
                    self._context_manager = AsyncPostgresSaver.from_conn_string(postgres_uri)
                    self._checkpointer = await self._context_manager.__aenter__()
                    
                    self.logger.info(f"检查点存储器初始化成功: {type(self._checkpointer).__name__}")
                    
                except Exception as e:
                    self.logger.error(f"PostgreSQL检查点存储器初始化失败: {e}")
                    self.logger.warning("降级使用内存存储")
                    
                    # 降级到内存存储
                    from langgraph.checkpoint.memory import MemorySaver
                    self._checkpointer = MemorySaver()
                    self.logger.info(f"检查点存储器初始化成功: {type(self._checkpointer).__name__}")
                
            elif self.storage_type == "sqlite":
                # 使用SQLite存储
                from langgraph.checkpoint.sqlite import SqliteSaver
                
                sqlite_path = self.settings.database.sqlite_path or "./checkpoints.db"
                self._checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{sqlite_path}")
                
                self.logger.info(f"检查点存储器初始化成功: {type(self._checkpointer).__name__}")
                
            else:
                # 使用内存存储
                from langgraph.checkpoint.memory import MemorySaver
                self._checkpointer = MemorySaver()
                
                self.logger.info(f"检查点存储器初始化成功: {type(self._checkpointer).__name__}")
            
            self._is_initialized = True
            self.logger.info(f"检查点存储器初始化成功: {type(self._checkpointer).__name__}")
            
            return self._checkpointer
            
        except Exception as e:
            self.logger.error(f"检查点存储器初始化失败: {e}")
            # 降级到内存存储
            if self.storage_type != "memory":
                self.logger.warning("PostgreSQL连接失败，降级使用内存存储")
                self.storage_type = "memory"
                self._checkpointer = MemorySaver()
                self._is_initialized = True
                return self._checkpointer
            raise
    
    async def get_checkpointer(self) -> BaseCheckpointSaver:
        """获取检查点存储器实例
        
        Returns:
            BaseCheckpointSaver: 检查点存储器实例
        """
        if not self._is_initialized:
            await self.initialize()
        return self._checkpointer
    
    async def save_checkpoint(
        self, 
        config: Dict[str, Any], 
        state: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """保存检查点
        
        Args:
            config: 检查点配置
            state: 状态数据
            metadata: 元数据
            
        Returns:
            str: 检查点ID
        """
        checkpointer = await self.get_checkpointer()
        
        # 准备检查点数据 - 按照LangGraph标准格式
        import uuid
        from datetime import datetime
        
        checkpoint_id = str(uuid.uuid4())
        
        # 确保config包含必需的参数
        if "configurable" not in config:
            config["configurable"] = {}
        
        # 添加必需的checkpoint_ns参数
        if "checkpoint_ns" not in config["configurable"]:
            config["configurable"]["checkpoint_ns"] = ""
        
        checkpoint = {
            "v": 1,  # 版本号
            "ts": datetime.utcnow().isoformat(),  # 时间戳
            "id": checkpoint_id,  # 检查点ID
            "channel_values": state,  # 状态数据
            "channel_versions": {},  # 通道版本
            "versions_seen": {}  # 已见版本
        }
        
        # 准备元数据
        checkpoint_metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        # 保存检查点 - 使用正确的参数格式
        saved_config = await checkpointer.aput(
            config, 
            checkpoint, 
            checkpoint_metadata, 
            {}  # writes (待写入数据)
        )
        
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        self.logger.info(f"保存检查点成功: thread_id={thread_id}, checkpoint_id={checkpoint_id}")
        
        return checkpoint_id
    
    async def load_checkpoint(
        self, 
        config: Dict[str, Any]
    ) -> Optional[CheckpointInfo]:
        """加载检查点
        
        Args:
            config: 检查点配置
            
        Returns:
            CheckpointInfo: 检查点信息，如果不存在则返回None
        """
        checkpointer = await self.get_checkpointer()
        
        # 使用正确的方法获取检查点元组
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if not checkpoint_tuple:
            return None
        
        checkpoint = checkpoint_tuple.checkpoint
        checkpoint_config = checkpoint_tuple.config
        checkpoint_metadata = checkpoint_tuple.metadata or {}
        
        # 解析元数据
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint.get("id", "unknown"),
            thread_id=config.get("configurable", {}).get("thread_id", "unknown"),
            timestamp=datetime.fromisoformat(
                checkpoint.get("ts", datetime.utcnow().isoformat())
            ),
            agent_type=checkpoint_metadata.get("agent_type"),
            user_id=checkpoint_metadata.get("user_id"),
            metadata=checkpoint_metadata
        )
        
        return CheckpointInfo(
            config=checkpoint_config,
            state=checkpoint.get("channel_values", {}),
            metadata=metadata
        )
    
    async def list_checkpoints(
        self, 
        config: Dict[str, Any], 
        limit: int = 10,
        before: Optional[str] = None
    ) -> List[CheckpointInfo]:
        """列出检查点历史
        
        Args:
            config: 检查点配置
            limit: 返回数量限制
            before: 在指定检查点之前的记录
            
        Returns:
            List[CheckpointInfo]: 检查点信息列表
        """
        checkpointer = await self.get_checkpointer()
        
        checkpoints = []
        # 使用正确的alist方法，传递before参数
        list_config = {**config}
        if before:
            list_config["before"] = before
            
        async for checkpoint_tuple in checkpointer.alist(
            list_config, 
            limit=limit
        ):
            checkpoint = checkpoint_tuple.checkpoint
            checkpoint_config = checkpoint_tuple.config
            checkpoint_metadata = checkpoint_tuple.metadata or {}
            
            # 解析元数据
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint.get("id", "unknown"),
                thread_id=config.get("configurable", {}).get("thread_id", "unknown"),
                timestamp=datetime.fromisoformat(
                    checkpoint.get("ts", datetime.utcnow().isoformat())
                ),
                agent_type=checkpoint_metadata.get("agent_type"),
                user_id=checkpoint_metadata.get("user_id"),
                metadata=checkpoint_metadata
            )
            
            checkpoints.append(CheckpointInfo(
                config=checkpoint_config,
                state=checkpoint.get("channel_values", {}),
                metadata=metadata
            ))
        
        return checkpoints
    
    async def delete_checkpoint(self, config: Dict[str, Any]) -> bool:
        """删除检查点
        
        Args:
            config: 检查点配置
            
        Returns:
            bool: 是否删除成功
        """
        try:
            checkpointer = await self.get_checkpointer()
            
            # 检查是否支持删除操作
            if hasattr(checkpointer, 'adelete'):
                await checkpointer.adelete(config)
                thread_id = config.get("configurable", {}).get("thread_id", "unknown")
                self.logger.info(f"删除检查点成功: thread_id={thread_id}")
                return True
            else:
                self.logger.warning("当前存储器不支持删除操作")
                return False
                
        except Exception as e:
            self.logger.error(f"删除检查点失败: {e}")
            return False
    
    async def cleanup_old_checkpoints(
        self, 
        days: int = 30,
        thread_id: Optional[str] = None
    ) -> int:
        """清理旧的检查点
        
        Args:
            days: 保留天数
            thread_id: 特定线程ID（可选）
            
        Returns:
            int: 清理的检查点数量
        """
        try:
            # 这里需要根据具体的存储器实现清理逻辑
            # 目前只是记录日志
            self.logger.info(f"清理{days}天前的检查点")
            return 0
            
        except Exception as e:
            self.logger.error(f"清理检查点失败: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息
        
        Returns:
            Dict: 存储统计信息
        """
        return {
            "storage_type": self.storage_type,
            "is_initialized": self._is_initialized,
            "checkpointer_type": type(self._checkpointer).__name__ if self._checkpointer else None,
        }
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 清理PostgreSQL上下文管理器
            if self._checkpointer_context:
                await self._checkpointer_context.__aexit__(None, None, None)
                self.logger.info("PostgreSQL检查点存储器资源清理完成")
            
            # 清理其他上下文管理器
            if self._context_manager:
                await self._context_manager.__aexit__(None, None, None)
                self.logger.info("检查点存储器资源清理完成")
            
            self.logger.info("检查点管理器清理完成")
            
        except Exception as e:
            self.logger.error(f"检查点管理器清理失败: {e}")
        finally:
            self._checkpointer = None
            self._checkpointer_context = None
            self._context_manager = None
            if hasattr(self, '_connection_pool'):
                self._connection_pool = None
            self._is_initialized = False


# 全局检查点管理器实例
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(storage_type: str = "postgres") -> CheckpointManager:
    """获取全局检查点管理器实例
    
    Args:
        storage_type: 存储类型
        
    Returns:
        CheckpointManager: 检查点管理器实例
    """
    global _checkpoint_manager
    
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager(storage_type)
    
    return _checkpoint_manager


@asynccontextmanager
async def checkpoint_manager_context(storage_type: str = "postgres"):
    """检查点管理器上下文管理器
    
    Args:
        storage_type: 存储类型
        
    Yields:
        CheckpointManager: 检查点管理器实例
    """
    manager = CheckpointManager(storage_type)
    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.cleanup()