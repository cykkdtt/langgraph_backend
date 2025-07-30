# LangMem 集成指南

## 概述

LangMem 是 LangChain 生态系统中的记忆管理组件，专为 LangGraph 应用设计，提供智能体的长期记忆存储和检索能力。本文档详细说明如何在多智能体协作平台中集成 LangMem。

## 核心特性

### 1. 记忆类型支持
- **语义记忆**: 存储概念、事实和知识
- **情节记忆**: 存储具体的交互历史和事件
- **程序记忆**: 存储技能、策略和行为模式

### 2. 智能记忆管理
- 自动记忆提取和存储
- 语义搜索和精确匹配
- 记忆整合和优化
- 命名空间隔离

### 3. LangGraph 原生集成
- 与 LangGraph Store 无缝集成
- 支持检查点系统
- 状态持久化

## 安装和配置

### 1. 依赖安装

```bash
pip install langmem langgraph[store] psycopg2-binary redis
```

### 2. 环境配置

```python
# config/memory_config.py
import os
from typing import Optional
from pydantic import BaseSettings

class MemoryConfig(BaseSettings):
    # 存储配置
    store_type: str = "postgres"
    postgres_url: str = os.getenv("LANGMEM_POSTGRES_URL", "postgresql://user:pass@localhost:5432/langmem")
    
    # 嵌入模型配置
    embedding_model: str = "openai:text-embedding-3-small"
    embedding_dims: int = 1536
    
    # 记忆管理配置
    max_memories_per_namespace: int = 10000
    auto_consolidate: bool = True
    consolidate_threshold: int = 1000
    
    # 缓存配置
    redis_url: Optional[str] = "redis://localhost:6379/0"
    cache_ttl: int = 3600
    
    # 清理配置
    cleanup_interval: int = 86400  # 24小时
    backup_enabled: bool = True
    backup_interval: int = 604800  # 7天

memory_config = MemoryConfig()
```

## 核心组件集成

### 1. 记忆存储管理器

```python
# app/core/memory/store_manager.py
from langmem import create_memory_store_manager
from langgraph.store import PostgresStore
from app.config.memory_config import memory_config

class MemoryStoreManager:
    def __init__(self):
        # 创建 PostgreSQL 存储
        self.store = PostgresStore(
            connection_string=memory_config.postgres_url,
            index_config={
                "dims": memory_config.embedding_dims,
                "embed": memory_config.embedding_model
            }
        )
        
        # 创建记忆存储管理器
        self.memory_manager = create_memory_store_manager(
            store=self.store,
            namespace_prefix="langgraph_agents"
        )
    
    async def initialize(self):
        """初始化存储"""
        await self.store.setup()
    
    async def cleanup(self):
        """清理资源"""
        await self.store.close()
    
    def get_namespace(self, agent_id: str, session_id: str) -> str:
        """生成命名空间"""
        return f"agent_{agent_id}_session_{session_id}"

# 全局实例
memory_store_manager = MemoryStoreManager()
```

### 2. 记忆工具创建器

```python
# app/core/memory/tools.py
from langmem import create_manage_memory_tool, create_search_memory_tool
from typing import List, Dict, Any
from app.core.memory.store_manager import memory_store_manager

class MemoryToolsFactory:
    @staticmethod
    def create_memory_tools(namespace: str) -> List[Dict[str, Any]]:
        """为指定命名空间创建记忆工具"""
        
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
        
        return [manage_tool, search_tool]
    
    @staticmethod
    def create_semantic_memory_tool(namespace: str):
        """创建语义记忆工具"""
        return create_manage_memory_tool(
            store=memory_store_manager.store,
            namespace=f"{namespace}_semantic",
            memory_type="semantic"
        )
    
    @staticmethod
    def create_episodic_memory_tool(namespace: str):
        """创建情节记忆工具"""
        return create_manage_memory_tool(
            store=memory_store_manager.store,
            namespace=f"{namespace}_episodic",
            memory_type="episodic"
        )
    
    @staticmethod
    def create_procedural_memory_tool(namespace: str):
        """创建程序记忆工具"""
        return create_manage_memory_tool(
            store=memory_store_manager.store,
            namespace=f"{namespace}_procedural",
            memory_type="procedural"
        )
```

### 3. 记忆增强的智能体基类

```python
# app/agents/memory_enhanced_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langgraph import StateGraph
from app.core.memory.tools import MemoryToolsFactory
from app.core.memory.store_manager import memory_store_manager
from app.agents.base import BaseAgent

class MemoryEnhancedAgent(BaseAgent, ABC):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.memory_namespace = None
        self.memory_tools = []
    
    async def setup_memory(self, session_id: str):
        """设置记忆系统"""
        self.memory_namespace = memory_store_manager.get_namespace(
            self.agent_id, session_id
        )
        
        # 创建记忆工具
        self.memory_tools = MemoryToolsFactory.create_memory_tools(
            self.memory_namespace
        )
        
        # 添加到智能体工具列表
        self.tools.extend(self.memory_tools)
    
    async def search_memories(self, query: str, memory_type: Optional[str] = None) -> List[Dict]:
        """搜索记忆"""
        search_namespace = self.memory_namespace
        if memory_type:
            search_namespace = f"{self.memory_namespace}_{memory_type}"
        
        search_tool = MemoryToolsFactory.create_search_memory_tool(search_namespace)
        return await search_tool.ainvoke({"query": query})
    
    async def store_memory(self, content: str, memory_type: str = "semantic", 
                          metadata: Optional[Dict] = None):
        """存储记忆"""
        namespace = f"{self.memory_namespace}_{memory_type}"
        manage_tool = MemoryToolsFactory.create_manage_memory_tool(namespace)
        
        memory_data = {
            "content": content,
            "metadata": metadata or {}
        }
        
        return await manage_tool.ainvoke({
            "action": "create",
            "memory": memory_data
        })
    
    async def inject_memory_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """注入记忆上下文"""
        if "messages" in state and state["messages"]:
            last_message = state["messages"][-1]
            
            # 搜索相关记忆
            relevant_memories = await self.search_memories(
                query=last_message.content,
                memory_type="semantic"
            )
            
            # 将记忆添加到上下文
            if relevant_memories:
                memory_context = "\n".join([
                    f"记忆: {mem['content']}" for mem in relevant_memories[:3]
                ])
                state["memory_context"] = memory_context
        
        return state
    
    async def update_memory_background(self, interaction_data: Dict[str, Any]):
        """后台更新记忆"""
        # 提取关键信息存储为语义记忆
        if "key_insights" in interaction_data:
            await self.store_memory(
                content=interaction_data["key_insights"],
                memory_type="semantic",
                metadata={"timestamp": interaction_data.get("timestamp")}
            )
        
        # 存储完整交互为情节记忆
        if "full_interaction" in interaction_data:
            await self.store_memory(
                content=interaction_data["full_interaction"],
                memory_type="episodic",
                metadata={
                    "session_id": interaction_data.get("session_id"),
                    "timestamp": interaction_data.get("timestamp")
                }
            )
```

## 智能体类型集成示例

### 1. 记忆增强的监督智能体

```python
# app/agents/memory_supervisor.py
from app.agents.memory_enhanced_base import MemoryEnhancedAgent
from app.agents.supervisor import SupervisorAgent

class MemoryEnhancedSupervisorAgent(MemoryEnhancedAgent, SupervisorAgent):
    async def coordinate_with_memory(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """基于记忆进行任务协调"""
        # 注入记忆上下文
        state = await self.inject_memory_context(state)
        
        # 搜索协作模式记忆
        collaboration_memories = await self.search_memories(
            query=f"协作模式 {state.get('task_type', '')}",
            memory_type="procedural"
        )
        
        # 基于记忆选择协作策略
        if collaboration_memories:
            state["collaboration_strategy"] = collaboration_memories[0]["content"]
        
        # 执行原有协调逻辑
        result = await super().coordinate(state)
        
        # 记录协作结果
        await self.store_memory(
            content=f"协作任务: {state.get('task_type')} 结果: {result.get('status')}",
            memory_type="episodic",
            metadata={"task_id": state.get("task_id")}
        )
        
        return result
```

### 2. 记忆增强的 RAG 智能体

```python
# app/agents/memory_rag.py
from app.agents.memory_enhanced_base import MemoryEnhancedAgent
from app.agents.rag import RAGAgent

class MemoryEnhancedRAGAgent(MemoryEnhancedAgent, RAGAgent):
    async def process_with_memory(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """结合记忆的 RAG 处理"""
        query = state.get("query", "")
        
        # 搜索用户偏好记忆
        preference_memories = await self.search_memories(
            query=f"用户偏好 {query}",
            memory_type="semantic"
        )
        
        # 基于记忆优化查询
        if preference_memories:
            enhanced_query = f"{query} 考虑用户偏好: {preference_memories[0]['content']}"
            state["enhanced_query"] = enhanced_query
        
        # 执行 RAG 处理
        result = await super().process(state)
        
        # 记录查询模式
        await self.store_memory(
            content=f"查询模式: {query} -> 结果类型: {result.get('result_type')}",
            memory_type="procedural",
            metadata={"query_timestamp": state.get("timestamp")}
        )
        
        return result
```

## API 集成

### 1. 记忆管理 API

```python
# app/api/memory.py
from fastapi import APIRouter, HTTPException, Depends
from app.core.memory.store_manager import memory_store_manager
from app.models.memory import MemoryManageRequest, MemorySearchRequest

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])

@router.post("/manage")
async def manage_memory(request: MemoryManageRequest):
    """管理记忆"""
    try:
        namespace = memory_store_manager.get_namespace(
            request.agent_id, request.session_id
        )
        
        if request.memory_type:
            namespace = f"{namespace}_{request.memory_type}"
        
        manage_tool = MemoryToolsFactory.create_manage_memory_tool(namespace)
        result = await manage_tool.ainvoke({
            "action": request.action,
            "memory": request.memory_data
        })
        
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_memory(request: MemorySearchRequest):
    """搜索记忆"""
    try:
        namespace = memory_store_manager.get_namespace(
            request.agent_id, request.session_id
        )
        
        if request.memory_type:
            namespace = f"{namespace}_{request.memory_type}"
        
        search_tool = MemoryToolsFactory.create_search_memory_tool(namespace)
        results = await search_tool.ainvoke({
            "query": request.query,
            "limit": request.limit
        })
        
        return {"status": "success", "memories": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 最佳实践

### 1. 命名空间设计
- 使用层次化命名空间: `agent_{agent_id}_session_{session_id}_{memory_type}`
- 为不同记忆类型使用独立命名空间
- 考虑多租户隔离需求

### 2. 记忆生命周期管理
- 定期清理过期记忆
- 实施记忆整合策略
- 备份重要记忆数据

### 3. 性能优化
- 使用 Redis 缓存频繁访问的记忆
- 批量处理记忆操作
- 异步处理记忆更新

### 4. 安全考虑
- 加密敏感记忆内容
- 实施访问控制
- 审计记忆操作日志

## 监控和调试

### 1. 记忆系统指标
- 记忆存储大小
- 搜索响应时间
- 记忆命中率
- 整合操作频率

### 2. 调试工具
- 记忆内容查看器
- 命名空间浏览器
- 搜索结果分析器

## 故障排除

### 常见问题
1. **记忆搜索无结果**: 检查命名空间配置和嵌入模型
2. **存储连接失败**: 验证 PostgreSQL 连接字符串
3. **记忆整合失败**: 检查存储空间和权限
4. **性能问题**: 优化索引配置和缓存策略

通过以上集成指南，可以在多智能体协作平台中充分利用 LangMem 的记忆管理能力，提升智能体的学习和适应能力。