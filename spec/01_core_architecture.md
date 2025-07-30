# 多智能体LangGraph项目 - 核心架构

## 1. 项目概述

### 1.1 项目目标
构建一个基于LangGraph的多智能体系统，支持多种类型的智能体协作，包括：
- **多智能体协作系统** (graph5.py) - Supervisor、Research、Chart智能体协作
- **Agentic RAG系统** (graph6.py) - 智能检索增强生成
- **专业化智能体系统** (graph7.py+) - 代码、数据分析、内容创作等专业智能体

### 1.2 核心特性
- **统一架构**: 所有智能体基于统一的BaseAgent抽象类
- **模块化设计**: 每种智能体类型独立模块，支持动态加载
- **状态管理**: 基于LangGraph的检查点系统实现状态持久化
- **流式响应**: 支持WebSocket和SSE的实时流式交互
- **工具集成**: 统一的工具管理和动态加载机制
- **中断处理**: 支持人工干预和审批流程

### 1.3 技术栈
- **核心框架**: LangGraph, LangChain
- **Web框架**: FastAPI
- **数据库**: PostgreSQL (检查点存储), Redis (缓存)
- **向量数据库**: Chroma/Pinecone (RAG系统)
- **消息队列**: Redis/RabbitMQ
- **容器化**: Docker, Docker Compose

## 2. 架构设计原则

### 2.1 核心原则
1. **模块化**: 每个智能体类型独立模块，便于维护和扩展
2. **可扩展性**: 支持新智能体类型的动态添加
3. **统一接口**: 所有智能体遵循统一的API接口规范
4. **状态一致性**: 基于检查点的状态管理确保数据一致性
5. **向后兼容**: 新功能不破坏现有接口

### 2.2 LangMem 长期记忆架构

#### 记忆管理系统设计

LangMem 作为核心记忆管理组件，提供跨会话的持久化记忆能力：

##### 记忆存储架构
```python
from langgraph.store.memory import InMemoryStore, AsyncPostgresStore
from langmem import create_manage_memory_tool, create_search_memory_tool

class MemoryConfig:
    """记忆配置管理"""
    
    def __init__(self, store_type: str = "postgres"):
        if store_type == "postgres":
            self.store = AsyncPostgresStore(
                connection_string="postgresql://...",
                index={
                    "dims": 1536,
                    "embed": "openai:text-embedding-3-small",
                }
            )
        else:
            self.store = InMemoryStore(
                index={
                    "dims": 1536,
                    "embed": "openai:text-embedding-3-small",
                }
            )
    
    def create_memory_tools(self, namespace_pattern: tuple[str, ...]):
        """创建记忆工具"""
        return [
            create_manage_memory_tool(namespace=namespace_pattern),
            create_search_memory_tool(namespace=namespace_pattern),
        ]
```

##### 命名空间组织策略
```python
# 用户级别记忆隔离
USER_MEMORY_NAMESPACE = ("memories", "{user_id}")

# 智能体级别记忆共享
AGENT_MEMORY_NAMESPACE = ("agent_memories", "{agent_type}")

# 组织级别记忆管理
ORG_MEMORY_NAMESPACE = ("org_memories", "{org_id}", "{user_id}")

# 专业化记忆分类
SPECIALIZED_MEMORY_NAMESPACE = ("specialized", "{agent_type}", "{user_id}")
```

##### 记忆类型实现
```python
from pydantic import BaseModel
from typing import List, Dict, Any

class SemanticMemory(BaseModel):
    """语义记忆：事实和知识"""
    content: str
    importance: float = 0.5
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class EpisodicMemory(BaseModel):
    """情节记忆：过去经验"""
    situation: str
    action_taken: str
    outcome: str
    success_score: float
    context: Dict[str, Any] = {}

class ProceduralMemory(BaseModel):
    """程序记忆：行为模式"""
    trigger_pattern: str
    response_template: str
    confidence: float
    usage_count: int = 0
```

### 2.3 智能体分类
#### 2.3.1 协作型智能体 (graph5.py)
- **Supervisor智能体**: 任务分解和协调
- **Research智能体**: 信息搜索和研究
- **Chart智能体**: 数据可视化和图表生成

#### 2.3.2 RAG型智能体 (graph6.py)
- **文档检索**: 基于向量相似度的智能检索
- **上下文生成**: 结合检索结果的智能回答
- **知识管理**: 文档索引和知识库维护

#### 2.3.3 专业化智能体 (graph7.py+)
- **代码智能体**: 代码生成、分析、审查
- **数据分析智能体**: 数据处理、分析、可视化
- **内容创作智能体**: 文本创作、编辑、优化

### 2.3 智能体管理架构
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator
from langgraph.graph import StateGraph
from datetime import datetime
import uuid

class BaseAgent(ABC):
    """智能体抽象基类"""
    
    def __init__(self, agent_type: str, config: Dict[str, Any]):
        self.agent_type = agent_type
        self.config = config
        self.agent_id = str(uuid.uuid4())
        self.created_at = datetime.utcnow()
        self.graph = self._build_graph()
    
    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """构建智能体的LangGraph图"""
        pass
    
    @abstractmethod
    async def chat(self, message: str, thread_id: str = None, **kwargs) -> Dict[str, Any]:
        """处理对话请求"""
        pass
    
    @abstractmethod
    async def astream(self, message: str, thread_id: str = None, **kwargs) -> AsyncGenerator:
        """流式处理对话请求"""
        pass
    
    async def cleanup(self):
        """清理资源"""
        pass
    
    def get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return self.config.get("capabilities", [])
    
    def get_tools(self) -> List[str]:
        """获取智能体工具列表"""
        return self.config.get("tools", [])

class AgentRegistry:
    """智能体注册表"""
    
    def __init__(self):
        self._agents = {}
        self._configs = {}
    
    def register_agent_type(self, agent_type: str, agent_class: type, config: Dict[str, Any]):
        """注册智能体类型"""
        self._agents[agent_type] = agent_class
        self._configs[agent_type] = config
    
    def get_agent_class(self, agent_type: str) -> Optional[type]:
        """获取智能体类"""
        return self._agents.get(agent_type)
    
    def get_agent_config(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """获取智能体配置"""
        return self._configs.get(agent_type)
    
    def list_agent_types(self) -> List[str]:
        """列出所有注册的智能体类型"""
        return list(self._agents.keys())

class AgentFactory:
    """智能体工厂"""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._instances = {}
    
    async def create_agent(self, agent_type: str, custom_config: Dict[str, Any] = None) -> BaseAgent:
        """创建智能体实例"""
        agent_class = self.registry.get_agent_class(agent_type)
        if not agent_class:
            raise ValueError(f"未知的智能体类型: {agent_type}")
        
        # 合并配置
        base_config = self.registry.get_agent_config(agent_type) or {}
        config = {**base_config, **(custom_config or {})}
        
        # 创建实例
        agent = agent_class(agent_type, config)
        
        # 缓存实例
        self._instances[agent.agent_id] = agent
        
        return agent
    
    async def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """获取智能体实例"""
        return self._instances.get(agent_id)
    
    async def cleanup_agent(self, agent_id: str):
        """清理智能体实例"""
        if agent_id in self._instances:
            await self._instances[agent_id].cleanup()
            del self._instances[agent_id]

# 全局注册表和工厂
agent_registry = AgentRegistry()
agent_factory = AgentFactory(agent_registry)
```

### 2.4 状态管理和持久化

#### 2.4.1 检查点系统
LangGraph提供强大的检查点系统用于状态持久化和恢复：

```python
from langgraph.checkpoint.postgres import PostgresCheckpointer
from langgraph.checkpoint.memory import MemoryCheckpointer
from langgraph.checkpoint.sqlite import SqliteCheckpointer

# 配置检查点存储
class CheckpointManager:
    def __init__(self, storage_type: str = "postgres"):
        self.storage_type = storage_type
        self.checkpointer = self._create_checkpointer()
    
    def _create_checkpointer(self):
        """创建检查点存储器"""
        if self.storage_type == "postgres":
            return PostgresCheckpointer.from_conn_string(
                conn_string=os.getenv("POSTGRES_URL"),
                sync_connection=True
            )
        elif self.storage_type == "sqlite":
            return SqliteCheckpointer.from_conn_string(
                conn_string="checkpoints.db"
            )
        else:
            return MemoryCheckpointer()
    
    async def save_checkpoint(self, config: dict, state: dict, metadata: dict = None):
        """保存检查点"""
        checkpoint = {
            "state": state,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        await self.checkpointer.aput(config, checkpoint)
    
    async def load_checkpoint(self, config: dict):
        """加载检查点"""
        return await self.checkpointer.aget(config)
    
    async def list_checkpoints(self, config: dict, limit: int = 10):
        """列出检查点历史"""
        checkpoints = []
        async for checkpoint in self.checkpointer.alist(config, limit=limit):
            checkpoints.append(checkpoint)
        return checkpoints
```

#### 2.4.2 线程状态管理
```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
import operator

class ThreadState(TypedDict):
    """线程状态定义"""
    messages: Annotated[list, operator.add]
    user_id: str
    thread_id: str
    agent_type: str
    context: dict
    memory: dict
    tools_used: list
    last_activity: str

class ThreadManager:
    """线程状态管理器"""
    
    def __init__(self, checkpointer):
        self.checkpointer = checkpointer
        self.active_threads = {}
    
    async def create_thread(self, user_id: str, agent_type: str) -> str:
        """创建新线程"""
        thread_id = str(uuid.uuid4())
        initial_state = ThreadState(
            messages=[],
            user_id=user_id,
            thread_id=thread_id,
            agent_type=agent_type,
            context={},
            memory={},
            tools_used=[],
            last_activity=datetime.now().isoformat()
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        await self.checkpointer.aput(config, initial_state)
        
        return thread_id
    
    async def get_thread_state(self, thread_id: str) -> ThreadState:
        """获取线程状态"""
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await self.checkpointer.aget(config)
        return checkpoint.get("state") if checkpoint else None
    
    async def update_thread_state(self, thread_id: str, updates: dict):
        """更新线程状态"""
        current_state = await self.get_thread_state(thread_id)
        if current_state:
            current_state.update(updates)
            current_state["last_activity"] = datetime.now().isoformat()
            
            config = {"configurable": {"thread_id": thread_id}}
            await self.checkpointer.aput(config, current_state)
    
    async def get_thread_history(self, thread_id: str, limit: int = 50):
        """获取线程历史"""
        config = {"configurable": {"thread_id": thread_id}}
        history = []
        async for checkpoint in self.checkpointer.alist(config, limit=limit):
            history.append({
                "timestamp": checkpoint.get("metadata", {}).get("timestamp"),
                "state": checkpoint.get("state"),
                "checkpoint_id": checkpoint.get("id")
            })
        return history
    
    async def fork_thread(self, source_thread_id: str, checkpoint_id: str = None) -> str:
        """从检查点分叉线程"""
        new_thread_id = str(uuid.uuid4())
        
        if checkpoint_id:
            # 从特定检查点分叉
            source_config = {"configurable": {"thread_id": source_thread_id}}
            checkpoint = await self.checkpointer.aget_tuple(source_config, checkpoint_id)
        else:
            # 从最新状态分叉
            checkpoint = await self.get_thread_state(source_thread_id)
        
        if checkpoint:
            new_state = checkpoint.copy()
            new_state["thread_id"] = new_thread_id
            new_state["last_activity"] = datetime.now().isoformat()
            
            new_config = {"configurable": {"thread_id": new_thread_id}}
            await self.checkpointer.aput(new_config, new_state)
        
        return new_thread_id
```

#### 2.4.3 智能体状态集成
```python
class StatefulAgent(BaseAgent):
    """有状态的智能体基类"""
    
    def __init__(self, agent_type: str, config: dict):
        super().__init__(agent_type, config)
        self.thread_manager = ThreadManager(CheckpointManager().checkpointer)
        self.graph = self._build_stateful_graph()
    
    def _build_stateful_graph(self) -> StateGraph:
        """构建有状态的图"""
        graph = StateGraph(ThreadState)
        
        # 添加节点
        graph.add_node("process_input", self._process_input)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("update_memory", self._update_memory)
        
        # 添加边
        graph.add_edge("process_input", "generate_response")
        graph.add_edge("generate_response", "update_memory")
        
        # 设置入口点
        graph.set_entry_point("process_input")
        graph.set_finish_point("update_memory")
        
        # 编译图并添加检查点
        return graph.compile(checkpointer=self.thread_manager.checkpointer)
    
    async def chat(self, message: str, thread_id: str = None, **kwargs) -> ChatResponse:
        """有状态的对话处理"""
        if not thread_id:
            thread_id = await self.thread_manager.create_thread(
                user_id=kwargs.get("user_id", "anonymous"),
                agent_type=self.agent_type
            )
        
        # 配置线程
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50
        }
        
        # 准备输入
        input_data = {
            "messages": [{"role": "user", "content": message}],
            "user_id": kwargs.get("user_id", "anonymous"),
            "thread_id": thread_id,
            "agent_type": self.agent_type
        }
        
        # 执行图
        result = await self.graph.ainvoke(input_data, config=config)
        
        return ChatResponse(
            message=result["messages"][-1]["content"],
            thread_id=thread_id,
            agent_type=self.agent_type,
            metadata={
                "tools_used": result.get("tools_used", []),
                "memory_updates": result.get("memory", {}),
                "context": result.get("context", {})
            }
        )
    
    async def astream(self, message: str, thread_id: str = None, **kwargs):
        """有状态的流式处理"""
        if not thread_id:
            thread_id = await self.thread_manager.create_thread(
                user_id=kwargs.get("user_id", "anonymous"),
                agent_type=self.agent_type
            )
        
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50
        }
        
        input_data = {
            "messages": [{"role": "user", "content": message}],
            "user_id": kwargs.get("user_id", "anonymous"),
            "thread_id": thread_id,
            "agent_type": self.agent_type
        }
        
        # 流式执行
        async for chunk in self.graph.astream(
            input_data, 
            config=config,
            stream_mode=["values", "events", "updates"]
        ):
            yield chunk
```

## 3. 开发优先级

### 阶段1: 核心架构 (Week 1-2)
- [ ] BaseAgent抽象类实现
- [ ] AgentRegistry和AgentFactory
- [ ] 基础API框架 (FastAPI)
- [ ] 检查点系统集成
- [ ] 基础测试框架

### 阶段2: 智能体集成 (Week 3-4)
- [ ] graph5.py多智能体系统集成
- [ ] graph6.py RAG系统集成
- [ ] graph7.py专业化智能体集成
- [ ] 统一API接口实现

### 阶段3: 扩展功能 (Week 5-6)
- [ ] 流式响应实现 (WebSocket + SSE)
- [ ] 工具管理系统
- [ ] 中断和人工干预
- [ ] 记忆管理系统

### 阶段4: 生产就绪 (Week 7-8)
- [ ] 错误处理和监控
- [ ] 安全性实现
- [ ] 性能优化
- [ ] 容器化部署
- [ ] 完整测试覆盖
- [ ] 文档完善