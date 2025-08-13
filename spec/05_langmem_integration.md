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
from typing import Optional, Literal
from pydantic import Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 兼容不同版本的 Pydantic
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class MemoryConfig(BaseSettings):
    # 存储配置
    store_type: Literal["postgres", "memory"] = Field(
        default="postgres", 
        description="存储类型：postgres 或 memory"
    )
    postgres_url: str = Field(
        default_factory=lambda: os.getenv(
            "POSTGRES_URI", 
            os.getenv("LANGMEM_POSTGRES_URL", "postgresql://user:pass@localhost:5432/langmem")
        ),
        description="PostgreSQL 连接URL"
    )
    
    # 嵌入模型配置（阿里云DashScope）
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("LLM_EMBEDDING_MODEL", "openai:text-embedding-v4"),
        description="嵌入模型名称（阿里云DashScope text-embedding-v4）"
    )
    embedding_dims: int = Field(
        default_factory=lambda: int(os.getenv("LLM_EMBEDDING_DIMENSIONS", "1024")),
        description="嵌入向量维度（text-embedding-v4为1024维）"
    )
    
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
# core/memory/store_manager.py
import os
import logging
from typing import Optional, Dict, Any
from langgraph.store.postgres import PostgresStore
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
        raise ValueError(f"不支持的嵌入模型: {embedding_model}")

class MemoryStoreManager:
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
                # 使用from_conn_string创建PostgresStore上下文管理器
                self._store_context = PostgresStore.from_conn_string(
                    memory_config.postgres_url,
                    index={
                        "embed": embeddings,  # 使用DashScopeEmbeddings实例
                        "dims": memory_config.embedding_dims,
                        "fields": ["$"]  # 对所有字段进行向量化
                    }
                )
                self.store = self._store_context.__enter__()
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
                store=self.store,
                namespace_prefix=memory_config.namespace_prefix
            )
            
            self._initialized = True
            logger.info("记忆存储管理器初始化完成")
            
        except Exception as e:
            logger.error(f"记忆存储管理器初始化失败: {e}")
            raise
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 如果是PostgresStore，需要退出上下文管理器
            if self._store_context:
                self._store_context.__exit__(None, None, None)
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

## 🚀 提示词优化功能

### 概述

LangMem的提示词优化功能是一个强大的工具，可以基于用户反馈和对话历史自动改进智能体的提示词。这个功能可以显著提升智能体系统的性能和用户满意度。

### 核心组件

#### 1. 提示词优化器 (PromptOptimizer)

```python
# core/optimization/prompt_optimizer.py
from typing import Dict, Any, List, Optional
from langmem import optimize_prompt
from core.memory.store_manager import memory_store_manager

class PromptOptimizer:
    """提示词优化器"""
    
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.model_name = "anthropic:claude-3-5-sonnet-latest"
        self.optimization_namespace = "prompt_optimization"
        
    async def initialize(self):
        """初始化优化器"""
        await self.memory_manager.initialize()
    
    async def optimize_agent_prompt(self, agent_type: str, current_prompt: str, 
                                  min_feedback_count: int = 10) -> Optional[str]:
        """优化单个智能体的提示词"""
        
        # 1. 收集反馈数据
        feedback_data = await self._collect_feedback_data(agent_type, min_feedback_count)
        
        if not feedback_data or len(feedback_data) < min_feedback_count:
            return None
        
        # 2. 分析反馈模式
        feedback_analysis = await self._analyze_feedback_patterns(feedback_data)
        
        # 3. 使用LangMem优化提示词
        optimized_prompt = await optimize_prompt(
            store=self.memory_manager.store,
            namespace=f"{self.optimization_namespace}_{agent_type}",
            current_prompt=current_prompt,
            feedback_data=feedback_analysis,
            model=self.model_name
        )
        
        # 4. 存储优化历史
        await self._store_optimization_history(agent_type, current_prompt, optimized_prompt, feedback_analysis)
        
        return optimized_prompt
    
    async def optimize_multi_agent_system(self, agent_prompts: Dict[str, str]) -> Dict[str, str]:
        """优化多智能体系统的协作效果"""
        
        # 1. 分析团队协作模式
        collaboration_patterns = await self._analyze_collaboration_patterns(agent_prompts.keys())
        
        # 2. 优化每个智能体的提示词
        optimized_prompts = {}
        for agent_type, current_prompt in agent_prompts.items():
            # 考虑协作上下文的优化
            optimized = await self._optimize_with_collaboration_context(
                agent_type, current_prompt, collaboration_patterns
            )
            optimized_prompts[agent_type] = optimized or current_prompt
        
        return optimized_prompts
    
    async def _collect_feedback_data(self, agent_type: str, min_count: int) -> List[Dict[str, Any]]:
        """收集反馈数据"""
        feedback_namespace = f"feedback_{agent_type}"
        
        # 从记忆存储中搜索反馈数据
        feedback_memories = await self.memory_manager.search_memories(
            query="用户反馈 满意度",
            namespace=feedback_namespace,
            limit=min_count * 2
        )
        
        return [memory.content for memory in feedback_memories[:min_count]]
    
    async def _analyze_feedback_patterns(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析反馈模式"""
        # 计算平均满意度
        satisfaction_scores = [fb.get("satisfaction_score", 0) for fb in feedback_data]
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0
        
        # 提取常见问题和建议
        common_issues = []
        improvement_suggestions = []
        
        for feedback in feedback_data:
            if feedback.get("feedback_text"):
                if feedback.get("satisfaction_score", 0) < 0.6:
                    common_issues.append(feedback["feedback_text"])
                else:
                    improvement_suggestions.append(feedback["feedback_text"])
        
        return {
            "avg_satisfaction": avg_satisfaction,
            "total_feedback_count": len(feedback_data),
            "common_issues": common_issues,
            "improvement_suggestions": improvement_suggestions
        }
    
    async def _store_optimization_history(self, agent_type: str, original_prompt: str, 
                                        optimized_prompt: str, analysis: Dict[str, Any]):
        """存储优化历史"""
        history_data = {
            "agent_type": agent_type,
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "optimization_analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.memory_manager.store_memory(
            content=f"提示词优化: {agent_type}",
            memory_type="procedural",
            namespace=f"{self.optimization_namespace}_history",
            metadata=history_data
        )
```

#### 2. 反馈收集器 (FeedbackCollector)

```python
class FeedbackCollector:
    """用户反馈收集器"""
    
    def __init__(self, prompt_optimizer: PromptOptimizer):
        self.prompt_optimizer = prompt_optimizer
        self.feedback_namespace = "prompt_feedback"
    
    async def collect_user_feedback(self, conversation_id: str, messages: List[Message],
                                  satisfaction_score: float, feedback_text: str = "",
                                  agent_type: str = "default"):
        """收集用户反馈"""
        
        feedback_data = {
            "conversation_id": conversation_id,
            "agent_type": agent_type,
            "satisfaction_score": satisfaction_score,
            "feedback_text": feedback_text,
            "message_count": len(messages),
            "timestamp": datetime.now().isoformat()
        }
        
        # 存储到记忆系统
        await self.prompt_optimizer.memory_manager.store_memory(
            content=f"用户反馈: 满意度{satisfaction_score} - {feedback_text}",
            memory_type="episodic",
            namespace=f"feedback_{agent_type}",
            metadata=feedback_data
        )
    
    async def collect_improvement_suggestion(self, agent_type: str, suggestion: str,
                                           context: Dict[str, Any] = None):
        """收集改进建议"""
        
        suggestion_data = {
            "agent_type": agent_type,
            "suggestion": suggestion,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        await self.prompt_optimizer.memory_manager.store_memory(
            content=f"改进建议: {suggestion}",
            memory_type="semantic",
            namespace=f"suggestions_{agent_type}",
            metadata=suggestion_data
        )
```

#### 3. 自动优化调度器 (AutoOptimizationScheduler)

```python
class AutoOptimizationScheduler:
    """自动优化调度器"""
    
    def __init__(self, prompt_optimizer: PromptOptimizer):
        self.prompt_optimizer = prompt_optimizer
        self.config = {
            "enabled": False,
            "interval_hours": 24,
            "min_feedback_count": 10,
            "optimization_strategy": "gradient"
        }
        self.is_running = False
        self.last_optimization = {}
    
    async def configure_auto_optimization(self, config: Dict[str, Any]):
        """配置自动优化"""
        self.config.update(config)
    
    async def run_optimization_cycle(self, agent_configs: Dict[str, str]) -> Dict[str, str]:
        """运行优化周期"""
        if self.is_running:
            return agent_configs
        
        self.is_running = True
        optimized_configs = {}
        
        try:
            for agent_type, current_prompt in agent_configs.items():
                # 检查是否需要优化
                if await self._should_optimize(agent_type):
                    optimized = await self.prompt_optimizer.optimize_agent_prompt(
                        agent_type=agent_type,
                        current_prompt=current_prompt,
                        min_feedback_count=self.config["min_feedback_count"]
                    )
                    optimized_configs[agent_type] = optimized or current_prompt
                else:
                    optimized_configs[agent_type] = current_prompt
            
            # 更新最后优化时间
            self.last_optimization = {
                "timestamp": datetime.now().isoformat(),
                "optimized_agents": list(optimized_configs.keys())
            }
            
        finally:
            self.is_running = False
        
        return optimized_configs
    
    async def _should_optimize(self, agent_type: str) -> bool:
        """判断是否应该优化"""
        # 检查反馈数量
        feedback_count = await self._get_feedback_count(agent_type)
        return feedback_count >= self.config["min_feedback_count"]
    
    async def _get_feedback_count(self, agent_type: str) -> int:
        """获取反馈数量"""
        feedback_memories = await self.prompt_optimizer.memory_manager.search_memories(
            query="用户反馈",
            namespace=f"feedback_{agent_type}",
            limit=100
        )
        return len(feedback_memories)
```

### API集成

#### 提示词优化API路由

```python
# core/optimization/prompt_optimization_api.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from models.base_models import BaseResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/optimization", tags=["提示词优化"])

class FeedbackRequest(BaseModel):
    conversation_id: str
    satisfaction_score: float
    feedback_text: str = ""
    agent_type: str = "default"
    messages: List[Dict[str, Any]] = []

class OptimizationRequest(BaseModel):
    current_prompt: str
    min_feedback_count: int = 10

class MultiAgentOptimizationRequest(BaseModel):
    agent_prompts: Dict[str, str]

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    feedback_collector: FeedbackCollector = Depends(get_feedback_collector)
):
    """提交用户反馈"""
    try:
        await feedback_collector.collect_user_feedback(
            conversation_id=request.conversation_id,
            messages=request.messages,
            satisfaction_score=request.satisfaction_score,
            feedback_text=request.feedback_text,
            agent_type=request.agent_type
        )
        return BaseResponse(message="反馈已收集")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_type}/optimize")
async def optimize_agent_prompt(
    agent_type: str,
    request: OptimizationRequest,
    prompt_optimizer: PromptOptimizer = Depends(get_prompt_optimizer)
):
    """优化单个智能体提示词"""
    try:
        optimized_prompt = await prompt_optimizer.optimize_agent_prompt(
            agent_type=agent_type,
            current_prompt=request.current_prompt,
            min_feedback_count=request.min_feedback_count
        )
        
        if optimized_prompt:
            return {
                "status": "success",
                "optimized_prompt": optimized_prompt,
                "improvement_detected": True
            }
        else:
            return {
                "status": "skipped",
                "message": "反馈数据不足或无需优化",
                "improvement_detected": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multi-agent/optimize")
async def optimize_multi_agent_system(
    request: MultiAgentOptimizationRequest,
    prompt_optimizer: PromptOptimizer = Depends(get_prompt_optimizer)
):
    """优化多智能体系统"""
    try:
        optimized_prompts = await prompt_optimizer.optimize_multi_agent_system(
            agent_prompts=request.agent_prompts
        )
        return {
            "status": "success",
            "optimized_prompts": optimized_prompts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 最佳实践

#### 1. 反馈收集策略
- **及时收集**：在每次对话结束后收集反馈
- **多维度评估**：收集满意度评分、具体建议、改进方向
- **用户友好**：使用简单的评分系统（1-5星）和可选的文字反馈

#### 2. 优化频率控制
- **定期优化**：建议每周执行一次自动优化
- **数据门槛**：确保有足够的反馈数据（建议≥10条）
- **渐进改进**：避免频繁的大幅度提示词变更

#### 3. 版本管理
- **记录历史**：保存所有优化历史和效果对比
- **回滚机制**：支持回退到之前的提示词版本
- **A/B测试**：对比新旧提示词的效果

## 智能体类型集成示例

### 1. 记忆增强的监督智能体

```python
# core/agents/supervisor_agent.py
import asyncio
from typing import Dict, Any, List
from langmem import create_manage_memory_tool, create_search_memory_tool
from core.memory.store_manager import memory_store_manager

class MemoryEnhancedSupervisorAgent:
    """带记忆功能的监督智能体"""
    
    def __init__(self, agent_id: str = "supervisor"):
        self.agent_id = agent_id
        self.namespace = f"supervisor_{agent_id}"
        
        # 记忆工具
        self.manage_memory_tool = None
        self.search_memory_tool = None
        
    async def initialize(self):
        """初始化记忆工具"""
        await memory_store_manager.initialize()
        
        self.manage_memory_tool = create_manage_memory_tool(
            store=memory_store_manager.store,
            namespace=self.namespace
        )
        
        self.search_memory_tool = create_search_memory_tool(
            store=memory_store_manager.store,
            namespace=self.namespace
        )
    
    async def coordinate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """协调任务执行，利用记忆优化决策"""
        if not self.search_memory_tool:
            await self.initialize()
        
        # 搜索相关的历史任务记忆
        similar_tasks = await self.search_memory_tool.ainvoke({
            "query": f"任务类型: {task.get('type', '')} 任务描述: {task.get('description', '')}",
            "limit": 3
        })
        
        # 基于历史经验做出决策
        coordination_result = await self._make_coordination_decision(task, similar_tasks)
        
        # 存储任务协调记忆
        await self.manage_memory_tool.ainvoke({
            "action": "store",
            "memory": {
                "content": f"任务协调: {task['description']} -> 分配给: {coordination_result['assigned_agents']}",
                "type": "procedural",
                "metadata": {
                    "task_id": task.get("id"),
                    "task_type": task.get("type"),
                    "assigned_agents": coordination_result["assigned_agents"],
                    "coordination_strategy": coordination_result["strategy"]
                }
            }
        })
        
        return coordination_result
    
    async def _make_coordination_decision(self, task: Dict[str, Any], 
                                        similar_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于历史记忆做出协调决策"""
        # 分析历史任务的成功模式
        successful_patterns = []
        for memory in similar_tasks.get("memories", []):
            if memory.get("metadata", {}).get("success_rate", 0) > 0.8:
                successful_patterns.append(memory["metadata"])
        
        # 基于成功模式选择智能体
        if successful_patterns:
            # 使用历史成功的智能体组合
            assigned_agents = successful_patterns[0].get("assigned_agents", ["default_agent"])
            strategy = "历史成功模式"
        else:
            # 默认分配策略
            assigned_agents = self._default_assignment(task)
            strategy = "默认分配"
        
        return {
            "assigned_agents": assigned_agents,
            "strategy": strategy,
            "confidence": 0.9 if successful_patterns else 0.6
        }
    
    def _default_assignment(self, task: Dict[str, Any]) -> List[str]:
        """默认任务分配逻辑"""
        task_type = task.get("type", "general")
        
        assignment_map = {
            "research": ["research_agent", "web_search_agent"],
            "analysis": ["analysis_agent", "data_agent"],
            "writing": ["writing_agent", "review_agent"],
            "coding": ["coding_agent", "test_agent"]
        }
        
        return assignment_map.get(task_type, ["general_agent"])
```

### 2. 记忆增强的RAG智能体

```python
# core/agents/rag_agent.py
import asyncio
from typing import Dict, Any, List, Optional
from langmem import create_manage_memory_tool, create_search_memory_tool
from core.memory.store_manager import memory_store_manager

class MemoryEnhancedRAGAgent:
    """带记忆功能的RAG智能体"""
    
    def __init__(self, agent_id: str = "rag_agent"):
        self.agent_id = agent_id
        self.namespace = f"rag_{agent_id}"
        
        # 记忆工具
        self.manage_memory_tool = None
        self.search_memory_tool = None
        
    async def initialize(self):
        """初始化记忆工具"""
        await memory_store_manager.initialize()
        
        self.manage_memory_tool = create_manage_memory_tool(
            store=memory_store_manager.store,
            namespace=self.namespace
        )
        
        self.search_memory_tool = create_search_memory_tool(
            store=memory_store_manager.store,
            namespace=self.namespace
        )
    
    async def enhanced_retrieval(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """记忆增强的检索"""
        if not self.search_memory_tool:
            await self.initialize()
        
        # 1. 搜索查询模式记忆
        query_patterns = await self.search_memory_tool.ainvoke({
            "query": f"查询模式: {query}",
            "limit": 3
        })
        
        # 2. 优化查询
        optimized_query = await self._optimize_query(query, query_patterns)
        
        # 3. 执行检索（这里应该调用实际的RAG检索逻辑）
        retrieval_results = await self._perform_retrieval(optimized_query)
        
        # 4. 存储查询模式记忆
        await self.manage_memory_tool.ainvoke({
            "action": "store",
            "memory": {
                "content": f"查询: {query} -> 优化查询: {optimized_query}",
                "type": "procedural",
                "metadata": {
                    "original_query": query,
                    "optimized_query": optimized_query,
                    "results_count": len(retrieval_results.get("documents", [])),
                    "context": context
                }
            }
        })
        
        return {
            "query": optimized_query,
            "documents": retrieval_results.get("documents", []),
            "optimization_applied": optimized_query != query
        }
    
    async def _optimize_query(self, query: str, query_patterns: Dict[str, Any]) -> str:
        """基于历史模式优化查询"""
        patterns = query_patterns.get("memories", [])
        
        if not patterns:
            return query
        
        # 分析成功的查询模式
        successful_patterns = [
            p for p in patterns 
            if p.get("metadata", {}).get("results_count", 0) > 0
        ]
        
        if successful_patterns:
            # 应用最成功的查询优化模式
            best_pattern = max(
                successful_patterns,
                key=lambda x: x.get("metadata", {}).get("results_count", 0)
            )
            
            # 这里可以实现更复杂的查询优化逻辑
            # 例如添加关键词、调整查询结构等
            optimized = query + " " + best_pattern.get("metadata", {}).get("optimization_keywords", "")
            return optimized.strip()
        
        return query
    
    async def _perform_retrieval(self, query: str) -> Dict[str, Any]:
        """执行实际的检索操作"""
        # 这里应该实现实际的RAG检索逻辑
        # 例如调用向量数据库、搜索引擎等
        
        # 模拟检索结果
        return {
            "documents": [
                {"content": f"检索到的文档1 for: {query}", "score": 0.9},
                {"content": f"检索到的文档2 for: {query}", "score": 0.8},
            ]
        }
    
    async def record_feedback(self, query: str, results: List[Dict[str, Any]], 
                            user_feedback: Dict[str, Any]):
        """记录用户反馈，用于改进检索"""
        if not self.manage_memory_tool:
            await self.initialize()
        
        await self.manage_memory_tool.ainvoke({
            "action": "store",
            "memory": {
                "content": f"用户反馈: 查询'{query}' 满意度: {user_feedback.get('satisfaction', 'unknown')}",
                "type": "episodic",
                "metadata": {
                    "query": query,
                    "results_count": len(results),
                    "satisfaction": user_feedback.get("satisfaction"),
                    "helpful_docs": user_feedback.get("helpful_docs", []),
                    "suggestions": user_feedback.get("suggestions", "")
                }
            }
        })
```

## 记忆管理API实现

### 核心记忆操作

```python
# core/memory/api.py
import asyncio
from typing import Dict, Any, List, Optional
from langmem import create_manage_memory_tool, create_search_memory_tool
from core.memory.store_manager import memory_store_manager

class MemoryAPI:
    """记忆管理API封装"""
    
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.manage_tool = None
        self.search_tool = None
        
    async def initialize(self):
        """初始化记忆工具"""
        await memory_store_manager.initialize()
        
        self.manage_tool = create_manage_memory_tool(
            store=memory_store_manager.store,
            namespace=self.namespace
        )
        
        self.search_tool = create_search_memory_tool(
            store=memory_store_manager.store,
            namespace=self.namespace
        )
    
    async def store_memory(self, content: str, memory_type: str = "episodic", 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """存储记忆"""
        if not self.manage_tool:
            await self.initialize()
        
        try:
            result = await self.manage_tool.ainvoke({
                "action": "store",
                "memory": {
                    "content": content,
                    "type": memory_type,
                    "metadata": metadata or {}
                }
            })
            return result.get("success", False)
        except Exception as e:
            print(f"存储记忆失败: {e}")
            return False
    
    async def search_memories(self, query: str, limit: int = 5, 
                            memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索记忆"""
        if not self.search_tool:
            await self.initialize()
        
        try:
            search_params = {
                "query": query,
                "limit": limit
            }
            
            if memory_type:
                search_params["filter"] = {"type": memory_type}
            
            result = await self.search_tool.ainvoke(search_params)
            return result.get("memories", [])
        except Exception as e:
            print(f"搜索记忆失败: {e}")
            return []
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新记忆"""
        if not self.manage_tool:
            await self.initialize()
        
        try:
            result = await self.manage_tool.ainvoke({
                "action": "update",
                "memory_id": memory_id,
                "updates": updates
            })
            return result.get("success", False)
        except Exception as e:
            print(f"更新记忆失败: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        if not self.manage_tool:
            await self.initialize()
        
        try:
            result = await self.manage_tool.ainvoke({
                "action": "delete",
                "memory_id": memory_id
            })
            return result.get("success", False)
        except Exception as e:
            print(f"删除记忆失败: {e}")
            return False

# 使用示例
async def example_usage():
    """记忆API使用示例"""
    # 创建记忆API实例
    memory_api = MemoryAPI("example_namespace")
    
    # 存储不同类型的记忆
    await memory_api.store_memory(
        content="用户喜欢技术类文章",
        memory_type="semantic",
        metadata={"category": "user_preference", "confidence": 0.9}
    )
    
    await memory_api.store_memory(
        content="2024-01-15: 用户询问了Python异步编程",
        memory_type="episodic",
        metadata={"timestamp": "2024-01-15", "topic": "python"}
    )
    
    await memory_api.store_memory(
        content="处理技术问题的标准流程：分析->搜索->验证->回答",
        memory_type="procedural",
        metadata={"process_type": "technical_support"}
    )
    
    # 搜索记忆
    tech_memories = await memory_api.search_memories(
        query="技术 Python",
        limit=3
    )
    
    print(f"找到 {len(tech_memories)} 条相关记忆")
    for memory in tech_memories:
        print(f"- {memory['content']}")

if __name__ == "__main__":
    asyncio.run(example_usage())
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