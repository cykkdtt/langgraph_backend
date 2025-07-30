# 多智能体LangGraph项目 - 智能体实现与工具集成

## LangMem 记忆增强智能体实现

### 记忆增强基类

```python
from langmem import create_manage_memory_tool, create_search_memory_tool, create_memory_manager
from langgraph.store.memory import InMemoryStore, AsyncPostgresStore
from typing import List, Dict, Any, Optional

class MemoryEnhancedAgent(BaseAgent):
    """记忆增强智能体基类"""
    
    def __init__(
        self,
        agent_id: str,
        model: str,
        store_config: Optional[Dict[str, Any]] = None,
        memory_namespace: Optional[tuple[str, ...]] = None
    ):
        super().__init__(agent_id, model)
        self.store = self._setup_memory_store(store_config)
        self.memory_namespace = memory_namespace or ("agent_memory", "{agent_id}", "{user_id}")
        self.memory_tools = self._create_memory_tools()
        self.memory_manager = create_memory_manager(
            model,
            instructions="Extract important facts, preferences, and patterns from conversations.",
            enable_inserts=True,
            enable_updates=True,
        )
    
    def _setup_memory_store(self, config: Optional[Dict[str, Any]]) -> BaseStore:
        """设置记忆存储"""
        if config and config.get("type") == "postgres":
            return AsyncPostgresStore(
                connection_string=config["connection_string"],
                index={
                    "dims": config.get("embedding_dims", 1536),
                    "embed": config.get("embedding_model", "openai:text-embedding-3-small"),
                }
            )
        else:
            return InMemoryStore(
                index={
                    "dims": 1536,
                    "embed": "openai:text-embedding-3-small",
                }
            )
    
    def _create_memory_tools(self) -> List:
        """创建记忆工具"""
        return [
            create_manage_memory_tool(
                namespace=self.memory_namespace,
                instructions="Store important information about user preferences, facts, and context.",
            ),
            create_search_memory_tool(
                namespace=self.memory_namespace,
                instructions="Search for relevant memories to inform your responses.",
            ),
        ]
    
    async def process_with_memory(
        self, 
        messages: List[Dict[str, Any]], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """带记忆处理的核心方法"""
        # 1. 搜索相关记忆
        relevant_memories = await self._search_relevant_memories(messages, config)
        
        # 2. 将记忆注入上下文
        enhanced_messages = self._inject_memory_context(messages, relevant_memories)
        
        # 3. 处理请求
        response = await self.process(enhanced_messages, config)
        
        # 4. 后台记忆提取和更新
        await self._update_memories(messages + [response], config)
        
        return response
    
    async def _search_relevant_memories(
        self, 
        messages: List[Dict[str, Any]], 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """搜索相关记忆"""
        if not messages:
            return []
        
        # 使用最后一条用户消息作为搜索查询
        last_user_message = next(
            (msg for msg in reversed(messages) if msg.get("role") == "user"),
            None
        )
        
        if not last_user_message:
            return []
        
        search_tool = self.memory_tools[1]  # search_memory_tool
        try:
            memories = await search_tool.ainvoke({
                "query": last_user_message["content"],
                "limit": 5
            }, config=config)
            return memories if isinstance(memories, list) else []
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []
    
    def _inject_memory_context(
        self, 
        messages: List[Dict[str, Any]], 
        memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """将记忆注入消息上下文"""
        if not memories:
            return messages
        
        memory_context = "\n".join([
            f"- {memory.get('content', '')}" for memory in memories
        ])
        
        system_message = {
            "role": "system",
            "content": f"""You have access to the following relevant memories from past interactions:

<memories>
{memory_context}
</memories>

Use these memories to provide more personalized and contextual responses."""
        }
        
        # 在消息开头插入系统消息
        return [system_message] + messages
    
    async def _update_memories(
        self, 
        conversation: List[Dict[str, Any]], 
        config: Dict[str, Any]
    ):
        """后台更新记忆"""
        try:
            # 使用记忆管理器提取新记忆
            extracted_memories = await self.memory_manager.ainvoke({
                "messages": conversation
            })
            
            # 存储提取的记忆
            manage_tool = self.memory_tools[0]  # manage_memory_tool
            for memory in extracted_memories:
                await manage_tool.ainvoke({
                    "content": memory.content,
                    "action": "create"
                }, config=config)
                
        except Exception as e:
            logger.warning(f"Memory update failed: {e}")
```

### 记忆增强多智能体协作

```python
class MemoryEnhancedSupervisorAgent(MemoryEnhancedAgent):
    """记忆增强监督智能体"""
    
    def __init__(self, agent_id: str, model: str, sub_agents: List[str]):
        super().__init__(
            agent_id, 
            model,
            memory_namespace=("supervisor_memory", "{user_id}")
        )
        self.sub_agents = sub_agents
    
    async def coordinate_with_memory(
        self, 
        task: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于记忆的任务协调"""
        # 搜索相关的协作模式记忆
        coordination_memories = await self._search_coordination_patterns(task, config)
        
        # 基于记忆选择最佳协作策略
        strategy = self._select_coordination_strategy(task, coordination_memories)
        
        # 执行协作任务
        result = await self._execute_coordination(task, strategy, config)
        
        # 记录协作结果和模式
        await self._record_coordination_outcome(task, strategy, result, config)
        
        return result
    
    async def _search_coordination_patterns(
        self, 
        task: str, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """搜索协作模式记忆"""
        search_tool = self.memory_tools[1]
        return await search_tool.ainvoke({
            "query": f"coordination patterns for: {task}",
            "limit": 3
        }, config=config)
    
    def _select_coordination_strategy(
        self, 
        task: str, 
        memories: List[Dict[str, Any]]
    ) -> str:
        """基于记忆选择协作策略"""
        if not memories:
            return "default_sequential"
        
        # 分析记忆中的成功模式
        successful_patterns = [
            memory for memory in memories 
            if memory.get("metadata", {}).get("success_score", 0) > 0.7
        ]
        
        if successful_patterns:
            return successful_patterns[0].get("metadata", {}).get("strategy", "default_sequential")
        
        return "default_sequential"
    
    async def _record_coordination_outcome(
        self, 
        task: str, 
        strategy: str, 
        result: Dict[str, Any], 
        config: Dict[str, Any]
    ):
        """记录协作结果"""
        success_score = result.get("success_score", 0.5)
        
        memory_content = f"""
        Coordination Pattern:
        - Task: {task}
        - Strategy: {strategy}
        - Success Score: {success_score}
        - Agents Used: {', '.join(self.sub_agents)}
        """
        
        manage_tool = self.memory_tools[0]
        await manage_tool.ainvoke({
            "content": memory_content,
            "action": "create",
            "metadata": {
                "type": "coordination_pattern",
                "strategy": strategy,
                "success_score": success_score,
                "task_type": self._classify_task_type(task)
            }
        }, config=config)
```

### 记忆增强 RAG 智能体

```python
class MemoryEnhancedRAGAgent(MemoryEnhancedAgent):
    """记忆增强 RAG 智能体"""
    
    def __init__(self, agent_id: str, model: str, retriever):
        super().__init__(
            agent_id, 
            model,
            memory_namespace=("rag_memory", "{user_id}")
        )
        self.retriever = retriever
    
    async def rag_with_memory(
        self, 
        query: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """结合记忆的 RAG 处理"""
        # 1. 搜索用户偏好和查询历史
        user_preferences = await self._search_user_preferences(query, config)
        
        # 2. 基于记忆优化检索查询
        optimized_query = self._optimize_query_with_memory(query, user_preferences)
        
        # 3. 执行文档检索
        retrieved_docs = await self.retriever.aretrieve(optimized_query)
        
        # 4. 结合记忆生成回答
        response = await self._generate_with_memory_context(
            query, retrieved_docs, user_preferences, config
        )
        
        # 5. 记录查询模式和偏好
        await self._record_query_pattern(query, response, config)
        
        return response
    
    async def _search_user_preferences(
        self, 
        query: str, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """搜索用户偏好记忆"""
        search_tool = self.memory_tools[1]
        return await search_tool.ainvoke({
            "query": f"user preferences for: {query}",
            "limit": 3
        }, config=config)
    
    def _optimize_query_with_memory(
        self, 
        query: str, 
        preferences: List[Dict[str, Any]]
    ) -> str:
        """基于记忆优化查询"""
        if not preferences:
            return query
        
        # 提取偏好关键词
        preference_keywords = []
        for pref in preferences:
            content = pref.get("content", "")
            if "prefers" in content.lower():
                preference_keywords.extend(
                    content.lower().split("prefers")[1].split()[:3]
                )
        
        if preference_keywords:
            return f"{query} {' '.join(preference_keywords)}"
        
        return query
    
    async def _record_query_pattern(
        self, 
        query: str, 
        response: Dict[str, Any], 
        config: Dict[str, Any]
    ):
        """记录查询模式"""
        satisfaction_score = response.get("satisfaction_score", 0.5)
        
        memory_content = f"""
        Query Pattern:
        - Query: {query}
        - Response Quality: {satisfaction_score}
        - Topics: {response.get("topics", [])}
        """
        
        manage_tool = self.memory_tools[0]
        await manage_tool.ainvoke({
            "content": memory_content,
            "action": "create",
            "metadata": {
                "type": "query_pattern",
                "satisfaction_score": satisfaction_score,
                "query_type": self._classify_query_type(query)
            }
        }, config=config)
```

## 1. 智能体实现规范

### 1.1 智能体基类实现
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator, List
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.checkpoint import BaseCheckpointSaver
import asyncio

class BaseAgent(ABC):
    """智能体基类，定义所有智能体的通用接口"""
    
    def __init__(self, config: Dict[str, Any], checkpointer: Optional[BaseCheckpointSaver] = None):
        self.config = config
        self.checkpointer = checkpointer
        self.graph = None
        self.tools = []
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """初始化智能体，子类必须实现"""
        pass
    
    @abstractmethod
    async def chat(self, message: str, thread_id: Optional[str] = None, 
                   user_id: str = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """处理对话请求"""
        pass
    
    @abstractmethod
    async def stream_chat(self, message: str, thread_id: Optional[str] = None,
                         user_id: str = None, context: Optional[Dict] = None) -> AsyncGenerator[Dict, None]:
        """流式对话处理"""
        pass
    
    def add_tool(self, tool):
        """添加工具到智能体"""
        self.tools.append(tool)
    
    def get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return self.config.get("capabilities", [])
```

### 1.2 多智能体协作实现 (graph5.py)
```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
import operator

class MultiAgentState(TypedDict):
    messages: Annotated[List, operator.add]
    next: str
    user_id: str
    thread_id: str

class SupervisorAgent(BaseAgent):
    """Supervisor智能体 - 协调其他智能体"""
    
    def _initialize(self):
        """初始化Supervisor智能体图"""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        self.llm = ChatOpenAI(
            model=self.config.get("supervisor_model", "deepseek-chat"),
            temperature=0
        )
        
        # 定义系统提示
        self.system_prompt = """你是一个智能体协调器。根据用户请求，决定调用哪个智能体：
        - research: 用于搜索和研究任务
        - chart: 用于数据可视化和图表生成
        - FINISH: 任务完成
        
        只返回智能体名称，不要其他内容。"""
        
        # 构建状态图
        workflow = StateGraph(MultiAgentState)
        
        # 添加节点
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("chart", self._chart_node)
        
        # 添加边
        workflow.add_edge("research", "supervisor")
        workflow.add_edge("chart", "supervisor")
        
        # 条件边
        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {
                "research": "research",
                "chart": "chart",
                "FINISH": END
            }
        )
        
        workflow.set_entry_point("supervisor")
        self.graph = workflow.compile(checkpointer=self.checkpointer)
    
    async def _supervisor_node(self, state: MultiAgentState):
        """Supervisor节点逻辑"""
        messages = state["messages"]
        response = await self.llm.ainvoke([
            SystemMessage(content=self.system_prompt),
            *messages
        ])
        
        return {
            "next": response.content.strip(),
            "messages": [response]
        }
    
    async def _research_node(self, state: MultiAgentState):
        """Research智能体节点"""
        # 这里调用实际的research智能体
        research_agent = ResearchAgent(self.config)
        result = await research_agent.process(state["messages"][-1].content)
        
        return {
            "messages": [result],
            "next": "supervisor"
        }
    
    async def _chart_node(self, state: MultiAgentState):
        """Chart智能体节点"""
        # 这里调用实际的chart智能体
        chart_agent = ChartAgent(self.config)
        result = await chart_agent.process(state["messages"][-1].content)
        
        return {
            "messages": [result],
            "next": "supervisor"
        }
    
    def _should_continue(self, state: MultiAgentState):
        """决定下一步执行哪个智能体"""
        return state["next"]
    
    async def chat(self, message: str, thread_id: Optional[str] = None,
                   user_id: str = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """处理对话请求"""
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "thread_id": thread_id
        }
        
        result = await self.graph.ainvoke(initial_state, config)
        
        return {
            "response": result["messages"][-1].content,
            "thread_id": thread_id,
            "agent_type": "multi_agent_supervisor",
            "agent_used": "supervisor",
            "execution_time": 0.0,
            "agent_chain": self._extract_agent_chain(result)
        }
    
    async def stream_chat(self, message: str, thread_id: Optional[str] = None,
                         user_id: str = None, context: Optional[Dict] = None) -> AsyncGenerator[Dict, None]:
        """流式对话处理"""
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "thread_id": thread_id
        }
        
        async for chunk in self.graph.astream(initial_state, config):
            for node_name, node_output in chunk.items():
                yield {
                    "type": "agent_switch",
                    "content": f"切换到 {node_name} 智能体",
                    "agent_name": node_name,
                    "metadata": node_output
                }
                
                if "messages" in node_output and node_output["messages"]:
                    yield {
                        "type": "message",
                        "content": node_output["messages"][-1].content,
                        "agent_name": node_name
                    }
```

### 1.3 RAG智能体实现 (graph6.py)
```python
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from typing import TypedDict, List

class RAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    user_id: str
    thread_id: str

class AgenticRAGAgent(BaseAgent):
    """智能RAG系统"""
    
    def _initialize(self):
        """初始化RAG智能体"""
        self.embeddings = OpenAIEmbeddings(
            model=self.config.get("embedding_model", "text-embedding-ada-002")
        )
        self.llm = ChatOpenAI(
            model=self.config.get("llm_model", "deepseek-chat"),
            temperature=0
        )
        
        # 初始化向量存储
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.config.get("vectorstore_path", "./chroma_db")
        )
        
        # 构建RAG图
        workflow = StateGraph(RAGState)
        
        # 添加节点
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("transform_query", self._transform_query)
        workflow.add_node("web_search", self._web_search)
        
        # 添加边
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate"
            }
        )
        workflow.add_edge("transform_query", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        
        self.graph = workflow.compile(checkpointer=self.checkpointer)
    
    async def _retrieve_documents(self, state: RAGState):
        """检索相关文档"""
        question = state["question"]
        documents = self.vectorstore.similarity_search(
            question, 
            k=self.config.get("top_k", 5)
        )
        
        return {"documents": documents}
    
    async def _grade_documents(self, state: RAGState):
        """评估文档相关性"""
        question = state["question"]
        documents = state["documents"]
        
        # 简单的相关性评估逻辑
        relevant_docs = []
        for doc in documents:
            # 这里可以使用更复杂的相关性评估
            if len(doc.page_content) > 50:  # 简单的长度过滤
                relevant_docs.append(doc)
        
        return {"documents": relevant_docs}
    
    def _decide_to_generate(self, state: RAGState):
        """决定是否生成回答或需要更多信息"""
        documents = state["documents"]
        
        if len(documents) >= 2:  # 有足够的相关文档
            return "generate"
        else:
            return "transform_query"
    
    async def _transform_query(self, state: RAGState):
        """转换查询以获得更好的搜索结果"""
        question = state["question"]
        
        # 使用LLM重写查询
        rewrite_prompt = f"重写以下查询以获得更好的搜索结果：{question}"
        response = await self.llm.ainvoke([{"role": "user", "content": rewrite_prompt}])
        
        return {"question": response.content}
    
    async def _web_search(self, state: RAGState):
        """网络搜索补充信息"""
        question = state["question"]
        
        # 这里集成网络搜索工具
        # 示例：使用Google搜索API
        search_results = []  # 实际搜索结果
        
        documents = [Document(page_content=result, metadata={"source": "web"}) 
                    for result in search_results]
        
        return {"documents": state["documents"] + documents}
    
    async def _generate_response(self, state: RAGState):
        """生成最终回答"""
        question = state["question"]
        documents = state["documents"]
        
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = f"""基于以下上下文回答问题：

上下文：
{context}

问题：{question}

请提供准确、有用的回答。如果上下文中没有足够信息，请说明。"""

        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        
        return {
            "generation": response.content,
            "retrieved_documents": [{"content": doc.page_content, "source": doc.metadata.get("source", "unknown")} 
                                  for doc in documents]
        }
    
    async def chat(self, message: str, thread_id: Optional[str] = None,
                   user_id: str = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """处理RAG对话请求"""
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        initial_state = {
            "question": message,
            "user_id": user_id,
            "thread_id": thread_id,
            "documents": [],
            "generation": ""
        }
        
        result = await self.graph.ainvoke(initial_state, config)
        
        return {
            "response": result["generation"],
            "thread_id": thread_id,
            "agent_type": "agentic_rag",
            "agent_used": "rag_agent",
            "execution_time": 0.0,
            "retrieved_documents": result.get("retrieved_documents", []),
            "retrieval_score": len(result.get("documents", []))
        }
```

### 1.4 专业化智能体实现
```python
class CodeAgent(BaseAgent):
    """代码生成和分析智能体"""
    
    def _initialize(self):
        """初始化代码智能体"""
        self.llm = ChatOpenAI(
            model=self.config.get("model", "deepseek-chat"),
            temperature=0.1
        )
        
        # 代码执行工具
        self.code_executor = self._setup_code_executor()
        
        # 构建代码处理图
        workflow = StateGraph(dict)
        
        workflow.add_node("analyze_request", self._analyze_code_request)
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("execute_code", self._execute_code)
        workflow.add_node("review_code", self._review_code)
        
        workflow.set_entry_point("analyze_request")
        workflow.add_edge("analyze_request", "generate_code")
        workflow.add_conditional_edges(
            "generate_code",
            self._should_execute,
            {
                "execute": "execute_code",
                "review": "review_code"
            }
        )
        workflow.add_edge("execute_code", "review_code")
        workflow.add_edge("review_code", END)
        
        self.graph = workflow.compile(checkpointer=self.checkpointer)
    
    def _setup_code_executor(self):
        """设置代码执行环境"""
        # 安全的代码执行环境
        return {
            "timeout": self.config.get("execution_timeout", 30),
            "max_output_length": 10000,
            "allowed_imports": self.config.get("allowed_imports", [])
        }
    
    async def _analyze_code_request(self, state: dict):
        """分析代码请求"""
        request = state["request"]
        
        analysis_prompt = f"""分析以下代码请求，确定需要执行的操作：
        
请求：{request}

请返回JSON格式：
{{
    "task_type": "generate|analyze|review|execute",
    "language": "python|javascript|etc",
    "complexity": "simple|medium|complex",
    "requires_execution": true|false
}}"""

        response = await self.llm.ainvoke([{"role": "user", "content": analysis_prompt}])
        
        # 解析响应（这里简化处理）
        analysis = {
            "task_type": "generate",
            "language": "python",
            "complexity": "medium",
            "requires_execution": False
        }
        
        return {"analysis": analysis}
    
    async def _generate_code(self, state: dict):
        """生成代码"""
        request = state["request"]
        analysis = state["analysis"]
        
        code_prompt = f"""根据以下要求生成{analysis['language']}代码：

要求：{request}

请提供：
1. 完整的代码实现
2. 必要的注释
3. 使用示例

代码："""

        response = await self.llm.ainvoke([{"role": "user", "content": code_prompt}])
        
        return {"generated_code": response.content}
    
    def _should_execute(self, state: dict):
        """决定是否执行代码"""
        analysis = state.get("analysis", {})
        return "execute" if analysis.get("requires_execution") else "review"
    
    async def _execute_code(self, state: dict):
        """执行代码"""
        code = state["generated_code"]
        
        # 安全执行代码（这里需要实现安全沙箱）
        try:
            # 实际执行逻辑
            execution_result = "代码执行成功"
            output = "示例输出"
        except Exception as e:
            execution_result = f"执行错误：{str(e)}"
            output = ""
        
        return {
            "execution_result": execution_result,
            "execution_output": output
        }
    
    async def _review_code(self, state: dict):
        """代码审查"""
        code = state["generated_code"]
        
        review_prompt = f"""请审查以下代码：

{code}

请提供：
1. 代码质量评估
2. 潜在问题
3. 改进建议
4. 安全性评估"""

        response = await self.llm.ainvoke([{"role": "user", "content": review_prompt}])
        
        return {"code_review": response.content}
    
    async def chat(self, message: str, thread_id: Optional[str] = None,
                   user_id: str = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """处理代码相关请求"""
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        initial_state = {
            "request": message,
            "user_id": user_id,
            "thread_id": thread_id
        }
        
        result = await self.graph.ainvoke(initial_state, config)
        
        specialized_output = {
            "generated_code": result.get("generated_code", ""),
            "execution_result": result.get("execution_result", ""),
            "execution_output": result.get("execution_output", ""),
            "code_review": result.get("code_review", "")
        }
        
        return {
            "response": result.get("generated_code", ""),
            "thread_id": thread_id,
            "agent_type": "code_agent",
            "agent_used": "code_agent",
            "execution_time": 0.0,
            "specialized_output": specialized_output
        }
```

## 2. 工具集成和中断处理

### 2.1 工具管理系统
```python
from typing import Dict, Any, Callable, List
from abc import ABC, abstractmethod
import asyncio
import inspect

class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具参数schema"""
        sig = inspect.signature(self.execute)
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            if param_name != "kwargs":
                schema["properties"][param_name] = {
                    "type": "string",  # 简化处理
                    "description": f"Parameter {param_name}"
                }
                if param.default == inspect.Parameter.empty:
                    schema["required"].append(param_name)
        
        return schema

class ToolManager:
    """工具管理器"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_bindings: Dict[str, List[str]] = {}  # agent_type -> tool_names
    
    def register_tool(self, tool: BaseTool, agent_types: List[str] = None):
        """注册工具"""
        self.tools[tool.name] = tool
        
        if agent_types:
            for agent_type in agent_types:
                if agent_type not in self.tool_bindings:
                    self.tool_bindings[agent_type] = []
                self.tool_bindings[agent_type].append(tool.name)
    
    def get_tools_for_agent(self, agent_type: str) -> List[BaseTool]:
        """获取智能体可用的工具"""
        tool_names = self.tool_bindings.get(agent_type, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """执行工具"""
        if tool_name not in self.tools:
            raise ValueError(f"工具 {tool_name} 不存在")
        
        tool = self.tools[tool_name]
        return await tool.execute(**kwargs)

# 示例工具实现
class WebSearchTool(BaseTool):
    """网络搜索工具"""
    
    def __init__(self):
        super().__init__("search_web", "搜索网络信息")
    
    async def execute(self, query: str, num_results: int = 5) -> List[Dict]:
        """执行网络搜索"""
        # 这里集成实际的搜索API
        await asyncio.sleep(1)  # 模拟API调用
        
        return [
            {
                "title": f"搜索结果 {i+1}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"关于 {query} 的搜索结果 {i+1}"
            }
            for i in range(num_results)
        ]

class CalculatorTool(BaseTool):
    """计算器工具"""
    
    def __init__(self):
        super().__init__("calculate", "执行数学计算")
    
    async def execute(self, expression: str) -> Dict[str, Any]:
        """执行数学计算"""
        try:
            # 安全的数学表达式计算
            result = eval(expression, {"__builtins__": {}}, {})
            return {
                "result": result,
                "expression": expression,
                "success": True
            }
        except Exception as e:
            return {
                "result": None,
                "expression": expression,
                "success": False,
                "error": str(e)
            }

class CodeExecutorTool(BaseTool):
    """代码执行工具"""
    
    def __init__(self):
        super().__init__("code_executor", "执行Python代码")
    
    async def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """执行Python代码"""
        try:
            # 这里需要实现安全的代码执行环境
            # 可以使用Docker容器或其他沙箱技术
            
            import subprocess
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "success": result.returncode == 0
                }
            finally:
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "代码执行超时",
                "return_code": -1,
                "success": False
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "success": False
            }

# 工具注册示例
def setup_tools() -> ToolManager:
    """设置工具管理器"""
    tool_manager = ToolManager()
    
    # 注册工具
    tool_manager.register_tool(
        WebSearchTool(), 
        ["multi_agent_supervisor", "agentic_rag"]
    )
    tool_manager.register_tool(
        CalculatorTool(), 
        ["multi_agent_supervisor", "code_agent"]
    )
    tool_manager.register_tool(
        CodeExecutorTool(), 
        ["code_agent"]
    )
    
    return tool_manager
```

### 2.2 中断和人工干预
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import BaseCheckpointSaver
from typing import Dict, Any, Optional
import asyncio

class InterruptibleAgent(BaseAgent):
    """支持中断和人工干预的智能体"""
    
    def _initialize(self):
        """初始化可中断的智能体图"""
        workflow = StateGraph(dict)
        
        # 添加节点
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("human_approval", self._human_approval)
        workflow.add_node("execute_action", self._execute_action)
        workflow.add_node("finalize", self._finalize)
        
        # 设置入口点
        workflow.set_entry_point("process_input")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "process_input",
            self._needs_approval,
            {
                "approve": "human_approval",
                "execute": "execute_action"
            }
        )
        
        # 人工审批后的路径
        workflow.add_conditional_edges(
            "human_approval",
            self._approval_result,
            {
                "approved": "execute_action",
                "rejected": "finalize",
                "modified": "process_input"
            }
        )
        
        workflow.add_edge("execute_action", "finalize")
        workflow.add_edge("finalize", END)
        
        self.graph = workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_approval"]  # 在人工审批前中断
        )
    
    async def _process_input(self, state: dict):
        """处理输入"""
        user_input = state["input"]
        
        # 分析输入，确定是否需要人工审批
        analysis = await self._analyze_input(user_input)
        
        return {
            "analysis": analysis,
            "requires_approval": analysis.get("risk_level", "low") == "high"
        }
    
    async def _analyze_input(self, user_input: str) -> Dict[str, Any]:
        """分析用户输入的风险级别"""
        # 这里可以使用LLM或规则引擎来分析
        risk_keywords = ["delete", "remove", "destroy", "harmful"]
        
        risk_level = "high" if any(keyword in user_input.lower() for keyword in risk_keywords) else "low"
        
        return {
            "risk_level": risk_level,
            "confidence": 0.8,
            "reasoning": f"检测到风险级别: {risk_level}"
        }
    
    def _needs_approval(self, state: dict):
        """判断是否需要人工审批"""
        return "approve" if state.get("requires_approval", False) else "execute"
    
    async def _human_approval(self, state: dict):
        """人工审批节点"""
        # 这个节点会被中断，等待人工干预
        return {
            "approval_status": "pending",
            "approval_message": "等待人工审批..."
        }
    
    def _approval_result(self, state: dict):
        """处理审批结果"""
        approval_status = state.get("approval_status", "pending")
        
        if approval_status == "approved":
            return "approved"
        elif approval_status == "rejected":
            return "rejected"
        elif approval_status == "modified":
            return "modified"
        else:
            # 默认等待审批
            return "approved"  # 简化处理
    
    async def _execute_action(self, state: dict):
        """执行动作"""
        analysis = state["analysis"]
        
        # 执行实际的动作
        result = f"执行完成: {analysis.get('reasoning', '未知操作')}"
        
        return {"result": result}
    
    async def _finalize(self, state: dict):
        """最终化处理"""
        return {
            "final_response": state.get("result", "操作已完成或被拒绝")
        }
    
    async def approve_action(self, thread_id: str, approval_status: str, 
                           modification: Optional[str] = None):
        """人工审批操作"""
        config = {"configurable": {"thread_id": thread_id}}
        
        # 更新状态以继续执行
        update_data = {"approval_status": approval_status}
        if modification:
            update_data["input"] = modification
        
        # 恢复执行
        result = await self.graph.ainvoke(None, config, input=update_data)
        return result
    
    async def chat(self, message: str, thread_id: Optional[str] = None,
                   user_id: str = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """处理对话请求（支持中断）"""
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        initial_state = {
            "input": message,
            "user_id": user_id,
            "thread_id": thread_id
        }
        
        try:
            result = await self.graph.ainvoke(initial_state, config)
            
            return {
                "response": result.get("final_response", "处理完成"),
                "thread_id": thread_id,
                "agent_type": "interruptible_agent",
                "agent_used": "interruptible_agent",
                "execution_time": 0.0,
                "requires_approval": result.get("requires_approval", False)
            }
        except Exception as e:
            if "interrupt" in str(e).lower():
                return {
                    "response": "操作需要人工审批，已暂停执行",
                    "thread_id": thread_id,
                    "agent_type": "interruptible_agent",
                    "agent_used": "interruptible_agent",
                    "execution_time": 0.0,
                    "requires_approval": True,
                    "status": "interrupted"
                }
            else:
                raise e
```

### 2.3 动态工具加载
```python
import importlib
import json
from typing import Dict, Any, List, Type

class DynamicToolLoader:
    """动态工具加载器"""
    
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
        self.loaded_tools: Dict[str, BaseTool] = {}
    
    async def load_tools_from_config(self, config_path: str):
        """从配置文件加载工具"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        for tool_config in config.get("tools", []):
            await self._load_tool_from_config(tool_config)
    
    async def _load_tool_from_config(self, tool_config: Dict[str, Any]):
        """从配置加载单个工具"""
        tool_type = tool_config.get("type")
        
        if tool_type == "function":
            tool = await self._load_function_tool(tool_config)
        elif tool_type == "api":
            tool = await self._load_api_tool(tool_config)
        elif tool_type == "database":
            tool = await self._load_database_tool(tool_config)
        else:
            raise ValueError(f"不支持的工具类型: {tool_type}")
        
        # 注册工具
        agent_types = tool_config.get("agent_types", [])
        self.tool_manager.register_tool(tool, agent_types)
        self.loaded_tools[tool.name] = tool
    
    async def _load_function_tool(self, config: Dict[str, Any]) -> BaseTool:
        """加载函数工具"""
        module_name = config["module"]
        function_name = config["function"]
        
        # 动态导入模块
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        
        class DynamicFunctionTool(BaseTool):
            def __init__(self, name: str, description: str, func: Callable):
                super().__init__(name, description)
                self.func = func
            
            async def execute(self, **kwargs) -> Any:
                if asyncio.iscoroutinefunction(self.func):
                    return await self.func(**kwargs)
                else:
                    return self.func(**kwargs)
        
        return DynamicFunctionTool(
            config["name"],
            config["description"],
            func
        )
    
    async def _load_api_tool(self, config: Dict[str, Any]) -> BaseTool:
        """加载API工具"""
        import aiohttp
        
        class DynamicAPITool(BaseTool):
            def __init__(self, name: str, description: str, api_config: Dict):
                super().__init__(name, description)
                self.api_config = api_config
            
            async def execute(self, **kwargs) -> Any:
                url = self.api_config["url"]
                method = self.api_config.get("method", "GET")
                headers = self.api_config.get("headers", {})
                
                async with aiohttp.ClientSession() as session:
                    if method.upper() == "GET":
                        async with session.get(url, params=kwargs, headers=headers) as response:
                            return await response.json()
                    elif method.upper() == "POST":
                        async with session.post(url, json=kwargs, headers=headers) as response:
                            return await response.json()
        
        return DynamicAPITool(
            config["name"],
            config["description"],
            config["api"]
        )
    
    async def _load_database_tool(self, config: Dict[str, Any]) -> BaseTool:
        """加载数据库工具"""
        import aiosqlite
        
        class DynamicDatabaseTool(BaseTool):
            def __init__(self, name: str, description: str, db_config: Dict):
                super().__init__(name, description)
                self.db_config = db_config
            
            async def execute(self, query: str, **kwargs) -> Any:
                db_path = self.db_config["path"]
                
                async with aiosqlite.connect(db_path) as db:
                    async with db.execute(query, kwargs) as cursor:
                        if query.strip().upper().startswith("SELECT"):
                            return await cursor.fetchall()
                        else:
                            await db.commit()
                            return {"affected_rows": cursor.rowcount}
        
        return DynamicDatabaseTool(
            config["name"],
            config["description"],
            config["database"]
        )

# 工具配置示例
TOOLS_CONFIG = {
    "tools": [
        {
            "name": "weather_api",
            "type": "api",
            "description": "获取天气信息",
            "api": {
                "url": "https://api.weather.com/v1/current",
                "method": "GET",
                "headers": {"API-Key": "your-api-key"}
            },
            "agent_types": ["multi_agent_supervisor"]
        },
        {
            "name": "user_database",
            "type": "database",
            "description": "查询用户数据库",
            "database": {
                "path": "./users.db"
            },
            "agent_types": ["agentic_rag"]
        },
        {
            "name": "custom_function",
            "type": "function",
            "description": "自定义处理函数",
            "module": "custom_tools",
            "function": "process_data",
            "agent_types": ["code_agent"]
        }
    ]
}
```

## 3. 流式响应实现规则

### 3.1 LangGraph流式模式
LangGraph SDK支持多种流式模式，每种模式适用于不同的场景：

```python
from langgraph.graph import StateGraph
from typing import AsyncGenerator, Dict, Any

class StreamingAgent(BaseAgent):
    """支持多种流式模式的智能体"""
    
    async def stream_with_mode(self, message: str, mode: str = "values", 
                              thread_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        """根据指定模式进行流式处理"""
        config = {"configurable": {"thread_id": thread_id or "default"}}
        initial_state = {"input": message}
        
        if mode == "values":
            # 流式输出完整状态值
            async for chunk in self.graph.astream(initial_state, config, stream_mode="values"):
                yield {
                    "type": "state_update",
                    "content": chunk,
                    "metadata": {"mode": "values"}
                }
        
        elif mode == "messages":
            # 流式输出消息
            async for chunk in self.graph.astream(initial_state, config, stream_mode="messages"):
                yield {
                    "type": "message",
                    "content": chunk.content if hasattr(chunk, 'content') else str(chunk),
                    "metadata": {"mode": "messages"}
                }
        
        elif mode == "updates":
            # 流式输出状态更新
            async for chunk in self.graph.astream(initial_state, config, stream_mode="updates"):
                yield {
                    "type": "update",
                    "content": chunk,
                    "metadata": {"mode": "updates"}
                }
        
        elif mode == "events":
            # 流式输出事件
            async for chunk in self.graph.astream_events(initial_state, config, version="v1"):
                yield {
                    "type": "event",
                    "content": chunk,
                    "metadata": {"mode": "events", "event_type": chunk.get("event")}
                }
        
        elif mode == "custom":
            # 自定义流式处理
            async for chunk in self._custom_stream(initial_state, config):
                yield chunk

# WebSocket流式实现
from fastapi import WebSocket
import json

class WebSocketStreamer:
    """WebSocket流式响应处理器"""
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
    
    async def handle_websocket(self, websocket: WebSocket, thread_id: str):
        """处理WebSocket连接"""
        await websocket.accept()
        
        try:
            while True:
                # 接收消息
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                message = message_data.get("message", "")
                stream_mode = message_data.get("stream_mode", "values")
                
                # 流式处理
                async for chunk in self.agent.stream_with_mode(
                    message, mode=stream_mode, thread_id=thread_id
                ):
                    await websocket.send_text(json.dumps(chunk))
                
                # 发送完成信号
                await websocket.send_text(json.dumps({
                    "type": "done",
                    "content": "Stream completed"
                }))
                
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": str(e)
            }))
        finally:
            await websocket.close()

# Server-Sent Events (SSE) 流式实现
from fastapi import Request
from fastapi.responses import StreamingResponse

class SSEStreamer:
    """Server-Sent Events流式响应处理器"""
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
    
    async def stream_response(self, message: str, thread_id: str, 
                            stream_mode: str = "values") -> StreamingResponse:
        """创建SSE流式响应"""
        
        async def generate():
            try:
                async for chunk in self.agent.stream_with_mode(
                    message, mode=stream_mode, thread_id=thread_id
                ):
                    # SSE格式
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # 发送完成事件
                yield f"data: {json.dumps({'type': 'done', 'content': 'Stream completed'})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
```

### 3.2 多智能体流式协调
```python
class MultiAgentStreamCoordinator:
    """多智能体流式协调器"""
    
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
        self.active_streams: Dict[str, AsyncGenerator] = {}
    
    async def coordinate_streams(self, message: str, thread_id: str) -> AsyncGenerator[Dict, None]:
        """协调多个智能体的流式响应"""
        
        # 确定需要激活的智能体
        active_agents = await self._determine_active_agents(message)
        
        # 启动多个流
        streams = {}
        for agent_name in active_agents:
            if agent_name in self.agents:
                streams[agent_name] = self.agents[agent_name].stream_chat(
                    message, thread_id=thread_id
                )
        
        # 协调流式输出
        async for chunk in self._merge_streams(streams):
            yield chunk
    
    async def _determine_active_agents(self, message: str) -> List[str]:
        """确定需要激活的智能体"""
        # 这里可以使用路由逻辑或LLM来决定
        # 简化实现：根据关键词判断
        
        active_agents = []
        
        if any(keyword in message.lower() for keyword in ["search", "research", "find"]):
            active_agents.append("research_agent")
        
        if any(keyword in message.lower() for keyword in ["chart", "graph", "visualize"]):
            active_agents.append("chart_agent")
        
        if any(keyword in message.lower() for keyword in ["code", "program", "script"]):
            active_agents.append("code_agent")
        
        # 默认包含supervisor
        if "supervisor_agent" not in active_agents:
            active_agents.insert(0, "supervisor_agent")
        
        return active_agents
    
    async def _merge_streams(self, streams: Dict[str, AsyncGenerator]) -> AsyncGenerator[Dict, None]:
        """合并多个流式输出"""
        import asyncio
        
        # 创建任务
        tasks = {}
        for agent_name, stream in streams.items():
            tasks[agent_name] = asyncio.create_task(self._stream_to_queue(stream, agent_name))
        
        # 使用队列来协调输出
        queue = asyncio.Queue()
        
        # 启动所有流
        for task in tasks.values():
            asyncio.create_task(self._queue_consumer(task, queue))
        
        # 输出合并的流
        while tasks:
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # 检测智能体切换
                if chunk.get("type") == "agent_switch":
                    yield {
                        "type": "agent_switch",
                        "content": f"切换到 {chunk['agent_name']} 智能体",
                        "agent_name": chunk["agent_name"],
                        "metadata": {"switch_reason": "task_handoff"}
                    }
                
                yield chunk
                
                # 检查是否有任务完成
                completed_tasks = [name for name, task in tasks.items() if task.done()]
                for name in completed_tasks:
                    del tasks[name]
                
            except asyncio.TimeoutError:
                # 检查是否所有任务都完成
                if all(task.done() for task in tasks.values()):
                    break
        
        yield {
            "type": "done",
            "content": "所有智能体处理完成"
        }
    
    async def _stream_to_queue(self, stream: AsyncGenerator, agent_name: str):
        """将流转换为队列项"""
        async for chunk in stream:
            chunk["agent_name"] = agent_name
            yield chunk
    
    async def _queue_consumer(self, stream_task, queue: asyncio.Queue):
        """队列消费者"""
        async for chunk in stream_task:
            await queue.put(chunk)
```