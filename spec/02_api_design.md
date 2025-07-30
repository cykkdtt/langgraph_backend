# 多智能体LangGraph项目 - API设计与数据模型

## 1. API设计规范

### 1.1 RESTful API结构
```
/api/v1/
├── /agents/                          # 智能体管理
│   ├── GET /                         # 获取所有智能体类型
│   ├── GET /{agent_type}             # 获取特定智能体信息
│   ├── POST /{agent_type}/chat       # 智能体对话
│   ├── POST /{agent_type}/stream     # 流式对话
│   └── /instances/                   # 智能体实例管理
│       ├── POST /                    # 创建实例
│       ├── GET /{instance_id}        # 获取实例信息
│       └── DELETE /{instance_id}     # 删除实例
├── /multi-agent/                     # 多智能体协作 (graph5.py)
│   ├── /supervisor/chat              # Supervisor智能体对话
│   ├── /research/chat                # Research智能体对话
│   └── /chart/chat                   # Chart智能体对话
├── /rag/                             # RAG智能体 (graph6.py)
│   ├── /chat                         # Agentic RAG对话
│   ├── /documents                    # 文档管理
│   │   ├── POST /upload              # 上传文档
│   │   ├── GET /list                 # 文档列表
│   │   └── DELETE /{doc_id}          # 删除文档
│   ├── /vectorstore                  # 向量存储管理
│   │   ├── POST /index               # 索引文档
│   │   ├── GET /search               # 向量搜索
│   │   └── DELETE /clear             # 清空索引
│   └── /retrieval                    # 检索配置
│       ├── GET /config               # 获取检索配置
│       └── PUT /config               # 更新检索配置
├── /specialized/                     # 专业化智能体 (graph7.py+)
│   ├── /code/                        # 代码智能体
│   │   ├── /generate                 # 代码生成
│   │   ├── /analyze                  # 代码分析
│   │   └── /review                   # 代码审查
│   ├── /data-analysis/               # 数据分析智能体
│   │   ├── /analyze                  # 数据分析
│   │   ├── /visualize                # 数据可视化
│   │   └── /insights                 # 洞察生成
│   └── /content/                     # 内容创作智能体
│       ├── /create                   # 内容创作
│       ├── /edit                     # 内容编辑
│       └── /optimize                 # 内容优化
├── /threads/                         # 会话管理 (支持所有智能体类型)
│   ├── POST /                        # 创建新会话
│   ├── GET /{thread_id}              # 获取会话信息
│   ├── DELETE /{thread_id}           # 删除会话
│   ├── GET /{thread_id}/history      # 获取会话历史
│   ├── PUT /{thread_id}/agent-type   # 切换会话的智能体类型
│   ├── POST /{thread_id}/interrupt   # 中断执行
│   └── POST /{thread_id}/resume      # 恢复执行
├── /memory/                          # 记忆管理 (跨智能体共享)
│   ├── GET /users/{user_id}          # 获取用户记忆
│   ├── POST /users/{user_id}         # 保存用户记忆
│   ├── DELETE /users/{user_id}/{memory_id} # 删除特定记忆
│   └── GET /global                   # 获取全局知识库
├── /memory/                          # 记忆管理 (跨智能体共享)
│   ├── GET /users/{user_id}          # 获取用户记忆
│   ├── POST /users/{user_id}         # 保存用户记忆
│   ├── DELETE /users/{user_id}/{memory_id} # 删除特定记忆
│   ├── GET /global                   # 获取全局知识库
│   ├── POST /manage                  # 创建/更新/删除记忆
│   ├── GET /search                   # 搜索记忆
│   ├── GET /namespaces               # 获取命名空间列表
│   ├── DELETE /namespace/{ns}        # 清空命名空间
│   ├── GET /stats                    # 记忆统计信息
│   ├── POST /consolidate             # 记忆整合
│   ├── GET /insights                 # 记忆洞察分析
│   ├── GET /config                   # 获取记忆配置
│   ├── PUT /config                   # 更新记忆配置
│   └── POST /namespace               # 创建新命名空间
└── /tools/                           # 工具管理
    ├── /search                       # 搜索工具接口
    ├── /chart                        # 图表生成工具接口
    ├── /retrieval                    # 检索工具接口
    ├── /mcp                          # MCP工具接口
    ├── GET /available                # 获取可用工具列表
    ├── POST /register                # 注册新工具
    └── POST /execute                 # 执行工具调用
```

### 1.2 智能体类型配置
```python
# 智能体类型配置示例
AGENT_TYPE_CONFIGS = {
    "multi_agent_supervisor": {
        "module": "graph5",
        "class": "SupervisorSystem",
        "description": "多智能体协作系统，包含supervisor、research和chart智能体",
        "capabilities": ["research", "chart_generation", "task_coordination"],
        "tools": ["google_search", "mcp_chart_tools"],
        "memory_enabled": True,
        "streaming_supported": True
    },
    "agentic_rag": {
        "module": "graph6", 
        "class": "AgenticRAGSystem",
        "description": "智能检索增强生成系统",
        "capabilities": ["document_retrieval", "context_generation", "query_understanding"],
        "tools": ["vector_search", "document_loader"],
        "memory_enabled": True,
        "streaming_supported": True,
        "requires_vectorstore": True
    },
    "code_agent": {
        "module": "graph7",
        "class": "CodeAgentSystem", 
        "description": "代码生成和分析智能体",
        "capabilities": ["code_generation", "code_analysis", "code_review"],
        "tools": ["code_executor", "static_analyzer"],
        "memory_enabled": False,
        "streaming_supported": True
    }
}
```

## 2. 数据模型设计

### 2.1 LangMem 记忆管理模型

#### 记忆操作模型
```python
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

class MemoryManageRequest(BaseModel):
    """记忆管理请求"""
    content: Optional[str] = None
    id: Optional[str] = None
    action: Literal["create", "update", "delete"] = "create"
    namespace: Optional[tuple[str, ...]] = None
    metadata: Optional[Dict[str, Any]] = None

class MemorySearchRequest(BaseModel):
    """记忆搜索请求"""
    query: str
    namespace: Optional[tuple[str, ...]] = None
    limit: int = 10
    offset: int = 0
    filter: Optional[Dict[str, Any]] = None

class MemoryItem(BaseModel):
    """记忆项"""
    id: str
    content: str
    namespace: tuple[str, ...]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    importance: float = 0.5
    access_count: int = 0

class MemorySearchResponse(BaseModel):
    """记忆搜索响应"""
    memories: List[MemoryItem]
    total: int
    has_more: bool

class MemoryStats(BaseModel):
    """记忆统计"""
    total_memories: int
    namespaces: List[str]
    memory_by_namespace: Dict[str, int]
    avg_importance: float
    last_updated: datetime

class MemoryConfig(BaseModel):
    """记忆配置"""
    store_type: Literal["memory", "postgres"] = "postgres"
    embedding_model: str = "openai:text-embedding-3-small"
    embedding_dims: int = 1536
    max_memories_per_namespace: int = 10000
    auto_consolidate: bool = True
    consolidate_threshold: int = 1000
```

#### 记忆类型模型
```python
class SemanticMemoryModel(BaseModel):
    """语义记忆模型"""
    content: str
    importance: float = 0.5
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    knowledge_type: Literal["fact", "preference", "rule"] = "fact"

class EpisodicMemoryModel(BaseModel):
    """情节记忆模型"""
    situation: str
    action_taken: str
    outcome: str
    success_score: float
    context: Dict[str, Any] = {}
    learned_patterns: List[str] = []

class ProceduralMemoryModel(BaseModel):
    """程序记忆模型"""
    trigger_pattern: str
    response_template: str
    confidence: float
    usage_count: int = 0
    effectiveness_score: float = 0.5
```

### 2.2 基础响应模型
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# 基础响应模型
class BaseResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    request_id: str
    timestamp: datetime

# 智能体类型信息模型
class AgentTypeInfo(BaseModel):
    type: str
    name: str
    description: str
    capabilities: List[str]
    tools: List[str]
    memory_enabled: bool
    streaming_supported: bool
    requires_vectorstore: bool = False
    config_schema: Optional[Dict] = None
```

### 2.2 请求/响应模型
```python
# 扩展的对话请求模型
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    user_id: str
    agent_type: str = "multi_agent_supervisor"  # 支持多种智能体类型
    agent_config: Optional[Dict[str, Any]] = {}
    context: Optional[Dict[str, Any]] = {}  # 额外上下文信息
    
    # RAG特定配置
    rag_config: Optional[Dict] = Field(default=None, description="RAG智能体特定配置")
    
    # 专业化智能体配置
    specialized_config: Optional[Dict] = Field(default=None, description="专业化智能体配置")

# 对话响应模型
class ChatResponse(BaseModel):
    response: str
    thread_id: str
    agent_type: str
    agent_used: str  # 实际执行的智能体名称
    execution_time: float
    has_attachments: bool = False
    attachments: Optional[List[Dict]] = []
    
    # RAG特定响应字段
    retrieved_documents: Optional[List[Dict]] = None
    retrieval_score: Optional[float] = None
    
    # 多智能体协作响应字段
    agent_chain: Optional[List[str]] = None  # 智能体执行链
    
    # 专业化智能体响应字段
    specialized_output: Optional[Dict] = None

# 流式响应模型
class StreamChunk(BaseModel):
    type: str  # "message", "agent_switch", "tool_call", "retrieval", "error", "done"
    content: str
    metadata: Optional[Dict] = {}
    agent_type: Optional[str] = None
    agent_name: Optional[str] = None
    
    # RAG特定流式字段
    retrieved_docs: Optional[List[Dict]] = None
    
    # 多智能体特定流式字段
    handoff_info: Optional[Dict] = None
```

### 2.3 RAG相关模型
```python
# RAG文档模型
class DocumentModel(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    vector_indexed: bool = False

# RAG检索配置模型
class RetrievalConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    similarity_threshold: float = 0.7
    embedding_model: str = "text-embedding-ada-002"
    retrieval_strategy: str = "similarity"  # similarity, mmr, threshold

# 文档上传请求
class DocumentUploadRequest(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}
    auto_index: bool = True

# 向量搜索请求
class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.7
    filter_metadata: Optional[Dict[str, Any]] = None
```

### 2.4 智能体实例管理模型
```python
# 智能体实例模型
class AgentInstance(BaseModel):
    instance_id: str
    agent_type: str
    user_id: str
    config: Dict[str, Any]
    status: str  # active, inactive, error
    created_at: datetime
    last_used: datetime
    memory_namespace: Optional[str] = None

# 创建智能体实例请求
class CreateAgentInstanceRequest(BaseModel):
    agent_type: str
    user_id: str
    config: Optional[Dict[str, Any]] = {}
    memory_namespace: Optional[str] = None

# 智能体实例响应
class AgentInstanceResponse(BaseModel):
    instance_id: str
    agent_type: str
    status: str
    capabilities: List[str]
    tools: List[str]
    created_at: datetime
    last_used: datetime
```

### 2.5 智能体特定配置模型
```python
# 多智能体协作配置
class MultiAgentConfig(BaseModel):
    supervisor_model: str = "deepseek-chat"
    research_model: str = "deepseek-chat" 
    chart_model: str = "qwen-plus"
    max_retries: int = 3
    timeout: int = 300
    enable_memory: bool = True
    memory_limit: int = 10

# RAG智能体配置
class RAGAgentConfig(BaseModel):
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "deepseek-chat"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    similarity_threshold: float = 0.7
    enable_reranking: bool = False
    reranker_model: Optional[str] = None

# 代码智能体配置
class CodeAgentConfig(BaseModel):
    language: str = "python"
    execution_timeout: int = 30
    max_code_length: int = 10000
    enable_static_analysis: bool = True
    allowed_imports: List[str] = []
    security_level: str = "strict"  # strict, moderate, permissive

# 数据分析智能体配置
class DataAnalysisConfig(BaseModel):
    max_dataset_size: int = 1000000  # 最大数据集大小
    supported_formats: List[str] = ["csv", "json", "xlsx", "parquet"]
    enable_visualization: bool = True
    chart_types: List[str] = ["bar", "line", "scatter", "histogram", "heatmap"]
    statistical_tests: List[str] = ["t_test", "chi_square", "correlation"]
```

### 2.6 线程和记忆管理模型
```python
# 线程创建请求
class CreateThreadRequest(BaseModel):
    user_id: str
    agent_type: str
    initial_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

# 线程信息响应
class ThreadInfoResponse(BaseModel):
    thread_id: str
    user_id: str
    agent_type: str
    status: str  # active, paused, completed, error
    created_at: datetime
    last_activity: datetime
    message_count: int
    metadata: Dict[str, Any]

# 线程历史响应
class ThreadHistoryResponse(BaseModel):
    thread_id: str
    messages: List[Dict[str, Any]]
    checkpoints: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int

# 用户记忆模型
class UserMemory(BaseModel):
    user_id: str
    memory_id: str
    content: Dict[str, Any]
    memory_type: str  # personal, preference, context, knowledge
    created_at: datetime
    updated_at: datetime
    expiry_date: Optional[datetime] = None
    tags: List[str] = []

# 记忆查询请求
class MemoryQueryRequest(BaseModel):
    user_id: str
    memory_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    limit: int = 50
    offset: int = 0
```

### 2.7 工具管理模型
```python
# 工具信息模型
class ToolInfo(BaseModel):
    name: str
    description: str
    tool_type: str  # function, api, database, mcp
    parameters: Dict[str, Any]
    agent_types: List[str]  # 支持的智能体类型
    enabled: bool = True
    version: str = "1.0.0"

# 工具注册请求
class ToolRegistrationRequest(BaseModel):
    name: str
    description: str
    tool_type: str
    implementation: Dict[str, Any]  # 工具实现配置
    parameters_schema: Dict[str, Any]
    agent_types: List[str]

# 工具执行请求
class ToolExecutionRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = {}
    timeout: int = 30

# 工具执行响应
class ToolExecutionResponse(BaseModel):
    tool_name: str
    success: bool
    result: Any
    execution_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
```

### 2.8 系统配置模型
```python
# 系统配置模型
class SystemConfig(BaseModel):
    # 智能体配置
    default_agent_type: str = "multi_agent_supervisor"
    max_concurrent_agents: int = 100
    agent_timeout: int = 300
    
    # 向量存储配置
    vectorstore_type: str = "chroma"  # chroma, pinecone, weaviate
    vectorstore_config: Dict[str, Any] = {}
    
    # 文件存储配置
    file_storage_type: str = "local"  # local, s3, gcs
    file_storage_config: Dict[str, Any] = {}
    
    # 缓存配置
    cache_type: str = "redis"  # redis, memory
    cache_config: Dict[str, Any] = {}
    
    # 监控配置
    enable_monitoring: bool = True
    metrics_endpoint: Optional[str] = None
    log_level: str = "INFO"

# 健康检查响应
class HealthCheckResponse(BaseModel):
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    version: str
    components: Dict[str, str]  # 各组件状态
    uptime: float
    memory_usage: float
    cpu_usage: float
```

## 3. API端点实现示例

### 3.1 智能体管理端点
```python
from fastapi import APIRouter, HTTPException, Depends
from typing import List

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])

@router.get("/", response_model=List[AgentTypeInfo])
async def list_agent_types():
    """获取所有智能体类型"""
    agent_types = []
    for agent_type in agent_registry.list_agent_types():
        config = agent_registry.get_agent_config(agent_type)
        agent_types.append(AgentTypeInfo(
            type=agent_type,
            name=config.get("name", agent_type),
            description=config.get("description", ""),
            capabilities=config.get("capabilities", []),
            tools=config.get("tools", []),
            memory_enabled=config.get("memory_enabled", False),
            streaming_supported=config.get("streaming_supported", False),
            requires_vectorstore=config.get("requires_vectorstore", False)
        ))
    return agent_types

@router.get("/{agent_type}", response_model=AgentTypeInfo)
async def get_agent_type_info(agent_type: str):
    """获取特定智能体类型信息"""
    config = agent_registry.get_agent_config(agent_type)
    if not config:
        raise HTTPException(status_code=404, detail="智能体类型不存在")
    
    return AgentTypeInfo(
        type=agent_type,
        name=config.get("name", agent_type),
        description=config.get("description", ""),
        capabilities=config.get("capabilities", []),
        tools=config.get("tools", []),
        memory_enabled=config.get("memory_enabled", False),
        streaming_supported=config.get("streaming_supported", False),
        requires_vectorstore=config.get("requires_vectorstore", False),
        config_schema=config.get("config_schema", {})
    )

@router.post("/{agent_type}/chat", response_model=ChatResponse)
async def chat_with_agent(agent_type: str, request: ChatRequest):
    """与智能体对话"""
    try:
        agent = await agent_factory.create_agent(agent_type, request.agent_config)
        response = await agent.chat(
            message=request.message,
            thread_id=request.thread_id,
            user_id=request.user_id,
            context=request.context
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3.2 线程管理端点
```python
@router.post("/threads/", response_model=ThreadInfoResponse)
async def create_thread(request: CreateThreadRequest):
    """创建新会话线程"""
    thread_manager = ThreadManager(CheckpointManager().checkpointer)
    thread_id = await thread_manager.create_thread(
        user_id=request.user_id,
        agent_type=request.agent_type
    )
    
    return ThreadInfoResponse(
        thread_id=thread_id,
        user_id=request.user_id,
        agent_type=request.agent_type,
        status="active",
        created_at=datetime.now(),
        last_activity=datetime.now(),
        message_count=0,
        metadata=request.metadata or {}
    )

@router.get("/threads/{thread_id}/history", response_model=ThreadHistoryResponse)
async def get_thread_history(thread_id: str, page: int = 1, page_size: int = 50):
    """获取线程历史"""
    thread_manager = ThreadManager(CheckpointManager().checkpointer)
    history = await thread_manager.get_thread_history(thread_id, limit=page_size)
    
    return ThreadHistoryResponse(
        thread_id=thread_id,
        messages=history,
        checkpoints=[],
        total_count=len(history),
        page=page,
        page_size=page_size
    )
```