# Models目录使用指南

`/models` 目录包含了整个系统的API数据模型定义，基于Pydantic构建，提供数据验证、序列化和类型安全功能。

## 📁 目录结构

```
models/
├── __init__.py           # 模型模块导出
├── base_models.py        # 基础响应模型
├── chat_models.py        # 聊天API模型
├── agent_models.py       # 智能体API模型
├── memory_models.py      # 记忆管理API模型
└── rag_models.py         # RAG系统API模型
```

## 🎯 主要功能

### 1. 数据验证和类型检查
- 自动验证输入数据格式
- 类型转换和约束检查
- 字段必填性验证

### 2. 序列化和反序列化
- JSON格式自动转换
- 字典格式支持
- 时间戳格式化

### 3. API标准化
- 统一的请求/响应格式
- 错误处理标准化
- 分页响应支持

## 📋 各模块详解

### base_models.py - 基础模型

**核心模型:**
- `BaseResponse` - 基础响应模型
- `SuccessResponse` - 成功响应
- `ErrorResponse` - 错误响应
- `PaginatedResponse` - 分页响应
- `HealthCheckResponse` - 健康检查响应

**使用示例:**
```python
from models.base_models import SuccessResponse, ErrorResponse

# 成功响应
success = SuccessResponse(
    message="操作成功",
    data={"result": "处理完成"},
    request_id="req_123"
)

# 错误响应
error = ErrorResponse(
    message="操作失败",
    error="参数验证错误"
)
```

### chat_models.py - 聊天模型

**核心模型:**
- `ChatRequest` - 聊天请求
- `ChatResponse` - 聊天响应
- `Message` - 消息模型
- `StreamChunk` - 流式响应块
- `ThreadInfo` - 会话线程信息

**使用示例:**
```python
from models.chat_models import ChatRequest, Message, MessageRole

# 聊天请求
request = ChatRequest(
    message="你好，请帮我分析数据",
    agent_id="agent_001",
    stream=True,
    temperature=0.7
)

# 消息模型
message = Message(
    role=MessageRole.ASSISTANT,
    content="我来帮您分析数据...",
    metadata={"confidence": 0.95}
)
```

### agent_models.py - 智能体模型

**核心模型:**
- `AgentInfo` - 智能体信息
- `CreateAgentRequest` - 创建智能体请求
- `AgentConfig` - 智能体配置
- `AgentInstanceResponse` - 实例响应

**使用示例:**
```python
from models.agent_models import CreateAgentRequest, AgentType, AgentConfig

# 智能体配置
config = AgentConfig(
    model="gpt-4-turbo",
    temperature=0.7,
    system_prompt="你是一个专业助手"
)

# 创建请求
request = CreateAgentRequest(
    name="数据分析师",
    type=AgentType.RAG,
    config=config
)
```

### memory_models.py - 记忆模型

**核心模型:**
- `MemoryItem` - 记忆项
- `MemoryCreateRequest` - 创建记忆请求
- `MemorySearchRequest` - 搜索记忆请求
- `MemorySearchResponse` - 搜索响应

**使用示例:**
```python
from models.memory_models import MemoryCreateRequest, MemoryType, MemoryImportance

# 创建记忆
memory_request = MemoryCreateRequest(
    content="用户偏好详细分析",
    type=MemoryType.LONG_TERM,
    importance=MemoryImportance.HIGH,
    agent_id="agent_001",
    tags=["用户偏好", "分析风格"]
)
```

### rag_models.py - RAG模型

**核心模型:**
- `DocumentModel` - 文档模型
- `DocumentUploadRequest` - 文档上传请求
- `RAGRequest` - RAG查询请求
- `VectorSearchRequest` - 向量搜索请求

**使用示例:**
```python
from models.rag_models import DocumentUploadRequest, DocumentType, RAGRequest

# 文档上传
upload_request = DocumentUploadRequest(
    title="市场分析报告",
    content="详细的市场数据...",
    type=DocumentType.PDF,
    tags=["市场", "分析"]
)

# RAG查询
rag_request = RAGRequest(
    query="市场趋势如何？",
    collection_name="reports",
    include_sources=True
)
```

## 🔧 在API中的使用

### FastAPI路由示例

```python
from fastapi import APIRouter, HTTPException
from models.chat_models import ChatRequest, ChatResponse
from models.base_models import BaseResponse

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """聊天API端点"""
    try:
        # 自动验证请求数据
        # request.message, request.agent_id 等字段已验证
        
        # 处理聊天逻辑
        response_content = await process_chat(request)
        
        # 返回标准化响应
        return ChatResponse(
            message=response_content,
            thread_id=request.thread_id or "default",
            agent_id=request.agent_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=BaseResponse)
async def health_endpoint():
    """健康检查端点"""
    return BaseResponse(
        success=True,
        message="系统运行正常",
        data={"status": "healthy"}
    )
```

### 数据验证示例

```python
from models.chat_models import ChatRequest
from pydantic import ValidationError

try:
    # 正确的数据
    valid_request = ChatRequest(
        message="测试消息",
        agent_id="agent_001",
        temperature=0.5  # 有效范围 0.0-2.0
    )
    print("验证成功")
except ValidationError as e:
    print(f"验证失败: {e}")

try:
    # 错误的数据
    invalid_request = ChatRequest(
        message="测试消息",
        agent_id="agent_001",
        temperature=3.0  # 超出有效范围
    )
except ValidationError as e:
    print(f"预期的验证错误: {e}")
```

## 💡 最佳实践

### 1. 字段约束设置
```python
from pydantic import BaseModel, Field

class MyModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="名称")
    age: int = Field(..., ge=0, le=150, description="年龄")
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$', description="邮箱")
```

### 2. 枚举类型使用
```python
from enum import Enum

class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

class MyModel(BaseModel):
    status: Status = Field(default=Status.PENDING)
```

### 3. 可选字段和默认值
```python
class MyModel(BaseModel):
    required_field: str
    optional_field: Optional[str] = None
    default_field: str = "default_value"
    list_field: List[str] = Field(default_factory=list)
```

### 4. 嵌套模型
```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    address: Address
    addresses: List[Address] = Field(default_factory=list)
```

## 🔄 序列化和反序列化

### JSON操作
```python
from models.chat_models import ChatRequest

# 创建模型实例
request = ChatRequest(message="测试", agent_id="agent_001")

# 序列化为JSON
json_str = request.model_dump_json()
print(json_str)

# 从JSON反序列化
parsed_request = ChatRequest.model_validate_json(json_str)
print(parsed_request.message)

# 转换为字典
dict_data = request.model_dump()
print(dict_data)

# 从字典创建
new_request = ChatRequest(**dict_data)
```

### 自定义序列化
```python
class MyModel(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

## 🚀 扩展和自定义

### 1. 继承基础模型
```python
from models.base_models import BaseResponse

class CustomResponse(BaseResponse):
    custom_field: str
    extra_data: Dict[str, Any] = Field(default_factory=dict)
```

### 2. 自定义验证器
```python
from pydantic import validator

class MyModel(BaseModel):
    email: str
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('无效的邮箱格式')
        return v.lower()
```

### 3. 根验证器
```python
from pydantic import root_validator

class MyModel(BaseModel):
    start_date: datetime
    end_date: datetime
    
    @root_validator
    def validate_dates(cls, values):
        start = values.get('start_date')
        end = values.get('end_date')
        if start and end and start >= end:
            raise ValueError('开始时间必须早于结束时间')
        return values
```

## 📖 运行演示

要查看完整的使用示例，可以运行演示脚本：

```bash
cd /Users/cykk/local/langchain-study/langgraph_study
python examples/models_usage_demo.py
```

这个脚本展示了：
- 各种模型的创建和使用
- 数据验证功能
- 序列化和反序列化
- API中的实际应用

## 🔗 相关文档

- [Pydantic官方文档](https://docs.pydantic.dev/)
- [FastAPI数据模型](https://fastapi.tiangolo.com/tutorial/body/)
- [项目API设计文档](./spec/02_api_design.md)

## 📝 注意事项

1. **版本兼容性**: 项目使用Pydantic v2，注意方法名变更（如`dict()` → `model_dump()`）
2. **性能考虑**: 复杂模型验证可能影响性能，合理设置验证规则
3. **错误处理**: 捕获`ValidationError`异常并提供友好的错误信息
4. **文档生成**: 使用`Field(description=...)`为字段添加描述，支持自动API文档生成