# 新智能体创建指南

本文档详细说明了在LangGraph多智能体系统中创建新智能体的位置和方法。

## 📍 创建位置

### 主要位置：`core/agents/` 目录

```
core/agents/
├── __init__.py              # 智能体模块导入
├── base.py                  # 基础智能体类
├── memory_enhanced.py       # 记忆增强智能体类
├── registry.py              # 智能体注册和管理
├── state.py                 # 智能体状态定义
├── templates/               # 智能体模板目录
│   └── new_agent_template.py # 新智能体创建模板
└── [你的新智能体].py        # 在这里创建新智能体
```

## 🚀 两种创建方式

### 方式一：基于 BaseAgent（简单智能体）

适用于：
- 不需要记忆功能的智能体
- 简单的任务处理智能体
- 工具调用型智能体

```python
from core.agents.base import BaseAgent

class MySimpleAgent(BaseAgent):
    def __init__(self, agent_id: str, llm, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name="我的简单智能体",
            description="智能体描述",
            llm=llm,
            tools=[my_tool1, my_tool2],
            **kwargs
        )
```

### 方式二：基于 MemoryEnhancedAgent（记忆增强智能体）

适用于：
- 需要记住用户偏好和历史的智能体
- 学习型智能体
- 个性化服务智能体

```python
from core.agents.memory_enhanced import MemoryEnhancedAgent

class MySmartAgent(MemoryEnhancedAgent):
    def __init__(self, agent_id: str, llm, **kwargs):
        memory_config = {
            "auto_store": True,
            "retrieval_limit": 8,
            "importance_threshold": 0.4
        }
        
        super().__init__(
            agent_id=agent_id,
            name="我的智能体",
            description="具有记忆功能的智能体",
            llm=llm,
            tools=[my_tool1, my_tool2],
            memory_config=memory_config,
            **kwargs
        )
```

## 📋 创建步骤

### 1. 使用模板快速开始

```bash
# 复制模板文件
cp core/agents/templates/new_agent_template.py core/agents/my_new_agent.py
```

### 2. 修改模板内容

1. **修改类名和基本信息**
   ```python
   class MyNewAgent(BaseAgent):  # 修改类名
       def __init__(self, agent_id: str, llm, **kwargs):
           super().__init__(
               agent_id=agent_id,
               name="我的新智能体",        # 修改名称
               description="智能体描述",   # 修改描述
               llm=llm,
               tools=[],                  # 添加工具
               **kwargs
           )
   ```

2. **定义专用工具**
   ```python
   @tool
   def my_custom_tool(input_param: str) -> str:
       """工具描述"""
       # 工具实现逻辑
       return "处理结果"
   ```

3. **构建处理图**
   ```python
   def _build_graph(self) -> StateGraph:
       graph = StateGraph(AgentState)
       
       # 添加节点
       graph.add_node("node1", self._node1_handler)
       graph.add_node("node2", self._node2_handler)
       
       # 定义流程
       graph.set_entry_point("node1")
       graph.add_edge("node1", "node2")
       graph.add_edge("node2", END)
       
       return graph
   ```

4. **实现处理逻辑**
   ```python
   async def _node1_handler(self, state: Dict[str, Any]) -> Dict[str, Any]:
       # 节点处理逻辑
       return state
   ```

### 3. 注册智能体

在 `core/agents/registry.py` 中注册：

```python
# 在 AgentRegistry.__init__ 中添加
self.register_agent_class("my_new_agent", MyNewAgent)

# 在 DEFAULT_AGENT_CONFIGS 中添加配置
"my_new_agent": AgentConfig(
    agent_type="my_new_agent",
    name="我的新智能体",
    description="智能体描述",
    llm_config={"provider": "qwen", "model": "qwen-plus"},
    tools=["my_custom_tool"],
    capabilities=["custom_capability"]
)
```

### 4. 更新模块导入

在 `core/agents/__init__.py` 中添加：

```python
from .my_new_agent import MyNewAgent

__all__ = [
    # ... 其他导入
    "MyNewAgent",
]
```

## 🛠️ 完整示例

查看以下文件获取完整示例：

1. **`create_new_agent_guide.py`** - 完整的创建指南和示例
2. **`create_custom_agent_demo.py`** - 自定义智能体演示
3. **`core/agents/templates/new_agent_template.py`** - 快速创建模板

## 📚 核心概念

### 智能体状态 (AgentState)

```python
from core.agents.state import AgentState

# 状态包含：
# - messages: 消息列表
# - user_id: 用户ID
# - session_id: 会话ID
# - metadata: 元数据
# - 其他自定义字段
```

### 工具定义

```python
from langchain_core.tools import tool

@tool
def my_tool(param1: str, param2: int = 10) -> str:
    """工具描述
    
    Args:
        param1: 参数1描述
        param2: 参数2描述，默认值10
        
    Returns:
        str: 返回结果描述
    """
    # 工具实现
    return f"结果: {param1}, {param2}"
```

### 记忆管理（仅MemoryEnhancedAgent）

```python
# 存储知识
await self.store_knowledge(
    content="重要信息",
    user_id="user123",
    memory_type=MemoryType.SEMANTIC,
    importance=0.8
)

# 获取记忆统计
stats = await self.get_memory_stats("user123", "session123")
```

## 🔧 测试智能体

```python
async def test_my_agent():
    from core.agents.registry import AgentFactory
    from config.settings import get_llm_by_name
    
    # 创建智能体
    factory = AgentFactory()
    agent = await factory.create_agent("test_001", config)
    
    # 测试对话
    request = ChatRequest(
        messages=[HumanMessage(content="测试消息")],
        user_id="test_user",
        session_id="test_session"
    )
    
    response = await agent.chat(request)
    print(response.message.content)
```

## 📖 最佳实践

1. **命名规范**
   - 类名使用 PascalCase：`MyCustomAgent`
   - 文件名使用 snake_case：`my_custom_agent.py`
   - 智能体类型使用 snake_case：`"my_custom"`

2. **工具设计**
   - 每个工具功能单一明确
   - 提供详细的文档字符串
   - 合理设置参数类型和默认值

3. **图设计**
   - 节点职责清晰
   - 流程逻辑简洁
   - 适当使用条件边

4. **错误处理**
   - 在关键节点添加异常处理
   - 提供降级方案
   - 记录详细日志

5. **性能优化**
   - 避免不必要的LLM调用
   - 合理使用缓存
   - 异步处理IO操作

## 🚀 快速开始

1. 复制模板文件
2. 修改类名和基本信息
3. 定义工具和处理逻辑
4. 注册智能体
5. 测试功能

现在你可以开始创建自己的智能体了！🎉