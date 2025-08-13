# 核心模块快速使用指南

本指南展示如何在你创建的智能体中使用项目的核心模块。

## 📁 核心模块概览

```
core/
├── memory/          # 长期记忆管理
├── tools/           # 工具管理和MCP集成
├── streaming/       # 流式处理
├── time_travel/     # 时间旅行和检查点
├── optimization/    # 提示词优化
├── workflows/       # 工作流构建
├── cache/          # 缓存管理
├── database/       # 数据库操作
├── error/          # 错误处理
├── interrupts/     # 中断处理
└── logging/        # 日志管理
```

## 🚀 快速开始

### 1. 基础导入模式

```python
# 安全导入模式 - 推荐
try:
    from core.memory import LangMemManager, MemoryType, MemoryScope
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

try:
    from core.tools import ToolRegistry, EnhancedToolManager
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
```

### 2. 在智能体中使用记忆模块

```python
from core.agents.memory_enhanced import MemoryEnhancedAgent

class MyAgent(MemoryEnhancedAgent):
    def __init__(self, agent_id: str, llm, **kwargs):
        super().__init__(
            agent_id=agent_id,
            llm=llm,
            memory_config={
                "auto_store": True,        # 自动存储对话
                "retrieval_limit": 10,     # 检索记忆数量
                "importance_threshold": 0.3 # 重要性阈值
            },
            **kwargs
        )
    
    async def store_custom_knowledge(self, content: str, user_id: str):
        """存储自定义知识"""
        await self.store_knowledge(
            content=content,
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,  # 语义记忆
            metadata={"category": "custom"},
            importance=0.8
        )
```

### 3. 使用工具管理器

```python
from core.tools.enhanced_tool_manager import get_enhanced_tool_manager
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """我的自定义工具"""
    return f"处理查询: {query}"

class MyAgent(BaseAgent):
    async def initialize(self):
        # 获取工具管理器
        self.tool_manager = get_enhanced_tool_manager()
        
        # 注册自定义工具
        await self.tool_manager.register_tool(
            my_custom_tool,
            metadata={"category": "custom", "agent_id": self.agent_id}
        )
        
        # 注册MCP工具
        mcp_count = await self.tool_manager.register_mcp_tools()
        print(f"注册了 {mcp_count} 个MCP工具")
```

### 4. 使用流式处理

```python
from core.streaming import get_stream_manager

class MyAgent(BaseAgent):
    async def stream_chat_enhanced(self, request):
        stream_manager = get_stream_manager()
        
        # 创建流式会话
        session = await stream_manager.create_stream_session(
            session_id=request.session_id,
            user_id=request.user_id,
            agent_id=self.agent_id
        )
        
        # 流式处理
        async for chunk in super().stream_chat(request):
            enhanced_chunk = await stream_manager.process_chunk(
                session.session_id, chunk
            )
            yield enhanced_chunk
```

### 5. 使用时间旅行功能

```python
from core.time_travel import get_time_travel_manager

class MyAgent(BaseAgent):
    async def chat_with_checkpoint(self, request):
        time_travel = get_time_travel_manager()
        
        # 创建检查点
        checkpoint_id = await time_travel.create_checkpoint(
            session_id=request.session_id,
            user_id=request.user_id,
            metadata={"type": "pre_chat"}
        )
        
        try:
            # 执行对话
            response = await super().chat(request)
            return response
        except Exception as e:
            # 出错时回滚
            await time_travel.rollback_to_checkpoint(
                checkpoint_id, request.session_id
            )
            raise e
```

### 6. 使用提示词优化

```python
from core.optimization import get_prompt_optimizer

class MyAgent(BaseAgent):
    async def chat(self, request):
        optimizer = get_prompt_optimizer()
        
        # 优化系统提示
        if request.messages and isinstance(request.messages[0], SystemMessage):
            optimized_prompt = await optimizer.optimize_prompt(
                prompt=request.messages[0].content,
                context={"agent_id": self.agent_id}
            )
            request.messages[0] = SystemMessage(content=optimized_prompt)
        
        return await super().chat(request)
```

## 🔧 完整集成示例

```python
from core.agents.memory_enhanced import MemoryEnhancedAgent
from core.tools.enhanced_tool_manager import get_enhanced_tool_manager
from core.streaming import get_stream_manager
from core.time_travel import get_time_travel_manager
from core.optimization import get_prompt_optimizer

class FullyIntegratedAgent(MemoryEnhancedAgent):
    def __init__(self, agent_id: str, llm, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name="完全集成智能体",
            llm=llm,
            memory_config={
                "auto_store": True,
                "retrieval_limit": 15,
                "importance_threshold": 0.3
            },
            **kwargs
        )
        
        # 核心模块管理器
        self.tool_manager = None
        self.stream_manager = None
        self.time_travel_manager = None
        self.prompt_optimizer = None
    
    async def initialize(self):
        """初始化所有核心模块"""
        await super().initialize()
        
        # 初始化各个管理器
        self.tool_manager = get_enhanced_tool_manager()
        self.stream_manager = get_stream_manager()
        self.time_travel_manager = get_time_travel_manager()
        self.prompt_optimizer = get_prompt_optimizer()
        
        # 注册工具
        for tool in self.tools:
            await self.tool_manager.register_tool(tool)
    
    async def enhanced_chat(self, request):
        """增强的对话处理"""
        # 1. 创建检查点
        checkpoint_id = await self.time_travel_manager.create_checkpoint(
            session_id=request.session_id,
            user_id=request.user_id,
            agent_id=self.agent_id
        )
        
        try:
            # 2. 优化提示词
            request = await self._optimize_request(request)
            
            # 3. 执行对话（包含记忆增强）
            response = await super().chat(request)
            
            return response
            
        except Exception as e:
            # 4. 错误恢复
            await self.time_travel_manager.rollback_to_checkpoint(
                checkpoint_id, request.session_id
            )
            raise e
    
    async def _optimize_request(self, request):
        """优化请求"""
        # 实现提示词优化逻辑
        return request
```

## 📝 使用建议

### 1. 模块选择
- **基础智能体**: 继承 `BaseAgent`
- **需要记忆**: 继承 `MemoryEnhancedAgent`
- **复杂功能**: 按需集成核心模块

### 2. 错误处理
```python
# 总是使用try-except包装核心模块调用
try:
    result = await self.tool_manager.execute_tool(tool_name, input_data)
except Exception as e:
    logger.warning(f"工具执行失败: {e}")
    # 降级处理
```

### 3. 性能考虑
- 只导入需要的模块
- 使用异步方法
- 适当设置缓存和限制

### 4. 配置管理
```python
# 在智能体配置中定义模块开关
config = {
    "memory_enabled": True,
    "tools_enabled": True,
    "streaming_enabled": False,
    "optimization_enabled": True
}
```

## 🔍 调试和监控

```python
async def get_agent_status(self):
    """获取智能体状态"""
    return {
        "agent_id": self.agent_id,
        "memory_stats": await self.get_memory_stats() if hasattr(self, 'memory_manager') else None,
        "tool_stats": await self.tool_manager.get_execution_stats() if self.tool_manager else None,
        "features_enabled": {
            "memory": hasattr(self, 'memory_manager'),
            "tools": self.tool_manager is not None,
            "streaming": self.stream_manager is not None
        }
    }
```

## 📚 更多资源

- **完整示例**: `core/agents/examples/integrated_agent_example.py`
- **模板文件**: `new_agent_template.py`
- **详细文档**: `HOW_TO_CREATE_NEW_AGENTS.md`

---

💡 **提示**: 从简单开始，逐步添加需要的核心模块功能。每个模块都有完善的错误处理和降级机制。