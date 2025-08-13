# 智能体使用核心模块指南

本指南展示如何在创建的智能体中使用项目的核心模块。

## 核心模块概览

### 1. core/memory - 长期记忆管理
- **功能**: 提供语义、情节、程序记忆存储和检索
- **主要组件**: `LangMemManager`, `MemoryNamespace`, `MemoryItem`
- **使用场景**: 智能体需要记住用户偏好、历史对话、学习内容

### 2. core/tools - 工具管理和MCP集成
- **功能**: 工具注册、执行监控、权限管理、MCP工具集成
- **主要组件**: `EnhancedToolManager`, `ToolRegistry`, `BaseManagedTool`
- **使用场景**: 智能体需要使用外部工具、API调用、并发执行

### 3. core/streaming - 流式处理
- **功能**: 实时流式响应处理和管理
- **主要组件**: `StreamManager`, `StreamChunk`
- **使用场景**: 需要实时响应的对话场景

### 4. core/time_travel - 时间旅行功能
- **功能**: 对话状态检查点、回滚、历史管理
- **主要组件**: `TimeTravelManager`, `CheckpointManager`
- **使用场景**: 需要撤销操作、状态恢复的场景

### 5. core/optimization - 提示词优化
- **功能**: 自动优化提示词以提高响应质量
- **主要组件**: `PromptOptimizer`
- **使用场景**: 需要动态优化对话质量的场景

## 快速集成示例

### 1. 集成记忆功能

```python
from core.memory import get_memory_manager, MemoryNamespace, MemoryScope, MemoryType, MemoryItem
from core.memory.tools import get_memory_tools

class MyMemoryAgent(MemoryEnhancedAgent):
    async def initialize(self):
        await super().initialize()
        
        # 获取记忆管理器
        self.memory_manager = get_memory_manager()
        
        # 添加记忆工具
        memory_tools = await get_memory_tools(f"agent_{self.agent_id}")
        self.tools.extend(memory_tools)
    
    async def store_custom_knowledge(self, content: str, user_id: str, category: str):
        """存储自定义知识"""
        namespace = MemoryNamespace(
            scope=MemoryScope.USER,
            identifier=user_id,
            sub_namespace=category
        )
        
        memory_item = MemoryItem(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            metadata={"category": category, "agent_id": self.agent_id},
            importance=0.7
        )
        
        return await self.memory_manager.store_memory(namespace, memory_item)
```

### 2. 集成工具管理

```python
from core.tools.enhanced_tool_manager import get_enhanced_tool_manager, ToolExecutionContext
from core.tools.mcp_manager import get_mcp_manager

class MyToolAgent(BaseAgent):
    async def initialize(self):
        await super().initialize()
        
        # 获取增强工具管理器
        self.enhanced_tool_manager = get_enhanced_tool_manager()
        
        # 注册自定义工具
        for tool in self.tools:
            await self.enhanced_tool_manager.register_tool(
                tool,
                metadata={"category": "custom", "agent_id": self.agent_id}
            )
        
        # 注册MCP工具
        await self.enhanced_tool_manager.register_mcp_tools()
    
    async def execute_tool_safely(self, tool_name: str, tool_input: dict, user_id: str):
        """安全执行工具"""
        context = ToolExecutionContext(
            user_id=user_id,
            agent_id=self.agent_id,
            execution_id=str(uuid.uuid4())
        )
        
        return await self.enhanced_tool_manager.execute_tool(
            tool_name, tool_input, context
        )
```

### 3. 集成流式处理

```python
from core.streaming import get_stream_manager

class MyStreamingAgent(BaseAgent):
    async def initialize(self):
        await super().initialize()
        self.stream_manager = get_stream_manager()
    
    async def stream_chat_enhanced(self, request: ChatRequest):
        """增强的流式对话"""
        # 创建流式会话
        stream_session = await self.stream_manager.create_stream_session(
            session_id=request.session_id,
            user_id=request.user_id,
            agent_id=self.agent_id
        )
        
        # 流式处理
        async for chunk in super().stream_chat(request):
            enhanced_chunk = await self.stream_manager.process_chunk(
                stream_session.session_id, chunk
            )
            yield enhanced_chunk
        
        # 结束会话
        await self.stream_manager.end_stream_session(stream_session.session_id)
```

### 4. 集成时间旅行

```python
from core.time_travel import get_time_travel_manager, CheckpointManager

class MyTimeTravelAgent(BaseAgent):
    async def initialize(self):
        await super().initialize()
        self.time_travel_manager = get_time_travel_manager()
        self.checkpoint_manager = CheckpointManager()
    
    async def chat_with_checkpoints(self, request: ChatRequest):
        """带检查点的对话"""
        # 创建检查点
        checkpoint_id = await self.checkpoint_manager.create_checkpoint(
            session_id=request.session_id,
            user_id=request.user_id,
            agent_id=self.agent_id
        )
        
        try:
            # 执行对话
            response = await super().chat(request)
            return response
        except Exception as e:
            # 出错时回滚
            await self.time_travel_manager.rollback_to_checkpoint(
                checkpoint_id, request.session_id
            )
            raise
```

### 5. 集成提示词优化

```python
from core.optimization import get_prompt_optimizer

class MyOptimizedAgent(BaseAgent):
    async def initialize(self):
        await super().initialize()
        self.prompt_optimizer = get_prompt_optimizer()
    
    async def chat_with_optimization(self, request: ChatRequest):
        """带优化的对话"""
        # 优化系统提示
        if request.messages and isinstance(request.messages[0], SystemMessage):
            optimized_prompt = await self.prompt_optimizer.optimize_prompt(
                prompt=request.messages[0].content,
                context={"agent_id": self.agent_id}
            )
            
            # 替换优化后的提示
            request.messages[0] = SystemMessage(content=optimized_prompt)
        
        return await super().chat(request)
```

## 全功能集成示例

```python
class FullyIntegratedAgent(MemoryEnhancedAgent):
    """集成所有核心模块的智能体"""
    
    def __init__(self, agent_id: str, llm, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name="全功能智能体",
            llm=llm,
            memory_config={
                "auto_store": True,
                "retrieval_limit": 10,
                "importance_threshold": 0.3
            },
            **kwargs
        )
    
    async def initialize(self):
        """初始化所有核心模块"""
        await super().initialize()
        
        # 初始化各个管理器
        self.enhanced_tool_manager = get_enhanced_tool_manager()
        self.stream_manager = get_stream_manager()
        self.time_travel_manager = get_time_travel_manager()
        self.prompt_optimizer = get_prompt_optimizer()
        
        # 注册工具
        for tool in self.tools:
            await self.enhanced_tool_manager.register_tool(tool)
    
    async def enhanced_chat(
        self,
        request: ChatRequest,
        enable_optimization: bool = True,
        create_checkpoint: bool = True
    ) -> ChatResponse:
        """增强的对话处理"""
        checkpoint_id = None
        
        try:
            # 1. 创建检查点
            if create_checkpoint:
                checkpoint_id = await self.time_travel_manager.create_checkpoint(
                    session_id=request.session_id,
                    user_id=request.user_id,
                    agent_id=self.agent_id
                )
            
            # 2. 优化提示词
            if enable_optimization:
                request = await self._optimize_request(request)
            
            # 3. 执行对话（自动使用记忆增强）
            response = await super().chat(request)
            
            return response
            
        except Exception as e:
            # 错误恢复
            if checkpoint_id:
                await self.time_travel_manager.rollback_to_checkpoint(
                    checkpoint_id, request.session_id
                )
            raise
    
    async def _optimize_request(self, request: ChatRequest) -> ChatRequest:
        """优化请求"""
        # 实现提示词优化逻辑
        return request
```

## 使用建议

### 1. 模块选择
- **简单任务**: 只使用 `core/tools`
- **需要记忆**: 使用 `core/memory` + `MemoryEnhancedAgent`
- **实时响应**: 添加 `core/streaming`
- **容错需求**: 添加 `core/time_travel`
- **质量优化**: 添加 `core/optimization`

### 2. 性能考虑
- 记忆功能会增加响应时间，适合长期对话
- 工具管理支持并发执行，适合复杂任务
- 流式处理适合长响应场景
- 检查点功能有存储开销，按需使用

### 3. 错误处理
- 所有模块都支持降级处理
- 建议在 `initialize()` 方法中处理模块初始化失败
- 使用 try-catch 包装核心功能调用

### 4. 最佳实践
- 在智能体的 `initialize()` 方法中初始化所需模块
- 使用配置参数控制模块功能的开启/关闭
- 为不同场景创建不同的智能体类
- 定期清理记忆和检查点以控制存储使用

## 完整示例

参考 `core/agents/templates/create_new_agent_guide.py` 文件中的完整示例代码，包含：

1. **MemoryIntegratedAgent** - 记忆集成示例
2. **ToolIntegratedAgent** - 工具管理示例  
3. **StreamingIntegratedAgent** - 流式处理示例
4. **TimeTravelIntegratedAgent** - 时间旅行示例
5. **OptimizationIntegratedAgent** - 优化集成示例
6. **FullyIntegratedAgent** - 全功能集成示例

每个示例都包含完整的初始化、使用和错误处理代码。