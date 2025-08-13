# 🤖 LangGraph Multi-Agent System

基于LangGraph的企业级多智能体协作系统，集成了LangMem长期记忆管理、MCP协议支持、复杂工作流编排、时间旅行功能和完整的API服务架构。

## ✨ 核心功能特性

### 🤖 多智能体协作架构
- **多智能体协作系统** (graph5.py) - Supervisor、Research、Chart智能体协作
- **Agentic RAG系统** (graph6.py) - 智能检索增强生成
- **专业化智能体系统** (graph7.py+) - 代码、数据分析、内容创作等专业智能体
- **统一智能体抽象** - 基于BaseAgent的模块化设计，支持动态加载
- **协作优化器** - 智能任务分配和负载均衡

### 🧠 智能记忆管理 (LangMem集成)
- **语义记忆** - 事实和知识的长期存储
- **情节记忆** - 过去经验和情境的记录
- **程序记忆** - 行为模式和响应模板
- **跨会话持久化** - 基于PostgreSQL的向量存储和语义搜索
- **命名空间隔离** - 用户级、智能体级、组织级记忆管理
- **🆕 提示词优化** - 基于用户反馈自动优化智能体提示词
  - **单智能体优化** - 基于对话历史和用户反馈改进个体提示词
  - **多智能体协同优化** - 优化整个智能体团队的协作效果
  - **持续学习** - 从用户满意度和反馈中持续改进
  - **A/B测试** - 支持提示词版本管理和效果对比
  - **自动化优化** - 定期自动执行提示词优化流程

### 🔄 复杂工作流编排
- **子图管理** - 支持嵌套工作流和模块化执行
- **条件路由** - 基于动态条件的智能分支
- **并行执行** - 多任务并发处理和结果聚合
- **工作流模板** - 可复用的工作流定义和实例化
- **人工干预** - 支持审批流程和人工决策点

### ⏰ 时间旅行功能
- **状态快照** - 执行过程的完整状态捕获
- **检查点管理** - 关键节点的状态保存和恢复
- **回滚机制** - 支持任意时间点的状态回退
- **分支管理** - 多分支执行和合并策略
- **历史追踪** - 完整的执行历史和时间线

### 🔧 工具与协议集成
- **MCP协议支持** - Model Context Protocol外部工具集成
- **丰富工具集** - 搜索、数据分析、可视化、代码执行等
- **动态工具加载** - 运行时工具注册和管理
- **工具缓存** - 智能缓存机制提升性能

### 🌐 企业级API架构
- **RESTful API** - 完整的REST API设计规范
- **流式响应** - WebSocket和SSE实时通信
- **状态管理** - 基于LangGraph的检查点系统
- **错误处理** - 完善的异常处理和恢复机制
- **性能监控** - 指标收集、日志记录、健康检查

### 🎨 前端开发支持
- **React前端方案** - 基于React 18 + TypeScript + Vite的现代化前端架构
- **完整UI组件** - 智能体聊天、记忆管理、工作流可视化、时间旅行等核心功能组件
- **实时通信** - WebSocket和SSE集成，支持流式输出和实时状态更新
- **状态管理** - Zustand + React Query的高效状态管理方案
- **响应式设计** - 支持桌面端和移动端的自适应界面
- **开发工具链** - 完整的开发、测试、构建和部署工具链

## 🛠️ 技术栈

### 核心框架
- **LangGraph** - 多智能体工作流编排
- **LangChain** - 大语言模型应用框架
- **FastAPI** - 高性能Web API框架
- **Pydantic** - 数据验证和序列化

### 记忆和存储
- **LangMem** - 向量记忆管理系统
- **PostgreSQL** - 主数据库
- **pgvector** - 向量数据库扩展
- **Redis** - 缓存和会话存储

### 工具集成
- **MCP (Model Context Protocol)** - 工具协议标准
- **WebSocket** - 实时通信
- **SSE (Server-Sent Events)** - 事件流

### 前端技术栈
- **React 18** - 现代化前端框架
- **TypeScript** - 类型安全的JavaScript
- **Vite** - 快速构建工具
- **Zustand** - 轻量级状态管理
- **React Query** - 服务端状态管理
- **Ant Design** - 企业级UI组件库
- **Tailwind CSS** - 实用优先的CSS框架
- **React Flow** - 工作流可视化
- **Socket.IO Client** - 实时通信客户端

### 开发工具
- **Python 3.11+** - 主要开发语言
- **asyncio** - 异步编程支持
- **pytest** - 测试框架

## 📊 项目进展状态

### 整体完成度: ~80%

- 🟢 **基础设施层 (95%完成)** - 环境配置、数据库、Redis缓存、核心配置
- 🟢 **核心抽象层 (90%完成)** - BaseAgent、CheckpointManager、MemoryEnhanced、完整模块注册
- 🟢 **数据模型层 (85%完成)** - API模型定义完善，包含所有核心模块
- 🟢 **智能体实现层 (80%完成)** - Graph5多智能体系统、智能体注册表、协作优化
- 🟢 **API服务层 (75%完成)** - FastAPI框架、MCP API、核心路由
- 🟢 **流式处理层 (95%完成)** - WebSocket、SSE、流管理器、事件处理、LangGraph适配器
- 🟡 **工作流编排层 (60%完成)** - 工作流构建器、条件路由、并行执行
- 🟡 **时间旅行层 (55%完成)** - 检查点管理、状态历史、回滚机制
- 🟡 **监控日志层 (70%完成)** - 错误处理、性能监控、结构化日志
- 🟡 **缓存管理层 (80%完成)** - Redis管理器、智能缓存、连接池
- 🟡 **人工干预层 (85%完成)** - 增强中断管理器、审批工作流、人工决策
- 🟢 **🆕 前端开发指南 (100%完成)** - React前端开发完整方案、技术栈选择、项目架构
- 🔴 **部署运维层 (25%完成)** - Docker配置，K8s待实现

## 📖 使用指南

### 快速开始

#### 后端服务启动
```bash
# 启动后端API服务
python main.py
```

启动后访问：
- **主页**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **系统状态**: http://localhost:8000/status

#### 🆕 前端开发
```bash
# 查看前端开发指南
cat REACT_FRONTEND_DEVELOPMENT_GUIDE.md

# 创建React前端项目
npm create vite@latest langgraph-frontend -- --template react-ts
cd langgraph-frontend

# 安装依赖
npm install zustand @tanstack/react-query antd tailwindcss
npm install @types/react @types/react-dom
npm install socket.io-client axios react-router-dom

# 启动开发服务器
npm run dev
```

前端开发服务启动后访问：
- **前端应用**: http://localhost:5173
- **开发工具**: 浏览器开发者工具

### 核心API接口

#### 多智能体协作
```bash
# Supervisor智能体对话
curl -X POST "http://localhost:8000/api/v1/multi-agent/supervisor/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "请帮我分析苹果公司的股价并生成图表",
    "user_id": "user123",
    "thread_id": "thread_456"
  }'

# 流式响应
curl -X POST "http://localhost:8000/api/v1/agents/supervisor/stream" \
  -H "Content-Type: application/json" \
  -d '{"content": "分析市场趋势", "user_id": "user123"}'
```

#### Agentic RAG系统
```bash
# 上传文档
curl -X POST "http://localhost:8000/api/v1/rag/documents/upload" \
  -F "file=@document.pdf" \
  -F "user_id=user123"

# RAG对话
curl -X POST "http://localhost:8000/api/v1/rag/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "根据上传的文档回答问题",
    "user_id": "user123"
  }'
```

#### 工作流管理
```bash
# 创建工作流
curl -X POST "http://localhost:8000/api/v1/workflows/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "数据分析工作流",
    "steps": [...],
    "user_id": "user123"
  }'

# 执行工作流
curl -X POST "http://localhost:8000/api/v1/workflows/{workflow_id}/execute" \
  -H "Content-Type: application/json" \
  -d '{"input_data": {...}}'
```

#### 记忆管理
```bash
# 搜索记忆
curl -X GET "http://localhost:8000/api/v1/memory/search?query=苹果股价&user_id=user123"

# 管理记忆
curl -X POST "http://localhost:8000/api/v1/memory/manage" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "create",
    "content": "重要的市场分析结果",
    "user_id": "user123"
  }'
```

#### 🆕 提示词优化
```bash
# 收集用户反馈
curl -X POST "http://localhost:8000/api/v1/optimization/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "satisfaction_score": 0.8,
    "feedback_text": "回答很好，但希望有更多例子",
    "agent_type": "technical_assistant"
  }'

# 优化单个智能体提示词
curl -X POST "http://localhost:8000/api/v1/optimization/agents/technical_assistant/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "current_prompt": "你是一个技术助手...",
    "min_feedback_count": 10
  }'

# 优化多智能体系统
curl -X POST "http://localhost:8000/api/v1/optimization/multi-agent/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_prompts": [
      {"name": "researcher", "prompt": "你是研究员..."},
      {"name": "writer", "prompt": "你是写作专家..."}
    ]
  }'

# 获取优化历史
curl -X GET "http://localhost:8000/api/v1/optimization/history?agent_type=technical_assistant"

# 启动自动优化
curl -X POST "http://localhost:8000/api/v1/optimization/auto-optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_configs": {
      "technical_assistant": "当前提示词...",
      "creative_writer": "当前提示词..."
    }
  }'
```

#### MCP工具集成
```bash
# 获取MCP服务器列表
curl -X GET "http://localhost:8000/api/v1/tools/mcp/servers"

# 调用MCP工具
curl -X POST "http://localhost:8000/api/v1/tools/mcp/tools/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "filesystem_read",
    "arguments": {"path": "/path/to/file"}
  }'
```

#### 时间旅行功能
```bash
# 获取时间旅行配置
curl -X GET "http://localhost:8000/api/v1/time-travel/config"

# 创建状态快照
curl -X POST "http://localhost:8000/api/v1/time-travel/snapshots" \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "thread_123",
    "description": "重要决策点快照",
    "snapshot_type": "manual"
  }'

# 获取快照列表
curl -X GET "http://localhost:8000/api/v1/time-travel/snapshots?thread_id=thread_123"

# 创建检查点
curl -X POST "http://localhost:8000/api/v1/time-travel/checkpoints" \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "thread_123",
    "name": "数据分析完成",
    "description": "完成数据分析阶段",
    "checkpoint_type": "milestone"
  }'

# 回滚到指定快照
curl -X POST "http://localhost:8000/api/v1/time-travel/rollback" \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "thread_123",
    "target_snapshot_id": "snapshot_456",
    "strategy": "soft"
  }'

# 创建分支
curl -X POST "http://localhost:8000/api/v1/time-travel/branches" \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "thread_123",
    "branch_name": "alternative_analysis",
    "description": "尝试不同的分析方法"
  }'

# 查询执行历史
curl -X GET "http://localhost:8000/api/v1/time-travel/history?thread_id=thread_123&limit=10"

# 获取系统状态
curl -X GET "http://localhost:8000/api/v1/time-travel/status"
```

### WebSocket实时通信

```javascript
// 连接WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/user123');

// 发送消息
ws.send(JSON.stringify({
  type: 'chat',
  content: '你好，请帮我分析数据',
  agent_type: 'supervisor'
}));

// 接收流式响应
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('收到响应:', data);
};
```

### 功能演示

```bash
# 运行完整功能演示
python examples/enhanced_components_demo.py

# 测试多智能体协作
python examples/test_multi_agent.py

# 测试RAG系统
python examples/test_rag_system.py

# 测试工作流编排
python examples/test_workflows.py

# 测试人工干预系统
python examples/langgraph_human_in_loop_demo.py

# 测试人工干预智能体
python examples/human_in_loop_agent_demo.py

# 测试流式处理功能
python examples/streaming_comprehensive_demo.py

# 测试流式处理实战演示
python examples/streaming_practical_demo.py

# 测试时间旅行功能
python examples/time_travel_demo.py

# 测试时间旅行API
python test_time_travel_api.py

# 🆕 测试LangMem提示词优化功能
python examples/langmem_prompt_optimization_demo.py

# 🆕 测试提示词优化集成
python -c "from core.optimization.prompt_optimizer import demo_integration; import asyncio; asyncio.run(demo_integration())"
```

## 🏗️ 项目架构

### 整体架构设计

项目采用前后端分离的微服务架构，包含后端API服务和前端Web应用两个主要部分：

```
┌─────────────────────────────────────────────────────────────┐
│                    前端Web应用 (Frontend)                    │
│              React + TypeScript + Vite                     │
├─────────────────────────────────────────────────────────────┤
│                    API网关层 (API Gateway)                  │
│                   RESTful API + WebSocket                  │
├─────────────────────────────────────────────────────────────┤
│                    后端服务层 (Backend Services)            │
│                   LangGraph多智能体系统                     │
└─────────────────────────────────────────────────────────────┘
```

### 后端分层架构设计

后端服务采用分层架构设计，共分为8个主要层级：

```
┌─────────────────────────────────────────────────────────────┐
│                    API服务层 (API Layer)                     │
│                   FastAPI Web应用                           │
├─────────────────────────────────────────────────────────────┤
│                   数据模型层 (Models Layer)                  │
│                  API请求/响应数据模型                        │
├─────────────────────────────────────────────────────────────┤
│                  智能体实现层 (Agents Layer)                 │
│              多智能体协作、记忆增强、MCP集成                  │
├─────────────────────────────────────────────────────────────┤
│                  工具集成层 (Tools Layer)                    │
│               MCP工具管理、外部工具集成                       │
├─────────────────────────────────────────────────────────────┤
│                  工作流编排层 (Workflows Layer)              │
│              复杂工作流构建、条件路由、并行执行                │
├─────────────────────────────────────────────────────────────┤
│                  核心抽象层 (Core Layer)                     │
│         记忆管理、流式处理、时间旅行、人工干预                │
├─────────────────────────────────────────────────────────────┤
│                  基础设施层 (Infrastructure Layer)           │
│         缓存、数据库、检查点、错误处理、日志系统              │
├─────────────────────────────────────────────────────────────┤
│                  配置管理层 (Configuration Layer)            │
│                  统一配置、环境管理                          │
└─────────────────────────────────────────────────────────────┘
```

#### 各层级详细说明

1. **API服务层** - 基于FastAPI的Web应用入口
   - 提供RESTful API接口
   - WebSocket连接管理
   - 中间件和路由管理

2. **数据模型层** - API数据模型定义
   - 请求/响应模型
   - 数据验证和序列化
   - 类型安全保障

3. **智能体实现层** - 智能体核心实现
   - 基础智能体抽象
   - 多智能体协作
   - 记忆增强智能体
   - MCP增强智能体

4. **工具集成层** - 工具管理和集成
   - MCP工具管理
   - 外部工具集成
   - 工具注册系统

5. **工作流编排层** - 复杂工作流管理
   - 工作流构建器
   - 条件路由和并行执行
   - 子图管理

6. **核心抽象层** - 核心功能模块
   - LangMem记忆管理
   - 流式处理
   - 时间旅行功能
   - 人工干预系统

7. **基础设施层** - 基础服务支持
   - Redis缓存管理
   - 数据库管理
   - 检查点管理
   - 错误处理和日志系统

8. **配置管理层** - 配置和环境管理
   - 统一配置系统
   - 环境变量管理
   - 模块初始化

### 目录结构

```
langgraph_study/
├── 📁 config/                    # 配置管理
│   ├── settings.py               # 统一配置系统
│   ├── memory_config.py          # LangMem记忆配置
│   └── __init__.py               # 配置模块初始化
├── 📁 core/                      # 核心模块
│   ├── __init__.py               # 核心模块导出
│   ├── agents/                   # 智能体实现
│   │   ├── __init__.py           # 智能体模块导出
│   │   ├── base.py               # 基础智能体抽象类
│   │   ├── collaborative.py      # 多智能体协作
│   │   ├── memory_enhanced.py    # 记忆增强智能体基类
│   │   ├── mcp_enhanced.py       # MCP增强智能体
│   │   ├── collaboration_optimizer.py # 协作优化器
│   │   ├── manager.py            # 智能体管理器
│   │   ├── registry.py           # 智能体注册系统
│   │   └── templates/            # 智能体模板和指南
│   │       ├── CORE_MODULES_QUICK_GUIDE.md # 核心模块快速指南
│   │       ├── HOW_TO_CREATE_NEW_AGENTS.md # 新智能体创建指南
│   │       ├── create_new_agent_guide.py # 智能体创建指导脚本
│   │       ├── integrated_agent_example.py # 集成智能体示例
│   │       ├── new_agent_template.py # 新智能体模板
│   │       └── simple_core_usage_example.py # 核心模块简单使用示例
│   ├── optimization/             # 🆕 提示词优化
│   │   ├── __init__.py           # 优化模块导出
│   │   ├── prompt_optimizer.py   # 提示词优化器(包含FeedbackCollector和AutoOptimizationScheduler)
│   │   └── prompt_optimization_api.py # 优化API接口
│   ├── cache/                    # 缓存管理
│   │   ├── __init__.py           # 缓存模块导出
│   │   └── redis_manager.py      # Redis缓存管理器
│   ├── checkpoint/               # 检查点管理
│   │   ├── __init__.py           # 检查点模块导出
│   │   └── manager.py            # 检查点管理器
│   ├── database/                 # 数据库管理
│   │   └── __init__.py           # 数据库模块导出
│   ├── error/                    # 错误处理
│   │   └── __init__.py           # 错误处理模块导出
│   ├── interrupts/               # 人工干预
│   │   ├── __init__.py           # 中断模块导出
│   │   ├── enhanced_interrupt_manager.py # 增强中断管理器
│   │   └── interrupt_types.py    # 中断类型定义
│   ├── logging/                  # 日志系统
│   │   └── __init__.py           # 日志模块导出
│   ├── memory/                   # LangMem记忆管理
│   │   ├── __init__.py           # 记忆模块导出
│   │   ├── store_manager.py      # 存储管理器
│   │   └── tools.py              # 记忆工具集成
│   ├── streaming/                # 流式处理
│   │   ├── __init__.py           # 流式处理模块导出
│   │   ├── stream_manager.py     # 流管理器
│   │   ├── stream_manager_enhanced.py # 增强流管理器
│   │   ├── websocket_handler.py  # WebSocket处理器
│   │   ├── sse_handler.py        # SSE处理器
│   │   ├── stream_types.py       # 流类型定义
│   │   └── langgraph_adapter.py  # LangGraph官方流式适配器
│   ├── time_travel/              # 时间旅行功能
│   │   ├── __init__.py           # 时间旅行模块导出
│   │   ├── time_travel_manager.py # 时间旅行管理器
│   │   ├── time_travel_api.py    # 时间旅行API接口
│   │   ├── checkpoint_manager.py # 检查点管理
│   │   ├── rollback_manager.py   # 回滚管理器
│   │   ├── state_history_manager.py # 状态历史管理
│   │   └── time_travel_types.py  # 时间旅行类型定义
│   ├── tools/                    # 工具集成
│   │   ├── __init__.py           # 工具模块导出
│   │   ├── mcp_manager.py        # MCP管理器
│   │   ├── mcp_api.py            # MCP API接口
│   │   ├── mcp_connection_manager.py # MCP连接管理
│   │   ├── mcp_cache_manager.py  # MCP缓存管理
│   │   └── enhanced_tool_manager.py # 增强工具管理器
│   └── workflows/                # 工作流编排
│       ├── __init__.py           # 工作流模块导出
│       ├── workflow_builder.py   # 工作流构建器
│       ├── conditional_router.py # 条件路由器
│       ├── parallel_executor.py  # 并行执行器
│       ├── subgraph_manager.py   # 子图管理器
│       └── workflow_types.py     # 工作流类型定义
├── 📁 models/                    # API数据模型
│   ├── __init__.py               # 模型模块导出
│   ├── base_models.py            # 基础响应模型
│   ├── chat_models.py            # 聊天API模型
│   ├── agent_models.py           # 智能体API模型
│   ├── memory_models.py          # 记忆管理API模型
│   └── rag_models.py             # RAG系统API模型
├── 📁 tools/                     # 外部工具集成
│   ├── __init__.py               # 工具集初始化
│   ├── searchticket.py           # 搜索工具
│   ├── weatherserver.py          # 天气服务工具
│   ├── code_tool.py              # 代码执行工具
│   ├── db_tool.py                # 数据库工具
│   ├── order.py                  # 订单管理工具
│   ├── cancelorder.py            # 取消订单工具
│   └── scheduled_order.py        # 定时订单工具
├── 📁 examples/                  # 示例代码
│   ├── advanced_time_travel_demo.py # 高级时间旅行演示
│   ├── create_custom_agent_demo.py # 自定义智能体创建演示
│   ├── enhanced_components_demo.py # 增强组件演示
│   ├── human_in_loop_agent_demo.py # 人工干预智能体演示
│   ├── langgraph_human_in_loop_demo.py # LangGraph官方人工干预演示
│   ├── langmem_practical_demo.py # LangMem实战演示
│   ├── 🆕 langmem_prompt_optimization_demo.py # LangMem提示词优化演示
│   ├── langmem_quick_start.py    # LangMem快速开始
│   ├── langmem_search_fixed.py   # LangMem搜索修复版
│   ├── mcp_demo.py               # MCP集成演示
│   ├── memory_enhanced_demo.py   # 记忆增强演示
│   ├── models_usage_demo.py      # 模型使用演示
│   ├── streaming_comprehensive_demo.py # 流式处理综合演示
│   ├── streaming_practical_demo.py # 流式处理实战演示
│   └── time_travel_demo.py       # 时间旅行功能演示
├── 📁 scripts/                   # 管理脚本
│   ├── create_store_vectors_table.py # 创建向量表
│   ├── initialize_database.py    # 数据库初始化
│   ├── initialize_system.py      # 系统初始化
│   ├── quick_check_langmem.py    # LangMem环境检查
│   ├── setup_pgvector.py         # pgvector扩展安装
│   ├── setup_project.py          # 项目设置
│   ├── simple_check_langmem.py   # 简化LangMem检查
│   ├── simple_initialize_database.py # 简化数据库初始化
│   ├── simple_setup_pgvector.py  # 简化pgvector安装
│   ├── test_store_vectors.py     # 测试向量存储
│   └── validate_langmem_requirements.py # 验证LangMem要求
├── 📁 spec/                      # 项目规范文档
│   ├── 05_langmem_integration.md # LangMem集成说明
│   ├── HUMAN_IN_LOOP_LEARNING_SUMMARY.md # 人工干预学习总结
│   ├── REACT_FRONTEND_DEVELOPMENT_GUIDE.md # React前端开发指南
│   ├── database_schema_reference.md # 数据库模式参考
│   ├── langgraph_official_guide.md # LangGraph官方指南
│   └── models_usage_guide.md     # 模型使用指南
├── 📁 test/                      # 测试代码
│   ├── test_agent_factory.py     # 智能体工厂测试
│   ├── test_all_components.py    # 所有组件集成测试
│   ├── test_checkpoint_fix.py    # 检查点修复测试
│   ├── test_enhanced_interrupt_manager.py # 增强中断管理器测试
│   ├── test_interrupt_comprehensive.py # 中断系统综合测试
│   ├── test_interrupt_official_demo.py # 官方中断演示测试
│   ├── test_langgraph_human_in_loop.py # LangGraph人工干预测试
│   ├── test_langmem_integration.py # LangMem集成测试
│   ├── test_langmem_official.py  # LangMem官方功能测试
│   ├── test_mcp.py               # MCP功能测试
│   ├── test_optimization.py      # 提示词优化功能测试
│   ├── test_redis_connection.py  # Redis连接测试
│   ├── test_redis_integration.py # Redis集成测试
│   ├── test_registration.py      # 注册系统测试
│   ├── test_time_travel.py       # 时间旅行功能测试
│   └── test_time_travel_api.py   # 时间旅行API测试
├── bootstrap.py                  # 系统启动器
├── main.py                       # FastAPI Web应用入口
├── graph5.py                     # 多智能体协作图
├── graph5_fixed.py               # 修复版多智能体图
├── requirements.txt              # Python依赖
├── langgraph.json                # LangGraph配置
├── servers_config.json           # MCP服务器配置
├── .env.template                 # 环境变量模板
├── .gitignore                    # Git忽略文件配置
├── LICENSE                       # 项目许可证
└── README.md                     # 项目说明文档
```

### 核心模块说明

#### 智能体系统
- **BaseAgent**: 统一的智能体抽象基类
- **多智能体协作**: Supervisor、Research、Chart智能体协作
- **RAG智能体**: 智能检索增强生成系统
- **专业化智能体**: 代码、数据分析、内容创作等专业智能体

#### 记忆管理 (LangMem)
- **存储管理**: PostgreSQL + pgvector向量存储
- **记忆类型**: 语义、情节、程序记忆
- **命名空间**: 用户级、智能体级、组织级隔离

#### 🆕 提示词优化系统
- **智能优化**: 基于用户反馈和对话历史自动改进提示词
- **单智能体优化**: 针对个体智能体的提示词精细化调优
- **多智能体协同优化**: 优化整个智能体团队的协作效果和一致性
- **持续学习机制**: 从用户满意度评分和反馈文本中持续学习
- **版本管理**: 支持提示词历史版本管理和A/B测试对比
- **自动化调度**: 定期自动执行优化流程，支持配置化策略
- **反馈收集**: 多维度用户反馈收集和分析系统
- **API集成**: 完整的RESTful API支持优化流程集成

#### 工作流编排
- **工作流构建**: 可视化工作流定义和构建
- **子图管理**: 嵌套工作流和模块化执行
- **条件路由**: 动态条件判断和分支
- **并行执行**: 多任务并发处理

#### 时间旅行功能
- **状态快照**: 完整的执行状态捕获和存储
- **检查点管理**: 关键节点状态保存和命名
- **回滚机制**: 任意时间点状态恢复，支持软回滚和硬回滚
- **分支管理**: 多分支执行和合并策略
- **历史追踪**: 完整的执行历史和时间线查询
- **REST API**: 完整的时间旅行API接口
- **配置管理**: 灵活的时间旅行配置和策略

#### 人工干预系统 (Human-in-the-Loop)
- **增强中断管理器**: 集成LangGraph官方`interrupt()`函数和`Command`原语
- **多种中断类型**: 审批请求、人工输入、工具审查、状态编辑
- **高级审批工作流**: 支持简单审批、多级审批、一致性审批等工作流类型
- **智能通知系统**: 审批通知、完成通知、超时通知、升级通知
- **中断响应处理**: 自动处理中断响应、状态恢复、过期清理

#### 流式处理系统
- **多种流式模式**: 支持VALUES、EVENTS、UPDATES、MESSAGES、DEBUG、ALL等流式模式
- **传输协议支持**: WebSocket双向通信和SSE服务器推送
- **LangGraph官方集成**: 通过适配器无缝集成LangGraph官方流式处理
- **事件类型覆盖**: 节点事件、工具事件、消息事件、状态事件、控制事件
- **高级功能**: 事件处理器、中断处理、状态管理、错误恢复、心跳机制
- **工具流式支持**: 为工具函数提供流式写入器支持

#### MCP集成
- **协议支持**: Model Context Protocol标准实现
- **工具管理**: 外部工具动态加载和管理
- **连接池**: 高效的MCP服务器连接管理
- **缓存机制**: 智能缓存提升性能

## 🚀 快速开始

### 环境要求

- **Python 3.9+** (推荐 3.11+)
- **PostgreSQL 13+** (推荐 14+，必须支持pgvector扩展)
- **Redis 6.0+** (可选，用于缓存和会话存储)
- **Node.js 18+** (可选，用于MCP服务器)

### 数据库要求详解

#### PostgreSQL + pgvector 配置

本项目使用 LangMem 进行智能体记忆管理，需要满足以下数据库要求：

**必需扩展：**
- `pgvector` 扩展（版本 0.5.0+）- 用于向量存储和语义搜索
- `uuid-ossp` 扩展 - 用于UUID生成

**数据库表结构：**
- `store` - 主存储表，存储记忆的键值对数据
- `store_vectors` - 向量存储表，存储嵌入向量用于语义搜索
- `checkpoints` - 检查点表，存储对话状态
- `checkpoint_blobs` - 检查点二进制数据表
- `checkpoint_writes` - 检查点写入记录表

**快速环境检查：**
```bash
# 检查数据库是否满足 LangMem 要求
python scripts/quick_check_langmem.py

# 自动安装和配置 pgvector 扩展
python scripts/setup_pgvector.py
```

### 一键启动 (推荐)

```bash
# 1. 克隆项目
git clone <repository-url>
cd langgraph_study

# 2. 安装Python依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.template .env
# 编辑 .env 文件，配置必要的API密钥和数据库连接

# 4. 一键启动系统 (自动完成数据库初始化、系统配置等)
python main.py
```

### 手动安装 (高级用户)

```bash
# 1. 项目环境设置
python scripts/setup_project.py

# 2. 数据库初始化 (创建表结构、安装扩展)
python scripts/initialize_database.py

# 3. 系统初始化 (配置智能体、工具、记忆系统)
python scripts/initialize_system.py

# 4. 启动Web服务
python main.py
```

### Docker 部署

```bash
# 使用 Docker Compose 一键部署
docker-compose up -d

# 或者单独构建和运行
docker build -t langgraph-multi-agent .
docker run -p 8000:8000 --env-file .env langgraph-multi-agent
```

## 🔧 配置说明

### 环境变量配置

主要配置项（详见 `.env.template`）：

```bash
# ===================
# LLM API配置
# ===================
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
TONGYI_API_KEY=your_tongyi_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# 默认LLM配置
DEFAULT_LLM_PROVIDER=deepseek
DEFAULT_MODEL_NAME=deepseek-chat
DEFAULT_TEMPERATURE=0.7

# ===================
# 数据库配置
# ===================
# PostgreSQL (主数据库 + LangMem记忆存储)
DB_POSTGRES_HOST=localhost
DB_POSTGRES_PORT=5432
DB_POSTGRES_USER=postgres
DB_POSTGRES_PASSWORD=your_password
DB_POSTGRES_DB=langgraph

# Redis (缓存和会话存储)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# 向量数据库 (RAG系统)
VECTOR_DB_TYPE=chroma  # chroma, pinecone, weaviate
VECTOR_DB_PATH=./data/chroma_db
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env

# ===================
# LangMem记忆配置
# ===================
LANGMEM_STORE_TYPE=postgres  # postgres, memory
LANGMEM_EMBEDDING_MODEL=openai:text-embedding-3-small
LANGMEM_EMBEDDING_DIMS=1536
LANGMEM_NAMESPACE_STRATEGY=user_isolated  # user_isolated, agent_shared, org_shared

# ===================
# MCP协议配置
# ===================
MCP_ENABLED=true
MCP_SERVERS_CONFIG_PATH=./config/mcp_servers.json
MCP_CONNECTION_TIMEOUT=30
MCP_MAX_CONNECTIONS=10
MCP_CACHE_TTL=300

# ===================
# 应用配置
# ===================
APP_NAME=LangGraph Multi-Agent System
APP_VERSION=1.0.0
APP_DEBUG=false
APP_HOST=0.0.0.0
APP_PORT=8000
APP_WORKERS=4

# API配置
API_VERSION=v1
API_PREFIX=/api/v1
API_RATE_LIMIT=100  # requests per minute
API_CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]

# ===================
# 安全配置
# ===================
SECRET_KEY=your_secret_key_here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440
ENCRYPTION_KEY=your_encryption_key

# ===================
# 日志配置
# ===================
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_STRUCTURED=true
LOG_FILE_PATH=./logs/app.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5

# ===================
# 监控配置
# ===================
MONITORING_ENABLED=true
METRICS_ENDPOINT=/metrics
HEALTH_CHECK_ENDPOINT=/health
PERFORMANCE_TRACKING=true

# ===================
# 工作流配置
# ===================
WORKFLOW_MAX_STEPS=100
WORKFLOW_TIMEOUT=3600  # seconds
WORKFLOW_PARALLEL_LIMIT=5
WORKFLOW_CHECKPOINT_INTERVAL=10  # steps

# ===================
# 时间旅行配置
# ===================
TIME_TRAVEL_ENABLED=true
SNAPSHOT_RETENTION_DAYS=30
CHECKPOINT_COMPRESSION=true
ROLLBACK_SAFETY_CHECK=true
```

### 智能体配置

系统支持多种智能体类型，每种都有独特的配置：

#### 1. 多智能体协作系统 (graph5.py)
```python
MULTI_AGENT_CONFIG = {
    "supervisor": {
        "model": "deepseek-chat",
        "temperature": 0.1,
        "max_iterations": 10,
        "tools": ["google_search", "mcp_tools"]
    },
    "research": {
        "model": "deepseek-chat", 
        "temperature": 0.3,
        "search_depth": 5,
        "tools": ["google_search", "web_scraper"]
    },
    "chart": {
        "model": "deepseek-chat",
        "temperature": 0.2,
        "chart_types": ["line", "bar", "pie", "scatter"],
        "tools": ["matplotlib", "plotly", "mcp_chart_tools"]
    }
}
```

#### 2. Agentic RAG系统 (graph6.py)
```python
RAG_CONFIG = {
    "retrieval": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 5,
        "similarity_threshold": 0.7
    },
    "generation": {
        "model": "deepseek-chat",
        "temperature": 0.2,
        "max_tokens": 2000,
        "context_window": 8000
    },
    "evaluation": {
        "relevance_threshold": 0.6,
        "quality_threshold": 0.7,
        "enable_reranking": true
    }
}
```

#### 3. 专业化智能体 (graph7.py+)
```python
SPECIALIZED_AGENTS_CONFIG = {
    "code_agent": {
        "model": "deepseek-coder",
        "temperature": 0.1,
        "languages": ["python", "javascript", "typescript", "go"],
        "tools": ["code_executor", "static_analyzer", "formatter"]
    },
    "data_agent": {
        "model": "deepseek-chat",
        "temperature": 0.2,
        "max_data_size": "100MB",
        "tools": ["pandas", "numpy", "matplotlib", "seaborn"]
    },
    "content_agent": {
        "model": "deepseek-chat",
        "temperature": 0.7,
        "content_types": ["article", "blog", "documentation", "marketing"],
        "tools": ["grammar_check", "seo_optimizer", "readability_analyzer"]
    }
}
```

### MCP服务器配置

配置外部MCP服务器以扩展工具能力：

```json
// config/mcp_servers.json
{
  "filesystem": {
    "command": "npx",
    "args": ["@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"],
    "env": {},
    "timeout": 30
  },
  "github": {
    "command": "npx", 
    "args": ["@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "your_github_token"
    },
    "timeout": 30
  },
  "postgres": {
    "command": "npx",
    "args": ["@modelcontextprotocol/server-postgres"],
    "env": {
      "POSTGRES_CONNECTION_STRING": "postgresql://user:pass@localhost/db"
    },
    "timeout": 30
  }
}
```

## 🛠️ 开发指南

### 项目开发进展

当前项目整体完成度约 **65%**，采用分层开发策略：

| 层级 | 完成度 | 状态 | 主要内容 |
|------|--------|------|----------|
| 基础设施层 | 90% | ✅ 完成 | 环境配置、数据库、Redis、向量数据库 |
| 核心抽象层 | 85% | ✅ 完成 | 配置系统、智能体基类、状态管理 |
| 数据模型层 | 80% | ✅ 完成 | Pydantic模型、数据库模型、API模型 |
| 智能体实现层 | 70% | 🚧 进行中 | 多智能体协作、RAG、专业化智能体 |
| API服务层 | 75% | 🚧 进行中 | RESTful API、WebSocket、路由管理 |
| 流式处理层 | 60% | 🚧 进行中 | 实时通信、事件流、状态同步 |
| 人工干预层 | 85% | 🟡 开发中 | 增强中断管理、审批工作流、人工决策 |
| 监控日志层 | 65% | 🚧 进行中 | 性能监控、日志聚合、告警系统 |
| 安全认证层 | 30% | 📋 计划中 | 用户认证、权限控制、数据加密 |
| 部署运维层 | 50% | 🚧 进行中 | Docker、K8s、CI/CD、备份恢复 |

### 添加新智能体

#### 1. 创建智能体类

```python
# agents/your_agent.py
from core.agents.base import BaseAgent
from core.tools.mcp_manager import MCPManager
from typing import Dict, Any, List

class YourAgent(BaseAgent):
    """自定义智能体实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mcp_manager = MCPManager()
        self.specialized_tools = self._load_specialized_tools()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入并返回结果"""
        # 1. 预处理输入
        processed_input = await self._preprocess(input_data)
        
        # 2. 调用LLM和工具
        result = await self._execute_with_tools(processed_input)
        
        # 3. 后处理结果
        final_result = await self._postprocess(result)
        
        return final_result
    
    def _load_specialized_tools(self) -> List[str]:
        """加载专用工具"""
        return ["tool1", "tool2", "tool3"]
```

#### 2. 注册智能体

```python
# core/agents/registry.py
from agents.your_agent import YourAgent

AGENT_REGISTRY = {
    "supervisor": SupervisorAgent,
    "research": ResearchAgent,
    "chart": ChartAgent,
    "rag": RAGAgent,
    "your_agent": YourAgent,  # 添加新智能体
}
```

#### 3. 配置智能体

```python
# config/agents.py
YOUR_AGENT_CONFIG = {
    "model": "deepseek-chat",
    "temperature": 0.3,
    "max_tokens": 2000,
    "tools": ["custom_tool1", "custom_tool2"],
    "memory_enabled": True,
    "mcp_enabled": True
}
```

### 添加新工具

#### 1. 创建工具类

```python
# tools/your_tool.py
from core.tools.base import BaseTool
from typing import Dict, Any

class YourTool(BaseTool):
    """自定义工具实现"""
    
    name = "your_tool"
    description = "工具描述"
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具逻辑"""
        # 实现工具功能
        result = await self._perform_action(kwargs)
        return {
            "success": True,
            "data": result,
            "metadata": {"tool": self.name}
        }
    
    async def _perform_action(self, params: Dict[str, Any]) -> Any:
        """执行具体操作"""
        pass
```

#### 2. 注册工具

```python
# core/tools/registry.py
from tools.your_tool import YourTool

TOOL_REGISTRY = {
    "google_search": GoogleSearchTool,
    "web_scraper": WebScraperTool,
    "chart_generator": ChartGeneratorTool,
    "your_tool": YourTool,  # 添加新工具
}
```

#### 3. MCP工具集成

```python
# 如果是MCP工具，添加到MCP服务器配置
# config/mcp_servers.json
{
  "your_mcp_server": {
    "command": "python",
    "args": ["-m", "your_mcp_server"],
    "env": {
      "API_KEY": "your_api_key"
    },
    "timeout": 30
  }
}
```

### 工作流开发

#### 1. 创建工作流图

```python
# workflows/your_workflow.py
from langgraph import StateGraph
from core.state import AgentState
from agents.supervisor import SupervisorAgent
from agents.your_agent import YourAgent

def create_your_workflow():
    """创建自定义工作流"""
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("supervisor", SupervisorAgent)
    workflow.add_node("your_agent", YourAgent)
    workflow.add_node("finish", lambda x: x)
    
    # 添加边
    workflow.add_edge("supervisor", "your_agent")
    workflow.add_edge("your_agent", "finish")
    
    # 设置入口点
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()
```

#### 2. 注册工作流

```python
# api/workflows.py
from workflows.your_workflow import create_your_workflow

WORKFLOW_REGISTRY = {
    "multi_agent": create_multi_agent_workflow,
    "rag_workflow": create_rag_workflow,
    "your_workflow": create_your_workflow,  # 添加新工作流
}
```

### 记忆系统扩展

#### 1. 自定义记忆存储

```python
# core/memory/custom_store.py
from langmem import BaseStore
from typing import Dict, Any, List

class CustomMemoryStore(BaseStore):
    """自定义记忆存储实现"""
    
    async def put(self, namespace: str, key: str, value: Dict[str, Any]):
        """存储记忆"""
        # 实现自定义存储逻辑
        pass
    
    async def get(self, namespace: str, key: str) -> Dict[str, Any]:
        """获取记忆"""
        # 实现自定义获取逻辑
        pass
    
    async def search(self, namespace: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """语义搜索记忆"""
        # 实现自定义搜索逻辑
        pass
```

#### 2. 记忆策略配置

```python
# config/memory.py
MEMORY_STRATEGIES = {
    "user_isolated": {
        "namespace_pattern": "user_{user_id}",
        "retention_days": 30,
        "max_memories": 1000
    },
    "agent_shared": {
        "namespace_pattern": "agent_{agent_type}",
        "retention_days": 7,
        "max_memories": 500
    },
    "custom_strategy": {
        "namespace_pattern": "custom_{context}",
        "retention_days": 60,
        "max_memories": 2000
    }
}
```

## 📊 监控和调试

### 系统健康检查

#### 1. 基础健康检查
```bash
# 检查系统整体状态
curl http://localhost:8000/health

# 检查详细健康状态
curl http://localhost:8000/health/detailed

# 检查特定组件
curl http://localhost:8000/health/database
curl http://localhost:8000/health/redis
curl http://localhost:8000/health/mcp
```

#### 2. 智能体状态监控
```bash
# 检查智能体状态
curl http://localhost:8000/api/v1/agents/status

# 检查特定智能体
curl http://localhost:8000/api/v1/agents/supervisor/status

# 检查智能体性能指标
curl http://localhost:8000/api/v1/agents/metrics
```

### 性能指标监控

#### 1. Prometheus指标
```bash
# 访问Prometheus指标端点
curl http://localhost:8000/metrics

# 主要指标包括：
# - agent_requests_total: 智能体请求总数
# - agent_request_duration_seconds: 请求处理时间
# - agent_memory_usage_bytes: 内存使用量
# - mcp_connections_active: 活跃MCP连接数
# - workflow_executions_total: 工作流执行总数
```

#### 2. 实时性能监控
```python
# 使用内置监控API
import requests

# 获取实时性能数据
response = requests.get("http://localhost:8000/api/v1/monitoring/performance")
metrics = response.json()

print(f"CPU使用率: {metrics['cpu_usage']}%")
print(f"内存使用率: {metrics['memory_usage']}%")
print(f"活跃连接数: {metrics['active_connections']}")
print(f"平均响应时间: {metrics['avg_response_time']}ms")
```

### 日志查看和分析

#### 1. 结构化日志查看
```bash
# 查看应用日志
tail -f logs/app.log

# 查看智能体日志
tail -f logs/agents.log

# 查看MCP日志
tail -f logs/mcp.log

# 查看工作流日志
tail -f logs/workflows.log
```

#### 2. 日志过滤和搜索
```bash
# 按级别过滤
grep "ERROR" logs/app.log

# 按智能体过滤
grep "supervisor" logs/agents.log

# 按时间范围过滤
grep "2024-01-01" logs/app.log

# 使用jq解析JSON日志
cat logs/app.log | jq '.level == "ERROR"'
```

#### 3. 日志聚合查询
```python
# 使用日志查询API
import requests

# 查询错误日志
response = requests.post("http://localhost:8000/api/v1/logs/query", json={
    "level": "ERROR",
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-01T23:59:59Z",
    "limit": 100
})

logs = response.json()
for log in logs["data"]:
    print(f"{log['timestamp']} - {log['message']}")
```

### 调试工具

#### 1. 智能体调试
```python
# 启用调试模式
import os
os.environ["APP_DEBUG"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"

# 使用调试API
response = requests.post("http://localhost:8000/api/v1/debug/agent", json={
    "agent_type": "supervisor",
    "input": "测试输入",
    "debug_level": "verbose"
})

debug_info = response.json()
print("执行步骤:", debug_info["steps"])
print("中间状态:", debug_info["intermediate_states"])
print("工具调用:", debug_info["tool_calls"])
```

#### 2. 工作流调试
```python
# 工作流步骤跟踪
response = requests.post("http://localhost:8000/api/v1/debug/workflow", json={
    "workflow_id": "multi_agent",
    "input": {"query": "测试查询"},
    "trace_enabled": True
})

trace = response.json()
for step in trace["execution_trace"]:
    print(f"步骤: {step['node']}")
    print(f"输入: {step['input']}")
    print(f"输出: {step['output']}")
    print(f"耗时: {step['duration']}ms")
```

#### 3. 内存和缓存调试
```bash
# 查看记忆系统状态
curl http://localhost:8000/api/v1/debug/memory/stats

# 查看缓存状态
curl http://localhost:8000/api/v1/debug/cache/stats

# 清理缓存
curl -X POST http://localhost:8000/api/v1/debug/cache/clear
```

### 告警和通知

#### 1. 告警规则配置
```yaml
# config/alerts.yaml
alerts:
  high_error_rate:
    condition: "error_rate > 0.05"
    duration: "5m"
    severity: "warning"
    message: "错误率过高: {{.Value}}"
  
  high_memory_usage:
    condition: "memory_usage > 0.8"
    duration: "2m"
    severity: "critical"
    message: "内存使用率过高: {{.Value}}"
  
  mcp_connection_failed:
    condition: "mcp_connections_failed > 0"
    duration: "1m"
    severity: "warning"
    message: "MCP连接失败: {{.Value}}"
```

#### 2. 通知渠道配置
```python
# config/notifications.py
NOTIFICATION_CHANNELS = {
    "email": {
        "enabled": True,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your_email@gmail.com",
        "password": "your_password",
        "recipients": ["admin@company.com"]
    },
    "slack": {
        "enabled": True,
        "webhook_url": "https://hooks.slack.com/services/...",
        "channel": "#alerts"
    },
    "webhook": {
        "enabled": True,
        "url": "https://your-webhook-endpoint.com/alerts",
        "headers": {"Authorization": "Bearer your_token"}
    }
}
```

## 🧪 测试

### 单元测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_agents.py
pytest tests/test_tools.py
pytest tests/test_workflows.py
pytest tests/test_memory.py

# 运行测试并生成覆盖率报告
pytest --cov=core --cov=agents --cov=tools --cov-report=html

# 运行性能测试
pytest tests/performance/ -v
```

### 集成测试

```bash
# 运行API集成测试
pytest tests/integration/test_api.py

# 运行智能体集成测试
pytest tests/integration/test_agents.py

# 运行工作流集成测试
pytest tests/integration/test_workflows.py

# 运行MCP集成测试
pytest tests/integration/test_mcp.py
```

### 端到端测试

```bash
# 运行完整的端到端测试
pytest tests/e2e/ -v

# 运行特定场景测试
pytest tests/e2e/test_multi_agent_collaboration.py
pytest tests/e2e/test_rag_workflow.py
pytest tests/e2e/test_time_travel.py
```

### 压力测试

```bash
# 使用locust进行压力测试
pip install locust
locust -f tests/load/locustfile.py --host=http://localhost:8000

# 或使用内置压力测试脚本
python tests/load/stress_test.py --concurrent=10 --duration=60
```

### 测试数据管理

```bash
# 初始化测试数据
python tests/fixtures/setup_test_data.py

# 清理测试数据
python tests/fixtures/cleanup_test_data.py

# 重置测试环境
python tests/fixtures/reset_test_env.py
```

## 📚 详细文档

### 核心架构文档
- **[LangMem集成说明](spec/05_langmem_integration.md)** - 长期记忆系统集成和配置
- **[人工干预学习总结](spec/HUMAN_IN_LOOP_LEARNING_SUMMARY.md)** - 人工干预系统学习和实践总结
- **[React前端开发指南](spec/REACT_FRONTEND_DEVELOPMENT_GUIDE.md)** - React + TypeScript前端开发详细方案
- **[数据库模式参考](spec/database_schema_reference.md)** - 数据库表结构、索引和关系设计
- **[LangGraph官方指南](spec/langgraph_official_guide.md)** - LangGraph框架使用和最佳实践
- **[模型使用指南](spec/models_usage_guide.md)** - 各种LLM模型的使用方法和配置

### 技术专题文档
- **[LangMem记忆系统](docs/langmem/)** - 智能体记忆管理、向量存储和语义搜索
  - [数据库架构](docs/langmem/database_schema.md)
  - [API参考](docs/langmem/api_reference.md)
  - [最佳实践](docs/langmem/best_practices.md)

- **[MCP协议集成](docs/mcp/)** - Model Context Protocol集成和工具扩展
  - [MCP服务器配置](docs/mcp/server_configuration.md)
  - [自定义工具开发](docs/mcp/custom_tools.md)
  - [故障排除](docs/mcp/troubleshooting.md)

- **[工作流编排](docs/workflows/)** - LangGraph工作流设计和状态管理
  - [状态图设计](docs/workflows/state_graph_design.md)
  - [条件路由](docs/workflows/conditional_routing.md)
  - [错误处理](docs/workflows/error_handling.md)

- **[时间旅行功能](docs/time_travel/)** - 状态回滚、快照管理和历史追踪
  - [快照策略](docs/time_travel/snapshot_strategies.md)
  - [回滚机制](docs/time_travel/rollback_mechanisms.md)
  - [性能优化](docs/time_travel/performance_optimization.md)

### API文档
- **[API参考手册](docs/api/)** - 完整的API接口文档
  - [智能体API](docs/api/agents.md)
  - [工作流API](docs/api/workflows.md)
  - [记忆管理API](docs/api/memory.md)
  - [工具管理API](docs/api/tools.md)
  - [监控API](docs/api/monitoring.md)

### 运维文档
- **[部署指南](docs/deployment/)** - 生产环境部署和配置
  - [Docker部署](docs/deployment/docker.md)
  - [Kubernetes部署](docs/deployment/kubernetes.md)
  - [负载均衡配置](docs/deployment/load_balancing.md)
  - [SSL/TLS配置](docs/deployment/ssl_tls.md)

- **[监控运维](docs/operations/)** - 系统监控、日志管理和故障排除
  - [Prometheus监控](docs/operations/prometheus.md)
  - [日志聚合](docs/operations/logging.md)
  - [告警配置](docs/operations/alerting.md)
  - [备份恢复](docs/operations/backup_recovery.md)

### 开发者指南
- **[贡献指南](CONTRIBUTING.md)** - 代码贡献、开发规范和提交流程
- **[代码规范](docs/development/coding_standards.md)** - Python代码风格和最佳实践
- **[测试指南](docs/development/testing_guide.md)** - 测试策略、工具使用和覆盖率要求
- **[性能优化](docs/development/performance_tuning.md)** - 性能分析、优化技巧和基准测试

### 示例和教程
- **[快速入门教程](docs/tutorials/)** - 从零开始的完整教程
  - [第一个智能体](docs/tutorials/first_agent.md)
  - [多智能体协作](docs/tutorials/multi_agent_collaboration.md)
  - [RAG系统构建](docs/tutorials/building_rag_system.md)
  - [自定义工具开发](docs/tutorials/custom_tool_development.md)
  - 🆕 [React前端开发入门](docs/tutorials/react_frontend_getting_started.md)
  - 🆕 [前端组件开发指南](docs/tutorials/frontend_component_development.md)

- **[示例项目](examples/)** - 完整的示例应用
  - [客服机器人](examples/customer_service_bot/)
  - [数据分析助手](examples/data_analysis_assistant/)
  - [内容创作系统](examples/content_creation_system/)
  - [代码审查助手](examples/code_review_assistant/)

### 故障排除
- **[常见问题FAQ](docs/faq.md)** - 常见问题和解决方案
- **[故障排除指南](docs/troubleshooting.md)** - 问题诊断和解决步骤
- **[错误代码参考](docs/error_codes.md)** - 错误代码含义和处理方法

## 🤝 贡献指南

我们欢迎所有形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 贡献方式

1. **代码贡献**
   - 修复bug
   - 添加新功能
   - 改进性能
   - 优化代码质量

2. **文档贡献**
   - 改进现有文档
   - 添加新的教程
   - 翻译文档
   - 修正错误

3. **测试贡献**
   - 添加测试用例
   - 改进测试覆盖率
   - 性能测试
   - 集成测试

4. **社区贡献**
   - 回答问题
   - 分享经验
   - 提供反馈
   - 推广项目

### 开发流程

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 Python 代码风格
- 使用 Black 进行代码格式化
- 使用 isort 进行导入排序
- 使用 mypy 进行类型检查
- 编写完整的文档字符串
- 添加适当的测试用例

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持

### 获取帮助

- **GitHub Issues**: [提交问题](https://github.com/your-repo/issues)
- **讨论区**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **文档**: [在线文档](https://your-docs-site.com)
- **邮件**: support@your-domain.com

### 社区

- **Discord**: [加入我们的Discord服务器](https://discord.gg/your-invite)
- **微信群**: 扫描二维码加入技术交流群
- **QQ群**: 123456789
- **知乎专栏**: [LangGraph多智能体系统](https://zhuanlan.zhihu.com/your-column)

### 商业支持

如需商业支持、定制开发或企业级服务，请联系：
- 邮箱: business@your-domain.com
- 电话: +86-xxx-xxxx-xxxx
- 官网: https://your-company.com

---

**⭐ 如果这个项目对你有帮助，请给我们一个星标！**

**🔔 关注项目获取最新更新和功能发布通知**

---

**Happy Coding! 🎉**