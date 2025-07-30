# 🤖 LangGraph Multi-Agent System

基于LangGraph的多智能体协作系统，集成了记忆管理、工具调用、错误处理和性能监控等完整功能。

## ✨ 功能特性

- 🤖 **多智能体协作架构** - Supervisor、Research、Chart等专业智能体
- 🧠 **智能记忆管理** - 基于LangMem的语义、情节、程序记忆
- 🔧 **丰富的工具集成** - 搜索、数据分析、可视化等工具
- 💾 **持久化存储支持** - PostgreSQL + Redis多层存储
- 🔄 **实时状态管理** - WebSocket实时通信
- 📊 **性能监控和日志** - 完整的错误处理和性能追踪
- 🛡️ **企业级架构** - 环境管理、配置验证、健康检查

## 🛠️ 技术栈

- **LangGraph**: 多智能体状态图框架
- **LangChain**: 语言模型和工具集成
- **FastAPI**: Web API框架
- **PostgreSQL**: 主数据库
- **Redis**: 缓存和会话存储
- **向量数据库**: Chroma/Pinecone（用于RAG）
- **Docker**: 容器化部署

## 📖 使用指南

### Web API

启动后访问：
- **主页**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **系统状态**: http://localhost:8000/status

### 聊天接口

```bash
# REST API
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "你好，请帮我分析苹果公司的股价",
    "user_id": "user123",
    "agent_type": "supervisor"
  }'

# WebSocket
# 连接到 ws://localhost:8000/ws/user123
```

### 功能演示

```bash
# 运行完整功能演示
python demo.py
```

## 🏗️ 项目架构

```
langgraph_study/
├── 📁 config/              # 配置管理
│   ├── settings.py         # 统一配置
│   └── memory_config.py    # 记忆配置
├── 📁 core/                # 核心模块
│   ├── agents/             # 智能体实现
│   │   ├── base.py         # 基础智能体
│   │   ├── collaborative.py # 协作智能体
│   │   └── memory_enhanced.py # 记忆增强智能体
│   ├── memory/             # 记忆管理
│   │   ├── store_manager.py # 存储管理
│   │   └── tools.py        # 记忆工具
│   ├── tools/              # 工具集成
│   ├── checkpoint/         # 检查点管理
│   ├── database/           # 数据库管理
│   ├── logging/            # 日志系统
│   ├── env/                # 环境管理
│   └── error/              # 错误处理
├── 📁 scripts/             # 管理脚本
│   ├── setup_project.py    # 项目设置
│   ├── initialize_database.py # 数据库初始化
│   └── initialize_system.py # 系统初始化
├── 📁 examples/            # 示例代码
├── 📁 spec/                # 项目规范
├── bootstrap.py            # 系统启动器
├── main.py                 # Web应用入口
├── start.py                # 快速启动脚本
├── demo.py                 # 功能演示
└── .env.template           # 环境变量模板
```

## 🚀 快速开始

### 环境要求

- Python 3.9+
- PostgreSQL 13+ （推荐 14+）
- Redis (可选)

#### LangMem 数据库要求

本项目使用 LangMem 进行智能体记忆管理，需要满足以下数据库要求：

**PostgreSQL 版本和扩展：**
- PostgreSQL 13+ （推荐 14+）
- `pgvector` 扩展（版本 0.5.0+）用于向量存储和语义搜索
- `uuid-ossp` 扩展用于UUID生成

**数据库表结构：**
- `store`: 主存储表，存储记忆的键值对数据
- `store_vectors`: 向量存储表，存储嵌入向量用于语义搜索
- `checkpoints`: 检查点表，存储对话状态
- `checkpoint_blobs`: 检查点二进制数据表
- `checkpoint_writes`: 检查点写入记录表

**快速检查数据库要求：**
```bash
# 检查数据库是否满足 LangMem 要求
python scripts/quick_check_langmem.py
```

**安装 pgvector 扩展：**
```bash
# 自动安装和配置 pgvector 扩展
python scripts/setup_pgvector.py
```

**详细信息：** 
- [LANGMEM_DATABASE_SCHEMA.md](LANGMEM_DATABASE_SCHEMA.md) - 数据库表结构详细说明
- [LANGMEM_INSTALLATION_GUIDE.md](LANGMEM_INSTALLATION_GUIDE.md) - 完整安装和配置指南

### 一键启动

```bash
# 克隆项目
git clone <repository-url>
cd langgraph_study

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.template .env
# 编辑 .env 文件，配置必要的API密钥和数据库连接

# 一键启动系统
python start.py
```

### 手动安装

```bash
# 1. 项目设置
python scripts/setup_project.py

# 2. 数据库初始化
python scripts/initialize_database.py

# 3. 系统初始化
python scripts/initialize_system.py

# 4. 启动Web服务
python main.py
```

## 🔧 配置说明

### 环境变量

主要配置项（详见 `.env.template`）：

```bash
# LLM API配置
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
TONGYI_API_KEY=your_tongyi_api_key

# 数据库配置
DB_POSTGRES_HOST=localhost
DB_POSTGRES_PORT=5432
DB_POSTGRES_USER=postgres
DB_POSTGRES_PASSWORD=your_password
DB_POSTGRES_DB=langgraph

# 应用配置
APP_DEBUG=false
APP_HOST=0.0.0.0
APP_PORT=8000

# 日志配置
LOG_LEVEL=INFO
LOG_STRUCTURED=true
```

### 智能体配置

系统包含三种主要智能体：

1. **Supervisor Agent** - 任务协调和分配
2. **Research Agent** - 信息研究和分析
3. **Chart Agent** - 数据可视化

## 🛠️ 开发指南

### 添加新智能体

```python
from core.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, agent_type="custom", **kwargs)
    
    async def process_message(self, content: str, context: dict) -> dict:
        # 实现自定义逻辑
        return {"content": "响应内容"}

# 注册智能体
from core.agents import get_agent_registry
registry = get_agent_registry()
registry.register_agent_class("custom", CustomAgent)
```

### 添加新工具

```python
from langchain.tools import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "自定义工具描述"
    
    def _run(self, query: str) -> str:
        # 实现工具逻辑
        return "工具执行结果"

# 注册工具
from core.tools import get_tool_registry
registry = get_tool_registry()
registry.register_tool(CustomTool())
```

## 📊 监控和调试

### 健康检查

```bash
curl http://localhost:8000/health
```

### 性能指标

```bash
curl http://localhost:8000/metrics
```

### 日志查看

```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
tail -f logs/error.log
```

## 🧪 测试

```bash
# 运行系统测试
python -m pytest test/

# 运行记忆集成测试
python test/test_langmem_integration.py

# 运行功能演示
python demo.py
```

## 📚 文档

详细文档请参考：

- [核心架构设计](spec/01_core_architecture.md)
- [API设计规范](spec/02_api_design.md)
- [智能体实现指南](spec/03_agent_implementation.md)
- [部署运维指南](spec/04_deployment_ops.md)
- [LangMem集成说明](spec/05_langmem_integration.md)

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🆘 支持

如有问题，请：

1. 查看 [FAQ](spec/FAQ.md)
2. 搜索 [Issues](../../issues)
3. 创建新的 [Issue](../../issues/new)

---

**Happy Coding! 🎉**