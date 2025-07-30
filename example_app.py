"""
多智能体LangGraph项目 - 示例应用

本模块提供一个完整的示例应用，展示如何使用新的架构：
- 系统初始化
- 智能体创建和注册
- 工具集成
- 协作任务处理
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List

from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from bootstrap import get_bootstrap, system_lifespan
from config.settings import get_settings
from core.agents.base import get_agent_registry, ChatRequest
from core.agents.collaborative import SupervisorAgent, ResearchAgent, ChartAgent
from core.tools import get_tool_registry, managed_tool, ToolCategory, ToolPermission
from core.memory import get_memory_manager, MemoryNamespace, MemoryScope, MemoryItem, MemoryType


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("example")


# 示例工具定义
@managed_tool(
    name="google_search",
    description="搜索互联网信息",
    category=ToolCategory.SEARCH,
    permissions=[ToolPermission.READ]
)
def google_search(query: str) -> str:
    """模拟Google搜索工具"""
    return f"搜索结果: {query} - 这是一个模拟的搜索结果"


@managed_tool(
    name="create_chart",
    description="创建图表",
    category=ToolCategory.GENERATION,
    permissions=[ToolPermission.WRITE]
)
def create_chart(data: str, chart_type: str = "bar") -> str:
    """模拟图表创建工具"""
    return f"已创建{chart_type}图表，数据: {data}"


@tool
def transfer_to_research_agent(task: str) -> str:
    """移交任务给研究智能体"""
    return f"任务已移交给研究智能体: {task}"


@tool
def transfer_to_chart_agent(task: str) -> str:
    """移交任务给图表智能体"""
    return f"任务已移交给图表智能体: {task}"


class ExampleApplication:
    """示例应用程序"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger("example.app")
        
        # 智能体实例
        self.supervisor_agent = None
        self.research_agent = None
        self.chart_agent = None
    
    async def setup_agents(self):
        """设置智能体"""
        try:
            self.logger.info("设置智能体...")
            
            # 创建语言模型
            llm = ChatDeepSeek(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                temperature=0.1
            )
            
            # 获取工具注册表
            tool_registry = get_tool_registry()
            
            # 获取工具实例
            search_tool = tool_registry.get_tool("google_search")
            chart_tool = tool_registry.get_tool("create_chart")
            
            # 创建智能体
            self.research_agent = ResearchAgent(
                llm=llm,
                search_tools=[search_tool] if search_tool else []
            )
            
            self.chart_agent = ChartAgent(
                llm=llm,
                chart_tools=[chart_tool] if chart_tool else []
            )
            
            self.supervisor_agent = SupervisorAgent(
                llm=llm,
                tools=[transfer_to_research_agent, transfer_to_chart_agent]
            )
            
            # 注册智能体
            agent_registry = get_agent_registry()
            agent_registry.register_agent("supervisor", self.supervisor_agent)
            agent_registry.register_agent("research", self.research_agent)
            agent_registry.register_agent("chart", self.chart_agent)
            
            self.logger.info("智能体设置完成")
            
        except Exception as e:
            self.logger.error(f"智能体设置失败: {e}")
            raise
    
    async def demo_memory_usage(self):
        """演示记忆功能"""
        try:
            self.logger.info("演示记忆功能...")
            
            # 获取记忆管理器
            memory_manager = get_memory_manager()
            
            # 创建用户命名空间
            user_namespace = MemoryNamespace(
                scope=MemoryScope.USER,
                identifier="demo_user"
            )
            
            # 存储一些记忆
            memories = [
                MemoryItem(
                    id="fact_1",
                    content="用户喜欢数据可视化",
                    memory_type=MemoryType.SEMANTIC,
                    importance=0.8,
                    metadata={"category": "preference"}
                ),
                MemoryItem(
                    id="event_1",
                    content="用户上次询问了关于销售数据的图表",
                    memory_type=MemoryType.EPISODIC,
                    importance=0.6,
                    metadata={"category": "interaction"}
                )
            ]
            
            for memory in memories:
                await memory_manager.store_memory(user_namespace, memory)
                self.logger.info(f"存储记忆: {memory.id}")
            
            # 搜索记忆
            from core.memory import MemoryQuery
            query = MemoryQuery(
                query="数据",
                limit=5
            )
            
            results = await memory_manager.search_memories(user_namespace, query)
            self.logger.info(f"搜索到 {len(results)} 条相关记忆")
            
            for result in results:
                self.logger.info(f"  - {result.content} (重要性: {result.importance})")
            
        except Exception as e:
            self.logger.error(f"记忆演示失败: {e}")
    
    async def demo_collaborative_task(self):
        """演示协作任务"""
        try:
            self.logger.info("演示协作任务...")
            
            # 创建任务请求
            task_request = ChatRequest(
                message="请帮我研究2024年AI市场趋势，并创建一个相关的图表",
                metadata={"task_type": "research_and_chart"}
            )
            
            # 使用主管智能体处理任务
            response = await self.supervisor_agent.chat(task_request)
            
            self.logger.info("协作任务完成")
            self.logger.info(f"回复: {response.message}")
            self.logger.info(f"元数据: {response.metadata}")
            
        except Exception as e:
            self.logger.error(f"协作任务演示失败: {e}")
    
    async def demo_streaming_task(self):
        """演示流式任务处理"""
        try:
            self.logger.info("演示流式任务处理...")
            
            # 创建流式任务请求
            task_request = ChatRequest(
                message="请分析当前的技术趋势",
                metadata={"stream": True}
            )
            
            # 流式处理
            async for chunk in self.research_agent.astream(task_request):
                self.logger.info(f"流式输出: {chunk.content[:100]}...")
            
            self.logger.info("流式任务完成")
            
        except Exception as e:
            self.logger.error(f"流式任务演示失败: {e}")
    
    async def run_demo(self):
        """运行完整演示"""
        try:
            # 设置智能体
            await self.setup_agents()
            
            # 演示记忆功能
            await self.demo_memory_usage()
            
            # 演示协作任务
            await self.demo_collaborative_task()
            
            # 演示流式任务
            await self.demo_streaming_task()
            
            self.logger.info("所有演示完成")
            
        except Exception as e:
            self.logger.error(f"演示运行失败: {e}")


async def main():
    """主函数"""
    logger.info("启动多智能体LangGraph示例应用")
    
    # 使用系统生命周期管理器
    async with system_lifespan() as bootstrap:
        # 检查系统状态
        status = bootstrap.get_system_status()
        logger.info("系统状态:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")
        
        if not status["is_initialized"]:
            logger.error("系统初始化失败，退出")
            return
        
        # 运行示例应用
        app = ExampleApplication()
        await app.run_demo()


if __name__ == "__main__":
    asyncio.run(main())