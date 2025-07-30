"""
多智能体LangGraph项目 - 功能演示脚本

本脚本演示系统的主要功能：
- 智能体创建和交互
- 记忆管理
- 工具使用
- 协作流程
"""

import asyncio
import json
from typing import Dict, Any

from bootstrap import get_bootstrap
from core.agents import get_agent_registry
from core.memory import get_memory_manager
from core.tools import get_tool_registry


class SystemDemo:
    """系统演示类"""
    
    def __init__(self):
        self.bootstrap = None
        self.agent_registry = None
        self.memory_manager = None
        self.tool_registry = None
    
    async def initialize(self):
        """初始化演示环境"""
        print("🔧 初始化演示环境...")
        
        # 初始化系统
        self.bootstrap = get_bootstrap()
        success = await self.bootstrap.initialize()
        
        if not success:
            raise RuntimeError("系统初始化失败")
        
        # 获取管理器实例
        self.agent_registry = get_agent_registry()
        self.memory_manager = get_memory_manager("postgres")
        self.tool_registry = get_tool_registry()
        
        print("✅ 演示环境初始化完成")
    
    async def demo_basic_chat(self):
        """演示基础聊天功能"""
        print("\n" + "="*50)
        print("📱 演示1: 基础聊天功能")
        print("="*50)
        
        # 创建supervisor智能体
        supervisor = await self.agent_registry.get_or_create_agent(
            agent_type="supervisor",
            user_id="demo_user",
            session_id="demo_session_1"
        )
        
        # 发送消息
        messages = [
            "你好，我是新用户",
            "请帮我分析一下苹果公司的股价趋势",
            "能帮我制作一个图表吗？"
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n👤 用户: {message}")
            
            response = await supervisor.process_message(
                content=message,
                context={"demo": True}
            )
            
            print(f"🤖 助手: {response.get('content', '无响应')}")
            
            # 显示元数据
            metadata = response.get('metadata', {})
            if metadata:
                print(f"📊 元数据: {json.dumps(metadata, ensure_ascii=False, indent=2)}")
    
    async def demo_memory_management(self):
        """演示记忆管理功能"""
        print("\n" + "="*50)
        print("🧠 演示2: 记忆管理功能")
        print("="*50)
        
        user_id = "demo_user"
        
        # 添加一些记忆
        memories = [
            {
                "content": "用户喜欢苹果公司的产品",
                "importance": 0.8,
                "memory_type": "preference"
            },
            {
                "content": "用户对股票投资感兴趣",
                "importance": 0.7,
                "memory_type": "interest"
            },
            {
                "content": "用户需要数据可视化帮助",
                "importance": 0.6,
                "memory_type": "need"
            }
        ]
        
        print("💾 添加记忆...")
        for memory in memories:
            await self.memory_manager.add_memory(
                user_id=user_id,
                content=memory["content"],
                importance=memory["importance"],
                metadata={"type": memory["memory_type"]}
            )
            print(f"  ✅ {memory['content']}")
        
        # 搜索记忆
        print("\n🔍 搜索相关记忆...")
        search_results = await self.memory_manager.search_memories(
            user_id=user_id,
            query="投资 股票",
            limit=5
        )
        
        for i, memory in enumerate(search_results, 1):
            print(f"  {i}. {memory.get('content', 'N/A')} (重要性: {memory.get('importance', 0)})")
    
    async def demo_tool_usage(self):
        """演示工具使用功能"""
        print("\n" + "="*50)
        print("🔧 演示3: 工具使用功能")
        print("="*50)
        
        # 获取所有可用工具
        tools = self.tool_registry.get_all_tools()
        print(f"📋 可用工具数量: {len(tools)}")
        
        for name, tool in tools.items():
            print(f"  🛠️  {name}: {getattr(tool, 'description', '无描述')}")
        
        # 演示工具调用（如果有搜索工具）
        if "search" in tools:
            print("\n🔍 演示搜索工具...")
            try:
                search_tool = tools["search"]
                result = await search_tool.arun("苹果公司最新新闻")
                print(f"搜索结果: {result[:200]}...")
            except Exception as e:
                print(f"搜索工具调用失败: {e}")
    
    async def demo_collaborative_workflow(self):
        """演示协作工作流"""
        print("\n" + "="*50)
        print("🤝 演示4: 智能体协作工作流")
        print("="*50)
        
        # 创建不同类型的智能体
        supervisor = await self.agent_registry.get_or_create_agent(
            agent_type="supervisor",
            user_id="demo_user",
            session_id="collab_session"
        )
        
        research_agent = await self.agent_registry.get_or_create_agent(
            agent_type="research",
            user_id="demo_user",
            session_id="collab_session"
        )
        
        chart_agent = await self.agent_registry.get_or_create_agent(
            agent_type="chart",
            user_id="demo_user",
            session_id="collab_session"
        )
        
        print("👥 创建的智能体:")
        print(f"  🎯 Supervisor: {supervisor.agent_type}")
        print(f"  🔍 Research: {research_agent.agent_type}")
        print(f"  📊 Chart: {chart_agent.agent_type}")
        
        # 模拟协作任务
        task = "请研究特斯拉公司的财务状况并制作可视化图表"
        print(f"\n📋 协作任务: {task}")
        
        # Supervisor分配任务
        print("\n🎯 Supervisor分析任务...")
        supervisor_response = await supervisor.process_message(
            content=task,
            context={"collaboration": True}
        )
        print(f"Supervisor: {supervisor_response.get('content', '')[:100]}...")
        
        # Research Agent执行研究
        print("\n🔍 Research Agent执行研究...")
        research_response = await research_agent.process_message(
            content="研究特斯拉公司的财务数据",
            context={"task_from": "supervisor"}
        )
        print(f"Research: {research_response.get('content', '')[:100]}...")
        
        # Chart Agent制作图表
        print("\n📊 Chart Agent制作图表...")
        chart_response = await chart_agent.process_message(
            content="基于研究结果制作财务图表",
            context={"research_data": research_response.get('content', '')}
        )
        print(f"Chart: {chart_response.get('content', '')[:100]}...")
    
    async def demo_system_monitoring(self):
        """演示系统监控功能"""
        print("\n" + "="*50)
        print("📊 演示5: 系统监控功能")
        print("="*50)
        
        # 获取系统状态
        status = self.bootstrap.get_system_status()
        print("🖥️  系统状态:")
        print(f"  初始化状态: {'✅' if status['initialized'] else '❌'}")
        print(f"  运行时间: {status['uptime']:.2f}秒")
        
        print("\n🔧 组件状态:")
        for component, active in status['components'].items():
            print(f"  {'✅' if active else '❌'} {component}")
        
        # 健康检查
        health = await self.bootstrap.health_check()
        print(f"\n🏥 健康状态: {health['status']}")
        
        for component, info in health['components'].items():
            status_icon = "✅" if info['status'] == "healthy" else "⚠️" if info['status'] == "degraded" else "❌"
            print(f"  {status_icon} {component}: {info['status']}")
    
    async def cleanup(self):
        """清理演示环境"""
        print("\n🧹 清理演示环境...")
        if self.bootstrap:
            await self.bootstrap.cleanup()
        print("✅ 清理完成")


async def main():
    """主演示函数"""
    print("🎭 LangGraph多智能体系统功能演示")
    print("=" * 60)
    
    demo = SystemDemo()
    
    try:
        # 初始化
        await demo.initialize()
        
        # 运行各项演示
        await demo.demo_basic_chat()
        await demo.demo_memory_management()
        await demo.demo_tool_usage()
        await demo.demo_collaborative_workflow()
        await demo.demo_system_monitoring()
        
        print("\n" + "="*60)
        print("🎉 演示完成！所有功能运行正常")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())