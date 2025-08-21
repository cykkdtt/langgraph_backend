#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试 MemoryEnhancedAgent 工具列表
检查工具列表中的对象类型
"""

import asyncio
from core.agents.registry import AgentRegistry, AgentFactory
from core.agents.memory_enhanced import MemoryEnhancedAgent
from core.memory.tools import MemoryToolsFactory
from langchain_core.tools import BaseTool

async def debug_memory_enhanced_agent_tools():
    """调试 MemoryEnhancedAgent 工具列表"""
    try:
        # 创建智能体注册表和工厂
        registry = AgentRegistry()
        factory = AgentFactory(registry)
        
        # 创建 supervisor 智能体实例
        from core.agents.base import AgentType
        agent_id = await factory.create_agent(
            agent_type=AgentType.SUPERVISOR,
            user_id="test_user",
            session_id="test_session",
            custom_config={"memory_enabled": True}
        )
        
        # 获取智能体实例
        agent = await factory.get_agent(agent_id)
        print(f"智能体类型: {type(agent)}")
        print(f"是否是 MemoryEnhancedAgent: {isinstance(agent, MemoryEnhancedAgent)}")
        
        # 检查初始工具列表
        print(f"\n初始工具列表长度: {len(agent.tools)}")
        for i, tool in enumerate(agent.tools):
            print(f"工具 {i}: 类型={type(tool)}, 是否有name属性={hasattr(tool, 'name')}, 是否是BaseTool={isinstance(tool, BaseTool)}")
            if hasattr(tool, 'name'):
                print(f"  工具名称: {tool.name}")
            elif hasattr(tool, '__name__'):
                print(f"  函数名称: {tool.__name__}")
            else:
                print(f"  对象: {tool}")
        
        # 手动调用 _add_memory_tools 来查看添加的工具
        print("\n=== 手动测试记忆工具创建 ===")
        namespace = f"agent_{agent.agent_id}"
        memory_tools = await MemoryToolsFactory.create_memory_tools(namespace)
        print(f"记忆工具数量: {len(memory_tools)}")
        for i, tool in enumerate(memory_tools):
            print(f"记忆工具 {i}: 类型={type(tool)}, 是否有name属性={hasattr(tool, 'name')}, 是否是BaseTool={isinstance(tool, BaseTool)}")
            if hasattr(tool, 'name'):
                print(f"  工具名称: {tool.name}")
            elif hasattr(tool, '__name__'):
                print(f"  函数名称: {tool.__name__}")
        
        # 调用智能体的 _add_memory_tools 方法
        print("\n=== 调用智能体的 _add_memory_tools ===")
        await agent._add_memory_tools()
        
        print(f"\n添加记忆工具后的工具列表长度: {len(agent.tools)}")
        for i, tool in enumerate(agent.tools):
            print(f"工具 {i}: 类型={type(tool)}, 是否有name属性={hasattr(tool, 'name')}, 是否是BaseTool={isinstance(tool, BaseTool)}")
            if hasattr(tool, 'name'):
                print(f"  工具名称: {tool.name}")
            elif hasattr(tool, '__name__'):
                print(f"  函数名称: {tool.__name__}")
            else:
                print(f"  对象: {tool}")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_memory_enhanced_agent_tools())