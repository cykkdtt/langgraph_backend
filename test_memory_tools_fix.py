#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试记忆工具修复
验证 MemoryEnhancedAgent 是否能正常处理消息而不出现工具相关错误
"""

import asyncio
import logging
from core.agents.registry import AgentRegistry, AgentFactory
from core.agents.base import ChatRequest
from langchain_core.messages import HumanMessage

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_memory_enhanced_agent():
    """测试记忆增强智能体的消息处理"""
    try:
        print("正在测试记忆增强智能体...")
        
        # 创建智能体注册表和工厂
        registry = AgentRegistry()
        factory = AgentFactory(registry)
        
        # 创建 SUPERVISOR 智能体实例
        instance_id = await factory.create_agent(
            agent_type="supervisor",
            user_id="test_user",
            session_id="test_session"
        )
        print(f"成功创建智能体实例: {instance_id}")
        
        # 获取智能体实例
        agent = await factory.get_agent(instance_id)
        if agent:
            print(f"智能体类型: {type(agent).__name__}")
            print(f"智能体ID: {agent.agent_id}")
            print(f"智能体名称: {agent.name}")
            print(f"工具数量: {len(agent.tools)}")
            
            # 测试消息处理
            print("\n开始测试消息处理...")
            chat_request = ChatRequest(
                messages=[HumanMessage(content="你好，请介绍一下你自己")],
                user_id="test_user",
                session_id="test_session"
            )
            
            # 处理消息
            response = await agent.chat(chat_request)
            print(f"智能体回复: {response.message.content[:100]}...")
            print("✅ 消息处理成功！")
            
        else:
            print("❌ 无法获取智能体实例")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_memory_enhanced_agent())