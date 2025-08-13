#!/usr/bin/env python3
"""
LangMem 实用演示 - 展示记忆功能的实际价值
这个演示展示了LangMem如何让AI智能体变得更加智能和个性化
"""

import asyncio
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到Python路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langgraph.store.postgres import AsyncPostgresStore
from config.memory_config import memory_config
import json
from datetime import datetime

class SmartAssistant:
    """智能助手 - 展示LangMem的实际应用"""
    
    def __init__(self):
        self.store = None
        self.store_context = None
        self.llm = None
        self.user_id = "demo_user"
    
    async def setup(self):
        """初始化智能助手"""
        print("🚀 初始化智能助手...")
        
        # 创建嵌入模型
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        # 创建存储
        self.store_context = AsyncPostgresStore.from_conn_string(
            memory_config.postgres_url,
            index={
                "embed": embeddings,
                "dims": 1024,
                "fields": ["$"]
            }
        )
        self.store = await self.store_context.__aenter__()
        
        # 创建LLM
        self.llm = init_chat_model("deepseek:deepseek-chat")
        
        print("✅ 智能助手初始化完成！")
    
    async def remember(self, content: str, memory_type: str = "general"):
        """存储记忆"""
        memory_data = {
            "content": content,
            "type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "user_id": self.user_id
        }
        
        namespace = (self.user_id, "memories")
        key = f"memory_{int(datetime.now().timestamp())}"
        
        await self.store.aput(namespace, key, memory_data)
        print(f"💾 已记住：{content}")
    
    async def recall(self, query: str, limit: int = 3):
        """回忆相关记忆"""
        namespace = (self.user_id, "memories")
        results = await self.store.asearch(namespace, query=query, limit=limit)
        
        memories = []
        for result in results:
            memories.append(result.value.get('content', ''))
        
        return memories
    
    async def chat_with_memory(self, message: str) -> str:
        """带记忆的对话"""
        # 先搜索相关记忆
        memories = await self.recall(message)
        
        # 构建包含记忆的提示
        context = ""
        if memories:
            context = f"相关记忆：\n" + "\n".join([f"- {memory}" for memory in memories]) + "\n\n"
        
        # 与LLM对话
        response = await self.llm.ainvoke([
            {"role": "system", "content": f"你是一个智能助手。{context}基于用户的历史记忆来回答问题。"},
            {"role": "user", "content": message}
        ])
        
        return response.content
    
    async def show_memories(self):
        """显示存储的记忆"""
        print("\n📚 当前存储的记忆：")
        
        # 搜索所有记忆
        namespace = (self.user_id, "memories")
        all_memories = await self.store.asearch(namespace, query="", limit=10)
        
        if all_memories:
            for i, memory in enumerate(all_memories, 1):
                data = memory.value
                content = data.get('content', '')
                memory_type = data.get('type', 'general')
                print(f"  {i}. [{memory_type}] {content}")
        else:
            print("  暂无记忆存储")
    
    async def cleanup(self):
        """清理资源"""
        if self.store_context:
            await self.store_context.__aexit__(None, None, None)

async def demo_scenario_1():
    """场景1：个人助手 - 记住用户偏好"""
    print("\n" + "="*60)
    print("🎭 场景1：个人助手 - 记住用户偏好")
    print("="*60)
    
    assistant = SmartAssistant()
    await assistant.setup()
    
    try:
        # 第一次对话 - 告诉助手偏好
        print("\n👤 用户：我喜欢简洁的界面设计，不喜欢太多颜色。")
        await assistant.remember("用户喜欢简洁的界面设计，不喜欢太多颜色", "偏好")
        
        # 第二次对话 - 询问推荐
        print("\n👤 用户：能推荐一个网站设计方案吗？")
        response = await assistant.chat_with_memory("能推荐一个网站设计方案吗？")
        print(f"🤖 助手：{response}")
        
        # 显示记忆
        await assistant.show_memories()
        
    finally:
        await assistant.cleanup()

async def demo_scenario_2():
    """场景2：客服助手 - 记住问题历史"""
    print("\n" + "="*60)
    print("🎭 场景2：客服助手 - 记住问题历史")
    print("="*60)
    
    assistant = SmartAssistant()
    await assistant.setup()
    
    try:
        # 第一次咨询
        print("\n👤 用户：我的订单还没发货，订单号是12345")
        await assistant.remember("用户订单12345还没发货，用户对此有疑问", "客服记录")
        
        # 第二次咨询 - 相关问题
        print("\n👤 用户：我想取消之前咨询的那个订单")
        response = await assistant.chat_with_memory("我想取消之前咨询的那个订单")
        print(f"🤖 助手：{response}")
        
        # 显示记忆
        await assistant.show_memories()
        
    finally:
        await assistant.cleanup()

async def demo_scenario_3():
    """场景3：学习助手 - 记住学习进度"""
    print("\n" + "="*60)
    print("🎭 场景3：学习助手 - 记住学习进度")
    print("="*60)
    
    assistant = SmartAssistant()
    await assistant.setup()
    
    try:
        # 记录学习进度
        print("\n👤 用户：我正在学习Python，已经掌握了基础语法和函数")
        await assistant.remember("用户正在学习Python，已掌握基础语法和函数", "学习进度")
        
        # 询问下一步学习建议
        print("\n👤 用户：我接下来应该学什么？")
        response = await assistant.chat_with_memory("我接下来应该学什么？")
        print(f"🤖 助手：{response}")
        
        # 显示记忆
        await assistant.show_memories()
        
    finally:
        await assistant.cleanup()

def explain_benefits():
    """解释LangMem的好处"""
    print("\n" + "="*60)
    print("💡 LangMem 的核心价值")
    print("="*60)
    
    benefits = [
        "🎯 个性化体验：记住用户偏好，提供定制化服务",
        "🔄 上下文连续性：跨对话记住重要信息",
        "📈 智能学习：从历史对话中学习用户习惯",
        "⚡ 效率提升：避免重复询问相同信息",
        "🎭 角色一致性：保持助手的个性和专业性",
        "📊 数据积累：为后续优化提供数据基础"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print("\n🚀 在你的项目中的应用场景：")
    scenarios = [
        "📱 聊天机器人：记住用户偏好和历史问题",
        "🛒 电商助手：记住购买偏好和浏览历史",
        "📚 学习平台：跟踪学习进度和知识点掌握",
        "💼 工作助手：记住项目信息和工作习惯",
        "🏥 医疗助手：记住病史和治疗偏好（注意隐私）"
    ]
    
    for scenario in scenarios:
        print(f"  {scenario}")

async def main():
    """主演示函数"""
    print("🧠 LangMem 实用价值演示")
    print("这个演示将展示LangMem如何让AI助手变得更加智能和有用")
    
    # 解释好处
    explain_benefits()
    
    # 运行演示场景
    await demo_scenario_1()
    await demo_scenario_2()
    await demo_scenario_3()
    
    print("\n" + "="*60)
    print("🎉 演示完成！")
    print("="*60)
    print("通过这些例子，你可以看到LangMem如何让AI助手：")
    print("1. 记住用户的个人偏好")
    print("2. 保持对话的连续性")
    print("3. 提供更加个性化的服务")
    print("4. 避免重复询问相同信息")
    print("\n这就是LangMem在你的项目中的核心价值！")

if __name__ == "__main__":
    asyncio.run(main())