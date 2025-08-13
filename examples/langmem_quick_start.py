#!/usr/bin/env python3
"""
LangMem 简单使用示例
展示如何在5分钟内为你的AI添加记忆功能
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

class SimpleMemoryBot:
    """5分钟搭建一个有记忆的AI助手"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.store = None
        self.store_context = None
        self.llm = None
    
    async def setup(self):
        """初始化（只需要这几行代码）"""
        print(f"🚀 为用户 {self.user_id} 初始化记忆功能...")
        
        # 1. 创建嵌入模型（用于语义搜索）
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        # 2. 创建记忆存储
        self.store_context = AsyncPostgresStore.from_conn_string(
            memory_config.postgres_url,
            index={"embed": embeddings, "dims": 1024, "fields": ["$"]}
        )
        self.store = await self.store_context.__aenter__()
        
        # 3. 创建LLM
        self.llm = init_chat_model("deepseek:deepseek-chat")
        
        print("✅ 记忆功能初始化完成！")
    
    async def remember(self, content: str):
        """记住某件事（核心功能1）"""
        namespace = (self.user_id, "memories")
        key = f"memory_{len(await self.store.asearch(namespace))}"
        
        await self.store.aput(namespace, key, {"content": content})
        print(f"💾 已记住：{content}")
    
    async def recall(self, query: str):
        """回忆相关内容（核心功能2）"""
        namespace = (self.user_id, "memories")
        results = await self.store.asearch(namespace, query=query, limit=3)
        
        memories = [r.value["content"] for r in results]
        return memories
    
    async def chat_with_memory(self, message: str):
        """带记忆的对话（核心功能3）"""
        # 搜索相关记忆
        memories = await self.recall(message)
        
        # 构建提示
        context = ""
        if memories:
            context = f"相关记忆：{'; '.join(memories)}\n\n"
        
        prompt = f"{context}用户问：{message}\n请基于记忆回答："
        
        # 生成回答
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content
    
    async def cleanup(self):
        """清理资源"""
        if self.store_context:
            await self.store_context.__aexit__(None, None, None)

async def quick_demo():
    """5分钟快速演示"""
    print("⚡ 5分钟LangMem快速演示")
    print("="*50)
    
    # 创建记忆机器人
    bot = SimpleMemoryBot("quick_demo_user")
    await bot.setup()
    
    try:
        print("\n📝 步骤1：让AI记住一些信息")
        await bot.remember("用户是程序员，喜欢Python")
        await bot.remember("用户正在学习LangChain")
        await bot.remember("用户偏好简洁的代码风格")
        
        print("\n💬 步骤2：测试AI是否记住了")
        
        questions = [
            "我是做什么工作的？",
            "推荐一个编程语言给我",
            "我在学什么技术？"
        ]
        
        for question in questions:
            print(f"\n👤 用户：{question}")
            answer = await bot.chat_with_memory(question)
            print(f"🤖 AI：{answer}")
        
        print("\n🎉 看到了吗？AI记住了你的信息！")
        
    finally:
        await bot.cleanup()

def show_integration_steps():
    """展示集成步骤"""
    print("\n" + "="*50)
    print("🔧 如何在你的项目中使用LangMem")
    print("="*50)
    
    print("\n📋 只需3个步骤：")
    
    print("\n1️⃣ 安装和配置")
    print("```bash")
    print("pip install langmem")
    print("# 配置环境变量：DASHSCOPE_API_KEY")
    print("```")
    
    print("\n2️⃣ 初始化记忆存储")
    print("```python")
    print("from langgraph.store.postgres import AsyncPostgresStore")
    print("from langchain_community.embeddings import DashScopeEmbeddings")
    print("")
    print("# 创建存储")
    print("embeddings = DashScopeEmbeddings(model='text-embedding-v4')")
    print("store = AsyncPostgresStore.from_conn_string(db_url, index={'embed': embeddings})")
    print("```")
    
    print("\n3️⃣ 在对话中使用")
    print("```python")
    print("# 存储记忆")
    print("await store.aput((user_id, 'memories'), 'key', {'content': '用户信息'})")
    print("")
    print("# 搜索记忆")
    print("memories = await store.asearch((user_id, 'memories'), query='搜索内容')")
    print("")
    print("# 在对话中使用记忆")
    print("context = '\\n'.join([m.value['content'] for m in memories])")
    print("prompt = f'{context}\\n\\n用户：{user_message}'")
    print("```")

async def main():
    """主函数"""
    print("🧠 LangMem 完全指南")
    print("从零开始，5分钟掌握AI记忆功能")
    
    # 快速演示
    await quick_demo()
    
    # 集成步骤
    show_integration_steps()
    
    print("\n" + "="*50)
    print("💡 总结：LangMem的价值")
    print("="*50)
    
    values = [
        "🎯 让AI记住用户信息，提供个性化服务",
        "🔄 跨对话保持上下文，避免重复介绍", 
        "📈 积累用户数据，持续改善体验",
        "⚡ 简单易用，几行代码就能集成",
        "🔒 数据安全，本地存储完全可控"
    ]
    
    for value in values:
        print(f"  {value}")
    
    print("\n🚀 现在你知道LangMem的作用了：")
    print("  让你的AI从'健忘症患者'变成'贴心助手'！")

if __name__ == "__main__":
    asyncio.run(main())