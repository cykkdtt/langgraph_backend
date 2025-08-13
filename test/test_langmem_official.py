#!/usr/bin/env python3
"""
LangMem 官方测试脚本

基于LangMem官方文档实现的测试脚本，包括：
1. 热路径记忆管理（智能体主动保存记忆）
2. 后台记忆提取（自动从对话中提取记忆）
3. 语义记忆提取（结构化事实存储）
4. 与阿里云嵌入模型的集成
"""

import asyncio
import logging
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel

# 首先加载环境变量
load_dotenv()

# LangMem 核心组件
from langmem import (
    create_manage_memory_tool,
    create_search_memory_tool,
    create_memory_store_manager,
    create_memory_manager,
    ReflectionExecutor
)

# LangGraph 组件
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.func import entrypoint

# LangChain 组件
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings

# 添加项目根目录到Python路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 本地配置
from config.memory_config import memory_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_embeddings():
    """创建阿里云嵌入模型实例"""
    return DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )


def create_store():
    """创建存储实例"""
    embeddings = create_embeddings()
    
    if memory_config.store_type == "postgres":
        # 使用异步PostgreSQL存储 - 返回上下文管理器
        return AsyncPostgresStore.from_conn_string(
            memory_config.postgres_url,
            index={
                "embed": embeddings,
                "dims": memory_config.embedding_dims,
                "fields": ["$"]
            }
        )
    else:
        # 使用内存存储
        return InMemoryStore(
            index={
                "embed": embeddings,
                "dims": memory_config.embedding_dims,
                "fields": ["$"]
            }
        )


class Triple(BaseModel):
    """语义记忆的三元组结构"""
    subject: str
    predicate: str
    object: str
    context: str | None = None


class LangMemTester:
    """LangMem 功能测试器"""
    
    def __init__(self):
        self.store = None
        self.store_context = None
        self.llm = None
        self.agent = None
        self.memory_manager = None
        self.semantic_manager = None
    
    async def setup(self):
        """设置测试环境"""
        print("🔧 设置LangMem测试环境...")
        
        # 初始化LLM - 使用DeepSeek模型
        self.llm = init_chat_model("deepseek:deepseek-chat")
        print("✅ LLM初始化完成")
        
        # 创建存储
        if memory_config.store_type == "postgres":
            self.store_context = create_store()
            self.store = await self.store_context.__aenter__()
            print("✅ PostgreSQL存储初始化完成")
        else:
            self.store = create_store()
            print("✅ 内存存储初始化完成")
        
        # 创建带记忆工具的智能体（热路径）
        self.agent = create_react_agent(
            self.llm,
            tools=[
                create_manage_memory_tool(
                    store=self.store,
                    namespace=("user_memories",)
                ),
                create_search_memory_tool(
                    store=self.store,
                    namespace=("user_memories",)
                ),
            ],
            store=self.store,
        )
        print("✅ 记忆增强智能体创建完成")
        
        # 创建后台记忆管理器
        self.memory_manager = create_memory_store_manager(
            "deepseek:deepseek-chat",
            store=self.store,
            namespace=("background_memories",)
        )
        print("✅ 后台记忆管理器创建完成")
        
        # 创建语义记忆管理器
        self.semantic_manager = create_memory_manager(
            "deepseek:deepseek-chat",
            schemas=[Triple],
            instructions="提取用户偏好和任何其他有用信息作为三元组",
            enable_inserts=True,
            enable_deletes=True,
        )
        print("✅ 语义记忆管理器创建完成")
        
        print("🎉 LangMem测试环境设置完成！")
    
    async def test_hot_path_memory(self):
        """测试热路径记忆管理（智能体主动保存记忆）"""
        print("\n🔥 测试热路径记忆管理...")
        
        try:
            # 让智能体记住用户偏好
            response1 = await self.agent.ainvoke({
                "messages": [{"role": "user", "content": "请记住我喜欢深色模式界面。"}]
            })
            print(f"智能体回应: {response1['messages'][-1].content}")
            
            # 询问之前的偏好
            response2 = await self.agent.ainvoke({
                "messages": [{"role": "user", "content": "我的界面偏好是什么？"}]
            })
            print(f"智能体回应: {response2['messages'][-1].content}")
            
            # 搜索存储的记忆
            memories = await self.store.asearch(("user_memories",))
            print(f"存储的记忆数量: {len(memories)}")
            for memory in memories:
                print(f"记忆内容: {memory.value}")
            
            print("✅ 热路径记忆管理测试完成")
            
        except Exception as e:
            print(f"❌ 热路径记忆管理测试失败: {e}")
            logger.error(f"热路径测试错误: {e}", exc_info=True)
    
    async def test_background_memory(self):
        """测试后台记忆提取"""
        print("\n🔄 测试后台记忆提取...")
        
        try:
            # 模拟对话
            conversation = {
                "messages": [
                    {"role": "user", "content": "我喜欢狗。我的狗叫Fido。"},
                    {"role": "assistant", "content": "那很棒！狗是很好的伙伴。Fido是个经典的狗名字。Fido是什么品种的狗？"}
                ]
            }
            
            # 后台提取记忆
            await self.memory_manager.ainvoke(conversation)
            print("✅ 后台记忆提取完成")
            
            # 查看提取的记忆
            background_memories = await self.store.asearch(("background_memories",))
            print(f"后台提取的记忆数量: {len(background_memories)}")
            for memory in background_memories:
                print(f"后台记忆: {memory.value}")
            
            print("✅ 后台记忆提取测试完成")
            
        except Exception as e:
            print(f"❌ 后台记忆提取测试失败: {e}")
            logger.error(f"后台记忆测试错误: {e}", exc_info=True)
    
    async def test_semantic_memory(self):
        """测试语义记忆提取（三元组）"""
        print("\n🧠 测试语义记忆提取...")
        
        try:
            # 第一次对话 - 提取三元组
            conversation1 = [
                {"role": "user", "content": "Alice管理ML团队并指导Bob，Bob也在这个团队。"}
            ]
            
            memories = self.semantic_manager.invoke({"messages": conversation1})
            print("第一次对话后的记忆:")
            for memory in memories:
                print(f"  {memory.content}")
            
            # 第二次对话 - 更新三元组
            conversation2 = [
                {"role": "user", "content": "Bob现在领导ML团队和NLP项目。"}
            ]
            
            updated_memories = self.semantic_manager.invoke({
                "messages": conversation2, 
                "existing": memories
            })
            print("\n第二次对话后的记忆更新:")
            for memory in updated_memories:
                print(f"  {memory.content}")
            
            # 第三次对话 - 删除相关记忆
            existing_triples = [m for m in updated_memories if isinstance(m.content, Triple)]
            conversation3 = [
                {"role": "user", "content": "Alice离开了公司。"}
            ]
            
            final_memories = self.semantic_manager.invoke({
                "messages": conversation3,
                "existing": existing_triples
            })
            print("\n第三次对话后的记忆:")
            for memory in final_memories:
                print(f"  {memory.content}")
            
            print("✅ 语义记忆提取测试完成")
            
        except Exception as e:
            print(f"❌ 语义记忆提取测试失败: {e}")
            logger.error(f"语义记忆测试错误: {e}", exc_info=True)
    
    async def test_memory_search(self):
        """测试记忆搜索功能"""
        print("\n🔍 测试记忆搜索功能...")
        
        try:
            # 搜索用户记忆
            user_memories = await self.store.asearch(("user_memories",), query="界面偏好")
            print(f"用户记忆搜索结果: {len(user_memories)} 条")
            for memory in user_memories:
                print(f"  {memory.value}")
            
            # 搜索后台记忆
            background_memories = await self.store.asearch(("background_memories",), query="狗")
            print(f"后台记忆搜索结果: {len(background_memories)} 条")
            for memory in background_memories:
                print(f"  {memory.value}")
            
            print("✅ 记忆搜索测试完成")
            
        except Exception as e:
            print(f"❌ 记忆搜索测试失败: {e}")
            logger.error(f"记忆搜索测试错误: {e}", exc_info=True)
    
    async def test_memory_persistence(self):
        """测试记忆持久化"""
        print("\n💾 测试记忆持久化...")
        
        try:
            # 直接向存储添加记忆
            test_memory = {
                "type": "preference",
                "content": "用户喜欢简洁的界面设计",
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            await self.store.aput(("test_memories",), "pref_001", test_memory)
            print("✅ 记忆存储成功")
            
            # 读取记忆
            retrieved = await self.store.aget(("test_memories",), "pref_001")
            print(f"读取的记忆: {retrieved}")
            
            # 列出所有命名空间
            namespaces = [item.namespace for item in await self.store.asearch(())]
            unique_namespaces = list(set(tuple(ns) for ns in namespaces))
            print(f"所有命名空间: {unique_namespaces}")
            
            print("✅ 记忆持久化测试完成")
            
        except Exception as e:
            print(f"❌ 记忆持久化测试失败: {e}")
            logger.error(f"记忆持久化测试错误: {e}", exc_info=True)
    
    async def cleanup(self):
        """清理测试环境"""
        print("\n🧹 清理测试环境...")
        
        try:
            if self.store_context:
                await self.store_context.__aexit__(None, None, None)
                print("✅ PostgreSQL存储连接已关闭")
            elif self.store and hasattr(self.store, 'close'):
                await self.store.close()
                print("✅ 存储连接已关闭")
            
            print("✅ 测试环境清理完成")
            
        except Exception as e:
            print(f"⚠️  清理过程中出现警告: {e}")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始LangMem官方功能测试")
        print(f"存储类型: {memory_config.store_type}")
        print(f"嵌入模型: {memory_config.embedding_model}")
        print(f"向量维度: {memory_config.embedding_dims}")
        
        try:
            await self.setup()
            
            # 运行各项测试
            await self.test_hot_path_memory()
            await self.test_background_memory()
            await self.test_semantic_memory()
            await self.test_memory_search()
            await self.test_memory_persistence()
            
            print("\n🎉 所有LangMem测试完成！")
            return True
            
        except Exception as e:
            print(f"\n❌ 测试过程中出现错误: {e}")
            logger.error(f"测试失败: {e}", exc_info=True)
            return False
        
        finally:
            await self.cleanup()


async def main():
    """主函数"""
    tester = LangMemTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ LangMem测试成功完成！")
    else:
        print("\n❌ LangMem测试失败！")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())