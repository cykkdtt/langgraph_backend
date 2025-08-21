#!/usr/bin/env python3
"""
LangMem 集成测试脚本

本脚本用于测试LangMem记忆管理系统的集成情况，包括：
- 记忆存储管理器初始化
- 记忆工具创建和使用
- 记忆增强智能体功能
- 端到端记忆功能测试
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.memory import (
    get_memory_manager,
    MemoryNamespace,
    MemoryScope,
    MemoryType,
    MemoryItem,
    MemoryQuery
)
from core.memory.store_manager import get_memory_store_manager
from core.memory.tools import get_memory_tools
from core.agents.memory_enhanced import MemoryEnhancedAgent
from config.settings import get_settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_memory_store_manager():
    """测试记忆存储管理器"""
    print("\n🔧 测试记忆存储管理器")
    print("-" * 40)
    
    try:
        # 获取存储管理器
        store_manager = await get_memory_store_manager()
        
        print("✅ 记忆存储管理器初始化成功")
        
        # 测试健康检查
        health = await store_manager.health_check()
        print(f"📊 健康检查: {health}")
        
        # 测试统计信息
        stats = await store_manager.get_stats()
        print(f"📈 统计信息: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 记忆存储管理器测试失败: {e}")
        return False


async def test_memory_tools():
    """测试记忆工具"""
    print("\n🛠️ 测试记忆工具")
    print("-" * 40)
    
    try:
        # 创建记忆工具
        namespace = "test_tools"
        tools = await get_memory_tools(namespace)
        
        print(f"✅ 创建记忆工具成功: {len(tools)} 个工具")
        
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ 记忆工具测试失败: {e}")
        return False


async def test_memory_manager():
    """测试记忆管理器"""
    print("\n🧠 测试记忆管理器")
    print("-" * 40)
    
    try:
        # 获取记忆管理器
        memory_manager = get_memory_manager()
        await memory_manager.initialize()
        
        print("✅ 记忆管理器初始化成功")
        
        # 创建测试命名空间
        namespace = MemoryNamespace(
            scope=MemoryScope.USER,
            identifier="test_user"
        )
        
        # 测试存储记忆
        test_memory = MemoryItem(
            id="test_memory_001",
            content="这是一个测试记忆，用于验证LangMem集成功能。",
            memory_type=MemoryType.SEMANTIC,
            metadata={"test": True, "category": "integration_test"},
            importance=0.8
        )
        
        memory_id = await memory_manager.store_memory(namespace, test_memory)
        print(f"✅ 存储记忆成功: {memory_id}")
        
        # 测试检索记忆
        retrieved_memory = await memory_manager.retrieve_memory(namespace, memory_id)
        if retrieved_memory:
            print(f"✅ 检索记忆成功: {retrieved_memory.content[:50]}...")
        else:
            print("❌ 检索记忆失败")
            return False
        
        # 测试搜索记忆
        query = MemoryQuery(
            query="测试记忆",
            memory_type=MemoryType.SEMANTIC,
            limit=5
        )
        
        search_results = await memory_manager.search_memories(namespace, query)
        print(f"✅ 搜索记忆成功: 找到 {len(search_results)} 条记忆")
        
        # 测试记忆统计
        stats = await memory_manager.get_memory_stats(namespace)
        print(f"📊 记忆统计: {stats}")
        
        # 清理测试记忆
        deleted = await memory_manager.delete_memory(namespace, memory_id)
        if deleted:
            print("✅ 清理测试记忆成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 记忆管理器测试失败: {e}")
        return False


async def test_memory_enhanced_agent():
    """测试记忆增强智能体"""
    print("\n🤖 测试记忆增强智能体")
    print("-" * 40)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from core.agents.base import ChatRequest
        
        settings = get_settings()
        
        # 创建语言模型
        llm = ChatOpenAI(
            model="qwen-turbo",
            temperature=0.7,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 创建记忆增强智能体
        agent = MemoryEnhancedAgent(
            agent_id="test_memory_agent",
            name="测试记忆智能体",
            description="用于测试记忆功能的智能体",
            llm=llm,
            memory_config={
                "auto_store": True,
                "retrieval_limit": 3,
                "importance_threshold": 0.3
            }
        )
        
        await agent.initialize()
        print("✅ 记忆增强智能体初始化成功")
        
        # 测试存储知识
        knowledge_id = await agent.store_knowledge(
            content="Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。",
            user_id="test_user",
            memory_type=MemoryType.SEMANTIC,
            importance=0.7
        )
        print(f"✅ 存储知识成功: {knowledge_id}")
        
        # 测试对话功能
        request = ChatRequest(
            messages=[HumanMessage(content="你好，请介绍一下Python编程语言。")],
            user_id="test_user",
            session_id="test_session",
            stream=False
        )
        
        response = await agent.chat(request)
        if response.message:
            print(f"✅ 对话测试成功")
            print(f"智能体回复: {response.message.content[:100]}...")
        else:
            print("❌ 对话测试失败")
            return False
        
        # 测试记忆统计
        stats = await agent.get_memory_stats("test_user", "test_session")
        print(f"📊 智能体记忆统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 记忆增强智能体测试失败: {e}")
        return False


async def test_langmem_integration():
    """测试LangMem原生集成"""
    print("\n🔗 测试LangMem原生集成")
    print("-" * 40)
    
    try:
        # 测试LangMem模块导入
        from langmem import (
            create_memory_manager,
            create_memory_store_manager,
            create_manage_memory_tool,
            create_search_memory_tool
        )
        from langchain_openai import ChatOpenAI
        from config.settings import get_settings
        print("✅ LangMem模块导入成功")
        
        # 创建语言模型实例
        settings = get_settings()
        llm = ChatOpenAI(
            model="qwen-turbo",
            temperature=0.7,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 测试创建记忆管理器
        langmem_manager = create_memory_manager(llm)
        print("✅ LangMem记忆管理器创建成功")
        
        # 测试创建记忆工具
        manage_tool = create_manage_memory_tool("test_namespace")
        search_tool = create_search_memory_tool("test_namespace")
        print("✅ LangMem记忆工具创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ LangMem原生集成测试失败: {e}")
        return False


async def run_integration_tests():
    """运行集成测试"""
    print("🚀 LangMem 集成测试开始")
    print("=" * 60)
    
    test_results = []
    
    # 测试列表
    tests = [
        ("LangMem原生集成", test_langmem_integration),
        ("记忆存储管理器", test_memory_store_manager),
        ("记忆工具", test_memory_tools),
        ("记忆管理器", test_memory_manager),
        ("记忆增强智能体", test_memory_enhanced_agent),
    ]
    
    # 运行测试
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"测试 {test_name} 出现异常: {e}")
            test_results.append((test_name, False))
    
    # 显示测试结果
    print("\n📋 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！LangMem集成成功！")
        return True
    else:
        print("⚠️ 部分测试失败，请检查配置和依赖")
        return False


async def main():
    """主函数"""
    try:
        success = await run_integration_tests()
        
        if success:
            print("\n🎯 LangMem集成测试完成，可以开始使用记忆增强功能！")
            print("\n下一步建议:")
            print("1. 运行示例应用: python examples/memory_enhanced_demo.py")
            print("2. 查看记忆配置: config/memory_config.py")
            print("3. 阅读文档: spc/05_langmem_integration.md")
        else:
            print("\n❌ 集成测试失败，请检查配置和依赖")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 测试被用户中断")
    except Exception as e:
        logger.error(f"测试运行出错: {e}")
        print(f"\n❌ 测试运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())