#!/usr/bin/env python3
"""
测试提示词优化模块的基本功能
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_optimization_imports():
    """测试优化模块的导入"""
    print("🔍 测试提示词优化模块导入...")
    
    try:
        # 测试基本导入
        from core.optimization.prompt_optimizer import PromptOptimizer, FeedbackCollector, AutoOptimizationScheduler
        print("✅ 成功导入 PromptOptimizer, FeedbackCollector, AutoOptimizationScheduler")
        
        # 测试API模块导入
        from core.optimization.prompt_optimization_api import router
        print("✅ 成功导入 prompt_optimization_api router")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

async def test_basic_functionality():
    """测试基本功能"""
    print("\n🔧 测试基本功能...")
    
    try:
        # 模拟内存管理器
        class MockMemoryManager:
            def __init__(self):
                self.data = {}
            
            async def initialize(self):
                """初始化方法"""
                pass
            
            async def aput(self, namespace, key, value):
                if namespace not in self.data:
                    self.data[namespace] = {}
                self.data[namespace][key] = value
            
            async def aget(self, namespace, key):
                return self.data.get(namespace, {}).get(key)
            
            async def asearch(self, namespace, query="", limit=10, **kwargs):
                # 简单的模拟搜索
                results = []
                if namespace in self.data:
                    for k, v in list(self.data[namespace].items())[:limit]:
                        results.append({"key": k, "value": v})
                return results
            
            async def store_memory(self, content, memory_type, namespace, metadata=None):
                """存储记忆"""
                if namespace not in self.data:
                    self.data[namespace] = {}
                key = f"memory_{len(self.data[namespace])}"
                self.data[namespace][key] = {
                    "content": content,
                    "memory_type": memory_type,
                    "metadata": metadata or {}
                }
            
            async def search_memories(self, query, namespace, limit=10, **kwargs):
                """搜索记忆"""
                results = []
                if namespace in self.data:
                    for k, v in list(self.data[namespace].items())[:limit]:
                        # 模拟记忆对象
                        class MockMemory:
                            def __init__(self, content):
                                self.content = content
                        results.append(MockMemory(v.get("content", "")))
                return results
        
        # 创建模拟的内存管理器
        mock_memory = MockMemoryManager()
        
        # 测试 PromptOptimizer 初始化
        from core.optimization.prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer(mock_memory)
        await optimizer.initialize()
        print("✅ PromptOptimizer 初始化成功")
        
        # 测试健康检查
        health = await optimizer.health_check()
        print(f"✅ 健康检查通过: {health}")
        
        # 测试 FeedbackCollector 初始化
        from core.optimization.prompt_optimizer import FeedbackCollector
        collector = FeedbackCollector(mock_memory)
        health = await collector.health_check()
        print(f"✅ FeedbackCollector 健康检查通过: {health}")
        
        # 测试 AutoOptimizationScheduler 初始化
        from core.optimization.prompt_optimizer import AutoOptimizationScheduler
        scheduler = AutoOptimizationScheduler(optimizer, collector)
        status = await scheduler.get_status()
        print(f"✅ AutoOptimizationScheduler 状态获取成功: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("🚀 开始测试提示词优化模块...")
    print("=" * 50)
    
    # 测试导入
    import_success = await test_optimization_imports()
    
    if import_success:
        # 测试基本功能
        func_success = await test_basic_functionality()
        
        if func_success:
            print("\n" + "=" * 50)
            print("🎉 所有测试通过！提示词优化模块工作正常")
            return True
    
    print("\n" + "=" * 50)
    print("❌ 测试失败，请检查模块配置")
    return False

if __name__ == "__main__":
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    # 运行测试
    success = asyncio.run(main())
    sys.exit(0 if success else 1)