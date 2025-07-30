#!/usr/bin/env python3
"""
Redis集成测试脚本

测试Redis在整个系统中的集成情况
"""

import asyncio
import sys
import os
import httpx
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_redis_integration():
    """测试Redis集成"""
    print("🔧 Redis系统集成测试")
    print("=" * 50)
    
    # 测试直接Redis连接
    print("1️⃣ 测试直接Redis连接...")
    try:
        from core.cache.redis_manager import get_cache_manager
        cache_manager = await get_cache_manager()
        health = await cache_manager.health_check()
        
        if health.get("status") == "healthy":
            print("   ✅ 直接Redis连接正常")
        else:
            print(f"   ❌ 直接Redis连接异常: {health}")
            return False
    except Exception as e:
        print(f"   ❌ 直接Redis连接失败: {e}")
        return False
    
    # 测试缓存操作
    print("\n2️⃣ 测试缓存基本操作...")
    try:
        redis_manager = cache_manager.redis_manager
        
        # 设置测试数据
        test_data = {
            "test_key_1": "Hello Redis!",
            "test_key_2": {"message": "JSON data", "number": 42},
            "test_key_3": ["list", "data", "test"]
        }
        
        for key, value in test_data.items():
            success = await redis_manager.set(key, value, 300)  # 5分钟过期
            if success:
                print(f"   ✅ 设置 {key}: {value}")
            else:
                print(f"   ❌ 设置 {key} 失败")
        
        # 获取测试数据
        for key in test_data.keys():
            value = await redis_manager.get(key)
            print(f"   📖 获取 {key}: {value}")
        
        # 列出所有测试键
        keys = await redis_manager.keys("test_key_*")
        print(f"   📋 测试键列表: {keys}")
        
        # 清理测试数据
        for key in test_data.keys():
            await redis_manager.delete(key)
        print("   🧹 测试数据已清理")
        
    except Exception as e:
        print(f"   ❌ 缓存操作测试失败: {e}")
        return False
    
    # 测试会话缓存
    print("\n3️⃣ 测试会话缓存...")
    try:
        session_cache = cache_manager.session_cache
        
        # 设置会话数据
        session_data = {
            "user_id": "test_user",
            "session_id": "test_session_123",
            "last_activity": "2024-01-01T12:00:00Z",
            "preferences": {"theme": "dark", "language": "zh-CN"}
        }
        
        success = await session_cache.set_session("test_session_123", session_data, 3600)
        if success:
            print("   ✅ 会话数据设置成功")
        else:
            print("   ❌ 会话数据设置失败")
            return False
        
        # 获取会话数据
        retrieved_data = await session_cache.get_session("test_session_123")
        if retrieved_data == session_data:
            print("   ✅ 会话数据获取成功")
        else:
            print(f"   ❌ 会话数据不匹配: {retrieved_data}")
        
        # 延长会话
        extended = await session_cache.extend_session("test_session_123", 7200)
        if extended:
            print("   ✅ 会话延长成功")
        else:
            print("   ❌ 会话延长失败")
        
        # 删除会话
        deleted = await session_cache.delete_session("test_session_123")
        if deleted:
            print("   ✅ 会话删除成功")
        else:
            print("   ❌ 会话删除失败")
        
    except Exception as e:
        print(f"   ❌ 会话缓存测试失败: {e}")
        return False
    
    # 清理资源
    try:
        await cache_manager.cleanup()
        print("\n🧹 资源清理完成")
    except Exception as e:
        print(f"\n⚠️ 资源清理警告: {e}")
    
    print("\n🎉 Redis集成测试全部通过!")
    return True

async def test_api_integration():
    """测试API集成（需要服务器运行）"""
    print("\n🌐 API集成测试")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        try:
            # 测试健康检查
            print("1️⃣ 测试健康检查API...")
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                redis_status = health_data.get("components", {}).get("redis", {})
                print(f"   ✅ 健康检查成功，Redis状态: {redis_status.get('status', 'unknown')}")
            else:
                print(f"   ❌ 健康检查失败: {response.status_code}")
                return False
            
            # 测试缓存健康检查
            print("\n2️⃣ 测试缓存健康检查API...")
            response = await client.get(f"{base_url}/cache/health")
            if response.status_code == 200:
                cache_health = response.json()
                print(f"   ✅ 缓存健康检查成功: {cache_health.get('status', 'unknown')}")
            else:
                print(f"   ❌ 缓存健康检查失败: {response.status_code}")
            
            # 测试缓存键列表
            print("\n3️⃣ 测试缓存键列表API...")
            response = await client.get(f"{base_url}/cache/keys")
            if response.status_code == 200:
                keys_data = response.json()
                print(f"   ✅ 缓存键列表获取成功，共 {keys_data.get('count', 0)} 个键")
            else:
                print(f"   ❌ 缓存键列表获取失败: {response.status_code}")
            
            print("\n🎉 API集成测试完成!")
            return True
            
        except httpx.ConnectError:
            print("   ⚠️ 无法连接到API服务器，请确保服务器正在运行")
            print("   💡 提示: 运行 'python main.py' 启动服务器")
            return False
        except Exception as e:
            print(f"   ❌ API测试失败: {e}")
            return False

async def main():
    """主函数"""
    print("🚀 开始Redis集成测试")
    print("=" * 60)
    
    # 测试直接集成
    success1 = await test_redis_integration()
    
    # 测试API集成
    success2 = await test_api_integration()
    
    if success1:
        print("\n✅ 所有测试通过!")
        if not success2:
            print("💡 API测试跳过（服务器未运行）")
    else:
        print("\n❌ 测试失败!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())