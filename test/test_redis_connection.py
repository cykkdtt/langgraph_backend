#!/usr/bin/env python3
"""
Redis连接测试脚本

测试Redis连接配置是否正确
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.cache.redis_manager import get_cache_manager, get_redis_manager
from config.settings import Settings
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_redis_connection():
    """测试Redis连接"""
    print("🔧 Redis连接测试")
    print("=" * 50)
    
    try:
        # 获取设置
        settings = Settings()
        print(f"📋 Redis配置:")
        print(f"   主机: {settings.database.redis_host}")
        print(f"   端口: {settings.database.redis_port}")
        print(f"   数据库: {settings.database.redis_db}")
        print(f"   连接URL: {settings.database.redis_url.replace(':' + (settings.database.redis_password or ''), ':***')}")
        print()
        
        # 获取缓存管理器
        cache_manager = await get_cache_manager()
        
        # 健康检查
        health = await cache_manager.health_check()
        print(f"🏥 健康检查结果:")
        for key, value in health.items():
            print(f"   {key}: {value}")
        print()
        
        if health.get("status") == "healthy":
            print("✅ Redis连接成功!")
            
            # 测试基本操作
            redis_manager = await get_redis_manager()
            
            # 测试设置和获取
            test_key = "test:connection"
            test_value = {"message": "Hello Redis!", "timestamp": "2024-01-01"}
            
            print("🧪 测试基本操作:")
            
            # 设置值
            set_result = await redis_manager.set(test_key, test_value, 60)
            print(f"   设置值: {set_result}")
            
            # 获取值
            get_result = await redis_manager.get(test_key)
            print(f"   获取值: {get_result}")
            
            # 检查存在
            exists_result = await redis_manager.exists(test_key)
            print(f"   键存在: {exists_result}")
            
            # 删除值
            delete_result = await redis_manager.delete(test_key)
            print(f"   删除值: {delete_result}")
            
            print("\n🎉 所有测试通过!")
            
        else:
            print("❌ Redis连接失败!")
            return False
            
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        print(f"❌ 测试失败: {e}")
        return False
    
    finally:
        # 清理资源
        try:
            cache_manager = await get_cache_manager()
            await cache_manager.cleanup()
        except:
            pass
    
    return True

async def main():
    """主函数"""
    success = await test_redis_connection()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())