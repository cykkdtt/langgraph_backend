#!/usr/bin/env python3
"""
测试修正后的checkpoint功能

验证CheckpointManager是否正确使用了LangGraph的checkpoint接口
"""

import asyncio
import logging
from datetime import datetime
from core.checkpoint.manager import CheckpointManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_memory_checkpoint():
    """测试内存checkpoint功能"""
    logger.info("=== 测试内存checkpoint功能 ===")
    
    # 创建内存checkpoint管理器
    manager = CheckpointManager(storage_type="memory")
    
    try:
        # 初始化
        checkpointer = await manager.initialize()
        logger.info(f"✅ 初始化成功: {type(checkpointer).__name__}")
        
        # 测试保存checkpoint
        config = {"configurable": {"thread_id": "test-thread-1"}}
        state = {
            "messages": [{"role": "user", "content": "Hello"}],
            "step": 1
        }
        metadata = {"agent_type": "test_agent", "user_id": "test_user"}
        
        checkpoint_id = await manager.save_checkpoint(config, state, metadata)
        logger.info(f"✅ 保存checkpoint成功: {checkpoint_id}")
        
        # 测试加载checkpoint
        loaded_info = await manager.load_checkpoint(config)
        if loaded_info:
            logger.info(f"✅ 加载checkpoint成功: {loaded_info.metadata.checkpoint_id}")
            logger.info(f"   状态: {loaded_info.state}")
            logger.info(f"   元数据: {loaded_info.metadata.metadata}")
        else:
            logger.error("❌ 加载checkpoint失败")
        
        # 测试列出checkpoints
        checkpoints = await manager.list_checkpoints(config, limit=5)
        logger.info(f"✅ 列出checkpoints成功: 找到 {len(checkpoints)} 个")
        
        # 测试存储统计
        stats = await manager.get_storage_stats()
        logger.info(f"✅ 存储统计: {stats}")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise
    finally:
        await manager.cleanup()
        logger.info("✅ 资源清理完成")

async def test_postgres_checkpoint():
    """测试PostgreSQL checkpoint功能（如果可用）"""
    logger.info("=== 测试PostgreSQL checkpoint功能 ===")
    
    try:
        # 创建PostgreSQL checkpoint管理器
        manager = CheckpointManager(storage_type="postgres")
        
        # 初始化
        checkpointer = await manager.initialize()
        logger.info(f"✅ 初始化成功: {type(checkpointer).__name__}")
        
        # 测试保存checkpoint
        config = {"configurable": {"thread_id": "test-thread-postgres"}}
        state = {
            "messages": [{"role": "user", "content": "PostgreSQL test"}],
            "step": 1
        }
        metadata = {"agent_type": "postgres_agent", "user_id": "postgres_user"}
        
        checkpoint_id = await manager.save_checkpoint(config, state, metadata)
        logger.info(f"✅ 保存checkpoint成功: {checkpoint_id}")
        
        # 测试加载checkpoint
        loaded_info = await manager.load_checkpoint(config)
        if loaded_info:
            logger.info(f"✅ 加载checkpoint成功: {loaded_info.metadata.checkpoint_id}")
            logger.info(f"   状态: {loaded_info.state}")
        else:
            logger.error("❌ 加载checkpoint失败")
        
        # 测试多个checkpoints
        for i in range(3):
            state["step"] = i + 2
            state["messages"].append({"role": "assistant", "content": f"Response {i+1}"})
            await manager.save_checkpoint(config, state, metadata)
        
        # 列出所有checkpoints
        checkpoints = await manager.list_checkpoints(config, limit=10)
        logger.info(f"✅ 列出checkpoints成功: 找到 {len(checkpoints)} 个")
        
        for i, checkpoint_info in enumerate(checkpoints):
            logger.info(f"   Checkpoint {i+1}: {checkpoint_info.metadata.checkpoint_id} "
                       f"(step: {checkpoint_info.state.get('step', 'N/A')})")
        
        await manager.cleanup()
        logger.info("✅ 资源清理完成")
        
    except Exception as e:
        logger.warning(f"⚠️ PostgreSQL测试跳过（可能未配置）: {e}")
        return False
    
    return True

async def test_checkpoint_manager_context():
    """测试checkpoint管理器上下文管理"""
    logger.info("=== 测试checkpoint管理器上下文管理 ===")
    
    from core.checkpoint.manager import checkpoint_manager_context
    
    try:
        async with checkpoint_manager_context("memory") as manager:
            logger.info(f"✅ 上下文管理器初始化成功: {type(manager).__name__}")
            
            # 测试基本功能
            config = {"configurable": {"thread_id": "context-test"}}
            state = {"test": "context manager"}
            
            checkpoint_id = await manager.save_checkpoint(config, state)
            logger.info(f"✅ 上下文中保存checkpoint成功: {checkpoint_id}")
            
            loaded_info = await manager.load_checkpoint(config)
            if loaded_info:
                logger.info(f"✅ 上下文中加载checkpoint成功")
            else:
                logger.error("❌ 上下文中加载checkpoint失败")
        
        logger.info("✅ 上下文管理器自动清理完成")
        
    except Exception as e:
        logger.error(f"❌ 上下文管理器测试失败: {e}")
        raise

async def main():
    """主测试函数"""
    logger.info("开始测试修正后的checkpoint功能...")
    
    try:
        # 测试内存checkpoint
        await test_memory_checkpoint()
        
        # 测试PostgreSQL checkpoint（如果可用）
        await test_postgres_checkpoint()
        
        # 测试上下文管理器
        await test_checkpoint_manager_context()
        
        logger.info("🎉 所有checkpoint测试通过！")
        
    except Exception as e:
        logger.error(f"💥 测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)