#!/usr/bin/env python3
"""
智能体工厂模式测试脚本

测试智能体注册表、工厂和管理器的功能。
"""

import asyncio
import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agents import (
    get_agent_registry,
    get_agent_factory,
    get_agent_manager,
    initialize_agent_manager,
    AgentType,
    AgentConfig,
    ChatRequest
)


async def test_agent_registry():
    """测试智能体注册表"""
    print("=== 测试智能体注册表 ===")
    
    registry = get_agent_registry()
    
    # 列出所有注册的智能体类型
    agent_types = registry.list_agent_types()
    print(f"注册的智能体类型: {[t.value for t in agent_types]}")
    
    # 获取智能体配置
    for agent_type in agent_types:
        config = registry.get_agent_config(agent_type)
        if config:
            print(f"{agent_type.value} 配置:")
            print(f"  名称: {config.name}")
            print(f"  描述: {config.description}")
            print(f"  工具: {config.tools}")
            print(f"  能力: {config.capabilities}")
            print()


async def test_agent_factory():
    """测试智能体工厂"""
    print("=== 测试智能体工厂 ===")
    
    factory = get_agent_factory()
    
    try:
        # 创建supervisor智能体实例
        instance_id = await factory.create_agent(
            agent_type=AgentType.SUPERVISOR,
            user_id="test_user",
            session_id="test_session"
        )
        print(f"创建supervisor智能体实例: {instance_id}")
        
        # 获取实例信息
        instance = await factory.get_instance_info(instance_id)
        if instance:
            print(f"实例信息:")
            print(f"  类型: {instance.agent_type.value}")
            print(f"  用户: {instance.user_id}")
            print(f"  状态: {instance.status.value}")
            print(f"  创建时间: {instance.created_at}")
        
        # 列出所有实例
        instances = await factory.list_instances()
        print(f"总实例数: {len(instances)}")
        
        # 清理实例
        await factory.cleanup_agent(instance_id)
        print(f"清理实例: {instance_id}")
        
    except Exception as e:
        print(f"工厂测试失败: {e}")


async def test_agent_manager():
    """测试智能体管理器"""
    print("=== 测试智能体管理器 ===")
    
    manager = await initialize_agent_manager()
    
    try:
        # 创建多个智能体实例
        supervisor_id = await manager.create_agent(
            agent_type=AgentType.SUPERVISOR,
            user_id="test_user_1",
            session_id="session_1"
        )
        print(f"创建supervisor实例: {supervisor_id}")
        
        research_id = await manager.create_agent(
            agent_type=AgentType.RESEARCH,
            user_id="test_user_2",
            session_id="session_2"
        )
        print(f"创建research实例: {research_id}")
        
        # 列出实例
        instances = await manager.list_instances()
        print(f"当前实例数: {len(instances)}")
        
        for inst in instances:
            print(f"  {inst.instance_id}: {inst.agent_type.value} ({inst.status.value})")
        
        # 测试按用户过滤
        user1_instances = await manager.list_instances(user_id="test_user_1")
        print(f"用户test_user_1的实例数: {len(user1_instances)}")
        
        # 测试按类型过滤
        supervisor_instances = await manager.list_instances(agent_type=AgentType.SUPERVISOR)
        print(f"supervisor类型实例数: {len(supervisor_instances)}")
        
        # 获取性能指标
        metrics = await manager.get_performance_metrics(supervisor_id)
        if metrics:
            print(f"supervisor性能指标:")
            print(f"  总请求数: {metrics.total_requests}")
            print(f"  成功率: {metrics.success_rate:.2%}")
            print(f"  运行时间: {metrics.uptime_hours:.2f}小时")
        
        # 健康检查
        health = await manager.health_check()
        print(f"管理器健康状态: {health}")
        
        # 清理实例
        await manager.cleanup_agent(supervisor_id)
        await manager.cleanup_agent(research_id)
        print("清理所有测试实例")
        
        # 停止管理器
        await manager.stop()
        print("停止智能体管理器")
        
    except Exception as e:
        print(f"管理器测试失败: {e}")
        await manager.stop()


async def test_agent_config():
    """测试智能体配置"""
    print("=== 测试智能体配置 ===")
    
    # 创建自定义配置
    custom_config = AgentConfig(
        agent_type=AgentType.CODE,
        name="Custom Code Agent",
        description="自定义代码智能体",
        llm_config={
            "model_type": "qwen",
            "temperature": 0.1,
            "max_tokens": 4000
        },
        tools=["code_executor", "syntax_checker"],
        capabilities=["code_generation", "code_review"],
        memory_enabled=True,
        custom_config={
            "programming_languages": ["python", "javascript", "go"],
            "code_style": "pep8"
        }
    )
    
    print(f"自定义配置:")
    print(f"  名称: {custom_config.name}")
    print(f"  描述: {custom_config.description}")
    print(f"  LLM配置: {custom_config.llm_config}")
    print(f"  工具: {custom_config.tools}")
    print(f"  能力: {custom_config.capabilities}")
    print(f"  自定义配置: {custom_config.custom_config}")


async def main():
    """主测试函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("开始智能体工厂模式测试")
    print("=" * 50)
    
    try:
        await test_agent_registry()
        print()
        
        await test_agent_factory()
        print()
        
        await test_agent_manager()
        print()
        
        await test_agent_config()
        print()
        
        print("=" * 50)
        print("所有测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())