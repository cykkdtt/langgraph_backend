#!/usr/bin/env python3
"""
MCP管理器测试脚本

用于验证重构后的MCP管理器是否正常工作
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.tools.mcp_manager import get_mcp_manager, initialize_mcp_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_mcp_manager():
    """测试MCP管理器功能"""
    logger.info("开始测试MCP管理器...")
    
    try:
        # 1. 初始化MCP管理器
        logger.info("1. 初始化MCP管理器")
        success = await initialize_mcp_manager()
        if not success:
            logger.error("MCP管理器初始化失败")
            return False
        
        manager = get_mcp_manager()
        logger.info(f"MCP管理器初始化成功，是否已初始化: {manager.is_initialized()}")
        
        # 2. 获取服务器列表
        logger.info("2. 获取服务器列表")
        server_names = manager.get_server_names()
        logger.info(f"配置的服务器: {server_names}")
        
        if not server_names:
            logger.warning("没有配置任何MCP服务器")
            return True
        
        # 3. 健康检查
        logger.info("3. 执行健康检查")
        health_status = await manager.health_check()
        for server_name, is_healthy in health_status.items():
            status = "健康" if is_healthy else "不健康"
            logger.info(f"  服务器 {server_name}: {status}")
        
        # 4. 获取工具
        logger.info("4. 获取MCP工具")
        all_tools = await manager.get_tools()
        logger.info(f"总共获取到 {len(all_tools)} 个工具")
        
        # 按服务器分组显示工具
        for server_name in server_names:
            try:
                server_tools = await manager.get_tools(server_name)
                logger.info(f"  服务器 {server_name}: {len(server_tools)} 个工具")
                for tool in server_tools[:3]:  # 只显示前3个工具
                    logger.info(f"    - {tool.name}: {tool.description}")
                if len(server_tools) > 3:
                    logger.info(f"    ... 还有 {len(server_tools) - 3} 个工具")
            except Exception as e:
                logger.warning(f"  服务器 {server_name} 获取工具失败: {e}")
        
        # 5. 测试资源获取（如果有可用的服务器）
        logger.info("5. 测试资源获取")
        healthy_servers = [name for name, healthy in health_status.items() if healthy]
        
        if healthy_servers:
            test_server = healthy_servers[0]
            try:
                resources = await manager.get_resources(test_server)
                logger.info(f"  服务器 {test_server}: {len(resources)} 个资源")
                for resource in resources[:2]:  # 只显示前2个资源
                    logger.info(f"    - {resource.source} ({resource.mimetype})")
            except Exception as e:
                logger.warning(f"  服务器 {test_server} 获取资源失败: {e}")
        else:
            logger.warning("  没有健康的服务器可以测试资源获取")
        
        # 6. 测试会话方式获取工具
        logger.info("6. 测试会话方式获取工具")
        if healthy_servers:
            test_server = healthy_servers[0]
            try:
                session_tools = await manager.get_tools_with_session(test_server)
                logger.info(f"  通过会话获取到 {len(session_tools)} 个工具 (服务器: {test_server})")
            except Exception as e:
                logger.warning(f"  会话方式获取工具失败: {e}")
        
        logger.info("MCP管理器测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        return False
    
    finally:
        # 清理资源
        try:
            manager = get_mcp_manager()
            await manager.close()
            logger.info("MCP管理器已关闭")
        except Exception as e:
            logger.warning(f"关闭MCP管理器时发生错误: {e}")


async def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("MCP管理器测试脚本")
    logger.info("=" * 50)
    
    success = await test_mcp_manager()
    
    if success:
        logger.info("✅ 测试成功完成")
        sys.exit(0)
    else:
        logger.error("❌ 测试失败")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())