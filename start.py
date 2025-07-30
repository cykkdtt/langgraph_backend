#!/usr/bin/env python3
"""
多智能体LangGraph项目 - 快速启动脚本

本脚本提供一键启动功能：
- 检查系统环境
- 初始化数据库
- 启动应用服务
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.setup_project import ProjectSetup
from bootstrap import get_bootstrap
from main import main as start_server


async def quick_start():
    """快速启动流程"""
    print("🚀 LangGraph多智能体系统启动中...")
    
    try:
        # 1. 项目设置
        print("\n📋 1. 检查项目设置...")
        setup = ProjectSetup()
        
        # 检查环境
        if not setup.check_environment():
            print("❌ 环境检查失败")
            return False
        
        # 创建必要目录
        setup.create_directories()
        print("✅ 目录结构检查完成")
        
        # 检查.env文件
        if not setup.setup_env_file():
            print("⚠️  .env文件设置可能不完整，请检查配置")
        
        # 2. 系统初始化
        print("\n🔧 2. 初始化系统...")
        bootstrap = get_bootstrap()
        
        success = await bootstrap.initialize()
        if not success:
            print("❌ 系统初始化失败")
            return False
        
        print("✅ 系统初始化完成")
        
        # 3. 健康检查
        print("\n🏥 3. 系统健康检查...")
        health = await bootstrap.health_check()
        
        print(f"系统状态: {health['status']}")
        for component, status in health['components'].items():
            status_icon = "✅" if status['status'] == "healthy" else "⚠️" if status['status'] == "degraded" else "❌"
            print(f"  {status_icon} {component}: {status['status']}")
        
        if health['status'] not in ['healthy', 'degraded']:
            print("❌ 系统健康检查未通过")
            return False
        
        # 4. 启动服务
        print("\n🌐 4. 启动Web服务...")
        print("服务将在 http://localhost:8000 启动")
        print("API文档: http://localhost:8000/docs")
        print("健康检查: http://localhost:8000/health")
        print("\n按 Ctrl+C 停止服务")
        
        return True
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("🤖 LangGraph多智能体协作系统")
    print("=" * 60)
    
    try:
        # 运行快速启动
        success = asyncio.run(quick_start())
        
        if success:
            # 启动Web服务
            start_server()
        else:
            print("\n❌ 系统启动失败，请检查配置和日志")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 系统已停止")
    except Exception as e:
        print(f"\n❌ 启动异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()