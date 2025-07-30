#!/usr/bin/env python
"""
多智能体LangGraph项目 - 系统初始化脚本

此脚本用于初始化系统环境，包括：
1. 检查环境变量配置
2. 初始化数据库
3. 创建必要的目录结构
4. 验证系统依赖
5. 初始化日志系统
"""

import os
import sys
import argparse
import logging
import time
import asyncio
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap import SystemBootstrap, get_bootstrap
from core.env import EnvironmentManager, get_environment_manager
from core.logging import get_logger
from core.database import get_database_manager
from core.error import (
    handle_errors, 
    ErrorSeverity, 
    ErrorCategory, 
    ConfigurationError,
    DatabaseError,
    SystemError
)

# 初始化日志
logger = get_logger("system.init")


class SystemInitializer:
    """系统初始化器"""
    
    def __init__(self):
        self.bootstrap = get_bootstrap()
        self.env_manager = get_environment_manager()
        self.db_manager = get_database_manager()
        self.start_time = time.time()
    
    @handle_errors(component="system.init")
    def create_directories(self) -> None:
        """创建必要的目录结构"""
        logger.info("创建必要的目录结构...")
        
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 需要创建的目录列表
        directories = [
            os.path.join(root_dir, "logs"),
            os.path.join(root_dir, "data"),
            os.path.join(root_dir, "temp"),
            os.path.join(root_dir, "uploads")
        ]
        
        # 创建目录
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"创建目录: {directory}")
            else:
                logger.debug(f"目录已存在: {directory}")
    
    @handle_errors(component="system.init")
    def check_environment(self) -> bool:
        """检查环境变量配置"""
        logger.info("检查环境变量配置...")
        
        # 验证环境变量
        validation_result = self.env_manager.validate_environment()
        
        if not validation_result.is_valid:
            logger.error(f"环境变量配置验证失败: {validation_result.errors}")
            
            # 生成环境变量模板
            if not os.path.exists(".env"):
                logger.info("未找到.env文件，正在生成模板...")
                self.env_manager.generate_env_template()
                logger.info("已生成.env.template文件，请复制为.env并填入实际配置值")
            
            return False
        
        logger.info("环境变量配置验证通过")
        return True
    
    @handle_errors(component="system.init")
    async def check_database(self) -> bool:
        """检查数据库连接"""
        logger.info("检查数据库连接...")
        
        # 初始化数据库
        try:
            await self.db_manager.initialize()
            
            # 检查数据库连接
            is_healthy = await self.db_manager.health_check()
            
            if not is_healthy:
                logger.error("数据库连接检查失败")
                return False
            
            logger.info("数据库连接检查通过")
            return True
        except Exception as e:
            logger.error(f"数据库连接检查失败: {e}")
            return False
    
    @handle_errors(component="system.init")
    def check_dependencies(self) -> bool:
        """检查系统依赖"""
        logger.info("检查系统依赖...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
            logger.error(f"Python版本不满足要求: 当前版本 {python_version.major}.{python_version.minor}.{python_version.micro}，需要 3.9 或更高版本")
            return False
        
        # 检查必要的包
        required_packages = [
            "langchain", "langgraph", "pydantic", "fastapi", "sqlalchemy", 
            "asyncpg", "redis", "uvicorn", "python-dotenv"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"缺少必要的包: {', '.join(missing_packages)}")
            logger.info(f"请使用以下命令安装: pip install {' '.join(missing_packages)}")
            return False
        
        logger.info("系统依赖检查通过")
        return True
    
    @handle_errors(component="system.init")
    async def initialize_system(self) -> bool:
        """初始化系统"""
        logger.info("开始初始化系统...")
        
        # 创建必要的目录结构
        self.create_directories()
        
        # 检查环境变量配置
        if not self.check_environment():
            return False
        
        # 检查系统依赖
        if not self.check_dependencies():
            return False
        
        # 检查数据库连接
        if not await self.check_database():
            return False
        
        # 初始化系统
        try:
            await self.bootstrap.initialize()
            logger.info("系统初始化完成")
            return True
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return False
    
    @handle_errors(component="system.init")
    async def run_health_check(self) -> Dict[str, Any]:
        """运行系统健康检查"""
        logger.info("运行系统健康检查...")
        
        health_status = await self.bootstrap.health_check()
        
        # 记录健康状态
        if health_status["status"] == "healthy":
            logger.info("系统健康检查通过")
        else:
            logger.warning(f"系统健康检查未通过: {health_status}")
        
        return health_status
    
    @handle_errors(component="system.init")
    def print_system_info(self) -> None:
        """打印系统信息"""
        # 获取系统状态
        system_status = self.bootstrap.get_system_status()
        
        # 计算初始化耗时
        elapsed_time = time.time() - self.start_time
        
        # 打印系统信息
        logger.info("=" * 50)
        logger.info("系统信息:")
        logger.info(f"- 初始化耗时: {elapsed_time:.2f}秒")
        logger.info(f"- 应用环境: {system_status['settings']['app_env']}")
        logger.info(f"- 调试模式: {system_status['settings']['debug']}")
        logger.info(f"- 日志级别: {system_status['settings']['log_level']}")
        logger.info(f"- 数据库类型: {system_status['settings']['database_type']}")
        logger.info(f"- LLM提供商: {system_status['settings']['llm_provider']}")
        logger.info(f"- 记忆功能: {'启用' if system_status['settings']['memory_enabled'] else '禁用'}")
        logger.info(f"- 工具功能: {'启用' if system_status['settings']['tools_enabled'] else '禁用'}")
        logger.info("=" * 50)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多智能体LangGraph项目 - 系统初始化脚本")
    parser.add_argument("--check-only", action="store_true", help="仅检查系统状态，不进行初始化")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建系统初始化器
    initializer = SystemInitializer()
    
    if args.check_only:
        # 仅检查系统状态
        logger.info("仅检查系统状态模式")
        
        # 检查环境变量配置
        env_check = initializer.check_environment()
        
        # 检查系统依赖
        dep_check = initializer.check_dependencies()
        
        # 检查数据库连接
        db_check = await initializer.check_database()
        
        # 运行系统健康检查
        if env_check and dep_check and db_check:
            health_status = await initializer.run_health_check()
            
            # 打印系统信息
            initializer.print_system_info()
            
            # 返回状态码
            return 0 if health_status["status"] == "healthy" else 1
        else:
            return 1
    else:
        # 初始化系统
        success = await initializer.initialize_system()
        
        if success:
            # 运行系统健康检查
            health_status = await initializer.run_health_check()
            
            # 打印系统信息
            initializer.print_system_info()
            
            # 返回状态码
            return 0 if health_status["status"] == "healthy" else 1
        else:
            return 1


if __name__ == "__main__":
    # 运行主函数
    exit_code = asyncio.run(main())
    sys.exit(exit_code)