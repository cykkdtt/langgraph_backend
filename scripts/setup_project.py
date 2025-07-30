#!/usr/bin/env python
"""
多智能体LangGraph项目 - 项目设置脚本

此脚本用于一键设置整个项目，包括：
1. 环境检查和配置
2. 依赖安装
3. 数据库初始化
4. 系统初始化
5. 验证和测试
"""

import os
import sys
import subprocess
import argparse
import asyncio
import logging
import shutil
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging import get_logger
from core.error import handle_errors

# 初始化日志
logger = get_logger("project.setup")


class ProjectSetup:
    """项目设置器"""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scripts_dir = os.path.join(self.project_root, "scripts")
    
    @handle_errors(component="project.setup")
    def check_python_version(self) -> bool:
        """检查Python版本"""
        logger.info("检查Python版本...")
        
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
            logger.error(f"Python版本不满足要求: 当前版本 {python_version.major}.{python_version.minor}.{python_version.micro}，需要 3.9 或更高版本")
            return False
        
        logger.info(f"Python版本检查通过: {python_version.major}.{python_version.minor}.{python_version.micro}")
        return True
    
    @handle_errors(component="project.setup")
    def check_required_tools(self) -> bool:
        """检查必要的工具"""
        logger.info("检查必要的工具...")
        
        required_tools = ["pip", "git"]
        missing_tools = []
        
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            logger.error(f"缺少必要的工具: {', '.join(missing_tools)}")
            return False
        
        logger.info("必要工具检查通过")
        return True
    
    @handle_errors(component="project.setup")
    def setup_environment_file(self) -> bool:
        """设置环境变量文件"""
        logger.info("设置环境变量文件...")
        
        env_file = os.path.join(self.project_root, ".env")
        env_template = os.path.join(self.project_root, ".env.template")
        
        # 如果.env文件不存在，从模板复制
        if not os.path.exists(env_file):
            if os.path.exists(env_template):
                shutil.copy2(env_template, env_file)
                logger.info(f"已从模板创建.env文件: {env_file}")
                logger.warning("请编辑.env文件并填入实际的配置值")
                return False  # 需要用户手动配置
            else:
                logger.error("未找到.env.template文件")
                return False
        else:
            logger.info(".env文件已存在")
            return True
    
    @handle_errors(component="project.setup")
    def install_dependencies(self) -> bool:
        """安装依赖"""
        logger.info("安装项目依赖...")
        
        requirements_file = os.path.join(self.project_root, "requirements.txt")
        
        if not os.path.exists(requirements_file):
            logger.warning("未找到requirements.txt文件，跳过依赖安装")
            return True
        
        try:
            # 升级pip
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True, text=True)
            
            # 安装依赖
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], 
                                  check=True, capture_output=True, text=True)
            
            logger.info("依赖安装完成")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"依赖安装失败: {e.stderr}")
            return False
    
    @handle_errors(component="project.setup")
    def create_directories(self) -> bool:
        """创建必要的目录"""
        logger.info("创建必要的目录...")
        
        directories = [
            "logs",
            "data",
            "temp",
            "uploads",
            "scripts"
        ]
        
        for directory in directories:
            dir_path = os.path.join(self.project_root, directory)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"创建目录: {dir_path}")
            else:
                logger.debug(f"目录已存在: {dir_path}")
        
        return True
    
    @handle_errors(component="project.setup")
    async def initialize_database(self) -> bool:
        """初始化数据库"""
        logger.info("初始化数据库...")
        
        # 运行数据库初始化脚本
        init_db_script = os.path.join(self.scripts_dir, "initialize_database.py")
        
        if not os.path.exists(init_db_script):
            logger.error("未找到数据库初始化脚本")
            return False
        
        try:
            # 导入并运行数据库初始化
            from scripts.initialize_database import DatabaseInitializer
            
            initializer = DatabaseInitializer()
            success = await initializer.run_full_initialization()
            
            if success:
                logger.info("数据库初始化完成")
                return True
            else:
                logger.error("数据库初始化失败")
                return False
        except Exception as e:
            logger.error(f"数据库初始化异常: {e}")
            return False
    
    @handle_errors(component="project.setup")
    async def initialize_system(self) -> bool:
        """初始化系统"""
        logger.info("初始化系统...")
        
        # 运行系统初始化脚本
        init_system_script = os.path.join(self.scripts_dir, "initialize_system.py")
        
        if not os.path.exists(init_system_script):
            logger.error("未找到系统初始化脚本")
            return False
        
        try:
            # 导入并运行系统初始化
            from scripts.initialize_system import SystemInitializer
            
            initializer = SystemInitializer()
            success = await initializer.initialize_system()
            
            if success:
                logger.info("系统初始化完成")
                return True
            else:
                logger.error("系统初始化失败")
                return False
        except Exception as e:
            logger.error(f"系统初始化异常: {e}")
            return False
    
    @handle_errors(component="project.setup")
    async def run_health_check(self) -> bool:
        """运行健康检查"""
        logger.info("运行系统健康检查...")
        
        try:
            from scripts.initialize_system import SystemInitializer
            
            initializer = SystemInitializer()
            health_status = await initializer.run_health_check()
            
            if health_status["status"] == "healthy":
                logger.info("系统健康检查通过")
                return True
            else:
                logger.warning(f"系统健康检查未完全通过: {health_status}")
                return False
        except Exception as e:
            logger.error(f"健康检查异常: {e}")
            return False
    
    @handle_errors(component="project.setup")
    def print_setup_summary(self, results: Dict[str, bool]) -> None:
        """打印设置摘要"""
        logger.info("=" * 60)
        logger.info("项目设置摘要:")
        logger.info("=" * 60)
        
        for step, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            logger.info(f"{step}: {status}")
        
        logger.info("=" * 60)
        
        # 检查是否所有步骤都成功
        all_success = all(results.values())
        
        if all_success:
            logger.info("🎉 项目设置完成！")
            logger.info("你现在可以开始使用多智能体LangGraph项目了。")
            logger.info("")
            logger.info("下一步:")
            logger.info("1. 检查并编辑 .env 文件中的配置")
            logger.info("2. 运行 python main.py 启动应用")
            logger.info("3. 访问 http://localhost:8000 查看API文档")
        else:
            logger.error("❌ 项目设置未完全成功，请检查上述错误并重新运行。")
            logger.info("")
            logger.info("常见问题解决:")
            logger.info("1. 确保已安装PostgreSQL并正在运行")
            logger.info("2. 检查.env文件中的数据库连接配置")
            logger.info("3. 确保所有必要的API密钥已正确配置")
    
    @handle_errors(component="project.setup")
    async def run_full_setup(self, skip_deps: bool = False, skip_db: bool = False) -> bool:
        """运行完整的项目设置"""
        logger.info("开始项目设置...")
        
        results = {}
        
        # 1. 检查Python版本
        results["Python版本检查"] = self.check_python_version()
        if not results["Python版本检查"]:
            self.print_setup_summary(results)
            return False
        
        # 2. 检查必要工具
        results["必要工具检查"] = self.check_required_tools()
        if not results["必要工具检查"]:
            self.print_setup_summary(results)
            return False
        
        # 3. 创建目录
        results["目录创建"] = self.create_directories()
        
        # 4. 设置环境文件
        results["环境文件设置"] = self.setup_environment_file()
        if not results["环境文件设置"]:
            logger.warning("请先配置.env文件，然后重新运行设置")
            self.print_setup_summary(results)
            return False
        
        # 5. 安装依赖（可选）
        if not skip_deps:
            results["依赖安装"] = self.install_dependencies()
        else:
            results["依赖安装"] = True
            logger.info("跳过依赖安装")
        
        # 6. 初始化数据库（可选）
        if not skip_db:
            results["数据库初始化"] = await self.initialize_database()
        else:
            results["数据库初始化"] = True
            logger.info("跳过数据库初始化")
        
        # 7. 初始化系统
        results["系统初始化"] = await self.initialize_system()
        
        # 8. 健康检查
        results["健康检查"] = await self.run_health_check()
        
        # 打印摘要
        self.print_setup_summary(results)
        
        return all(results.values())


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多智能体LangGraph项目 - 项目设置脚本")
    parser.add_argument("--skip-deps", action="store_true", help="跳过依赖安装")
    parser.add_argument("--skip-db", action="store_true", help="跳过数据库初始化")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建项目设置器
    setup = ProjectSetup()
    
    # 运行完整设置
    success = await setup.run_full_setup(
        skip_deps=args.skip_deps,
        skip_db=args.skip_db
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    # 运行主函数
    exit_code = asyncio.run(main())
    sys.exit(exit_code)