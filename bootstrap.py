"""
多智能体LangGraph项目 - 系统启动和初始化模块

本模块负责系统的启动和初始化，包括：
- 配置加载和验证
- 数据库连接初始化
- 日志系统配置
- 存储系统初始化
- 工具和智能体系统初始化
- 系统健康检查
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from config.settings import get_settings
from core.logging import LoggerManager
from core.env import EnvironmentManager
from core.database import DatabaseManager
from core.error import (
    ErrorHandler, 
    PerformanceMonitor,
    handle_async_errors,
    monitor_async_performance,
    get_error_handler,
    get_performance_monitor
)
from core.checkpoint.manager import get_checkpoint_manager
from core.memory import get_memory_manager
from core.tools import get_tool_registry
from core.agents import get_agent_registry


class SystemBootstrap:
    """系统启动器
    
    负责系统的初始化、配置和启动
    """
    
    def __init__(self):
        # 基础设施
        self.logger = None
        self.env_manager = None
        self.db_manager = None
        
        # 错误处理和性能监控
        self.error_handler = None
        self.performance_monitor = None
        
        # 管理器实例
        self.checkpoint_manager = None
        self.memory_manager = None
        self.tool_registry = None
        self.agent_registry = None
        
        # 状态
        self.is_initialized = False
        self.start_time = None
        self.initialization_errors = []
    
    def initialize_logging(self) -> bool:
        """初始化日志系统
        
        Returns:
            bool: 是否初始化成功
        """
        try:
            # 获取日志管理器
            self.logger = LoggerManager()
            
            # 初始化日志配置
            self.logger.initialize()
            
            # 获取logger实例
            self.logger = self.logger.get_logger("bootstrap")
            
            self.logger.info("日志系统初始化完成")
            return True
        except Exception as e:
            print(f"日志系统初始化失败: {e}")
            return False
    
    def validate_environment(self) -> bool:
        """验证环境变量
        
        Returns:
            bool: 验证是否通过
        """
        try:
            self.logger.info("验证环境变量...")
            
            # 获取环境管理器
            self.env_manager = EnvironmentManager()
            
            # 验证环境变量
            validation_result = self.env_manager.validate_environment()
            
            if validation_result.is_valid:
                self.logger.info("环境变量验证通过")
                return True
            else:
                error_msg = f"环境变量验证失败: {validation_result.errors}"
                self.logger.error(error_msg)
                self.initialization_errors.append(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"环境变量验证异常: {e}"
            self.logger.error(error_msg)
            self.initialization_errors.append(error_msg)
            return False
    
    async def initialize_error_handling(self) -> bool:
        """初始化错误处理和性能监控"""
        try:
            self.logger.info("初始化错误处理和性能监控...")
            
            # 初始化错误处理器
            self.error_handler = get_error_handler()
            
            # 初始化性能监控器
            self.performance_monitor = get_performance_monitor()
            
            self.logger.info("错误处理和性能监控初始化完成")
            return True
        except Exception as e:
            self.logger.error(f"错误处理和性能监控初始化失败: {e}")
            return False
    
    async def initialize_database(self) -> bool:
        """初始化数据库
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info("初始化数据库...")
            
            # 获取数据库管理器
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # 检查数据库健康状态
            is_healthy = await self.db_manager.health_check()
            
            if is_healthy:
                self.logger.info("数据库初始化完成")
                return True
            else:
                error_msg = "数据库健康检查失败"
                self.logger.error(error_msg)
                self.initialization_errors.append(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"数据库初始化失败: {e}"
            self.logger.error(error_msg)
            self.initialization_errors.append(error_msg)
            return False
    
    async def initialize_storage(self) -> bool:
        """初始化存储系统
        
        Returns:
            bool: 是否初始化成功
        """
        try:
            self.logger.info("初始化存储系统...")
            
            # 初始化检查点管理器
            self.checkpoint_manager = get_checkpoint_manager("postgres")
            await self.checkpoint_manager.initialize()
            self.logger.info("检查点管理器初始化完成")
            
            # 初始化记忆管理器
            self.memory_manager = get_memory_manager("postgres")
            await self.memory_manager.initialize()
            self.logger.info("记忆管理器初始化完成")
            
            return True
            
        except Exception as e:
            error_msg = f"存储系统初始化失败: {e}"
            self.logger.error(error_msg)
            self.initialization_errors.append(error_msg)
            return False
    
    def initialize_tools(self) -> bool:
        """初始化工具系统
        
        Returns:
            bool: 是否初始化成功
        """
        try:
            self.logger.info("初始化工具系统...")
            
            # 获取工具注册表
            self.tool_registry = get_tool_registry()
            
            # 这里可以注册默认工具
            # 实际的工具注册会在具体的智能体初始化时进行
            
            self.logger.info("工具系统初始化完成")
            return True
            
        except Exception as e:
            error_msg = f"工具系统初始化失败: {e}"
            self.logger.error(error_msg)
            self.initialization_errors.append(error_msg)
            return False
    
    async def initialize_agents(self) -> bool:
        """初始化智能体系统
        
        Returns:
            bool: 是否初始化成功
        """
        try:
            self.logger.info("初始化智能体系统...")
            
            # 获取智能体注册表
            self.agent_registry = get_agent_registry()
            
            # 这里可以注册默认智能体类型
            # 实际的智能体实例化会在需要时进行
            
            self.logger.info("智能体系统初始化完成")
            return True
            
        except Exception as e:
            error_msg = f"智能体系统初始化失败: {e}"
            self.logger.error(error_msg)
            self.initialization_errors.append(error_msg)
            return False
    
    @handle_async_errors(component="bootstrap")
    @monitor_async_performance(name="system_initialize", component="bootstrap")
    async def initialize(self) -> bool:
        """完整系统初始化
        
        Returns:
            bool: 是否初始化成功
        """
        if self.is_initialized:
            if self.logger:
                self.logger.info("系统已经初始化")
            return True
        
        self.start_time = time.time()
        
        # 清空之前的错误
        self.initialization_errors.clear()
        
        # 初始化各个子系统
        logging_ok = self.initialize_logging()
        env_ok = self.validate_environment()
        
        # 如果环境变量验证失败，则不继续初始化
        if not env_ok:
            self.logger.error("环境变量验证失败，系统初始化中止")
            return False
        
        # 初始化数据库
        db_ok = await self.initialize_database()
        
        # 如果数据库初始化失败，则不继续初始化
        if not db_ok:
            self.logger.error("数据库初始化失败，系统初始化中止")
            return False
        
        # 初始化错误处理和性能监控
        error_handling_ok = await self.initialize_error_handling()
        
        # 初始化存储系统
        storage_ok = await self.initialize_storage()
        
        # 初始化工具和智能体
        tools_ok = self.initialize_tools()
        agents_ok = await self.initialize_agents()
        
        # 检查初始化结果
        all_ok = logging_ok and env_ok and db_ok and error_handling_ok and storage_ok and tools_ok and agents_ok
        
        if all_ok:
            self.is_initialized = True
            elapsed_time = time.time() - self.start_time
            self.logger.info(f"系统初始化完成，耗时: {elapsed_time:.2f}秒")
            return True
        else:
            self.logger.error("系统初始化失败")
            for error in self.initialization_errors:
                self.logger.error(f"  - {error}")
            return False
    
    async def cleanup(self):
        """清理系统资源"""
        try:
            if self.logger:
                self.logger.info("开始系统清理...")
            
            # 清理数据库连接
            if self.db_manager:
                await self.db_manager.cleanup()
            
            # 清理检查点管理器
            if self.checkpoint_manager:
                await self.checkpoint_manager.cleanup()
            
            # 清理记忆管理器
            if self.memory_manager:
                await self.memory_manager.cleanup()
            
            # 清理智能体注册表
            if self.agent_registry:
                await self.agent_registry.cleanup_all()
            
            self.is_initialized = False
            if self.logger:
                self.logger.info("系统清理完成")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"系统清理失败: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查
        
        Returns:
            Dict[str, Any]: 健康检查结果
        """
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": time.time()
        }
        
        try:
            # 检查数据库
            if self.db_manager:
                db_healthy = await self.db_manager.health_check()
                health_status["components"]["database"] = {
                    "status": "healthy" if db_healthy else "unhealthy",
                    "details": "Database connection check"
                }
            else:
                health_status["components"]["database"] = {
                    "status": "not_initialized",
                    "details": "Database manager not initialized"
                }
            
            # 检查检查点管理器
            if self.checkpoint_manager:
                checkpoint_healthy = await self.checkpoint_manager.health_check()
                health_status["components"]["checkpoint"] = {
                    "status": "healthy" if checkpoint_healthy else "unhealthy",
                    "details": "Checkpoint manager check"
                }
            else:
                health_status["components"]["checkpoint"] = {
                    "status": "not_initialized",
                    "details": "Checkpoint manager not initialized"
                }
            
            # 检查记忆管理器
            if self.memory_manager:
                memory_healthy = await self.memory_manager.health_check()
                health_status["components"]["memory"] = {
                    "status": "healthy" if memory_healthy else "unhealthy",
                    "details": "Memory manager check"
                }
            else:
                health_status["components"]["memory"] = {
                    "status": "not_initialized",
                    "details": "Memory manager not initialized"
                }
            
            # 检查工具注册表
            if self.tool_registry:
                health_status["components"]["tools"] = {
                    "status": "healthy",
                    "details": f"Tool registry with {len(self.tool_registry.get_all_tools())} tools"
                }
            else:
                health_status["components"]["tools"] = {
                    "status": "not_initialized",
                    "details": "Tool registry not initialized"
                }
            
            # 检查智能体注册表
            if self.agent_registry:
                health_status["components"]["agents"] = {
                    "status": "healthy",
                    "details": f"Agent registry with {len(self.agent_registry.get_all_agents())} agents"
                }
            else:
                health_status["components"]["agents"] = {
                    "status": "not_initialized",
                    "details": "Agent registry not initialized"
                }
            
            # 检查整体状态
            unhealthy_components = [
                name for name, component in health_status["components"].items()
                if component["status"] in ["unhealthy", "not_initialized"]
            ]
            
            if unhealthy_components:
                health_status["status"] = "degraded" if len(unhealthy_components) < len(health_status["components"]) else "unhealthy"
                health_status["unhealthy_components"] = unhealthy_components
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
            if self.logger:
                self.logger.error(f"健康检查失败: {e}")
        
        return health_status
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态
        
        Returns:
            Dict[str, Any]: 系统状态信息
        """
        settings = get_settings()
        
        return {
            "initialized": self.is_initialized,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "components": {
                "database": self.db_manager is not None,
                "checkpoint": self.checkpoint_manager is not None,
                "memory": self.memory_manager is not None,
                "tools": self.tool_registry is not None,
                "agents": self.agent_registry is not None
            },
            "settings": {
                "app_env": settings.app.env,
                "debug": settings.app.debug,
                "database_type": "postgresql",
                "llm_provider": "deepseek",
                "memory_enabled": settings.memory.enabled,
                "tools_enabled": settings.tools.enabled,
                "log_level": settings.logging.level,
                "structured_logging": settings.logging.structured,
                "json_logging": settings.logging.json_format
            }
        }


# 全局实例
_bootstrap: Optional[SystemBootstrap] = None


def get_bootstrap() -> SystemBootstrap:
    """获取系统启动器实例
    
    Returns:
        SystemBootstrap: 系统启动器实例
    """
    global _bootstrap
    if _bootstrap is None:
        _bootstrap = SystemBootstrap()
    return _bootstrap


@asynccontextmanager
async def system_lifespan():
    """系统生命周期管理器
    
    用于管理系统的启动和关闭
    """
    bootstrap = get_bootstrap()
    
    try:
        # 启动系统
        success = await bootstrap.initialize()
        if not success:
            raise RuntimeError("系统初始化失败")
        
        yield bootstrap
        
    finally:
        # 关闭系统
        await bootstrap.cleanup()


# 便捷函数
async def initialize_system() -> bool:
    """初始化系统
    
    Returns:
        bool: 是否初始化成功
    """
    bootstrap = get_bootstrap()
    return await bootstrap.initialize()


async def health_check() -> Dict[str, Any]:
    """系统健康检查
    
    Returns:
        Dict[str, Any]: 健康检查结果
    """
    bootstrap = get_bootstrap()
    return await bootstrap.health_check()


async def main():
    """主函数，用于测试系统初始化"""
    print("开始系统初始化测试...")
    
    bootstrap = get_bootstrap()
    
    try:
        # 初始化系统
        success = await bootstrap.initialize()
        
        if success:
            print("✓ 系统初始化成功")
            
            # 获取系统状态
            status = bootstrap.get_system_status()
            print(f"系统状态: {status}")
            
            # 运行健康检查
            health = await bootstrap.health_check()
            print(f"健康检查: {health}")
            
        else:
            print("✗ 系统初始化失败")
            
    except Exception as e:
        print(f"系统初始化异常: {e}")
        
    finally:
        # 清理资源
        await bootstrap.cleanup()


if __name__ == "__main__":
    asyncio.run(main())