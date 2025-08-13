"""
系统启动器 - Bootstrap模块

提供系统初始化、启动和生命周期管理功能，包括：
- 系统组件初始化
- 配置验证和加载
- 数据库连接管理
- 健康检查和状态监控
- 优雅关闭处理
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime

from config.settings import get_settings
from core.database import get_database_manager
from core.cache.redis_manager import get_cache_manager
from core.memory.store_manager import get_memory_store_manager
from core.tools.mcp_manager import get_mcp_manager
from core.agents import get_agent_registry, get_agent_manager
from core.error import get_error_handler, get_performance_monitor
from core.streaming import get_streaming_manager
from core.workflows import get_workflow_manager
from core.optimization import get_optimization_manager
from core.interrupts import get_interrupt_manager
from core.time_travel import get_time_travel_manager
from core.checkpoint import get_checkpoint_manager
from core.tools import get_enhanced_tool_manager
from core.logging import logger_manager

logger = logging.getLogger(__name__)


class SystemBootstrap:
    """系统启动器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.start_time = time.time()
        self.components = {}
        self.health_status = {}
        
    async def initialize(self):
        """初始化系统组件"""
        logger.info("开始系统初始化...")
        
        try:
            # 1. 初始化数据库连接
            logger.info("初始化数据库连接...")
            self.components['database'] = await self._init_database()
            
            # 2. 初始化缓存管理器
            logger.info("初始化缓存管理器...")
            self.components['cache'] = await self._init_cache()
            
            # 3. 初始化记忆管理器
            logger.info("初始化记忆管理器...")
            self.components['memory'] = await self._init_memory()
            
            # 4. 初始化MCP管理器
            logger.info("初始化MCP管理器...")
            self.components['mcp'] = await self._init_mcp()
            
            # 5. 初始化智能体系统
            logger.info("初始化智能体系统...")
            self.components['agents'] = await self._init_agents()
            
            # 6. 初始化错误处理和监控
            logger.info("初始化错误处理和监控...")
            self.components['error_handler'] = get_error_handler()
            self.components['performance_monitor'] = get_performance_monitor()
            
            # 7. 初始化流式管理器
            logger.info("初始化流式管理器...")
            self.components['streaming'] = await self._init_streaming()
            
            # 8. 初始化工作流管理器
            logger.info("初始化工作流管理器...")
            self.components['workflows'] = await self._init_workflows()
            
            # 9. 初始化提示词优化器
            logger.info("初始化提示词优化器...")
            self.components['optimization'] = await self._init_optimization()
            
            # 10. 初始化中断管理器
            logger.info("初始化中断管理器...")
            self.components['interrupts'] = await self._init_interrupts()
            
            # 11. 初始化时间旅行管理器
            logger.info("初始化时间旅行管理器...")
            self.components['time_travel'] = await self._init_time_travel()
            
            # 12. 初始化检查点管理器
            logger.info("初始化检查点管理器...")
            self.components['checkpoint'] = await self._init_checkpoint()
            
            # 13. 初始化增强工具管理器
            logger.info("初始化增强工具管理器...")
            self.components['tools'] = await self._init_tools()
            
            # 14. 初始化日志管理器
            logger.info("初始化日志管理器...")
            self.components['logging'] = self._init_logging()
            
            logger.info("系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            await self.cleanup()
            raise
    
    async def _init_database(self):
        """初始化数据库"""
        try:
            db_manager = await get_database_manager()
            return db_manager
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    async def _init_cache(self):
        """初始化缓存"""
        try:
            cache_manager = await get_cache_manager()
            await cache_manager.initialize()
            return cache_manager
        except Exception as e:
            logger.error(f"缓存初始化失败: {e}")
            raise
    
    async def _init_memory(self):
        """初始化记忆管理器"""
        try:
            memory_manager = await get_memory_store_manager()
            await memory_manager.initialize()
            return memory_manager
        except Exception as e:
            logger.error(f"记忆管理器初始化失败: {e}")
            raise
    
    async def _init_mcp(self):
        """初始化MCP管理器"""
        try:
            mcp_manager = get_mcp_manager()
            await mcp_manager.initialize()
            return mcp_manager
        except Exception as e:
            logger.warning(f"MCP管理器初始化失败: {e}")
            # MCP不是必需的，可以继续运行
            return None
    
    async def _init_agents(self):
        """初始化智能体系统"""
        try:
            agent_registry = get_agent_registry()
            agent_manager = get_agent_manager()
            
            # 注册默认智能体类型
            await self._register_default_agents(agent_registry)
            
            return {
                'registry': agent_registry,
                'manager': agent_manager
            }
        except Exception as e:
            logger.error(f"智能体系统初始化失败: {e}")
            raise
    
    async def _register_default_agents(self, registry):
        """注册默认智能体类型"""
        try:
            # 这里可以注册具体的智能体类型
            # registry.register_agent_type(AgentType.SUPERVISOR, SupervisorAgent)
            # registry.register_agent_type(AgentType.RESEARCH, ResearchAgent)
            # registry.register_agent_type(AgentType.CHART, ChartAgent)
            logger.info("默认智能体类型注册完成")
        except Exception as e:
            logger.warning(f"智能体类型注册失败: {e}")
    
    async def _init_streaming(self):
        """初始化流式管理器"""
        try:
            streaming_manager = get_streaming_manager()
            # 如果有初始化方法，调用它
            if hasattr(streaming_manager, 'initialize'):
                await streaming_manager.initialize()
            return streaming_manager
        except Exception as e:
            logger.error(f"流式管理器初始化失败: {e}")
            raise
    
    async def _init_workflows(self):
        """初始化工作流管理器"""
        try:
            workflow_manager = get_workflow_manager()
            # 如果有初始化方法，调用它
            if hasattr(workflow_manager, 'initialize'):
                await workflow_manager.initialize()
            return workflow_manager
        except Exception as e:
            logger.error(f"工作流管理器初始化失败: {e}")
            raise
    
    async def _init_optimization(self):
        """初始化提示词优化器"""
        try:
            optimization_manager = get_optimization_manager()
            # 如果有初始化方法，调用它
            if hasattr(optimization_manager, 'initialize'):
                await optimization_manager.initialize()
            return optimization_manager
        except Exception as e:
            logger.error(f"提示词优化器初始化失败: {e}")
            raise
    
    async def _init_interrupts(self):
        """初始化中断管理器"""
        try:
            interrupt_manager = get_interrupt_manager()
            # 如果有初始化方法，调用它
            if hasattr(interrupt_manager, 'initialize'):
                await interrupt_manager.initialize()
            return interrupt_manager
        except Exception as e:
            logger.error(f"中断管理器初始化失败: {e}")
            raise
    
    async def _init_time_travel(self):
        """初始化时间旅行管理器"""
        try:
            time_travel_manager = get_time_travel_manager()
            # 如果有初始化方法，调用它
            if hasattr(time_travel_manager, 'initialize'):
                await time_travel_manager.initialize()
            return time_travel_manager
        except Exception as e:
            logger.error(f"时间旅行管理器初始化失败: {e}")
            raise
    
    async def _init_checkpoint(self):
        """初始化检查点管理器"""
        try:
            checkpoint_manager = get_checkpoint_manager()
            # 如果有初始化方法，调用它
            if hasattr(checkpoint_manager, 'initialize'):
                await checkpoint_manager.initialize()
            return checkpoint_manager
        except Exception as e:
            logger.error(f"检查点管理器初始化失败: {e}")
            raise
    
    async def _init_tools(self):
        """初始化增强工具管理器"""
        try:
            tools_manager = get_enhanced_tool_manager()
            # 如果有初始化方法，调用它
            if hasattr(tools_manager, 'initialize'):
                await tools_manager.initialize()
            return tools_manager
        except Exception as e:
            logger.error(f"增强工具管理器初始化失败: {e}")
            raise
    
    def _init_logging(self):
        """初始化日志管理器"""
        try:
            # 日志管理器是同步的
            logger_manager.initialize()
            return logger_manager
        except Exception as e:
            logger.error(f"日志管理器初始化失败: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "components": {}
        }
        
        # 检查各组件健康状态
        for component_name, component in self.components.items():
            try:
                if hasattr(component, 'health_check'):
                    component_health = await component.health_check()
                    health_data["components"][component_name] = component_health
                else:
                    health_data["components"][component_name] = {
                        "status": "healthy",
                        "available": True
                    }
            except Exception as e:
                health_data["components"][component_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "available": False
                }
                health_data["status"] = "degraded"
        
        return health_data
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "status": "running",
            "uptime": time.time() - self.start_time,
            "start_time": self.start_time,
            "components": {
                name: bool(component) for name, component in self.components.items()
            },
            "settings": {
                "app_name": self.settings.app.app_name,
                "version": self.settings.app.version,
                "debug": self.settings.app.debug,
                "host": self.settings.app.host,
                "port": self.settings.app.port
            }
        }
    
    async def cleanup(self):
        """清理系统资源"""
        logger.info("开始系统清理...")
        
        # 按相反顺序清理组件
        cleanup_order = ['logging', 'tools', 'checkpoint', 'time_travel', 'interrupts', 'optimization', 'workflows', 'streaming', 'agents', 'mcp', 'memory', 'cache', 'database']
        
        for component_name in cleanup_order:
            if component_name in self.components:
                component = self.components[component_name]
                try:
                    if hasattr(component, 'cleanup'):
                        await component.cleanup()
                    elif hasattr(component, 'close'):
                        await component.close()
                    logger.info(f"{component_name} 组件清理完成")
                except Exception as e:
                    logger.warning(f"{component_name} 组件清理失败: {e}")
        
        logger.info("系统清理完成")


# 全局实例
_bootstrap_instance: Optional[SystemBootstrap] = None


def get_bootstrap() -> SystemBootstrap:
    """获取系统启动器实例"""
    global _bootstrap_instance
    if _bootstrap_instance is None:
        _bootstrap_instance = SystemBootstrap()
    return _bootstrap_instance


@asynccontextmanager
async def system_lifespan():
    """系统生命周期管理器"""
    bootstrap = get_bootstrap()
    
    try:
        # 初始化系统
        await bootstrap.initialize()
        logger.info("系统启动完成")
        
        yield bootstrap
        
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        raise
    finally:
        # 清理系统资源
        await bootstrap.cleanup()
        logger.info("系统已关闭")


# 兼容性函数
async def initialize_system():
    """初始化系统（兼容性函数）"""
    bootstrap = get_bootstrap()
    return await bootstrap.initialize()


async def cleanup_system():
    """清理系统（兼容性函数）"""
    bootstrap = get_bootstrap()
    await bootstrap.cleanup()