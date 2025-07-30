"""
多智能体LangGraph项目 - 工具管理器

本模块提供统一的工具管理功能，支持：
- 工具注册和发现
- 工具权限管理
- 工具执行监控
- MCP工具集成
- 自定义工具扩展
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union, Type
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field

from config.settings import get_settings


class ToolCategory(str, Enum):
    """工具分类"""
    SEARCH = "search"          # 搜索工具
    ANALYSIS = "analysis"      # 分析工具
    GENERATION = "generation"  # 生成工具
    COMMUNICATION = "communication"  # 通信工具
    DATA = "data"              # 数据处理工具
    SYSTEM = "system"          # 系统工具
    CUSTOM = "custom"          # 自定义工具


class ToolPermission(str, Enum):
    """工具权限"""
    READ = "read"              # 只读权限
    WRITE = "write"            # 写入权限
    EXECUTE = "execute"        # 执行权限
    ADMIN = "admin"            # 管理权限


class ToolMetadata(BaseModel):
    """工具元数据"""
    name: str = Field(description="工具名称")
    description: str = Field(description="工具描述")
    category: ToolCategory = Field(description="工具分类")
    version: str = Field(default="1.0.0", description="工具版本")
    author: Optional[str] = Field(default=None, description="工具作者")
    permissions: List[ToolPermission] = Field(default_factory=list, description="所需权限")
    dependencies: List[str] = Field(default_factory=list, description="依赖项")
    tags: List[str] = Field(default_factory=list, description="标签")
    is_async: bool = Field(default=False, description="是否异步工具")
    timeout: Optional[int] = Field(default=30, description="超时时间（秒）")


class ToolExecutionResult(BaseModel):
    """工具执行结果"""
    tool_name: str = Field(description="工具名称")
    success: bool = Field(description="是否成功")
    result: Any = Field(description="执行结果")
    error: Optional[str] = Field(default=None, description="错误信息")
    execution_time: float = Field(description="执行时间（秒）")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="执行时间戳")


class ToolExecutionContext(BaseModel):
    """工具执行上下文"""
    user_id: Optional[str] = Field(default=None, description="用户ID")
    agent_id: Optional[str] = Field(default=None, description="智能体ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    permissions: List[ToolPermission] = Field(default_factory=list, description="用户权限")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class BaseManagedTool(BaseTool, ABC):
    """托管工具基类
    
    扩展LangChain的BaseTool，添加权限管理和监控功能。
    """
    
    metadata: ToolMetadata = Field(description="工具元数据")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(f"tool.{self.metadata.name}")
    
    @abstractmethod
    async def _arun_with_context(
        self, 
        context: ToolExecutionContext,
        *args, 
        **kwargs
    ) -> Any:
        """带上下文的异步执行方法"""
        pass
    
    def _run_with_context(
        self, 
        context: ToolExecutionContext,
        *args, 
        **kwargs
    ) -> Any:
        """带上下文的同步执行方法"""
        # 默认实现：调用异步方法
        return asyncio.run(self._arun_with_context(context, *args, **kwargs))
    
    def _run(
        self, 
        *args, 
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> Any:
        """同步执行方法"""
        # 创建默认上下文
        context = ToolExecutionContext()
        return self._run_with_context(context, *args, **kwargs)
    
    async def _arun(
        self, 
        *args, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs
    ) -> Any:
        """异步执行方法"""
        # 创建默认上下文
        context = ToolExecutionContext()
        return await self._arun_with_context(context, *args, **kwargs)
    
    def check_permissions(self, context: ToolExecutionContext) -> bool:
        """检查权限"""
        required_permissions = set(self.metadata.permissions)
        user_permissions = set(context.permissions)
        
        # 检查是否有足够权限
        if required_permissions and not required_permissions.issubset(user_permissions):
            missing = required_permissions - user_permissions
            self.logger.warning(f"权限不足: 缺少 {missing}")
            return False
        
        return True


class ToolRegistry:
    """工具注册表
    
    管理所有可用工具的注册、发现和权限控制。
    """
    
    def __init__(self):
        self.logger = logging.getLogger("tool.registry")
        self._tools: Dict[str, BaseManagedTool] = {}
        self._tool_metadata: Dict[str, ToolMetadata] = {}
        self._execution_stats: Dict[str, List[ToolExecutionResult]] = {}
    
    def register_tool(self, tool: BaseManagedTool) -> bool:
        """注册工具
        
        Args:
            tool: 工具实例
            
        Returns:
            bool: 是否注册成功
        """
        try:
            tool_name = tool.metadata.name
            
            if tool_name in self._tools:
                self.logger.warning(f"工具已存在，将被覆盖: {tool_name}")
            
            self._tools[tool_name] = tool
            self._tool_metadata[tool_name] = tool.metadata
            self._execution_stats[tool_name] = []
            
            self.logger.info(f"工具注册成功: {tool_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"工具注册失败: {e}")
            return False
    
    def unregister_tool(self, tool_name: str) -> bool:
        """注销工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            bool: 是否注销成功
        """
        try:
            if tool_name not in self._tools:
                self.logger.warning(f"工具不存在: {tool_name}")
                return False
            
            del self._tools[tool_name]
            del self._tool_metadata[tool_name]
            del self._execution_stats[tool_name]
            
            self.logger.info(f"工具注销成功: {tool_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"工具注销失败: {e}")
            return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseManagedTool]:
        """获取工具实例
        
        Args:
            tool_name: 工具名称
            
        Returns:
            BaseManagedTool: 工具实例，如果不存在则返回None
        """
        return self._tools.get(tool_name)
    
    def list_tools(
        self, 
        category: Optional[ToolCategory] = None,
        permissions: Optional[List[ToolPermission]] = None
    ) -> List[ToolMetadata]:
        """列出工具
        
        Args:
            category: 工具分类过滤
            permissions: 权限过滤
            
        Returns:
            List[ToolMetadata]: 工具元数据列表
        """
        tools = []
        
        for metadata in self._tool_metadata.values():
            # 分类过滤
            if category and metadata.category != category:
                continue
            
            # 权限过滤
            if permissions:
                required_permissions = set(metadata.permissions)
                user_permissions = set(permissions)
                if required_permissions and not required_permissions.issubset(user_permissions):
                    continue
            
            tools.append(metadata)
        
        return tools
    
    def search_tools(self, query: str) -> List[ToolMetadata]:
        """搜索工具
        
        Args:
            query: 搜索查询
            
        Returns:
            List[ToolMetadata]: 匹配的工具元数据列表
        """
        query_lower = query.lower()
        matching_tools = []
        
        for metadata in self._tool_metadata.values():
            # 在名称、描述、标签中搜索
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                matching_tools.append(metadata)
        
        return matching_tools
    
    async def execute_tool(
        self, 
        tool_name: str, 
        context: ToolExecutionContext,
        *args, 
        **kwargs
    ) -> ToolExecutionResult:
        """执行工具
        
        Args:
            tool_name: 工具名称
            context: 执行上下文
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            ToolExecutionResult: 执行结果
        """
        start_time = datetime.utcnow()
        
        try:
            # 获取工具
            tool = self.get_tool(tool_name)
            if not tool:
                raise ValueError(f"工具不存在: {tool_name}")
            
            # 检查权限
            if not tool.check_permissions(context):
                raise PermissionError(f"权限不足，无法执行工具: {tool_name}")
            
            # 执行工具
            if tool.metadata.is_async:
                result = await tool._arun_with_context(context, *args, **kwargs)
            else:
                result = tool._run_with_context(context, *args, **kwargs)
            
            # 计算执行时间
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # 创建执行结果
            execution_result = ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
            # 记录统计信息
            self._execution_stats[tool_name].append(execution_result)
            
            self.logger.info(f"工具执行成功: {tool_name}, 耗时: {execution_time:.2f}s")
            return execution_result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            execution_result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time
            )
            
            # 记录统计信息
            if tool_name in self._execution_stats:
                self._execution_stats[tool_name].append(execution_result)
            
            self.logger.error(f"工具执行失败: {tool_name}, 错误: {e}")
            return execution_result
    
    def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """获取工具统计信息
        
        Args:
            tool_name: 工具名称
            
        Returns:
            Dict: 统计信息
        """
        if tool_name not in self._execution_stats:
            return {}
        
        executions = self._execution_stats[tool_name]
        if not executions:
            return {"total_executions": 0}
        
        successful_executions = [e for e in executions if e.success]
        failed_executions = [e for e in executions if not e.success]
        
        stats = {
            "total_executions": len(executions),
            "successful_executions": len(successful_executions),
            "failed_executions": len(failed_executions),
            "success_rate": len(successful_executions) / len(executions) if executions else 0,
            "avg_execution_time": sum(e.execution_time for e in executions) / len(executions),
            "last_execution": executions[-1].timestamp.isoformat() if executions else None
        }
        
        if successful_executions:
            stats["avg_successful_time"] = sum(e.execution_time for e in successful_executions) / len(successful_executions)
        
        return stats
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息
        
        Returns:
            Dict: 统计信息
        """
        total_tools = len(self._tools)
        total_executions = sum(len(stats) for stats in self._execution_stats.values())
        
        # 按分类统计
        by_category = {}
        for metadata in self._tool_metadata.values():
            category = metadata.category.value
            by_category[category] = by_category.get(category, 0) + 1
        
        return {
            "total_tools": total_tools,
            "total_executions": total_executions,
            "tools_by_category": by_category,
            "registered_tools": list(self._tools.keys())
        }


# 全局工具注册表实例
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册表实例
    
    Returns:
        ToolRegistry: 工具注册表实例
    """
    global _tool_registry
    
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    
    return _tool_registry


# 装饰器：简化工具注册
def managed_tool(
    name: str,
    description: str,
    category: ToolCategory = ToolCategory.CUSTOM,
    permissions: Optional[List[ToolPermission]] = None,
    **metadata_kwargs
):
    """托管工具装饰器
    
    Args:
        name: 工具名称
        description: 工具描述
        category: 工具分类
        permissions: 所需权限
        **metadata_kwargs: 其他元数据参数
    """
    def decorator(func: Callable):
        # 创建工具元数据
        tool_metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            permissions=permissions or [],
            **metadata_kwargs
        )
        
        # 创建托管工具类
        class DecoratedTool(BaseManagedTool):
            name = tool_metadata.name
            description = tool_metadata.description
            metadata = tool_metadata
            
            async def _arun_with_context(self, context: ToolExecutionContext, *args, **kwargs):
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        # 注册工具
        tool_instance = DecoratedTool()
        registry = get_tool_registry()
        registry.register_tool(tool_instance)
        
        return func
    
    return decorator