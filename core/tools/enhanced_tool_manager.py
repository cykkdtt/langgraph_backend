"""
增强的工具管理器

基于LangGraph官方最佳实践的工具管理优化实现。
参考: https://langchain-ai.github.io/langgraph/agents/tools/
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Callable, Type
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field, ValidationError

from .mcp_manager import get_mcp_manager


class ToolExecutionMode(str, Enum):
    """工具执行模式"""
    SYNC = "sync"
    ASYNC = "async"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class ToolValidationLevel(str, Enum):
    """工具验证级别"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    CUSTOM = "custom"


@dataclass
class ToolExecutionContext:
    """工具执行上下文"""
    user_id: str
    session_id: str
    agent_id: str
    execution_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ToolExecutionResult:
    """工具执行结果"""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    context: Optional[ToolExecutionContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolValidator:
    """工具验证器"""
    
    def __init__(self, validation_level: ToolValidationLevel = ToolValidationLevel.BASIC):
        self.validation_level = validation_level
        self.custom_validators: Dict[str, Callable] = {}
        self.logger = logging.getLogger("tool.validator")
    
    def register_custom_validator(self, tool_name: str, validator: Callable):
        """注册自定义验证器"""
        self.custom_validators[tool_name] = validator
        self.logger.info(f"注册自定义验证器: {tool_name}")
    
    async def validate_input(self, tool: BaseTool, input_data: Dict[str, Any]) -> bool:
        """验证工具输入"""
        if self.validation_level == ToolValidationLevel.NONE:
            return True
        
        try:
            # 基础验证：检查必需参数
            if self.validation_level in [ToolValidationLevel.BASIC, ToolValidationLevel.STRICT]:
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    # 使用Pydantic模型验证
                    tool.args_schema(**input_data)
            
            # 严格验证：额外的业务逻辑检查
            if self.validation_level == ToolValidationLevel.STRICT:
                await self._strict_validation(tool, input_data)
            
            # 自定义验证
            if self.validation_level == ToolValidationLevel.CUSTOM:
                if tool.name in self.custom_validators:
                    return await self.custom_validators[tool.name](input_data)
            
            return True
            
        except ValidationError as e:
            self.logger.error(f"工具输入验证失败 {tool.name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"工具验证异常 {tool.name}: {e}")
            return False
    
    async def _strict_validation(self, tool: BaseTool, input_data: Dict[str, Any]):
        """严格验证逻辑"""
        # 检查敏感操作
        sensitive_operations = ['delete', 'remove', 'drop', 'truncate']
        tool_name_lower = tool.name.lower()
        
        if any(op in tool_name_lower for op in sensitive_operations):
            # 需要额外确认的敏感操作
            if not input_data.get('confirm_sensitive_operation', False):
                raise ValueError(f"敏感操作 {tool.name} 需要明确确认")


class EnhancedToolManager:
    """增强的工具管理器
    
    基于LangGraph最佳实践，提供：
    - 工具生命周期管理
    - 执行监控和性能分析
    - 错误处理和重试机制
    - 并发控制和限流
    - 工具验证和安全检查
    """
    
    def __init__(
        self,
        max_concurrent_executions: int = 10,
        default_timeout: float = 30.0,
        enable_metrics: bool = True
    ):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self.default_timeout = default_timeout
        self.enable_metrics = enable_metrics
        
        # 组件
        self.validator = ToolValidator()
        self.mcp_manager = get_mcp_manager()
        self.logger = logging.getLogger("tool.manager.enhanced")
        
        # 监控数据
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        self.active_executions: Dict[str, ToolExecutionContext] = {}
        
        # 错误处理
        self.error_handlers: Dict[str, Callable] = {}
        self.retry_strategies: Dict[str, Callable] = {}
    
    async def register_tool(
        self,
        tool: BaseTool,
        metadata: Optional[Dict[str, Any]] = None,
        validation_level: ToolValidationLevel = ToolValidationLevel.BASIC,
        custom_validator: Optional[Callable] = None
    ) -> bool:
        """注册工具"""
        try:
            # 验证工具
            if not await self._validate_tool(tool):
                return False
            
            # 注册工具
            self.tools[tool.name] = tool
            self.tool_metadata[tool.name] = metadata or {}
            
            # 设置验证器
            if custom_validator:
                self.validator.register_custom_validator(tool.name, custom_validator)
            
            # 初始化统计
            if self.enable_metrics:
                self.execution_stats[tool.name] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'failed_executions': 0,
                    'total_execution_time': 0.0,
                    'average_execution_time': 0.0,
                    'last_execution': None,
                    'error_rate': 0.0
                }
            
            self.logger.info(f"工具注册成功: {tool.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"工具注册失败 {tool.name}: {e}")
            return False
    
    async def register_mcp_tools(self, server_name: Optional[str] = None) -> int:
        """注册MCP工具"""
        try:
            mcp_tools = await self.mcp_manager.get_tools(server_name)
            registered_count = 0
            
            for tool in mcp_tools:
                if await self.register_tool(tool, {'source': 'mcp', 'server': server_name}):
                    registered_count += 1
            
            self.logger.info(f"注册MCP工具: {registered_count} 个 (服务器: {server_name or '全部'})")
            return registered_count
            
        except Exception as e:
            self.logger.error(f"注册MCP工具失败: {e}")
            return 0
    
    async def execute_tool(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        context: ToolExecutionContext,
        execution_mode: ToolExecutionMode = ToolExecutionMode.ASYNC
    ) -> ToolExecutionResult:
        """执行工具"""
        start_time = datetime.utcnow()
        execution_id = context.execution_id
        
        # 检查工具是否存在
        if tool_name not in self.tools:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"工具不存在: {tool_name}",
                context=context
            )
        
        tool = self.tools[tool_name]
        
        try:
            # 并发控制
            async with self.execution_semaphore:
                # 记录活跃执行
                self.active_executions[execution_id] = context
                
                # 输入验证
                if not await self.validator.validate_input(tool, input_data):
                    return ToolExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        error="输入验证失败",
                        context=context
                    )
                
                # 执行工具
                result = await self._execute_with_timeout(
                    tool, input_data, context, execution_mode
                )
                
                # 计算执行时间
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # 更新统计
                if self.enable_metrics:
                    await self._update_execution_stats(tool_name, True, execution_time)
                
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    context=context
                )
        
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # 更新统计
            if self.enable_metrics:
                await self._update_execution_stats(tool_name, False, execution_time)
            
            # 错误处理
            error_message = await self._handle_execution_error(tool_name, e, context)
            
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=error_message,
                execution_time=execution_time,
                context=context
            )
        
        finally:
            # 清理活跃执行记录
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def execute_tools_parallel(
        self,
        tool_requests: List[Dict[str, Any]],
        context: ToolExecutionContext
    ) -> List[ToolExecutionResult]:
        """并行执行多个工具"""
        tasks = []
        
        for i, request in enumerate(tool_requests):
            # 为每个请求创建独立的上下文
            request_context = ToolExecutionContext(
                user_id=context.user_id,
                session_id=context.session_id,
                agent_id=context.agent_id,
                execution_id=f"{context.execution_id}_{i}",
                metadata=context.metadata,
                timeout=request.get('timeout', context.timeout),
                max_retries=context.max_retries
            )
            
            task = self.execute_tool(
                tool_name=request['tool_name'],
                input_data=request['input_data'],
                context=request_context,
                execution_mode=ToolExecutionMode.PARALLEL
            )
            tasks.append(task)
        
        # 并行执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ToolExecutionResult(
                    tool_name=tool_requests[i]['tool_name'],
                    success=False,
                    error=str(result),
                    context=context
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_with_timeout(
        self,
        tool: BaseTool,
        input_data: Dict[str, Any],
        context: ToolExecutionContext,
        execution_mode: ToolExecutionMode
    ) -> Any:
        """带超时的工具执行"""
        timeout = context.timeout or self.default_timeout
        
        if execution_mode == ToolExecutionMode.ASYNC:
            # 异步执行
            if hasattr(tool, 'ainvoke'):
                return await asyncio.wait_for(
                    tool.ainvoke(input_data),
                    timeout=timeout
                )
            else:
                # 同步工具的异步包装
                return await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, tool.invoke, input_data
                    ),
                    timeout=timeout
                )
        else:
            # 同步执行
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, tool.invoke, input_data
                ),
                timeout=timeout
            )
    
    async def _validate_tool(self, tool: BaseTool) -> bool:
        """验证工具"""
        try:
            # 检查必需属性
            if not hasattr(tool, 'name') or not tool.name:
                self.logger.error("工具缺少名称")
                return False
            
            if not hasattr(tool, 'description') or not tool.description:
                self.logger.error(f"工具 {tool.name} 缺少描述")
                return False
            
            # 检查是否有调用方法
            if not (hasattr(tool, 'invoke') or hasattr(tool, 'ainvoke')):
                self.logger.error(f"工具 {tool.name} 缺少调用方法")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"工具验证异常: {e}")
            return False
    
    async def _update_execution_stats(
        self,
        tool_name: str,
        success: bool,
        execution_time: float
    ):
        """更新执行统计"""
        if tool_name not in self.execution_stats:
            return
        
        stats = self.execution_stats[tool_name]
        stats['total_executions'] += 1
        stats['total_execution_time'] += execution_time
        stats['last_execution'] = datetime.utcnow().isoformat()
        
        if success:
            stats['successful_executions'] += 1
        else:
            stats['failed_executions'] += 1
        
        # 计算平均执行时间
        stats['average_execution_time'] = (
            stats['total_execution_time'] / stats['total_executions']
        )
        
        # 计算错误率
        stats['error_rate'] = (
            stats['failed_executions'] / stats['total_executions']
        )
    
    async def _handle_execution_error(
        self,
        tool_name: str,
        error: Exception,
        context: ToolExecutionContext
    ) -> str:
        """处理执行错误"""
        error_message = str(error)
        
        # 记录错误
        self.logger.error(f"工具执行失败 {tool_name}: {error_message}")
        
        # 调用自定义错误处理器
        if tool_name in self.error_handlers:
            try:
                custom_message = await self.error_handlers[tool_name](error, context)
                if custom_message:
                    error_message = custom_message
            except Exception as e:
                self.logger.error(f"自定义错误处理器失败: {e}")
        
        return error_message
    
    def get_tool_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具统计信息"""
        return self.execution_stats.get(tool_name)
    
    def get_all_tools(self) -> List[str]:
        """获取所有工具名称"""
        return list(self.tools.keys())
    
    def get_active_executions(self) -> Dict[str, ToolExecutionContext]:
        """获取活跃执行"""
        return self.active_executions.copy()


# 全局实例
_enhanced_tool_manager: Optional[EnhancedToolManager] = None


def get_enhanced_tool_manager() -> EnhancedToolManager:
    """获取增强工具管理器实例"""
    global _enhanced_tool_manager
    if _enhanced_tool_manager is None:
        _enhanced_tool_manager = EnhancedToolManager()
    return _enhanced_tool_manager