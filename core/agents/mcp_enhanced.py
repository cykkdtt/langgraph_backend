"""
MCP增强智能体基类

提供集成MCP (Model Context Protocol) 功能的智能体基类，包括：
- MCP工具集成和调用
- 资源访问和管理
- 多服务器协调
- 智能工具路由
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from abc import abstractmethod

from ..base import BaseAgent
from ...tools.mcp_manager import get_mcp_manager, MCPServerConfig, MCPToolInfo, MCPResourceInfo
from ...tools.mcp_connection_manager import get_connection_manager
from ...tools.mcp_cache_manager import get_cache_manager


class MCPEnhancedAgent(BaseAgent):
    """MCP增强智能体基类"""
    
    def __init__(self, agent_type: str, config: Dict[str, Any]):
        super().__init__(agent_type, config)
        self.logger = logging.getLogger(f"agent.mcp.{agent_type}")
        
        # MCP组件
        self.mcp_manager = get_mcp_manager()
        self.connection_manager = get_connection_manager()
        self.cache_manager = get_cache_manager()
        
        # MCP配置
        self.mcp_servers: Dict[str, MCPServerConfig] = {}
        self.available_tools: Dict[str, MCPToolInfo] = {}
        self.available_resources: Dict[str, MCPResourceInfo] = {}
        
        # 工具路由配置
        self.tool_routing: Dict[str, str] = {}  # tool_name -> server_name
        self.server_priority: List[str] = []  # 服务器优先级
        
        # 初始化MCP
        self._mcp_initialized = False
    
    async def setup_mcp_tools(self, server_configs: List[MCPServerConfig]) -> bool:
        """设置MCP工具"""
        try:
            # 配置服务器
            for config in server_configs:
                self.mcp_servers[config.name] = config
                
                # 添加连接
                await self.connection_manager.add_connection(config.to_connection_config())
            
            # 初始化MCP管理器
            await self.mcp_manager.initialize_servers(self.mcp_servers)
            
            # 加载工具和资源
            await self._load_tools_and_resources()
            
            # 设置工具路由
            self._setup_tool_routing()
            
            self._mcp_initialized = True
            self.logger.info(f"MCP工具设置完成，加载了 {len(self.available_tools)} 个工具")
            return True
            
        except Exception as e:
            self.logger.error(f"MCP工具设置失败: {e}")
            return False
    
    async def _load_tools_and_resources(self):
        """加载工具和资源"""
        try:
            # 加载所有工具
            all_tools = await self.mcp_manager.get_tools()
            for tool in all_tools:
                self.available_tools[tool.name] = tool
            
            # 加载所有资源
            all_resources = await self.mcp_manager.get_resources()
            for resource in all_resources:
                self.available_resources[resource.uri] = resource
                
        except Exception as e:
            self.logger.error(f"加载工具和资源失败: {e}")
    
    def _setup_tool_routing(self):
        """设置工具路由"""
        # 按服务器优先级设置路由
        for tool_name, tool_info in self.available_tools.items():
            if tool_name not in self.tool_routing:
                self.tool_routing[tool_name] = tool_info.server_name
        
        # 设置服务器优先级（如果未配置）
        if not self.server_priority:
            self.server_priority = list(self.mcp_servers.keys())
    
    async def call_mcp_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        server_name: Optional[str] = None,
        use_cache: bool = True
    ) -> Any:
        """调用MCP工具"""
        if not self._mcp_initialized:
            raise RuntimeError("MCP未初始化，请先调用setup_mcp_tools")
        
        # 确定服务器
        target_server = server_name or self.tool_routing.get(tool_name)
        if not target_server:
            raise ValueError(f"未找到工具 {tool_name} 的服务器")
        
        try:
            # 检查缓存
            if use_cache:
                cached_result = await self.cache_manager.get(
                    target_server, "tool_call", tool_name=tool_name, arguments=arguments
                )
                if cached_result is not None:
                    self.logger.debug(f"从缓存获取工具调用结果: {tool_name}")
                    return cached_result
            
            # 调用工具
            result = await self.mcp_manager.call_tool(target_server, tool_name, arguments)
            
            # 缓存结果
            if use_cache and result is not None:
                await self.cache_manager.set(
                    target_server, "tool_call", result, 
                    tool_name=tool_name, arguments=arguments
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"MCP工具调用失败 {tool_name}: {e}")
            raise
    
    async def get_mcp_resources(
        self, 
        resource_uri: str,
        server_name: Optional[str] = None,
        use_cache: bool = True
    ) -> Any:
        """获取MCP资源"""
        if not self._mcp_initialized:
            raise RuntimeError("MCP未初始化，请先调用setup_mcp_tools")
        
        # 确定服务器
        target_server = server_name
        if not target_server:
            # 从资源URI中推断服务器
            resource_info = self.available_resources.get(resource_uri)
            if resource_info:
                target_server = resource_info.server_name
        
        if not target_server:
            raise ValueError(f"未找到资源 {resource_uri} 的服务器")
        
        try:
            # 检查缓存
            if use_cache:
                cached_result = await self.cache_manager.get(
                    target_server, "resource_get", resource_uri=resource_uri
                )
                if cached_result is not None:
                    self.logger.debug(f"从缓存获取资源: {resource_uri}")
                    return cached_result
            
            # 获取资源
            result = await self.mcp_manager.get_resource(target_server, resource_uri)
            
            # 缓存结果
            if use_cache and result is not None:
                await self.cache_manager.set(
                    target_server, "resource_get", result,
                    resource_uri=resource_uri
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"MCP资源获取失败 {resource_uri}: {e}")
            raise
    
    async def list_available_tools(self, server_name: Optional[str] = None) -> List[MCPToolInfo]:
        """列出可用工具"""
        if server_name:
            return [tool for tool in self.available_tools.values() 
                   if tool.server_name == server_name]
        return list(self.available_tools.values())
    
    async def list_available_resources(self, server_name: Optional[str] = None) -> List[MCPResourceInfo]:
        """列出可用资源"""
        if server_name:
            return [resource for resource in self.available_resources.values() 
                   if resource.server_name == server_name]
        return list(self.available_resources.values())
    
    async def health_check_mcp(self) -> Dict[str, Any]:
        """MCP健康检查"""
        try:
            # 检查连接状态
            connection_status = await self.connection_manager.health_check_all()
            
            # 检查工具可用性
            tool_status = {}
            for server_name in self.mcp_servers.keys():
                try:
                    tools = await self.mcp_manager.get_tools(server_name)
                    tool_status[server_name] = {
                        "status": "healthy",
                        "tool_count": len(tools)
                    }
                except Exception as e:
                    tool_status[server_name] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return {
                "mcp_initialized": self._mcp_initialized,
                "server_count": len(self.mcp_servers),
                "tool_count": len(self.available_tools),
                "resource_count": len(self.available_resources),
                "connections": {name: status.value for name, status in connection_status.items()},
                "tools": tool_status
            }
            
        except Exception as e:
            self.logger.error(f"MCP健康检查失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def reload_mcp_config(self) -> bool:
        """重新加载MCP配置"""
        try:
            # 重新加载工具和资源
            await self._load_tools_and_resources()
            
            # 重新设置工具路由
            self._setup_tool_routing()
            
            self.logger.info("MCP配置重新加载完成")
            return True
            
        except Exception as e:
            self.logger.error(f"MCP配置重新加载失败: {e}")
            return False
    
    def get_tool_by_capability(self, capability: str) -> List[MCPToolInfo]:
        """根据能力获取工具"""
        matching_tools = []
        for tool in self.available_tools.values():
            if capability.lower() in tool.description.lower():
                matching_tools.append(tool)
        return matching_tools
    
    async def execute_tool_chain(self, tool_chain: List[Dict[str, Any]]) -> List[Any]:
        """执行工具链"""
        results = []
        
        for step in tool_chain:
            tool_name = step.get("tool")
            arguments = step.get("arguments", {})
            server_name = step.get("server")
            
            if not tool_name:
                raise ValueError("工具链步骤必须包含tool字段")
            
            try:
                result = await self.call_mcp_tool(tool_name, arguments, server_name)
                results.append(result)
                
                # 如果有下一步，可以将当前结果传递给下一步
                if "pass_result_to_next" in step and step["pass_result_to_next"]:
                    next_step_index = tool_chain.index(step) + 1
                    if next_step_index < len(tool_chain):
                        next_step = tool_chain[next_step_index]
                        if "arguments" not in next_step:
                            next_step["arguments"] = {}
                        next_step["arguments"]["previous_result"] = result
                
            except Exception as e:
                self.logger.error(f"工具链执行失败在步骤 {tool_name}: {e}")
                if step.get("continue_on_error", False):
                    results.append({"error": str(e)})
                else:
                    raise
        
        return results
    
    async def cleanup_mcp(self):
        """清理MCP资源"""
        try:
            # 清理缓存
            for server_name in self.mcp_servers.keys():
                await self.cache_manager.clear_server_cache(server_name)
            
            # 清理连接
            for server_name in self.mcp_servers.keys():
                await self.connection_manager.remove_connection(server_name)
            
            # 重置状态
            self.mcp_servers.clear()
            self.available_tools.clear()
            self.available_resources.clear()
            self.tool_routing.clear()
            self._mcp_initialized = False
            
            self.logger.info("MCP资源清理完成")
            
        except Exception as e:
            self.logger.error(f"MCP资源清理失败: {e}")


class FileSystemMCPAgent(MCPEnhancedAgent):
    """文件系统MCP智能体"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("filesystem_mcp", config)
        
    async def initialize(self):
        """初始化文件系统MCP智能体"""
        filesystem_config = MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"],
            env={}
        )
        
        return await self.setup_mcp_tools([filesystem_config])
    
    async def read_file(self, file_path: str) -> str:
        """读取文件"""
        return await self.call_mcp_tool("read_file", {"path": file_path})
    
    async def write_file(self, file_path: str, content: str) -> bool:
        """写入文件"""
        result = await self.call_mcp_tool("write_file", {"path": file_path, "content": content})
        return result.get("success", False)
    
    async def list_directory(self, dir_path: str) -> List[str]:
        """列出目录内容"""
        result = await self.call_mcp_tool("list_directory", {"path": dir_path})
        return result.get("files", [])


class GitMCPAgent(MCPEnhancedAgent):
    """Git操作MCP智能体"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("git_mcp", config)
    
    async def initialize(self):
        """初始化Git MCP智能体"""
        git_config = MCPServerConfig(
            name="git",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-git", "/path/to/git/repo"],
            env={}
        )
        
        return await self.setup_mcp_tools([git_config])
    
    async def git_status(self) -> Dict[str, Any]:
        """获取Git状态"""
        return await self.call_mcp_tool("git_status", {})
    
    async def git_commit(self, message: str) -> bool:
        """提交更改"""
        result = await self.call_mcp_tool("git_commit", {"message": message})
        return result.get("success", False)
    
    async def git_log(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取Git日志"""
        result = await self.call_mcp_tool("git_log", {"limit": limit})
        return result.get("commits", [])


class DatabaseMCPAgent(MCPEnhancedAgent):
    """数据库查询MCP智能体"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("database_mcp", config)
    
    async def initialize(self):
        """初始化数据库MCP智能体"""
        db_config = MCPServerConfig(
            name="database",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres"],
            env={
                "POSTGRES_CONNECTION_STRING": self.config.get("database_url", "")
            }
        )
        
        return await self.setup_mcp_tools([db_config])
    
    async def execute_query(self, query: str, params: Optional[List] = None) -> Dict[str, Any]:
        """执行SQL查询"""
        return await self.call_mcp_tool("execute_query", {
            "query": query,
            "params": params or []
        })
    
    async def get_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """获取数据库模式"""
        args = {}
        if table_name:
            args["table"] = table_name
        return await self.call_mcp_tool("get_schema", args)