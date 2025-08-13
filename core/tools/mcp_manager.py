"""
MCP工具管理器

基于LangGraph官方MCP文档实现的标准MCP工具管理系统。
严格按照官方最佳实践，支持多服务器连接、工具加载、资源管理和提示管理。

参考文档: https://langchain-ai.github.io/langgraph/reference/mcp/
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_core.document_loaders import Blob
from pydantic import BaseModel, Field

from config.settings import get_settings


class MCPServerConfig(BaseModel):
    """MCP服务器配置 - 符合官方标准"""
    name: str = Field(description="服务器名称")
    command: Optional[str] = Field(default=None, description="启动命令")
    args: List[str] = Field(default_factory=list, description="命令参数")
    url: Optional[str] = Field(default=None, description="服务器URL")
    transport: str = Field(default="stdio", description="传输协议")
    env: Dict[str, str] = Field(default_factory=dict, description="环境变量")
    enabled: bool = Field(default=True, description="是否启用")


class MCPManager:
    """MCP管理器 - 基于官方文档的标准实现
    
    严格按照官方文档实现，支持：
    - 多服务器连接管理（使用官方推荐的连接方式）
    - 工具动态加载（使用官方load_mcp_tools函数）
    - 资源管理（使用官方get_resources方法）
    - 提示管理（使用官方get_prompt方法）
    - 正确的会话管理（使用session()方法）
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化MCP管理器
        
        Args:
            config_path: MCP配置文件路径，默认为 servers_config.json
        """
        self.logger = logging.getLogger("mcp.manager")
        self.config_path = config_path or "servers_config.json"
        
        # 状态管理
        self._client: Optional[MultiServerMCPClient] = None
        self._connections: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        
        # 加载配置
        self._load_config()
    
    def _load_config(self) -> None:
        """加载MCP服务器配置"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self.logger.warning(f"MCP配置文件不存在: {self.config_path}")
                return
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            mcp_servers = config_data.get("mcpServers", {})
            
            for server_name, server_config in mcp_servers.items():
                try:
                    # 跳过禁用的服务器
                    if not server_config.get("enabled", True):
                        self.logger.info(f"跳过禁用的MCP服务器: {server_name}")
                        continue
                    
                    # 构建连接配置 - 按照官方文档格式
                    connection_config = {
                        "transport": server_config.get("transport", "stdio")
                    }
                    
                    if server_config.get("command"):
                        connection_config["command"] = server_config["command"]
                        connection_config["args"] = server_config.get("args", [])
                        if server_config.get("env"):
                            connection_config["env"] = server_config["env"]
                    elif server_config.get("url"):
                        connection_config["url"] = server_config["url"]
                    else:
                        self.logger.error(f"服务器配置无效 {server_name}: 缺少command或url")
                        continue
                    
                    self._connections[server_name] = connection_config
                    self.logger.info(f"加载MCP服务器配置: {server_name}")
                    
                except Exception as e:
                    self.logger.error(f"解析服务器配置失败 {server_name}: {e}")
            
            self.logger.info(f"成功加载 {len(self._connections)} 个MCP服务器配置")
            
        except Exception as e:
            self.logger.error(f"加载MCP配置失败: {e}")
    
    async def initialize(self) -> bool:
        """初始化MCP客户端 - 按照官方文档方式
        
        Returns:
            bool: 是否初始化成功
        """
        if self._initialized:
            return True
        
        try:
            if not self._connections:
                self.logger.warning("没有可用的MCP服务器配置")
                return False
            
            # 创建MCP客户端 - 使用官方推荐方式
            self._client = MultiServerMCPClient(self._connections)
            self.logger.info(f"MCP客户端初始化成功，配置了 {len(self._connections)} 个服务器")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"MCP客户端初始化失败: {e}")
            return False
    
    async def get_tools(self, server_name: Optional[str] = None) -> List[BaseTool]:
        """获取MCP工具 - 使用官方推荐方式
        
        Args:
            server_name: 服务器名称，None表示获取所有服务器的工具
            
        Returns:
            List[BaseTool]: 工具列表
        """
        if not await self.initialize():
            return []
        
        try:
            # 使用官方推荐的get_tools方法，注意使用关键字参数
            tools = await self._client.get_tools(server_name=server_name)
            
            self.logger.info(f"获取到 {len(tools)} 个MCP工具 (服务器: {server_name or '全部'})")
            return tools
            
        except Exception as e:
            self.logger.error(f"获取MCP工具失败: {e}")
            return []
    
    async def get_tools_with_session(self, server_name: str) -> List[BaseTool]:
        """使用显式会话获取工具 - 官方推荐的高级用法
        
        Args:
            server_name: 服务器名称
            
        Returns:
            List[BaseTool]: 工具列表
        """
        if not await self.initialize():
            return []
        
        try:
            # 使用官方推荐的session方法
            async with self._client.session(server_name) as session:
                tools = await load_mcp_tools(session)
                self.logger.info(f"通过会话获取到 {len(tools)} 个工具 (服务器: {server_name})")
                return tools
                
        except Exception as e:
            self.logger.error(f"通过会话获取工具失败: {e}")
            return []
    
    async def get_resources(self, server_name: str, uris: Optional[Union[str, List[str]]] = None) -> List[Blob]:
        """获取MCP资源 - 使用官方方法
        
        Args:
            server_name: 服务器名称
            uris: 资源URI或URI列表，None表示获取所有资源
            
        Returns:
            List[Blob]: 资源列表
        """
        if not await self.initialize():
            return []
        
        try:
            # 使用官方推荐的get_resources方法，注意参数格式
            resources = await self._client.get_resources(server_name, uris=uris)
            
            self.logger.info(f"获取到 {len(resources)} 个资源 (服务器: {server_name})")
            return resources
            
        except Exception as e:
            self.logger.error(f"获取MCP资源失败: {e}")
            return []
    
    async def get_prompt(self, server_name: str, name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[List[Any]]:
        """获取MCP提示 - 使用官方方法
        
        Args:
            server_name: 服务器名称
            name: 提示名称
            arguments: 提示参数
            
        Returns:
            Optional[List[Any]]: 提示消息列表
        """
        if not await self.initialize():
            return None
        
        try:
            # 使用官方推荐的get_prompt方法，注意参数格式
            prompt_messages = await self._client.get_prompt(server_name, name, arguments=arguments)
            
            self.logger.info(f"获取到提示: {name} (服务器: {server_name})")
            return prompt_messages
            
        except Exception as e:
            self.logger.error(f"获取MCP提示失败: {e}")
            return None
    
    async def health_check(self, server_name: Optional[str] = None) -> Dict[str, bool]:
        """健康检查 - 检查MCP服务器连接状态
        
        Args:
            server_name: 服务器名称，None表示检查所有服务器
            
        Returns:
            Dict[str, bool]: 服务器名称到健康状态的映射
        """
        if not await self.initialize():
            return {}
        
        health_status = {}
        
        if server_name:
            # 检查特定服务器
            servers_to_check = [server_name]
        else:
            # 检查所有服务器
            servers_to_check = list(self._connections.keys())
        
        for srv_name in servers_to_check:
            try:
                # 尝试获取工具来测试连接
                tools = await self.get_tools(srv_name)
                health_status[srv_name] = True
                self.logger.debug(f"服务器 {srv_name} 健康检查通过")
            except Exception as e:
                health_status[srv_name] = False
                self.logger.warning(f"服务器 {srv_name} 健康检查失败: {e}")
        
        return health_status
    
    def get_server_names(self) -> List[str]:
        """获取所有配置的服务器名称
        
        Returns:
            List[str]: 服务器名称列表
        """
        return list(self._connections.keys())
    
    def is_initialized(self) -> bool:
        """检查是否已初始化
        
        Returns:
            bool: 是否已初始化
        """
        return self._initialized
    
    async def close(self):
        """关闭MCP客户端连接"""
        if self._client:
            # MultiServerMCPClient 没有显式的close方法
            # 但我们可以清理状态
            self._client = None
            self._initialized = False
            self.logger.info("MCP客户端已关闭")


# 全局MCP管理器实例
_mcp_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    """获取全局MCP管理器实例
    
    Returns:
        MCPManager: MCP管理器实例
    """
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
    return _mcp_manager


async def initialize_mcp_manager() -> bool:
    """初始化全局MCP管理器
    
    Returns:
        bool: 是否初始化成功
    """
    manager = get_mcp_manager()
    return await manager.initialize()


# 便捷函数
async def get_all_mcp_tools() -> List[BaseTool]:
    """获取所有MCP工具的便捷函数
    
    Returns:
        List[BaseTool]: 所有MCP工具列表
    """
    manager = get_mcp_manager()
    return await manager.get_tools()


async def get_mcp_tools_by_server(server_name: str) -> List[BaseTool]:
    """按服务器获取MCP工具的便捷函数
    
    Args:
        server_name: 服务器名称
        
    Returns:
        List[BaseTool]: 指定服务器的工具列表
    """
    manager = get_mcp_manager()
    return await manager.get_tools(server_name)


async def get_mcp_resources_by_server(server_name: str, uris: Optional[Union[str, List[str]]] = None) -> List[Blob]:
    """按服务器获取MCP资源的便捷函数
    
    Args:
        server_name: 服务器名称
        uris: 资源URI或URI列表
        
    Returns:
        List[Blob]: 资源列表
    """
    manager = get_mcp_manager()
    return await manager.get_resources(server_name, uris)