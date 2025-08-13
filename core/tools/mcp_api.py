"""
MCP API路由模块

提供MCP工具管理的REST API接口，包括：
- MCP服务器管理
- MCP工具发现和执行
- MCP资源管理
- MCP提示管理
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import logging

from core.tools.mcp_manager import get_mcp_manager, MCPManager
from core.tools import get_tool_registry, ToolExecutionContext, ToolExecutionResult

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/mcp", tags=["MCP"])


# Pydantic模型
class MCPServerInfo(BaseModel):
    """MCP服务器信息"""
    name: str
    status: str
    connected: bool
    tools_count: int
    resources_count: int
    prompts_count: int
    last_ping: Optional[str] = None
    error: Optional[str] = None


class MCPToolInfo(BaseModel):
    """MCP工具信息"""
    name: str
    description: str
    server_name: str
    input_schema: Dict[str, Any]
    is_available: bool


class MCPResourceInfo(BaseModel):
    """MCP资源信息"""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    server_name: str


class MCPPromptInfo(BaseModel):
    """MCP提示信息"""
    name: str
    description: Optional[str] = None
    arguments: List[Dict[str, Any]]
    server_name: str


class ToolExecutionRequest(BaseModel):
    """工具执行请求"""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None


class ResourceReadRequest(BaseModel):
    """资源读取请求"""
    uri: str


class PromptExecuteRequest(BaseModel):
    """提示执行请求"""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


# 依赖注入
async def get_mcp_manager_dependency() -> MCPManager:
    """获取MCP管理器依赖"""
    try:
        return get_mcp_manager()
    except Exception as e:
        logger.error(f"获取MCP管理器失败: {e}")
        raise HTTPException(status_code=500, detail="MCP服务不可用")


# API端点
@router.get("/servers", response_model=List[MCPServerInfo])
async def list_servers(
    mcp_manager: MCPManager = Depends(get_mcp_manager_dependency)
) -> List[MCPServerInfo]:
    """列出所有MCP服务器"""
    try:
        servers_info = []
        
        # 获取健康检查状态
        health_status = await mcp_manager.health_check()
        
        for server_name in mcp_manager.get_server_names():
            try:
                is_connected = health_status.get(server_name, False)
                
                # 获取工具、资源数量
                tools_count = 0
                resources_count = 0
                
                if is_connected:
                    try:
                        tools = await mcp_manager.get_tools(server_name)
                        tools_count = len(tools)
                        
                        resources = await mcp_manager.get_resources(server_name)
                        resources_count = len(resources)
                    except Exception:
                        pass
                
                server_info = MCPServerInfo(
                    name=server_name,
                    status="connected" if is_connected else "disconnected",
                    connected=is_connected,
                    tools_count=tools_count,
                    resources_count=resources_count,
                    prompts_count=0,  # 暂时设为0，可以后续添加
                    last_ping=None,
                    error=None
                )
                
            except Exception as e:
                server_info = MCPServerInfo(
                    name=server_name,
                    status="error",
                    connected=False,
                    tools_count=0,
                    resources_count=0,
                    prompts_count=0,
                    error=str(e)
                )
            
            servers_info.append(server_info)
        
        return servers_info
        
    except Exception as e:
        logger.error(f"列出MCP服务器失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/servers/{server_name}/reconnect")
async def reconnect_server(
    server_name: str,
    background_tasks: BackgroundTasks,
    mcp_manager: MCPManager = Depends(get_mcp_manager_dependency)
):
    """重连MCP服务器"""
    try:
        # 在后台任务中重连
        background_tasks.add_task(mcp_manager.reconnect_server, server_name)
        
        return {"message": f"正在重连服务器 {server_name}"}
        
    except Exception as e:
        logger.error(f"重连MCP服务器失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools", response_model=List[MCPToolInfo])
async def list_tools(
    server_name: Optional[str] = None,
    mcp_manager: MCPManager = Depends(get_mcp_manager_dependency)
) -> List[MCPToolInfo]:
    """列出MCP工具"""
    try:
        if server_name:
            # 列出特定服务器的工具
            tools = await mcp_manager.get_tools(server_name)
            tools_info = [
                MCPToolInfo(
                    name=tool.name,
                    description=tool.description,
                    server_name=server_name,
                    input_schema=getattr(tool, 'input_schema', {}),
                    is_available=True
                )
                for tool in tools
            ]
        else:
            # 列出所有工具
            tools_info = []
            
            for srv_name in mcp_manager.get_server_names():
                try:
                    tools = await mcp_manager.get_tools(srv_name)
                    for tool in tools:
                        tools_info.append(MCPToolInfo(
                            name=tool.name,
                            description=tool.description,
                            server_name=srv_name,
                            input_schema=getattr(tool, 'input_schema', {}),
                            is_available=True
                        ))
                except Exception as e:
                    logger.warning(f"获取服务器 {srv_name} 工具失败: {e}")
                    continue
        
        return tools_info
        
    except Exception as e:
        logger.error(f"列出MCP工具失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/execute", response_model=Dict[str, Any])
async def execute_tool(
    request: ToolExecutionRequest,
    mcp_manager: MCPManager = Depends(get_mcp_manager_dependency)
) -> Dict[str, Any]:
    """执行MCP工具"""
    try:
        # 通过工具注册表执行
        tool_registry = get_tool_registry()
        
        # 创建执行上下文
        context = ToolExecutionContext(
            user_id=request.context.get("user_id", "anonymous") if request.context else "anonymous",
            session_id=request.context.get("session_id", "default") if request.context else "default",
            permissions=request.context.get("permissions", []) if request.context else [],
            metadata=request.context.get("metadata", {}) if request.context else {}
        )
        
        # 执行工具
        result = await tool_registry.execute_tool(
            request.tool_name,
            context,
            **request.arguments
        )
        
        return {
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "execution_time": result.execution_time,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"执行MCP工具失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resources", response_model=List[MCPResourceInfo])
async def list_resources(
    server_name: Optional[str] = None,
    mcp_manager: MCPManager = Depends(get_mcp_manager_dependency)
) -> List[MCPResourceInfo]:
    """列出MCP资源"""
    try:
        resources_info = []
        
        if server_name:
            # 列出特定服务器的资源
            resources = await mcp_manager.get_resources(server_name)
            for resource in resources:
                resources_info.append(MCPResourceInfo(
                    uri=resource.uri,
                    name=resource.name,
                    description=resource.description,
                    mime_type=resource.mimeType,
                    server_name=server_name
                ))
        else:
            # 列出所有服务器的资源
            for srv_name in mcp_manager.clients.keys():
                try:
                    resources = await mcp_manager.get_resources(srv_name)
                    for resource in resources:
                        resources_info.append(MCPResourceInfo(
                            uri=resource.uri,
                            name=resource.name,
                            description=resource.description,
                            mime_type=resource.mimeType,
                            server_name=srv_name
                        ))
                except Exception as e:
                    logger.warning(f"获取服务器 {srv_name} 资源失败: {e}")
                    continue
        
        return resources_info
        
    except Exception as e:
        logger.error(f"列出MCP资源失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resources/read")
async def read_resource(
    request: ResourceReadRequest,
    mcp_manager: MCPManager = Depends(get_mcp_manager_dependency)
) -> Dict[str, Any]:
    """读取MCP资源"""
    try:
        content = await mcp_manager.read_resource(request.uri)
        
        return {
            "uri": request.uri,
            "content": content,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"读取MCP资源失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompts", response_model=List[MCPPromptInfo])
async def list_prompts(
    server_name: Optional[str] = None,
    mcp_manager: MCPManager = Depends(get_mcp_manager_dependency)
) -> List[MCPPromptInfo]:
    """列出MCP提示"""
    try:
        prompts_info = []
        
        if server_name:
            # 列出特定服务器的提示
            prompts = await mcp_manager.get_prompts(server_name)
            for prompt in prompts:
                prompts_info.append(MCPPromptInfo(
                    name=prompt.name,
                    description=prompt.description,
                    arguments=prompt.arguments,
                    server_name=server_name
                ))
        else:
            # 列出所有服务器的提示
            for srv_name in mcp_manager.clients.keys():
                try:
                    prompts = await mcp_manager.get_prompts(srv_name)
                    for prompt in prompts:
                        prompts_info.append(MCPPromptInfo(
                            name=prompt.name,
                            description=prompt.description,
                            arguments=prompt.arguments,
                            server_name=srv_name
                        ))
                except Exception as e:
                    logger.warning(f"获取服务器 {srv_name} 提示失败: {e}")
                    continue
        
        return prompts_info
        
    except Exception as e:
        logger.error(f"列出MCP提示失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompts/execute")
async def execute_prompt(
    request: PromptExecuteRequest,
    mcp_manager: MCPManager = Depends(get_mcp_manager_dependency)
) -> Dict[str, Any]:
    """执行MCP提示"""
    try:
        result = await mcp_manager.get_prompt(request.name, request.arguments)
        
        return {
            "name": request.name,
            "result": result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"执行MCP提示失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
async def refresh_mcp_tools(
    background_tasks: BackgroundTasks,
    mcp_manager: MCPManager = Depends(get_mcp_manager_dependency)
):
    """刷新MCP工具到工具注册表"""
    try:
        # 在后台任务中刷新
        tool_registry = get_tool_registry()
        background_tasks.add_task(tool_registry.refresh_mcp_tools)
        
        return {"message": "正在刷新MCP工具"}
        
    except Exception as e:
        logger.error(f"刷新MCP工具失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_mcp_status(
    mcp_manager: MCPManager = Depends(get_mcp_manager_dependency)
) -> Dict[str, Any]:
    """获取MCP系统状态"""
    try:
        # 统计信息
        total_servers = len(mcp_manager.clients)
        connected_servers = 0
        total_tools = 0
        total_resources = 0
        total_prompts = 0
        
        for server_name in mcp_manager.clients.keys():
            try:
                if await mcp_manager.ping_server(server_name):
                    connected_servers += 1
                    
                    tools = await mcp_manager.get_tools(server_name)
                    resources = await mcp_manager.get_resources(server_name)
                    prompts = await mcp_manager.get_prompts(server_name)
                    
                    total_tools += len(tools)
                    total_resources += len(resources)
                    total_prompts += len(prompts)
                    
            except Exception:
                continue
        
        return {
            "total_servers": total_servers,
            "connected_servers": connected_servers,
            "disconnected_servers": total_servers - connected_servers,
            "total_tools": total_tools,
            "total_resources": total_resources,
            "total_prompts": total_prompts,
            "mcp_available": True
        }
        
    except Exception as e:
        logger.error(f"获取MCP状态失败: {e}")
        return {
            "mcp_available": False,
            "error": str(e)
        }