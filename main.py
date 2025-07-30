"""
多智能体LangGraph项目 - 主应用入口

本模块是整个系统的主入口，提供：
- FastAPI Web服务
- 智能体API接口
- 系统管理接口
- WebSocket实时通信
- 健康检查和监控
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.settings import get_settings
from bootstrap import get_bootstrap, system_lifespan
from core.agents import (
    get_agent_registry, 
    get_agent_manager, 
    initialize_agent_manager,
    AgentType
)
from core.tools import get_tool_registry
from core.memory import get_memory_manager
from core.cache.redis_manager import get_cache_manager
from core.error import get_error_handler, get_performance_monitor


# 数据模型
class ChatMessage(BaseModel):
    """聊天消息模型"""
    content: str
    user_id: str = "default"
    session_id: Optional[str] = None
    agent_type: str = "supervisor"
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """聊天响应模型"""
    content: str
    agent_type: str
    session_id: str
    metadata: Optional[Dict[str, Any]] = None


class SystemStatus(BaseModel):
    """系统状态模型"""
    status: str
    uptime: float
    components: Dict[str, bool]
    settings: Dict[str, Any]


class HealthCheck(BaseModel):
    """健康检查模型"""
    status: str
    components: Dict[str, Dict[str, Any]]
    timestamp: float


# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化系统
    async with system_lifespan() as bootstrap:
        app.state.bootstrap = bootstrap
        app.state.agent_registry = get_agent_registry()
        app.state.agent_manager = await initialize_agent_manager()
        app.state.tool_registry = get_tool_registry()
        app.state.memory_manager = get_memory_manager("postgres")
        app.state.cache_manager = await get_cache_manager()
        app.state.error_handler = get_error_handler()
        app.state.performance_monitor = get_performance_monitor()
        
        logger = logging.getLogger("app")
        logger.info("应用启动完成")
        
        yield
        
        # 关闭时清理资源
        logger.info("应用正在关闭...")
        await app.state.agent_manager.stop()
        await app.state.cache_manager.cleanup()


# 创建FastAPI应用
def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    settings = get_settings()
    
    app = FastAPI(
        title="LangGraph Multi-Agent System",
        description="基于LangGraph的多智能体协作系统",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=settings.app.cors_methods,
        allow_headers=settings.app.cors_headers,
    )
    
    return app


app = create_app()


# WebSocket连接管理
class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """接受WebSocket连接"""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """发送个人消息"""
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        """广播消息"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # 连接已断开，移除
                self.disconnect(connection)


manager = ConnectionManager()


# API路由
@app.get("/")
async def root():
    """根路径"""
    return {"message": "LangGraph Multi-Agent System", "version": "1.0.0"}


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """系统健康检查"""
    try:
        bootstrap = app.state.bootstrap
        health_data = await bootstrap.health_check()
        
        # 添加Redis缓存健康检查
        cache_manager = app.state.cache_manager
        redis_health = await cache_manager.health_check()
        health_data["components"]["redis"] = redis_health
        
        # 添加智能体管理器健康检查
        agent_manager = app.state.agent_manager
        agent_health = await agent_manager.health_check()
        health_data["components"]["agent_manager"] = agent_health
        
        return HealthCheck(**health_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@app.get("/status", response_model=SystemStatus)
async def system_status():
    """获取系统状态"""
    try:
        bootstrap = app.state.bootstrap
        status_data = bootstrap.get_system_status()
        return SystemStatus(**status_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """聊天接口"""
    try:
        agent_manager = app.state.agent_manager
        
        # 解析智能体类型
        try:
            agent_type = AgentType(message.agent_type)
        except ValueError:
            agent_type = AgentType.SUPERVISOR  # 默认使用supervisor
        
        # 创建或获取智能体实例
        instances = await agent_manager.list_instances(
            user_id=message.user_id,
            agent_type=agent_type
        )
        
        if instances:
            # 使用现有实例
            instance_id = instances[0].instance_id
        else:
            # 创建新实例
            instance_id = await agent_manager.create_agent(
                agent_type=agent_type,
                user_id=message.user_id,
                session_id=message.session_id
            )
        
        # 构建聊天请求
        from core.agents.base import ChatRequest
        chat_request = ChatRequest(
            message=message.content,
            user_id=message.user_id,
            session_id=message.session_id or "default",
            context=message.context or {}
        )
        
        # 处理消息
        response = await agent_manager.process_message(instance_id, chat_request)
        
        return ChatResponse(
            content=response.content,
            agent_type=message.agent_type,
            session_id=response.session_id,
            metadata=response.metadata or {}
        )
        
    except Exception as e:
        error_handler = app.state.error_handler
        await error_handler.handle_error(e, context={
            "endpoint": "/chat",
            "user_id": message.user_id,
            "agent_type": message.agent_type
        })
        raise HTTPException(status_code=500, detail=f"聊天处理失败: {str(e)}")


@app.get("/agents")
async def list_agents():
    """获取所有智能体实例"""
    try:
        agent_manager = app.state.agent_manager
        instances = await agent_manager.list_instances()
        
        return {
            "instances": [
                {
                    "instance_id": inst.instance_id,
                    "agent_type": inst.agent_type.value,
                    "user_id": inst.user_id,
                    "session_id": inst.session_id,
                    "status": inst.status.value,
                    "created_at": inst.created_at.isoformat(),
                    "last_used": inst.last_used.isoformat()
                }
                for inst in instances
            ],
            "total": len(instances)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取智能体列表失败: {str(e)}")


@app.get("/agents/{instance_id}")
async def get_agent(instance_id: str):
    """获取特定智能体实例信息"""
    try:
        agent_manager = app.state.agent_manager
        instance = await agent_manager.get_instance_info(instance_id)
        
        if not instance:
            raise HTTPException(status_code=404, detail="智能体实例不存在")
        
        # 获取性能指标
        metrics = await agent_manager.get_performance_metrics(instance_id)
        
        return {
            "instance_id": instance.instance_id,
            "agent_type": instance.agent_type.value,
            "user_id": instance.user_id,
            "session_id": instance.session_id,
            "status": instance.status.value,
            "created_at": instance.created_at.isoformat(),
            "last_used": instance.last_used.isoformat(),
            "config": instance.config.dict(),
            "metadata": instance.metadata,
            "performance": {
                "total_requests": metrics.total_requests if metrics else 0,
                "success_rate": metrics.success_rate if metrics else 0.0,
                "avg_response_time": metrics.avg_response_time if metrics else 0.0
            } if metrics else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取智能体信息失败: {str(e)}")


@app.delete("/agents/{instance_id}")
async def delete_agent(instance_id: str):
    """删除智能体实例"""
    try:
        agent_manager = app.state.agent_manager
        instance = await agent_manager.get_instance_info(instance_id)
        
        if not instance:
            raise HTTPException(status_code=404, detail="智能体实例不存在")
        
        await agent_manager.cleanup_agent(instance_id)
        
        return {"message": f"智能体实例 {instance_id} 已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除智能体失败: {str(e)}")


@app.post("/agents")
async def create_agent(
    agent_type: str,
    user_id: str,
    session_id: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
):
    """创建新的智能体实例"""
    try:
        agent_manager = app.state.agent_manager
        
        # 解析智能体类型
        try:
            agent_type_enum = AgentType(agent_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效的智能体类型: {agent_type}")
        
        instance_id = await agent_manager.create_agent(
            agent_type=agent_type_enum,
            user_id=user_id,
            session_id=session_id,
            custom_config=custom_config
        )
        
        return {
            "instance_id": instance_id,
            "agent_type": agent_type,
            "user_id": user_id,
            "session_id": session_id,
            "message": "智能体实例创建成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建智能体失败: {str(e)}")


@app.get("/agents/types")
async def list_agent_types():
    """获取所有可用的智能体类型"""
    try:
        agent_registry = app.state.agent_registry
        agent_types = agent_registry.list_agent_types()
        
        return {
            "agent_types": [
                {
                    "type": agent_type.value,
                    "config": agent_registry.get_agent_config(agent_type).dict() if agent_registry.get_agent_config(agent_type) else None
                }
                for agent_type in agent_types
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取智能体类型失败: {str(e)}")


@app.get("/agents/performance")
async def get_agents_performance():
    """获取所有智能体的性能指标"""
    try:
        agent_manager = app.state.agent_manager
        all_metrics = await agent_manager.get_all_performance_metrics()
        
        return {
            "performance_metrics": {
                instance_id: {
                    "total_requests": metrics.total_requests,
                    "success_rate": metrics.success_rate,
                    "error_rate": metrics.error_rate,
                    "avg_response_time": metrics.avg_response_time,
                    "min_response_time": metrics.min_response_time,
                    "max_response_time": metrics.max_response_time,
                    "uptime_hours": metrics.uptime_hours,
                    "last_request_time": metrics.last_request_time.isoformat() if metrics.last_request_time else None
                }
                for instance_id, metrics in all_metrics.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")


@app.get("/tools")
async def list_tools():
    """获取所有工具"""
    try:
        tool_registry = app.state.tool_registry
        tools = tool_registry.get_all_tools()
        return {"tools": [{"name": name, "description": tool.description} for name, tool in tools.items()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工具列表失败: {str(e)}")


@app.get("/memory/{user_id}")
async def get_user_memory(user_id: str, limit: int = 10):
    """获取用户记忆"""
    try:
        memory_manager = app.state.memory_manager
        memories = await memory_manager.search_memories(
            user_id=user_id,
            query="",
            limit=limit
        )
        return {"memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户记忆失败: {str(e)}")


@app.delete("/memory/{user_id}")
async def clear_user_memory(user_id: str):
    """清除用户记忆"""
    try:
        memory_manager = app.state.memory_manager
        await memory_manager.clear_user_memories(user_id)
        return {"message": f"用户 {user_id} 的记忆已清除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除用户记忆失败: {str(e)}")


@app.get("/cache/health")
async def cache_health():
    """缓存健康检查"""
    try:
        cache_manager = app.state.cache_manager
        health = await cache_manager.health_check()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"缓存健康检查失败: {str(e)}")


@app.get("/cache/keys")
async def list_cache_keys(pattern: str = "*"):
    """列出缓存键"""
    try:
        cache_manager = app.state.cache_manager
        keys = await cache_manager.redis_manager.keys(pattern)
        return {"keys": keys, "count": len(keys)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取缓存键失败: {str(e)}")


@app.delete("/cache/clear")
async def clear_cache():
    """清空缓存"""
    try:
        cache_manager = app.state.cache_manager
        success = await cache_manager.redis_manager.flushdb()
        if success:
            return {"message": "缓存已清空"}
        else:
            raise HTTPException(status_code=500, detail="清空缓存失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空缓存失败: {str(e)}")


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket聊天接口"""
    await manager.connect(websocket)
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            
            # 这里可以解析JSON消息并处理
            try:
                import json
                message_data = json.loads(data)
                
                # 创建聊天消息
                message = ChatMessage(
                    content=message_data.get("content", ""),
                    user_id=user_id,
                    session_id=message_data.get("session_id"),
                    agent_type=message_data.get("agent_type", "supervisor"),
                    context=message_data.get("context")
                )
                
                # 处理消息
                agent_registry = app.state.agent_registry
                agent = await agent_registry.get_or_create_agent(
                    agent_type=message.agent_type,
                    user_id=message.user_id,
                    session_id=message.session_id
                )
                
                response = await agent.process_message(
                    content=message.content,
                    context=message.context or {}
                )
                
                # 发送响应
                response_data = {
                    "content": response.get("content", ""),
                    "agent_type": message.agent_type,
                    "session_id": response.get("session_id", message.session_id or "default"),
                    "metadata": response.get("metadata", {})
                }
                
                await manager.send_personal_message(
                    json.dumps(response_data, ensure_ascii=False),
                    websocket
                )
                
            except json.JSONDecodeError:
                # 如果不是JSON，直接作为文本处理
                await manager.send_personal_message(f"收到消息: {data}", websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/metrics")
async def get_metrics():
    """获取性能指标"""
    try:
        performance_monitor = app.state.performance_monitor
        metrics = performance_monitor.get_all_metrics()
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")


# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    try:
        error_handler = app.state.error_handler
        await error_handler.handle_error(exc, context={
            "url": str(request.url),
            "method": request.method
        })
    except:
        pass  # 避免错误处理器本身出错
    
    return JSONResponse(
        status_code=500,
        content={"detail": f"内部服务器错误: {str(exc)}"}
    )


def main():
    """主函数"""
    settings = get_settings()
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, settings.logging.log_level.upper()),
        format=settings.logging.log_format
    )
    
    # 启动服务器
    uvicorn.run(
        "main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.reload,
        log_level=settings.logging.log_level.lower()
    )


if __name__ == "__main__":
    main()