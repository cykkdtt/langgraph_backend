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
from datetime import datetime
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
from core.memory.store_manager import get_store_manager
from core.cache.redis_manager import get_cache_manager
from core.error import get_error_handler, get_performance_monitor

# API路由导入
from core.tools.mcp_api import router as mcp_router
from core.time_travel.time_travel_api import router as core_time_travel_router
from core.optimization.prompt_optimization_api import router as prompt_optimization_router
from api.auth import router as auth_router
from api.chat import router as chat_router
from api.websocket import router as websocket_router
from api.threads import router as threads_router
from api.workflows import router as workflows_router
from api.memory import router as memory_router
from api.time_travel import router as time_travel_router
from models.auth_models import UserInfo
from models.chat_models import ChatRequest


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
    logger = logging.getLogger("app")
    
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
        
        # 初始化提示词优化器
        try:
            from core.optimization.prompt_optimizer import PromptOptimizer, FeedbackCollector, AutoOptimizationScheduler
            
            # 创建提示词优化器实例
            prompt_optimizer = PromptOptimizer(app.state.memory_manager)
            feedback_collector = FeedbackCollector(app.state.memory_manager)
            auto_scheduler = AutoOptimizationScheduler(prompt_optimizer, feedback_collector)
            
            # 初始化优化器
            await prompt_optimizer.initialize()
            
            # 存储到应用状态
            app.state.prompt_optimizer = prompt_optimizer
            app.state.feedback_collector = feedback_collector
            app.state.auto_scheduler = auto_scheduler
            
            logger.info("提示词优化器初始化完成")
        except Exception as e:
            logger.warning(f"提示词优化器初始化失败: {e}")
        
        # 加载MCP工具到工具注册表
        try:
            await app.state.tool_registry.load_mcp_tools()
            logger.info("MCP工具加载完成")
        except Exception as e:
            logger.warning(f"MCP工具加载失败: {e}")
        
        logger.info("应用启动完成")
        
        yield
        
        # 关闭时清理资源
        logger.info("应用正在关闭...")
        
        # 停止自动优化调度器
        if hasattr(app.state, 'auto_scheduler'):
            try:
                await app.state.auto_scheduler.stop_auto_optimization()
                logger.info("自动优化调度器已停止")
            except Exception as e:
                logger.warning(f"停止自动优化调度器失败: {e}")
        
        # 清理记忆管理器
        if hasattr(app.state, 'memory_manager'):
            try:
                await app.state.memory_manager.cleanup()
                logger.info("记忆管理器已清理")
            except Exception as e:
                logger.warning(f"清理记忆管理器失败: {e}")
        
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
        allow_origins=settings.cors.allow_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )
    
    # 注册路由
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/api/v1")
    app.include_router(threads_router, prefix="/api/v1")
    app.include_router(workflows_router, prefix="/api/v1")
    app.include_router(memory_router, prefix="/api/v1")
    app.include_router(time_travel_router, prefix="/api/v1")
    app.include_router(core_time_travel_router, prefix="/api/v1/checkpoints")
    app.include_router(mcp_router, prefix="/api/v1")
    app.include_router(prompt_optimization_router, prefix="/api/v1")
    
    return app


app = create_app()


# 导入WebSocket连接管理器
from api.websocket import connection_manager as manager


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
        
        # 构建聊天请求（使用core.agents.base.ChatRequest格式）
        from langchain_core.messages import HumanMessage
        from core.agents.base import ChatRequest as BaseChatRequest
        chat_request = BaseChatRequest(
            messages=[HumanMessage(content=message.content)],
            user_id=message.user_id,
            session_id=message.session_id or "default",
            stream=False,
            metadata=message.context or {}
        )
        
        # 处理消息
        try:
            response = await agent_manager.process_message(instance_id, chat_request)
            # 暂时使用固定消息，避免Pydantic属性访问问题
            message_content = "智能体响应成功"
        except Exception as e:
            message_content = f"智能体处理失败: {str(e)}"
            response = None
        
        # 返回API响应格式（使用main.py中定义的ChatResponse）
        return ChatResponse(
            content=message_content,
            agent_type=message.agent_type,
            session_id=message.session_id or "default",
            metadata={}
        )
        
    except Exception as e:
        error_handler = app.state.error_handler
        error_handler.handle_error(e, context={
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
    """WebSocket连接端点"""
    logger = logging.getLogger("websocket")
    
    # 建立连接并获取connection_id（ConnectionManager会处理accept）
    connection_id = await manager.connect(websocket, user_id)
    logger.info(f"WebSocket连接建立: user_id={user_id}, connection_id={connection_id}")
    print(f"WebSocket连接请求: user_id={user_id}, connection_id={connection_id}")
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            logger.info(f"收到WebSocket消息: {data}")
            print(f"收到WebSocket消息: {data}")
            
            # 解析JSON消息并处理
            try:
                import json
                import uuid
                import time
                from datetime import timezone
                message_data = json.loads(data)
                
                # 检查是否为心跳消息
                if isinstance(message_data, dict) and message_data.get('type') == 'ping':
                    # 响应心跳ping消息
                    pong_response = {
                        "type": "pong",
                        "timestamp": message_data.get('timestamp', time.time())
                    }
                    await websocket.send_text(json.dumps(pong_response))
                    logger.debug(f"Sent pong response to user {user_id}")
                    continue
                
                # 检查是否为非聊天消息类型（包括ping、pong等其他类型）
                if isinstance(message_data, dict):
                    msg_type = message_data.get('type')
                    # 如果消息有type字段且不是chat_message，则跳过处理
                    if msg_type and msg_type != 'chat_message':
                        logger.debug(f"Skipping non-chat message type: {msg_type}")
                        continue
                
                # 检查消息格式，支持前端WebSocketMessage格式
                if "type" in message_data and "data" in message_data:
                    # 前端WebSocketMessage格式: {type: 'chat_message', data: {...}}
                    data_payload = message_data["data"]
                    content = data_payload.get("message", data_payload.get("content", ""))
                    session_id = data_payload.get("session_id", "default")
                    agent_type = data_payload.get("agent_type", "supervisor")
                    context = data_payload.get("context")
                    print(f"解析WebSocket消息: type={message_data['type']}, data={data_payload}")
                    print(f"提取的内容: message={content}, session_id={session_id}, agent_type={agent_type}")
                elif "type" in message_data and "payload" in message_data:
                    # 旧的前端WebSocketMessage格式
                    payload = message_data["payload"]
                    content = payload.get("content", payload.get("message", ""))
                    session_id = payload.get("session_id", "default")
                    agent_type = payload.get("agent_type", "supervisor")
                    context = payload.get("context")
                    print(f"解析消息: type={message_data['type']}, payload={payload}")
                else:
                    # 直接的消息格式
                    content = message_data.get("content", message_data.get("message", ""))
                    session_id = message_data.get("session_id", "default")
                    agent_type = message_data.get("agent_type", "supervisor")
                    context = message_data.get("context")
                    print(f"解析直接消息格式: content={content}, agent_type={agent_type}")
                
                if not content:
                    logger.warning("收到空消息内容")
                    continue
                
                # 创建聊天消息
                message = ChatMessage(
                    content=content,
                    user_id=user_id,
                    session_id=session_id,
                    agent_type=agent_type,
                    context=context
                )
                
                logger.info(f"处理聊天消息: {message.content[:50]}...")
                print(f"处理聊天消息: content={content}, agent_type={agent_type}")
                
                # 使用agent_manager处理消息
                agent_manager = app.state.agent_manager
                
                # 解析智能体类型
                try:
                    agent_type_enum = AgentType(message.agent_type)
                except ValueError:
                    agent_type_enum = AgentType.SUPERVISOR  # 默认使用supervisor
                
                # 创建或获取智能体实例
                instances = await agent_manager.list_instances(
                    user_id=message.user_id,
                    agent_type=agent_type_enum
                )
                
                if instances:
                    # 使用现有实例
                    instance_id = instances[0].instance_id
                    logger.info(f"使用现有智能体实例: {instance_id}")
                    print(f"使用现有智能体实例: {instance_id}")
                else:
                    # 创建新实例
                    instance_id = await agent_manager.create_agent(
                        agent_type=agent_type_enum,
                        user_id=message.user_id,
                        session_id=message.session_id
                    )
                    logger.info(f"创建新智能体实例: {instance_id}")
                    print(f"创建新智能体实例: {instance_id}")
                
                # 构建聊天请求
                from langchain_core.messages import HumanMessage
                from core.agents.base import ChatRequest as BaseChatRequest
                chat_request = BaseChatRequest(
                    messages=[HumanMessage(content=message.content)],
                    user_id=message.user_id,
                    session_id=message.session_id,
                    stream=False,
                    metadata=message.context or {}
                )
                
                # 处理消息
                try:
                    # 记录记忆相关信息
                    logger.info(f"开始处理消息，准备记忆保存 - 用户ID: {message.user_id}, 会话ID: {message.session_id}")
                    logger.info(f"用户消息内容: {message.content[:100]}...")
                    
                    # 检查消息中是否包含重要信息（如姓名等）
                    important_keywords = ['小白', '我叫', '我是', '名字', '姓名']
                    has_important_info = any(keyword in message.content for keyword in important_keywords)
                    if has_important_info:
                        logger.info(f"检测到重要信息，消息包含关键词: {[kw for kw in important_keywords if kw in message.content]}")
                    
                    response = await agent_manager.process_message(instance_id, chat_request)
                    
                    # 提取响应内容 - 修复消息格式问题
                    response_content = "智能体响应成功"  # 默认值
                    
                    if hasattr(response, 'content'):
                        # 如果response有content属性，直接使用
                        response_content = response.content
                    elif isinstance(response, dict):
                        # 如果是字典，尝试获取content字段
                        response_content = response.get('content', '智能体响应成功')
                    elif isinstance(response, str):
                        # 如果是字符串，直接使用
                        response_content = response
                    else:
                        # 对于其他类型（如AIMessage对象），尝试提取content
                        response_str = str(response)
                        
                        # 检查是否是AIMessage对象的字符串表示
                        if 'AIMessage(content=' in response_str:
                            # 使用正则表达式提取content内容
                            import re
                            match = re.search(r"AIMessage\(content='([^']*)'.*\)", response_str)
                            if match:
                                response_content = match.group(1)
                            else:
                                # 如果正则匹配失败，尝试其他方法
                                try:
                                    # 尝试从字符串中提取content
                                    start_idx = response_str.find("content='")
                                    if start_idx != -1:
                                        start_idx += len("content='")
                                        end_idx = response_str.find("'", start_idx)
                                        if end_idx != -1:
                                            response_content = response_str[start_idx:end_idx]
                                        else:
                                            response_content = "智能体响应成功"
                                    else:
                                        response_content = "智能体响应成功"
                                except Exception:
                                    response_content = "智能体响应成功"
                        else:
                            # 如果不是AIMessage格式，直接使用字符串
                            response_content = response_str
                    
                    logger.info(f"智能体响应: {response_content[:50]}...")
                    print(f"智能体响应: {response_content}")
                    
                    # 记录记忆保存相关信息
                    logger.info(f"消息处理完成，记忆保存应该已触发 - 智能体类型: {agent_type_enum.value}")
                    if has_important_info:
                        logger.info(f"重要信息已处理，LangMem应该已保存相关记忆")
                    
                except Exception as e:
                    logger.error(f"智能体处理失败: {str(e)}")
                    logger.error(f"记忆保存可能受到影响 - 错误详情: {str(e)}")
                    print(f"智能体处理失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    response_content = f"抱歉，处理您的消息时出现了错误: {str(e)}"
                
                # 发送响应 - 确保消息格式正确
                response_data = {
                    "content": response_content,
                    "agent_type": message.agent_type,
                    "session_id": message.session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message_id": str(uuid.uuid4())
                }
                
                from api.websocket import WebSocketMessage, WebSocketMessageType
                response_message = WebSocketMessage(
                    type=WebSocketMessageType.CHAT_MESSAGE,
                    data=response_data,
                    user_id=user_id,
                    session_id=message.session_id
                )
                
                # 确保消息被正确序列化
                message_json = response_message.model_dump_json()
                logger.info(f"发送响应消息: {message_json}")
                
                await websocket.send_text(message_json)
                logger.info("WebSocket响应已发送")
                print(f"响应已发送: {connection_id}")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {str(e)}")
                print(f"JSON解析错误: {e}")
                error_data = {
                    "content": "消息格式错误，请发送有效的JSON格式消息",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                from api.websocket import WebSocketMessage, WebSocketMessageType
                error_message = WebSocketMessage(
                    type=WebSocketMessageType.CHAT_MESSAGE,
                    data=error_data,
                    user_id=user_id
                )
                await websocket.send_text(error_message.model_dump_json())
            except Exception as e:
                logger.error(f"WebSocket消息处理错误: {str(e)}")
                print(f"消息处理错误: {e}")
                import traceback
                traceback.print_exc()
                error_data = {
                    "content": f"处理消息时发生错误: {str(e)}",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                from api.websocket import WebSocketMessage, WebSocketMessageType
                error_message = WebSocketMessage(
                    type=WebSocketMessageType.CHAT_MESSAGE,
                    data=error_data,
                    user_id=user_id
                )
                await websocket.send_text(error_message.model_dump_json())
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: user_id={user_id}, connection_id={connection_id}")
        print(f"WebSocket正常断开: {connection_id}")
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket连接异常: {str(e)}")
        print(f"WebSocket异常: {e}")
        import traceback
        traceback.print_exc()
        await manager.disconnect(connection_id)
        logger.info(f"清理WebSocket连接: {connection_id}")


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
        level=getattr(logging, settings.monitoring.log_level.value.upper()),
        format=settings.monitoring.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.monitoring.log_file) if settings.monitoring.log_file else logging.NullHandler()
        ]
    )
    
    # 启动服务器
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()