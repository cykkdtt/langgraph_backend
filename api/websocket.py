"""WebSocket连接管理和实时聊天实现

本模块实现了LangGraph多智能体系统的WebSocket功能，包括：
- WebSocket连接管理
- 实时聊天通信
- 智能体状态广播
- 会话状态同步
- 连接池管理
"""

from typing import Dict, List, Optional, Set, Any
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.routing import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
import json
import asyncio
from datetime import datetime, timedelta, timezone
import uuid
from collections import defaultdict
import weakref

from models.chat_models import (
    WebSocketMessage, WebSocketMessageType, ChatRequest, StreamChunk,
    StreamEventType, AgentStatus, ConnectionInfo, ChatError
)
from models.database_models import User, Session
from models.auth_models import UserInfo
from api.auth import get_current_user_websocket
from core.database import get_async_session
from core.agents import get_agent_manager
from core.logging import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/ws", tags=["WebSocket"])


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 活跃连接: connection_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        
        # 用户连接映射: user_id -> Set[connection_id]
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # 会话连接映射: session_id -> Set[connection_id]
        self.session_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # 连接信息: connection_id -> ConnectionInfo
        self.connection_info: Dict[str, ConnectionInfo] = {}
        
        # 心跳任务: connection_id -> asyncio.Task
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
        
        # 连接锁
        self._lock = asyncio.Lock()
    
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: str, 
        session_id: Optional[str] = None,
        client_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """建立WebSocket连接"""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        
        async with self._lock:
            # 存储连接
            self.active_connections[connection_id] = websocket
            self.user_connections[user_id].add(connection_id)
            
            if session_id:
                self.session_connections[session_id].add(connection_id)
            
            # 存储连接信息
            self.connection_info[connection_id] = ConnectionInfo(
                connection_id=connection_id,
                user_id=user_id,
                session_id=session_id,
                client_info=client_info or {}
            )
            
            # 启动心跳任务
            self.heartbeat_tasks[connection_id] = asyncio.create_task(
                self._heartbeat_task(connection_id)
            )
        
        logger.info(f"WebSocket连接建立: {connection_id}, 用户: {user_id}")
        
        # 发送连接确认消息
        await self.send_to_connection(
            connection_id,
            WebSocketMessage(
                type=WebSocketMessageType.CHAT_MESSAGE,
                data={
                    "event": "connected",
                    "connection_id": connection_id,
                    "message": "WebSocket连接已建立"
                },
                user_id=user_id,
                session_id=session_id
            )
        )
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """断开WebSocket连接"""
        async with self._lock:
            if connection_id not in self.active_connections:
                return
            
            # 获取连接信息
            conn_info = self.connection_info.get(connection_id)
            if conn_info:
                user_id = conn_info.user_id
                session_id = conn_info.session_id
                
                # 从映射中移除
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
                
                if session_id and session_id in self.session_connections:
                    self.session_connections[session_id].discard(connection_id)
                    if not self.session_connections[session_id]:
                        del self.session_connections[session_id]
            
            # 移除连接
            del self.active_connections[connection_id]
            
            if connection_id in self.connection_info:
                del self.connection_info[connection_id]
            
            # 取消心跳任务
            if connection_id in self.heartbeat_tasks:
                self.heartbeat_tasks[connection_id].cancel()
                del self.heartbeat_tasks[connection_id]
        
        logger.info(f"WebSocket连接断开: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, message):
        """向指定连接发送消息"""
        websocket = self.active_connections.get(connection_id)
        if websocket:
            try:
                # 处理不同类型的消息对象
                if isinstance(message, dict):
                    # 如果是字典，直接使用json.dumps序列化
                    message_text = json.dumps(message)
                elif hasattr(message, 'model_dump_json'):
                    # 如果是Pydantic模型，使用model_dump_json方法
                    message_text = message.model_dump_json()
                else:
                    # 其他情况，尝试转换为字符串
                    message_text = str(message)
                
                await websocket.send_text(message_text)
                
                # 更新最后活动时间
                if connection_id in self.connection_info:
                    self.connection_info[connection_id].last_activity = datetime.now(timezone.utc)
                    
            except Exception as e:
                logger.error(f"发送消息失败: {connection_id}, {e}")
                await self.disconnect(connection_id)
    
    async def send_to_user(self, user_id: str, message: WebSocketMessage):
        """向用户的所有连接发送消息"""
        connection_ids = self.user_connections.get(user_id, set()).copy()
        for connection_id in connection_ids:
            await self.send_to_connection(connection_id, message)
    
    async def send_to_session(self, session_id: str, message: WebSocketMessage):
        """向会话的所有连接发送消息"""
        connection_ids = self.session_connections.get(session_id, set()).copy()
        for connection_id in connection_ids:
            await self.send_to_connection(connection_id, message)
    
    async def broadcast(self, message: WebSocketMessage, exclude_connections: Optional[Set[str]] = None):
        """广播消息到所有连接"""
        exclude_connections = exclude_connections or set()
        connection_ids = set(self.active_connections.keys()) - exclude_connections
        
        for connection_id in connection_ids:
            await self.send_to_connection(connection_id, message)
    
    async def get_user_connections(self, user_id: str) -> List[str]:
        """获取用户的所有连接ID"""
        return list(self.user_connections.get(user_id, set()))
    
    async def get_session_connections(self, session_id: str) -> List[str]:
        """获取会话的所有连接ID"""
        return list(self.session_connections.get(session_id, set()))
    
    async def get_connection_count(self) -> int:
        """获取活跃连接数"""
        return len(self.active_connections)
    
    async def get_user_count(self) -> int:
        """获取在线用户数"""
        return len(self.user_connections)
    
    async def cleanup_inactive_connections(self, timeout_minutes: int = 30):
        """清理非活跃连接"""
        timeout = timedelta(minutes=timeout_minutes)
        current_time = datetime.now(timezone.utc)
        
        inactive_connections = []
        
        for connection_id, conn_info in self.connection_info.items():
            if current_time - conn_info.last_activity > timeout:
                inactive_connections.append(connection_id)
        
        for connection_id in inactive_connections:
            await self.disconnect(connection_id)
        
        if inactive_connections:
            logger.info(f"清理了 {len(inactive_connections)} 个非活跃连接")
    
    async def _heartbeat_task(self, connection_id: str):
        """心跳任务"""
        try:
            while connection_id in self.active_connections:
                await asyncio.sleep(30)  # 30秒心跳间隔
                
                ping_message = WebSocketMessage(
                    type=WebSocketMessageType.PING,
                    data={"timestamp": datetime.now(timezone.utc).isoformat()}
                )
                
                await self.send_to_connection(connection_id, ping_message)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"心跳任务异常: {connection_id}, {e}")
            await self.disconnect(connection_id)


# 全局连接管理器实例
connection_manager = ConnectionManager()


class WebSocketChatHandler:
    """WebSocket聊天处理器"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def handle_chat_message(
        self, 
        connection_id: str, 
        message_data: Dict[str, Any],
        db: AsyncSession
    ):
        """处理聊天消息"""
        try:
            # 解析聊天请求
            chat_request = ChatRequest(**message_data)
            
            # 获取连接信息
            conn_info = self.connection_manager.connection_info.get(connection_id)
            if not conn_info:
                raise ValueError("连接信息不存在")
            
            # 获取或创建会话
            session = await self._get_or_create_session(
                db, chat_request.session_id, conn_info.user_id, chat_request.mode
            )
            
            # 选择智能体
            agent = await self._select_agent(chat_request)
            
            # 发送智能体状态更新
            await self._broadcast_agent_status(
                session.id, agent.agent_type, "thinking"
            )
            
            # 流式处理聊天
            if chat_request.stream:
                await self._handle_stream_chat(
                    connection_id, chat_request, session, agent, db
                )
            else:
                await self._handle_regular_chat(
                    connection_id, chat_request, session, agent, db
                )
                
        except Exception as e:
            logger.error(f"处理聊天消息失败: {e}")
            await self._send_error(connection_id, str(e))
    
    async def _handle_stream_chat(
        self, 
        connection_id: str, 
        request: ChatRequest, 
        session, 
        agent, 
        db: AsyncSession
    ):
        """处理流式聊天"""
        try:
            # 获取聊天历史
            history = await self._get_chat_history(db, session.id, request.max_history)
            
            # 发送开始事件
            start_message = WebSocketMessage(
                type=WebSocketMessageType.STREAM_CHUNK,
                data={
                    "event_type": StreamEventType.MESSAGE_START,
                    "data": {"agent_type": agent.agent_type, "session_id": session.id}
                },
                session_id=session.id
            )
            await self.connection_manager.send_to_session(session.id, start_message)
            
            # 流式处理
            full_content = ""
            async for chunk in agent.stream_chat(
                message=request.message,
                history=history,
                context=request.context,
                tools_enabled=request.include_tools
            ):
                if chunk.event_type == StreamEventType.MESSAGE_DELTA:
                    full_content += chunk.data.get("content", "")
                
                # 广播流式块
                stream_message = WebSocketMessage(
                    type=WebSocketMessageType.STREAM_CHUNK,
                    data={
                        "event_type": chunk.event_type,
                        "data": chunk.data
                    },
                    session_id=session.id
                )
                await self.connection_manager.send_to_session(session.id, stream_message)
            
            # 保存完整消息
            await self._save_messages(db, session.id, request.message, full_content)
            
            # 发送结束事件
            end_message = WebSocketMessage(
                type=WebSocketMessageType.STREAM_CHUNK,
                data={
                    "event_type": StreamEventType.MESSAGE_END,
                    "data": {"total_content": full_content}
                },
                session_id=session.id
            )
            await self.connection_manager.send_to_session(session.id, end_message)
            
        except Exception as e:
            logger.error(f"流式聊天处理失败: {e}")
            await self._send_error(connection_id, str(e))
        finally:
            # 更新智能体状态为空闲
            await self._broadcast_agent_status(
                session.id, agent.agent_type, "idle"
            )
    
    async def _handle_regular_chat(
        self, 
        connection_id: str, 
        request: ChatRequest, 
        session, 
        agent, 
        db: AsyncSession
    ):
        """处理常规聊天"""
        try:
            # 获取聊天历史
            history = await self._get_chat_history(db, session.id, request.max_history)
            
            # 调用智能体
            response = await agent.chat(
                message=request.message,
                history=history,
                context=request.context,
                tools_enabled=request.include_tools
            )
            
            # 保存消息
            await self._save_messages(db, session.id, request.message, response.content)
            
            # 发送响应
            response_message = WebSocketMessage(
                type=WebSocketMessageType.CHAT_MESSAGE,
                data={
                    "message_id": str(uuid.uuid4()),
                    "session_id": session.id,
                    "content": response.content,
                    "agent_type": agent.agent_type,
                    "tool_calls": [call.dict() for call in response.tool_calls],
                    "metadata": response.metadata
                },
                session_id=session.id
            )
            await self.connection_manager.send_to_session(session.id, response_message)
            
        except Exception as e:
            logger.error(f"常规聊天处理失败: {e}")
            await self._send_error(connection_id, str(e))
        finally:
            # 更新智能体状态为空闲
            await self._broadcast_agent_status(
                session.id, agent.agent_type, "idle"
            )
    
    async def _get_or_create_session(self, db: AsyncSession, session_id: Optional[str], user_id: str, mode):
        """获取或创建会话"""
        # 这里复用chat.py中的逻辑
        from api.chat import get_or_create_session
        return await get_or_create_session(db, session_id, user_id, mode)
    
    async def _select_agent(self, request: ChatRequest):
        """选择智能体"""
        # 这里复用chat.py中的逻辑
        from api.chat import select_agent
        return await select_agent(request, None)  # session参数在这里不需要
    
    async def _get_chat_history(self, db: AsyncSession, session_id: str, max_history: int):
        """获取聊天历史"""
        from api.chat import get_chat_history_for_agent
        return await get_chat_history_for_agent(db, session_id, max_history)
    
    async def _save_messages(self, db: AsyncSession, session_id: str, user_message: str, agent_response: str):
        """保存用户消息和智能体响应"""
        from api.chat import save_message
        from models.chat_models import MessageRole, MessageType
        
        # 这里需要获取用户ID，简化处理
        user_id = "system"  # 实际应该从连接信息获取
        
        # 保存用户消息
        await save_message(
            db, session_id, user_id, MessageRole.USER, user_message, MessageType.TEXT
        )
        
        # 保存智能体响应
        await save_message(
            db, session_id, user_id, MessageRole.ASSISTANT, agent_response, MessageType.TEXT
        )
    
    async def _broadcast_agent_status(self, session_id: str, agent_type: str, status: str):
        """广播智能体状态"""
        status_message = WebSocketMessage(
            type=WebSocketMessageType.AGENT_STATUS,
            data=AgentStatus(
                agent_type=agent_type,
                agent_id=f"{agent_type}_{session_id}",
                status=status,
                updated_at=datetime.utcnow()
            ).dict(),
            session_id=session_id
        )
        await self.connection_manager.send_to_session(session_id, status_message)
    
    async def _send_error(self, connection_id: str, error_message: str):
        """发送错误消息"""
        error_msg = WebSocketMessage(
            type=WebSocketMessageType.ERROR,
            data=ChatError(
                error_code="CHAT_ERROR",
                error_message=error_message,
                error_type="ChatProcessingError"
            ).dict()
        )
        await self.connection_manager.send_to_connection(connection_id, error_msg)


# WebSocket聊天处理器实例
chat_handler = WebSocketChatHandler(connection_manager)


@router.websocket("/chat")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    token: Optional[str] = None,
    session_id: Optional[str] = None
):
    """WebSocket聊天端点"""
    connection_id = None
    
    try:
        # 验证用户身份
        if not token:
            await websocket.close(code=4001, reason="缺少认证令牌")
            return
        
        # 这里应该验证JWT令牌，简化处理
        user_id = "test_user"  # 实际应该从JWT解析
        
        # 建立连接
        connection_id = await connection_manager.connect(
            websocket, user_id, session_id,
            client_info={"user_agent": "WebSocket Client"}
        )
        
        # 获取数据库会话
        async with get_async_session() as db:
            # 消息处理循环
            while True:
                try:
                    # 接收消息
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    message_type = message_data.get("type")
                    
                    if message_type == WebSocketMessageType.CHAT_MESSAGE:
                        # 处理聊天消息
                        await chat_handler.handle_chat_message(
                            connection_id, message_data.get("data", {}), db
                        )
                    
                    elif message_type == WebSocketMessageType.PONG:
                        # 处理心跳响应
                        logger.debug(f"收到心跳响应: {connection_id}")
                    
                    else:
                        logger.warning(f"未知消息类型: {message_type}")
                
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await chat_handler._send_error(connection_id, "无效的JSON格式")
                except Exception as e:
                    logger.error(f"处理WebSocket消息失败: {e}")
                    await chat_handler._send_error(connection_id, str(e))
    
    except Exception as e:
        logger.error(f"WebSocket连接失败: {e}")
        if connection_id:
            await connection_manager.disconnect(connection_id)
    
    finally:
        if connection_id:
            await connection_manager.disconnect(connection_id)


@router.get("/connections/stats")
async def get_connection_stats():
    """获取连接统计信息"""
    return {
        "active_connections": await connection_manager.get_connection_count(),
        "online_users": await connection_manager.get_user_count(),
        "timestamp": datetime.utcnow().isoformat()
    }


# 定期清理任务
async def cleanup_task():
    """定期清理非活跃连接"""
    while True:
        try:
            await asyncio.sleep(300)  # 5分钟清理一次
            await connection_manager.cleanup_inactive_connections()
        except Exception as e:
            logger.error(f"清理任务异常: {e}")


# 注意：清理任务需要在应用启动时手动启动
# 可以在main.py中的startup事件中启动此任务