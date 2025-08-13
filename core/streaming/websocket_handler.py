"""
WebSocket 处理器

提供基于WebSocket的实时双向通信功能。
"""

import json
import asyncio
import uuid
from typing import Dict, Any, Optional, List, Set
from fastapi import WebSocket, WebSocketDisconnect
import logging

from .stream_types import StreamChunk, StreamConfig, StreamEventType


class WebSocketHandler:
    """WebSocket处理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("websocket.handler")
    
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """建立WebSocket连接
        
        Args:
            websocket: WebSocket连接
            user_id: 用户ID
            metadata: 连接元数据
            
        Returns:
            str: 连接ID
        """
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "connected_at": "now",
            **(metadata or {})
        }
        
        self.logger.info(f"WebSocket连接建立: {connection_id} (用户: {user_id})")
        
        # 发送连接确认
        await self.send_message(connection_id, {
            "type": "connection",
            "status": "connected",
            "connection_id": connection_id
        })
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """断开WebSocket连接
        
        Args:
            connection_id: 连接ID
        """
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if connection_id in self.connection_metadata:
            user_id = self.connection_metadata[connection_id].get("user_id")
            del self.connection_metadata[connection_id]
            self.logger.info(f"WebSocket连接断开: {connection_id} (用户: {user_id})")
    
    async def send_message(
        self, 
        connection_id: str, 
        message: Dict[str, Any]
    ) -> bool:
        """发送消息到指定连接
        
        Args:
            connection_id: 连接ID
            message: 消息内容
            
        Returns:
            bool: 是否发送成功
        """
        if connection_id not in self.active_connections:
            return False
        
        websocket = self.active_connections[connection_id]
        
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
            return True
        except Exception as e:
            self.logger.error(f"发送WebSocket消息失败: {e}")
            await self.disconnect(connection_id)
            return False
    
    async def broadcast_message(
        self, 
        message: Dict[str, Any],
        user_filter: Optional[List[str]] = None
    ):
        """广播消息到所有连接
        
        Args:
            message: 消息内容
            user_filter: 用户过滤列表（可选）
        """
        disconnected_connections = []
        
        for connection_id, websocket in self.active_connections.items():
            # 检查用户过滤
            if user_filter:
                user_id = self.connection_metadata.get(connection_id, {}).get("user_id")
                if user_id not in user_filter:
                    continue
            
            try:
                await websocket.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                self.logger.error(f"广播消息失败: {e}")
                disconnected_connections.append(connection_id)
        
        # 清理断开的连接
        for connection_id in disconnected_connections:
            await self.disconnect(connection_id)
    
    async def stream_to_websocket(
        self,
        connection_id: str,
        stream_generator,
        config: StreamConfig = None
    ):
        """将流式数据发送到WebSocket
        
        Args:
            connection_id: 连接ID
            stream_generator: 流式生成器
            config: 流式配置
        """
        if connection_id not in self.active_connections:
            self.logger.warning(f"连接不存在: {connection_id}")
            return
        
        if config is None:
            config = StreamConfig()
        
        try:
            async for chunk in stream_generator:
                # 检查连接是否仍然活跃
                if connection_id not in self.active_connections:
                    break
                
                # 过滤事件
                if self._should_include_event(chunk, config):
                    message = self._chunk_to_websocket_message(chunk)
                    success = await self.send_message(connection_id, message)
                    
                    if not success:
                        break
                
                # 处理完成事件
                if chunk.chunk_type == StreamEventType.COMPLETE:
                    break
        
        except Exception as e:
            self.logger.error(f"WebSocket流式传输错误: {e}")
            await self.send_message(connection_id, {
                "type": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
    
    def _chunk_to_websocket_message(self, chunk: StreamChunk) -> Dict[str, Any]:
        """将流式块转换为WebSocket消息
        
        Args:
            chunk: 流式块
            
        Returns:
            Dict: WebSocket消息
        """
        return {
            "type": chunk.chunk_type.value,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "timestamp": chunk.timestamp.isoformat() if chunk.timestamp else None,
            "node_id": chunk.node_id,
            "run_id": chunk.run_id
        }
    
    def _should_include_event(
        self, 
        chunk: StreamChunk, 
        config: StreamConfig
    ) -> bool:
        """判断是否应该包含事件
        
        Args:
            chunk: 流式块
            config: 流式配置
            
        Returns:
            bool: 是否包含
        """
        # 检查排除列表
        if chunk.chunk_type in config.exclude_events:
            return False
        
        # 检查包含列表（如果指定了）
        if config.include_events and chunk.chunk_type not in config.include_events:
            return False
        
        return True
    
    async def handle_client_message(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """处理客户端消息
        
        Args:
            connection_id: 连接ID
            message: 客户端消息
            
        Returns:
            Optional[Dict]: 响应消息
        """
        message_type = message.get("type")
        
        if message_type == "ping":
            return {"type": "pong", "timestamp": "now"}
        
        elif message_type == "subscribe":
            # 处理订阅请求
            return await self._handle_subscribe(connection_id, message)
        
        elif message_type == "unsubscribe":
            # 处理取消订阅请求
            return await self._handle_unsubscribe(connection_id, message)
        
        elif message_type == "interrupt":
            # 处理中断请求
            return await self._handle_interrupt_request(connection_id, message)
        
        else:
            self.logger.warning(f"未知消息类型: {message_type}")
            return {"type": "error", "error": f"未知消息类型: {message_type}"}
    
    async def _handle_subscribe(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理订阅请求
        
        Args:
            connection_id: 连接ID
            message: 订阅消息
            
        Returns:
            Dict: 响应消息
        """
        # 这里可以实现订阅逻辑
        # 例如订阅特定的事件类型或运行ID
        return {"type": "subscribe_ack", "status": "success"}
    
    async def _handle_unsubscribe(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理取消订阅请求
        
        Args:
            connection_id: 连接ID
            message: 取消订阅消息
            
        Returns:
            Dict: 响应消息
        """
        # 这里可以实现取消订阅逻辑
        return {"type": "unsubscribe_ack", "status": "success"}
    
    async def _handle_interrupt_request(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理中断请求
        
        Args:
            connection_id: 连接ID
            message: 中断消息
            
        Returns:
            Dict: 响应消息
        """
        # 这里可以实现中断处理逻辑
        run_id = message.get("run_id")
        if not run_id:
            return {"type": "error", "error": "缺少run_id"}
        
        # 实际的中断处理逻辑需要与StreamManager集成
        return {"type": "interrupt_ack", "run_id": run_id, "status": "requested"}
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """获取连接信息
        
        Args:
            connection_id: 连接ID
            
        Returns:
            Optional[Dict]: 连接信息
        """
        return self.connection_metadata.get(connection_id)
    
    def list_active_connections(self) -> List[Dict[str, Any]]:
        """列出所有活跃连接
        
        Returns:
            List[Dict]: 连接信息列表
        """
        return [
            {"connection_id": conn_id, **metadata}
            for conn_id, metadata in self.connection_metadata.items()
        ]
    
    def get_connections_by_user(self, user_id: str) -> List[str]:
        """获取指定用户的所有连接
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[str]: 连接ID列表
        """
        return [
            conn_id for conn_id, metadata in self.connection_metadata.items()
            if metadata.get("user_id") == user_id
        ]


class WebSocketConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.handler = WebSocketHandler()
        self.logger = logging.getLogger("websocket.manager")
    
    async def handle_websocket_connection(
        self,
        websocket: WebSocket,
        user_id: str,
        metadata: Dict[str, Any] = None
    ):
        """处理WebSocket连接
        
        Args:
            websocket: WebSocket连接
            user_id: 用户ID
            metadata: 连接元数据
        """
        connection_id = await self.handler.connect(websocket, user_id, metadata)
        
        try:
            while True:
                # 接收客户端消息
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    response = await self.handler.handle_client_message(
                        connection_id, message
                    )
                    
                    if response:
                        await self.handler.send_message(connection_id, response)
                
                except json.JSONDecodeError:
                    await self.handler.send_message(connection_id, {
                        "type": "error",
                        "error": "无效的JSON格式"
                    })
                
                except Exception as e:
                    self.logger.error(f"处理WebSocket消息错误: {e}")
                    await self.handler.send_message(connection_id, {
                        "type": "error",
                        "error": str(e)
                    })
        
        except WebSocketDisconnect:
            self.logger.info(f"WebSocket客户端断开连接: {connection_id}")
        
        except Exception as e:
            self.logger.error(f"WebSocket连接错误: {e}")
        
        finally:
            await self.handler.disconnect(connection_id)