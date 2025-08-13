"""
SSE (Server-Sent Events) 处理器

提供基于SSE的流式响应处理功能。
"""

import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from fastapi.responses import StreamingResponse
from fastapi import Request
import logging

from .stream_types import StreamChunk, StreamConfig, StreamEventType


class SSEHandler:
    """SSE处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger("sse.handler")
    
    async def create_sse_response(
        self,
        stream_generator: AsyncGenerator[StreamChunk, None],
        request: Request,
        config: StreamConfig = None
    ) -> StreamingResponse:
        """创建SSE响应
        
        Args:
            stream_generator: 流式生成器
            request: FastAPI请求对象
            config: 流式配置
            
        Returns:
            StreamingResponse: SSE响应
        """
        if config is None:
            config = StreamConfig()
        
        async def event_stream():
            """SSE事件流生成器"""
            try:
                # 发送连接建立事件
                yield self._format_sse_event(
                    event_type="connection",
                    data={"status": "connected", "timestamp": "now"}
                )
                
                # 心跳任务
                heartbeat_task = None
                if config.enable_heartbeat:
                    heartbeat_task = asyncio.create_task(
                        self._heartbeat_generator(config.heartbeat_interval)
                    )
                
                try:
                    async for chunk in stream_generator:
                        # 检查客户端是否断开连接
                        if await request.is_disconnected():
                            self.logger.info("客户端断开连接，停止SSE流")
                            break
                        
                        # 过滤事件
                        if self._should_include_event(chunk, config):
                            sse_data = self._chunk_to_sse_data(chunk)
                            yield self._format_sse_event(
                                event_type=chunk.chunk_type.value,
                                data=sse_data,
                                event_id=chunk.run_id
                            )
                        
                        # 处理完成事件
                        if chunk.chunk_type == StreamEventType.COMPLETE:
                            break
                
                finally:
                    # 清理心跳任务
                    if heartbeat_task:
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except asyncio.CancelledError:
                            pass
                
                # 发送连接关闭事件
                yield self._format_sse_event(
                    event_type="connection",
                    data={"status": "closed", "timestamp": "now"}
                )
                
            except Exception as e:
                self.logger.error(f"SSE流处理错误: {e}")
                yield self._format_sse_event(
                    event_type="error",
                    data={"error": str(e), "error_type": type(e).__name__}
                )
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    def _format_sse_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        event_id: Optional[str] = None
    ) -> str:
        """格式化SSE事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
            event_id: 事件ID
            
        Returns:
            str: 格式化的SSE事件
        """
        lines = []
        
        if event_id:
            lines.append(f"id: {event_id}")
        
        lines.append(f"event: {event_type}")
        
        # 处理数据
        if isinstance(data, dict):
            data_str = json.dumps(data, ensure_ascii=False)
        else:
            data_str = str(data)
        
        # 处理多行数据
        for line in data_str.split('\n'):
            lines.append(f"data: {line}")
        
        lines.append("")  # 空行表示事件结束
        
        return "\n".join(lines) + "\n"
    
    def _chunk_to_sse_data(self, chunk: StreamChunk) -> Dict[str, Any]:
        """将流式块转换为SSE数据
        
        Args:
            chunk: 流式块
            
        Returns:
            Dict: SSE数据
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
    
    async def _heartbeat_generator(self, interval: int):
        """心跳生成器
        
        Args:
            interval: 心跳间隔（秒）
        """
        try:
            while True:
                await asyncio.sleep(interval)
                yield self._format_sse_event(
                    event_type="heartbeat",
                    data={"timestamp": "now"}
                )
        except asyncio.CancelledError:
            pass


class SSEEventFormatter:
    """SSE事件格式化器"""
    
    @staticmethod
    def format_message_chunk(chunk: StreamChunk) -> Dict[str, Any]:
        """格式化消息块
        
        Args:
            chunk: 流式块
            
        Returns:
            Dict: 格式化的数据
        """
        return {
            "type": "message",
            "content": chunk.content,
            "node": chunk.metadata.get("node"),
            "timestamp": chunk.timestamp.isoformat() if chunk.timestamp else None
        }
    
    @staticmethod
    def format_state_update(chunk: StreamChunk) -> Dict[str, Any]:
        """格式化状态更新
        
        Args:
            chunk: 流式块
            
        Returns:
            Dict: 格式化的数据
        """
        return {
            "type": "state_update",
            "node": chunk.metadata.get("node"),
            "state": chunk.metadata.get("state"),
            "timestamp": chunk.timestamp.isoformat() if chunk.timestamp else None
        }
    
    @staticmethod
    def format_error(chunk: StreamChunk) -> Dict[str, Any]:
        """格式化错误
        
        Args:
            chunk: 流式块
            
        Returns:
            Dict: 格式化的数据
        """
        return {
            "type": "error",
            "error": chunk.content,
            "error_type": chunk.metadata.get("error_type"),
            "node": chunk.node_id,
            "timestamp": chunk.timestamp.isoformat() if chunk.timestamp else None
        }