"""
流式处理管理器

提供统一的流式处理管理功能，支持多种流式模式和事件处理。
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
import logging

from langgraph.graph import StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from .stream_types import (
    StreamChunk, StreamEvent, StreamState, StreamMode, 
    StreamEventType, StreamConfig, InterruptRequest, InterruptResponse
)


class StreamManager:
    """流式处理管理器"""
    
    def __init__(self, checkpoint_saver: Optional[BaseCheckpointSaver] = None):
        self.checkpoint_saver = checkpoint_saver
        self.active_streams: Dict[str, StreamState] = {}
        self.interrupt_requests: Dict[str, InterruptRequest] = {}
        self.event_handlers: Dict[StreamEventType, List[callable]] = {}
        self.logger = logging.getLogger("stream.manager")
        
        # 初始化事件处理器
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """设置默认事件处理器"""
        self.register_event_handler(
            StreamEventType.NODE_ERROR, 
            self._handle_node_error
        )
        self.register_event_handler(
            StreamEventType.INTERRUPT, 
            self._handle_interrupt
        )
    
    def register_event_handler(
        self, 
        event_type: StreamEventType, 
        handler: callable
    ):
        """注册事件处理器
        
        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def stream_graph_execution(
        self,
        graph: StateGraph,
        input_data: Dict[str, Any],
        config: Dict[str, Any],
        stream_config: StreamConfig = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """流式执行图
        
        Args:
            graph: LangGraph图实例
            input_data: 输入数据
            config: 执行配置
            stream_config: 流式配置
            
        Yields:
            StreamChunk: 流式响应块
        """
        if stream_config is None:
            stream_config = StreamConfig()
        
        run_id = str(uuid.uuid4())
        
        # 创建流式状态
        stream_state = StreamState(
            run_id=run_id,
            status="running"
        )
        self.active_streams[run_id] = stream_state
        
        try:
            # 发送开始事件
            yield StreamChunk(
                chunk_type=StreamEventType.NODE_START,
                content="",
                metadata={"run_id": run_id, "status": "started"},
                run_id=run_id
            )
            
            # 流式执行图
            async for chunk in graph.astream(
                input_data,
                config=config,
                stream_mode=stream_config.modes
            ):
                # 处理不同类型的流式输出
                processed_chunks = await self._process_stream_chunk(
                    chunk, run_id, stream_config
                )
                
                for processed_chunk in processed_chunks:
                    # 触发事件处理器
                    await self._trigger_event_handlers(processed_chunk)
                    yield processed_chunk
                
                # 检查中断
                if await self._check_interrupts(run_id):
                    yield StreamChunk(
                        chunk_type=StreamEventType.INTERRUPT,
                        content="执行被中断",
                        metadata={"run_id": run_id},
                        run_id=run_id
                    )
                    break
            
            # 更新状态为完成
            stream_state.status = "completed"
            stream_state.updated_at = datetime.utcnow()
            
            # 发送完成事件
            yield StreamChunk(
                chunk_type=StreamEventType.COMPLETE,
                content="",
                metadata={"run_id": run_id, "status": "completed"},
                run_id=run_id
            )
            
        except Exception as e:
            # 更新状态为错误
            stream_state.status = "error"
            stream_state.error = str(e)
            stream_state.updated_at = datetime.utcnow()
            
            self.logger.error(f"流式执行错误 {run_id}: {e}")
            
            # 发送错误事件
            yield StreamChunk(
                chunk_type=StreamEventType.ERROR,
                content=str(e),
                metadata={"run_id": run_id, "error_type": type(e).__name__},
                run_id=run_id
            )
        
        finally:
            # 清理资源
            if run_id in self.active_streams:
                del self.active_streams[run_id]
    
    async def _process_stream_chunk(
        self,
        chunk: Any,
        run_id: str,
        stream_config: StreamConfig
    ) -> List[StreamChunk]:
        """处理流式块
        
        Args:
            chunk: 原始流式块
            run_id: 运行ID
            stream_config: 流式配置
            
        Returns:
            List[StreamChunk]: 处理后的流式块列表
        """
        processed_chunks = []
        
        if isinstance(chunk, dict):
            # 处理字典类型的块
            for node_name, node_output in chunk.items():
                if isinstance(node_output, dict):
                    # 处理消息
                    if "messages" in node_output:
                        for message in node_output["messages"]:
                            processed_chunks.append(StreamChunk(
                                chunk_type=StreamEventType.MESSAGE_CHUNK,
                                content=str(message.content) if hasattr(message, 'content') else str(message),
                                metadata={
                                    "node": node_name,
                                    "message_type": type(message).__name__
                                },
                                node_id=node_name,
                                run_id=run_id
                            ))
                    
                    # 处理状态更新
                    if "state" in node_output:
                        processed_chunks.append(StreamChunk(
                            chunk_type=StreamEventType.STATE_UPDATE,
                            content="",
                            metadata={
                                "node": node_name,
                                "state": node_output["state"]
                            },
                            node_id=node_name,
                            run_id=run_id
                        ))
                else:
                    # 处理其他类型的输出
                    processed_chunks.append(StreamChunk(
                        chunk_type=StreamEventType.NODE_END,
                        content=str(node_output),
                        metadata={"node": node_name},
                        node_id=node_name,
                        run_id=run_id
                    ))
        else:
            # 处理非字典类型的块
            processed_chunks.append(StreamChunk(
                chunk_type=StreamEventType.MESSAGE_CHUNK,
                content=str(chunk),
                metadata={},
                run_id=run_id
            ))
        
        return processed_chunks
    
    async def _trigger_event_handlers(self, chunk: StreamChunk):
        """触发事件处理器
        
        Args:
            chunk: 流式块
        """
        handlers = self.event_handlers.get(chunk.chunk_type, [])
        for handler in handlers:
            try:
                await handler(chunk)
            except Exception as e:
                self.logger.error(f"事件处理器错误: {e}")
    
    async def _check_interrupts(self, run_id: str) -> bool:
        """检查中断
        
        Args:
            run_id: 运行ID
            
        Returns:
            bool: 是否有中断
        """
        # 检查是否有待处理的中断请求
        for interrupt_id, interrupt_req in self.interrupt_requests.items():
            if interrupt_req.run_id == run_id:
                return True
        return False
    
    async def _handle_node_error(self, chunk: StreamChunk):
        """处理节点错误
        
        Args:
            chunk: 错误块
        """
        self.logger.error(f"节点执行错误: {chunk.content}")
        
        # 更新流式状态
        if chunk.run_id in self.active_streams:
            stream_state = self.active_streams[chunk.run_id]
            stream_state.status = "error"
            stream_state.error = chunk.content
            stream_state.updated_at = datetime.utcnow()
    
    async def _handle_interrupt(self, chunk: StreamChunk):
        """处理中断事件
        
        Args:
            chunk: 中断块
        """
        self.logger.info(f"处理中断事件: {chunk.run_id}")
        
        # 更新流式状态
        if chunk.run_id in self.active_streams:
            stream_state = self.active_streams[chunk.run_id]
            stream_state.status = "interrupted"
            stream_state.updated_at = datetime.utcnow()
    
    async def request_interrupt(
        self,
        run_id: str,
        node_id: str,
        message: str,
        context: Dict[str, Any] = None
    ) -> str:
        """请求中断
        
        Args:
            run_id: 运行ID
            node_id: 节点ID
            message: 中断消息
            context: 上下文信息
            
        Returns:
            str: 中断请求ID
        """
        interrupt_id = str(uuid.uuid4())
        interrupt_request = InterruptRequest(
            run_id=run_id,
            node_id=node_id,
            message=message,
            context=context or {}
        )
        
        self.interrupt_requests[interrupt_id] = interrupt_request
        self.logger.info(f"创建中断请求: {interrupt_id}")
        
        return interrupt_id
    
    async def respond_to_interrupt(
        self,
        interrupt_id: str,
        response_data: Dict[str, Any],
        approved: bool = True
    ) -> bool:
        """响应中断
        
        Args:
            interrupt_id: 中断请求ID
            response_data: 响应数据
            approved: 是否批准
            
        Returns:
            bool: 是否成功响应
        """
        if interrupt_id not in self.interrupt_requests:
            return False
        
        interrupt_request = self.interrupt_requests[interrupt_id]
        
        # 创建响应
        response = InterruptResponse(
            request_id=interrupt_id,
            response_data=response_data,
            approved=approved
        )
        
        # 清理中断请求
        del self.interrupt_requests[interrupt_id]
        
        self.logger.info(f"响应中断请求: {interrupt_id}, 批准: {approved}")
        
        return True
    
    def get_stream_state(self, run_id: str) -> Optional[StreamState]:
        """获取流式状态
        
        Args:
            run_id: 运行ID
            
        Returns:
            Optional[StreamState]: 流式状态
        """
        return self.active_streams.get(run_id)
    
    def list_active_streams(self) -> List[StreamState]:
        """列出活跃的流式执行
        
        Returns:
            List[StreamState]: 活跃流式状态列表
        """
        return list(self.active_streams.values())
    
    def list_pending_interrupts(self) -> List[InterruptRequest]:
        """列出待处理的中断请求
        
        Returns:
            List[InterruptRequest]: 中断请求列表
        """
        return list(self.interrupt_requests.values())