"""
LangGraph官方流式处理适配器

将LangGraph官方流式输出适配到项目的流式处理系统中。
"""

import asyncio
from typing import AsyncGenerator, Dict, Any, List, Union
from langgraph.graph import StateGraph
from langgraph.config import get_stream_writer

from .stream_types import StreamChunk, StreamEventType, StreamConfig, StreamMode
from .stream_manager import StreamManager


class LangGraphStreamAdapter:
    """LangGraph流式输出适配器"""
    
    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
    
    def map_langgraph_mode_to_internal(self, langgraph_mode: str) -> StreamMode:
        """映射LangGraph流式模式到内部模式"""
        mapping = {
            "values": StreamMode.VALUES,
            "updates": StreamMode.UPDATES, 
            "messages": StreamMode.MESSAGES,
            "custom": StreamMode.EVENTS,  # 自定义事件映射到事件流
            "debug": StreamMode.DEBUG
        }
        return mapping.get(langgraph_mode, StreamMode.VALUES)
    
    def map_internal_mode_to_langgraph(self, internal_mode: StreamMode) -> str:
        """映射内部模式到LangGraph流式模式"""
        mapping = {
            StreamMode.VALUES: "values",
            StreamMode.UPDATES: "updates",
            StreamMode.MESSAGES: "messages", 
            StreamMode.EVENTS: "custom",
            StreamMode.DEBUG: "debug"
        }
        return mapping.get(internal_mode, "values")
    
    async def adapt_langgraph_stream(
        self,
        graph: StateGraph,
        input_data: Dict[str, Any],
        config: Dict[str, Any],
        stream_config: StreamConfig
    ) -> AsyncGenerator[StreamChunk, None]:
        """适配LangGraph流式输出到内部格式
        
        Args:
            graph: LangGraph图实例
            input_data: 输入数据
            config: 执行配置
            stream_config: 流式配置
            
        Yields:
            StreamChunk: 适配后的流式块
        """
        # 转换流式模式
        langgraph_modes = [
            self.map_internal_mode_to_langgraph(mode) 
            for mode in stream_config.modes
        ]
        
        run_id = config.get("configurable", {}).get("thread_id", "unknown")
        
        try:
            # 处理单模式
            if len(langgraph_modes) == 1:
                async for chunk in graph.astream(
                    input_data, 
                    config=config, 
                    stream_mode=langgraph_modes[0]
                ):
                    adapted_chunk = self._adapt_chunk(
                        chunk, langgraph_modes[0], run_id
                    )
                    if adapted_chunk:
                        yield adapted_chunk
            
            # 处理多模式
            else:
                async for stream_mode, chunk in graph.astream(
                    input_data,
                    config=config,
                    stream_mode=langgraph_modes
                ):
                    adapted_chunk = self._adapt_chunk(chunk, stream_mode, run_id)
                    if adapted_chunk:
                        yield adapted_chunk
        
        except Exception as e:
            # 发送错误事件
            yield StreamChunk(
                chunk_type=StreamEventType.ERROR,
                content=str(e),
                metadata={"error_type": type(e).__name__},
                run_id=run_id
            )
    
    def _adapt_chunk(
        self, 
        chunk: Any, 
        stream_mode: str, 
        run_id: str
    ) -> StreamChunk:
        """适配单个流式块
        
        Args:
            chunk: LangGraph原始块
            stream_mode: 流式模式
            run_id: 运行ID
            
        Returns:
            StreamChunk: 适配后的流式块
        """
        if stream_mode == "updates":
            return self._adapt_updates_chunk(chunk, run_id)
        elif stream_mode == "values":
            return self._adapt_values_chunk(chunk, run_id)
        elif stream_mode == "messages":
            return self._adapt_messages_chunk(chunk, run_id)
        elif stream_mode == "custom":
            return self._adapt_custom_chunk(chunk, run_id)
        elif stream_mode == "debug":
            return self._adapt_debug_chunk(chunk, run_id)
        
        return None
    
    def _adapt_updates_chunk(self, chunk: Dict[str, Any], run_id: str) -> StreamChunk:
        """适配updates模式的块"""
        if isinstance(chunk, dict):
            for node_name, node_output in chunk.items():
                return StreamChunk(
                    chunk_type=StreamEventType.STATE_UPDATE,
                    content=str(node_output),
                    metadata={
                        "node": node_name,
                        "update_type": "node_output"
                    },
                    node_id=node_name,
                    run_id=run_id
                )
        
        return StreamChunk(
            chunk_type=StreamEventType.STATE_UPDATE,
            content=str(chunk),
            metadata={"update_type": "general"},
            run_id=run_id
        )
    
    def _adapt_values_chunk(self, chunk: Dict[str, Any], run_id: str) -> StreamChunk:
        """适配values模式的块"""
        return StreamChunk(
            chunk_type=StreamEventType.STATE_UPDATE,
            content="",
            metadata={
                "state_values": chunk,
                "update_type": "full_state"
            },
            run_id=run_id
        )
    
    def _adapt_messages_chunk(self, chunk: tuple, run_id: str) -> StreamChunk:
        """适配messages模式的块"""
        if isinstance(chunk, tuple) and len(chunk) == 2:
            token, metadata = chunk
            return StreamChunk(
                chunk_type=StreamEventType.MESSAGE_CHUNK,
                content=str(token),
                metadata=metadata or {},
                run_id=run_id
            )
        
        return StreamChunk(
            chunk_type=StreamEventType.MESSAGE_CHUNK,
            content=str(chunk),
            metadata={},
            run_id=run_id
        )
    
    def _adapt_custom_chunk(self, chunk: Any, run_id: str) -> StreamChunk:
        """适配custom模式的块"""
        return StreamChunk(
            chunk_type=StreamEventType.CUSTOM_EVENT,
            content=str(chunk),
            metadata={"custom_data": chunk},
            run_id=run_id
        )
    
    def _adapt_debug_chunk(self, chunk: Any, run_id: str) -> StreamChunk:
        """适配debug模式的块"""
        return StreamChunk(
            chunk_type=StreamEventType.DEBUG,
            content=str(chunk),
            metadata={"debug_info": chunk},
            run_id=run_id
        )


class StreamWriterIntegration:
    """流式写入器集成"""
    
    @staticmethod
    def create_stream_writer_tool(tool_func):
        """为工具函数添加流式写入器支持
        
        Args:
            tool_func: 原始工具函数
            
        Returns:
            包装后的工具函数
        """
        def wrapper(*args, **kwargs):
            try:
                # 尝试获取流式写入器
                writer = get_stream_writer()
                
                # 将写入器添加到工具上下文
                if hasattr(tool_func, '__globals__'):
                    tool_func.__globals__['stream_writer'] = writer
                
                return tool_func(*args, **kwargs)
            
            except Exception:
                # 如果无法获取写入器，正常执行工具
                return tool_func(*args, **kwargs)
        
        return wrapper
    
    @staticmethod
    def send_progress_update(message: str):
        """发送进度更新
        
        Args:
            message: 进度消息
        """
        try:
            writer = get_stream_writer()
            writer(message)
        except Exception:
            # 如果无法获取写入器，忽略更新
            pass