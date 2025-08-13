#!/usr/bin/env python3
"""
LangGraph 流式处理实战演示

基于官方文档学习的流式处理功能实现，展示：
1. 多种流式模式的使用
2. 工具中的自定义流式更新
3. LLM令牌流式处理
4. 错误处理和恢复
5. 与项目流式管理器的集成
"""

import asyncio
import json
import time
from typing import Dict, Any, List, AsyncGenerator, Optional
from dataclasses import dataclass
from enum import Enum

from langgraph import StateGraph, END
from langgraph.graph import Graph
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import MemorySaver

# 导入项目的流式管理器
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.streaming.stream_manager import StreamManager, StreamConfig, StreamMode
from core.streaming.stream_types import StreamEventType


class DemoState(Dict[str, Any]):
    """演示状态类"""
    messages: List[Dict[str, str]]
    progress: float
    current_task: str
    results: List[str]
    error_count: int


class StreamingDemoApp:
    """LangGraph流式处理演示应用"""
    
    def __init__(self):
        self.checkpointer = MemorySaver()
        self.stream_manager = StreamManager(self.checkpointer)
        
    def create_demo_graph(self) -> StateGraph:
        """创建演示图"""
        
        def start_node(state: DemoState) -> DemoState:
            """开始节点"""
            writer = get_stream_writer()
            writer("🚀 开始执行工作流...")
            
            return {
                **state,
                "current_task": "初始化",
                "progress": 0.1,
                "messages": state.get("messages", []) + [
                    {"role": "system", "content": "工作流已启动"}
                ]
            }
        
        def data_processing_node(state: DemoState) -> DemoState:
            """数据处理节点 - 展示工具中的流式更新"""
            writer = get_stream_writer()
            
            # 模拟数据处理过程
            data_items = ["用户数据", "产品信息", "订单记录", "分析报告", "统计数据"]
            results = []
            
            writer("📊 开始数据处理...")
            
            for i, item in enumerate(data_items):
                # 模拟处理时间
                time.sleep(0.5)
                
                # 发送进度更新
                progress = (i + 1) / len(data_items)
                writer(f"处理 {item}... ({progress:.1%})")
                
                results.append(f"已处理: {item}")
            
            writer("✅ 数据处理完成!")
            
            return {
                **state,
                "current_task": "数据处理",
                "progress": 0.5,
                "results": results
            }
        
        def llm_simulation_node(state: DemoState) -> DemoState:
            """LLM模拟节点 - 展示令牌流式处理"""
            writer = get_stream_writer()
            
            # 模拟LLM生成过程
            response_text = "基于处理的数据，我们可以得出以下结论：数据质量良好，用户活跃度较高，产品销售趋势积极。"
            
            writer("🤖 LLM开始生成响应...")
            
            # 模拟令牌流式生成
            for i, char in enumerate(response_text):
                if i % 5 == 0:  # 每5个字符发送一次更新
                    writer(f"生成进度: {response_text[:i+5]}")
                time.sleep(0.1)
            
            writer("🎯 LLM响应生成完成!")
            
            return {
                **state,
                "current_task": "LLM生成",
                "progress": 0.8,
                "messages": state.get("messages", []) + [
                    {"role": "assistant", "content": response_text}
                ]
            }
        
        def finalization_node(state: DemoState) -> DemoState:
            """完成节点"""
            writer = get_stream_writer()
            writer("🎉 工作流执行完成!")
            
            return {
                **state,
                "current_task": "完成",
                "progress": 1.0
            }
        
        # 构建图
        graph = StateGraph(DemoState)
        
        # 添加节点
        graph.add_node("start", start_node)
        graph.add_node("data_processing", data_processing_node)
        graph.add_node("llm_simulation", llm_simulation_node)
        graph.add_node("finalization", finalization_node)
        
        # 添加边
        graph.add_edge("start", "data_processing")
        graph.add_edge("data_processing", "llm_simulation")
        graph.add_edge("llm_simulation", "finalization")
        graph.add_edge("finalization", END)
        
        # 设置入口点
        graph.set_entry_point("start")
        
        return graph.compile(checkpointer=self.checkpointer)
    
    async def demo_basic_streaming(self):
        """演示基础流式处理"""
        print("\n" + "="*60)
        print("🔄 基础流式处理演示 (updates模式)")
        print("="*60)
        
        graph = self.create_demo_graph()
        
        input_data = {
            "messages": [{"role": "user", "content": "请处理数据并生成报告"}],
            "progress": 0.0,
            "current_task": "",
            "results": [],
            "error_count": 0
        }
        
        config = {"configurable": {"thread_id": "demo_basic"}}
        
        async for chunk in graph.astream(input_data, config, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                print(f"📍 节点: {node_name}")
                print(f"   任务: {node_output.get('current_task', 'N/A')}")
                print(f"   进度: {node_output.get('progress', 0):.1%}")
                print()
    
    async def demo_custom_streaming(self):
        """演示自定义流式处理"""
        print("\n" + "="*60)
        print("🎨 自定义流式处理演示 (custom模式)")
        print("="*60)
        
        graph = self.create_demo_graph()
        
        input_data = {
            "messages": [{"role": "user", "content": "请处理数据并生成报告"}],
            "progress": 0.0,
            "current_task": "",
            "results": [],
            "error_count": 0
        }
        
        config = {"configurable": {"thread_id": "demo_custom"}}
        
        async for chunk in graph.astream(input_data, config, stream_mode="custom"):
            print(f"💬 自定义更新: {chunk}")
    
    async def demo_multi_mode_streaming(self):
        """演示多模式流式处理"""
        print("\n" + "="*60)
        print("🔀 多模式流式处理演示 (updates + custom)")
        print("="*60)
        
        graph = self.create_demo_graph()
        
        input_data = {
            "messages": [{"role": "user", "content": "请处理数据并生成报告"}],
            "progress": 0.0,
            "current_task": "",
            "results": [],
            "error_count": 0
        }
        
        config = {"configurable": {"thread_id": "demo_multi"}}
        
        async for stream_mode, chunk in graph.astream(
            input_data, config, stream_mode=["updates", "custom"]
        ):
            if stream_mode == "updates":
                for node_name, node_output in chunk.items():
                    print(f"📊 [更新] 节点: {node_name}, 进度: {node_output.get('progress', 0):.1%}")
            elif stream_mode == "custom":
                print(f"💬 [自定义] {chunk}")
    
    async def demo_stream_manager_integration(self):
        """演示与项目流式管理器的集成"""
        print("\n" + "="*60)
        print("🔧 流式管理器集成演示")
        print("="*60)
        
        graph = self.create_demo_graph()
        
        input_data = {
            "messages": [{"role": "user", "content": "请处理数据并生成报告"}],
            "progress": 0.0,
            "current_task": "",
            "results": [],
            "error_count": 0
        }
        
        config = {"configurable": {"thread_id": "demo_manager"}}
        
        # 配置流式管理器
        stream_config = StreamConfig(
            modes=[StreamMode.UPDATES, StreamMode.CUSTOM],
            buffer_size=100,
            timeout=30
        )
        
        # 注册事件处理器
        def on_progress_update(event):
            if event.event_type == StreamEventType.PROGRESS_UPDATE:
                print(f"🎯 [管理器] 进度更新: {event.data}")
        
        def on_custom_event(event):
            if event.event_type == StreamEventType.CUSTOM_EVENT:
                print(f"🎨 [管理器] 自定义事件: {event.data}")
        
        self.stream_manager.register_event_handler(StreamEventType.PROGRESS_UPDATE, on_progress_update)
        self.stream_manager.register_event_handler(StreamEventType.CUSTOM_EVENT, on_custom_event)
        
        try:
            async for chunk in self.stream_manager.stream_graph_execution(
                graph, input_data, config, stream_config
            ):
                print(f"📦 [管理器] 流式块: {chunk.chunk_type.value} - {chunk.content}")
        except Exception as e:
            print(f"❌ 流式处理错误: {e}")
    
    async def demo_error_handling(self):
        """演示错误处理"""
        print("\n" + "="*60)
        print("⚠️  错误处理演示")
        print("="*60)
        
        def error_node(state: DemoState) -> DemoState:
            """故意产生错误的节点"""
            writer = get_stream_writer()
            writer("⚠️ 即将触发错误...")
            
            # 模拟错误
            if state.get("error_count", 0) < 1:
                writer("❌ 发生错误，正在重试...")
                raise ValueError("模拟的处理错误")
            
            writer("✅ 错误已恢复!")
            return {**state, "current_task": "错误恢复"}
        
        # 创建包含错误处理的图
        graph = StateGraph(DemoState)
        graph.add_node("error_node", error_node)
        graph.add_edge("error_node", END)
        graph.set_entry_point("error_node")
        
        compiled_graph = graph.compile(checkpointer=self.checkpointer)
        
        input_data = {
            "messages": [],
            "progress": 0.0,
            "current_task": "",
            "results": [],
            "error_count": 0
        }
        
        config = {"configurable": {"thread_id": "demo_error"}}
        
        try:
            async for chunk in compiled_graph.astream(input_data, config, stream_mode="custom"):
                print(f"💬 错误处理更新: {chunk}")
        except Exception as e:
            print(f"❌ 捕获到错误: {e}")
            print("🔄 实现错误恢复逻辑...")
            
            # 更新错误计数并重试
            input_data["error_count"] = 1
            config["configurable"]["thread_id"] = "demo_error_retry"
            
            print("🔄 重试执行...")
            async for chunk in compiled_graph.astream(input_data, config, stream_mode="custom"):
                print(f"💬 重试更新: {chunk}")
    
    async def run_all_demos(self):
        """运行所有演示"""
        print("🎬 LangGraph 流式处理实战演示")
        print("基于官方文档学习的流式处理功能")
        
        # 运行各种演示
        await self.demo_basic_streaming()
        await self.demo_custom_streaming()
        await self.demo_multi_mode_streaming()
        await self.demo_stream_manager_integration()
        await self.demo_error_handling()
        
        print("\n" + "="*60)
        print("🎉 所有演示完成!")
        print("="*60)
        print("\n📚 学习总结:")
        print("1. ✅ 掌握了多种流式模式的使用")
        print("2. ✅ 学会了在工具中发送自定义更新")
        print("3. ✅ 了解了LLM令牌流式处理")
        print("4. ✅ 实现了错误处理和恢复机制")
        print("5. ✅ 集成了项目的流式管理器")
        print("\n🚀 现在可以在实际项目中应用这些流式处理技术!")


async def main():
    """主函数"""
    demo_app = StreamingDemoApp()
    await demo_app.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())