#!/usr/bin/env python3
"""
LangGraph 流式处理综合演示

基于官方文档学习，演示LangGraph的各种流式处理功能：
1. 工作流进度流式处理 - 获取图节点执行后的状态更新
2. LLM令牌流式处理 - 流式传输语言模型生成的令牌
3. 自定义更新流式处理 - 发出用户定义的信号
4. 多种流式模式组合使用
5. 工具中的流式更新
6. 流式处理的错误处理和中断机制

参考文档：
- https://langchain-ai.github.io/langgraph/concepts/streaming/
- https://langchain-ai.github.io/langgraph/how-tos/streaming/
- https://langchain-ai.github.io/langgraph/cloud/how-tos/streaming/
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, AsyncGenerator, Optional
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.prebuilt import create_react_agent

# 导入项目中的流式处理组件
from core.streaming import (
    StreamManager, StreamConfig, StreamMode, StreamEventType,
    StreamChunk, StreamState
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("streaming_demo")


@dataclass
class DemoState:
    """演示状态类"""
    messages: List[Any]
    current_step: str = ""
    progress: float = 0.0
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


# 创建支持流式更新的工具
@tool
def weather_tool(city: str) -> str:
    """获取天气信息的工具，支持流式更新"""
    # 获取流式写入器
    writer = get_stream_writer()
    
    # 发送进度更新
    writer(f"🌍 正在查询 {city} 的天气信息...")
    
    # 模拟API调用延迟
    import time
    time.sleep(1)
    
    writer(f"📡 连接到天气服务...")
    time.sleep(0.5)
    
    writer(f"📊 解析天气数据...")
    time.sleep(0.5)
    
    # 返回结果
    result = f"今天{city}的天气是晴朗的，温度25°C"
    writer(f"✅ 天气查询完成")
    
    return result


@tool
def data_analysis_tool(data: Dict[str, Any]) -> str:
    """数据分析工具，支持流式进度更新"""
    writer = get_stream_writer()
    
    total_items = len(data)
    writer(f"📈 开始分析 {total_items} 项数据...")
    
    # 模拟数据处理过程
    for i, (key, value) in enumerate(data.items(), 1):
        progress = (i / total_items) * 100
        writer(f"🔍 处理项目 {i}/{total_items}: {key} ({progress:.1f}%)")
        import time
        time.sleep(0.3)
    
    writer(f"✅ 数据分析完成，发现 {total_items} 个关键指标")
    
    return f"分析完成：处理了{total_items}项数据，发现关键趋势和模式"


class StreamingWorkflowDemo:
    """流式工作流演示类"""
    
    def __init__(self):
        self.checkpointer = MemorySaver()
        self.stream_manager = StreamManager(self.checkpointer)
        
    def create_simple_workflow(self) -> StateGraph:
        """创建简单的工作流图"""
        
        def step1_node(state: DemoState) -> DemoState:
            """第一步：数据准备"""
            logger.info("执行步骤1：数据准备")
            state.current_step = "数据准备"
            state.progress = 0.25
            state.data = {"users": 100, "orders": 250, "revenue": 15000}
            state.messages = add_messages(
                state.messages,
                [AIMessage(content="步骤1完成：数据准备就绪")]
            )
            return state
        
        def step2_node(state: DemoState) -> DemoState:
            """第二步：数据处理"""
            logger.info("执行步骤2：数据处理")
            state.current_step = "数据处理"
            state.progress = 0.5
            # 调用数据分析工具
            result = data_analysis_tool.invoke(state.data)
            state.messages = add_messages(
                state.messages,
                [AIMessage(content=f"步骤2完成：{result}")]
            )
            return state
        
        def step3_node(state: DemoState) -> DemoState:
            """第三步：结果生成"""
            logger.info("执行步骤3：结果生成")
            state.current_step = "结果生成"
            state.progress = 1.0
            state.messages = add_messages(
                state.messages,
                [AIMessage(content="步骤3完成：报告已生成")]
            )
            return state
        
        # 构建图
        workflow = StateGraph(DemoState)
        workflow.add_node("step1", step1_node)
        workflow.add_node("step2", step2_node)
        workflow.add_node("step3", step3_node)
        
        workflow.add_edge(START, "step1")
        workflow.add_edge("step1", "step2")
        workflow.add_edge("step2", "step3")
        workflow.add_edge("step3", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def demo_basic_streaming(self):
        """演示基础流式处理"""
        logger.info("=== 基础流式处理演示 ===")
        
        graph = self.create_simple_workflow()
        
        # 初始状态
        initial_state = DemoState(
            messages=[HumanMessage(content="开始数据分析流程")]
        )
        
        config = {"configurable": {"thread_id": "demo_basic_streaming"}}
        
        print("\n🔄 流式模式: updates (状态更新)")
        print("-" * 50)
        
        # 使用 updates 模式流式处理
        async for chunk in graph.astream(initial_state, config, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                print(f"📍 节点: {node_name}")
                print(f"   当前步骤: {node_output.current_step}")
                print(f"   进度: {node_output.progress * 100:.1f}%")
                if node_output.messages:
                    latest_message = node_output.messages[-1]
                    print(f"   消息: {latest_message.content}")
                print()
    
    async def demo_values_streaming(self):
        """演示完整状态值流式处理"""
        logger.info("=== 完整状态值流式处理演示 ===")
        
        graph = self.create_simple_workflow()
        
        initial_state = DemoState(
            messages=[HumanMessage(content="开始完整状态流式处理")]
        )
        
        config = {"configurable": {"thread_id": "demo_values_streaming"}}
        
        print("\n🔄 流式模式: values (完整状态)")
        print("-" * 50)
        
        # 使用 values 模式流式处理
        async for state in graph.astream(initial_state, config, stream_mode="values"):
            print(f"📊 完整状态更新:")
            print(f"   当前步骤: {state.current_step}")
            print(f"   进度: {state.progress * 100:.1f}%")
            print(f"   数据项: {len(state.data) if state.data else 0}")
            print(f"   消息数: {len(state.messages)}")
            print()
    
    async def demo_custom_streaming(self):
        """演示自定义流式处理"""
        logger.info("=== 自定义流式处理演示 ===")
        
        # 创建包含工具的简单图
        def tool_node(state: DemoState) -> DemoState:
            """调用工具的节点"""
            # 调用天气工具
            weather_result = weather_tool.invoke({"city": "北京"})
            
            # 调用数据分析工具
            analysis_result = data_analysis_tool.invoke({
                "sales": 1000,
                "users": 500,
                "conversion": 0.05
            })
            
            state.messages = add_messages(
                state.messages,
                [
                    AIMessage(content=f"天气查询结果: {weather_result}"),
                    AIMessage(content=f"数据分析结果: {analysis_result}")
                ]
            )
            return state
        
        workflow = StateGraph(DemoState)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "tools")
        workflow.add_edge("tools", END)
        
        graph = workflow.compile(checkpointer=self.checkpointer)
        
        initial_state = DemoState(
            messages=[HumanMessage(content="执行工具调用")]
        )
        
        config = {"configurable": {"thread_id": "demo_custom_streaming"}}
        
        print("\n🔄 流式模式: custom (自定义更新)")
        print("-" * 50)
        
        # 使用 custom 模式流式处理
        async for chunk in graph.astream(initial_state, config, stream_mode="custom"):
            print(f"🔧 自定义更新: {chunk}")
    
    async def demo_multiple_stream_modes(self):
        """演示多种流式模式组合"""
        logger.info("=== 多种流式模式组合演示 ===")
        
        graph = self.create_simple_workflow()
        
        initial_state = DemoState(
            messages=[HumanMessage(content="多模式流式处理")]
        )
        
        config = {"configurable": {"thread_id": "demo_multiple_modes"}}
        
        print("\n🔄 流式模式: ['updates', 'custom']")
        print("-" * 50)
        
        # 使用多种流式模式
        async for stream_mode, chunk in graph.astream(
            initial_state, 
            config, 
            stream_mode=["updates", "custom"]
        ):
            print(f"📡 模式: {stream_mode}")
            if stream_mode == "updates":
                for node_name, node_output in chunk.items():
                    print(f"   节点更新: {node_name} -> {node_output.current_step}")
            elif stream_mode == "custom":
                print(f"   自定义数据: {chunk}")
            print()
    
    async def demo_stream_manager_integration(self):
        """演示与项目流式管理器的集成"""
        logger.info("=== 流式管理器集成演示 ===")
        
        graph = self.create_simple_workflow()
        
        initial_state = DemoState(
            messages=[HumanMessage(content="流式管理器集成测试")]
        )
        
        config = {"configurable": {"thread_id": "demo_stream_manager"}}
        
        # 创建流式配置
        stream_config = StreamConfig(
            modes=[StreamMode.UPDATES, StreamMode.VALUES],
            buffer_size=50,
            timeout=60
        )
        
        print("\n🔄 使用项目流式管理器")
        print("-" * 50)
        
        # 使用项目的流式管理器
        async for chunk in self.stream_manager.stream_graph_execution(
            graph, initial_state, config, stream_config
        ):
            print(f"🎯 {chunk.chunk_type.value}: {chunk.content}")
            if chunk.metadata:
                print(f"   元数据: {json.dumps(chunk.metadata, ensure_ascii=False, indent=2)}")
            print()
    
    async def demo_error_handling_streaming(self):
        """演示流式处理中的错误处理"""
        logger.info("=== 流式处理错误处理演示 ===")
        
        def error_node(state: DemoState) -> DemoState:
            """会产生错误的节点"""
            state.current_step = "错误处理测试"
            # 故意抛出异常
            raise ValueError("这是一个测试错误")
        
        def recovery_node(state: DemoState) -> DemoState:
            """恢复节点"""
            state.current_step = "错误恢复"
            state.messages = add_messages(
                state.messages,
                [AIMessage(content="已从错误中恢复")]
            )
            return state
        
        workflow = StateGraph(DemoState)
        workflow.add_node("error_step", error_node)
        workflow.add_node("recovery", recovery_node)
        
        workflow.add_edge(START, "error_step")
        workflow.add_edge("error_step", "recovery")
        workflow.add_edge("recovery", END)
        
        graph = workflow.compile(checkpointer=self.checkpointer)
        
        initial_state = DemoState(
            messages=[HumanMessage(content="错误处理测试")]
        )
        
        config = {"configurable": {"thread_id": "demo_error_handling"}}
        
        print("\n🔄 流式处理错误处理")
        print("-" * 50)
        
        try:
            async for chunk in graph.astream(initial_state, config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    print(f"📍 节点: {node_name}")
                    print(f"   步骤: {node_output.current_step}")
                    print()
        except Exception as e:
            print(f"❌ 捕获到错误: {e}")
            print("🔧 错误处理机制已激活")


async def main():
    """主演示函数"""
    print("🚀 LangGraph 流式处理综合演示")
    print("=" * 60)
    print("基于官方文档学习的流式处理功能演示")
    print("参考文档:")
    print("- https://langchain-ai.github.io/langgraph/concepts/streaming/")
    print("- https://langchain-ai.github.io/langgraph/how-tos/streaming/")
    print("- https://langchain-ai.github.io/langgraph/cloud/how-tos/streaming/")
    print("=" * 60)
    
    demo = StreamingWorkflowDemo()
    
    # 运行各种演示
    await demo.demo_basic_streaming()
    await asyncio.sleep(1)
    
    await demo.demo_values_streaming()
    await asyncio.sleep(1)
    
    await demo.demo_custom_streaming()
    await asyncio.sleep(1)
    
    await demo.demo_multiple_stream_modes()
    await asyncio.sleep(1)
    
    await demo.demo_stream_manager_integration()
    await asyncio.sleep(1)
    
    await demo.demo_error_handling_streaming()
    
    print("\n✅ 流式处理演示完成!")
    print("\n📚 学习总结:")
    print("1. LangGraph支持多种流式模式：values, updates, messages, custom, debug")
    print("2. 可以在工具中使用get_stream_writer()发送自定义更新")
    print("3. 支持多种流式模式同时使用")
    print("4. 流式处理具有完善的错误处理机制")
    print("5. 可以与项目自定义的流式管理器集成")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
    except Exception as e:
        logger.error(f"演示执行失败: {e}")
        import traceback
        traceback.print_exc()