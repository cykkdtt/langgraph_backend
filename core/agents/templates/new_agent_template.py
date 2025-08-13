"""
新智能体快速创建模板

使用此模板可以快速创建新的智能体类型。
只需要修改相应的部分即可创建自定义智能体。
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 选择基类：BaseAgent 或 MemoryEnhancedAgent
from ..base import BaseAgent, ChatRequest, ChatResponse
# from ..memory_enhanced import MemoryEnhancedAgent  # 如果需要记忆功能

logger = logging.getLogger(__name__)


# ============================================================================
# 第一步：定义智能体专用工具
# ============================================================================

@tool
def example_tool(input_text: str) -> str:
    """示例工具 - 请替换为你的实际工具"""
    return f"处理结果: {input_text}"

@tool
def another_example_tool(param1: str, param2: int = 10) -> str:
    """另一个示例工具"""
    return f"参数1: {param1}, 参数2: {param2}"


# ============================================================================
# 第二步：创建智能体类
# ============================================================================

class MyCustomAgent(BaseAgent):  # 或继承 MemoryEnhancedAgent
    """自定义智能体
    
    请修改以下内容：
    1. 类名和文档字符串
    2. 智能体的名称和描述
    3. 工具列表
    4. 处理逻辑
    """
    
    def __init__(self, agent_id: str, llm, **kwargs):
        super().__init__(
            agent_id=agent_id,
            name="我的自定义智能体",  # 修改名称
            description="这是一个自定义智能体的描述",  # 修改描述
            llm=llm,
            tools=[example_tool, another_example_tool],  # 修改工具列表
            **kwargs
        )
    
    def _build_graph(self) -> StateGraph:
        """构建智能体的处理图
        
        请根据你的业务逻辑修改节点和边的定义
        """
        from ..state import AgentState
        
        # 创建图
        graph = StateGraph(AgentState)
        
        # ========================================
        # 第三步：定义处理节点
        # ========================================
        
        # 添加你的处理节点
        graph.add_node("analyze_input", self._analyze_input_node)
        graph.add_node("process_request", self._process_request_node)
        graph.add_node("generate_response", self._generate_response_node)
        
        # 添加工具节点（如果有工具）
        if self.tools:
            tool_node = ToolNode(self.tools)
            graph.add_node("tools", tool_node)
        
        # ========================================
        # 第四步：定义图的流程
        # ========================================
        
        # 设置入口点
        graph.set_entry_point("analyze_input")
        
        # 添加边（定义节点之间的流转）
        graph.add_edge("analyze_input", "process_request")
        
        # 条件边示例
        graph.add_conditional_edges(
            "process_request",
            self._should_use_tools,  # 条件函数
            {
                "use_tools": "tools",
                "generate": "generate_response"
            }
        )
        
        # 工具执行后的流转
        graph.add_edge("tools", "generate_response")
        graph.add_edge("generate_response", END)
        
        return graph
    
    # ========================================
    # 第五步：实现处理节点的逻辑
    # ========================================
    
    async def _analyze_input_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """分析输入节点 - 请修改为你的逻辑"""
        messages = state.get("messages", [])
        
        if messages:
            last_message = messages[-1].content.lower()
            
            # 示例：简单的意图识别
            if "工具" in last_message or "tool" in last_message:
                state["needs_tools"] = True
            else:
                state["needs_tools"] = False
        
        return state
    
    async def _process_request_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求节点 - 请修改为你的逻辑"""
        messages = state.get("messages", [])
        
        # 示例：添加系统提示
        system_prompt = """
你是一个专业的助手。请根据用户的需求：
1. 理解用户的问题
2. 如果需要，使用可用的工具
3. 提供有帮助的回答

请修改这个系统提示以符合你的智能体角色。
"""
        
        # 调用LLM处理
        system_message = SystemMessage(content=system_prompt)
        response = await self.llm.ainvoke([system_message] + messages)
        
        state["messages"].append(response)
        
        return state
    
    async def _generate_response_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终响应节点"""
        # 这里可以添加响应后处理逻辑
        # 例如：格式化输出、添加元数据等
        
        return state
    
    def _should_use_tools(self, state: Dict[str, Any]) -> str:
        """判断是否需要使用工具的条件函数"""
        
        # 检查是否有工具调用
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "use_tools"
        
        # 检查状态标记
        if state.get("needs_tools", False):
            return "use_tools"
        
        return "generate"


# ============================================================================
# 第六步：注册智能体（可选）
# ============================================================================

def register_my_custom_agent():
    """注册自定义智能体到系统"""
    from ..registry import AgentRegistry, AgentConfig
    
    # 获取注册表
    registry = AgentRegistry()
    
    # 注册智能体类
    registry.register_agent_class("my_custom", MyCustomAgent)
    
    # 创建默认配置
    config = AgentConfig(
        agent_type="my_custom",
        name="我的自定义智能体",
        description="自定义智能体的描述",
        llm_config={
            "provider": "qwen",  # 或其他LLM提供商
            "model": "qwen-plus"
        },
        tools=["example_tool", "another_example_tool"],
        capabilities=["custom_capability"],  # 定义智能体能力
        memory_config={
            "enabled": False  # 如果继承MemoryEnhancedAgent则设为True
        }
    )
    
    return config


# ============================================================================
# 第七步：测试智能体
# ============================================================================

async def test_my_custom_agent():
    """测试自定义智能体"""
    
    # 注册智能体
    config = register_my_custom_agent()
    
    # 创建智能体实例
    from config.settings import get_llm_by_name
    from ..registry import AgentFactory
    
    llm = get_llm_by_name("qwen")
    factory = AgentFactory()
    
    agent = await factory.create_agent("custom_001", config)
    
    # 测试对话
    request = ChatRequest(
        messages=[HumanMessage(content="你好，请介绍一下你的功能")],
        user_id="test_user",
        session_id="test_session"
    )
    
    response = await agent.chat(request)
    print(f"智能体回复: {response.message.content}")
    
    return agent


# ============================================================================
# 使用说明
# ============================================================================

"""
使用此模板创建新智能体的步骤：

1. 复制此文件并重命名
2. 修改类名 MyCustomAgent 为你的智能体名称
3. 定义专用工具（@tool装饰器）
4. 修改智能体的名称、描述和工具列表
5. 根据业务逻辑修改 _build_graph 方法中的节点和边
6. 实现各个处理节点的逻辑
7. 可选：注册智能体到系统
8. 测试智能体功能

记忆功能：
- 如果需要记忆功能，继承 MemoryEnhancedAgent 而不是 BaseAgent
- 在 memory_config 中设置 enabled: True
- 可以使用 self.store_knowledge() 存储知识
- 智能体会自动检索相关记忆增强对话

工具集成：
- 使用 @tool 装饰器定义工具函数
- 在 tools 列表中添加工具
- 在图中添加 ToolNode 处理工具调用
- 使用条件边判断是否需要调用工具

部署：
- 将新智能体文件放在 core/agents/ 目录下
- 在 core/agents/__init__.py 中导入新智能体
- 在 registry.py 中注册智能体类型
- 在配置文件中添加智能体配置
"""

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_my_custom_agent())