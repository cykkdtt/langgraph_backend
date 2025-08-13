#!/usr/bin/env python3
"""
创建自定义智能体演示

本示例展示如何基于BaseAgent创建新的自定义智能体。
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from core.agents.base import BaseAgent, AgentType, AgentStatus
from core.agents.registry import AgentConfig, get_agent_registry, get_agent_factory
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("custom_agent_demo")


# 1. 创建自定义工具
@tool
def custom_analysis_tool(data: str) -> str:
    """自定义分析工具"""
    return f"分析结果: {data} 已被处理"


@tool  
def custom_report_tool(content: str) -> str:
    """自定义报告工具"""
    return f"报告生成: {content}"


class CustomAnalysisAgent(BaseAgent):
    """自定义分析智能体
    
    这是一个专门用于数据分析的智能体示例。
    """
    
    def __init__(self, agent_id: str, **kwargs):
        # 初始化LLM
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
        
        # 定义工具
        tools = [custom_analysis_tool, custom_report_tool]
        
        super().__init__(
            agent_id=agent_id,
            name="Custom Analysis Agent",
            description="专门用于数据分析的自定义智能体",
            llm=llm,
            tools=tools,
            agent_type=AgentType.DATA_ANALYSIS,
            **kwargs
        )
    
    async def _build_graph(self) -> None:
        """构建智能体的LangGraph图"""
        # 创建状态图
        self.graph = StateGraph(self.AgentState)
        
        # 添加节点
        self.graph.add_node("analyze", self._analyze_node)
        self.graph.add_node("report", self._report_node)
        self.graph.add_node("chat", self._chat_node)
        
        # 设置路由
        self.graph.add_edge(START, "chat")
        self.graph.add_conditional_edges(
            "chat",
            self._route_decision,
            {
                "analyze": "analyze",
                "report": "report", 
                "end": END
            }
        )
        self.graph.add_edge("analyze", "report")
        self.graph.add_edge("report", END)
    
    def _route_decision(self, state: Dict[str, Any]) -> str:
        """路由决策函数"""
        messages = state.get("messages", [])
        if not messages:
            return "end"
            
        last_message = messages[-1]
        content = last_message.content.lower()
        
        if "分析" in content or "analysis" in content:
            return "analyze"
        elif "报告" in content or "report" in content:
            return "report"
        else:
            return "end"
    
    async def _analyze_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """分析节点"""
        messages = state["messages"]
        
        # 调用分析工具
        analysis_result = custom_analysis_tool.invoke({"data": "用户数据"})
        
        # 生成分析响应
        response = AIMessage(content=f"数据分析完成: {analysis_result}")
        
        return {"messages": [response]}
    
    async def _report_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """报告节点"""
        messages = state["messages"]
        
        # 调用报告工具
        report_result = custom_report_tool.invoke({"content": "分析结果摘要"})
        
        # 生成报告响应
        response = AIMessage(content=f"报告生成完成: {report_result}")
        
        return {"messages": [response]}


class CustomChatbotAgent(BaseAgent):
    """自定义聊天机器人智能体
    
    这是一个简单的聊天机器人示例。
    """
    
    def __init__(self, agent_id: str, **kwargs):
        # 初始化LLM
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
        
        super().__init__(
            agent_id=agent_id,
            name="Custom Chatbot Agent",
            description="友好的聊天机器人智能体",
            llm=llm,
            agent_type=AgentType.CONTENT_CREATION,
            **kwargs
        )
    
    async def _build_graph(self) -> None:
        """构建简单的聊天图"""
        self.graph = StateGraph(self.AgentState)
        
        # 添加聊天节点
        self.graph.add_node("chat", self._enhanced_chat_node)
        
        # 设置路由
        self.graph.add_edge(START, "chat")
        self.graph.add_edge("chat", END)
    
    async def _enhanced_chat_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """增强的聊天节点"""
        messages = state["messages"]
        user_id = state.get("user_id", "unknown")
        
        # 添加系统提示
        system_prompt = f"""你是一个友好的AI助手。
        当前用户ID: {user_id}
        请用友好、专业的语气回答用户的问题。
        """
        
        # 构建完整的消息列表
        full_messages = [AIMessage(content=system_prompt)] + messages
        
        # 调用LLM
        response = await self.llm.ainvoke(full_messages)
        
        return {"messages": [response]}


async def demo_create_custom_agents():
    """演示创建和使用自定义智能体"""
    logger.info("=== 创建自定义智能体演示 ===")
    
    # 1. 创建自定义分析智能体
    analysis_agent = CustomAnalysisAgent(
        agent_id="custom_analysis_001",
        debug=True
    )
    
    # 初始化智能体
    await analysis_agent.initialize()
    logger.info(f"创建分析智能体: {analysis_agent.name}")
    
    # 2. 创建自定义聊天机器人
    chatbot_agent = CustomChatbotAgent(
        agent_id="custom_chatbot_001",
        debug=True
    )
    
    # 初始化智能体
    await chatbot_agent.initialize()
    logger.info(f"创建聊天机器人: {chatbot_agent.name}")
    
    # 3. 测试分析智能体
    logger.info("\n--- 测试分析智能体 ---")
    analysis_request = {
        "messages": [HumanMessage(content="请帮我分析这些数据")],
        "user_id": "demo_user",
        "session_id": "demo_session_001",
        "metadata": {}
    }
    
    analysis_response = await analysis_agent.chat(analysis_request)
    logger.info(f"分析智能体响应: {analysis_response}")
    
    # 4. 测试聊天机器人
    logger.info("\n--- 测试聊天机器人 ---")
    chat_request = {
        "messages": [HumanMessage(content="你好，请介绍一下你自己")],
        "user_id": "demo_user", 
        "session_id": "demo_session_002",
        "metadata": {}
    }
    
    chat_response = await chatbot_agent.chat(chat_request)
    logger.info(f"聊天机器人响应: {chat_response}")


async def demo_register_custom_agents():
    """演示如何注册自定义智能体到系统中"""
    logger.info("=== 注册自定义智能体演示 ===")
    
    # 获取智能体注册表和工厂
    registry = get_agent_registry()
    factory = get_agent_factory()
    
    # 1. 注册自定义分析智能体配置
    analysis_config = AgentConfig(
        agent_type=AgentType.DATA_ANALYSIS,
        name="Custom Analysis Agent",
        description="专门用于数据分析的自定义智能体",
        llm_config={
            "model_type": "deepseek",
            "temperature": 0.3,
            "max_tokens": 3000
        },
        tools=["custom_analysis_tool", "custom_report_tool"],
        capabilities=["data_analysis", "report_generation", "statistical_analysis"],
        memory_enabled=True,
        collaboration_enabled=True
    )
    
    # 注册配置
    registry.register_agent_config(AgentType.DATA_ANALYSIS, analysis_config)
    logger.info("注册自定义分析智能体配置")
    
    # 2. 注册自定义聊天机器人配置
    chatbot_config = AgentConfig(
        agent_type=AgentType.CONTENT_CREATION,
        name="Custom Chatbot Agent", 
        description="友好的聊天机器人智能体",
        llm_config={
            "model_type": "deepseek",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        capabilities=["conversation", "content_creation", "user_assistance"],
        memory_enabled=True
    )
    
    # 注册配置
    registry.register_agent_config(AgentType.CONTENT_CREATION, chatbot_config)
    logger.info("注册自定义聊天机器人配置")
    
    # 3. 使用工厂创建智能体实例
    try:
        # 创建分析智能体实例
        analysis_instance = await factory.create_agent(
            agent_type=AgentType.DATA_ANALYSIS,
            user_id="demo_user",
            session_id="demo_session"
        )
        logger.info(f"通过工厂创建分析智能体: {analysis_instance.instance_id}")
        
        # 创建聊天机器人实例
        chatbot_instance = await factory.create_agent(
            agent_type=AgentType.CONTENT_CREATION,
            user_id="demo_user",
            session_id="demo_session"
        )
        logger.info(f"通过工厂创建聊天机器人: {chatbot_instance.instance_id}")
        
    except Exception as e:
        logger.error(f"创建智能体实例失败: {e}")
    
    # 4. 列出所有注册的智能体类型
    available_types = registry.get_available_agent_types()
    logger.info(f"可用的智能体类型: {available_types}")


if __name__ == "__main__":
    async def main():
        """主函数"""
        try:
            # 演示创建自定义智能体
            await demo_create_custom_agents()
            
            print("\n" + "="*50 + "\n")
            
            # 演示注册自定义智能体
            await demo_register_custom_agents()
            
        except Exception as e:
            logger.error(f"演示过程中出现错误: {e}")
    
    # 运行演示
    asyncio.run(main())