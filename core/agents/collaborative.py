"""
多智能体LangGraph项目 - 协作型智能体实现

本模块实现基于LangGraph的协作型智能体，包括：
- Supervisor智能体（主管协调）
- Research智能体（研究分析）
- Chart智能体（图表生成）
- 智能体间的任务移交和协作
"""

import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.agents.base import (
    BaseAgent, AgentType, AgentStatus, AgentCapability, 
    ChatRequest, ChatResponse, StreamChunk
)
from core.tools import get_tool_registry, ToolExecutionContext, ToolPermission
from config.settings import get_settings


class CollaborativeState(BaseModel):
    """协作状态模型"""
    messages: List[BaseMessage] = Field(default_factory=list, description="消息历史")
    current_agent: Optional[str] = Field(default=None, description="当前处理的智能体")
    task_description: Optional[str] = Field(default=None, description="任务描述")
    research_results: Optional[Dict[str, Any]] = Field(default=None, description="研究结果")
    chart_results: Optional[Dict[str, Any]] = Field(default=None, description="图表结果")
    final_answer: Optional[str] = Field(default=None, description="最终答案")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class SupervisorAgent(BaseAgent):
    """主管智能体
    
    负责协调其他智能体，分配任务，整合结果。
    """
    
    def __init__(self, llm, tools: Optional[List[BaseTool]] = None):
        """初始化主管智能体
        
        Args:
            llm: 语言模型
            tools: 工具列表
        """
        super().__init__(
            agent_id="supervisor",
            agent_type=AgentType.COLLABORATIVE,
            name="Supervisor Agent",
            description="主管智能体，负责任务协调和结果整合"
        )
        
        self.llm = llm
        self.tools = tools or []
        self.logger = logging.getLogger("agent.supervisor")
        
        # 添加能力
        self.capabilities.extend([
            AgentCapability.TASK_COORDINATION,
            AgentCapability.RESULT_INTEGRATION,
            AgentCapability.DECISION_MAKING
        ])
    
    def _build_graph(self) -> StateGraph:
        """构建主管智能体的处理图"""
        # 创建基础的React智能体
        self._react_agent = create_react_agent(
            self.llm,
            self.tools,
            state_modifier="You are a supervisor agent responsible for coordinating tasks between research and chart agents. "
                          "Analyze the user's request and decide which agents to involve. "
                          "Use transfer tools to delegate tasks to appropriate agents."
        )
        
        # 创建状态图
        workflow = StateGraph(CollaborativeState)
        
        # 添加节点
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # 设置入口点
        workflow.set_entry_point("supervisor")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {
                "continue": "supervisor",
                "finalize": "finalize",
                "end": END
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def _supervisor_node(self, state: CollaborativeState) -> CollaborativeState:
        """主管节点处理逻辑"""
        try:
            # 获取最新消息
            messages = state.messages
            if not messages:
                return state
            
            # 调用React智能体
            result = await self._react_agent.ainvoke({
                "messages": messages
            })
            
            # 更新状态
            if "messages" in result:
                state.messages = result["messages"]
            
            # 检查是否有最终答案
            last_message = state.messages[-1] if state.messages else None
            if last_message and isinstance(last_message, AIMessage):
                # 简单检查是否包含最终答案的标识
                if "最终答案" in last_message.content or "final answer" in last_message.content.lower():
                    state.final_answer = last_message.content
            
            state.current_agent = "supervisor"
            
            return state
            
        except Exception as e:
            self.logger.error(f"主管节点处理失败: {e}")
            # 添加错误消息
            error_message = AIMessage(content=f"处理过程中出现错误: {str(e)}")
            state.messages.append(error_message)
            return state
    
    async def _finalize_node(self, state: CollaborativeState) -> CollaborativeState:
        """最终化节点"""
        try:
            # 整合所有结果
            summary_parts = []
            
            if state.research_results:
                summary_parts.append(f"研究结果: {state.research_results}")
            
            if state.chart_results:
                summary_parts.append(f"图表结果: {state.chart_results}")
            
            if state.final_answer:
                summary_parts.append(f"最终答案: {state.final_answer}")
            
            # 创建最终总结
            final_summary = "\n\n".join(summary_parts) if summary_parts else "任务完成"
            
            final_message = AIMessage(content=final_summary)
            state.messages.append(final_message)
            
            return state
            
        except Exception as e:
            self.logger.error(f"最终化节点处理失败: {e}")
            error_message = AIMessage(content=f"最终化过程中出现错误: {str(e)}")
            state.messages.append(error_message)
            return state
    
    def _should_continue(self, state: CollaborativeState) -> str:
        """判断是否继续处理"""
        # 检查是否有最终答案
        if state.final_answer:
            return "finalize"
        
        # 检查消息数量，避免无限循环
        if len(state.messages) > 20:
            return "end"
        
        # 检查最后一条消息是否是工具调用结果
        last_message = state.messages[-1] if state.messages else None
        if last_message and isinstance(last_message, AIMessage):
            # 如果包含工具调用，继续处理
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "continue"
            
            # 如果是普通回复且没有明确的结束标识，也继续处理
            if not any(keyword in last_message.content.lower() 
                      for keyword in ["完成", "结束", "finished", "done", "final"]):
                return "continue"
        
        return "finalize"
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """处理聊天请求"""
        try:
            self.status = AgentStatus.PROCESSING
            
            # 创建初始状态
            initial_state = CollaborativeState(
                messages=[HumanMessage(content=request.message)],
                task_description=request.message,
                metadata=request.metadata
            )
            
            # 执行图
            final_state = await self.graph.ainvoke(initial_state)
            
            # 获取最终回复
            last_message = final_state.messages[-1] if final_state.messages else None
            response_content = last_message.content if last_message else "处理完成"
            
            self.status = AgentStatus.IDLE
            
            return ChatResponse(
                message=response_content,
                agent_id=self.agent_id,
                metadata={
                    "total_messages": len(final_state.messages),
                    "research_results": final_state.research_results,
                    "chart_results": final_state.chart_results,
                    **final_state.metadata
                }
            )
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"聊天处理失败: {e}")
            return ChatResponse(
                message=f"处理请求时出现错误: {str(e)}",
                agent_id=self.agent_id,
                metadata={"error": str(e)}
            )
    
    async def astream(self, request: ChatRequest) -> AsyncGenerator[StreamChunk, None]:
        """流式处理聊天请求"""
        try:
            self.status = AgentStatus.PROCESSING
            
            # 创建初始状态
            initial_state = CollaborativeState(
                messages=[HumanMessage(content=request.message)],
                task_description=request.message,
                metadata=request.metadata
            )
            
            # 流式执行图
            async for chunk in self.graph.astream(initial_state):
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, CollaborativeState):
                        # 发送最新消息
                        if node_output.messages:
                            last_message = node_output.messages[-1]
                            yield StreamChunk(
                                content=last_message.content,
                                chunk_type="message",
                                metadata={
                                    "node": node_name,
                                    "current_agent": node_output.current_agent
                                }
                            )
            
            self.status = AgentStatus.IDLE
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"流式处理失败: {e}")
            yield StreamChunk(
                content=f"处理请求时出现错误: {str(e)}",
                chunk_type="error",
                metadata={"error": str(e)}
            )


class ResearchAgent(BaseAgent):
    """研究智能体
    
    专门负责信息搜索和研究分析任务。
    """
    
    def __init__(self, llm, search_tools: Optional[List[BaseTool]] = None):
        """初始化研究智能体
        
        Args:
            llm: 语言模型
            search_tools: 搜索工具列表
        """
        super().__init__(
            agent_id="research",
            agent_type=AgentType.COLLABORATIVE,
            name="Research Agent",
            description="研究智能体，专门负责信息搜索和分析"
        )
        
        self.llm = llm
        self.search_tools = search_tools or []
        self.logger = logging.getLogger("agent.research")
        
        # 添加能力
        self.capabilities.extend([
            AgentCapability.INFORMATION_RETRIEVAL,
            AgentCapability.DATA_ANALYSIS,
            AgentCapability.RESEARCH
        ])
    
    def _build_graph(self) -> StateGraph:
        """构建研究智能体的处理图"""
        # 创建基础的React智能体
        self._react_agent = create_react_agent(
            self.llm,
            self.search_tools,
            state_modifier="You are a research agent specialized in information gathering and analysis. "
                          "Use search tools to find relevant information and provide comprehensive research results. "
                          "Focus on accuracy, relevance, and depth of analysis."
        )
        
        # 创建简单的状态图
        workflow = StateGraph(dict)
        workflow.add_node("research", self._research_node)
        workflow.set_entry_point("research")
        workflow.add_edge("research", END)
        
        return workflow.compile()
    
    async def _research_node(self, state: dict) -> dict:
        """研究节点处理逻辑"""
        try:
            # 调用React智能体进行研究
            result = await self._react_agent.ainvoke(state)
            return result
            
        except Exception as e:
            self.logger.error(f"研究节点处理失败: {e}")
            return {
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"研究过程中出现错误: {str(e)}")
                ]
            }
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """处理聊天请求"""
        try:
            self.status = AgentStatus.PROCESSING
            
            # 执行研究
            result = await self.graph.ainvoke({
                "messages": [HumanMessage(content=request.message)]
            })
            
            # 获取回复
            last_message = result["messages"][-1] if result.get("messages") else None
            response_content = last_message.content if last_message else "研究完成"
            
            self.status = AgentStatus.IDLE
            
            return ChatResponse(
                message=response_content,
                agent_id=self.agent_id,
                metadata={"research_completed": True}
            )
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"研究处理失败: {e}")
            return ChatResponse(
                message=f"研究过程中出现错误: {str(e)}",
                agent_id=self.agent_id,
                metadata={"error": str(e)}
            )
    
    async def astream(self, request: ChatRequest) -> AsyncGenerator[StreamChunk, None]:
        """流式处理聊天请求"""
        try:
            self.status = AgentStatus.PROCESSING
            
            # 流式执行研究
            async for chunk in self.graph.astream({
                "messages": [HumanMessage(content=request.message)]
            }):
                for node_name, node_output in chunk.items():
                    if "messages" in node_output and node_output["messages"]:
                        last_message = node_output["messages"][-1]
                        yield StreamChunk(
                            content=last_message.content,
                            chunk_type="message",
                            metadata={"node": node_name}
                        )
            
            self.status = AgentStatus.IDLE
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"流式研究处理失败: {e}")
            yield StreamChunk(
                content=f"研究过程中出现错误: {str(e)}",
                chunk_type="error",
                metadata={"error": str(e)}
            )


class ChartAgent(BaseAgent):
    """图表智能体
    
    专门负责数据可视化和图表生成任务。
    """
    
    def __init__(self, llm, chart_tools: Optional[List[BaseTool]] = None):
        """初始化图表智能体
        
        Args:
            llm: 语言模型
            chart_tools: 图表工具列表
        """
        super().__init__(
            agent_id="chart",
            agent_type=AgentType.COLLABORATIVE,
            name="Chart Agent",
            description="图表智能体，专门负责数据可视化和图表生成"
        )
        
        self.llm = llm
        self.chart_tools = chart_tools or []
        self.logger = logging.getLogger("agent.chart")
        
        # 添加能力
        self.capabilities.extend([
            AgentCapability.DATA_VISUALIZATION,
            AgentCapability.CHART_GENERATION,
            AgentCapability.DATA_ANALYSIS
        ])
    
    def _build_graph(self) -> StateGraph:
        """构建图表智能体的处理图"""
        # 创建基础的React智能体
        self._react_agent = create_react_agent(
            self.llm,
            self.chart_tools,
            state_modifier="You are a chart agent specialized in data visualization and chart generation. "
                          "Use chart tools to create appropriate visualizations based on the data and requirements. "
                          "Focus on clarity, accuracy, and visual appeal of the charts."
        )
        
        # 创建简单的状态图
        workflow = StateGraph(dict)
        workflow.add_node("chart", self._chart_node)
        workflow.set_entry_point("chart")
        workflow.add_edge("chart", END)
        
        return workflow.compile()
    
    async def _chart_node(self, state: dict) -> dict:
        """图表节点处理逻辑"""
        try:
            # 调用React智能体进行图表生成
            result = await self._react_agent.ainvoke(state)
            return result
            
        except Exception as e:
            self.logger.error(f"图表节点处理失败: {e}")
            return {
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"图表生成过程中出现错误: {str(e)}")
                ]
            }
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """处理聊天请求"""
        try:
            self.status = AgentStatus.PROCESSING
            
            # 执行图表生成
            result = await self.graph.ainvoke({
                "messages": [HumanMessage(content=request.message)]
            })
            
            # 获取回复
            last_message = result["messages"][-1] if result.get("messages") else None
            response_content = last_message.content if last_message else "图表生成完成"
            
            self.status = AgentStatus.IDLE
            
            return ChatResponse(
                message=response_content,
                agent_id=self.agent_id,
                metadata={"chart_completed": True}
            )
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"图表处理失败: {e}")
            return ChatResponse(
                message=f"图表生成过程中出现错误: {str(e)}",
                agent_id=self.agent_id,
                metadata={"error": str(e)}
            )
    
    async def astream(self, request: ChatRequest) -> AsyncGenerator[StreamChunk, None]:
        """流式处理聊天请求"""
        try:
            self.status = AgentStatus.PROCESSING
            
            # 流式执行图表生成
            async for chunk in self.graph.astream({
                "messages": [HumanMessage(content=request.message)]
            }):
                for node_name, node_output in chunk.items():
                    if "messages" in node_output and node_output["messages"]:
                        last_message = node_output["messages"][-1]
                        yield StreamChunk(
                            content=last_message.content,
                            chunk_type="message",
                            metadata={"node": node_name}
                        )
            
            self.status = AgentStatus.IDLE
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"流式图表处理失败: {e}")
            yield StreamChunk(
                content=f"图表生成过程中出现错误: {str(e)}",
                chunk_type="error",
                metadata={"error": str(e)}
            )