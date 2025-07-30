#!/usr/bin/env python3
"""
修复版本的 Graph5 多智能体系统
主要修复：
1. MCP 会话管理问题
2. 异步生成器冲突
3. 更好的错误处理
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# LangChain 和 LangGraph 导入
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

# MCP 相关导入
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
mcp_client = None
chart_tools = []
supervisor = None
checkpointer = None
store = None

def get_mcp_servers():
    """获取MCP服务器配置"""
    try:
        with open("servers_config.json", "r") as f:
            return json.load(f).get("mcpServers", {})
    except FileNotFoundError:
        logger.warning("servers_config.json 文件未找到，将不使用MCP工具")
        return {}
    except Exception as e:
        logger.error(f"读取MCP配置时出错: {e}")
        return {}

def get_postgres_uri():
    """获取PostgreSQL连接URI"""
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'password')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'langgraph_db')}"
    )

# 初始化MCP客户端
mcpServers = get_mcp_servers()
if mcpServers:
    mcp_client = MultiServerMCPClient(mcpServers)
    logger.info(f"初始化MCP客户端，服务器: {list(mcpServers.keys())}")
else:
    logger.info("未配置MCP服务器，将跳过MCP工具加载")

# 配置搜索工具
search = None
serper_api_key = os.getenv("SERPER_API_KEY")

if serper_api_key:
    try:
        search = GoogleSerperAPIWrapper(
            serper_api_key=serper_api_key,
            k=5
        )
        logger.info("Google Serper搜索工具初始化成功")
    except Exception as e:
        logger.warning(f"Google Serper搜索工具初始化失败: {e}")
        search = None
else:
    logger.warning("未找到SERPER_API_KEY，将跳过搜索功能")

@tool
def google_search(query: str) -> str:
    """使用Google搜索获取信息"""
    if not search:
        return "搜索功能暂时不可用：未配置SERPER_API_KEY"
    
    try:
        return search.run(query)
    except Exception as e:
        return f"搜索失败: {str(e)}"

# 配置LLM模型
def get_llm_model(model_type: str = "tongyi"):
    """获取LLM模型"""
    if model_type == "tongyi":
        return ChatTongyi(
            model="qwen-plus",
            temperature=0.7,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
    elif model_type == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# 创建智能体模型
research_llm = get_llm_model("tongyi")
supervisor_llm = get_llm_model("tongyi")

async def load_mcp_tools_safe():
    """安全地加载MCP工具"""
    global chart_tools
    
    if not mcp_client or not mcpServers:
        logger.info("跳过MCP工具加载：未配置MCP服务器")
        return []
    
    try:
        # 使用新的API方式获取工具
        tools = await mcp_client.get_tools("mcp-server-chart")
        logger.info(f"成功加载 {len(tools)} 个MCP工具")
        return tools
    except Exception as e:
        logger.error(f"加载MCP工具失败: {e}")
        logger.info("将继续运行，但不包含图表生成功能")
        return []

def create_handoff_tool(target_agent: str):
    """创建智能体间的任务转移工具"""
    @tool
    def handoff_to_agent(task_description: str) -> str:
        """将任务转移给指定智能体
        
        Args:
            task_description: 要转移的任务描述
            
        Returns:
            确认转移的消息
        """
        return f"任务已转移给{target_agent}: {task_description}"
    
    handoff_to_agent.name = f"handoff_to_{target_agent}"
    return handoff_to_agent

# 创建智能体工具
research_tools = [google_search, create_handoff_tool("chart_agent")]
supervisor_tools = [
    create_handoff_tool("research_agent"),
    create_handoff_tool("chart_agent")
]

async def enhanced_chart_agent_node(state: MessagesState):
    """增强的图表智能体节点，包含错误处理和重试机制"""
    global chart_tools
    
    # 如果还没有加载MCP工具，尝试加载
    if not chart_tools and mcp_client:
        chart_tools = await load_mcp_tools_safe()
    
    # 创建图表智能体的工具列表
    agent_tools = chart_tools + [create_handoff_tool("research_agent")]
    
    if not chart_tools:
        # 如果没有MCP工具，返回一个说明消息
        return {
            "messages": [
                AIMessage(content="抱歉，图表生成功能暂时不可用。MCP服务器连接失败。我可以为您提供数据分析的文字描述。")
            ]
        }
    
    # 创建图表智能体
    chart_agent = create_react_agent(
        supervisor_llm,
        agent_tools,
        state_modifier="你是一个专业的数据可视化专家。你可以创建各种类型的图表来展示数据。"
    )
    
    try:
        result = await chart_agent.ainvoke(state)
        return result
    except Exception as e:
        logger.error(f"图表智能体执行失败: {e}")
        return {
            "messages": [
                AIMessage(content=f"图表生成过程中出现错误: {str(e)}。请尝试重新描述您的需求。")
            ]
        }

# 创建智能体
research_agent = create_react_agent(
    research_llm,
    research_tools,
    state_modifier="你是一个专业的研究助手。你擅长搜索和分析信息，为用户提供准确、全面的研究结果。"
)

supervisor_agent = create_react_agent(
    supervisor_llm,
    supervisor_tools,
    state_modifier="""你是一个智能的任务协调者。你需要：
1. 理解用户的需求
2. 决定是否需要调用其他智能体
3. 协调多个智能体的工作
4. 整合结果并提供最终回答

可用的智能体：
- research_agent: 负责信息搜索和研究
- chart_agent: 负责数据可视化和图表生成
"""
)

# Store工具函数
async def create_user_memory_namespace(user_id: str) -> str:
    """为用户创建记忆命名空间"""
    return f"user_memory_{user_id}"

async def save_user_memory(user_id: str, memory_key: str, memory_value: dict):
    """保存用户记忆到Store"""
    global store
    if not store:
        logger.warning("Store未初始化，无法保存用户记忆")
        return
    
    try:
        namespace = await create_user_memory_namespace(user_id)
        await store.aput(namespace, memory_key, memory_value)
        logger.info(f"已保存用户 {user_id} 的记忆: {memory_key}")
    except Exception as e:
        logger.error(f"保存用户记忆失败: {e}")

async def get_user_memories(user_id: str, query: str = None, limit: int = 5) -> List[dict]:
    """从Store获取用户记忆"""
    global store
    if not store:
        logger.warning("Store未初始化，无法获取用户记忆")
        return []
    
    try:
        namespace = await create_user_memory_namespace(user_id)
        
        if query:
            # 使用语义搜索
            results = await store.asearch(namespace, query=query, limit=limit)
            return [{"key": r.key, "value": r.value} for r in results]
        else:
            # 获取所有记忆
            items = await store.alist(namespace, limit=limit)
            return [{"key": item.key, "value": item.value} for item in items]
    except Exception as e:
        logger.error(f"获取用户记忆失败: {e}")
        return []

async def save_user_preference(user_id: str, preference_key: str, preference_value: any):
    """保存用户偏好"""
    await save_user_memory(user_id, f"preference_{preference_key}", {
        "type": "preference",
        "key": preference_key,
        "value": preference_value,
        "timestamp": asyncio.get_event_loop().time()
    })

async def get_user_preferences(user_id: str) -> dict:
    """获取用户偏好"""
    memories = await get_user_memories(user_id)
    preferences = {}
    for memory in memories:
        if memory["value"].get("type") == "preference":
            preferences[memory["value"]["key"]] = memory["value"]["value"]
    return preferences

async def memory_aware_supervisor_agent(state: MessagesState, user_id: str = "default_user"):
    """具有记忆感知的supervisor智能体"""
    global store
    
    # 获取用户记忆和偏好
    user_memories = await get_user_memories(user_id, limit=3)
    user_preferences = await get_user_preferences(user_id)
    
    # 构建增强的上下文
    context_parts = []
    if user_memories:
        context_parts.append("用户历史记忆:")
        for memory in user_memories:
            context_parts.append(f"- {memory['key']}: {memory['value']}")
    
    if user_preferences:
        context_parts.append("用户偏好:")
        for key, value in user_preferences.items():
            context_parts.append(f"- {key}: {value}")
    
    enhanced_context = "\n".join(context_parts) if context_parts else "无历史记忆"
    
    # 修改状态以包含记忆上下文
    enhanced_state = state.copy()
    if enhanced_state["messages"]:
        last_message = enhanced_state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            enhanced_content = f"{last_message.content}\n\n[上下文记忆]\n{enhanced_context}"
            enhanced_state["messages"][-1] = HumanMessage(content=enhanced_content)
    
    # 调用supervisor智能体
    result = await supervisor_agent.ainvoke(enhanced_state)
    
    # 保存重要的交互到记忆
    if result.get("messages"):
        last_ai_message = result["messages"][-1]
        if isinstance(last_ai_message, AIMessage):
            await save_user_memory(user_id, f"interaction_{asyncio.get_event_loop().time()}", {
                "type": "interaction",
                "user_input": state["messages"][-1].content if state["messages"] else "",
                "ai_response": last_ai_message.content,
                "timestamp": asyncio.get_event_loop().time()
            })
    
    return result

async def supervisor_node_with_memory(state: MessagesState):
    """带记忆的supervisor节点"""
    # 这里可以从state中提取user_id，暂时使用默认值
    user_id = "default_user"
    return await memory_aware_supervisor_agent(state, user_id)

# 路由函数
def route_to_agent(state: MessagesState) -> str:
    """根据最后一条消息决定路由到哪个智能体"""
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage):
        content = last_message.content.lower()
        
        # 检查是否需要图表生成
        if any(keyword in content for keyword in ["图表", "chart", "可视化", "visualization", "画图", "绘制"]):
            return "chart_agent"
        
        # 检查是否需要研究
        if any(keyword in content for keyword in ["搜索", "search", "研究", "research", "查找", "信息"]):
            return "research_agent"
        
        # 检查handoff工具调用
        if "handoff_to_research_agent" in content:
            return "research_agent"
        elif "handoff_to_chart_agent" in content:
            return "chart_agent"
    
    return "supervisor"

async def initialize_system():
    """初始化多智能体系统"""
    global supervisor, checkpointer, store, chart_tools
    
    logger.info("开始初始化多智能体系统...")
    
    try:
        # 获取PostgreSQL连接URI
        postgres_uri = get_postgres_uri()
        logger.info("PostgreSQL URI配置完成")
        
        # 初始化嵌入模型
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        logger.info("嵌入模型初始化完成")
        
        # 初始化检查点管理器（使用上下文管理器）
        checkpointer_context = AsyncPostgresSaver.from_conn_string(postgres_uri)
        checkpointer = await checkpointer_context.__aenter__()
        
        # 设置检查点表结构
        if hasattr(checkpointer, 'setup'):
            await checkpointer.setup()
        logger.info("检查点管理器初始化完成")
        
        # 初始化Store（使用上下文管理器）
        store_context = AsyncPostgresStore.from_conn_string(
            postgres_uri,
            index={
                "embed": embeddings,
                "dims": 1024,
                "fields": ["$"]
            }
        )
        store = await store_context.__aenter__()
        
        # 设置Store表结构
        if hasattr(store, 'setup'):
            await store.setup()
        logger.info("Store初始化完成")
        
        # 加载MCP工具
        chart_tools = await load_mcp_tools_safe()
        
        # 创建图
        graph = StateGraph(MessagesState)
        
        # 添加节点
        graph.add_node("supervisor", supervisor_node_with_memory)
        graph.add_node("research_agent", research_agent)
        graph.add_node("chart_agent", enhanced_chart_agent_node)
        
        # 添加边
        graph.add_edge("research_agent", "supervisor")
        graph.add_edge("chart_agent", "supervisor")
        
        # 添加条件边
        graph.add_conditional_edges(
            "supervisor",
            route_to_agent,
            {
                "supervisor": "__end__",
                "research_agent": "research_agent",
                "chart_agent": "chart_agent"
            }
        )
        
        # 设置入口点
        graph.set_entry_point("supervisor")
        
        # 编译图
        supervisor = graph.compile(
            checkpointer=checkpointer,
            store=store
        )
        
        logger.info("多智能体系统初始化完成")
        return supervisor
        
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        # 尝试降级到内存存储
        logger.info("尝试降级到内存存储...")
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.store.memory import InMemoryStore
            
            checkpointer = MemorySaver()
            store = InMemoryStore()
            
            # 创建图（简化版本）
            graph = StateGraph(MessagesState)
            graph.add_node("supervisor", supervisor_node_with_memory)
            graph.add_node("research_agent", research_agent)
            graph.add_node("chart_agent", enhanced_chart_agent_node)
            
            graph.add_edge("research_agent", "supervisor")
            graph.add_edge("chart_agent", "supervisor")
            
            graph.add_conditional_edges(
                "supervisor",
                route_to_agent,
                {
                    "supervisor": "__end__",
                    "research_agent": "research_agent",
                    "chart_agent": "chart_agent"
                }
            )
            
            graph.set_entry_point("supervisor")
            
            supervisor = graph.compile(
                checkpointer=checkpointer,
                store=store
            )
            
            logger.info("降级到内存存储成功")
            return supervisor
            
        except Exception as fallback_error:
            logger.error(f"降级到内存存储也失败: {fallback_error}")
            raise

async def cleanup_system():
    """清理系统资源"""
    global checkpointer, store
    
    logger.info("开始清理系统资源...")
    
    try:
        if checkpointer:
            await checkpointer.aclose()
            logger.info("检查点管理器已清理")
        
        if store:
            await store.aclose()
            logger.info("Store已清理")
            
        logger.info("系统资源清理完成")
        
    except Exception as e:
        logger.error(f"清理系统资源时出错: {e}")

async def main():
    """主函数"""
    try:
        # 初始化系统
        supervisor_graph = await initialize_system()
        
        # 测试对话
        test_messages = [
            "你好，我想了解一下人工智能的发展历史",
            "请帮我搜索最新的AI技术趋势",
            "能否为我创建一个展示AI发展历程的图表？"
        ]
        
        config = {"configurable": {"thread_id": "test_thread_1"}}
        
        for message in test_messages:
            logger.info(f"用户输入: {message}")
            
            result = await supervisor_graph.ainvoke(
                {"messages": [HumanMessage(content=message)]},
                config=config
            )
            
            if result.get("messages"):
                last_message = result["messages"][-1]
                logger.info(f"AI回复: {last_message.content}")
            
            print("-" * 50)
        
    except Exception as e:
        logger.error(f"运行过程中出错: {e}")
    finally:
        # 清理资源
        await cleanup_system()

if __name__ == "__main__":
    asyncio.run(main())