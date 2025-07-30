from dotenv import load_dotenv
load_dotenv(override=True)
import ssl
import aiohttp
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchResults,DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import convert_to_messages
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
import json
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
import os
import asyncio
import httpx
import logging
import re
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

# PostgreSQL持久化存储
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# 添加Store功能
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.store.base import BaseStore
import psycopg
from psycopg.rows import dict_row

os.environ["SERPER_API_KEY"] = "4dd04756479b347e0f2359b5b4fc8db5261da440"

# 全局变量用于管理长期运行的资源
checkpointer_context_manager = None
store_context_manager = None

# 全局变量来保持MCP会话、智能体和持久化存储
mcp_session = None
chart_agent = None
supervisor = None
checkpointer = None
store = None
local_supervisor_agent = None

def get_mcp_servers():
    with open("servers_config.json", "r", encoding="utf-8") as f:
        return json.load(f).get("mcpServers", {})

# 创建搜索工具 - 将 GoogleSerperAPIWrapper 转换为工具
search_wrapper = GoogleSerperAPIWrapper()
search = Tool(
    name="google_search",
    description="搜索互联网信息。输入应该是搜索查询字符串。",
    func=search_wrapper.run,
)

model = ChatTongyi(model="qwen-plus")
model2 = ChatDeepSeek(model="deepseek-chat")

mcpServers = get_mcp_servers()
client = MultiServerMCPClient(mcpServers)

async def validate_image_url(url: str, max_retries: int = 2) -> bool:
    """验证图片URL是否可访问"""
    if not url or not url.startswith('http'):
        return False
        
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(max_retries + 1):
            try:
                response = await client.head(url)
                if response.status_code == 200:
                    logging.info(f"图片URL验证成功: {url}")
                    return True
                else:
                    logging.warning(f"图片URL返回状态码 {response.status_code}: {url}")
                    if attempt < max_retries:
                        await asyncio.sleep(1)  # 等待1秒后重试
                        continue
                    return False
            except Exception as e:
                logging.warning(f"图片URL验证失败 (尝试 {attempt + 1}/{max_retries + 1}): {url}, 错误: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                return False
        return False

research_agent = create_react_agent(
    model=model2,
    tools=[search],
    prompt=(
       "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any chart/visualization\n"
        "- After completing your research, respond with 'TASK COMPLETED:' followed by your findings\n"
        "- Do NOT ask for more tasks or continue working after completing the assigned task\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )

    return handoff_tool

# Handoffs
assign_to_chart_agent = create_handoff_tool(
    agent_name="chart_agent",
    description="Assign task to a chart agent.",
)

assign_to_research_agent = create_handoff_tool(
    agent_name="research_agent",
    description="Assign task to a researcher agent.",
)

supervisor_agent = create_react_agent(
    model=model2,
    tools=[assign_to_chart_agent, assign_to_research_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a chart agent. Assign chart/visualization-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "When an agent responds with 'TASK COMPLETED:', the task is finished.\n"
        "Do not assign more work after a task is completed.\n"
        "For general conversations or simple questions, you can respond directly without assigning to other agents.\n"
        "Only assign to specialized agents when specific research or chart/visualization work is needed."
    ),
    name="supervisor",
)

async def enhanced_chart_agent_node(state: MessagesState, max_retries: int = 3):
    """增强的chart_agent节点，带URL验证和重试机制"""
    global chart_agent
    
    original_messages = state["messages"]
    
    for attempt in range(max_retries):
        try:
            # 调用原始的chart_agent
            logging.info(f"Chart agent 尝试 {attempt + 1}/{max_retries}")
            result = await chart_agent.ainvoke(state)
            
            # 检查结果中的图片URL
            if result and "messages" in result:
                messages = result["messages"]
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        content = last_message.content
                        
                        # 使用正则表达式查找图片URL
                        url_pattern = r'https://[^\s<>\[\]{}"|\'`]*\.(?:png|jpg|jpeg|gif|webp|svg)|https://mdn\.alipayobjects\.com/[^\s<>\[\]{}"|\'`]*'
                        urls = re.findall(url_pattern, content, re.IGNORECASE)
                        
                        if urls:
                            # 验证所有找到的图片URL
                            all_valid = True
                            for url in urls:
                                is_valid = await validate_image_url(url)
                                if not is_valid:
                                    all_valid = False
                                    logging.error(f"图片URL验证失败: {url}")
                                    break
                            
                            if all_valid:
                                logging.info("所有图片URL验证成功，返回结果")
                                return result
                            else:
                                logging.warning(f"图片URL验证失败，尝试重新生成 ({attempt + 1}/{max_retries})")
                                if attempt < max_retries - 1:
                                    # 添加重试提示到消息
                                    retry_message = HumanMessage(
                                        content="上一次生成的图片链接无效，请重新生成图表，确保图片链接可访问。"
                                    )
                                    state = {**state, "messages": original_messages + [retry_message]}
                                    continue
                                else:
                                    logging.error("达到最大重试次数，返回最后一次结果")
                                    return result
                        else:
                            # 没有找到图片URL，直接返回
                            logging.info("未找到图片URL，直接返回结果")
                            return result
                    else:
                        logging.info("消息没有content属性，直接返回结果")
                        return result
                else:
                    logging.warning("结果中没有消息")
                    return result
            else:
                logging.warning("Chart agent 返回空结果")
                return result
                
        except Exception as e:
            logging.error(f"Chart agent 执行出错 (尝试 {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            else:
                # 返回错误消息
                error_message = AIMessage(
                    content=f"抱歉，图表生成失败。错误: {str(e)}"
                )
                return {"messages": original_messages + [error_message]}
    
    # 如果所有尝试都失败了
    error_message = AIMessage(
        content="抱歉，经过多次尝试仍无法生成有效的图表。"
    )
    return {"messages": original_messages + [error_message]}

async def initialize_system():
    """初始化多智能体系统，建立持久的MCP连接、PostgreSQL持久化存储和跨线程Store"""
    global mcp_session, chart_agent, supervisor, checkpointer, store, local_supervisor_agent
    global checkpointer_context_manager, store_context_manager
    
    # 建立PostgreSQL持久化存储
    # 使用默认的PostgreSQL连接或从环境变量读取
    postgres_conn_string = os.getenv(
        "POSTGRES_URL", 
        "postgresql://postgres:postgres123@47.107.169.40:5432/langgraph?sslmode=disable"
    )
    
    logging.info(f"初始化PostgreSQL持久化存储，连接字符串: {postgres_conn_string.split('@')[0] + '@***'}")
    
    try:
        # 初始化Checkpointer (会话持久化) - 保持上下文管理器
        checkpointer_context_manager = AsyncPostgresSaver.from_conn_string(postgres_conn_string)
        checkpointer = await checkpointer_context_manager.__aenter__()
        
        # 初始化Store (跨线程长期记忆) - 保持上下文管理器
        store_context_manager = AsyncPostgresStore.from_conn_string(postgres_conn_string)
        store = await store_context_manager.__aenter__()
        
        # 首次使用时需要设置表结构
        try:
            await checkpointer.setup()
            await store.setup()
            logging.info("📋 数据库表结构初始化成功")
        except Exception as setup_error:
            logging.warning(f"表结构设置警告（可能已存在）: {setup_error}")
        
        logging.info("✅ AsyncPostgresSaver + AsyncPostgresStore 持久化存储初始化成功")
        logging.info(f"🔍 DEBUG: checkpointer = {type(checkpointer)} {id(checkpointer)}")
        logging.info(f"🔍 DEBUG: store = {type(store)} {id(store)}")
        logging.info(f"🔍 DEBUG: store_context_manager = {type(store_context_manager)} {id(store_context_manager)}")
        
    except Exception as e:
        logging.error(f"存储初始化失败: {e}")
        logging.info("降级使用内存存储")
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore
        checkpointer = MemorySaver()
        store = InMemoryStore()
        # 清空上下文管理器
        checkpointer_context_manager = None
        store_context_manager = None
    
    # 建立持久的MCP会话
    mcp_session = client.session("mcp-server-chart")
    session = await mcp_session.__aenter__()
    tools = await load_mcp_tools(session)
        
    # 在函数内部重新定义所有agents以确保一致性
    # 创建 chart_agent（使用MCP工具）
    chart_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=(
            "- Assist ONLY with chart/visualization-related tasks\n"
            "- After completing your chart/visualization, respond with 'TASK COMPLETED:' followed by your results\n"
            "- Do NOT ask for more tasks or continue working after completing the assigned task\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text.\n"
            "- NEVER fabricate success messages when tools actually failed."
        ),
        name="chart_agent",
    )
        
    # 创建 research_agent（使用搜索工具）
    local_research_agent = create_react_agent(
        model=model2,
        tools=[search],
        prompt=(
           "You are a research agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, DO NOT do any chart/visualization\n"
            "- After completing your research, respond with 'TASK COMPLETED:' followed by your findings\n"
            "- Do NOT ask for more tasks or continue working after completing the assigned task\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="research_agent",
    )
    
    # 重新定义handoff工具
    def create_local_handoff_tool(*, agent_name: str, description: str | None = None):
        name = f"transfer_to_{agent_name}"
        description = description or f"Ask {agent_name} for help."

        @tool(name, description=description)
        def handoff_tool(
            state: Annotated[MessagesState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            tool_message = {
                "role": "tool",
                "content": f"Successfully transferred to {agent_name}",
                "name": name,
                "tool_call_id": tool_call_id,
            }
            return Command(
                goto=agent_name,  
                update={**state, "messages": state["messages"] + [tool_message]},  
                graph=Command.PARENT,  
            )
        return handoff_tool

    # 本地handoff工具
    local_assign_to_chart_agent = create_local_handoff_tool(
        agent_name="chart_agent",
        description="Assign task to a chart agent.",
    )

    local_assign_to_research_agent = create_local_handoff_tool(
        agent_name="research_agent",
        description="Assign task to a researcher agent.",
    )

    # 创建supervisor_agent（使用本地handoff工具）
    local_supervisor_agent = create_react_agent(
        model=model2,
        tools=[local_assign_to_chart_agent, local_assign_to_research_agent],
        prompt=(
            "You are a supervisor managing two agents:\n"
            "- a research agent. Assign research-related tasks to this agent\n"
            "- a chart agent. Assign chart/visualization-related tasks to this agent\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "When an agent responds with 'TASK COMPLETED:', the task is finished.\n"
            "Do not assign more work after a task is completed.\n"
            "For general conversations or simple questions, you can respond directly without assigning to other agents.\n"
            "Only assign to specialized agents when specific research or chart/visualization work is needed."
        ),
        name="supervisor",
    )
    
    # 创建增强的chart_agent节点函数
    async def enhanced_chart_agent_node_with_tools(state: MessagesState, max_retries: int = 3):
        """增强的chart_agent节点，带URL验证和重试机制"""
        for attempt in range(max_retries + 1):
            try:
                # 调用chart_agent（包含MCP工具）
                result = await chart_agent.ainvoke(state)
                
                # 检查结果中是否包含图片URL
                if result and "messages" in result:
                    for message in result["messages"]:
                        if hasattr(message, 'content') and isinstance(message.content, str):
                            # 查找图片URL
                            import re
                            url_pattern = r'https?://[^\s)]+\.(png|jpg|jpeg|gif|webp)'
                            urls = re.findall(url_pattern, message.content)
                            
                            # 验证每个图片URL
                            for url in urls:
                                is_valid = await validate_image_url(url)
                                if not is_valid:
                                    if attempt < max_retries:
                                        logging.warning(f"图片URL验证失败，第{attempt + 1}次重试: {url}")
                                        # 添加重试提示到状态
                                        retry_message = f"Previous image URL failed to load: {url}. Please generate a new chart."
                                        retry_state = {
                                            **state,
                                            "messages": state["messages"] + [{"role": "user", "content": retry_message}]
                                        }
                                        state = retry_state
                                        continue
                                    else:
                                        logging.error(f"图片URL在{max_retries}次重试后仍然无效: {url}")
                
                return result
                
            except Exception as e:
                logging.error(f"Chart agent执行错误 (尝试 {attempt + 1}): {e}")
                if attempt == max_retries:
                    raise
                continue
    
    # 创建具有记忆功能的supervisor节点包装函数
    async def supervisor_node_with_memory(state: MessagesState, config: RunnableConfig = None):
        """Supervisor节点，集成记忆功能"""
        return await memory_aware_supervisor_agent(state, store=store, config=config)
    
    # 创建 supervisor graph with checkpointer 和 store
    supervisor = (
        StateGraph(MessagesState)
        .add_node("supervisor", supervisor_node_with_memory)  # 使用具有记忆功能的supervisor
        .add_node("chart_agent", enhanced_chart_agent_node_with_tools)
        .add_node("research_agent", local_research_agent)  # 使用正确的名称
        .add_edge(START, "supervisor")
        .add_edge("research_agent", "supervisor")
        .add_edge("chart_agent", "supervisor")
        .add_conditional_edges(
            "supervisor",
            lambda x: x.get("next", END),  # supervisor决定下一步
            {"chart_agent": "chart_agent", "research_agent": "research_agent", END: END}
        )
        .compile(checkpointer=checkpointer, store=store)
    )
    
    logging.info("多智能体系统与持久化存储(Checkpointer + Store)初始化完成")
    logging.info(f"🔍 DEBUG: 函数结束前 store = {type(store) if store else 'None'} {id(store) if store else 'N/A'}")
    return supervisor

# ================================
# Store工具函数 - 跨线程长期记忆管理
# ================================

def create_user_memory_namespace(user_id: str) -> tuple:
    """创建用户记忆的namespace"""
    return ("user", user_id, "memories")

def create_user_preferences_namespace(user_id: str) -> tuple:
    """创建用户偏好的namespace"""  
    return ("user", user_id, "preferences")

def create_global_knowledge_namespace() -> tuple:
    """创建全局知识库的namespace"""
    return ("knowledge", "global")

async def save_user_memory(store: BaseStore, user_id: str, memory_key: str, memory_value: dict):
    """保存用户记忆到Store"""
    namespace = create_user_memory_namespace(user_id)
    await store.aput(namespace, memory_key, memory_value)
    logging.info(f"💾 保存用户记忆: {user_id} -> {memory_key}")

async def get_user_memories(store: BaseStore, user_id: str, query: str = None, limit: int = 5):
    """获取用户记忆"""
    namespace = create_user_memory_namespace(user_id)
    if query:
        # 语义搜索
        memories = await store.asearch(namespace, query=query, limit=limit)
    else:
        # 获取所有记忆
        memories = await store.asearch(namespace, limit=limit)
    
    logging.info(f"🔍 检索用户记忆: {user_id}, 找到 {len(memories)} 条记忆")
    return memories

async def save_user_preference(store: BaseStore, user_id: str, preference_key: str, preference_value: dict):
    """保存用户偏好"""
    namespace = create_user_preferences_namespace(user_id)
    await store.aput(namespace, preference_key, preference_value)
    logging.info(f"⚙️ 保存用户偏好: {user_id} -> {preference_key}")

async def get_user_preferences(store: BaseStore, user_id: str):
    """获取用户偏好"""
    namespace = create_user_preferences_namespace(user_id)
    preferences = await store.asearch(namespace)
    logging.info(f"⚙️ 检索用户偏好: {user_id}, 找到 {len(preferences)} 条偏好")
    return preferences

# Store功能的智能体节点示例
async def memory_aware_supervisor_agent(state: MessagesState, store: BaseStore = None, config: dict = None):
    """具有记忆功能的supervisor智能体"""
    enhanced_state = state.copy()
    
    if store and config:
        user_id = config.get("configurable", {}).get("user_id", "default_user")
        
        # 获取用户记忆和偏好
        if state["messages"]:
            last_message_content = str(state["messages"][-1].content)
        else:
            last_message_content = ""
        memories = await get_user_memories(store, user_id, query=last_message_content, limit=3)
        preferences = await get_user_preferences(store, user_id)
        
        # 构建增强的上下文
        memory_context = "\n".join([f"记忆: {m.value}" for m in memories]) if memories else ""
        preference_context = "\n".join([f"偏好: {p.value}" for p in preferences]) if preferences else ""
        
        if memory_context or preference_context:
            # 创建增强的系统消息，包含用户记忆和偏好
            enhanced_prompt = f"""
基于用户历史信息协调智能体任务:

用户记忆信息:
{memory_context}

用户偏好设置:
{preference_context}

请基于以上用户上下文信息来更好地理解和处理用户请求。
            """.strip()
            
            # 在消息列表开头插入系统消息，不修改用户消息内容
            enhanced_messages = []
            if state["messages"]:
                # 在用户消息之前插入系统消息
                from langchain_core.messages import SystemMessage
                system_msg = SystemMessage(content=enhanced_prompt)
                enhanced_messages.append(system_msg)
                
                # 保持原始用户消息不变
                enhanced_messages.extend(state["messages"])
                
                enhanced_state = {**state, "messages": enhanced_messages}
            else:
                enhanced_state = state
            
            logging.info(f"🧠 Supervisor使用了 {len(memories)} 条记忆和 {len(preferences)} 条偏好")
        else:
            enhanced_state = state
    
    # 使用增强的状态调用原始supervisor
    return await local_supervisor_agent.ainvoke(enhanced_state)

async def cleanup_system():
    """清理系统资源"""
    global mcp_session, checkpointer_context_manager, store_context_manager
    
    # 清理 MCP 会话
    if mcp_session:
        try:
            await mcp_session.__aexit__(None, None, None)
            logging.info("MCP会话已清理")
        except Exception as e:
            logging.error(f"清理MCP会话时出错: {e}")
    
    # 清理 Checkpointer 上下文
    if checkpointer_context_manager:
        try:
            await checkpointer_context_manager.__aexit__(None, None, None)
            logging.info("Checkpointer上下文已清理")
        except Exception as e:
            logging.error(f"清理Checkpointer上下文时出错: {e}")
    
    # 清理 Store 上下文
    if store_context_manager:
        try:
            await store_context_manager.__aexit__(None, None, None)
            logging.info("Store上下文已清理")
        except Exception as e:
            logging.error(f"清理Store上下文时出错: {e}")
    
    logging.info("🧹 系统资源清理完成")

async def main():
    """主函数，用于测试"""
    try:
        await initialize_system()
        print("✅ 多智能体系统初始化完成")
        
        # 可选：添加测试代码
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        await cleanup_system()
        raise

if __name__ == "__main__": 
    asyncio.run(main())