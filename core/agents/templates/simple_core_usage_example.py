#!/usr/bin/env python3
"""
简单的核心模块使用示例

这个文件展示了如何在智能体中安全地使用核心模块，
包括适当的错误处理和降级机制。

运行方式:
python simple_core_usage_example.py
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 安全导入核心模块
# ============================================================================

# 1. 导入基础智能体
try:
    from core.agents.base import BaseAgent, ChatRequest, ChatResponse
    from core.agents.memory_enhanced import MemoryEnhancedAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"智能体模块导入失败: {e}")
    AGENTS_AVAILABLE = False

# 2. 导入记忆模块
try:
    from core.memory import MemoryType, MemoryScope
    MEMORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"记忆模块导入失败: {e}")
    MEMORY_AVAILABLE = False

# 3. 导入工具模块
try:
    from core.tools.enhanced_tool_manager import get_enhanced_tool_manager
    TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"工具模块导入失败: {e}")
    TOOLS_AVAILABLE = False

# 4. 导入流式处理模块
try:
    from core.streaming import get_stream_manager
    STREAMING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"流式处理模块导入失败: {e}")
    STREAMING_AVAILABLE = False

# 5. 导入时间旅行模块
try:
    from core.time_travel import get_time_travel_manager
    TIME_TRAVEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"时间旅行模块导入失败: {e}")
    TIME_TRAVEL_AVAILABLE = False

# ============================================================================
# 定义自定义工具
# ============================================================================

@tool
def calculate_tool(expression: str) -> str:
    """简单的计算工具"""
    try:
        # 安全的数学表达式计算
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        else:
            return "错误: 包含不允许的字符"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def weather_tool(city: str) -> str:
    """模拟天气查询工具"""
    weather_data = {
        "北京": "晴天，温度 15°C",
        "上海": "多云，温度 18°C",
        "广州": "小雨，温度 22°C",
        "深圳": "晴天，温度 25°C"
    }
    return weather_data.get(city, f"抱歉，暂无{city}的天气信息")

@tool
def knowledge_search_tool(query: str) -> str:
    """模拟知识搜索工具"""
    knowledge_base = {
        "python": "Python是一种高级编程语言，以其简洁和可读性著称。",
        "ai": "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习": "机器学习是人工智能的一个子集，使计算机能够在没有明确编程的情况下学习和改进。"
    }
    
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return f"找到相关信息: {value}"
    
    return f"抱歉，没有找到关于'{query}'的相关信息。"

# ============================================================================
# 模拟LLM类
# ============================================================================

class MockLLM:
    """模拟的LLM，用于演示"""
    
    async def ainvoke(self, messages):
        """模拟异步调用"""
        if not messages:
            return AIMessage(content="你好！我是一个演示智能体。")
        
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            content = last_message.content.lower()
            
            if "计算" in content or "算" in content:
                return AIMessage(content="我可以帮你进行简单的数学计算。请告诉我要计算的表达式。")
            elif "天气" in content:
                return AIMessage(content="我可以查询天气信息。请告诉我你想查询哪个城市的天气。")
            elif "知识" in content or "搜索" in content:
                return AIMessage(content="我可以搜索知识库。请告诉我你想了解什么。")
            elif "功能" in content or "能力" in content:
                return AIMessage(content="我具有以下功能：数学计算、天气查询、知识搜索。我还集成了记忆、工具管理等核心模块。")
            else:
                return AIMessage(content=f"你说: {last_message.content}。我是一个演示智能体，可以进行计算、查询天气和搜索知识。")
        
        return AIMessage(content="我是一个演示智能体，很高兴为你服务！")

# ============================================================================
# 简单的智能体实现
# ============================================================================

class SimpleAgent:
    """简单的智能体实现，展示核心模块使用"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.name = "简单演示智能体"
        self.llm = MockLLM()
        self.tools = [calculate_tool, weather_tool, knowledge_search_tool]
        
        # 核心模块管理器
        self.tool_manager = None
        self.stream_manager = None
        self.time_travel_manager = None
        
        # 功能开关
        self.features = {
            "agents": AGENTS_AVAILABLE,
            "memory": MEMORY_AVAILABLE,
            "tools": TOOLS_AVAILABLE,
            "streaming": STREAMING_AVAILABLE,
            "time_travel": TIME_TRAVEL_AVAILABLE
        }
        
        logger.info(f"智能体创建完成: {agent_id}")
        logger.info(f"可用功能: {self.features}")
    
    async def initialize(self):
        """初始化智能体和核心模块"""
        logger.info("开始初始化智能体...")
        
        # 1. 初始化工具管理器
        if self.features["tools"]:
            try:
                self.tool_manager = get_enhanced_tool_manager()
                
                # 注册自定义工具
                for tool in self.tools:
                    await self.tool_manager.register_tool(
                        tool,
                        metadata={
                            "category": "demo",
                            "agent_id": self.agent_id,
                            "created_at": datetime.utcnow().isoformat()
                        }
                    )
                
                logger.info(f"工具管理器初始化成功，注册了 {len(self.tools)} 个工具")
                
            except Exception as e:
                logger.error(f"工具管理器初始化失败: {e}")
                self.features["tools"] = False
        
        # 2. 初始化流式管理器
        if self.features["streaming"]:
            try:
                self.stream_manager = get_stream_manager()
                logger.info("流式管理器初始化成功")
            except Exception as e:
                logger.error(f"流式管理器初始化失败: {e}")
                self.features["streaming"] = False
        
        # 3. 初始化时间旅行管理器
        if self.features["time_travel"]:
            try:
                self.time_travel_manager = get_time_travel_manager()
                logger.info("时间旅行管理器初始化成功")
            except Exception as e:
                logger.error(f"时间旅行管理器初始化失败: {e}")
                self.features["time_travel"] = False
        
        logger.info("智能体初始化完成")
        logger.info(f"最终功能状态: {self.features}")
    
    async def chat(self, message: str, user_id: str = "demo_user", session_id: str = "demo_session") -> str:
        """简单的对话处理"""
        try:
            # 创建消息
            human_message = HumanMessage(content=message)
            
            # 调用LLM
            response = await self.llm.ainvoke([human_message])
            
            return response.content
            
        except Exception as e:
            logger.error(f"对话处理失败: {e}")
            return f"抱歉，处理你的消息时出现了错误: {e}"
    
    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """执行工具"""
        if not self.features["tools"] or not self.tool_manager:
            # 降级到直接工具调用
            for tool in self.tools:
                if tool.name == tool_name:
                    try:
                        result = tool.invoke(tool_input)
                        return str(result)
                    except Exception as e:
                        return f"工具执行失败: {e}"
            return f"未找到工具: {tool_name}"
        
        try:
            # 使用工具管理器执行
            from core.tools import ToolExecutionContext
            
            context = ToolExecutionContext(
                user_id="demo_user",
                session_id="demo_session",
                agent_id=self.agent_id,
                execution_id=f"exec_{datetime.utcnow().timestamp()}",
                metadata={"timestamp": datetime.utcnow().isoformat()}
            )
            
            result = await self.tool_manager.execute_tool(
                tool_name,
                tool_input,
                context
            )
            
            return str(result.output) if result else "工具执行失败"
            
        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            return f"工具执行失败: {e}"
    
    async def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        status = {
            "agent_id": self.agent_id,
            "name": self.name,
            "features": self.features,
            "tools": [tool.name for tool in self.tools],
            "initialized": True
        }
        
        # 添加工具统计（如果可用）
        if self.features["tools"] and self.tool_manager:
            try:
                tool_stats = await self.tool_manager.get_execution_stats()
                status["tool_stats"] = tool_stats
            except Exception as e:
                logger.warning(f"获取工具统计失败: {e}")
        
        return status

# ============================================================================
# 使用记忆增强智能体的示例（如果可用）
# ============================================================================

class MemoryAwareAgent:
    """使用记忆功能的智能体示例"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.name = "记忆增强智能体"
        self.llm = MockLLM()
        
        # 如果记忆和智能体模块都可用，使用MemoryEnhancedAgent
        if AGENTS_AVAILABLE and MEMORY_AVAILABLE:
            try:
                # 这里需要根据实际的MemoryEnhancedAgent实现来调整
                logger.info("记忆增强智能体可用")
                self.memory_enabled = True
            except Exception as e:
                logger.warning(f"记忆增强智能体初始化失败: {e}")
                self.memory_enabled = False
        else:
            self.memory_enabled = False
            logger.info("记忆功能不可用，使用基础智能体")
    
    async def chat_with_memory(self, message: str, user_id: str = "demo_user") -> str:
        """带记忆的对话"""
        if not self.memory_enabled:
            return await self.llm.ainvoke([HumanMessage(content=message)])
        
        # 这里可以添加记忆相关的逻辑
        # 例如：检索相关记忆、存储对话等
        
        response = await self.llm.ainvoke([HumanMessage(content=message)])
        
        # 模拟存储对话记忆
        logger.info(f"存储对话记忆: 用户说'{message}'，AI回复'{response.content}'")
        
        return response.content

# ============================================================================
# 演示和测试函数
# ============================================================================

async def demo_simple_agent():
    """演示简单智能体"""
    print("=" * 60)
    print("🤖 简单智能体演示")
    print("=" * 60)
    
    # 创建智能体
    agent = SimpleAgent("demo_agent_001")
    await agent.initialize()
    
    # 获取状态
    status = await agent.get_status()
    print(f"\n📊 智能体状态:")
    print(f"  ID: {status['agent_id']}")
    print(f"  名称: {status['name']}")
    print(f"  可用功能: {status['features']}")
    print(f"  工具列表: {status['tools']}")
    
    # 测试对话
    print(f"\n💬 对话测试:")
    test_messages = [
        "你好！",
        "你有什么功能？",
        "帮我计算 2 + 3 * 4",
        "查询北京的天气",
        "搜索关于Python的知识"
    ]
    
    for message in test_messages:
        print(f"\n👤 用户: {message}")
        response = await agent.chat(message)
        print(f"🤖 智能体: {response}")
    
    # 测试工具执行
    print(f"\n🔧 工具执行测试:")
    tool_tests = [
        ("calculate_tool", {"expression": "10 + 5 * 2"}),
        ("weather_tool", {"city": "上海"}),
        ("knowledge_search_tool", {"query": "机器学习"})
    ]
    
    for tool_name, tool_input in tool_tests:
        print(f"\n🛠️  执行工具: {tool_name}")
        print(f"   输入: {tool_input}")
        result = await agent.execute_tool(tool_name, tool_input)
        print(f"   结果: {result}")

async def demo_memory_agent():
    """演示记忆智能体"""
    print("\n" + "=" * 60)
    print("🧠 记忆增强智能体演示")
    print("=" * 60)
    
    agent = MemoryAwareAgent("memory_agent_001")
    
    print(f"记忆功能状态: {'✅ 可用' if agent.memory_enabled else '❌ 不可用'}")
    
    # 测试对话
    test_messages = [
        "我叫张三，今年25岁",
        "我喜欢编程和阅读",
        "你还记得我的名字吗？",
        "我的爱好是什么？"
    ]
    
    for message in test_messages:
        print(f"\n👤 用户: {message}")
        response = await agent.chat_with_memory(message)
        print(f"🤖 智能体: {response}")

async def main():
    """主函数"""
    print("🚀 核心模块使用演示开始")
    print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 演示简单智能体
        await demo_simple_agent()
        
        # 演示记忆智能体
        await demo_memory_agent()
        
        print("\n" + "=" * 60)
        print("✅ 演示完成！")
        print("=" * 60)
        
        print(f"\n📝 总结:")
        print(f"  - 智能体模块: {'✅' if AGENTS_AVAILABLE else '❌'}")
        print(f"  - 记忆模块: {'✅' if MEMORY_AVAILABLE else '❌'}")
        print(f"  - 工具模块: {'✅' if TOOLS_AVAILABLE else '❌'}")
        print(f"  - 流式处理: {'✅' if STREAMING_AVAILABLE else '❌'}")
        print(f"  - 时间旅行: {'✅' if TIME_TRAVEL_AVAILABLE else '❌'}")
        
        print(f"\n💡 提示:")
        print(f"  - 如果某些模块不可用，智能体会自动降级到基础功能")
        print(f"  - 查看日志了解详细的初始化过程")
        print(f"  - 参考 CORE_MODULES_QUICK_GUIDE.md 了解更多用法")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())