"""
LangMem 记忆增强智能体示例

本示例展示如何使用记忆增强的智能体，包括：
- 创建记忆增强的智能体
- 自动存储和检索对话记忆
- 知识记忆管理
- 记忆统计和清理
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from core.agents.memory_enhanced import MemoryEnhancedAgent
from core.agents.base import ChatRequest, AgentType
from core.memory import MemoryType, MemoryScope
from config.settings import get_settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryDemoAgent(MemoryEnhancedAgent):
    """记忆演示智能体
    
    专门用于演示记忆功能的智能体
    """
    
    def __init__(self):
        settings = get_settings()
        
        # 初始化语言模型
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=settings.openai.api_key
        )
        
        # 记忆配置
        memory_config = {
            "auto_store": True,
            "retrieval_limit": 5,
            "importance_threshold": 0.3
        }
        
        super().__init__(
            agent_id="memory_demo_agent",
            name="记忆演示智能体",
            description="一个展示LangMem记忆功能的智能体，能够记住用户的偏好、历史对话和学习内容",
            llm=llm,
            agent_type=AgentType.CHAT,
            memory_config=memory_config
        )
    
    async def demonstrate_memory_features(self, user_id: str = "demo_user"):
        """演示记忆功能"""
        session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n🧠 LangMem 记忆增强智能体演示")
        print(f"用户ID: {user_id}")
        print(f"会话ID: {session_id}")
        print("=" * 60)
        
        # 1. 基础对话 - 建立记忆
        print("\n📝 第一轮对话 - 建立记忆")
        await self._demo_conversation(
            user_id, session_id,
            "你好！我叫张三，我是一名软件工程师，喜欢Python编程和机器学习。"
        )
        
        # 2. 存储专业知识
        print("\n📚 存储专业知识")
        await self.store_knowledge(
            content="Python中的装饰器是一种设计模式，允许在不修改函数代码的情况下扩展函数功能。常用的装饰器包括@property、@staticmethod、@classmethod等。",
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            metadata={"topic": "Python编程", "category": "技术知识"},
            importance=0.8
        )
        print("✅ 已存储Python装饰器相关知识")
        
        # 3. 存储学习经历
        await self.store_knowledge(
            content="用户张三在2024年完成了深度学习课程，掌握了TensorFlow和PyTorch框架，并成功实现了一个图像分类项目。",
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            metadata={"topic": "学习经历", "year": "2024"},
            importance=0.7
        )
        print("✅ 已存储学习经历")
        
        # 4. 存储技能信息
        await self.store_knowledge(
            content="张三擅长使用Python进行数据分析，熟练掌握pandas、numpy、matplotlib等库，能够进行数据清洗、可视化和统计分析。",
            user_id=user_id,
            memory_type=MemoryType.PROCEDURAL,
            metadata={"topic": "技能", "skill_type": "数据分析"},
            importance=0.8
        )
        print("✅ 已存储技能信息")
        
        # 5. 等待一下让记忆系统处理
        await asyncio.sleep(1)
        
        # 6. 测试记忆检索 - 相关话题
        print("\n🔍 第二轮对话 - 测试记忆检索")
        await self._demo_conversation(
            user_id, session_id,
            "我想学习更多关于Python装饰器的内容，你能给我一些建议吗？"
        )
        
        # 7. 测试个人信息记忆
        print("\n👤 第三轮对话 - 测试个人信息记忆")
        await self._demo_conversation(
            user_id, session_id,
            "根据我的背景，你觉得我应该学习哪些新技术？"
        )
        
        # 8. 显示记忆统计
        print("\n📊 记忆统计信息")
        stats = await self.get_memory_stats(user_id, session_id)
        self._print_memory_stats(stats)
        
        # 9. 演示记忆清理（可选）
        print("\n🧹 记忆清理演示")
        cleanup_result = await self.cleanup_old_memories(
            user_id, session_id, 
            days=0,  # 立即清理（仅用于演示）
            min_importance=0.9  # 只清理重要性很低的记忆
        )
        print(f"清理结果: {cleanup_result}")
    
    async def _demo_conversation(self, user_id: str, session_id: str, user_message: str):
        """演示对话"""
        print(f"\n用户: {user_message}")
        
        # 创建对话请求
        request = ChatRequest(
            messages=[HumanMessage(content=user_message)],
            user_id=user_id,
            session_id=session_id,
            stream=False
        )
        
        # 处理对话
        response = await self.chat(request)
        
        if response.message:
            print(f"智能体: {response.message.content}")
        else:
            print("智能体: [无响应]")
    
    def _print_memory_stats(self, stats: Dict[str, Any]):
        """打印记忆统计信息"""
        print(f"总记忆数量: {stats.get('total_memories', 0)}")
        
        user_memories = stats.get('user_memories', {})
        if user_memories:
            print(f"\n用户记忆:")
            print(f"  总数: {user_memories.get('total_count', 0)}")
            by_type = user_memories.get('by_type', {})
            for memory_type, count in by_type.items():
                print(f"  {memory_type}: {count}")
        
        session_memories = stats.get('session_memories', {})
        if session_memories:
            print(f"\n会话记忆:")
            print(f"  总数: {session_memories.get('total_count', 0)}")
            by_type = session_memories.get('by_type', {})
            for memory_type, count in by_type.items():
                print(f"  {memory_type}: {count}")


async def interactive_demo():
    """交互式演示"""
    print("\n🎯 LangMem 交互式演示")
    print("输入 'quit' 退出演示")
    print("=" * 60)
    
    # 创建智能体
    agent = MemoryDemoAgent()
    await agent.initialize()
    
    user_id = "interactive_user"
    session_id = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"用户ID: {user_id}")
    print(f"会话ID: {session_id}")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
            
            if not user_input:
                continue
            
            # 特殊命令处理
            if user_input.startswith('/'):
                await handle_special_command(agent, user_input, user_id, session_id)
                continue
            
            # 创建对话请求
            request = ChatRequest(
                messages=[HumanMessage(content=user_input)],
                user_id=user_id,
                session_id=session_id,
                stream=False
            )
            
            # 处理对话
            response = await agent.chat(request)
            
            if response.message:
                print(f"智能体: {response.message.content}")
            else:
                print("智能体: [无响应]")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")
    
    print("\n👋 演示结束")


async def handle_special_command(agent, command: str, user_id: str, session_id: str):
    """处理特殊命令"""
    if command == '/stats':
        # 显示记忆统计
        stats = await agent.get_memory_stats(user_id, session_id)
        agent._print_memory_stats(stats)
    
    elif command.startswith('/store '):
        # 存储知识
        content = command[7:]  # 移除 '/store '
        memory_id = await agent.store_knowledge(
            content=content,
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            importance=0.7
        )
        print(f"✅ 已存储知识，记忆ID: {memory_id}")
    
    elif command == '/cleanup':
        # 清理记忆
        result = await agent.cleanup_old_memories(user_id, session_id, days=30, min_importance=0.2)
        print(f"🧹 清理完成: {result}")
    
    elif command == '/help':
        # 显示帮助
        print("\n可用命令:")
        print("/stats - 显示记忆统计")
        print("/store <内容> - 存储知识")
        print("/cleanup - 清理旧记忆")
        print("/help - 显示帮助")
        print("quit - 退出演示")
    
    else:
        print("未知命令，输入 /help 查看可用命令")


async def main():
    """主函数"""
    print("🚀 LangMem 记忆增强智能体演示程序")
    print("\n选择演示模式:")
    print("1. 自动演示 - 展示所有记忆功能")
    print("2. 交互式演示 - 与智能体对话")
    
    while True:
        choice = input("\n请选择 (1/2): ").strip()
        
        if choice == '1':
            # 自动演示
            agent = MemoryDemoAgent()
            await agent.initialize()
            await agent.demonstrate_memory_features()
            break
        
        elif choice == '2':
            # 交互式演示
            await interactive_demo()
            break
        
        else:
            print("无效选择，请输入 1 或 2")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
        print(f"\n❌ 程序运行出错: {e}")