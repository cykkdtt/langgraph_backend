#!/usr/bin/env python3
"""
LangGraph时间旅行功能演示

基于LangGraph官方文档实现的时间旅行功能演示，包括：
1. 状态快照和检查点管理
2. 时间旅行和状态回滚
3. 分支执行和状态修改
4. 执行历史查询和分析

参考文档：
- https://langchain-ai.github.io/langgraph/concepts/time-travel/
- https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/
"""

import asyncio
import uuid
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
from typing_extensions import NotRequired

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model

# 导入项目的时间旅行模块
from core.time_travel import (
    TimeTravelManager, StateHistoryManager,
    TimeTravelConfig, SnapshotType, CheckpointType
)


class JokeState(TypedDict):
    """笑话生成状态"""
    topic: NotRequired[str]
    joke: NotRequired[str]
    rating: NotRequired[int]
    feedback: NotRequired[str]
    iteration: NotRequired[int]


class TimeTravelDemo:
    """时间旅行功能演示"""
    
    def __init__(self):
        # 初始化LLM
        self.llm = init_chat_model(
            "openai:gpt-4o-mini",
            temperature=0.7,
        )
        
        # 初始化检查点保存器
        self.checkpointer = InMemorySaver()
        
        # 初始化时间旅行管理器
        self.time_travel_config = TimeTravelConfig(
            auto_snapshot=True,
            snapshot_interval=1,  # 每步都创建快照
            auto_checkpoint=True,
            checkpoint_on_milestone=True
        )
        self.time_travel_manager = TimeTravelManager(self.time_travel_config)
        
        # 构建图
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """构建笑话生成工作流图"""
        workflow = StateGraph(JokeState)
        
        # 添加节点
        workflow.add_node("generate_topic", self._generate_topic)
        workflow.add_node("write_joke", self._write_joke)
        workflow.add_node("rate_joke", self._rate_joke)
        workflow.add_node("improve_joke", self._improve_joke)
        
        # 添加边
        workflow.add_edge(START, "generate_topic")
        workflow.add_edge("generate_topic", "write_joke")
        workflow.add_edge("write_joke", "rate_joke")
        workflow.add_edge("rate_joke", "improve_joke")
        workflow.add_edge("improve_joke", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _generate_topic(self, state: JokeState) -> JokeState:
        """生成笑话主题"""
        print("🎯 生成笑话主题...")
        
        msg = self.llm.invoke("给我一个有趣的笑话主题，用中文回答")
        topic = msg.content
        
        print(f"主题: {topic}")
        return {"topic": topic, "iteration": 1}
    
    def _write_joke(self, state: JokeState) -> JokeState:
        """根据主题写笑话"""
        print(f"✍️ 根据主题'{state['topic']}'写笑话...")
        
        prompt = f"根据主题'{state['topic']}'写一个简短有趣的笑话，用中文回答"
        msg = self.llm.invoke(prompt)
        joke = msg.content
        
        print(f"笑话: {joke}")
        return {"joke": joke}
    
    def _rate_joke(self, state: JokeState) -> JokeState:
        """评价笑话质量"""
        print("⭐ 评价笑话质量...")
        
        prompt = f"请对这个笑话进行评分(1-10分)并给出简短评价：\n{state['joke']}\n只返回数字分数和一句话评价，格式：分数|评价"
        msg = self.llm.invoke(prompt)
        response = msg.content
        
        try:
            parts = response.split('|')
            rating = int(parts[0].strip())
            feedback = parts[1].strip() if len(parts) > 1 else "无评价"
        except:
            rating = 5
            feedback = "评价解析失败"
        
        print(f"评分: {rating}/10")
        print(f"评价: {feedback}")
        
        return {"rating": rating, "feedback": feedback}
    
    def _improve_joke(self, state: JokeState) -> JokeState:
        """改进笑话"""
        if state.get("rating", 0) >= 8:
            print("✅ 笑话质量很好，无需改进")
            return state
        
        print("🔧 改进笑话...")
        
        prompt = f"""
        请改进这个笑话，使其更有趣：
        原笑话: {state['joke']}
        评价: {state['feedback']}
        
        请写一个改进版本，用中文回答
        """
        
        msg = self.llm.invoke(prompt)
        improved_joke = msg.content
        
        print(f"改进后的笑话: {improved_joke}")
        
        iteration = state.get("iteration", 1) + 1
        return {"joke": improved_joke, "iteration": iteration}

    async def run_basic_demo(self):
        """运行基础演示"""
        print("=" * 60)
        print("🚀 开始基础笑话生成演示")
        print("=" * 60)
        
        # 创建配置
        config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
            }
        }
        
        # 运行图
        result = self.graph.invoke({}, config)
        
        print("\n📊 最终结果:")
        print(f"主题: {result.get('topic', 'N/A')}")
        print(f"笑话: {result.get('joke', 'N/A')}")
        print(f"评分: {result.get('rating', 'N/A')}/10")
        print(f"迭代次数: {result.get('iteration', 'N/A')}")
        
        return config, result

    async def demonstrate_time_travel(self, config: Dict[str, Any]):
        """演示时间旅行功能"""
        print("\n" + "=" * 60)
        print("⏰ 时间旅行功能演示")
        print("=" * 60)
        
        # 1. 获取执行历史 <mcreference link="https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/" index="1">1</mcreference>
        print("\n📜 获取执行历史:")
        states = list(self.graph.get_state_history(config))
        
        for i, state in enumerate(states):
            print(f"  {i+1}. 检查点: {state.config['configurable']['checkpoint_id'][:8]}...")
            print(f"     下一步: {state.next}")
            if state.values:
                topic = state.values.get('topic', 'N/A')[:30]
                joke = state.values.get('joke', 'N/A')[:50]
                print(f"     主题: {topic}...")
                print(f"     笑话: {joke}...")
            print()
        
        # 2. 选择一个检查点进行时间旅行 <mcreference link="https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/" index="1">1</mcreference>
        if len(states) >= 2:
            selected_state = states[1]  # 选择第二个状态（生成主题后）
            print(f"🎯 选择检查点进行时间旅行:")
            print(f"   检查点ID: {selected_state.config['configurable']['checkpoint_id']}")
            print(f"   下一步: {selected_state.next}")
            print(f"   状态: {selected_state.values}")
            
            # 3. 修改状态并从检查点恢复 <mcreference link="https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/" index="1">1</mcreference>
            print("\n🔄 修改状态并从检查点恢复:")
            new_config = self.graph.update_state(
                config,
                {"topic": "程序员和Bug的爱恨情仇"},
                checkpoint_id=selected_state.config["configurable"]["checkpoint_id"]
            )
            
            print(f"   新检查点ID: {new_config['configurable']['checkpoint_id']}")
            
            # 4. 从修改后的状态继续执行 <mcreference link="https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/" index="1">1</mcreference>
            print("\n▶️ 从修改后的状态继续执行:")
            alternative_result = self.graph.invoke(
                None,  # 输入为None表示从检查点继续
                new_config
            )
            
            print("\n📊 替代时间线的结果:")
            print(f"主题: {alternative_result.get('topic', 'N/A')}")
            print(f"笑话: {alternative_result.get('joke', 'N/A')}")
            print(f"评分: {alternative_result.get('rating', 'N/A')}/10")
            
            # 5. 比较两个时间线 <mcreference link="https://langchain-ai.github.io/langgraph/concepts/time-travel/" index="0">0</mcreference>
            print("\n🔍 时间线比较:")
            print("原始时间线 vs 替代时间线")
            print("-" * 40)
            
            # 获取两个时间线的最终状态
            original_states = list(self.graph.get_state_history(config))
            alternative_states = list(self.graph.get_state_history(new_config))
            
            if original_states and alternative_states:
                orig_final = original_states[0].values
                alt_final = alternative_states[0].values
                
                print(f"主题: '{orig_final.get('topic', 'N/A')}' -> '{alt_final.get('topic', 'N/A')}'")
                print(f"评分: {orig_final.get('rating', 'N/A')} -> {alt_final.get('rating', 'N/A')}")
                
            return new_config, alternative_result
        
        return None, None

    async def demonstrate_branching(self, config: Dict[str, Any]):
        """演示分支功能"""
        print("\n" + "=" * 60)
        print("🌳 分支执行演示")
        print("=" * 60)
        
        # 获取历史状态
        states = list(self.graph.get_state_history(config))
        if len(states) < 2:
            print("❌ 历史状态不足，无法演示分支功能")
            return
        
        # 从同一个检查点创建多个分支 <mcreference link="https://langchain-ai.github.io/langgraph/concepts/time-travel/" index="0">0</mcreference>
        base_state = states[1]  # 选择基础状态
        
        print(f"🎯 从检查点创建多个分支:")
        print(f"   基础检查点: {base_state.config['configurable']['checkpoint_id'][:8]}...")
        
        # 创建分支1：科技主题
        branch1_config = self.graph.update_state(
            config,
            {"topic": "人工智能的日常生活"},
            checkpoint_id=base_state.config["configurable"]["checkpoint_id"]
        )
        
        # 创建分支2：生活主题
        branch2_config = self.graph.update_state(
            config,
            {"topic": "减肥路上的奇遇"},
            checkpoint_id=base_state.config["configurable"]["checkpoint_id"]
        )
        
        print("\n🌿 分支1 - 科技主题:")
        branch1_result = self.graph.invoke(None, branch1_config)
        print(f"   主题: {branch1_result.get('topic', 'N/A')}")
        print(f"   笑话: {branch1_result.get('joke', 'N/A')[:100]}...")
        print(f"   评分: {branch1_result.get('rating', 'N/A')}/10")
        
        print("\n🌿 分支2 - 生活主题:")
        branch2_result = self.graph.invoke(None, branch2_config)
        print(f"   主题: {branch2_result.get('topic', 'N/A')}")
        print(f"   笑话: {branch2_result.get('joke', 'N/A')[:100]}...")
        print(f"   评分: {branch2_result.get('rating', 'N/A')}/10")
        
        # 分析分支结果
        print("\n📈 分支分析:")
        ratings = [
            branch1_result.get('rating', 0),
            branch2_result.get('rating', 0)
        ]
        best_branch = 1 if ratings[0] > ratings[1] else 2
        print(f"   最佳分支: 分支{best_branch} (评分: {max(ratings)}/10)")
        
        return [branch1_config, branch2_config], [branch1_result, branch2_result]

    async def demonstrate_debugging(self, config: Dict[str, Any]):
        """演示调试功能"""
        print("\n" + "=" * 60)
        print("🐞 调试功能演示")
        print("=" * 60)
        
        # 分析执行路径 <mcreference link="https://langchain-ai.github.io/langgraph/concepts/time-travel/" index="0">0</mcreference>
        print("🔍 分析执行路径:")
        states = list(self.graph.get_state_history(config))
        
        for i, state in enumerate(reversed(states)):
            step_num = i + 1
            checkpoint_id = state.config['configurable']['checkpoint_id'][:8]
            next_step = state.next[0] if state.next else "END"
            
            print(f"   步骤 {step_num}: {next_step} (检查点: {checkpoint_id}...)")
            
            # 显示关键状态变化
            if state.values:
                if 'topic' in state.values and i == 1:
                    print(f"      ✓ 生成主题: {state.values['topic'][:50]}...")
                elif 'joke' in state.values and i == 2:
                    print(f"      ✓ 生成笑话: {state.values['joke'][:50]}...")
                elif 'rating' in state.values and i == 3:
                    print(f"      ✓ 评分: {state.values['rating']}/10")
        
        # 性能分析
        print("\n⚡ 性能分析:")
        if len(states) > 1:
            total_steps = len(states)
            print(f"   总步骤数: {total_steps}")
            print(f"   平均每步耗时: ~0.5秒 (模拟)")
            
            # 找出可能的优化点
            final_state = states[0].values
            if final_state.get('rating', 0) < 7:
                print("   💡 优化建议: 笑话质量较低，可能需要改进提示词")
            if final_state.get('iteration', 1) > 2:
                print("   💡 优化建议: 迭代次数较多，考虑优化初始生成质量")

    async def run_complete_demo(self):
        """运行完整演示"""
        print("🎭 LangGraph时间旅行功能完整演示")
        print("基于官方文档实现的时间旅行、分支和调试功能")
        print("=" * 80)
        
        try:
            # 1. 基础演示
            config, result = await self.run_basic_demo()
            
            # 2. 时间旅行演示
            alt_config, alt_result = await self.demonstrate_time_travel(config)
            
            # 3. 分支演示
            if alt_config:
                branch_configs, branch_results = await self.demonstrate_branching(alt_config)
            
            # 4. 调试演示
            await self.demonstrate_debugging(config)
            
            print("\n" + "=" * 80)
            print("✅ 时间旅行功能演示完成！")
            print("\n🎯 演示要点总结:")
            print("1. ✓ 状态快照和检查点管理")
            print("2. ✓ 时间旅行和状态回滚")
            print("3. ✓ 分支执行和状态修改")
            print("4. ✓ 执行历史查询和调试")
            print("\n📚 参考文档:")
            print("- LangGraph时间旅行概念: https://langchain-ai.github.io/langgraph/concepts/time-travel/")
            print("- 时间旅行使用指南: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/")
            
        except Exception as e:
            print(f"❌ 演示过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数"""
    demo = TimeTravelDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())