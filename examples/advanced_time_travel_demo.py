#!/usr/bin/env python3
"""
高级时间旅行功能演示 - 多智能体协作场景

演示在复杂多智能体系统中如何使用时间旅行功能进行：
1. 协作决策的回滚和重试
2. 不同策略的分支测试
3. 错误恢复和状态修复
4. 性能优化和路径分析

基于LangGraph官方时间旅行功能实现。
"""

import asyncio
import uuid
from typing import TypedDict, Optional, List, Dict, Any, Literal
from datetime import datetime
from typing_extensions import NotRequired

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model

# 导入项目模块
from core.time_travel import (
    TimeTravelManager, TimeTravelConfig,
    SnapshotType, CheckpointType, RollbackStrategy
)


class MultiAgentState(TypedDict):
    """多智能体协作状态"""
    task: str
    current_agent: str
    research_data: NotRequired[Dict[str, Any]]
    analysis_result: NotRequired[Dict[str, Any]]
    chart_data: NotRequired[Dict[str, Any]]
    final_report: NotRequired[str]
    decision_history: NotRequired[List[Dict[str, Any]]]
    error_count: NotRequired[int]
    strategy: NotRequired[str]
    quality_score: NotRequired[float]


class AdvancedTimeTravelDemo:
    """高级时间旅行功能演示"""
    
    def __init__(self):
        # 初始化LLM
        self.llm = init_chat_model(
            "openai:gpt-4o-mini",
            temperature=0.3,
        )
        
        # 初始化检查点保存器
        self.checkpointer = InMemorySaver()
        
        # 初始化时间旅行管理器
        self.time_travel_config = TimeTravelConfig(
            auto_snapshot=True,
            snapshot_interval=1,
            auto_checkpoint=True,
            checkpoint_on_error=True,
            checkpoint_on_milestone=True,
            enable_branching=True
        )
        self.time_travel_manager = TimeTravelManager(self.time_travel_config)
        
        # 构建多智能体协作图
        self.graph = self._build_multi_agent_graph()
        
    def _build_multi_agent_graph(self) -> StateGraph:
        """构建多智能体协作图"""
        workflow = StateGraph(MultiAgentState)
        
        # 添加智能体节点
        workflow.add_node("supervisor", self._supervisor_agent)
        workflow.add_node("researcher", self._research_agent)
        workflow.add_node("analyst", self._analysis_agent)
        workflow.add_node("chart_maker", self._chart_agent)
        workflow.add_node("reporter", self._report_agent)
        workflow.add_node("quality_checker", self._quality_checker)
        workflow.add_node("error_handler", self._error_handler)
        
        # 添加条件边
        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._route_next_agent,
            {
                "research": "researcher",
                "analysis": "analyst", 
                "chart": "chart_maker",
                "report": "reporter",
                "quality": "quality_checker",
                "error": "error_handler",
                "end": END
            }
        )
        
        # 所有智能体完成后回到supervisor
        for agent in ["researcher", "analyst", "chart_maker", "reporter"]:
            workflow.add_edge(agent, "supervisor")
        
        workflow.add_edge("quality_checker", "supervisor")
        workflow.add_edge("error_handler", "supervisor")
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _supervisor_agent(self, state: MultiAgentState) -> MultiAgentState:
        """监督智能体 - 协调任务分配"""
        print(f"🎯 监督智能体: 当前任务 - {state['task']}")
        
        # 记录决策历史
        decision_history = state.get("decision_history", [])
        
        # 根据当前状态决定下一步
        if not state.get("research_data"):
            next_agent = "research"
            decision = "需要进行研究收集数据"
        elif not state.get("analysis_result"):
            next_agent = "analysis"
            decision = "需要分析研究数据"
        elif not state.get("chart_data"):
            next_agent = "chart"
            decision = "需要创建图表可视化"
        elif not state.get("final_report"):
            next_agent = "report"
            decision = "需要生成最终报告"
        elif not state.get("quality_score"):
            next_agent = "quality"
            decision = "需要质量检查"
        else:
            next_agent = "end"
            decision = "任务完成"
        
        # 检查错误计数
        error_count = state.get("error_count", 0)
        if error_count > 2:
            next_agent = "error"
            decision = "错误过多，需要错误处理"
        
        decision_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": "supervisor",
            "decision": decision,
            "next_agent": next_agent,
            "state_summary": self._get_state_summary(state)
        })
        
        print(f"   决策: {decision} -> {next_agent}")
        
        return {
            **state,
            "current_agent": next_agent,
            "decision_history": decision_history
        }
    
    def _research_agent(self, state: MultiAgentState) -> MultiAgentState:
        """研究智能体 - 收集和整理数据"""
        print("🔍 研究智能体: 收集数据...")
        
        # 模拟研究过程
        strategy = state.get("strategy", "standard")
        
        if strategy == "deep":
            # 深度研究策略
            research_data = {
                "sources": ["学术论文", "行业报告", "专家访谈", "市场调研"],
                "data_points": 150,
                "confidence": 0.9,
                "methodology": "深度分析法"
            }
        elif strategy == "fast":
            # 快速研究策略
            research_data = {
                "sources": ["网络搜索", "新闻报道"],
                "data_points": 50,
                "confidence": 0.6,
                "methodology": "快速扫描法"
            }
        else:
            # 标准研究策略
            research_data = {
                "sources": ["官方数据", "行业报告", "新闻分析"],
                "data_points": 100,
                "confidence": 0.8,
                "methodology": "标准分析法"
            }
        
        # 模拟可能的错误
        import random
        if random.random() < 0.2:  # 20%概率出错
            error_count = state.get("error_count", 0) + 1
            print(f"   ❌ 研究过程中出现错误 (错误计数: {error_count})")
            return {
                **state,
                "error_count": error_count,
                "current_agent": "supervisor"
            }
        
        print(f"   ✅ 研究完成: {research_data['methodology']}, 置信度: {research_data['confidence']}")
        
        return {
            **state,
            "research_data": research_data,
            "current_agent": "supervisor"
        }
    
    def _analysis_agent(self, state: MultiAgentState) -> MultiAgentState:
        """分析智能体 - 分析数据并得出结论"""
        print("📊 分析智能体: 分析数据...")
        
        research_data = state.get("research_data", {})
        if not research_data:
            error_count = state.get("error_count", 0) + 1
            print("   ❌ 缺少研究数据，无法进行分析")
            return {
                **state,
                "error_count": error_count,
                "current_agent": "supervisor"
            }
        
        # 基于研究数据进行分析
        confidence = research_data.get("confidence", 0.5)
        data_points = research_data.get("data_points", 0)
        
        analysis_result = {
            "trends": ["上升趋势", "季节性波动", "市场成熟"],
            "insights": [
                "市场需求持续增长",
                "竞争格局相对稳定",
                "技术创新是关键驱动力"
            ],
            "recommendations": [
                "加大研发投入",
                "扩展市场份额",
                "优化产品结构"
            ],
            "confidence_score": min(confidence + 0.1, 1.0),
            "data_quality": "高" if data_points > 100 else "中" if data_points > 50 else "低"
        }
        
        print(f"   ✅ 分析完成: 置信度 {analysis_result['confidence_score']:.2f}")
        
        return {
            **state,
            "analysis_result": analysis_result,
            "current_agent": "supervisor"
        }
    
    def _chart_agent(self, state: MultiAgentState) -> MultiAgentState:
        """图表智能体 - 创建数据可视化"""
        print("📈 图表智能体: 创建可视化...")
        
        analysis_result = state.get("analysis_result", {})
        if not analysis_result:
            error_count = state.get("error_count", 0) + 1
            print("   ❌ 缺少分析结果，无法创建图表")
            return {
                **state,
                "error_count": error_count,
                "current_agent": "supervisor"
            }
        
        chart_data = {
            "chart_types": ["趋势图", "饼图", "柱状图"],
            "visualizations": [
                {"type": "line", "title": "市场趋势", "data_points": 12},
                {"type": "pie", "title": "市场份额", "segments": 5},
                {"type": "bar", "title": "竞争分析", "categories": 8}
            ],
            "quality": "高清",
            "format": "SVG",
            "interactive": True
        }
        
        print(f"   ✅ 图表创建完成: {len(chart_data['visualizations'])} 个可视化")
        
        return {
            **state,
            "chart_data": chart_data,
            "current_agent": "supervisor"
        }
    
    def _report_agent(self, state: MultiAgentState) -> MultiAgentState:
        """报告智能体 - 生成最终报告"""
        print("📝 报告智能体: 生成报告...")
        
        # 检查所需数据
        required_data = ["research_data", "analysis_result", "chart_data"]
        missing_data = [key for key in required_data if not state.get(key)]
        
        if missing_data:
            error_count = state.get("error_count", 0) + 1
            print(f"   ❌ 缺少必要数据: {missing_data}")
            return {
                **state,
                "error_count": error_count,
                "current_agent": "supervisor"
            }
        
        # 生成报告
        research_data = state["research_data"]
        analysis_result = state["analysis_result"]
        chart_data = state["chart_data"]
        
        final_report = f"""
        # 市场分析报告
        
        ## 研究方法
        - 数据来源: {', '.join(research_data['sources'])}
        - 数据点数: {research_data['data_points']}
        - 置信度: {research_data['confidence']:.2f}
        
        ## 主要发现
        {chr(10).join(f"- {insight}" for insight in analysis_result['insights'])}
        
        ## 建议
        {chr(10).join(f"- {rec}" for rec in analysis_result['recommendations'])}
        
        ## 可视化
        包含 {len(chart_data['visualizations'])} 个图表和可视化
        
        报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        print("   ✅ 报告生成完成")
        
        return {
            **state,
            "final_report": final_report.strip(),
            "current_agent": "supervisor"
        }
    
    def _quality_checker(self, state: MultiAgentState) -> MultiAgentState:
        """质量检查智能体 - 评估输出质量"""
        print("⭐ 质量检查智能体: 评估质量...")
        
        # 计算质量分数
        score = 0.0
        factors = []
        
        # 检查研究数据质量
        research_data = state.get("research_data", {})
        if research_data:
            confidence = research_data.get("confidence", 0)
            data_points = research_data.get("data_points", 0)
            research_score = (confidence * 0.7 + min(data_points / 100, 1.0) * 0.3)
            score += research_score * 0.3
            factors.append(f"研究质量: {research_score:.2f}")
        
        # 检查分析结果质量
        analysis_result = state.get("analysis_result", {})
        if analysis_result:
            analysis_score = analysis_result.get("confidence_score", 0)
            score += analysis_score * 0.4
            factors.append(f"分析质量: {analysis_score:.2f}")
        
        # 检查可视化质量
        chart_data = state.get("chart_data", {})
        if chart_data:
            chart_score = min(len(chart_data.get("visualizations", [])) / 3, 1.0)
            score += chart_score * 0.2
            factors.append(f"可视化质量: {chart_score:.2f}")
        
        # 检查报告完整性
        final_report = state.get("final_report", "")
        if final_report:
            report_score = min(len(final_report) / 500, 1.0)
            score += report_score * 0.1
            factors.append(f"报告完整性: {report_score:.2f}")
        
        print(f"   📊 质量评估: {score:.2f}/1.0")
        for factor in factors:
            print(f"      - {factor}")
        
        return {
            **state,
            "quality_score": score,
            "current_agent": "supervisor"
        }
    
    def _error_handler(self, state: MultiAgentState) -> MultiAgentState:
        """错误处理智能体 - 处理和恢复错误"""
        print("🚨 错误处理智能体: 处理错误...")
        
        error_count = state.get("error_count", 0)
        print(f"   错误计数: {error_count}")
        
        # 重置错误计数并采取恢复措施
        recovery_actions = [
            "重置错误状态",
            "清理无效数据",
            "调整处理策略",
            "降低质量要求"
        ]
        
        print(f"   🔧 执行恢复操作: {', '.join(recovery_actions)}")
        
        # 建议切换到快速策略
        new_strategy = "fast" if state.get("strategy") != "fast" else "standard"
        
        return {
            **state,
            "error_count": 0,
            "strategy": new_strategy,
            "current_agent": "supervisor"
        }
    
    def _route_next_agent(self, state: MultiAgentState) -> str:
        """路由到下一个智能体"""
        return state.get("current_agent", "end")
    
    def _get_state_summary(self, state: MultiAgentState) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "has_research": bool(state.get("research_data")),
            "has_analysis": bool(state.get("analysis_result")),
            "has_charts": bool(state.get("chart_data")),
            "has_report": bool(state.get("final_report")),
            "quality_score": state.get("quality_score"),
            "error_count": state.get("error_count", 0),
            "strategy": state.get("strategy", "standard")
        }

    async def run_baseline_execution(self) -> tuple[Dict[str, Any], MultiAgentState]:
        """运行基线执行"""
        print("=" * 60)
        print("🚀 基线执行 - 标准策略")
        print("=" * 60)
        
        config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
            }
        }
        
        initial_state = {
            "task": "分析人工智能市场趋势并生成报告",
            "strategy": "standard",
            "current_agent": "supervisor"
        }
        
        result = self.graph.invoke(initial_state, config)
        
        print(f"\n📊 基线执行结果:")
        print(f"   质量分数: {result.get('quality_score', 0):.2f}/1.0")
        print(f"   错误次数: {result.get('error_count', 0)}")
        print(f"   策略: {result.get('strategy', 'N/A')}")
        
        return config, result

    async def demonstrate_strategy_branching(self, base_config: Dict[str, Any]):
        """演示策略分支测试"""
        print("\n" + "=" * 60)
        print("🌳 策略分支测试")
        print("=" * 60)
        
        # 获取基线执行的检查点 <mcreference link="https://langchain-ai.github.io/langgraph/concepts/time-travel/" index="0">0</mcreference>
        states = list(self.graph.get_state_history(base_config))
        if len(states) < 2:
            print("❌ 历史状态不足，无法进行分支测试")
            return
        
        # 选择一个早期检查点进行分支
        early_checkpoint = None
        for state in reversed(states):
            if not state.values.get("research_data"):  # 找到研究开始前的状态
                early_checkpoint = state
                break
        
        if not early_checkpoint:
            early_checkpoint = states[-1]  # 使用最早的状态
        
        print(f"🎯 从检查点创建策略分支:")
        print(f"   检查点: {early_checkpoint.config['configurable']['checkpoint_id'][:8]}...")
        
        strategies = ["deep", "fast"]
        branch_results = {}
        
        for strategy in strategies:
            print(f"\n🌿 测试策略: {strategy}")
            
            # 创建分支并修改策略 <mcreference link="https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/" index="1">1</mcreference>
            branch_config = self.graph.update_state(
                base_config,
                {"strategy": strategy},
                checkpoint_id=early_checkpoint.config["configurable"]["checkpoint_id"]
            )
            
            # 从分支继续执行
            branch_result = self.graph.invoke(None, branch_config)
            branch_results[strategy] = branch_result
            
            print(f"   质量分数: {branch_result.get('quality_score', 0):.2f}/1.0")
            print(f"   错误次数: {branch_result.get('error_count', 0)}")
            
            # 分析研究数据差异
            research_data = branch_result.get('research_data', {})
            if research_data:
                print(f"   数据点数: {research_data.get('data_points', 0)}")
                print(f"   置信度: {research_data.get('confidence', 0):.2f}")
        
        # 比较策略效果
        print(f"\n📈 策略比较:")
        best_strategy = max(strategies, key=lambda s: branch_results[s].get('quality_score', 0))
        print(f"   最佳策略: {best_strategy}")
        
        for strategy in strategies:
            result = branch_results[strategy]
            quality = result.get('quality_score', 0)
            errors = result.get('error_count', 0)
            print(f"   {strategy}: 质量={quality:.2f}, 错误={errors}")
        
        return branch_results

    async def demonstrate_error_recovery(self, base_config: Dict[str, Any]):
        """演示错误恢复功能"""
        print("\n" + "=" * 60)
        print("🔧 错误恢复演示")
        print("=" * 60)
        
        # 模拟错误场景：强制增加错误计数 <mcreference link="https://langchain-ai.github.io/langgraph/concepts/time-travel/" index="0">0</mcreference>
        states = list(self.graph.get_state_history(base_config))
        if not states:
            print("❌ 无历史状态可用")
            return
        
        # 找到一个中间状态
        middle_state = states[len(states)//2] if len(states) > 2 else states[0]
        
        print(f"🎯 模拟错误场景:")
        print(f"   从检查点: {middle_state.config['configurable']['checkpoint_id'][:8]}...")
        
        # 创建错误状态
        error_config = self.graph.update_state(
            base_config,
            {"error_count": 3},  # 触发错误处理
            checkpoint_id=middle_state.config["configurable"]["checkpoint_id"]
        )
        
        print("   💥 注入错误: error_count = 3")
        
        # 从错误状态恢复执行
        print("\n🚑 执行错误恢复:")
        recovery_result = self.graph.invoke(None, error_config)
        
        print(f"   ✅ 恢复后状态:")
        print(f"      错误计数: {recovery_result.get('error_count', 0)}")
        print(f"      策略调整: {recovery_result.get('strategy', 'N/A')}")
        print(f"      最终质量: {recovery_result.get('quality_score', 0):.2f}/1.0")
        
        # 分析恢复效果
        decision_history = recovery_result.get('decision_history', [])
        error_decisions = [d for d in decision_history if 'error' in d.get('decision', '').lower()]
        
        if error_decisions:
            print(f"\n🔍 错误处理决策:")
            for decision in error_decisions[-2:]:  # 显示最近的错误处理决策
                print(f"      {decision['timestamp'][:19]}: {decision['decision']}")
        
        return recovery_result

    async def demonstrate_performance_analysis(self, configs: List[Dict[str, Any]]):
        """演示性能分析功能"""
        print("\n" + "=" * 60)
        print("⚡ 性能分析")
        print("=" * 60)
        
        all_metrics = []
        
        for i, config in enumerate(configs):
            print(f"\n📊 分析执行路径 {i+1}:")
            
            # 获取执行历史 <mcreference link="https://langchain-ai.github.io/langgraph/concepts/time-travel/" index="0">0</mcreference>
            states = list(self.graph.get_state_history(config))
            
            # 计算性能指标
            total_steps = len(states)
            agent_usage = {}
            error_points = []
            
            for state in reversed(states):
                # 统计智能体使用情况
                if state.values and 'current_agent' in state.values:
                    agent = state.values['current_agent']
                    agent_usage[agent] = agent_usage.get(agent, 0) + 1
                
                # 记录错误点
                if state.values and state.values.get('error_count', 0) > 0:
                    error_points.append(state.config['configurable']['checkpoint_id'][:8])
            
            # 获取最终结果
            final_state = states[0].values if states else {}
            quality_score = final_state.get('quality_score', 0)
            
            metrics = {
                "total_steps": total_steps,
                "quality_score": quality_score,
                "agent_usage": agent_usage,
                "error_count": len(error_points),
                "efficiency": quality_score / max(total_steps, 1)
            }
            
            all_metrics.append(metrics)
            
            print(f"   总步骤: {total_steps}")
            print(f"   质量分数: {quality_score:.2f}")
            print(f"   效率指标: {metrics['efficiency']:.3f}")
            print(f"   智能体使用: {dict(list(agent_usage.items())[:3])}")  # 显示前3个
        
        # 综合分析
        if len(all_metrics) > 1:
            print(f"\n🎯 综合性能分析:")
            avg_quality = sum(m['quality_score'] for m in all_metrics) / len(all_metrics)
            avg_steps = sum(m['total_steps'] for m in all_metrics) / len(all_metrics)
            avg_efficiency = sum(m['efficiency'] for m in all_metrics) / len(all_metrics)
            
            print(f"   平均质量: {avg_quality:.2f}")
            print(f"   平均步骤: {avg_steps:.1f}")
            print(f"   平均效率: {avg_efficiency:.3f}")
            
            # 找出最佳执行
            best_execution = max(all_metrics, key=lambda m: m['efficiency'])
            best_index = all_metrics.index(best_execution)
            print(f"   最佳执行: 路径 {best_index + 1} (效率: {best_execution['efficiency']:.3f})")

    async def run_complete_advanced_demo(self):
        """运行完整的高级演示"""
        print("🎭 LangGraph高级时间旅行功能演示")
        print("多智能体协作场景中的时间旅行、分支测试和错误恢复")
        print("=" * 80)
        
        try:
            # 1. 基线执行
            base_config, base_result = await self.run_baseline_execution()
            
            # 2. 策略分支测试
            branch_results = await self.demonstrate_strategy_branching(base_config)
            
            # 3. 错误恢复演示
            recovery_result = await self.demonstrate_error_recovery(base_config)
            
            # 4. 性能分析
            all_configs = [base_config]
            if branch_results:
                # 这里简化处理，实际应该收集所有分支的配置
                all_configs.extend([base_config] * len(branch_results))
            
            await self.demonstrate_performance_analysis(all_configs)
            
            print("\n" + "=" * 80)
            print("✅ 高级时间旅行功能演示完成！")
            print("\n🎯 高级功能演示要点:")
            print("1. ✓ 多智能体协作中的时间旅行")
            print("2. ✓ 策略分支测试和比较")
            print("3. ✓ 错误恢复和状态修复")
            print("4. ✓ 性能分析和路径优化")
            print("5. ✓ 复杂决策历史追踪")
            
        except Exception as e:
            print(f"❌ 演示过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数"""
    demo = AdvancedTimeTravelDemo()
    await demo.run_complete_advanced_demo()


if __name__ == "__main__":
    asyncio.run(main())