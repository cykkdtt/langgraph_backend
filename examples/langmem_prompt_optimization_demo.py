#!/usr/bin/env python3
"""
LangMem 提示词优化演示

展示如何使用LangMem的提示词优化功能来改进智能体的系统提示词。
包括：
1. 单个智能体提示词优化
2. 多智能体系统协同优化
3. 基于用户反馈的持续改进
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 检查是否安装了langmem
try:
    from langmem import create_prompt_optimizer, create_multi_prompt_optimizer
    LANGMEM_AVAILABLE = True
except ImportError:
    LANGMEM_AVAILABLE = False
    print("⚠️  LangMem未安装，将使用模拟演示")


class PromptOptimizationDemo:
    """提示词优化演示类"""
    
    def __init__(self):
        self.model_name = "anthropic:claude-3-5-sonnet-latest"
        
    async def demo_single_agent_optimization(self):
        """演示单个智能体的提示词优化"""
        print("\n" + "="*60)
        print("🎯 单个智能体提示词优化演示")
        print("="*60)
        
        # 模拟对话历史和反馈
        trajectories = [
            # 对话1：用户不满意回答的详细程度
            (
                [
                    {"role": "user", "content": "解释一下机器学习"},
                    {"role": "assistant", "content": "机器学习是人工智能的一个分支，让计算机从数据中学习。"},
                    {"role": "user", "content": "太简单了，我需要更详细的解释"}
                ],
                {"feedback": "回答太简单，需要更详细的技术解释"}
            ),
            
            # 对话2：用户满意的回答
            (
                [
                    {"role": "user", "content": "什么是深度学习？"},
                    {"role": "assistant", "content": "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。它包括以下关键概念：\n1. 神经网络架构\n2. 反向传播算法\n3. 梯度下降优化\n4. 激活函数的作用\n\n应用领域包括图像识别、自然语言处理、语音识别等。"},
                ],
                {"score": 0.9, "comment": "详细且结构化，很好的解释"}
            ),
            
            # 对话3：需要更多实例
            (
                [
                    {"role": "user", "content": "神经网络是怎么工作的？"},
                    {"role": "assistant", "content": "神经网络通过层层连接的节点处理信息，每个连接都有权重。"},
                    {"role": "user", "content": "能给个具体例子吗？"}
                ],
                {"feedback": "需要具体的例子来说明概念"}
            )
        ]
        
        # 原始提示词
        original_prompt = "你是一个AI助手，帮助用户解答问题。"
        
        print(f"📝 原始提示词：\n{original_prompt}\n")
        
        if LANGMEM_AVAILABLE:
            # 使用gradient策略优化
            optimizer = create_prompt_optimizer(
                self.model_name,
                kind="gradient",
                config={
                    "max_reflection_steps": 2,
                    "min_reflection_steps": 1
                }
            )
            
            try:
                optimized_prompt = await optimizer.ainvoke({
                    "trajectories": trajectories,
                    "prompt": original_prompt
                })
                
                print(f"✨ 优化后的提示词：\n{optimized_prompt}\n")
                
            except Exception as e:
                print(f"❌ 优化失败: {e}")
                self._show_mock_optimization()
        else:
            self._show_mock_optimization()
    
    def _show_mock_optimization(self):
        """显示模拟的优化结果"""
        mock_optimized = """你是一个专业的AI技术助手，专门帮助用户深入理解技术概念。在回答时请遵循以下原则：

1. **详细解释**：提供全面、深入的技术解释，不要过于简化
2. **结构化回答**：使用清晰的结构，包括要点列表、步骤说明等
3. **具体示例**：总是包含具体的例子来说明抽象概念
4. **渐进式解释**：从基础概念开始，逐步深入到技术细节
5. **实际应用**：说明概念在实际中的应用场景

确保你的回答既有技术深度，又易于理解。"""
        
        print(f"✨ 模拟优化后的提示词：\n{mock_optimized}\n")
    
    async def demo_multi_agent_optimization(self):
        """演示多智能体系统的协同优化"""
        print("\n" + "="*60)
        print("🤝 多智能体系统协同优化演示")
        print("="*60)
        
        # 定义多个智能体的提示词
        agent_prompts = [
            {
                "name": "researcher",
                "prompt": "你是一个研究员，负责收集和分析技术信息。"
            },
            {
                "name": "writer",
                "prompt": "你是一个技术写作专家，负责将研究结果写成清晰的报告。"
            },
            {
                "name": "reviewer",
                "prompt": "你是一个质量审核员，负责检查报告的准确性和完整性。"
            }
        ]
        
        # 模拟团队协作的对话历史
        team_conversations = [
            # 协作案例1：缺少技术细节
            (
                [
                    {"role": "user", "content": "研究一下最新的Transformer架构"},
                    {"role": "assistant", "content": "找到了一些关于Transformer的基本信息..."},  # researcher
                    {"role": "assistant", "content": "基于研究，Transformer是一种注意力机制..."},  # writer
                    {"role": "assistant", "content": "报告缺少具体的技术实现细节"},  # reviewer
                    {"role": "user", "content": "确实需要更多技术细节"}
                ],
                {"feedback": "研究不够深入，写作缺少技术细节"}
            ),
            
            # 协作案例2：成功的协作
            (
                [
                    {"role": "user", "content": "分析BERT模型的创新点"},
                    {"role": "assistant", "content": "深入分析了BERT的双向编码、预训练策略、微调方法..."},  # researcher
                    {"role": "assistant", "content": "基于详细研究，撰写了包含架构图、算法流程、性能对比的完整报告..."},  # writer
                    {"role": "assistant", "content": "报告结构清晰，技术细节准确，建议发布"},  # reviewer
                ],
                {"score": 0.95, "comment": "团队协作完美，报告质量很高"}
            )
        ]
        
        print("👥 原始智能体提示词：")
        for agent in agent_prompts:
            print(f"  {agent['name']}: {agent['prompt']}")
        
        if LANGMEM_AVAILABLE:
            # 创建多智能体优化器
            multi_optimizer = create_multi_prompt_optimizer(
                self.model_name,
                kind="gradient",
                config={"max_reflection_steps": 2}
            )
            
            try:
                optimized_prompts = await multi_optimizer.ainvoke({
                    "trajectories": team_conversations,
                    "prompts": agent_prompts
                })
                
                print("\n✨ 优化后的智能体提示词：")
                for prompt_info in optimized_prompts:
                    print(f"\n{prompt_info['name']}:")
                    print(f"  {prompt_info['prompt']}")
                    
            except Exception as e:
                print(f"❌ 多智能体优化失败: {e}")
                self._show_mock_multi_optimization()
        else:
            self._show_mock_multi_optimization()
    
    def _show_mock_multi_optimization(self):
        """显示模拟的多智能体优化结果"""
        mock_optimized_agents = [
            {
                "name": "researcher",
                "prompt": """你是一个深度技术研究员，专门负责收集和分析前沿技术信息。在研究时请：
1. 深入挖掘技术细节和实现原理
2. 收集最新的论文、代码和实验数据
3. 分析技术的创新点和局限性
4. 提供详细的技术架构和算法流程
5. 包含性能基准和对比分析"""
            },
            {
                "name": "writer", 
                "prompt": """你是一个技术写作专家，负责将复杂的研究结果转化为清晰、结构化的技术报告。写作时请：
1. 使用清晰的层次结构组织内容
2. 包含技术架构图和流程图
3. 提供具体的代码示例和实现细节
4. 添加性能数据和对比表格
5. 确保技术准确性和可读性的平衡"""
            },
            {
                "name": "reviewer",
                "prompt": """你是一个严格的技术质量审核员，负责确保报告的准确性和完整性。审核时请：
1. 验证技术细节的准确性
2. 检查是否遗漏重要的技术要点
3. 确认代码示例的正确性
4. 评估报告的逻辑结构和可读性
5. 提供具体的改进建议"""
            }
        ]
        
        print("\n✨ 模拟优化后的智能体提示词：")
        for agent in mock_optimized_agents:
            print(f"\n{agent['name']}:")
            print(f"  {agent['prompt']}")
    
    async def demo_continuous_improvement(self):
        """演示基于用户反馈的持续改进"""
        print("\n" + "="*60)
        print("🔄 持续改进演示")
        print("="*60)
        
        print("📊 模拟持续改进流程：")
        
        improvement_steps = [
            {
                "step": 1,
                "description": "收集用户反馈",
                "data": "用户反馈：回答太技术化，需要更通俗易懂"
            },
            {
                "step": 2, 
                "description": "分析反馈模式",
                "data": "发现模式：80%用户希望更简单的解释"
            },
            {
                "step": 3,
                "description": "优化提示词",
                "data": "添加：'用通俗易懂的语言解释，避免过多专业术语'"
            },
            {
                "step": 4,
                "description": "测试新提示词",
                "data": "A/B测试：新提示词满意度提升25%"
            },
            {
                "step": 5,
                "description": "部署改进版本",
                "data": "正式部署，继续收集反馈"
            }
        ]
        
        for step in improvement_steps:
            print(f"\n步骤 {step['step']}: {step['description']}")
            print(f"  📝 {step['data']}")
            await asyncio.sleep(0.5)  # 模拟处理时间
        
        print("\n✅ 持续改进循环建立完成！")
    
    def show_integration_suggestions(self):
        """显示集成建议"""
        print("\n" + "="*60)
        print("💡 项目集成建议")
        print("="*60)
        
        suggestions = [
            {
                "area": "智能体系统",
                "suggestion": "为每个智能体类型建立提示词优化流程",
                "implementation": "在core/agents/中添加prompt_optimizer模块"
            },
            {
                "area": "用户反馈收集",
                "suggestion": "在API响应中添加反馈收集机制",
                "implementation": "扩展models/chat_models.py添加反馈字段"
            },
            {
                "area": "A/B测试",
                "suggestion": "实现提示词版本管理和A/B测试",
                "implementation": "在core/experiments/中添加ab_testing模块"
            },
            {
                "area": "监控指标",
                "suggestion": "跟踪提示词性能指标",
                "implementation": "扩展core/logging/添加prompt_metrics"
            },
            {
                "area": "自动化优化",
                "suggestion": "定期自动优化提示词",
                "implementation": "在scripts/中添加auto_optimize_prompts.py"
            }
        ]
        
        for suggestion in suggestions:
            print(f"\n🎯 {suggestion['area']}:")
            print(f"  建议: {suggestion['suggestion']}")
            print(f"  实现: {suggestion['implementation']}")


async def main():
    """主函数"""
    print("🚀 LangMem 提示词优化功能演示")
    print("展示如何使用LangMem自动改进智能体的提示词")
    
    if not LANGMEM_AVAILABLE:
        print("\n⚠️  注意：LangMem未安装，使用模拟演示")
        print("安装命令：pip install langmem")
    
    demo = PromptOptimizationDemo()
    
    # 演示各种优化功能
    await demo.demo_single_agent_optimization()
    await demo.demo_multi_agent_optimization()
    await demo.demo_continuous_improvement()
    
    # 显示集成建议
    demo.show_integration_suggestions()
    
    print("\n" + "="*60)
    print("🎉 提示词优化演示完成！")
    print("这个功能可以显著提升智能体系统的性能")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())