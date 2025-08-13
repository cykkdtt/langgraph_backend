"""
LangMem 提示词优化集成方案

这个模块展示了如何在现有项目中集成LangMem的提示词优化功能，
包括与现有智能体系统的集成、反馈收集机制和自动优化流程。
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path

# 项目现有模块
from core.memory.store_manager import MemoryStoreManager
from core.agents.base import BaseAgent
from models.chat_models import Message, ChatResponse

# LangMem 提示词优化 (需要安装)
try:
    from langmem import create_prompt_optimizer, create_multi_prompt_optimizer
    LANGMEM_AVAILABLE = True
except ImportError:
    LANGMEM_AVAILABLE = False


class PromptOptimizer:
    """提示词优化器 - 集成LangMem的提示词优化功能"""
    
    def __init__(self, memory_manager: MemoryStoreManager, model_name: str = "anthropic:claude-3-5-sonnet-latest"):
        self.memory_manager = memory_manager
        self.model_name = model_name
        self.feedback_namespace = "prompt_feedback"
        self.optimization_namespace = "prompt_optimization"
        self.initialized = False
        
        # 初始化优化器
        if LANGMEM_AVAILABLE:
            self.single_optimizer = create_prompt_optimizer(
                model_name,
                kind="gradient",
                config={"max_reflection_steps": 2}
            )
            self.multi_optimizer = create_multi_prompt_optimizer(
                model_name,
                kind="gradient",
                config={"max_reflection_steps": 2}
            )
        else:
            self.single_optimizer = None
            self.multi_optimizer = None
    
    async def initialize(self):
        """初始化优化器"""
        if not self.initialized:
            await self.memory_manager.initialize()
            self.initialized = True
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy" if LANGMEM_AVAILABLE else "limited",
            "langmem_available": LANGMEM_AVAILABLE,
            "initialized": self.initialized,
            "model_name": self.model_name
        }
    
    async def collect_feedback(self, 
                             conversation_id: str,
                             messages: List[Message],
                             feedback: Dict[str, Any]) -> None:
        """收集用户反馈用于提示词优化"""
        
        feedback_data = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": [msg.dict() for msg in messages],
            "feedback": feedback,
            "agent_type": feedback.get("agent_type", "unknown")
        }
        
        # 存储到记忆系统
        await self.memory_manager.store_memory(
            content=json.dumps(feedback_data),
            memory_type="episodic",
            namespace=self.feedback_namespace,
            metadata={
                "conversation_id": conversation_id,
                "agent_type": feedback.get("agent_type"),
                "satisfaction_score": feedback.get("score", 0),
                "feedback_type": "user_feedback"
            }
        )
    
    async def optimize_agent_prompt(self, 
                                  agent_type: str,
                                  user_id: Optional[str] = None,
                                  strategy: str = "gradient",
                                  max_iterations: int = 5,
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """优化单个智能体的提示词"""
        
        if not LANGMEM_AVAILABLE:
            raise Exception("LangMem未安装，无法执行提示词优化")
        
        # 获取当前提示词（这里需要从智能体配置中获取）
        current_prompt = context.get("current_prompt", f"你是一个{agent_type}智能体。")
        
        # 获取该智能体类型的反馈数据
        query_filter = {"agent_type": agent_type}
        if user_id:
            query_filter["user_id"] = user_id
            
        feedback_memories = await self.memory_manager.search_memories(
            query=f"agent_type:{agent_type}",
            namespace=self.feedback_namespace,
            limit=50
        )
        
        min_feedback_count = context.get("min_feedback_count", 10)
        if len(feedback_memories) < min_feedback_count:
            raise Exception(f"反馈数据不足 ({len(feedback_memories)}/{min_feedback_count})")
        
        # 转换为优化器需要的格式
        trajectories = []
        for memory in feedback_memories:
            try:
                data = json.loads(memory.content)
                messages = data["messages"]
                feedback = data["feedback"]
                
                # 构建对话轨迹
                trajectory = (messages, feedback)
                trajectories.append(trajectory)
                
            except (json.JSONDecodeError, KeyError) as e:
                continue
        
        if not trajectories:
            raise Exception("没有有效的反馈数据")
        
        try:
            # 执行提示词优化
            optimized_prompt = await self.single_optimizer.ainvoke({
                "trajectories": trajectories,
                "prompt": current_prompt
            })
            
            # 计算改进分数（简化版本）
            improvement_score = len(trajectories) * 0.1  # 简化计算
            
            # 生成优化ID
            optimization_id = f"opt_{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 保存优化结果
            await self._save_optimization_result(
                agent_type, current_prompt, optimized_prompt, trajectories, optimization_id
            )
            
            return {
                "optimization_id": optimization_id,
                "original_prompt": current_prompt,
                "optimized_prompt": optimized_prompt,
                "improvement_score": improvement_score,
                "details": {
                    "strategy": strategy,
                    "feedback_count": len(trajectories),
                    "iterations": max_iterations,
                    "agent_type": agent_type,
                    "user_id": user_id
                }
            }
            
        except Exception as e:
            raise Exception(f"提示词优化失败: {e}")
    
    async def optimize_multi_agent_system(self, 
                                        agent_types: List[str],
                                        user_id: Optional[str] = None,
                                        strategy: str = "gradient",
                                        max_iterations: int = 5,
                                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """优化多智能体系统的协同提示词"""
        
        if not LANGMEM_AVAILABLE:
            raise Exception("LangMem未安装，无法执行多智能体优化")
        
        # 构建智能体提示词列表
        agent_prompts = []
        for agent_type in agent_types:
            prompt = context.get(f"{agent_type}_prompt", f"你是一个{agent_type}智能体。")
            agent_prompts.append({"agent_type": agent_type, "prompt": prompt})
        
        # 获取多智能体协作的反馈数据
        feedback_memories = await self.memory_manager.search_memories(
            query="multi_agent OR collaboration OR team",
            namespace=self.feedback_namespace,
            limit=100
        )
        
        min_feedback_count = context.get("min_feedback_count", 20)
        if len(feedback_memories) < min_feedback_count:
            raise Exception(f"多智能体反馈数据不足 ({len(feedback_memories)}/{min_feedback_count})")
        
        # 构建团队对话轨迹
        team_trajectories = []
        for memory in feedback_memories:
            try:
                data = json.loads(memory.content)
                if data.get("feedback", {}).get("type") == "multi_agent":
                    messages = data["messages"]
                    feedback = data["feedback"]
                    team_trajectories.append((messages, feedback))
            except (json.JSONDecodeError, KeyError):
                continue
        
        if not team_trajectories:
            raise Exception("没有有效的多智能体反馈数据")
        
        try:
            # 执行多智能体优化
            optimized_prompts = await self.multi_optimizer.ainvoke({
                "trajectories": team_trajectories,
                "prompts": agent_prompts
            })
            
            # 计算整体改进分数
            overall_improvement = len(team_trajectories) * 0.15
            
            # 生成优化ID
            optimization_id = f"multi_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 保存优化结果
            await self._save_multi_optimization_result(
                agent_prompts, optimized_prompts, team_trajectories, optimization_id
            )
            
            return {
                "optimization_id": optimization_id,
                "optimized_agents": optimized_prompts,
                "overall_improvement": overall_improvement,
                "details": {
                    "strategy": strategy,
                    "feedback_count": len(team_trajectories),
                    "iterations": max_iterations,
                    "agent_types": agent_types,
                    "user_id": user_id
                }
            }
            
        except Exception as e:
            raise Exception(f"多智能体优化失败: {e}")
    
    async def _save_optimization_result(self,
                                      agent_type: str,
                                      original_prompt: str,
                                      optimized_prompt: str,
                                      trajectories: List[Tuple],
                                      optimization_id: str) -> None:
        """保存单智能体优化结果"""
        
        result_data = {
            "optimization_id": optimization_id,
            "agent_type": agent_type,
            "timestamp": datetime.now().isoformat(),
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "feedback_count": len(trajectories),
            "optimization_type": "single_agent"
        }
        
        await self.memory_manager.store_memory(
            content=json.dumps(result_data),
            memory_type="procedural",
            namespace=self.optimization_namespace,
            metadata={
                "optimization_id": optimization_id,
                "agent_type": agent_type,
                "optimization_type": "single_agent",
                "feedback_count": len(trajectories)
            }
        )
    
    async def _save_multi_optimization_result(self,
                                            original_prompts: List[Dict[str, str]],
                                            optimized_prompts: List[Dict[str, str]],
                                            trajectories: List[Tuple],
                                            optimization_id: str) -> None:
        """保存多智能体优化结果"""
        
        result_data = {
            "optimization_id": optimization_id,
            "timestamp": datetime.now().isoformat(),
            "original_prompts": original_prompts,
            "optimized_prompts": optimized_prompts,
            "feedback_count": len(trajectories),
            "optimization_type": "multi_agent"
        }
        
        await self.memory_manager.store_memory(
            content=json.dumps(result_data),
            memory_type="procedural",
            namespace=self.optimization_namespace,
            metadata={
                "optimization_id": optimization_id,
                "optimization_type": "multi_agent",
                "agent_count": len(original_prompts),
                "feedback_count": len(trajectories)
            }
        )
    
    async def get_optimization_history(self, 
                                     filter_dict: Optional[Dict[str, Any]] = None,
                                     limit: int = 50,
                                     offset: int = 0) -> List[Dict[str, Any]]:
        """获取优化历史记录"""
        
        # 构建查询条件
        if filter_dict:
            query_parts = []
            for key, value in filter_dict.items():
                query_parts.append(f"{key}:{value}")
            query = " AND ".join(query_parts)
        else:
            query = "optimization"
        
        optimization_memories = await self.memory_manager.search_memories(
            query=query,
            namespace=self.optimization_namespace,
            limit=limit + offset  # 获取更多数据用于分页
        )
        
        history = []
        for memory in optimization_memories:
            try:
                data = json.loads(memory.content)
                history.append(data)
            except json.JSONDecodeError:
                continue
        
        # 排序并分页
        sorted_history = sorted(history, key=lambda x: x["timestamp"], reverse=True)
        return sorted_history[offset:offset + limit]


class EnhancedChatResponse(ChatResponse):
    """增强的聊天响应，包含反馈收集功能"""
    
    feedback_collected: bool = False
    conversation_id: str = ""
    agent_type: str = ""


class FeedbackCollector:
    """反馈收集器 - 集成到现有API中"""
    
    def __init__(self, memory_manager: MemoryStoreManager):
        self.memory_manager = memory_manager
        self.feedback_namespace = "prompt_feedback"
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "namespace": self.feedback_namespace
        }
    
    async def collect_feedback(self,
                             agent_type: str,
                             user_id: str,
                             session_id: str,
                             satisfaction_score: int,
                             feedback_text: Optional[str] = None,
                             improvement_suggestions: List[str] = None,
                             context: Dict[str, Any] = None) -> str:
        """收集用户反馈"""
        
        feedback_id = f"feedback_{agent_type}_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        feedback_data = {
            "feedback_id": feedback_id,
            "agent_type": agent_type,
            "user_id": user_id,
            "session_id": session_id,
            "satisfaction_score": satisfaction_score,
            "feedback_text": feedback_text,
            "improvement_suggestions": improvement_suggestions or [],
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "type": "user_feedback"
        }
        
        # 存储到记忆系统
        await self.memory_manager.store_memory(
            content=json.dumps(feedback_data),
            memory_type="episodic",
            namespace=self.feedback_namespace,
            metadata={
                "feedback_id": feedback_id,
                "agent_type": agent_type,
                "user_id": user_id,
                "session_id": session_id,
                "satisfaction_score": satisfaction_score,
                "feedback_type": "user_feedback"
            }
        )
        
        return feedback_id
    
    async def get_feedback_stats(self,
                               agent_type: str,
                               user_id: Optional[str] = None,
                               days: int = 30) -> Dict[str, Any]:
        """获取反馈统计"""
        
        # 构建查询条件
        query = f"agent_type:{agent_type}"
        if user_id:
            query += f" AND user_id:{user_id}"
        
        # 获取指定时间范围内的反馈
        feedback_memories = await self.memory_manager.search_memories(
            query=query,
            namespace=self.feedback_namespace,
            limit=1000  # 获取足够多的数据用于统计
        )
        
        # 过滤时间范围
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_feedback = []
        
        for memory in feedback_memories:
            try:
                data = json.loads(memory.content)
                feedback_time = datetime.fromisoformat(data["timestamp"])
                if feedback_time >= cutoff_date:
                    recent_feedback.append(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        
        if not recent_feedback:
            return {
                "total_count": 0,
                "average_satisfaction": 0,
                "satisfaction_distribution": {},
                "common_suggestions": []
            }
        
        # 计算统计信息
        satisfaction_scores = [f["satisfaction_score"] for f in recent_feedback]
        average_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
        
        # 满意度分布
        satisfaction_distribution = {}
        for score in satisfaction_scores:
            satisfaction_distribution[score] = satisfaction_distribution.get(score, 0) + 1
        
        # 常见建议
        all_suggestions = []
        for f in recent_feedback:
            all_suggestions.extend(f.get("improvement_suggestions", []))
        
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
        
        common_suggestions = sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_count": len(recent_feedback),
            "average_satisfaction": round(average_satisfaction, 2),
            "satisfaction_distribution": satisfaction_distribution,
            "common_suggestions": [{"suggestion": s[0], "count": s[1]} for s in common_suggestions]
        }
    
    async def collect_user_feedback(self,
                                  conversation_id: str,
                                  messages: List[Message],
                                  satisfaction_score: float,
                                  feedback_text: str = "",
                                  agent_type: str = "general") -> None:
        """收集用户满意度反馈（保持向后兼容）"""
        
        feedback = {
            "score": satisfaction_score,
            "text": feedback_text,
            "agent_type": agent_type,
            "type": "satisfaction",
            "timestamp": datetime.now().isoformat()
        }
        
        await self.collect_feedback_legacy(
            conversation_id, messages, feedback
        )
    
    async def collect_improvement_suggestion(self,
                                           conversation_id: str,
                                           messages: List[Message],
                                           suggestion: str,
                                           agent_type: str = "general") -> None:
        """收集改进建议（保持向后兼容）"""
        
        feedback = {
            "suggestion": suggestion,
            "agent_type": agent_type,
            "type": "improvement",
            "timestamp": datetime.now().isoformat()
        }
        
        await self.collect_feedback_legacy(
            conversation_id, messages, feedback
        )
    
    async def collect_feedback_legacy(self, 
                                    conversation_id: str,
                                    messages: List[Message],
                                    feedback: Dict[str, Any]) -> None:
        """收集用户反馈用于提示词优化（旧版本兼容）"""
        
        feedback_data = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": [msg.dict() for msg in messages],
            "feedback": feedback,
            "agent_type": feedback.get("agent_type", "unknown")
        }
        
        # 存储到记忆系统
        await self.memory_manager.store_memory(
            content=json.dumps(feedback_data),
            memory_type="episodic",
            namespace=self.feedback_namespace,
            metadata={
                "conversation_id": conversation_id,
                "agent_type": feedback.get("agent_type"),
                "satisfaction_score": feedback.get("score", 0),
                "feedback_type": "user_feedback"
            }
        )


class AutoOptimizationScheduler:
    """自动优化调度器 - 定期执行提示词优化"""
    
    def __init__(self, prompt_optimizer: PromptOptimizer, feedback_collector: FeedbackCollector):
        self.prompt_optimizer = prompt_optimizer
        self.feedback_collector = feedback_collector
        self.optimization_interval = timedelta(days=7)  # 每周优化一次
        self.last_optimization = {}
        self.is_running = False
        self.optimization_task = None
        self.config = {
            "enabled": False,
            "interval_hours": 24,
            "min_feedback_count": 10,
            "optimization_strategy": "gradient"
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            "is_running": self.is_running,
            "config": self.config,
            "last_optimization": {
                agent_type: time.isoformat() 
                for agent_type, time in self.last_optimization.items()
            }
        }
    
    async def start_auto_optimization(self,
                                    interval_hours: int = 24,
                                    min_feedback_count: int = 10,
                                    optimization_strategy: str = "gradient"):
        """启动自动优化"""
        self.config.update({
            "enabled": True,
            "interval_hours": interval_hours,
            "min_feedback_count": min_feedback_count,
            "optimization_strategy": optimization_strategy
        })
        
        self.optimization_interval = timedelta(hours=interval_hours)
        
        if not self.is_running:
            self.is_running = True
            self.optimization_task = asyncio.create_task(self._optimization_loop())
    
    async def stop_auto_optimization(self):
        """停止自动优化"""
        self.config["enabled"] = False
        self.is_running = False
        
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
            self.optimization_task = None
    
    async def _optimization_loop(self):
        """自动优化循环"""
        while self.is_running:
            try:
                await self.run_optimization_cycle()
                await asyncio.sleep(self.optimization_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"自动优化循环出错: {e}")
                await asyncio.sleep(3600)  # 出错时等待1小时后重试
    
    async def run_optimization_cycle(self):
        """运行一次优化周期"""
        if not self.config["enabled"]:
            return
        
        # 获取需要优化的智能体类型
        agent_types = ["supervisor", "researcher", "writer", "reviewer"]  # 可配置
        
        for agent_type in agent_types:
            try:
                if await self.should_optimize(agent_type):
                    await self._optimize_agent(agent_type)
            except Exception as e:
                print(f"优化智能体 {agent_type} 失败: {e}")
    
    async def should_optimize(self, agent_type: str) -> bool:
        """检查是否需要优化"""
        
        last_time = self.last_optimization.get(agent_type)
        if not last_time:
            return True
        
        time_since_last = datetime.now() - last_time
        return time_since_last > self.optimization_interval
    
    async def _optimize_agent(self, agent_type: str):
        """优化单个智能体"""
        
        # 检查反馈数量是否足够
        stats = await self.feedback_collector.get_feedback_stats(
            agent_type=agent_type,
            days=self.optimization_interval.days
        )
        
        if stats["total_count"] < self.config["min_feedback_count"]:
            print(f"智能体 {agent_type} 反馈数量不足，跳过优化")
            return
        
        try:
            # 执行优化
            result = await self.prompt_optimizer.optimize_agent_prompt(
                agent_type=agent_type,
                strategy=self.config["optimization_strategy"],
                context={
                    "min_feedback_count": self.config["min_feedback_count"],
                    "current_prompt": f"你是一个{agent_type}智能体。"  # 这里应该从配置中获取
                }
            )
            
            self.last_optimization[agent_type] = datetime.now()
            print(f"智能体 {agent_type} 优化完成: {result['optimization_id']}")
            
        except Exception as e:
            print(f"智能体 {agent_type} 优化失败: {e}")
    
    async def auto_optimize_agent(self, agent_type: str, current_prompt: str) -> Optional[str]:
        """自动优化智能体提示词（保持向后兼容）"""
        
        if not await self.should_optimize(agent_type):
            return None
        
        print(f"🔄 开始自动优化 {agent_type} 智能体...")
        
        try:
            result = await self.prompt_optimizer.optimize_agent_prompt(
                agent_type=agent_type,
                context={"current_prompt": current_prompt}
            )
            
            if result:
                self.last_optimization[agent_type] = datetime.now()
                print(f"✅ {agent_type} 智能体优化完成")
                return result["optimized_prompt"]
            else:
                print(f"⚠️ {agent_type} 智能体优化跳过")
                return None
                
        except Exception as e:
            print(f"❌ {agent_type} 智能体优化失败: {e}")
            return None
    
    async def run_optimization_cycle_legacy(self, agent_configs: Dict[str, str]) -> Dict[str, str]:
        """运行完整的优化周期（保持向后兼容）"""
        
        optimized_configs = {}
        
        for agent_type, current_prompt in agent_configs.items():
            optimized_prompt = await self.auto_optimize_agent(agent_type, current_prompt)
            optimized_configs[agent_type] = optimized_prompt or current_prompt
        
        return optimized_configs


# 使用示例
async def demo_integration():
    """演示集成使用"""
    
    # 初始化组件
    memory_manager = MemoryStoreManager()
    await memory_manager.initialize()
    
    prompt_optimizer = PromptOptimizer(memory_manager)
    feedback_collector = FeedbackCollector(prompt_optimizer)
    auto_scheduler = AutoOptimizationScheduler(prompt_optimizer)
    
    # 模拟收集反馈
    messages = [
        Message(role="user", content="解释一下机器学习"),
        Message(role="assistant", content="机器学习是...")
    ]
    
    await feedback_collector.collect_user_feedback(
        conversation_id="conv_123",
        messages=messages,
        satisfaction_score=0.8,
        feedback_text="回答很好，但希望有更多例子",
        agent_type="technical_assistant"
    )
    
    # 模拟自动优化
    agent_configs = {
        "technical_assistant": "你是一个技术助手，帮助用户理解技术概念。",
        "creative_writer": "你是一个创意写作助手，帮助用户创作内容。"
    }
    
    optimized_configs = await auto_scheduler.run_optimization_cycle(agent_configs)
    
    print("🎉 集成演示完成！")
    print(f"优化结果: {len(optimized_configs)} 个智能体配置已更新")


if __name__ == "__main__":
    asyncio.run(demo_integration())