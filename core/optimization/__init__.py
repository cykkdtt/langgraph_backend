"""
提示词优化模块

提供LangMem集成的提示词优化功能，包括：
- 提示词优化器
- 反馈收集器
- 自动优化调度器
"""

from .prompt_optimizer import PromptOptimizer, FeedbackCollector, AutoOptimizationScheduler

# 全局优化管理器实例
_optimization_manager = None

async def get_optimization_manager():
    """获取全局优化管理器实例"""
    global _optimization_manager
    if _optimization_manager is None:
        from core.memory.store_manager import get_memory_store_manager
        memory_manager = await get_memory_store_manager()
        _optimization_manager = PromptOptimizer(memory_manager)
        await _optimization_manager.initialize()
    return _optimization_manager

__all__ = [
    "PromptOptimizer",
    "FeedbackCollector", 
    "AutoOptimizationScheduler",
    "get_optimization_manager"
]