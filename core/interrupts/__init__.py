"""
中断和人工干预模块

提供智能体执行过程中的中断处理和人工干预功能，
包括审批流程、人工输入、工具调用审查等。
"""

from .interrupt_types import (
    InterruptType,
    InterruptStatus,
    InterruptPriority,
    InterruptRequest,
    InterruptResponse,
    ApprovalRequest,
    ApprovalResponse,
    HumanInputRequest,
    HumanInputResponse,
    InterruptContext,
    InterruptNotification
)

from .enhanced_interrupt_manager import (
    EnhancedInterruptManager,
    ApprovalWorkflowType,
    ApprovalRule,
    ApprovalWorkflow
)

# 全局实例
_interrupt_manager_instance = None

def get_interrupt_manager() -> EnhancedInterruptManager:
    """获取中断管理器实例"""
    global _interrupt_manager_instance
    if _interrupt_manager_instance is None:
        _interrupt_manager_instance = EnhancedInterruptManager()
    return _interrupt_manager_instance

__all__ = [
    # 中断类型和状态
    "InterruptType",
    "InterruptStatus", 
    "InterruptPriority",
    
    # 请求和响应模型
    "InterruptRequest",
    "InterruptResponse",
    "ApprovalRequest",
    "ApprovalResponse",
    "HumanInputRequest",
    "HumanInputResponse",
    
    # 上下文和通知
    "InterruptContext",
    "InterruptNotification",
    
    # 增强版管理器
    "EnhancedInterruptManager",
    
    # 审批工作流
    "ApprovalWorkflowType",
    "ApprovalRule",
    "ApprovalWorkflow",
    
    # 管理器函数
    "get_interrupt_manager"
]