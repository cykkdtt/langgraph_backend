"""协作工具模块

提供智能体间协作所需的工具函数。
"""

import logging
from typing import Dict, Any, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def handoff_to_research(task_description: str, context: Optional[str] = None) -> str:
    """移交任务给研究智能体
    
    Args:
        task_description: 任务描述
        context: 可选的上下文信息
        
    Returns:
        str: 移交结果
    """
    logger.info(f"移交任务给研究智能体: {task_description}")
    
    result = {
        "action": "handoff_to_research",
        "target_agent": "research",
        "task": task_description,
        "context": context,
        "status": "pending"
    }
    
    return f"任务已移交给研究智能体: {task_description}"


@tool
def handoff_to_chart(data_description: str, chart_type: Optional[str] = None) -> str:
    """移交任务给图表智能体
    
    Args:
        data_description: 数据描述
        chart_type: 可选的图表类型
        
    Returns:
        str: 移交结果
    """
    logger.info(f"移交任务给图表智能体: {data_description}")
    
    result = {
        "action": "handoff_to_chart",
        "target_agent": "chart",
        "data": data_description,
        "chart_type": chart_type,
        "status": "pending"
    }
    
    return f"任务已移交给图表智能体: {data_description}"


@tool
def task_coordinator(task_list: str, priority: str = "normal") -> str:
    """任务协调工具
    
    Args:
        task_list: 任务列表描述
        priority: 任务优先级 (low, normal, high)
        
    Returns:
        str: 协调结果
    """
    logger.info(f"协调任务: {task_list}, 优先级: {priority}")
    
    result = {
        "action": "task_coordination",
        "tasks": task_list,
        "priority": priority,
        "status": "coordinated"
    }
    
    return f"任务协调完成: {task_list} (优先级: {priority})"


# 工具映射字典，用于根据字符串名称获取工具对象
COLLABORATION_TOOLS = {
    "handoff_to_research": handoff_to_research,
    "handoff_to_chart": handoff_to_chart,
    "task_coordinator": task_coordinator
}


def get_collaboration_tool(tool_name: str):
    """根据工具名称获取工具对象
    
    Args:
        tool_name: 工具名称
        
    Returns:
        工具对象或None
    """
    return COLLABORATION_TOOLS.get(tool_name)


def get_all_collaboration_tools():
    """获取所有协作工具
    
    Returns:
        List: 所有协作工具对象列表
    """
    return list(COLLABORATION_TOOLS.values())