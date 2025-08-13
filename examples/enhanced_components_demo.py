#!/usr/bin/env python3
"""
增强组件演示

演示如何使用新注册的增强工具管理器和智能体协作优化器。
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# 从核心模块导入新注册的组件
from core import (
    # 增强工具管理器
    get_enhanced_tool_manager,
    ToolExecutionMode,
    ToolValidationLevel,
    EnhancedToolExecutionContext,
    ToolValidator,
    
    # 协作优化器
    get_collaboration_orchestrator,
    CollaborationMode,
    MessageType,
    CollaborationMessage,
    CollaborationTask,
    CollaborationContext,
    
    # 基础组件
    BaseAgent,
    AgentType,
    AgentStatus,
    get_tool_registry,
    managed_tool
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_demo")


# 使用装饰器创建示例工具
@managed_tool(
    name="calculate_sum",
    description="计算两个数字的和",
    category="analysis",
    permissions=["execute"]
)
def calculate_sum(a: int, b: int) -> int:
    """计算两个数字的和"""
    return a + b


@managed_tool(
    name="generate_report",
    description="生成数据报告",
    category="generation",
    permissions=["read", "execute"]
)
async def generate_report(data: Dict[str, Any]) -> str:
    """生成数据报告"""
    await asyncio.sleep(1)  # 模拟异步处理
    return f"报告生成完成，处理了 {len(data)} 项数据"


class DemoAgent(BaseAgent):
    """演示智能体"""
    
    def __init__(self, agent_id: str, name: str):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.RESEARCH,
            name=name,
            description=f"演示智能体 {name}",
            capabilities=["analysis", "reporting"]
        )
    
    async def _build_graph(self):
        """构建智能体图"""
        # 简化实现，实际应该构建LangGraph
        pass
    
    async def chat(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理对话"""
        task_description = input_data.get("task_description", "")
        input_data_content = input_data.get("input_data", {})
        
        # 模拟智能体处理
        await asyncio.sleep(0.5)
        
        result = {
            "agent_id": self.agent_id,
            "processed_task": task_description,
            "result": f"智能体 {self.name} 处理完成: {task_description}",
            "input_data": input_data_content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return result


async def demo_enhanced_tool_manager():
    """演示增强工具管理器"""
    logger.info("=== 增强工具管理器演示 ===")
    
    # 获取增强工具管理器
    tool_manager = get_enhanced_tool_manager()
    
    # 注册MCP工具（如果可用）
    try:
        mcp_count = await tool_manager.register_mcp_tools()
        logger.info(f"注册了 {mcp_count} 个MCP工具")
    except Exception as e:
        logger.warning(f"MCP工具注册失败: {e}")
    
    # 创建执行上下文
    context = EnhancedToolExecutionContext(
        user_id="demo_user",
        session_id="demo_session",
        agent_id="demo_agent",
        execution_id="demo_exec_001",
        timeout=10.0
    )
    
    # 演示工具验证
    validator = ToolValidator(ToolValidationLevel.BASIC)
    
    # 演示并行工具执行
    tool_requests = [
        {
            "tool_name": "calculate_sum",
            "input_data": {"a": 10, "b": 20}
        },
        {
            "tool_name": "generate_report", 
            "input_data": {"data": {"item1": "value1", "item2": "value2"}}
        }
    ]
    
    logger.info("开始并行执行工具...")
    results = await tool_manager.execute_tools_parallel(tool_requests, context)
    
    for result in results:
        if result.success:
            logger.info(f"工具 {result.tool_name} 执行成功: {result.result}")
        else:
            logger.error(f"工具 {result.tool_name} 执行失败: {result.error}")
    
    # 获取工具统计信息
    all_tools = tool_manager.get_all_tools()
    logger.info(f"当前注册的工具: {all_tools}")
    
    active_executions = tool_manager.get_active_executions()
    logger.info(f"活跃执行数量: {len(active_executions)}")


async def demo_collaboration_orchestrator():
    """演示智能体协作编排器"""
    logger.info("=== 智能体协作编排器演示 ===")
    
    # 获取协作编排器
    orchestrator = get_collaboration_orchestrator()
    
    # 创建演示智能体
    agents = [
        DemoAgent("agent_001", "分析师"),
        DemoAgent("agent_002", "研究员"),
        DemoAgent("agent_003", "报告员")
    ]
    
    # 注册智能体
    for agent in agents:
        await orchestrator.register_agent(agent)
        logger.info(f"注册智能体: {agent.name}")
    
    # 创建协作会话
    session_id = "demo_collaboration_session"
    context = await orchestrator.create_collaboration_session(
        session_id=session_id,
        user_id="demo_user",
        mode=CollaborationMode.SEQUENTIAL,
        participating_agents=[agent.agent_id for agent in agents]
    )
    
    logger.info(f"创建协作会话: {session_id}")
    
    # 执行协作任务
    task_description = "分析市场数据并生成报告"
    task_data = {
        "input": {
            "market_data": ["数据1", "数据2", "数据3"],
            "analysis_type": "trend_analysis"
        }
    }
    
    logger.info("开始执行协作任务...")
    result = await orchestrator.execute_collaborative_task(
        session_id=session_id,
        task_description=task_description,
        task_data=task_data
    )
    
    if result["success"]:
        logger.info(f"协作任务执行成功!")
        logger.info(f"任务ID: {result['task_id']}")
        logger.info(f"执行时间: {result['execution_time']:.2f}秒")
        logger.info(f"结果: {result['result']}")
    else:
        logger.error(f"协作任务执行失败: {result['error']}")
    
    # 获取协作统计信息
    stats = orchestrator.get_collaboration_stats(session_id)
    logger.info(f"协作统计: {stats}")


async def demo_different_collaboration_modes():
    """演示不同的协作模式"""
    logger.info("=== 不同协作模式演示 ===")
    
    orchestrator = get_collaboration_orchestrator()
    
    # 创建智能体
    agents = [
        DemoAgent("parallel_agent_001", "并行处理器1"),
        DemoAgent("parallel_agent_002", "并行处理器2"),
        DemoAgent("parallel_agent_003", "并行处理器3")
    ]
    
    for agent in agents:
        await orchestrator.register_agent(agent)
    
    # 演示并行模式
    logger.info("--- 并行协作模式 ---")
    parallel_session = "parallel_session"
    await orchestrator.create_collaboration_session(
        session_id=parallel_session,
        user_id="demo_user",
        mode=CollaborationMode.PARALLEL,
        participating_agents=[agent.agent_id for agent in agents]
    )
    
    parallel_result = await orchestrator.execute_collaborative_task(
        session_id=parallel_session,
        task_description="并行处理数据",
        task_data={"input": {"data_chunk": "chunk_data"}}
    )
    
    logger.info(f"并行模式结果: {parallel_result['success']}")
    
    # 演示层次化模式
    logger.info("--- 层次化协作模式 ---")
    hierarchical_session = "hierarchical_session"
    await orchestrator.create_collaboration_session(
        session_id=hierarchical_session,
        user_id="demo_user",
        mode=CollaborationMode.HIERARCHICAL,
        participating_agents=[agent.agent_id for agent in agents]
    )
    
    hierarchical_result = await orchestrator.execute_collaborative_task(
        session_id=hierarchical_session,
        task_description="层次化任务分解",
        task_data={"input": {"complex_task": "multi_step_analysis"}}
    )
    
    logger.info(f"层次化模式结果: {hierarchical_result['success']}")


async def main():
    """主函数"""
    logger.info("开始增强组件演示")
    
    try:
        # 演示增强工具管理器
        await demo_enhanced_tool_manager()
        
        # 等待一下
        await asyncio.sleep(1)
        
        # 演示协作编排器
        await demo_collaboration_orchestrator()
        
        # 等待一下
        await asyncio.sleep(1)
        
        # 演示不同协作模式
        await demo_different_collaboration_modes()
        
        logger.info("所有演示完成!")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())