"""
集成Human-in-the-Loop功能的智能体示例

展示如何在现有的BaseAgent架构中集成LangGraph官方的
Human-in-the-Loop功能。
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

# 导入项目现有模块
from core.agents.base import BaseAgent
from core.interrupts.enhanced_interrupt_manager import (
    EnhancedInterruptManager,
    create_approval_node_interrupt,
    create_human_input_node_interrupt,
    create_tool_review_node_interrupt,
    create_state_edit_node_interrupt
)
from core.interrupts.interrupt_types import InterruptPriority
from models.agent_models import AgentConfig


class HumanInLoopAgent(BaseAgent):
    """集成Human-in-the-Loop功能的智能体"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.interrupt_manager = EnhancedInterruptManager(
            checkpointer=self.checkpointer
        )
        
    def _initialize(self):
        """初始化包含人工干预的智能体图"""
        workflow = StateGraph(dict)
        
        # 添加节点
        workflow.add_node("analyze_request", self._analyze_request)
        workflow.add_node("security_check", self._security_check)
        workflow.add_node("plan_execution", self._plan_execution)
        workflow.add_node("execute_with_approval", self._execute_with_approval)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("finalize", self._finalize)
        
        # 添加边
        workflow.add_edge(START, "analyze_request")
        workflow.add_edge("analyze_request", "security_check")
        workflow.add_edge("security_check", "plan_execution")
        workflow.add_edge("plan_execution", "execute_with_approval")
        workflow.add_edge("execute_with_approval", "validate_results")
        workflow.add_edge("validate_results", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _analyze_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """分析请求"""
        self.logger.info("分析用户请求...")
        
        user_message = state.get("messages", [])[-1]["content"] if state.get("messages") else ""
        
        # 分析请求复杂度和风险
        analysis = {
            "complexity": "high" if len(user_message) > 100 else "medium",
            "risk_level": "high" if any(word in user_message.lower() 
                                     for word in ["删除", "修改", "支付", "转账"]) else "low",
            "estimated_time": "30分钟",
            "required_tools": ["search", "analysis", "report_generation"]
        }
        
        return {
            **state,
            "analysis": analysis,
            "current_step": "analyze_request",
            "step_completed_at": datetime.now().isoformat()
        }
    
    async def _security_check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """安全检查 - 可能需要审批"""
        self.logger.info("执行安全检查...")
        
        analysis = state.get("analysis", {})
        
        # 高风险操作需要审批
        if analysis.get("risk_level") == "high":
            self.logger.info("检测到高风险操作，需要安全审批...")
            
            # 使用增强的中断管理器创建审批中断
            approval_result = create_approval_node_interrupt(
                self.interrupt_manager,
                title="安全审批请求",
                description=f"检测到高风险操作，需要安全审批。操作内容：{state.get('messages', [])[-1]['content'] if state.get('messages') else ''}",
                context={
                    "user_id": state.get("user_id", "unknown"),
                    "risk_level": analysis.get("risk_level"),
                    "operation_type": "security_sensitive",
                    "run_id": state.get("run_id", str(uuid.uuid4())),
                    "node_id": "security_check"
                },
                priority=InterruptPriority.HIGH,
                required_approvers=["security_admin", "supervisor"],
                timeout_seconds=3600,  # 1小时超时
                options=[
                    {"value": "approve", "label": "批准执行"},
                    {"value": "reject", "label": "拒绝执行"},
                    {"value": "modify", "label": "修改后执行"}
                ]
            )
            
            # 处理审批结果
            if approval_result.get("decision") == "reject":
                return {
                    **state,
                    "security_status": "rejected",
                    "execution_stopped": True,
                    "rejection_reason": approval_result.get("reason", "安全审批被拒绝")
                }
            elif approval_result.get("decision") == "modify":
                # 如果需要修改，可以更新执行计划
                state["execution_modifications"] = approval_result.get("modifications", [])
        
        return {
            **state,
            "security_status": "approved",
            "security_check_completed_at": datetime.now().isoformat()
        }
    
    async def _plan_execution(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """制定执行计划 - 可能需要人工输入"""
        self.logger.info("制定执行计划...")
        
        # 如果执行被停止，直接返回
        if state.get("execution_stopped"):
            return state
        
        analysis = state.get("analysis", {})
        
        # 复杂任务需要人工输入额外参数
        if analysis.get("complexity") == "high":
            self.logger.info("复杂任务需要人工输入执行参数...")
            
            human_input = create_human_input_node_interrupt(
                self.interrupt_manager,
                prompt="请提供执行此复杂任务所需的详细参数",
                input_type="form",
                context={
                    "user_id": state.get("user_id", "unknown"),
                    "task_complexity": analysis.get("complexity"),
                    "run_id": state.get("run_id", str(uuid.uuid4())),
                    "node_id": "plan_execution"
                },
                validation_rules={
                    "priority": {"required": True, "type": "string"},
                    "deadline": {"required": False, "type": "datetime"},
                    "quality_level": {"required": True, "type": "string"}
                },
                options=[
                    {
                        "name": "priority",
                        "type": "select",
                        "label": "任务优先级",
                        "options": ["low", "medium", "high", "urgent"]
                    },
                    {
                        "name": "deadline",
                        "type": "datetime",
                        "label": "期望完成时间"
                    },
                    {
                        "name": "quality_level",
                        "type": "select",
                        "label": "质量要求",
                        "options": ["basic", "standard", "premium"]
                    },
                    {
                        "name": "special_requirements",
                        "type": "text",
                        "label": "特殊要求"
                    }
                ]
            )
            
            # 将人工输入整合到执行计划中
            execution_plan = {
                "priority": human_input.get("priority", "medium"),
                "deadline": human_input.get("deadline"),
                "quality_level": human_input.get("quality_level", "standard"),
                "special_requirements": human_input.get("special_requirements", ""),
                "estimated_steps": 5,
                "tools_needed": analysis.get("required_tools", [])
            }
        else:
            # 简单任务使用默认计划
            execution_plan = {
                "priority": "medium",
                "quality_level": "standard",
                "estimated_steps": 3,
                "tools_needed": analysis.get("required_tools", [])
            }
        
        return {
            **state,
            "execution_plan": execution_plan,
            "plan_created_at": datetime.now().isoformat()
        }
    
    async def _execute_with_approval(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务 - 工具调用需要审查"""
        self.logger.info("执行任务...")
        
        if state.get("execution_stopped"):
            return state
        
        execution_plan = state.get("execution_plan", {})
        
        # 准备工具调用
        proposed_tools = []
        for tool_name in execution_plan.get("tools_needed", []):
            if tool_name == "search":
                proposed_tools.append({
                    "tool_name": "web_search",
                    "arguments": {
                        "query": state.get("messages", [])[-1]["content"] if state.get("messages") else "",
                        "max_results": 10
                    },
                    "description": "搜索相关信息"
                })
            elif tool_name == "analysis":
                proposed_tools.append({
                    "tool_name": "data_analyzer",
                    "arguments": {
                        "data_source": "search_results",
                        "analysis_type": "comprehensive"
                    },
                    "description": "分析搜索结果"
                })
            elif tool_name == "report_generation":
                proposed_tools.append({
                    "tool_name": "report_generator",
                    "arguments": {
                        "format": "markdown",
                        "include_charts": True,
                        "quality": execution_plan.get("quality_level", "standard")
                    },
                    "description": "生成分析报告"
                })
        
        # 审查工具调用
        if proposed_tools:
            self.logger.info(f"需要审查 {len(proposed_tools)} 个工具调用...")
            
            tool_review_result = create_tool_review_node_interrupt(
                self.interrupt_manager,
                proposed_tools=proposed_tools,
                context={
                    "user_id": state.get("user_id", "unknown"),
                    "execution_plan": execution_plan,
                    "run_id": state.get("run_id", str(uuid.uuid4())),
                    "node_id": "execute_with_approval"
                },
                allow_modifications=True
            )
            
            # 使用审查后的工具
            approved_tools = tool_review_result.get("approved_tools", proposed_tools)
            
            # 模拟工具执行
            tool_results = []
            for tool in approved_tools:
                result = {
                    "tool_name": tool["tool_name"],
                    "status": "success",
                    "result": f"模拟执行结果 for {tool['tool_name']}",
                    "executed_at": datetime.now().isoformat()
                }
                tool_results.append(result)
                self.logger.info(f"执行工具: {tool['tool_name']}")
        else:
            tool_results = []
        
        return {
            **state,
            "tool_results": tool_results,
            "execution_completed_at": datetime.now().isoformat()
        }
    
    async def _validate_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """验证结果 - 可能需要编辑状态"""
        self.logger.info("验证执行结果...")
        
        if state.get("execution_stopped"):
            return state
        
        # 生成初始结果
        tool_results = state.get("tool_results", [])
        execution_plan = state.get("execution_plan", {})
        
        initial_output = {
            "summary": f"任务执行完成，共执行了 {len(tool_results)} 个工具",
            "results": tool_results,
            "quality_score": 85,  # 模拟质量评分
            "execution_time": "25分钟",
            "status": "completed"
        }
        
        # 高质量要求需要人工验证
        if execution_plan.get("quality_level") in ["premium", "high"]:
            self.logger.info("高质量要求，需要人工验证结果...")
            
            validation_result = create_state_edit_node_interrupt(
                self.interrupt_manager,
                current_state=initial_output,
                editable_fields=["summary", "quality_score", "status", "notes"],
                context={
                    "user_id": state.get("user_id", "unknown"),
                    "quality_requirement": execution_plan.get("quality_level"),
                    "run_id": state.get("run_id", str(uuid.uuid4())),
                    "node_id": "validate_results"
                },
                validation_schema={
                    "quality_score": {"type": "number", "min": 0, "max": 100},
                    "status": {"type": "string", "enum": ["completed", "needs_revision", "failed"]}
                }
            )
            
            # 使用验证后的结果
            final_output = validation_result.get("final_state", initial_output)
        else:
            final_output = initial_output
        
        return {
            **state,
            "final_output": final_output,
            "validation_completed_at": datetime.now().isoformat()
        }
    
    async def _finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """完成任务"""
        self.logger.info("完成任务处理...")
        
        if state.get("execution_stopped"):
            final_message = {
                "role": "assistant",
                "content": f"任务执行被停止：{state.get('rejection_reason', '未知原因')}",
                "metadata": {"status": "stopped"}
            }
        else:
            final_output = state.get("final_output", {})
            final_message = {
                "role": "assistant",
                "content": f"任务执行完成！{final_output.get('summary', '')}",
                "metadata": {
                    "status": "completed",
                    "quality_score": final_output.get("quality_score"),
                    "execution_summary": {
                        "security_approved": state.get("security_status") == "approved",
                        "tools_executed": len(state.get("tool_results", [])),
                        "human_interactions": {
                            "security_approval": state.get("security_status") == "approved",
                            "planning_input": "execution_plan" in state,
                            "tool_review": "tool_results" in state,
                            "result_validation": "final_output" in state
                        }
                    }
                }
            }
        
        return {
            **state,
            "messages": state.get("messages", []) + [final_message],
            "completed_at": datetime.now().isoformat()
        }


async def demo_human_in_loop_agent():
    """演示Human-in-the-Loop智能体"""
    print("🚀 启动Human-in-the-Loop智能体演示")
    print("=" * 60)
    
    # 创建智能体配置
    config = AgentConfig(
        agent_id="human_in_loop_demo",
        agent_type="human_in_loop",
        name="Human-in-the-Loop演示智能体",
        description="集成人工干预功能的智能体",
        model_config={
            "model": "gpt-4",
            "temperature": 0.7
        }
    )
    
    # 创建智能体
    agent = HumanInLoopAgent(config)
    
    # 测试场景1：高风险操作
    print("📋 场景1：高风险操作（需要安全审批）")
    print("-" * 40)
    
    initial_state = {
        "messages": [{
            "role": "user",
            "content": "请帮我删除数据库中所有过期的用户数据"
        }],
        "user_id": "user123",
        "run_id": str(uuid.uuid4())
    }
    
    config_dict = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    try:
        # 运行直到第一个中断
        result = agent.graph.invoke(initial_state, config=config_dict)
        
        if "__interrupt__" in result:
            print(f"🔄 检测到中断: {result['__interrupt__'][0]['value']['type']}")
            
            # 模拟安全审批
            approval_response = {
                "decision": "approve",
                "approver_id": "security_admin",
                "reason": "已确认删除范围合理，仅删除30天前的数据",
                "conditions": ["保留审计日志", "分批删除"]
            }
            
            print(f"👤 安全审批: {approval_response['decision']}")
            
            # 继续执行
            result = agent.graph.invoke(Command(resume=approval_response), config=config_dict)
            
            # 处理后续中断...
            while "__interrupt__" in result:
                interrupt_type = result["__interrupt__"][0]["value"]["type"]
                print(f"🔄 处理中断: {interrupt_type}")
                
                if interrupt_type == "human_input_request":
                    # 模拟人工输入
                    human_input_response = {
                        "priority": "high",
                        "deadline": "2024-12-25T18:00:00",
                        "quality_level": "premium",
                        "special_requirements": "需要详细的删除报告"
                    }
                    print(f"📝 人工输入: {human_input_response}")
                    result = agent.graph.invoke(Command(resume=human_input_response), config=config_dict)
                
                elif interrupt_type == "tool_review_request":
                    # 模拟工具审查
                    tool_review_response = {
                        "action": "modify",
                        "approved_tools": [
                            {
                                "tool_name": "web_search",
                                "arguments": {"query": "数据删除最佳实践", "max_results": 5},
                                "description": "搜索数据删除最佳实践"
                            }
                        ]
                    }
                    print(f"🔧 工具审查: {tool_review_response['action']}")
                    result = agent.graph.invoke(Command(resume=tool_review_response), config=config_dict)
                
                elif interrupt_type == "state_edit_request":
                    # 模拟状态验证
                    validation_response = {
                        "action": "edit",
                        "final_state": {
                            "summary": "安全删除了1,234条过期用户数据",
                            "quality_score": 95,
                            "status": "completed",
                            "notes": "所有操作已记录到审计日志"
                        }
                    }
                    print(f"✅ 结果验证: {validation_response['action']}")
                    result = agent.graph.invoke(Command(resume=validation_response), config=config_dict)
        
        print(f"\n🎉 任务完成！最终结果:")
        if result.get("messages"):
            print(f"响应: {result['messages'][-1]['content']}")
        
    except Exception as e:
        print(f"❌ 执行出错: {e}")


if __name__ == "__main__":
    asyncio.run(demo_human_in_loop_agent())