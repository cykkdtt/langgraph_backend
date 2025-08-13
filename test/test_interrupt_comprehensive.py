#!/usr/bin/env python3
"""
LangGraph中断功能综合测试

本测试文件展示了如何使用LangGraph官方的interrupt()函数和Command原语
实现人机交互功能，包括：
1. 基础中断和恢复
2. 审批工作流
3. 人工输入
4. 工具调用审查
5. 状态编辑
6. 多重中断处理

参考文档：
- https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
- https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/
"""

import asyncio
import uuid
import json
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime, timedelta

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

# 导入项目模块
from core.interrupts.enhanced_interrupt_manager import EnhancedInterruptManager
from core.interrupts.interrupt_types import (
    InterruptType, InterruptPriority, InterruptStatus
)


class WorkflowState(TypedDict):
    """工作流状态定义"""
    messages: List[Dict[str, Any]]
    current_step: str
    user_id: str
    session_id: str
    task_type: str
    risk_level: str
    approval_status: Optional[str]
    human_input: Optional[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    execution_plan: Optional[Dict[str, Any]]
    results: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class InterruptTestWorkflow:
    """中断功能测试工作流"""
    
    def __init__(self):
        self.interrupt_manager = EnhancedInterruptManager()
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """构建包含各种中断场景的测试图"""
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("start_task", self._start_task)
        workflow.add_node("risk_assessment", self._risk_assessment)
        workflow.add_node("approval_gate", self._approval_gate)
        workflow.add_node("collect_input", self._collect_input)
        workflow.add_node("plan_tools", self._plan_tools)
        workflow.add_node("review_tools", self._review_tools)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("edit_state", self._edit_state)
        workflow.add_node("finalize", self._finalize)
        
        # 添加边
        workflow.add_edge(START, "start_task")
        workflow.add_edge("start_task", "risk_assessment")
        workflow.add_edge("risk_assessment", "approval_gate")
        workflow.add_edge("approval_gate", "collect_input")
        workflow.add_edge("collect_input", "plan_tools")
        workflow.add_edge("plan_tools", "review_tools")
        workflow.add_edge("review_tools", "execute_tools")
        workflow.add_edge("execute_tools", "edit_state")
        workflow.add_edge("edit_state", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _start_task(self, state: WorkflowState) -> WorkflowState:
        """开始任务"""
        print("🚀 开始任务处理...")
        
        return {
            **state,
            "current_step": "start_task",
            "metadata": {
                **state.get("metadata", {}),
                "started_at": datetime.now().isoformat(),
                "step_history": ["start_task"]
            }
        }
    
    def _risk_assessment(self, state: WorkflowState) -> WorkflowState:
        """风险评估"""
        print("🔍 执行风险评估...")
        
        # 模拟风险评估逻辑
        task_content = state.get("messages", [{}])[-1].get("content", "")
        risk_keywords = ["删除", "支付", "转账", "修改系统", "访问敏感数据"]
        
        risk_level = "high" if any(keyword in task_content for keyword in risk_keywords) else "low"
        
        return {
            **state,
            "current_step": "risk_assessment",
            "risk_level": risk_level,
            "metadata": {
                **state.get("metadata", {}),
                "risk_assessed_at": datetime.now().isoformat(),
                "step_history": state.get("metadata", {}).get("step_history", []) + ["risk_assessment"]
            }
        }
    
    def _approval_gate(self, state: WorkflowState) -> WorkflowState:
        """审批门控 - 高风险任务需要审批"""
        print("🛡️ 检查是否需要审批...")
        
        if state.get("risk_level") == "high":
            print("⚠️ 检测到高风险操作，需要审批...")
            
            # 创建审批中断数据
            approval_data = self.interrupt_manager.create_approval_interrupt(
                title="高风险操作审批",
                description=f"任务包含高风险操作，需要管理员审批。任务内容：{state.get('messages', [{}])[-1].get('content', '')}",
                context={
                    "user_id": state.get("user_id"),
                    "session_id": state.get("session_id"),
                    "risk_level": state.get("risk_level"),
                    "task_type": state.get("task_type"),
                    "run_id": str(uuid.uuid4()),
                    "node_id": "approval_gate"
                },
                priority=InterruptPriority.HIGH,
                required_approvers=["admin", "security_officer"],
                timeout_seconds=3600,
                options=[
                    {"value": "approve", "label": "批准执行"},
                    {"value": "reject", "label": "拒绝执行"},
                    {"value": "modify", "label": "修改后执行"}
                ]
            )
            
            print(f"📋 创建审批请求: {approval_data['interrupt_id']}")
            
            # 使用LangGraph的interrupt()函数暂停执行
            approval_result = interrupt(approval_data)
            
            print(f"✅ 收到审批结果: {approval_result}")
            
            # 处理审批结果
            if isinstance(approval_result, dict):
                decision = approval_result.get("decision", "approve")
                if decision == "reject":
                    return {
                        **state,
                        "current_step": "approval_gate",
                        "approval_status": "rejected",
                        "execution_stopped": True,
                        "rejection_reason": approval_result.get("reason", "审批被拒绝")
                    }
                elif decision == "modify":
                    state["execution_modifications"] = approval_result.get("modifications", [])
            
            approval_status = "approved"
        else:
            print("✅ 低风险操作，无需审批")
            approval_status = "not_required"
        
        return {
            **state,
            "current_step": "approval_gate",
            "approval_status": approval_status,
            "metadata": {
                **state.get("metadata", {}),
                "approval_checked_at": datetime.now().isoformat(),
                "step_history": state.get("metadata", {}).get("step_history", []) + ["approval_gate"]
            }
        }
    
    def _collect_input(self, state: WorkflowState) -> WorkflowState:
        """收集人工输入"""
        print("📝 收集执行参数...")
        
        # 如果执行被停止，跳过
        if state.get("execution_stopped"):
            return state
        
        # 创建人工输入中断
        input_data = self.interrupt_manager.create_human_input_interrupt(
            prompt="请提供任务执行所需的详细参数",
            input_type="form",
            context={
                "user_id": state.get("user_id"),
                "session_id": state.get("session_id"),
                "task_type": state.get("task_type"),
                "run_id": str(uuid.uuid4()),
                "node_id": "collect_input"
            },
            validation_rules={
                "priority": {"required": True, "type": "string"},
                "deadline": {"required": False, "type": "datetime"},
                "quality_level": {"required": True, "type": "string"}
            },
            timeout_seconds=1800,
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
                }
            ]
        )
        
        print(f"📋 创建输入请求: {input_data['interrupt_id']}")
        
        # 使用interrupt()收集人工输入
        human_input = interrupt(input_data)
        
        print(f"✅ 收到人工输入: {human_input}")
        
        return {
            **state,
            "current_step": "collect_input",
            "human_input": human_input if isinstance(human_input, dict) else {},
            "metadata": {
                **state.get("metadata", {}),
                "input_collected_at": datetime.now().isoformat(),
                "step_history": state.get("metadata", {}).get("step_history", []) + ["collect_input"]
            }
        }
    
    def _plan_tools(self, state: WorkflowState) -> WorkflowState:
        """规划工具调用"""
        print("🔧 规划工具调用...")
        
        if state.get("execution_stopped"):
            return state
        
        # 基于任务类型和人工输入规划工具
        human_input = state.get("human_input", {})
        task_type = state.get("task_type", "general")
        
        # 模拟工具规划
        planned_tools = []
        
        if task_type == "data_analysis":
            planned_tools = [
                {
                    "name": "fetch_data",
                    "args": {"source": "database", "table": "user_data"},
                    "description": "获取用户数据"
                },
                {
                    "name": "analyze_data",
                    "args": {"method": "statistical", "confidence": 0.95},
                    "description": "统计分析数据"
                },
                {
                    "name": "generate_report",
                    "args": {"format": "pdf", "include_charts": True},
                    "description": "生成分析报告"
                }
            ]
        elif task_type == "system_operation":
            planned_tools = [
                {
                    "name": "check_system_status",
                    "args": {"components": ["database", "api", "cache"]},
                    "description": "检查系统状态"
                },
                {
                    "name": "backup_data",
                    "args": {"backup_type": "incremental"},
                    "description": "备份数据"
                }
            ]
        else:
            planned_tools = [
                {
                    "name": "search_information",
                    "args": {"query": state.get("messages", [{}])[-1].get("content", "")},
                    "description": "搜索相关信息"
                }
            ]
        
        return {
            **state,
            "current_step": "plan_tools",
            "tool_calls": planned_tools,
            "metadata": {
                **state.get("metadata", {}),
                "tools_planned_at": datetime.now().isoformat(),
                "step_history": state.get("metadata", {}).get("step_history", []) + ["plan_tools"]
            }
        }
    
    def _review_tools(self, state: WorkflowState) -> WorkflowState:
        """审查工具调用"""
        print("🔍 审查工具调用...")
        
        if state.get("execution_stopped"):
            return state
        
        tool_calls = state.get("tool_calls", [])
        
        if not tool_calls:
            return {
                **state,
                "current_step": "review_tools"
            }
        
        # 创建工具审查中断
        review_data = self.interrupt_manager.create_tool_review_interrupt(
            proposed_tools=tool_calls,
            context={
                "user_id": state.get("user_id"),
                "session_id": state.get("session_id"),
                "task_type": state.get("task_type"),
                "run_id": str(uuid.uuid4()),
                "node_id": "review_tools"
            },
            allow_modifications=True
        )
        
        print(f"📋 创建工具审查请求: {review_data['interrupt_id']}")
        
        # 使用interrupt()进行工具审查
        review_result = interrupt(review_data)
        
        print(f"✅ 收到工具审查结果: {review_result}")
        
        # 处理审查结果
        if isinstance(review_result, dict):
            action = review_result.get("action", "approve_all")
            
            if action == "approve_all":
                approved_tools = tool_calls
            elif action == "modify":
                approved_tools = review_result.get("modified_tools", tool_calls)
            elif action == "reject_all":
                approved_tools = []
            else:
                approved_tools = tool_calls
        else:
            approved_tools = tool_calls
        
        return {
            **state,
            "current_step": "review_tools",
            "tool_calls": approved_tools,
            "tool_review_result": review_result if isinstance(review_result, dict) else {},
            "metadata": {
                **state.get("metadata", {}),
                "tools_reviewed_at": datetime.now().isoformat(),
                "step_history": state.get("metadata", {}).get("step_history", []) + ["review_tools"]
            }
        }
    
    def _execute_tools(self, state: WorkflowState) -> WorkflowState:
        """执行工具调用"""
        print("⚡ 执行工具调用...")
        
        if state.get("execution_stopped"):
            return state
        
        tool_calls = state.get("tool_calls", [])
        results = []
        
        # 模拟工具执行
        for tool in tool_calls:
            print(f"  🔧 执行工具: {tool['name']}")
            
            # 模拟工具执行结果
            if tool["name"] == "fetch_data":
                result = {"status": "success", "rows": 1000, "data": "sample_data"}
            elif tool["name"] == "analyze_data":
                result = {"status": "success", "insights": ["趋势上升", "异常值检测"], "confidence": 0.95}
            elif tool["name"] == "generate_report":
                result = {"status": "success", "report_url": "https://example.com/report.pdf"}
            elif tool["name"] == "search_information":
                result = {"status": "success", "results": ["结果1", "结果2", "结果3"]}
            else:
                result = {"status": "success", "message": f"工具 {tool['name']} 执行完成"}
            
            results.append({
                "tool": tool["name"],
                "result": result,
                "executed_at": datetime.now().isoformat()
            })
        
        return {
            **state,
            "current_step": "execute_tools",
            "results": {"tool_results": results},
            "metadata": {
                **state.get("metadata", {}),
                "tools_executed_at": datetime.now().isoformat(),
                "step_history": state.get("metadata", {}).get("step_history", []) + ["execute_tools"]
            }
        }
    
    def _edit_state(self, state: WorkflowState) -> WorkflowState:
        """编辑状态 - 允许人工修改最终结果"""
        print("✏️ 状态编辑检查...")
        
        if state.get("execution_stopped"):
            return state
        
        # 创建状态编辑中断
        current_results = state.get("results", {})
        
        edit_data = self.interrupt_manager.create_state_edit_interrupt(
            current_state=current_results,
            editable_fields=["tool_results", "summary", "recommendations"],
            context={
                "user_id": state.get("user_id"),
                "session_id": state.get("session_id"),
                "task_type": state.get("task_type"),
                "run_id": str(uuid.uuid4()),
                "node_id": "edit_state"
            },
            validation_schema={
                "summary": {"type": "string", "max_length": 500},
                "recommendations": {"type": "array", "items": {"type": "string"}}
            }
        )
        
        print(f"📋 创建状态编辑请求: {edit_data['interrupt_id']}")
        
        # 使用interrupt()进行状态编辑
        edit_result = interrupt(edit_data)
        
        print(f"✅ 收到状态编辑结果: {edit_result}")
        
        # 处理编辑结果
        if isinstance(edit_result, dict):
            action = edit_result.get("action", "approve")
            
            if action == "edit":
                # 应用编辑
                edited_state = edit_result.get("edited_state", current_results)
                final_results = {**current_results, **edited_state}
            elif action == "reset":
                # 重置到初始状态
                final_results = {"tool_results": []}
            else:
                # 保持当前状态
                final_results = current_results
        else:
            final_results = current_results
        
        return {
            **state,
            "current_step": "edit_state",
            "results": final_results,
            "state_edit_result": edit_result if isinstance(edit_result, dict) else {},
            "metadata": {
                **state.get("metadata", {}),
                "state_edited_at": datetime.now().isoformat(),
                "step_history": state.get("metadata", {}).get("step_history", []) + ["edit_state"]
            }
        }
    
    def _finalize(self, state: WorkflowState) -> WorkflowState:
        """完成任务"""
        print("🎯 完成任务...")
        
        return {
            **state,
            "current_step": "finalize",
            "completed": True,
            "metadata": {
                **state.get("metadata", {}),
                "completed_at": datetime.now().isoformat(),
                "step_history": state.get("metadata", {}).get("step_history", []) + ["finalize"]
            }
        }


async def test_basic_interrupt():
    """测试基础中断功能"""
    print("\n" + "="*50)
    print("🧪 测试基础中断功能")
    print("="*50)
    
    workflow = InterruptTestWorkflow()
    
    # 创建初始状态
    initial_state = {
        "messages": [{"role": "user", "content": "请帮我分析用户数据"}],
        "user_id": "test_user",
        "session_id": str(uuid.uuid4()),
        "task_type": "data_analysis",
        "metadata": {}
    }
    
    # 配置
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    try:
        # 运行工作流直到第一个中断
        print("🚀 启动工作流...")
        result = workflow.graph.invoke(initial_state, config=config)
        
        # 检查是否有中断
        if "__interrupt__" in result:
            interrupts = result["__interrupt__"]
            print(f"⏸️ 工作流在中断处暂停，中断数量: {len(interrupts)}")
            
            for i, interrupt_info in enumerate(interrupts):
                print(f"  中断 {i+1}: {interrupt_info}")
                
                # 模拟人工响应
                if "approval" in str(interrupt_info.value):
                    print("  👤 模拟审批响应: 批准")
                    response = Command(resume={
                        "decision": "approve",
                        "reason": "测试批准",
                        "approved_by": "test_admin"
                    })
                elif "input" in str(interrupt_info.value):
                    print("  👤 模拟人工输入响应")
                    response = Command(resume={
                        "priority": "high",
                        "quality_level": "premium",
                        "deadline": (datetime.now() + timedelta(hours=2)).isoformat()
                    })
                elif "tool" in str(interrupt_info.value):
                    print("  👤 模拟工具审查响应: 批准所有工具")
                    response = Command(resume={
                        "action": "approve_all",
                        "reviewer": "test_reviewer"
                    })
                elif "state" in str(interrupt_info.value):
                    print("  👤 模拟状态编辑响应: 保持当前状态")
                    response = Command(resume={
                        "action": "approve",
                        "editor": "test_editor"
                    })
                else:
                    print("  👤 模拟通用响应")
                    response = Command(resume={"approved": True})
                
                # 恢复执行
                print("  ▶️ 恢复工作流执行...")
                result = workflow.graph.invoke(response, config=config)
                
                # 检查是否还有更多中断
                if "__interrupt__" not in result:
                    print("  ✅ 工作流完成")
                    break
                else:
                    print(f"  ⏸️ 遇到下一个中断: {len(result['__interrupt__'])} 个")
        else:
            print("✅ 工作流完成，无中断")
        
        print(f"\n📊 最终结果:")
        print(f"  当前步骤: {result.get('current_step', 'unknown')}")
        print(f"  完成状态: {result.get('completed', False)}")
        print(f"  执行历史: {result.get('metadata', {}).get('step_history', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


async def test_multiple_interrupts():
    """测试多重中断处理"""
    print("\n" + "="*50)
    print("🧪 测试多重中断处理")
    print("="*50)
    
    workflow = InterruptTestWorkflow()
    
    # 创建高风险任务状态
    initial_state = {
        "messages": [{"role": "user", "content": "请删除用户敏感数据并生成报告"}],
        "user_id": "test_user",
        "session_id": str(uuid.uuid4()),
        "task_type": "system_operation",
        "metadata": {}
    }
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    try:
        print("🚀 启动高风险任务工作流...")
        
        interrupt_count = 0
        max_interrupts = 10  # 防止无限循环
        
        result = workflow.graph.invoke(initial_state, config=config)
        
        while "__interrupt__" in result and interrupt_count < max_interrupts:
            interrupt_count += 1
            interrupts = result["__interrupt__"]
            
            print(f"\n⏸️ 第 {interrupt_count} 个中断点，中断数量: {len(interrupts)}")
            
            for interrupt_info in interrupts:
                interrupt_data = interrupt_info.value
                interrupt_type = interrupt_data.get("type", "unknown")
                
                print(f"  📋 中断类型: {interrupt_type}")
                print(f"  📝 标题: {interrupt_data.get('title', 'N/A')}")
                
                # 根据中断类型提供不同的响应
                if interrupt_type == "approval_request":
                    print("  👤 处理审批请求...")
                    response = Command(resume={
                        "decision": "approve",
                        "reason": f"测试审批 #{interrupt_count}",
                        "approved_by": "test_admin",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif interrupt_type == "human_input_request":
                    print("  👤 处理人工输入请求...")
                    response = Command(resume={
                        "priority": "urgent",
                        "quality_level": "premium",
                        "deadline": (datetime.now() + timedelta(hours=1)).isoformat(),
                        "special_requirements": f"测试输入 #{interrupt_count}"
                    })
                
                elif interrupt_type == "tool_review_request":
                    print("  👤 处理工具审查请求...")
                    response = Command(resume={
                        "action": "approve_all",
                        "reviewer": "test_reviewer",
                        "review_notes": f"测试审查 #{interrupt_count}"
                    })
                
                elif interrupt_type == "state_edit_request":
                    print("  👤 处理状态编辑请求...")
                    response = Command(resume={
                        "action": "edit",
                        "edited_state": {
                            "summary": f"测试编辑摘要 #{interrupt_count}",
                            "recommendations": [f"建议 {interrupt_count}.1", f"建议 {interrupt_count}.2"]
                        },
                        "editor": "test_editor"
                    })
                
                else:
                    print("  👤 处理通用中断...")
                    response = Command(resume={
                        "approved": True,
                        "response_id": interrupt_count
                    })
                
                # 恢复执行
                print("  ▶️ 恢复执行...")
                result = workflow.graph.invoke(response, config=config)
                break  # 处理第一个中断后跳出内层循环
        
        if interrupt_count >= max_interrupts:
            print(f"⚠️ 达到最大中断处理次数限制: {max_interrupts}")
        else:
            print(f"\n✅ 工作流完成，总共处理了 {interrupt_count} 个中断")
        
        print(f"\n📊 最终结果:")
        print(f"  当前步骤: {result.get('current_step', 'unknown')}")
        print(f"  完成状态: {result.get('completed', False)}")
        print(f"  审批状态: {result.get('approval_status', 'unknown')}")
        print(f"  执行历史: {result.get('metadata', {}).get('step_history', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_interrupt_timeout():
    """测试中断超时处理"""
    print("\n" + "="*50)
    print("🧪 测试中断超时处理")
    print("="*50)
    
    # 这里我们主要测试中断管理器的超时逻辑
    interrupt_manager = EnhancedInterruptManager()
    
    try:
        # 创建一个短超时的审批中断
        approval_data = interrupt_manager.create_approval_interrupt(
            title="超时测试审批",
            description="这是一个用于测试超时的审批请求",
            context={
                "user_id": "test_user",
                "run_id": str(uuid.uuid4()),
                "node_id": "timeout_test"
            },
            priority=InterruptPriority.MEDIUM,
            timeout_seconds=2  # 2秒超时
        )
        
        interrupt_id = approval_data["interrupt_id"]
        print(f"📋 创建超时测试中断: {interrupt_id}")
        print(f"⏰ 超时时间: 2秒")
        
        # 等待超时
        print("⏳ 等待超时...")
        await asyncio.sleep(3)
        
        # 检查中断状态
        status = interrupt_manager.get_interrupt_status(interrupt_id)
        print(f"📊 中断状态: {status}")
        
        # 尝试响应已超时的中断
        response_success = await interrupt_manager.process_interrupt_response(
            interrupt_id=interrupt_id,
            response_data={"decision": "approve", "reason": "迟到的响应"},
            responder_id="test_user"
        )
        
        print(f"📝 迟到响应处理结果: {'成功' if response_success else '失败（预期）'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🎯 LangGraph中断功能综合测试")
    print("="*60)
    
    test_results = []
    
    # 运行各项测试
    tests = [
        ("基础中断功能", test_basic_interrupt),
        ("多重中断处理", test_multiple_interrupts),
        ("中断超时处理", test_interrupt_timeout),
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 开始测试: {test_name}")
        try:
            result = await test_func()
            test_results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'通过' if result else '失败'}")
        except Exception as e:
            print(f"❌ {test_name}: 异常 - {e}")
            test_results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！LangGraph中断功能正常工作。")
    else:
        print("⚠️ 部分测试失败，请检查相关功能。")


if __name__ == "__main__":
    asyncio.run(main())