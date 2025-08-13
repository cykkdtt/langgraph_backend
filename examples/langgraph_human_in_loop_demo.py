#!/usr/bin/env python3
"""
LangGraph官方Human-in-the-Loop功能集成示例

演示如何使用LangGraph v1.0+的新版interrupt()函数和Command原语
实现人工干预功能，包括四种典型设计模式。
"""

import asyncio
import uuid
from typing import TypedDict, Annotated, Any, Dict, List
from datetime import datetime

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.graph.message import add_messages

# 导入项目现有的中断类型
from core.interrupts.interrupt_types import (
    InterruptType, InterruptPriority, InterruptContext
)


class AgentState(TypedDict):
    """智能体状态定义"""
    messages: Annotated[List[Dict[str, Any]], add_messages]
    user_id: str
    current_task: str
    approval_required: bool
    tool_calls: List[Dict[str, Any]]
    human_input: Dict[str, Any]
    execution_context: Dict[str, Any]


class LangGraphHumanInLoopDemo:
    """LangGraph Human-in-the-Loop 功能演示"""
    
    def __init__(self):
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """构建包含人工干预的图"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("analyze_request", self._analyze_request)
        workflow.add_node("approval_gate", self._approval_gate)
        workflow.add_node("collect_human_input", self._collect_human_input)
        workflow.add_node("review_tool_calls", self._review_tool_calls)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("validate_output", self._validate_output)
        workflow.add_node("finalize_response", self._finalize_response)
        
        # 添加边
        workflow.add_edge(START, "analyze_request")
        workflow.add_edge("analyze_request", "approval_gate")
        workflow.add_edge("approval_gate", "collect_human_input")
        workflow.add_edge("collect_human_input", "review_tool_calls")
        workflow.add_edge("review_tool_calls", "execute_tools")
        workflow.add_edge("execute_tools", "validate_output")
        workflow.add_edge("validate_output", "finalize_response")
        workflow.add_edge("finalize_response", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _analyze_request(self, state: AgentState) -> AgentState:
        """分析请求并确定是否需要人工干预"""
        print(f"🔍 分析请求: {state['current_task']}")
        
        # 模拟分析逻辑
        sensitive_keywords = ["删除", "支付", "转账", "重要"]
        approval_required = any(keyword in state['current_task'] for keyword in sensitive_keywords)
        
        return {
            **state,
            "approval_required": approval_required,
            "execution_context": {
                "analysis_time": datetime.now().isoformat(),
                "risk_level": "high" if approval_required else "low"
            }
        }
    
    async def _approval_gate(self, state: AgentState) -> AgentState:
        """模式1: 批准或拒绝 - 在关键步骤前暂停审批"""
        if not state["approval_required"]:
            print("✅ 无需审批，直接通过")
            return state
        
        print("⏸️ 需要人工审批，暂停执行...")
        
        # 使用LangGraph官方interrupt()函数
        approval_result = interrupt({
            "type": "approval_request",
            "title": "任务执行审批",
            "description": f"请审批以下任务: {state['current_task']}",
            "context": {
                "user_id": state["user_id"],
                "task": state["current_task"],
                "risk_level": state["execution_context"]["risk_level"]
            },
            "options": [
                {"value": "approve", "label": "批准执行"},
                {"value": "reject", "label": "拒绝执行"},
                {"value": "modify", "label": "修改后执行"}
            ]
        })
        
        print(f"📋 收到审批结果: {approval_result}")
        
        if approval_result.get("decision") == "reject":
            # 如果被拒绝，可以提前结束或采取其他行动
            return {
                **state,
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": "任务被拒绝执行"
                }]
            }
        
        return {
            **state,
            "execution_context": {
                **state["execution_context"],
                "approval_result": approval_result,
                "approved_at": datetime.now().isoformat()
            }
        }
    
    async def _collect_human_input(self, state: AgentState) -> AgentState:
        """模式4: 验证人工输入 - 收集必要的人工输入"""
        print("📝 收集人工输入...")
        
        # 使用interrupt()收集人工输入
        human_input = interrupt({
            "type": "human_input_request",
            "prompt": "请提供执行此任务所需的额外信息",
            "input_type": "form",
            "fields": [
                {
                    "name": "priority",
                    "type": "select",
                    "label": "任务优先级",
                    "options": ["low", "medium", "high", "urgent"],
                    "required": True
                },
                {
                    "name": "deadline",
                    "type": "datetime",
                    "label": "截止时间",
                    "required": False
                },
                {
                    "name": "notes",
                    "type": "text",
                    "label": "备注信息",
                    "required": False
                }
            ],
            "validation_rules": {
                "priority": {"required": True},
                "deadline": {"format": "datetime"}
            }
        })
        
        print(f"📥 收到人工输入: {human_input}")
        
        return {
            **state,
            "human_input": human_input,
            "execution_context": {
                **state["execution_context"],
                "human_input_received_at": datetime.now().isoformat()
            }
        }
    
    async def _review_tool_calls(self, state: AgentState) -> AgentState:
        """模式3: 审查工具调用 - 在工具执行前审查和编辑"""
        print("🔧 准备工具调用...")
        
        # 模拟生成工具调用
        proposed_tools = [
            {
                "tool_name": "search_database",
                "arguments": {"query": state["current_task"]},
                "description": "搜索相关数据"
            },
            {
                "tool_name": "generate_report",
                "arguments": {"format": "pdf", "include_charts": True},
                "description": "生成报告"
            }
        ]
        
        print("⏸️ 暂停以审查工具调用...")
        
        # 使用interrupt()审查工具调用
        reviewed_tools = interrupt({
            "type": "tool_review_request",
            "title": "工具调用审查",
            "description": "请审查以下工具调用是否合适",
            "proposed_tools": proposed_tools,
            "context": {
                "task": state["current_task"],
                "user_input": state["human_input"]
            },
            "actions": [
                {"value": "approve_all", "label": "批准所有工具"},
                {"value": "modify", "label": "修改工具调用"},
                {"value": "reject_all", "label": "拒绝所有工具"}
            ]
        })
        
        print(f"🔍 工具调用审查结果: {reviewed_tools}")
        
        return {
            **state,
            "tool_calls": reviewed_tools.get("approved_tools", proposed_tools),
            "execution_context": {
                **state["execution_context"],
                "tools_reviewed_at": datetime.now().isoformat()
            }
        }
    
    async def _execute_tools(self, state: AgentState) -> AgentState:
        """执行已审查的工具调用"""
        print("⚡ 执行工具调用...")
        
        results = []
        for tool_call in state["tool_calls"]:
            print(f"  🔧 执行: {tool_call['tool_name']}")
            # 模拟工具执行
            result = {
                "tool_name": tool_call["tool_name"],
                "result": f"模拟执行结果 for {tool_call['tool_name']}",
                "executed_at": datetime.now().isoformat()
            }
            results.append(result)
        
        return {
            **state,
            "execution_context": {
                **state["execution_context"],
                "tool_results": results,
                "tools_executed_at": datetime.now().isoformat()
            }
        }
    
    async def _validate_output(self, state: AgentState) -> AgentState:
        """模式2: 编辑图状态 - 验证和编辑输出"""
        print("✅ 验证输出结果...")
        
        # 生成初始输出
        initial_output = {
            "task": state["current_task"],
            "results": state["execution_context"]["tool_results"],
            "priority": state["human_input"].get("priority", "medium"),
            "completed_at": datetime.now().isoformat()
        }
        
        print("⏸️ 暂停以验证输出...")
        
        # 使用interrupt()验证输出
        validated_output = interrupt({
            "type": "output_validation",
            "title": "输出结果验证",
            "description": "请验证以下输出结果是否正确",
            "initial_output": initial_output,
            "validation_options": [
                {"value": "approve", "label": "输出正确"},
                {"value": "edit", "label": "需要编辑"},
                {"value": "regenerate", "label": "重新生成"}
            ],
            "editable_fields": ["results", "priority", "notes"]
        })
        
        print(f"📊 输出验证结果: {validated_output}")
        
        return {
            **state,
            "execution_context": {
                **state["execution_context"],
                "final_output": validated_output.get("final_output", initial_output),
                "validated_at": datetime.now().isoformat()
            }
        }
    
    async def _finalize_response(self, state: AgentState) -> AgentState:
        """完成响应"""
        print("🎉 任务执行完成!")
        
        final_message = {
            "role": "assistant",
            "content": f"任务 '{state['current_task']}' 已完成执行",
            "metadata": {
                "execution_summary": state["execution_context"],
                "human_interactions": {
                    "approval_required": state["approval_required"],
                    "human_input_provided": bool(state["human_input"]),
                    "tools_reviewed": len(state["tool_calls"]) > 0
                }
            }
        }
        
        return {
            **state,
            "messages": state["messages"] + [final_message]
        }
    
    async def run_demo(self, task: str, user_id: str = "demo_user"):
        """运行演示"""
        print(f"🚀 开始执行任务: {task}")
        print("=" * 60)
        
        # 创建初始状态
        initial_state = {
            "messages": [{
                "role": "user",
                "content": task
            }],
            "user_id": user_id,
            "current_task": task,
            "approval_required": False,
            "tool_calls": [],
            "human_input": {},
            "execution_context": {}
        }
        
        # 配置线程
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        try:
            # 运行图直到第一个中断
            result = self.graph.invoke(initial_state, config=config)
            
            # 检查是否有中断
            if "__interrupt__" in result:
                print("\n🔄 检测到中断，需要人工干预...")
                for interrupt_info in result["__interrupt__"]:
                    print(f"中断类型: {interrupt_info.get('value', {}).get('type', 'unknown')}")
                    print(f"中断描述: {interrupt_info.get('value', {}).get('description', 'N/A')}")
                
                return config, result
            else:
                print("\n✅ 任务完成，无需人工干预")
                return config, result
                
        except Exception as e:
            print(f"❌ 执行出错: {e}")
            return None, None
    
    async def resume_with_response(self, config: Dict, response_data: Dict):
        """使用Command原语恢复执行"""
        print(f"\n🔄 使用响应数据恢复执行: {response_data}")
        
        try:
            # 使用Command原语恢复
            result = self.graph.invoke(Command(resume=response_data), config=config)
            
            # 检查是否还有更多中断
            if "__interrupt__" in result:
                print("\n⏸️ 检测到更多中断...")
                return config, result
            else:
                print("\n🎉 任务完全完成!")
                return config, result
                
        except Exception as e:
            print(f"❌ 恢复执行出错: {e}")
            return None, None


async def main():
    """主演示函数"""
    demo = LangGraphHumanInLoopDemo()
    
    # 演示1: 需要审批的敏感任务
    print("📋 演示1: 敏感任务审批流程")
    print("=" * 60)
    
    config, result = await demo.run_demo("删除用户数据库中的过期记录", "user123")
    
    if config and "__interrupt__" in result:
        # 模拟人工审批
        approval_response = {
            "decision": "approve",
            "approver_id": "admin001",
            "reason": "已确认删除范围合理",
            "conditions": ["仅删除30天前的记录", "保留审计日志"]
        }
        
        print(f"\n👤 管理员审批: {approval_response}")
        config, result = await demo.resume_with_response(config, approval_response)
        
        if config and "__interrupt__" in result:
            # 模拟人工输入
            human_input_response = {
                "priority": "high",
                "deadline": "2024-12-25T18:00:00",
                "notes": "年底数据清理任务"
            }
            
            print(f"\n📝 提供人工输入: {human_input_response}")
            config, result = await demo.resume_with_response(config, human_input_response)
            
            if config and "__interrupt__" in result:
                # 模拟工具审查
                tool_review_response = {
                    "action": "modify",
                    "approved_tools": [
                        {
                            "tool_name": "search_database",
                            "arguments": {"query": "过期记录", "limit": 1000},
                            "description": "搜索过期数据（限制1000条）"
                        }
                    ]
                }
                
                print(f"\n🔧 工具审查结果: {tool_review_response}")
                config, result = await demo.resume_with_response(config, tool_review_response)
                
                if config and "__interrupt__" in result:
                    # 模拟输出验证
                    validation_response = {
                        "action": "approve",
                        "final_output": {
                            "task": "删除用户数据库中的过期记录",
                            "results": "已安全删除856条过期记录",
                            "priority": "high",
                            "notes": "删除操作已完成，审计日志已保存"
                        }
                    }
                    
                    print(f"\n✅ 输出验证: {validation_response}")
                    config, result = await demo.resume_with_response(config, validation_response)
    
    print("\n" + "=" * 60)
    print("📋 演示2: 普通任务（无需人工干预）")
    print("=" * 60)
    
    # 演示2: 普通任务
    await demo.run_demo("查询今天的天气情况", "user456")


if __name__ == "__main__":
    asyncio.run(main())