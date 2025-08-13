#!/usr/bin/env python3
"""
LangGraph官方中断功能示例

基于官方文档的简单示例，展示interrupt()和Command的基础用法：
- https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/

这个示例展示了：
1. 基础的interrupt()使用
2. Command(resume=value)恢复执行
3. 多种中断场景
4. 状态持久化
"""

import uuid
from typing import TypedDict, Dict, Any
from datetime import datetime

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command


class State(TypedDict):
    """简单的状态定义"""
    messages: list
    user_input: str
    approval_needed: bool
    approved: bool
    final_result: str


def analyze_request(state: State) -> State:
    """分析用户请求"""
    print("🔍 分析用户请求...")
    
    user_message = state.get("messages", [])[-1] if state.get("messages") else ""
    
    # 检查是否需要审批
    sensitive_keywords = ["删除", "支付", "转账", "修改系统"]
    needs_approval = any(keyword in str(user_message) for keyword in sensitive_keywords)
    
    print(f"📋 请求内容: {user_message}")
    print(f"🛡️ 需要审批: {needs_approval}")
    
    return {
        **state,
        "approval_needed": needs_approval
    }


def approval_gate(state: State) -> State:
    """审批门控 - 演示基础interrupt()用法"""
    print("🚪 进入审批门控...")
    
    if not state.get("approval_needed", False):
        print("✅ 无需审批，直接通过")
        return {
            **state,
            "approved": True
        }
    
    print("⚠️ 需要审批，暂停执行等待人工干预...")
    
    # 使用LangGraph官方的interrupt()函数
    # 这会暂停图的执行，等待外部输入
    approval_result = interrupt({
        "type": "approval_request",
        "message": "请审批此操作",
        "details": {
            "user_request": state.get("messages", [])[-1] if state.get("messages") else "",
            "timestamp": datetime.now().isoformat(),
            "risk_level": "high"
        },
        "options": [
            {"value": "approve", "label": "批准"},
            {"value": "reject", "label": "拒绝"}
        ]
    })
    
    print(f"✅ 收到审批结果: {approval_result}")
    
    # 处理审批结果
    if isinstance(approval_result, dict):
        approved = approval_result.get("decision") == "approve"
    else:
        # 如果直接返回字符串
        approved = str(approval_result).lower() in ["approve", "approved", "yes", "true"]
    
    return {
        **state,
        "approved": approved
    }


def collect_user_input(state: State) -> State:
    """收集用户输入 - 演示人工输入场景"""
    print("📝 收集用户输入...")
    
    if not state.get("approved", False):
        print("❌ 未获得审批，跳过用户输入收集")
        return state
    
    print("💬 请提供额外信息...")
    
    # 使用interrupt()收集用户输入
    user_input = interrupt({
        "type": "user_input_request",
        "prompt": "请提供执行此任务所需的额外信息",
        "input_type": "text",
        "placeholder": "请输入详细要求...",
        "validation": {
            "required": True,
            "min_length": 5
        }
    })
    
    print(f"✅ 收到用户输入: {user_input}")
    
    return {
        **state,
        "user_input": str(user_input) if user_input else ""
    }


def process_task(state: State) -> State:
    """处理任务"""
    print("⚡ 处理任务...")
    
    if not state.get("approved", False):
        result = "任务被拒绝，未执行"
    else:
        user_input = state.get("user_input", "")
        original_request = state.get("messages", [])[-1] if state.get("messages") else ""
        
        # 模拟任务处理
        result = f"任务已完成。原始请求: {original_request}，用户输入: {user_input}"
    
    print(f"📊 处理结果: {result}")
    
    return {
        **state,
        "final_result": result
    }


def review_result(state: State) -> State:
    """审查结果 - 演示结果编辑场景"""
    print("📋 审查处理结果...")
    
    current_result = state.get("final_result", "")
    
    if not current_result or "被拒绝" in current_result:
        print("⏭️ 无需审查，直接返回")
        return state
    
    print("🔍 请审查处理结果，可以进行编辑...")
    
    # 使用interrupt()进行结果审查和编辑
    review_result = interrupt({
        "type": "result_review",
        "message": "请审查处理结果",
        "current_result": current_result,
        "actions": [
            {"value": "approve", "label": "批准结果"},
            {"value": "edit", "label": "编辑结果"},
            {"value": "regenerate", "label": "重新生成"}
        ]
    })
    
    print(f"✅ 收到审查结果: {review_result}")
    
    # 处理审查结果
    if isinstance(review_result, dict):
        action = review_result.get("action", "approve")
        if action == "edit":
            final_result = review_result.get("edited_result", current_result)
        elif action == "regenerate":
            final_result = "结果已重新生成: " + current_result
        else:
            final_result = current_result
    else:
        final_result = current_result
    
    return {
        **state,
        "final_result": final_result
    }


def build_graph():
    """构建包含中断的工作流图"""
    workflow = StateGraph(State)
    
    # 添加节点
    workflow.add_node("analyze", analyze_request)
    workflow.add_node("approval", approval_gate)
    workflow.add_node("input", collect_user_input)
    workflow.add_node("process", process_task)
    workflow.add_node("review", review_result)
    
    # 添加边
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "approval")
    workflow.add_edge("approval", "input")
    workflow.add_edge("input", "process")
    workflow.add_edge("process", "review")
    workflow.add_edge("review", END)
    
    # 使用内存检查点保存器
    checkpointer = InMemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)


def test_basic_interrupt():
    """测试基础中断功能"""
    print("🧪 测试基础中断功能")
    print("="*50)
    
    graph = build_graph()
    
    # 创建初始状态
    initial_state = {
        "messages": ["请帮我删除用户数据"],  # 这会触发审批
        "user_input": "",
        "approval_needed": False,
        "approved": False,
        "final_result": ""
    }
    
    # 配置线程ID
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print("🚀 启动工作流...")
    
    # 第一次运行 - 会在审批处中断
    result = graph.invoke(initial_state, config=config)
    
    print(f"\n📊 第一次运行结果:")
    print(f"  当前状态: {result}")
    
    # 检查是否有中断
    if "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        print(f"\n⏸️ 工作流暂停，中断数量: {len(interrupts)}")
        
        for i, interrupt_info in enumerate(interrupts):
            print(f"\n📋 中断 {i+1}:")
            print(f"  ID: {interrupt_info.id}")
            print(f"  值: {interrupt_info.value}")
            print(f"  可恢复: {interrupt_info.resumable}")
        
        # 模拟审批响应
        print("\n👤 模拟管理员审批: 批准")
        approval_response = Command(resume={
            "decision": "approve",
            "approved_by": "admin",
            "timestamp": datetime.now().isoformat()
        })
        
        # 恢复执行
        print("▶️ 恢复工作流执行...")
        result = graph.invoke(approval_response, config=config)
        
        # 检查是否还有中断（用户输入）
        if "__interrupt__" in result:
            interrupts = result["__interrupt__"]
            print(f"\n⏸️ 遇到第二个中断，中断数量: {len(interrupts)}")
            
            for interrupt_info in interrupts:
                print(f"📋 中断信息: {interrupt_info.value}")
            
            # 模拟用户输入
            print("\n👤 模拟用户输入: 请小心处理，这是重要数据")
            input_response = Command(resume="请小心处理，这是重要数据")
            
            # 再次恢复执行
            print("▶️ 再次恢复工作流执行...")
            result = graph.invoke(input_response, config=config)
            
            # 检查是否还有中断（结果审查）
            if "__interrupt__" in result:
                interrupts = result["__interrupt__"]
                print(f"\n⏸️ 遇到第三个中断，中断数量: {len(interrupts)}")
                
                # 模拟结果审查
                print("\n👤 模拟结果审查: 批准结果")
                review_response = Command(resume={
                    "action": "approve",
                    "reviewer": "supervisor"
                })
                
                # 最后一次恢复执行
                print("▶️ 最后一次恢复工作流执行...")
                result = graph.invoke(review_response, config=config)
    
    print(f"\n✅ 工作流完成!")
    print(f"📊 最终结果:")
    print(f"  审批状态: {result.get('approved', False)}")
    print(f"  用户输入: {result.get('user_input', '')}")
    print(f"  最终结果: {result.get('final_result', '')}")
    
    return result


def test_rejection_flow():
    """测试拒绝流程"""
    print("\n🧪 测试拒绝流程")
    print("="*50)
    
    graph = build_graph()
    
    # 创建初始状态
    initial_state = {
        "messages": ["请帮我转账到海外账户"],  # 这会触发审批
        "user_input": "",
        "approval_needed": False,
        "approved": False,
        "final_result": ""
    }
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print("🚀 启动工作流...")
    
    # 运行到审批中断
    result = graph.invoke(initial_state, config=config)
    
    if "__interrupt__" in result:
        print("⏸️ 工作流在审批处暂停")
        
        # 模拟拒绝审批
        print("👤 模拟管理员审批: 拒绝")
        rejection_response = Command(resume={
            "decision": "reject",
            "reason": "高风险操作，不予批准",
            "rejected_by": "security_admin"
        })
        
        # 恢复执行
        result = graph.invoke(rejection_response, config=config)
    
    print(f"\n✅ 拒绝流程完成!")
    print(f"📊 最终结果:")
    print(f"  审批状态: {result.get('approved', False)}")
    print(f"  最终结果: {result.get('final_result', '')}")
    
    return result


def test_no_approval_needed():
    """测试无需审批的流程"""
    print("\n🧪 测试无需审批流程")
    print("="*50)
    
    graph = build_graph()
    
    # 创建低风险初始状态
    initial_state = {
        "messages": ["请帮我查询天气信息"],  # 这不会触发审批
        "user_input": "",
        "approval_needed": False,
        "approved": False,
        "final_result": ""
    }
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print("🚀 启动工作流...")
    
    # 运行工作流
    result = graph.invoke(initial_state, config=config)
    
    # 处理可能的用户输入中断
    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        interrupt_info = interrupts[0]
        
        if "user_input" in str(interrupt_info.value):
            print("📝 提供用户输入")
            response = Command(resume="请提供详细的天气预报")
        elif "review" in str(interrupt_info.value):
            print("📋 审查结果")
            response = Command(resume={"action": "approve"})
        else:
            response = Command(resume="继续")
        
        result = graph.invoke(response, config=config)
    
    print(f"\n✅ 无审批流程完成!")
    print(f"📊 最终结果:")
    print(f"  审批状态: {result.get('approved', False)}")
    print(f"  最终结果: {result.get('final_result', '')}")
    
    return result


def main():
    """主函数"""
    print("🎯 LangGraph官方中断功能示例")
    print("="*60)
    
    try:
        # 测试基础中断功能
        test_basic_interrupt()
        
        # 测试拒绝流程
        test_rejection_flow()
        
        # 测试无需审批流程
        test_no_approval_needed()
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()