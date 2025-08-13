#!/usr/bin/env python3
"""
基于LangGraph官方文档的Human-in-the-Loop实际工作流测试

参考文档：
- https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
- https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/

展示如何在实际工作流中集成人工干预功能
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Annotated
from datetime import datetime

# LangGraph相关导入
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# 项目模块导入
from core.interrupts.enhanced_interrupt_manager import EnhancedInterruptManager
from core.interrupts.interrupt_types import InterruptType, InterruptPriority


# 定义状态结构
class WorkflowState(dict):
    """工作流状态"""
    messages: Annotated[List[Dict], add_messages]
    user_input: Optional[str] = None
    approval_status: Optional[str] = None
    task_result: Optional[str] = None
    error_count: int = 0


class HumanInLoopWorkflow:
    """Human-in-the-Loop工作流示例"""
    
    def __init__(self):
        self.interrupt_manager = EnhancedInterruptManager()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建包含人工干预的工作流图"""
        
        # 创建状态图
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("analyze_request", self.analyze_request)
        workflow.add_node("approval_gate", self.approval_gate)
        workflow.add_node("collect_user_input", self.collect_user_input)
        workflow.add_node("process_task", self.process_task)
        workflow.add_node("review_result", self.review_result)
        workflow.add_node("handle_error", self.handle_error)
        
        # 定义流程
        workflow.add_edge(START, "analyze_request")
        workflow.add_edge("analyze_request", "approval_gate")
        workflow.add_edge("approval_gate", "collect_user_input")
        workflow.add_edge("collect_user_input", "process_task")
        workflow.add_edge("process_task", "review_result")
        workflow.add_edge("review_result", END)
        workflow.add_edge("handle_error", "analyze_request")
        
        # 编译图
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory, interrupt_before=["approval_gate", "collect_user_input"])
    
    async def analyze_request(self, state: WorkflowState) -> WorkflowState:
        """分析请求"""
        print("🔍 分析请求...")
        
        # 模拟请求分析
        request_type = state.get("request_type", "unknown")
        risk_level = state.get("risk_level", "medium")
        
        analysis_result = {
            "type": request_type,
            "risk": risk_level,
            "requires_approval": risk_level in ["high", "critical"],
            "timestamp": datetime.now().isoformat()
        }
        
        state["analysis"] = analysis_result
        state["messages"] = [{"role": "system", "content": f"请求分析完成: {analysis_result}"}]
        
        print(f"✅ 分析结果: {analysis_result}")
        return state
    
    async def approval_gate(self, state: WorkflowState) -> WorkflowState:
        """审批门控"""
        print("🚪 进入审批门控...")
        
        analysis = state.get("analysis", {})
        
        if analysis.get("requires_approval", False):
            print("⚠️ 需要审批，创建中断...")
            
            # 使用interrupt()函数创建中断
            approval_request = interrupt({
                "type": "approval",
                "title": "高风险操作审批",
                "description": f"请求类型: {analysis.get('type')}, 风险级别: {analysis.get('risk')}",
                "context": {
                    "analysis": analysis,
                    "state": dict(state)
                },
                "options": [
                    {"value": "approve", "label": "批准"},
                    {"value": "reject", "label": "拒绝"},
                    {"value": "modify", "label": "修改后批准"}
                ]
            })
            
            print(f"📋 创建审批中断: {approval_request}")
            
            # 等待审批结果（这里会被中断）
            # 在实际应用中，这里会暂停执行直到收到人工响应
            state["approval_status"] = "pending"
        else:
            print("✅ 低风险操作，自动通过")
            state["approval_status"] = "auto_approved"
        
        return state
    
    async def collect_user_input(self, state: WorkflowState) -> WorkflowState:
        """收集用户输入"""
        print("📝 收集用户输入...")
        
        approval_status = state.get("approval_status")
        
        if approval_status == "pending":
            print("⏳ 等待审批结果...")
            # 这里会被中断，等待人工审批
            return state
        elif approval_status == "reject":
            print("❌ 审批被拒绝，终止流程")
            state["task_result"] = "rejected"
            return state
        
        # 如果需要额外的用户输入
        if state.get("needs_user_input", False):
            print("📋 需要用户输入，创建中断...")
            
            user_input_request = interrupt({
                "type": "user_input",
                "prompt": "请提供额外的参数信息",
                "input_type": "text",
                "validation": {
                    "required": True,
                    "min_length": 1
                }
            })
            
            print(f"📝 创建用户输入中断: {user_input_request}")
        else:
            print("✅ 无需额外用户输入")
            state["user_input"] = "default_input"
        
        return state
    
    async def process_task(self, state: WorkflowState) -> WorkflowState:
        """处理任务"""
        print("⚙️ 处理任务...")
        
        try:
            # 模拟任务处理
            task_type = state.get("analysis", {}).get("type", "unknown")
            user_input = state.get("user_input", "")
            
            # 模拟可能的错误
            if state.get("error_count", 0) < 1 and task_type == "error_test":
                state["error_count"] += 1
                raise Exception("模拟任务处理错误")
            
            result = {
                "status": "completed",
                "task_type": task_type,
                "input_used": user_input,
                "timestamp": datetime.now().isoformat(),
                "output": f"处理结果: {task_type} - {user_input}"
            }
            
            state["task_result"] = result
            state["messages"].append({"role": "system", "content": f"任务处理完成: {result}"})
            
            print(f"✅ 任务完成: {result}")
            
        except Exception as e:
            print(f"❌ 任务处理失败: {e}")
            state["error"] = str(e)
            # 使用Command重定向到错误处理
            return Command(goto="handle_error")
        
        return state
    
    async def review_result(self, state: WorkflowState) -> WorkflowState:
        """审查结果"""
        print("👀 审查结果...")
        
        task_result = state.get("task_result")
        
        if isinstance(task_result, dict) and task_result.get("status") == "completed":
            # 对于关键结果，可能需要人工审查
            if state.get("analysis", {}).get("risk") == "critical":
                print("🔍 关键结果需要人工审查...")
                
                review_request = interrupt({
                    "type": "review",
                    "title": "结果审查",
                    "description": "请审查任务执行结果",
                    "context": {
                        "result": task_result,
                        "analysis": state.get("analysis")
                    },
                    "options": [
                        {"value": "accept", "label": "接受结果"},
                        {"value": "reject", "label": "拒绝结果"},
                        {"value": "modify", "label": "修改结果"}
                    ]
                })
                
                print(f"👁️ 创建结果审查中断: {review_request}")
            else:
                print("✅ 结果自动通过审查")
                state["review_status"] = "auto_approved"
        else:
            print("❌ 无有效结果可审查")
            state["review_status"] = "no_result"
        
        return state
    
    async def handle_error(self, state: WorkflowState) -> WorkflowState:
        """处理错误"""
        print("🚨 处理错误...")
        
        error = state.get("error", "未知错误")
        error_count = state.get("error_count", 0)
        
        print(f"❌ 错误信息: {error}")
        print(f"📊 错误次数: {error_count}")
        
        if error_count >= 3:
            print("🛑 错误次数过多，终止流程")
            state["task_result"] = "failed_max_retries"
            return Command(goto=END)
        
        # 创建错误处理中断
        error_handling_request = interrupt({
            "type": "error_handling",
            "title": "错误处理",
            "description": f"处理过程中发生错误: {error}",
            "context": {
                "error": error,
                "error_count": error_count,
                "state": dict(state)
            },
            "options": [
                {"value": "retry", "label": "重试"},
                {"value": "skip", "label": "跳过"},
                {"value": "abort", "label": "中止"}
            ]
        })
        
        print(f"🔧 创建错误处理中断: {error_handling_request}")
        
        # 清除错误状态
        if "error" in state:
            del state["error"]
        
        return state


async def test_basic_workflow():
    """测试基础工作流"""
    print("🧪 测试基础工作流")
    print("="*50)
    
    workflow = HumanInLoopWorkflow()
    
    # 测试配置
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # 初始状态
    initial_state = {
        "request_type": "data_processing",
        "risk_level": "low",
        "needs_user_input": False,
        "messages": []
    }
    
    try:
        print("🚀 启动工作流...")
        
        # 运行工作流
        result = await workflow.graph.ainvoke(initial_state, config)
        
        print("✅ 工作流完成")
        print(f"📊 最终状态: {result.get('task_result')}")
        print(f"📝 消息数量: {len(result.get('messages', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工作流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_approval_workflow():
    """测试需要审批的工作流"""
    print("\n🧪 测试审批工作流")
    print("="*50)
    
    workflow = HumanInLoopWorkflow()
    
    # 测试配置
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # 高风险操作状态
    initial_state = {
        "request_type": "system_modification",
        "risk_level": "high",
        "needs_user_input": True,
        "messages": []
    }
    
    try:
        print("🚀 启动高风险工作流...")
        
        # 运行到第一个中断点
        result = None
        async for event in workflow.graph.astream(initial_state, config):
            print(f"📊 事件: {event}")
            if "approval_gate" in event:
                result = event["approval_gate"]
                break
        
        if result:
            print("⏸️ 工作流在审批门控处中断")
            print(f"📋 审批状态: {result.get('approval_status')}")
        else:
            print("⚠️ 未达到预期的中断点")
        
        return True
        
    except Exception as e:
        print(f"❌ 审批工作流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_handling_workflow():
    """测试错误处理工作流"""
    print("\n🧪 测试错误处理工作流")
    print("="*50)
    
    workflow = HumanInLoopWorkflow()
    
    # 测试配置
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # 错误测试状态
    initial_state = {
        "request_type": "error_test",
        "risk_level": "low",
        "needs_user_input": False,
        "messages": []
    }
    
    try:
        print("🚀 启动错误测试工作流...")
        
        # 运行工作流
        events = []
        async for event in workflow.graph.astream(initial_state, config):
            events.append(event)
            print(f"📊 事件: {list(event.keys())}")
            
            # 检查是否到达错误处理
            if "handle_error" in event:
                print("🚨 到达错误处理节点")
                break
        
        print(f"📊 总共处理了 {len(events)} 个事件")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误处理工作流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_state_persistence():
    """测试状态持久化"""
    print("\n🧪 测试状态持久化")
    print("="*50)
    
    workflow = HumanInLoopWorkflow()
    
    # 使用固定的线程ID来测试持久化
    thread_id = "test_persistence_thread"
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "request_type": "persistence_test",
        "risk_level": "medium",
        "messages": []
    }
    
    try:
        print("💾 测试状态保存...")
        
        # 第一次运行
        print("🚀 第一次运行工作流...")
        first_result = await workflow.graph.ainvoke(initial_state, config)
        
        print(f"📊 第一次运行结果: {first_result.get('task_result')}")
        
        # 获取状态快照
        state_snapshot = workflow.graph.get_state(config)
        print(f"💾 状态快照: {state_snapshot}")
        
        # 第二次运行（应该从保存的状态继续）
        print("🔄 第二次运行工作流...")
        second_result = await workflow.graph.ainvoke({"additional_data": "test"}, config)
        
        print(f"📊 第二次运行结果: {second_result.get('task_result')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 状态持久化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("🎯 LangGraph Human-in-the-Loop 工作流测试")
    print("="*60)
    
    test_functions = [
        ("基础工作流", test_basic_workflow),
        ("审批工作流", test_approval_workflow),
        ("错误处理工作流", test_error_handling_workflow),
        ("状态持久化", test_state_persistence),
    ]
    
    results = []
    
    for test_name, test_func in test_functions:
        print(f"\n🧪 开始测试: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'通过' if result else '失败'}")
        except Exception as e:
            print(f"❌ {test_name}: 异常 - {e}")
            results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！Human-in-the-Loop工作流功能正常。")
    else:
        print("⚠️ 部分测试失败，请检查相关功能。")


if __name__ == "__main__":
    asyncio.run(main())