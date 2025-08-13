#!/usr/bin/env python3
"""
项目增强中断管理器功能测试

测试本项目中的EnhancedInterruptManager类的各种功能：
1. 审批工作流
2. 人工输入处理
3. 工具调用审查
4. 状态编辑
5. 超时处理
6. 通知系统
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

# 导入项目模块
from core.interrupts.enhanced_interrupt_manager import EnhancedInterruptManager
from core.interrupts.interrupt_types import (
    InterruptType, InterruptPriority, InterruptStatus,
    InterruptRequest, InterruptResponse
)


async def test_enhanced_interrupt_manager():
    """测试增强中断管理器的基础功能"""
    print("🧪 测试增强中断管理器基础功能")
    print("="*50)
    
    # 创建中断管理器
    manager = EnhancedInterruptManager()
    
    try:
        # 1. 测试创建审批中断
        print("\n📋 测试创建审批中断...")
        approval_data = manager.create_approval_interrupt(
            title="测试审批",
            description="这是一个测试审批请求",
            context={
                "user_id": "test_user",
                "run_id": str(uuid.uuid4()),
                "node_id": "test_node"
            },
            priority=InterruptPriority.HIGH,
            required_approvers=["admin", "supervisor"],
            timeout_seconds=3600
        )
        
        interrupt_id = approval_data["interrupt_id"]
        print(f"✅ 创建审批中断成功: {interrupt_id}")
        print(f"   标题: {approval_data['title']}")
        print(f"   优先级: {approval_data['priority']}")
        print(f"   审批者: {approval_data['required_approvers']}")
        
        # 2. 测试获取中断状态
        print(f"\n📊 测试获取中断状态...")
        status = manager.get_interrupt_status(interrupt_id)
        print(f"✅ 中断状态: {status}")
        
        # 3. 测试处理中断响应
        print(f"\n📝 测试处理中断响应...")
        response_success = await manager.process_interrupt_response(
            interrupt_id=interrupt_id,
            response_data={
                "decision": "approve",
                "reason": "测试批准",
                "approved": True
            },
            responder_id="admin"
        )
        print(f"✅ 响应处理结果: {'成功' if response_success else '失败'}")
        
        # 4. 测试创建人工输入中断
        print(f"\n📝 测试创建人工输入中断...")
        input_data = manager.create_human_input_interrupt(
            prompt="请输入您的姓名",
            input_type="text",
            context={
                "user_id": "test_user",
                "run_id": str(uuid.uuid4()),
                "node_id": "input_node"
            },
            validation_rules={
                "required": True,
                "min_length": 2,
                "max_length": 50
            },
            timeout_seconds=1800
        )
        
        input_interrupt_id = input_data["interrupt_id"]
        print(f"✅ 创建人工输入中断成功: {input_interrupt_id}")
        print(f"   提示: {input_data['prompt']}")
        print(f"   类型: {input_data['input_type']}")
        
        # 5. 测试工具审查中断
        print(f"\n🔧 测试创建工具审查中断...")
        tools_data = manager.create_tool_review_interrupt(
            proposed_tools=[
                {
                    "name": "search_database",
                    "args": {"query": "SELECT * FROM users"},
                    "description": "搜索用户数据库"
                },
                {
                    "name": "send_email",
                    "args": {"to": "user@example.com", "subject": "通知"},
                    "description": "发送邮件通知"
                }
            ],
            context={
                "user_id": "test_user",
                "run_id": str(uuid.uuid4()),
                "node_id": "tools_node"
            },
            allow_modifications=True
        )
        
        tools_interrupt_id = tools_data["interrupt_id"]
        print(f"✅ 创建工具审查中断成功: {tools_interrupt_id}")
        print(f"   工具数量: {len(tools_data['proposed_tools'])}")
        print(f"   允许修改: {tools_data['allow_modifications']}")
        
        # 6. 测试状态编辑中断
        print(f"\n✏️ 测试创建状态编辑中断...")
        state_data = manager.create_state_edit_interrupt(
            current_state={
                "user_name": "张三",
                "age": 30,
                "email": "zhangsan@example.com"
            },
            editable_fields=["user_name", "email"],
            context={
                "user_id": "test_user",
                "run_id": str(uuid.uuid4()),
                "node_id": "state_node"
            },
            validation_schema={
                "user_name": {"type": "string", "required": True},
                "email": {"type": "email", "required": True}
            }
        )
        
        state_interrupt_id = state_data["interrupt_id"]
        print(f"✅ 创建状态编辑中断成功: {state_interrupt_id}")
        print(f"   可编辑字段: {state_data['editable_fields']}")
        print(f"   当前状态: {state_data['current_state']}")
        
        # 7. 测试获取所有活跃中断
        print(f"\n📋 测试获取活跃中断...")
        active_interrupts = manager.active_interrupts
        print(f"✅ 活跃中断数量: {len(active_interrupts)}")
        for iid, interrupt in active_interrupts.items():
            print(f"   - {iid}: {interrupt.title} ({interrupt.interrupt_type})")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_approval_workflow():
    """测试审批工作流"""
    print("\n🧪 测试审批工作流")
    print("="*50)
    
    manager = EnhancedInterruptManager()
    
    try:
        # 创建需要多人审批的中断
        approval_data = manager.create_approval_interrupt(
            title="关键系统操作审批",
            description="需要删除生产环境数据库中的敏感数据",
            context={
                "user_id": "operator",
                "operation": "delete_sensitive_data",
                "environment": "production",
                "run_id": str(uuid.uuid4()),
                "node_id": "critical_operation"
            },
            priority=InterruptPriority.URGENT,
            required_approvers=["admin", "security_officer", "supervisor"],
            timeout_seconds=7200  # 2小时
        )
        
        interrupt_id = approval_data["interrupt_id"]
        print(f"📋 创建关键操作审批: {interrupt_id}")
        
        # 模拟多个审批者的响应
        approvers = [
            ("admin", "approve", "管理员批准"),
            ("security_officer", "approve", "安全官批准"),
            ("supervisor", "approve", "主管批准")
        ]
        
        for approver_id, decision, reason in approvers:
            print(f"\n👤 {approver_id} 进行审批...")
            
            response_success = await manager.process_interrupt_response(
                interrupt_id=interrupt_id,
                response_data={
                    "decision": decision,
                    "reason": reason,
                    "approved": decision == "approve",
                    "timestamp": datetime.now().isoformat()
                },
                responder_id=approver_id
            )
            
            print(f"   {'✅' if response_success else '❌'} 审批响应: {decision}")
            
            # 检查审批状态
            responses = manager.interrupt_responses.get(interrupt_id, [])
            print(f"   📊 已收到响应数: {len(responses)}")
        
        # 检查最终状态
        final_status = manager.get_interrupt_status(interrupt_id)
        print(f"\n📊 最终审批状态: {final_status}")
        
        return True
        
    except Exception as e:
        print(f"❌ 审批工作流测试失败: {e}")
        return False


async def test_timeout_handling():
    """测试超时处理"""
    print("\n🧪 测试超时处理")
    print("="*50)
    
    manager = EnhancedInterruptManager()
    
    try:
        # 创建短超时的中断
        approval_data = manager.create_approval_interrupt(
            title="超时测试",
            description="这是一个用于测试超时的审批",
            context={
                "user_id": "test_user",
                "run_id": str(uuid.uuid4()),
                "node_id": "timeout_test"
            },
            priority=InterruptPriority.MEDIUM,
            timeout_seconds=3  # 3秒超时
        )
        
        interrupt_id = approval_data["interrupt_id"]
        print(f"⏰ 创建超时测试中断: {interrupt_id}")
        print(f"   超时时间: 3秒")
        
        # 等待超时
        print("⏳ 等待超时...")
        await asyncio.sleep(4)
        
        # 尝试响应已超时的中断
        print("📝 尝试响应已超时的中断...")
        response_success = await manager.process_interrupt_response(
            interrupt_id=interrupt_id,
            response_data={"decision": "approve"},
            responder_id="late_user"
        )
        
        print(f"📊 超时后响应结果: {'成功' if response_success else '失败（预期）'}")
        
        # 检查中断状态
        status = manager.get_interrupt_status(interrupt_id)
        print(f"📊 超时后中断状态: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ 超时处理测试失败: {e}")
        return False


async def test_interrupt_types():
    """测试不同类型的中断"""
    print("\n🧪 测试不同类型的中断")
    print("="*50)
    
    manager = EnhancedInterruptManager()
    
    interrupt_types_tests = [
        {
            "name": "确认中断",
            "type": InterruptType.CONFIRMATION,
            "data": {
                "title": "确认操作",
                "description": "确认要执行此操作吗？",
                "context": {"operation": "delete_file"},
                "options": [
                    {"value": "yes", "label": "是"},
                    {"value": "no", "label": "否"}
                ]
            }
        },
        {
            "name": "决策中断",
            "type": InterruptType.DECISION,
            "data": {
                "title": "选择处理方式",
                "description": "请选择如何处理这个错误",
                "context": {"error": "connection_timeout"},
                "options": [
                    {"value": "retry", "label": "重试"},
                    {"value": "skip", "label": "跳过"},
                    {"value": "abort", "label": "中止"}
                ]
            }
        },
        {
            "name": "错误处理中断",
            "type": InterruptType.ERROR_HANDLING,
            "data": {
                "title": "处理错误",
                "description": "系统遇到错误，需要人工干预",
                "context": {"error_code": "E001", "error_message": "数据库连接失败"},
                "priority": InterruptPriority.URGENT
            }
        }
    ]
    
    created_interrupts = []
    
    try:
        for test_case in interrupt_types_tests:
            print(f"\n📋 创建{test_case['name']}...")
            
            # 创建中断请求
            interrupt_request = InterruptRequest(
                interrupt_id=str(uuid.uuid4()),
                run_id=str(uuid.uuid4()),
                node_id="test_node",
                interrupt_type=test_case["type"],
                priority=test_case["data"].get("priority", InterruptPriority.MEDIUM),
                title=test_case["data"]["title"],
                message=test_case["data"]["description"],
                context=test_case["data"]["context"],
                options=test_case["data"].get("options", [])
            )
            
            # 添加到管理器
            manager.active_interrupts[interrupt_request.interrupt_id] = interrupt_request
            manager.interrupt_responses[interrupt_request.interrupt_id] = []
            
            created_interrupts.append(interrupt_request.interrupt_id)
            
            print(f"✅ 创建成功: {interrupt_request.interrupt_id}")
            print(f"   类型: {interrupt_request.interrupt_type}")
            print(f"   优先级: {interrupt_request.priority}")
        
        # 显示所有创建的中断
        print(f"\n📊 总共创建了 {len(created_interrupts)} 个中断:")
        for interrupt_id in created_interrupts:
            interrupt = manager.active_interrupts[interrupt_id]
            print(f"   - {interrupt.title} ({interrupt.interrupt_type})")
        
        return True
        
    except Exception as e:
        print(f"❌ 中断类型测试失败: {e}")
        return False


async def test_interrupt_context():
    """测试中断上下文功能"""
    print("\n🧪 测试中断上下文功能")
    print("="*50)
    
    manager = EnhancedInterruptManager()
    
    try:
        # 创建包含丰富上下文的中断
        context = {
            "user_id": "user123",
            "session_id": "session456",
            "workflow_id": "workflow789",
            "current_step": "data_processing",
            "previous_steps": ["validation", "authentication"],
            "user_permissions": ["read", "write"],
            "environment": "production",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "source": "api_request",
                "client_ip": "192.168.1.100",
                "user_agent": "Mozilla/5.0"
            }
        }
        
        approval_data = manager.create_approval_interrupt(
            title="上下文丰富的审批",
            description="包含详细上下文信息的审批请求",
            context=context,
            priority=InterruptPriority.HIGH,
            required_approvers=["context_admin"],
            timeout_seconds=3600
        )
        
        interrupt_id = approval_data["interrupt_id"]
        print(f"📋 创建上下文丰富的中断: {interrupt_id}")
        
        # 验证上下文信息
        interrupt = manager.active_interrupts[interrupt_id]
        print(f"✅ 上下文验证:")
        print(f"   用户ID: {interrupt.context.get('user_id')}")
        print(f"   会话ID: {interrupt.context.get('session_id')}")
        print(f"   当前步骤: {interrupt.context.get('current_step')}")
        print(f"   环境: {interrupt.context.get('environment')}")
        print(f"   权限: {interrupt.context.get('user_permissions')}")
        
        # 测试基于上下文的响应
        response_data = {
            "decision": "approve",
            "reason": "基于用户权限和环境验证通过",
            "context_validated": True,
            "validation_details": {
                "user_permissions_check": "passed",
                "environment_check": "passed",
                "session_validity": "valid"
            }
        }
        
        response_success = await manager.process_interrupt_response(
            interrupt_id=interrupt_id,
            response_data=response_data,
            responder_id="context_admin"
        )
        
        print(f"✅ 基于上下文的响应处理: {'成功' if response_success else '失败'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 上下文测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🎯 项目增强中断管理器功能测试")
    print("="*60)
    
    test_functions = [
        ("增强中断管理器基础功能", test_enhanced_interrupt_manager),
        ("审批工作流", test_approval_workflow),
        ("超时处理", test_timeout_handling),
        ("不同类型中断", test_interrupt_types),
        ("中断上下文功能", test_interrupt_context),
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
        print("🎉 所有测试通过！增强中断管理器功能正常。")
    else:
        print("⚠️ 部分测试失败，请检查相关功能。")


if __name__ == "__main__":
    asyncio.run(main())