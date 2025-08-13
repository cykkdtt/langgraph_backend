#!/usr/bin/env python3
"""
测试所有核心模块组件的导入
"""

import sys
import traceback
from typing import List, Tuple

def test_component_imports() -> Tuple[List[str], List[str]]:
    """测试所有组件导入"""
    success_imports = []
    failed_imports = []
    
    # 测试主要模块导入
    test_modules = [
        # 智能体相关
        ("core.agents", [
            "BaseAgent", "AgentType", "AgentStatus", "AgentCapability", 
            "AgentMetadata", "ChatRequest", "ChatResponse", "StreamChunk",
            "MemoryEnhancedAgent", "AgentConfig", "AgentInstance", 
            "AgentRegistry", "AgentFactory", "AgentPerformanceMetrics",
            "AgentManager", "CollaborationMode", "MessageType",
            "CollaborationMessage", "CollaborationTask", "CollaborationContext",
            "AgentCollaborationOrchestrator", "TaskScheduler", "LoadBalancer"
        ]),
        
        # 工具相关
        ("core.tools", [
            "ToolCategory", "ToolPermission", "ToolMetadata",
            "ToolExecutionResult", "ToolExecutionContext", "BaseManagedTool",
            "ToolRegistry"
        ]),
        
        # 记忆管理
        ("core.memory", [
            "MemoryType", "MemoryScope", "MemoryItem", "MemoryQuery",
            "MemoryNamespace", "LangMemManager"
        ]),
        
        # 流式处理
        ("core.streaming", [
            "StreamManager", "StreamMode", "StreamEvent"
        ]),
        
        # 中断处理
        ("core.interrupts", [
            "InterruptType", "InterruptStatus", "InterruptPriority",
            "InterruptRequest", "InterruptResponse", "InterruptContext",
            "InterruptManager"
        ]),
        
        # 工作流编排
        ("core.workflows", [
            "WorkflowBuilder", "WorkflowDefinition", "WorkflowStep", "Condition"
        ]),
        
        # 时间旅行
        ("core.time_travel", [
            "TimeTravelManager", "CheckpointManager", "RollbackManager",
            "StateHistoryManager"
        ]),
        
        # 缓存管理
        ("core.cache", [
            "RedisManager", "SessionCache", "CacheManager"
        ]),
        
        # 检查点管理
        ("core.checkpoint", [
            "CheckpointMetadata", "CheckpointInfo", "CheckpointManager"
        ]),
        
        # 数据库管理
        ("core.database", [
            "DatabaseManager"
        ]),
        
        # 错误处理
        ("core.error", [
            "ErrorSeverity", "ErrorCategory", "ErrorContext", "ErrorInfo",
            "BaseError", "SystemError", "ConfigurationError", "DatabaseError",
            "ConnectionError", "APIError", "AuthenticationError", "AuthorizationError",
            "ErrorHandler", "PerformanceMetric", "PerformanceMonitor"
        ]),
        
        # 日志管理
        ("core.logging", [
            "StructuredFormatter", "LoggerManager"
        ])
    ]
    
    for module_name, components in test_modules:
        print(f"\n测试模块: {module_name}")
        
        for component in components:
            try:
                exec(f"from {module_name} import {component}")
                success_imports.append(f"{module_name}.{component}")
                print(f"  ✓ {component}")
            except Exception as e:
                failed_imports.append(f"{module_name}.{component}: {str(e)}")
                print(f"  ✗ {component}: {str(e)}")
    
    return success_imports, failed_imports


def test_core_module_import():
    """测试核心模块整体导入"""
    print("\n" + "="*60)
    print("测试核心模块整体导入")
    print("="*60)
    
    try:
        import core
        print("✓ 核心模块导入成功")
        
        # 测试一些关键组件
        key_components = [
            "BaseAgent", "AgentRegistry", "ToolRegistry", "LangMemManager",
            "StreamManager", "InterruptManager", "WorkflowBuilder",
            "TimeTravelManager", "RedisManager", "DatabaseManager",
            "ErrorHandler", "LoggerManager"
        ]
        
        available_components = []
        missing_components = []
        
        for component in key_components:
            if hasattr(core, component):
                available_components.append(component)
                print(f"  ✓ {component}")
            else:
                missing_components.append(component)
                print(f"  ✗ {component} (未找到)")
        
        print(f"\n可用组件: {len(available_components)}")
        print(f"缺失组件: {len(missing_components)}")
        
        return True, available_components, missing_components
        
    except Exception as e:
        print(f"✗ 核心模块导入失败: {str(e)}")
        traceback.print_exc()
        return False, [], []


def test_functionality():
    """测试基本功能"""
    print("\n" + "="*60)
    print("测试基本功能")
    print("="*60)
    
    try:
        # 测试错误处理器
        from core.error import get_error_handler
        error_handler = get_error_handler()
        print("✓ 错误处理器创建成功")
        
        # 测试日志管理器
        from core.logging import get_logger
        logger = get_logger("test")
        print("✓ 日志器创建成功")
        
        # 测试工具注册表
        from core.tools import get_tool_registry
        tool_registry = get_tool_registry()
        print("✓ 工具注册表创建成功")
        
        # 测试智能体注册表
        from core.agents import get_agent_registry
        agent_registry = get_agent_registry()
        print("✓ 智能体注册表创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 功能测试失败: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("开始测试所有核心模块组件...")
    
    # 测试组件导入
    success_imports, failed_imports = test_component_imports()
    
    print("\n" + "="*60)
    print("组件导入测试结果")
    print("="*60)
    print(f"成功导入: {len(success_imports)} 个组件")
    print(f"导入失败: {len(failed_imports)} 个组件")
    
    if failed_imports:
        print("\n失败的导入:")
        for failed in failed_imports:
            print(f"  - {failed}")
    
    # 测试核心模块导入
    core_success, available, missing = test_core_module_import()
    
    # 测试基本功能
    if core_success:
        func_success = test_functionality()
    else:
        func_success = False
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"组件导入成功率: {len(success_imports)}/{len(success_imports) + len(failed_imports)}")
    print(f"核心模块导入: {'成功' if core_success else '失败'}")
    print(f"功能测试: {'成功' if func_success else '失败'}")
    
    if len(failed_imports) == 0 and core_success and func_success:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())