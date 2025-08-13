#!/usr/bin/env python3
"""
组件注册验证脚本

验证新创建的组件是否已正确注册到项目模块系统中。
"""

import sys
import traceback
from typing import List, Tuple


def test_import(module_path: str, component_name: str) -> Tuple[bool, str]:
    """测试导入组件"""
    try:
        module = __import__(module_path, fromlist=[component_name])
        component = getattr(module, component_name)
        return True, f"✅ {component_name} 导入成功"
    except ImportError as e:
        return False, f"❌ {component_name} 导入失败 (ImportError): {e}"
    except AttributeError as e:
        return False, f"❌ {component_name} 导入失败 (AttributeError): {e}"
    except Exception as e:
        return False, f"❌ {component_name} 导入失败 (其他错误): {e}"


def main():
    """主函数"""
    print("🔍 开始验证组件注册状态...")
    print("=" * 60)
    
    # 定义要测试的组件
    test_cases = [
        # 核心模块导入测试
        ("core", "BaseAgent"),
        ("core", "AgentType"),
        ("core", "get_tool_registry"),
        ("core", "managed_tool"),
        
        # 增强工具管理器组件
        ("core", "get_enhanced_tool_manager"),
        ("core", "ToolExecutionMode"),
        ("core", "ToolValidationLevel"),
        ("core", "EnhancedToolExecutionContext"),
        ("core", "ToolValidator"),
        ("core", "EnhancedToolManager"),
        
        # 协作优化器组件
        ("core", "get_collaboration_orchestrator"),
        ("core", "CollaborationMode"),
        ("core", "MessageType"),
        ("core", "CollaborationMessage"),
        ("core", "CollaborationTask"),
        ("core", "CollaborationContext"),
        ("core", "AgentCollaborationOrchestrator"),
        ("core", "TaskScheduler"),
        ("core", "LoadBalancer"),
        
        # 子模块导入测试
        ("core.agents", "BaseAgent"),
        ("core.agents", "AgentCollaborationOrchestrator"),
        ("core.tools", "ToolRegistry"),
        ("core.tools", "EnhancedToolManager"),
    ]
    
    # 执行测试
    success_count = 0
    total_count = len(test_cases)
    failed_imports = []
    
    for module_path, component_name in test_cases:
        success, message = test_import(module_path, component_name)
        print(message)
        
        if success:
            success_count += 1
        else:
            failed_imports.append((module_path, component_name, message))
    
    print("=" * 60)
    print(f"📊 测试结果: {success_count}/{total_count} 组件导入成功")
    
    if failed_imports:
        print("\n❌ 失败的导入:")
        for module_path, component_name, error_msg in failed_imports:
            print(f"   {module_path}.{component_name}: {error_msg}")
    else:
        print("\n🎉 所有组件都已成功注册!")
    
    # 测试功能性导入
    print("\n🧪 功能性测试...")
    try:
        from core import get_enhanced_tool_manager, get_collaboration_orchestrator
        
        # 测试获取管理器实例
        tool_manager = get_enhanced_tool_manager()
        orchestrator = get_collaboration_orchestrator()
        
        print("✅ 增强工具管理器实例创建成功")
        print("✅ 协作编排器实例创建成功")
        
        # 测试基本功能
        all_tools = tool_manager.get_all_tools()
        print(f"✅ 工具管理器功能正常，当前工具数量: {len(all_tools)}")
        
        agent_count = len(orchestrator.agents)
        print(f"✅ 协作编排器功能正常，当前智能体数量: {agent_count}")
        
    except Exception as e:
        print(f"❌ 功能性测试失败: {e}")
        traceback.print_exc()
    
    print("\n✨ 验证完成!")


if __name__ == "__main__":
    main()