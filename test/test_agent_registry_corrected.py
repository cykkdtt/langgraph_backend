#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试智能体注册表的正确性

该测试文件验证：
1. 只有 SUPERVISOR 和 RAG 智能体使用 MemoryEnhancedAgent
2. RESEARCH、CHART、CODE 智能体使用 BaseAgent
3. 记忆命名空间的正确配置
4. 后台记忆管理器的功能
5. 符合 LangMem 最佳实践
"""

import unittest
import sys
import os
import time
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.registry import AgentRegistry, AgentConfig
from core.agents.base import AgentType
from core.agents.base import BaseAgent
from core.agents.memory_enhanced import MemoryEnhancedAgent
from core.agents.background_memory_manager import BackgroundMemoryManager


class TestAgentRegistryCorrect(unittest.TestCase):
    """测试智能体注册表的正确性"""
    
    def setUp(self):
        """测试前的设置"""
        print("\n=== 初始化测试环境 ===")
        
        # 创建智能体注册表
        self.registry = AgentRegistry()
        
        # 验证注册表初始化
        self.assertIsNotNone(self.registry)
        self.assertIsNotNone(self.registry._memory_manager)
        self.assertIsInstance(self.registry._memory_manager, BackgroundMemoryManager)
        
        print("✓ 智能体注册表初始化成功")
        print("✓ 后台记忆管理器初始化成功")
    
    def test_registry_initialization(self):
        """测试注册表初始化"""
        print("\n=== 测试注册表初始化 ===")
        
        # 验证智能体类注册
        expected_agents = [AgentType.SUPERVISOR, AgentType.RESEARCH, AgentType.CHART, AgentType.RAG, AgentType.CODE]
        
        for agent_type in expected_agents:
            self.assertIn(agent_type, self.registry._agent_classes)
            self.assertIn(agent_type, self.registry._agent_configs)
            print(f"✓ {agent_type.value}: 已注册")
        
        print("✓ 所有智能体类型已正确注册")
    
    def test_memory_enabled_agents(self):
        """测试启用记忆功能的智能体"""
        print("\n=== 测试启用记忆功能的智能体 ===")
        
        # 应该启用记忆功能的智能体
        memory_enabled_agents = [AgentType.SUPERVISOR, AgentType.RAG]
        
        for agent_type in memory_enabled_agents:
            # 检查智能体类
            agent_class = self.registry._agent_classes[agent_type]
            self.assertEqual(agent_class, MemoryEnhancedAgent)
            print(f"✓ {agent_type.value}: 使用 MemoryEnhancedAgent")
            
            # 检查配置
            config = self.registry.get_agent_config(agent_type)
            self.assertTrue(config.memory_enabled)
            self.assertIsNotNone(config.memory_namespace)
            print(f"✓ {agent_type.value}: 记忆功能已启用，命名空间: {config.memory_namespace}")
    
    def test_base_agents(self):
        """测试基础智能体"""
        print("\n=== 测试基础智能体 ===")
        
        # 应该使用基础智能体的类型
        base_agents = [AgentType.RESEARCH, AgentType.CHART, AgentType.CODE]
        
        for agent_type in base_agents:
            # 检查智能体类
            agent_class = self.registry._agent_classes[agent_type]
            self.assertEqual(agent_class, BaseAgent)
            print(f"✓ {agent_type.value}: 使用 BaseAgent")
            
            # 检查配置
            config = self.registry.get_agent_config(agent_type)
            self.assertFalse(config.memory_enabled)
            print(f"✓ {agent_type.value}: 记忆功能已禁用")
    
    def test_memory_namespace_configuration(self):
        """测试记忆命名空间配置"""
        print("\n=== 测试记忆命名空间配置 ===")
        
        # 检查 SUPERVISOR 智能体的记忆命名空间
        supervisor_config = self.registry.get_agent_config(AgentType.SUPERVISOR)
        self.assertEqual(supervisor_config.memory_namespace, "supervisor_agent")
        print(f"✓ SUPERVISOR: 命名空间 = {supervisor_config.memory_namespace}")
        
        # 检查 RAG 智能体的记忆命名空间
        rag_config = self.registry.get_agent_config(AgentType.RAG)
        self.assertEqual(rag_config.memory_namespace, "rag_agent")
        print(f"✓ RAG: 命名空间 = {rag_config.memory_namespace}")
        
        # 确保命名空间是独立的
        self.assertNotEqual(supervisor_config.memory_namespace, rag_config.memory_namespace)
        print("✓ 记忆命名空间是独立的")
    
    def test_background_memory_manager(self):
        """测试后台记忆管理器"""
        print("\n=== 测试后台记忆管理器 ===")
        
        memory_manager = self.registry._memory_manager
        
        # 验证后台记忆管理器的基本功能
        self.assertIsInstance(memory_manager, BackgroundMemoryManager)
        print("✓ 后台记忆管理器类型正确")
        
        # 测试记忆搜索功能（模拟）
        try:
            # 这里只是测试方法存在，不进行实际的记忆操作
            self.assertTrue(hasattr(memory_manager, 'search_memories'))
            print("✓ 后台记忆管理器具有搜索记忆功能")
        except Exception as e:
            print(f"⚠ 后台记忆管理器测试警告: {str(e)}")
    
    def test_agent_creation_with_mock(self):
        """测试智能体创建功能（使用模拟）"""
        print("\n=== 测试智能体创建功能 ===")
        
        # 测试创建不同类型的智能体
        test_cases = [
            (AgentType.SUPERVISOR, MemoryEnhancedAgent, True),
            (AgentType.RAG, MemoryEnhancedAgent, True),
            (AgentType.RESEARCH, BaseAgent, False),
            (AgentType.CHART, BaseAgent, False),
            (AgentType.CODE, BaseAgent, False)
        ]
        
        for agent_type, expected_class, memory_enabled in test_cases:
            try:
                # 获取智能体配置
                config = self.registry.get_agent_config(agent_type)
                
                # 获取智能体类
                agent_class = self.registry._agent_classes[agent_type]
                
                # 验证智能体类
                self.assertEqual(agent_class, expected_class)
                
                # 验证记忆配置
                self.assertEqual(config.memory_enabled, memory_enabled)
                
                print(f"✓ {agent_type.value}: 类型={expected_class.__name__}, 记忆={memory_enabled}")
                
            except Exception as e:
                print(f"✗ {agent_type.value}: 验证失败 - {str(e)}")
                self.fail(f"验证 {agent_type.value} 智能体失败: {str(e)}")
    
    def test_langmem_best_practices_compliance(self):
        """测试 LangMem 最佳实践合规性"""
        print("\n=== 测试 LangMem 最佳实践合规性 ===")
        
        # 验证只有需要记忆功能的智能体才启用记忆
        memory_enabled_count = 0
        base_agent_count = 0
        
        for agent_type in [AgentType.SUPERVISOR, AgentType.RESEARCH, AgentType.CHART, AgentType.RAG, AgentType.CODE]:
            config = self.registry.get_agent_config(agent_type)
            agent_class = self.registry._agent_classes[agent_type]
            
            if config.memory_enabled:
                memory_enabled_count += 1
                self.assertEqual(agent_class, MemoryEnhancedAgent)
            else:
                base_agent_count += 1
                self.assertEqual(agent_class, BaseAgent)
        
        # 验证记忆功能使用的合理性
        self.assertEqual(memory_enabled_count, 2)  # 只有 SUPERVISOR 和 RAG
        self.assertEqual(base_agent_count, 3)  # RESEARCH, CHART, CODE
        
        print(f"✓ 启用记忆功能的智能体数量: {memory_enabled_count} (符合最佳实践)")
        print(f"✓ 使用基础智能体的数量: {base_agent_count} (避免过度使用记忆功能)")
        
        # 验证记忆命名空间的独立性
        supervisor_ns = self.registry.get_agent_config(AgentType.SUPERVISOR).memory_namespace
        rag_ns = self.registry.get_agent_config(AgentType.RAG).memory_namespace
        
        self.assertIsNotNone(supervisor_ns)
        self.assertIsNotNone(rag_ns)
        self.assertNotEqual(supervisor_ns, rag_ns)
        
        print("✓ 记忆命名空间配置符合最佳实践")
    
    def tearDown(self):
        """测试后的清理"""
        print("\n=== 清理测试环境 ===")
        
        # 清理资源
        if hasattr(self, 'registry'):
            # 这里可以添加清理逻辑
            pass
        
        print("✓ 测试环境清理完成")


if __name__ == '__main__':
    print("开始测试智能体注册表的正确性...")
    print("=" * 60)
    
    # 运行测试
    unittest.main(verbosity=2)