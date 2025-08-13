#!/usr/bin/env python3
"""
时间旅行API测试脚本

测试时间旅行API的各个端点功能
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, Any

import httpx
from pydantic import BaseModel


class TimeTravelAPITester:
    """时间旅行API测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1/time-travel"
        self.test_thread_id = f"test_thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    async def test_api_endpoints(self):
        """测试所有API端点"""
        print("🚀 开始测试时间旅行API端点...")
        print(f"📍 API基础URL: {self.api_base}")
        print(f"🧵 测试会话ID: {self.test_thread_id}")
        print("-" * 60)
        
        async with httpx.AsyncClient() as client:
            # 测试配置获取
            await self._test_get_config(client)
            
            # 测试快照管理
            await self._test_snapshot_management(client)
            
            # 测试检查点管理
            await self._test_checkpoint_management(client)
            
            # 测试分支管理
            await self._test_branch_management(client)
            
            # 测试执行历史
            await self._test_execution_history(client)
            
            # 测试状态查询
            await self._test_status_query(client)
            
            # 测试数据清理
            await self._test_cleanup(client)
        
        print("-" * 60)
        print("✅ 时间旅行API测试完成!")
    
    async def _test_get_config(self, client: httpx.AsyncClient):
        """测试配置获取"""
        print("📋 测试配置获取...")
        try:
            response = await client.get(f"{self.api_base}/config")
            if response.status_code == 200:
                config = response.json()
                print(f"   ✅ 配置获取成功: {json.dumps(config, indent=2, ensure_ascii=False)}")
            else:
                print(f"   ❌ 配置获取失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ 配置获取异常: {e}")
    
    async def _test_snapshot_management(self, client: httpx.AsyncClient):
        """测试快照管理"""
        print("📸 测试快照管理...")
        
        # 创建快照
        try:
            snapshot_data = {
                "thread_id": self.test_thread_id,
                "snapshot_type": "manual",
                "description": "测试快照",
                "metadata": {"test": True, "created_by": "api_test"}
            }
            
            response = await client.post(f"{self.api_base}/snapshots", json=snapshot_data)
            if response.status_code == 200:
                snapshot = response.json()
                snapshot_id = snapshot["snapshot_id"]
                print(f"   ✅ 快照创建成功: {snapshot_id}")
                
                # 获取快照列表
                response = await client.get(f"{self.api_base}/snapshots/{self.test_thread_id}")
                if response.status_code == 200:
                    snapshots = response.json()
                    print(f"   ✅ 快照列表获取成功: 共{len(snapshots)}个快照")
                else:
                    print(f"   ❌ 快照列表获取失败: {response.status_code}")
                
                # 删除快照
                response = await client.delete(f"{self.api_base}/snapshots/{snapshot_id}")
                if response.status_code == 200:
                    print(f"   ✅ 快照删除成功")
                else:
                    print(f"   ❌ 快照删除失败: {response.status_code}")
                    
            else:
                print(f"   ❌ 快照创建失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ 快照管理异常: {e}")
    
    async def _test_checkpoint_management(self, client: httpx.AsyncClient):
        """测试检查点管理"""
        print("🔖 测试检查点管理...")
        
        try:
            checkpoint_data = {
                "thread_id": self.test_thread_id,
                "checkpoint_name": "test_checkpoint",
                "description": "测试检查点",
                "auto_cleanup": True
            }
            
            response = await client.post(f"{self.api_base}/checkpoints", json=checkpoint_data)
            if response.status_code == 200:
                checkpoint = response.json()
                checkpoint_id = checkpoint["checkpoint_id"]
                print(f"   ✅ 检查点创建成功: {checkpoint_id}")
                
                # 获取检查点列表
                response = await client.get(f"{self.api_base}/checkpoints/{self.test_thread_id}")
                if response.status_code == 200:
                    checkpoints = response.json()
                    print(f"   ✅ 检查点列表获取成功: 共{len(checkpoints)}个检查点")
                else:
                    print(f"   ❌ 检查点列表获取失败: {response.status_code}")
                
                # 测试恢复检查点
                restore_data = {
                    "checkpoint_id": checkpoint_id,
                    "create_backup": True
                }
                response = await client.post(f"{self.api_base}/checkpoints/restore", json=restore_data)
                if response.status_code == 200:
                    print(f"   ✅ 检查点恢复成功")
                else:
                    print(f"   ❌ 检查点恢复失败: {response.status_code}")
                
                # 删除检查点
                response = await client.delete(f"{self.api_base}/checkpoints/{checkpoint_id}")
                if response.status_code == 200:
                    print(f"   ✅ 检查点删除成功")
                else:
                    print(f"   ❌ 检查点删除失败: {response.status_code}")
                    
            else:
                print(f"   ❌ 检查点创建失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ 检查点管理异常: {e}")
    
    async def _test_branch_management(self, client: httpx.AsyncClient):
        """测试分支管理"""
        print("🌿 测试分支管理...")
        
        try:
            branch_data = {
                "thread_id": self.test_thread_id,
                "branch_name": "test_branch",
                "from_step": 0,
                "description": "测试分支"
            }
            
            response = await client.post(f"{self.api_base}/branches", json=branch_data)
            if response.status_code == 200:
                branch = response.json()
                branch_id = branch["branch_id"]
                print(f"   ✅ 分支创建成功: {branch_id}")
                
                # 获取分支列表
                response = await client.get(f"{self.api_base}/branches/{self.test_thread_id}")
                if response.status_code == 200:
                    branches = response.json()
                    print(f"   ✅ 分支列表获取成功: 共{len(branches)}个分支")
                else:
                    print(f"   ❌ 分支列表获取失败: {response.status_code}")
                    
            else:
                print(f"   ❌ 分支创建失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ 分支管理异常: {e}")
    
    async def _test_execution_history(self, client: httpx.AsyncClient):
        """测试执行历史"""
        print("📜 测试执行历史...")
        
        try:
            response = await client.get(
                f"{self.api_base}/history/{self.test_thread_id}",
                params={
                    "include_snapshots": True,
                    "include_checkpoints": True,
                    "limit": 100
                }
            )
            if response.status_code == 200:
                history = response.json()
                print(f"   ✅ 执行历史获取成功: 共{history['total_steps']}步")
                print(f"      - 快照数量: {len(history['snapshots'])}")
                print(f"      - 检查点数量: {len(history['checkpoints'])}")
            else:
                print(f"   ❌ 执行历史获取失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ 执行历史异常: {e}")
    
    async def _test_status_query(self, client: httpx.AsyncClient):
        """测试状态查询"""
        print("📊 测试状态查询...")
        
        try:
            response = await client.get(f"{self.api_base}/status/{self.test_thread_id}")
            if response.status_code == 200:
                status = response.json()
                print(f"   ✅ 状态查询成功:")
                print(f"      - 时间旅行启用: {status['time_travel_enabled']}")
                print(f"      - 快照总数: {status['statistics']['total_snapshots']}")
                print(f"      - 检查点总数: {status['statistics']['total_checkpoints']}")
                print(f"      - 总步骤数: {status['statistics']['total_steps']}")
                print(f"      - 估计存储: {status['statistics']['estimated_storage_mb']} MB")
            else:
                print(f"   ❌ 状态查询失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ 状态查询异常: {e}")
    
    async def _test_cleanup(self, client: httpx.AsyncClient):
        """测试数据清理"""
        print("🧹 测试数据清理...")
        
        try:
            cleanup_data = {
                "keep_recent_days": 7,
                "keep_important": True
            }
            response = await client.post(
                f"{self.api_base}/cleanup/{self.test_thread_id}",
                params=cleanup_data
            )
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ 数据清理启动成功: {result['message']}")
            else:
                print(f"   ❌ 数据清理失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ 数据清理异常: {e}")
    
    async def test_rollback_functionality(self, client: httpx.AsyncClient):
        """测试回滚功能"""
        print("⏪ 测试回滚功能...")
        
        try:
            rollback_data = {
                "thread_id": self.test_thread_id,
                "target_step": 0,
                "rollback_type": "soft"
            }
            
            response = await client.post(f"{self.api_base}/rollback", json=rollback_data)
            if response.status_code == 200:
                rollback = response.json()
                print(f"   ✅ 回滚创建成功: {rollback['rollback_id']}")
                print(f"      - 从步骤 {rollback['from_step']} 回滚到 {rollback['to_step']}")
            else:
                print(f"   ❌ 回滚失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ 回滚异常: {e}")


async def main():
    """主函数"""
    print("🧪 时间旅行API测试工具")
    print("=" * 60)
    
    # 检查服务器是否运行
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            if response.status_code == 200:
                print("✅ 服务器运行正常")
            else:
                print(f"❌ 服务器响应异常: {response.status_code}")
                return
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        print("💡 请确保服务器正在运行: python main.py")
        return
    
    # 运行测试
    tester = TimeTravelAPITester()
    await tester.test_api_endpoints()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        sys.exit(1)