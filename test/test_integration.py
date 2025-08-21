#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前后端集成测试脚本
测试前端API调用和WebSocket连接功能
"""

import requests
import json
import time
import asyncio
import websockets
from typing import Dict, List, Any
from datetime import datetime

class IntegrationTester:
    def __init__(self, backend_url="http://localhost:8000", frontend_url="http://localhost:5173"):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.ws_url = "ws://localhost:8000/ws/test_user"
        self.test_results = []
        self.session = requests.Session()
        
    def log_test(self, test_name: str, success: bool, details: str = "", response_time: float = 0):
        """记录测试结果"""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({response_time:.2f}ms) - {details}")
        
    def test_frontend_server(self):
        """测试前端服务器响应"""
        try:
            start_time = time.time()
            response = self.session.get(self.frontend_url, timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                self.log_test("前端服务器响应", True, f"状态码: {response.status_code}", response_time)
                return True
            else:
                self.log_test("前端服务器响应", False, f"状态码: {response.status_code}", response_time)
                return False
        except Exception as e:
            self.log_test("前端服务器响应", False, f"错误: {str(e)}")
            return False
            
    def test_backend_health(self):
        """测试后端健康检查"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.backend_url}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                self.log_test("后端健康检查", True, f"状态码: {response.status_code}", response_time)
                return True
            else:
                self.log_test("后端健康检查", False, f"状态码: {response.status_code}", response_time)
                return False
        except Exception as e:
            self.log_test("后端健康检查", False, f"错误: {str(e)}")
            return False
            
    def test_api_endpoints(self):
        """测试主要API端点"""
        endpoints = [
            ("/", "GET", "根路径"),
            ("/agents", "GET", "智能体列表"),
            ("/api/v1/threads", "GET", "对话列表"),
            ("/api/v1/workflows", "GET", "工作流列表"),
            ("/api/v1/memory", "GET", "记忆列表"),
            ("/api/v1/time-travel", "GET", "时间线"),
        ]
        
        for endpoint, method, description in endpoints:
            try:
                start_time = time.time()
                url = f"{self.backend_url}{endpoint}"
                
                if method == "GET":
                    response = self.session.get(url, timeout=5)
                elif method == "POST":
                    response = self.session.post(url, json={}, timeout=5)
                    
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code in [200, 201]:
                    self.log_test(f"API端点 - {description}", True, 
                                f"状态码: {response.status_code}", response_time)
                elif response.status_code in [401, 403]:
                    self.log_test(f"API端点 - {description}", True, 
                                f"需要认证 (状态码: {response.status_code})", response_time)
                else:
                    self.log_test(f"API端点 - {description}", False, 
                                f"状态码: {response.status_code}", response_time)
                    
            except Exception as e:
                self.log_test(f"API端点 - {description}", False, f"错误: {str(e)}")
                
    def test_chat_api(self):
        """测试聊天API"""
        try:
            # 测试发送消息
            start_time = time.time()
            chat_data = {
                "content": "Hello, this is a test message",
                "user_id": "test_user",
                "agent_type": "general"
            }
            
            response = self.session.post(
                f"{self.backend_url}/chat", 
                json=chat_data, 
                timeout=10
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code in [200, 201]:
                self.log_test("聊天API", True, 
                            f"状态码: {response.status_code}", response_time)
                return True
            else:
                self.log_test("聊天API", False, 
                            f"状态码: {response.status_code}, 响应: {response.text[:100]}", response_time)
                return False
                
        except Exception as e:
            self.log_test("聊天API", False, f"错误: {str(e)}")
            return False
            
    async def test_websocket_connection(self):
        """测试WebSocket连接"""
        try:
            start_time = time.time()
            
            # 尝试连接WebSocket
            async with websockets.connect(self.ws_url, timeout=5) as websocket:
                response_time = (time.time() - start_time) * 1000
                
                # 发送测试消息
                test_message = {
                    "type": "ping",
                    "payload": {"message": "test"},
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(test_message))
                
                # 等待响应
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    self.log_test("WebSocket连接", True, 
                                f"连接成功，收到响应", response_time)
                    return True
                except asyncio.TimeoutError:
                    self.log_test("WebSocket连接", True, 
                                f"连接成功，但无响应", response_time)
                    return True
                    
        except Exception as e:
            self.log_test("WebSocket连接", False, f"错误: {str(e)}")
            return False
            
    def test_cors_headers(self):
        """测试CORS配置"""
        try:
            start_time = time.time()
            
            # 发送OPTIONS请求测试CORS
            headers = {
                'Origin': self.frontend_url,
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            }
            
            response = self.session.options(
                f"{self.backend_url}/chat", 
                headers=headers, 
                timeout=5
            )
            response_time = (time.time() - start_time) * 1000
            
            cors_headers = {
                'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
            }
            
            if response.status_code == 200 and cors_headers['Access-Control-Allow-Origin']:
                self.log_test("CORS配置", True, 
                            f"CORS头部正确配置", response_time)
                return True
            else:
                self.log_test("CORS配置", False, 
                            f"状态码: {response.status_code}, CORS头部: {cors_headers}", response_time)
                return False
                
        except Exception as e:
            self.log_test("CORS配置", False, f"错误: {str(e)}")
            return False
            
    def run_all_tests(self):
        """运行所有集成测试"""
        print("\n🚀 开始前后端集成测试...\n")
        
        # 基础连接测试
        print("📡 基础连接测试:")
        self.test_frontend_server()
        self.test_backend_health()
        
        # API端点测试
        print("\n🔌 API端点测试:")
        self.test_api_endpoints()
        self.test_chat_api()
        
        # WebSocket测试
        print("\n🌐 WebSocket测试:")
        try:
            asyncio.run(self.test_websocket_connection())
        except Exception as e:
            self.log_test("WebSocket连接", False, f"异步测试错误: {str(e)}")
        
        # CORS测试
        print("\n🔒 CORS配置测试:")
        self.test_cors_headers()
        
        # 生成测试报告
        self.generate_report()
        
    def generate_report(self):
        """生成测试报告"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*60)
        print("📊 前后端集成测试报告")
        print("="*60)
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests}")
        print(f"失败: {failed_tests}")
        print(f"成功率: {success_rate:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ 失败的测试:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test_name']}: {result['details']}")
        
        print("\n✅ 成功的测试:")
        for result in self.test_results:
            if result['success']:
                print(f"  - {result['test_name']} ({result['response_time']:.2f}ms)")
        
        # 保存详细报告到文件
        report_file = "integration_test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": success_rate,
                    "test_time": datetime.now().isoformat()
                },
                "results": self.test_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
if __name__ == "__main__":
    tester = IntegrationTester()
    tester.run_all_tests()