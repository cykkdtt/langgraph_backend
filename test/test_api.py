#!/usr/bin/env python3
"""
API测试脚本
用于测试后端API接口的功能
"""

import requests
import json
import time
from typing import Dict, Any


class APITester:
    """API测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def test_endpoint(self, method: str, endpoint: str, data: Dict[Any, Any] = None, 
                     headers: Dict[str, str] = None) -> Dict[str, Any]:
        """测试单个API端点"""
        url = f"{self.base_url}{endpoint}"
        
        if headers is None:
            headers = {"Content-Type": "application/json"}
        
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, headers=headers)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, headers=headers)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            response_time = time.time() - start_time
            
            result = {
                "endpoint": endpoint,
                "method": method.upper(),
                "status_code": response.status_code,
                "response_time": round(response_time * 1000, 2),  # 毫秒
                "success": 200 <= response.status_code < 300,
                "response_data": None,
                "error": None
            }
            
            try:
                result["response_data"] = response.json()
            except:
                result["response_data"] = response.text
            
            if not result["success"]:
                result["error"] = f"HTTP {response.status_code}: {response.text}"
            
        except Exception as e:
            result = {
                "endpoint": endpoint,
                "method": method.upper(),
                "status_code": None,
                "response_time": None,
                "success": False,
                "response_data": None,
                "error": str(e)
            }
        
        self.test_results.append(result)
        return result
    
    def run_basic_tests(self):
        """运行基础API测试"""
        print("开始API测试...\n")
        
        # 1. 健康检查
        print("1. 测试健康检查接口")
        health_result = self.test_endpoint("GET", "/health")
        self.print_result(health_result)
        
        # 2. 系统状态
        print("\n2. 测试系统状态接口")
        status_result = self.test_endpoint("GET", "/status")
        self.print_result(status_result)
        
        # 3. 根路径
        print("\n3. 测试根路径")
        root_result = self.test_endpoint("GET", "/")
        self.print_result(root_result)
        
        # 4. 聊天接口
        print("\n4. 测试聊天接口")
        chat_data = {
            "content": "Hello, how are you?",
            "user_id": "test_user",
            "agent_type": "supervisor"
        }
        chat_result = self.test_endpoint("POST", "/chat", chat_data)
        self.print_result(chat_result)
        
        # 5. 智能体列表
        print("\n5. 测试智能体列表接口")
        agents_result = self.test_endpoint("GET", "/agents")
        self.print_result(agents_result)
        
        # 6. API v1 路由测试
        print("\n6. 测试API v1路由")
        
        # 6.1 聊天API v1
        print("\n6.1 测试聊天API v1")
        chat_v1_result = self.test_endpoint("POST", "/api/v1/chat", chat_data)
        self.print_result(chat_v1_result)
        
        # 6.2 线程API v1
        print("\n6.2 测试线程API v1")
        threads_result = self.test_endpoint("GET", "/api/v1/threads")
        self.print_result(threads_result)
        
        # 6.3 内存API v1
        print("\n6.3 测试内存API v1")
        memory_result = self.test_endpoint("GET", "/api/v1/memory")
        self.print_result(memory_result)
    
    def print_result(self, result: Dict[str, Any]):
        """打印测试结果"""
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"   {status} {result['method']} {result['endpoint']}")
        print(f"   状态码: {result['status_code']}")
        
        if result["response_time"]:
            print(f"   响应时间: {result['response_time']}ms")
        
        if result["error"]:
            print(f"   错误: {result['error']}")
        elif result["response_data"]:
            if isinstance(result["response_data"], dict):
                print(f"   响应: {json.dumps(result['response_data'], ensure_ascii=False, indent=2)[:200]}...")
            else:
                print(f"   响应: {str(result['response_data'])[:200]}...")
    
    def generate_report(self):
        """生成测试报告"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r["success"])
        failed_tests = total_tests - successful_tests
        
        avg_response_time = sum(
            r["response_time"] for r in self.test_results 
            if r["response_time"] is not None
        ) / max(1, sum(1 for r in self.test_results if r["response_time"] is not None))
        
        print("\n" + "="*50)
        print("API测试报告")
        print("="*50)
        print(f"总测试数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失败: {failed_tests}")
        print(f"成功率: {(successful_tests/total_tests*100):.1f}%")
        print(f"平均响应时间: {avg_response_time:.2f}ms")
        
        if failed_tests > 0:
            print("\n失败的测试:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['method']} {result['endpoint']}: {result['error']}")
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": successful_tests/total_tests*100,
            "avg_response_time": avg_response_time,
            "results": self.test_results
        }


if __name__ == "__main__":
    tester = APITester()
    tester.run_basic_tests()
    report = tester.generate_report()