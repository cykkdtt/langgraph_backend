#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统性能测试脚本
测试响应时间、并发性能和资源使用情况
"""

import requests
import time
import threading
import json
import statistics
import psutil
import concurrent.futures
from typing import List, Dict, Any
from datetime import datetime

class PerformanceTester:
    def __init__(self, backend_url="http://localhost:8000", frontend_url="http://localhost:5173"):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.test_results = []
        self.session = requests.Session()
        
    def log_test(self, test_name: str, metrics: Dict[str, Any]):
        """记录测试结果"""
        result = {
            "test_name": test_name,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        print(f"📊 {test_name}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print()
        
    def measure_response_time(self, url: str, method: str = "GET", data: Dict = None, iterations: int = 10) -> Dict[str, float]:
        """测量响应时间"""
        response_times = []
        successful_requests = 0
        
        for _ in range(iterations):
            try:
                start_time = time.time()
                
                if method == "GET":
                    response = self.session.get(url, timeout=10)
                elif method == "POST":
                    response = self.session.post(url, json=data, timeout=10)
                    
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # 转换为毫秒
                
                if response.status_code in [200, 201]:
                    response_times.append(response_time)
                    successful_requests += 1
                    
            except Exception as e:
                print(f"请求失败: {e}")
                
        if response_times:
            return {
                "平均响应时间(ms)": statistics.mean(response_times),
                "最小响应时间(ms)": min(response_times),
                "最大响应时间(ms)": max(response_times),
                "中位数响应时间(ms)": statistics.median(response_times),
                "成功请求数": successful_requests,
                "总请求数": iterations,
                "成功率(%)": (successful_requests / iterations) * 100
            }
        else:
            return {
                "平均响应时间(ms)": 0,
                "最小响应时间(ms)": 0,
                "最大响应时间(ms)": 0,
                "中位数响应时间(ms)": 0,
                "成功请求数": 0,
                "总请求数": iterations,
                "成功率(%)": 0
            }
            
    def test_concurrent_requests(self, url: str, concurrent_users: int = 10, requests_per_user: int = 5) -> Dict[str, Any]:
        """测试并发请求性能"""
        def make_request():
            try:
                start_time = time.time()
                response = self.session.get(url, timeout=10)
                end_time = time.time()
                return {
                    "response_time": (end_time - start_time) * 1000,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "response_time": 0,
                    "status_code": 0,
                    "success": False,
                    "error": str(e)
                }
                
        # 创建线程池执行并发请求
        all_results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # 提交所有任务
            futures = []
            for _ in range(concurrent_users * requests_per_user):
                future = executor.submit(make_request)
                futures.append(future)
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                all_results.append(result)
                
        end_time = time.time()
        total_time = end_time - start_time
        
        # 分析结果
        successful_requests = [r for r in all_results if r['success']]
        response_times = [r['response_time'] for r in successful_requests]
        
        if response_times:
            return {
                "并发用户数": concurrent_users,
                "每用户请求数": requests_per_user,
                "总请求数": len(all_results),
                "成功请求数": len(successful_requests),
                "失败请求数": len(all_results) - len(successful_requests),
                "成功率(%)": (len(successful_requests) / len(all_results)) * 100,
                "总耗时(s)": total_time,
                "平均响应时间(ms)": statistics.mean(response_times),
                "最大响应时间(ms)": max(response_times),
                "最小响应时间(ms)": min(response_times),
                "吞吐量(req/s)": len(successful_requests) / total_time if total_time > 0 else 0
            }
        else:
            return {
                "并发用户数": concurrent_users,
                "每用户请求数": requests_per_user,
                "总请求数": len(all_results),
                "成功请求数": 0,
                "失败请求数": len(all_results),
                "成功率(%)": 0,
                "总耗时(s)": total_time,
                "平均响应时间(ms)": 0,
                "最大响应时间(ms)": 0,
                "最小响应时间(ms)": 0,
                "吞吐量(req/s)": 0
            }
            
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统资源使用情况"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "CPU使用率(%)": cpu_percent,
                "内存使用率(%)": memory.percent,
                "可用内存(GB)": memory.available / (1024**3),
                "总内存(GB)": memory.total / (1024**3),
                "磁盘使用率(%)": disk.percent,
                "可用磁盘空间(GB)": disk.free / (1024**3)
            }
        except Exception as e:
            return {"错误": f"无法获取系统指标: {str(e)}"}
            
    def test_endpoint_performance(self):
        """测试各个端点的性能"""
        endpoints = [
            (f"{self.backend_url}/health", "GET", None, "后端健康检查"),
            (f"{self.backend_url}/", "GET", None, "后端根路径"),
            (f"{self.backend_url}/agents", "GET", None, "智能体列表"),
            (f"{self.frontend_url}", "GET", None, "前端首页")
        ]
        
        for url, method, data, description in endpoints:
            print(f"🔍 测试 {description} 性能...")
            metrics = self.measure_response_time(url, method, data, iterations=20)
            self.log_test(f"{description} - 响应时间测试", metrics)
            
    def test_concurrent_performance(self):
        """测试并发性能"""
        test_cases = [
            (f"{self.backend_url}/health", 5, 3, "后端健康检查 - 轻负载"),
            (f"{self.backend_url}/health", 10, 5, "后端健康检查 - 中负载"),
            (f"{self.backend_url}/agents", 5, 3, "智能体列表 - 轻负载"),
            (f"{self.frontend_url}", 5, 3, "前端首页 - 轻负载")
        ]
        
        for url, concurrent_users, requests_per_user, description in test_cases:
            print(f"🚀 测试 {description} 并发性能...")
            metrics = self.test_concurrent_requests(url, concurrent_users, requests_per_user)
            self.log_test(f"{description} - 并发测试", metrics)
            
    def test_load_performance(self):
        """测试负载性能"""
        print("⚡ 测试系统负载性能...")
        
        # 测试不同负载级别
        load_tests = [
            (f"{self.backend_url}/health", 20, 10, "高负载测试"),
            (f"{self.backend_url}/agents", 15, 8, "中高负载测试")
        ]
        
        for url, concurrent_users, requests_per_user, description in load_tests:
            print(f"📈 执行 {description}...")
            
            # 测试前获取系统指标
            before_metrics = self.get_system_metrics()
            
            # 执行负载测试
            performance_metrics = self.test_concurrent_requests(url, concurrent_users, requests_per_user)
            
            # 测试后获取系统指标
            time.sleep(2)  # 等待系统稳定
            after_metrics = self.get_system_metrics()
            
            # 合并指标
            combined_metrics = {
                **performance_metrics,
                "测试前CPU使用率(%)": before_metrics.get("CPU使用率(%)", 0),
                "测试后CPU使用率(%)": after_metrics.get("CPU使用率(%)", 0),
                "测试前内存使用率(%)": before_metrics.get("内存使用率(%)", 0),
                "测试后内存使用率(%)": after_metrics.get("内存使用率(%)", 0)
            }
            
            self.log_test(f"{description}", combined_metrics)
            
    def run_all_performance_tests(self):
        """运行所有性能测试"""
        print("\n🚀 开始系统性能测试...\n")
        
        # 获取初始系统状态
        print("📊 系统初始状态:")
        initial_metrics = self.get_system_metrics()
        self.log_test("系统初始状态", initial_metrics)
        
        # 端点响应时间测试
        print("⏱️ 端点响应时间测试:")
        self.test_endpoint_performance()
        
        # 并发性能测试
        print("🔄 并发性能测试:")
        self.test_concurrent_performance()
        
        # 负载性能测试
        print("📈 负载性能测试:")
        self.test_load_performance()
        
        # 生成性能报告
        self.generate_performance_report()
        
    def generate_performance_report(self):
        """生成性能测试报告"""
        print("\n" + "="*60)
        print("📊 系统性能测试报告")
        print("="*60)
        
        # 分析响应时间
        response_time_tests = [r for r in self.test_results if "响应时间测试" in r['test_name']]
        if response_time_tests:
            avg_response_times = [r['metrics'].get('平均响应时间(ms)', 0) for r in response_time_tests]
            print(f"\n⏱️ 响应时间分析:")
            print(f"  平均响应时间: {statistics.mean(avg_response_times):.2f}ms")
            print(f"  最快响应时间: {min(avg_response_times):.2f}ms")
            print(f"  最慢响应时间: {max(avg_response_times):.2f}ms")
        
        # 分析并发性能
        concurrent_tests = [r for r in self.test_results if "并发测试" in r['test_name']]
        if concurrent_tests:
            throughputs = [r['metrics'].get('吞吐量(req/s)', 0) for r in concurrent_tests]
            success_rates = [r['metrics'].get('成功率(%)', 0) for r in concurrent_tests]
            print(f"\n🚀 并发性能分析:")
            print(f"  平均吞吐量: {statistics.mean(throughputs):.2f} req/s")
            print(f"  最高吞吐量: {max(throughputs):.2f} req/s")
            print(f"  平均成功率: {statistics.mean(success_rates):.1f}%")
        
        # 性能评级
        print(f"\n🏆 性能评级:")
        if response_time_tests:
            avg_response = statistics.mean([r['metrics'].get('平均响应时间(ms)', 0) for r in response_time_tests])
            if avg_response < 100:
                print("  响应时间: 优秀 (< 100ms)")
            elif avg_response < 500:
                print("  响应时间: 良好 (< 500ms)")
            elif avg_response < 1000:
                print("  响应时间: 一般 (< 1000ms)")
            else:
                print("  响应时间: 需要优化 (> 1000ms)")
        
        if concurrent_tests:
            avg_success_rate = statistics.mean([r['metrics'].get('成功率(%)', 0) for r in concurrent_tests])
            if avg_success_rate >= 95:
                print("  稳定性: 优秀 (≥ 95%)")
            elif avg_success_rate >= 90:
                print("  稳定性: 良好 (≥ 90%)")
            elif avg_success_rate >= 80:
                print("  稳定性: 一般 (≥ 80%)")
            else:
                print("  稳定性: 需要优化 (< 80%)")
        
        # 保存详细报告
        report_file = "performance_test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_tests": len(self.test_results),
                    "test_time": datetime.now().isoformat()
                },
                "results": self.test_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细性能报告已保存到: {report_file}")
        
if __name__ == "__main__":
    tester = PerformanceTester()
    tester.run_all_performance_tests()