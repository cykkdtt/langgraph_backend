#!/usr/bin/env python3
"""
前端功能测试脚本
使用Selenium测试前端界面功能
"""

import time
import json
from typing import Dict, List, Any
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class FrontendTester:
    """前端功能测试器"""
    
    def __init__(self, base_url: str = "http://localhost:5173"):
        self.base_url = base_url
        self.driver = None
        self.test_results = []
        
    def setup_driver(self):
        """设置浏览器驱动"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # 无头模式
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            return True
        except Exception as e:
            print(f"浏览器驱动设置失败: {e}")
            return False
    
    def teardown_driver(self):
        """关闭浏览器驱动"""
        if self.driver:
            self.driver.quit()
    
    def test_page_load(self, path: str = "/") -> Dict[str, Any]:
        """测试页面加载"""
        url = f"{self.base_url}{path}"
        start_time = time.time()
        
        try:
            self.driver.get(url)
            
            # 等待页面加载完成
            WebDriverWait(self.driver, 10).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            load_time = time.time() - start_time
            
            result = {
                "test": f"页面加载测试 - {path}",
                "success": True,
                "load_time": round(load_time * 1000, 2),
                "page_title": self.driver.title,
                "url": self.driver.current_url,
                "error": None
            }
            
        except Exception as e:
            result = {
                "test": f"页面加载测试 - {path}",
                "success": False,
                "load_time": None,
                "page_title": None,
                "url": url,
                "error": str(e)
            }
        
        self.test_results.append(result)
        return result
    
    def test_element_presence(self, selector: str, description: str) -> Dict[str, Any]:
        """测试元素是否存在"""
        try:
            element = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            
            result = {
                "test": f"元素存在测试 - {description}",
                "success": True,
                "element_found": True,
                "element_text": element.text[:100] if element.text else None,
                "error": None
            }
            
        except TimeoutException:
            result = {
                "test": f"元素存在测试 - {description}",
                "success": False,
                "element_found": False,
                "element_text": None,
                "error": f"元素未找到: {selector}"
            }
        except Exception as e:
            result = {
                "test": f"元素存在测试 - {description}",
                "success": False,
                "element_found": False,
                "element_text": None,
                "error": str(e)
            }
        
        self.test_results.append(result)
        return result
    
    def test_click_interaction(self, selector: str, description: str) -> Dict[str, Any]:
        """测试点击交互"""
        try:
            element = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            
            # 记录点击前的状态
            before_url = self.driver.current_url
            
            element.click()
            time.sleep(1)  # 等待响应
            
            # 记录点击后的状态
            after_url = self.driver.current_url
            
            result = {
                "test": f"点击交互测试 - {description}",
                "success": True,
                "clicked": True,
                "url_changed": before_url != after_url,
                "before_url": before_url,
                "after_url": after_url,
                "error": None
            }
            
        except Exception as e:
            result = {
                "test": f"点击交互测试 - {description}",
                "success": False,
                "clicked": False,
                "url_changed": False,
                "before_url": None,
                "after_url": None,
                "error": str(e)
            }
        
        self.test_results.append(result)
        return result
    
    def test_input_interaction(self, selector: str, test_text: str, description: str) -> Dict[str, Any]:
        """测试输入交互"""
        try:
            element = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            
            # 清空并输入文本
            element.clear()
            element.send_keys(test_text)
            
            # 验证输入
            actual_value = element.get_attribute('value')
            
            result = {
                "test": f"输入交互测试 - {description}",
                "success": actual_value == test_text,
                "input_successful": True,
                "expected_value": test_text,
                "actual_value": actual_value,
                "error": None if actual_value == test_text else f"输入值不匹配: 期望 '{test_text}', 实际 '{actual_value}'"
            }
            
        except Exception as e:
            result = {
                "test": f"输入交互测试 - {description}",
                "success": False,
                "input_successful": False,
                "expected_value": test_text,
                "actual_value": None,
                "error": str(e)
            }
        
        self.test_results.append(result)
        return result
    
    def run_comprehensive_tests(self):
        """运行综合前端测试"""
        print("开始前端功能测试...\n")
        
        if not self.setup_driver():
            print("❌ 浏览器驱动设置失败，无法进行测试")
            return
        
        try:
            # 1. 页面加载测试
            print("1. 测试页面加载")
            load_result = self.test_page_load("/")
            self.print_result(load_result)
            
            if not load_result["success"]:
                print("❌ 页面加载失败，跳过后续测试")
                return
            
            # 2. 基础元素存在测试
            print("\n2. 测试基础界面元素")
            
            # 测试主要容器
            container_result = self.test_element_presence(".App", "主应用容器")
            self.print_result(container_result)
            
            # 测试导航元素
            nav_result = self.test_element_presence("nav, .ant-menu", "导航菜单")
            self.print_result(nav_result)
            
            # 测试内容区域
            content_result = self.test_element_presence("main, .ant-layout-content", "主内容区域")
            self.print_result(content_result)
            
            # 3. 统计卡片测试
            print("\n3. 测试统计卡片")
            stats_result = self.test_element_presence(".ant-statistic", "统计卡片")
            self.print_result(stats_result)
            
            # 4. 聊天界面测试
            print("\n4. 测试聊天界面")
            
            # 测试聊天容器
            chat_container_result = self.test_element_presence("[class*='chat'], .ant-input", "聊天界面")
            self.print_result(chat_container_result)
            
            # 测试输入框
            if chat_container_result["success"]:
                input_result = self.test_element_presence("textarea, input[type='text']", "消息输入框")
                self.print_result(input_result)
                
                # 测试输入功能
                if input_result["success"]:
                    input_test_result = self.test_input_interaction(
                        "textarea, input[type='text']", 
                        "测试消息", 
                        "消息输入功能"
                    )
                    self.print_result(input_test_result)
            
            # 5. 按钮交互测试
            print("\n5. 测试按钮交互")
            button_result = self.test_element_presence("button", "按钮元素")
            self.print_result(button_result)
            
            # 6. 响应式测试
            print("\n6. 测试响应式设计")
            self.test_responsive_design()
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
        
        finally:
            self.teardown_driver()
    
    def test_responsive_design(self):
        """测试响应式设计"""
        screen_sizes = [
            (1920, 1080, "桌面"),
            (768, 1024, "平板"),
            (375, 667, "手机")
        ]
        
        for width, height, device in screen_sizes:
            try:
                self.driver.set_window_size(width, height)
                time.sleep(1)
                
                # 检查页面是否正常显示
                body = self.driver.find_element(By.TAG_NAME, "body")
                
                result = {
                    "test": f"响应式测试 - {device} ({width}x{height})",
                    "success": True,
                    "screen_size": f"{width}x{height}",
                    "device_type": device,
                    "page_visible": body.is_displayed(),
                    "error": None
                }
                
            except Exception as e:
                result = {
                    "test": f"响应式测试 - {device} ({width}x{height})",
                    "success": False,
                    "screen_size": f"{width}x{height}",
                    "device_type": device,
                    "page_visible": False,
                    "error": str(e)
                }
            
            self.test_results.append(result)
            self.print_result(result)
    
    def print_result(self, result: Dict[str, Any]):
        """打印测试结果"""
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"   {status} {result['test']}")
        
        if "load_time" in result and result["load_time"]:
            print(f"   加载时间: {result['load_time']}ms")
        
        if result.get("error"):
            print(f"   错误: {result['error']}")
        
        if result.get("page_title"):
            print(f"   页面标题: {result['page_title']}")
    
    def generate_report(self):
        """生成测试报告"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r["success"])
        failed_tests = total_tests - successful_tests
        
        print("\n" + "="*50)
        print("前端功能测试报告")
        print("="*50)
        print(f"总测试数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失败: {failed_tests}")
        print(f"成功率: {(successful_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print("\n失败的测试:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result.get('error', '未知错误')}")
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": successful_tests/total_tests*100,
            "results": self.test_results
        }


def simple_frontend_test():
    """简单的前端测试（不需要Selenium）"""
    import requests
    
    print("开始简单前端测试...\n")
    
    try:
        # 测试前端服务器是否响应
        response = requests.get("http://localhost:5173", timeout=5)
        
        print(f"✅ 前端服务器响应正常")
        print(f"   状态码: {response.status_code}")
        print(f"   响应时间: {response.elapsed.total_seconds() * 1000:.2f}ms")
        print(f"   内容长度: {len(response.text)} 字符")
        
        # 检查HTML内容
        if "<!DOCTYPE html>" in response.text or "<html" in response.text:
            print(f"   ✅ HTML内容正常")
        else:
            print(f"   ❌ HTML内容异常")
        
        # 检查是否包含React相关内容
        if "react" in response.text.lower() or "vite" in response.text.lower():
            print(f"   ✅ 检测到React/Vite相关内容")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到前端服务器 (http://localhost:5173)")
        print("   请确保前端开发服务器正在运行")
        return False
    except Exception as e:
        print(f"❌ 前端测试失败: {e}")
        return False


if __name__ == "__main__":
    # 首先进行简单测试
    if simple_frontend_test():
        print("\n" + "="*50)
        print("注意: 完整的UI测试需要安装Chrome浏览器和ChromeDriver")
        print("如需进行完整测试，请运行:")
        print("pip install selenium")
        print("并确保已安装Chrome浏览器")
        print("="*50)
        
        # 如果需要完整测试，取消下面的注释
        # tester = FrontendTester()
        # tester.run_comprehensive_tests()
        # tester.generate_report()
    else:
        print("\n前端服务器未运行，跳过UI测试")