from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import requests, json

class CancelOrder(BaseModel):
    ID: str = Field(description="用户ID", default="")
    Mobile: str = Field(description="用户手机号", default="")
    Action: str = Field(description="操作类型：查询订单、取消", default="查询订单")
    OrderID: str = Field(description="订单ID（取消订单时必填）", default="")

@tool(args_schema=CancelOrder)
def cancel_order(ID: str = "125748", Mobile: str = "", Action: str = "查询订单", OrderID: str = ""):
    """
    取消订单工具
    
    功能：查询会员未完成的订单或取消指定订单
    
    参数说明：
    - ID: 用户ID（可选，与Mobile二选一）
    - Mobile: 用户手机号（可选，与ID二选一）
    - Action: 操作类型，支持"查询订单"或"取消"
    - OrderID: 订单ID（仅在取消订单时需要）
    
    操作说明：
    1. 查询订单：返回用户所有未完成的订单列表
    2. 取消订单：取消指定的订单（需要提供OrderID）
    
    返回格式：
    - Result: 结果状态（1=成功，0=失败，-1=用户验证失败）
    - Message: 结果消息
    - OrderList: 订单列表（查询订单时返回）
    
    订单状态说明：
    - 0: 未处理
    - 1: 已接单
    - 2: 配送中
    - 3: 已完成
    - 4: 已废弃
    """
    
    # API 接口地址
    api_url = "https://api.fly96089.com/api/LLM_Api/CancelOrder.php"
    
    # 构建请求参数
    params = {
        "ID": ID,
        "Mobile": Mobile,
        "action": Action,
        "OrderID": OrderID
    }
    
    # 移除空值参数
    params = {k: v for k, v in params.items() if v != ""}
    
    # 验证参数
    if Action == "取消" and OrderID == "":
        return "❌ 取消订单需要提供订单ID"
    
    try:
        # 发送 POST 请求
        response = requests.post(api_url, data=params, timeout=30)
        response.raise_for_status()  # 检查HTTP错误
        
        # 解析响应
        result = response.json()
        
        # 直接返回原始结果，让大模型自主组织语言
        return result
        
    except requests.exceptions.RequestException as e:
        return f"❌ 网络请求失败：{str(e)}"
    except json.JSONDecodeError as e:
        return f"❌ 响应解析失败：{str(e)}"
    except Exception as e:
        return f"❌ 未知错误：{str(e)}"