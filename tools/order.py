from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import requests, json
from datetime import datetime, timedelta

class AutoOrder(BaseModel):
    ID: str = Field(description="用户ID", default="")
    Mobile: str = Field(description="用户手机号", default="")
    Action: str = Field(description="操作类型，如：订水", default="订水")
    GoodsID: int = Field(description="商品ID，1=飞龙雨，5=小飞龙，3=山知水心", default=1)
    GoodsName: str = Field(description="商品名称：飞龙雨、小飞龙、山知水心", default="")
    Quantity: int = Field(description="订购数量", default=1)
    AppointmentKind: int = Field(description="预约类型：0=即刻订水，1=预约订水", default=0)
    AppointmentTime: str = Field(description="预约时间，格式：Y-m-d H:i:s", default="")
    Province: str = Field(description="省份", default="")
    City: str = Field(description="城市", default="")
    District: str = Field(description="区县", default="")
    Address: str = Field(description="详细地址", default="")
    Mark: str = Field(description="备注", default="")
    OrderNum: str = Field(description="订单号，为空时自动生成", default="")

@tool(args_schema=AutoOrder)
def auto_order(
    ID: str = "",
    Mobile: str = "",
    Action: str = "订水",
    GoodsID: int = 1,
    GoodsName: str = "",
    Quantity: int = 1,
    AppointmentKind: int = 0,
    AppointmentTime: str = "",
    Province: str = "",
    City: str = "",
    District: str = "",
    Address: str = "",
    Mark: str = "",
    OrderNum: str = ""
):
    """
    自动订水工具
    
    参数说明：
    - ID: 用户ID（可选，与Mobile二选一）
    - Mobile: 用户手机号（可选，与ID二选一）
    - Action: 操作类型，默认为"订水"
    - GoodsID: 商品ID，1=飞龙雨，5=小飞龙，3=山知水心
    - GoodsName: 商品名称，支持：飞龙雨、小飞龙、山知水心
    - Quantity: 订购数量，默认为1
    - AppointmentKind: 预约类型，0=即刻订水，1=预约订水
    - AppointmentTime: 预约时间（预约订水时必填）
    - Province: 省份（可选，为空时使用默认地址）
    - City: 城市（可选，为空时使用默认地址）
    - District: 区县（可选，为空时使用默认地址）
    - Address: 详细地址（可选，为空时使用默认地址）
    - Mark: 备注信息
    - OrderNum: 订单号（可选，为空时自动生成）
    
    返回：
    - Result: 结果状态（1=成功，0=失败，-1=用户验证失败）
    - Message: 结果消息
    - OrderNum: 订单号（成功时返回）
    - Address: 配送地址（成功时返回）
    """
    
    # API 接口地址
    api_url = "https://api.fly96089.com/api/LLM_Api/AutoOrderWater.php"
    
    # 构建请求参数
    params = {
        "ID": ID,
        "Mobile": Mobile,
        "action": Action,
        "GoodsID": GoodsID,
        "GoodsName": GoodsName,
        "Quantity": Quantity,
        "AppointmentKind": AppointmentKind,
        "AppointmentTime": AppointmentTime,
        "Province": Province,
        "City": City,
        "District": District,
        "Address": Address,
        "Mark": Mark,
        "OrderNum": OrderNum
    }
    
    # 移除空值参数
    params = {k: v for k, v in params.items() if v != "" and v != 0}
    
    try:
        # 发送 POST 请求
        response = requests.post(api_url, data=params, timeout=30)
        response.raise_for_status()  # 检查HTTP错误
        
        # 解析响应
        result = response.json()
        
        # 格式化返回结果
        if result.get('Result') == 1:
            # 成功
            return result
            # message = f"✅ 订单创建成功！\n"
            # message += f"📋 订单号：{result.get('OrderNum', '')}\n"
            # message += f"📍 配送地址：{result.get('Address', '')}\n"
            # message += f"💬 详情：{result.get('Message', '')}"
        elif result.get('Result') == -1:
            # 用户验证失败
            message = f"❌ 用户验证失败：{result.get('Message', '')}"
        else:
            # 其他错误
            message = f"❌ 订单创建失败：{result.get('Message', '')}"
        
        return message
        
    except requests.exceptions.RequestException as e:
        return f"❌ 网络请求失败：{str(e)}"
    except json.JSONDecodeError as e:
        return f"❌ 响应解析失败：{str(e)}"
    except Exception as e:
        return f"❌ 未知错误：{str(e)}"
    

# # 获取日期和时间信息工具
# class GetDateInfo(BaseModel):
#     format_type: str = Field(description="返回格式类型，可选值为'date'(仅日期)、'time'(仅时间)、'datetime'(日期和时间)", default="datetime")
#     days_offset: int = Field(description="日期偏移量，0表示今天，1表示明天，-1表示昨天", default=0)

# @tool(args_schema=GetDateInfo)
# def get_date_info(format_type: str = "datetime", days_offset: int = 0) -> str:
#     """
#     获取日期和时间信息
#     当用户要求预约订水时，必须按以下步骤执行：
#     1. 先调用get_date_info获取预约时间
#     2. 将返回的时间字符串填入AppointmentTime参数
    
#     示例流程：
#     用户："订一桶明天10点的飞龙雨"
#     → 第一步：调用get_date_info获取明天10点的时间
#     → 第二步：调用auto_order(GoodsName="飞龙雨", AppointmentTime="get_date_info的结果")

#     :param format_type: 返回格式类型，可选值为"date"(仅日期)、"time"(仅时间)、"datetime"(日期和时间)
#     :param days_offset: 日期偏移量，0表示今天，1表示明天，-1表示昨天
#     :return: 根据指定格式返回的日期/时间字符串
#     """
#     target_date = datetime.now() + timedelta(days=days_offset)
    
#     if format_type == "date":
#         return target_date.strftime("%Y-%m-%d")
#     elif format_type == "time":
#         return target_date.strftime("%H:%M:%S")
#     else:  # 默认返回日期和时间
#         return target_date.strftime("%Y-%m-%d %H:%M:%S")
    
