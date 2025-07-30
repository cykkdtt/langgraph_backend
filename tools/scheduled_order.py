from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import datetime
from .order import auto_order

class ScheduledOrderWater(BaseModel):
    day: str = Field(description="日期描述，如'今天'、'明天'、'后天'或具体日期如'2023-05-20'")
    time: str = Field(description="时间描述，如'上午8点'、'下午3点'、'晚上7点半'或具体时间如'14:30'", default="")
    goods_name: Optional[str] = Field(description="商品名称，可选值为'飞龙雨'、'小飞龙'、'山知水心'", default=None)
    quantity: Optional[int] = Field(description="商品数量（整数）", default=1)
    ID: Optional[str] = Field(description="用户ID，用于身份验证", default="")

@tool(args_schema=ScheduledOrderWater)
def order_water_scheduled(day: str, time: str = "", goods_name: Optional[str] = None, quantity: Optional[int] = 1, ID: Optional[str] = "") -> str:
    """
    按照指定日期和时间预约订水
    
    这个工具可以解析自然语言的日期和时间描述，并自动调用订水服务。
    
    支持的日期格式：
    - 相对日期：今天、明天、后天
    - 具体日期：2023-05-20 格式
    
    支持的时间格式：
    - 模糊时间：上午、下午、晚上、中午
    - 具体时间：上午8点、下午3点、晚上7点半
    - 24小时格式：14:30
    
    :param day: 日期描述，如"今天"、"明天"、"后天"或具体日期如"2023-05-20"
    :param time: 时间描述，如"上午8点"、"下午3点"、"晚上7点半"或具体时间如"14:30"
    :param goods_name: 商品名称，可选值为"飞龙雨"、"小飞龙"、"山知水心"
    :param quantity: 商品数量（整数）
    :return: 预约结果信息
    """
    
    # 处理日期
    today = datetime.datetime.now()
    target_date = today  # 默认为今天，确保变量始终被定义
    
    if day == "今天":
        target_date = today
    elif day == "明天":
        target_date = today + datetime.timedelta(days=1)
    elif day == "后天":
        target_date = today + datetime.timedelta(days=2)
    else:
        # 尝试解析具体日期
        try:
            target_date = datetime.datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            return f"⚠️ 无法识别日期格式: {day}，请使用'今天'、'明天'、'后天'或'YYYY-MM-DD'格式"
    
    # 处理时间
    hour, minute = 8, 0  # 默认为早上8点
    
    # 处理时间描述
    if not time or time in ["早上", "上午", "早晨"]:
        pass  # 使用默认时间
    elif time == "中午":
        hour = 12
    elif time in ["下午", "晚上"]:
        hour = 18
    else:
        # 处理常见时间表述
        am_pm = None
        if "上午" in time or "早上" in time:
            time = time.replace("上午", "").replace("早上", "")
            am_pm = "上午"
        elif "下午" in time or "晚上" in time:
            time = time.replace("下午", "").replace("晚上", "")
            am_pm = "下午"
        
        # 提取小时和分钟
        if "点" in time:
            parts = time.split("点")
            try:
                hour = int(parts[0].strip())
                # 处理下午时间
                if am_pm == "下午" and hour < 12:
                    hour += 12
                    
                # 处理分钟
                if len(parts) > 1 and parts[1].strip():
                    minute_part = parts[1].strip()
                    if "半" in minute_part:
                        minute = 30
                    else:
                        minute_part = minute_part.replace("分", "")
                        if minute_part:
                            minute = int(minute_part)
            except ValueError:
                return f"⚠️ 无法识别时间格式: {time}"
        else:
            # 尝试解析具体时间 HH:MM
            try:
                time_parts = time.split(":")
                hour = int(time_parts[0].strip())
                if len(time_parts) > 1:
                    minute = int(time_parts[1].strip())
                    
                # 处理下午时间
                if am_pm == "下午" and hour < 12:
                    hour += 12
            except ValueError:
                return f"⚠️ 无法识别时间格式: {time}"
    
    # 组合日期和时间
    target_datetime = target_date.replace(hour=hour, minute=minute, second=0)
    formatted_time = target_datetime.strftime("%Y-%m-%d %H:%M:%S")
    
    # 准备商品信息
    goods_id = 1  # 默认飞龙雨
    if goods_name:
        if goods_name == "飞龙雨":
            goods_id = 1
        elif goods_name == "小飞龙":
            goods_id = 5
        elif goods_name == "山知水心":
            goods_id = 3
        else:
            return f"⚠️ 不支持的商品: {goods_name}，请选择'飞龙雨'、'小飞龙'或'山知水心'"
    
    # 调用订水工具
    try:
        result = auto_order.invoke({
            "ID": ID,  # 使用传入的用户ID
            "Action": "订水",
            "GoodsID": goods_id,
            "GoodsName": goods_name or "飞龙雨",
            "Quantity": quantity,
            "AppointmentKind": 1,  # 预约订水
            "AppointmentTime": formatted_time
        })
        
        # 格式化返回结果
        if isinstance(result, dict):
            if result.get('Result') == 1:
                return f"✅ 预约订水成功！\n📅 预约时间: {formatted_time}\n📦 商品: {goods_name or '飞龙雨'} x{quantity}\n📋 订单号: {result.get('OrderNum', '')}\n📍 配送地址: {result.get('Address', '')}"
            else:
                return f"❌ 预约失败: {result.get('Message', '')}"
        else:
            return str(result)
            
    except Exception as e:
        return f"❌ 预约订水失败: {str(e)}" 