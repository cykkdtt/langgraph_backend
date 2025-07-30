from pydantic import BaseModel, Field
from langchain_core.tools import tool
import requests, json

class SearchTicket(BaseModel):
    ID: str = Field(description="用户ID", default="")
    Mobile: str = Field(description="用户手机号", default="")
    Action: str = Field(description="操作类型", default="查询")

@tool(args_schema=SearchTicket)
def search_ticket(ID: str = "125748", Mobile: str = "", Action: str = "查询"):
    """
    查询水票工具
    
    功能：查询会员剩余的水票数量
    
    参数说明：
    - ID: 用户ID（可选，与Mobile二选一）
    - Mobile: 用户手机号（可选，与ID二选一）
    - Action: 操作类型，默认为"查询"
    
    返回格式：
    - Result: 结果状态（1=成功，0=失败，-1=用户验证失败）
    - Message: 结果消息
    - Tickets: 水票列表，包含各种商品的水票余量
      - GoodsID: 商品ID
      - GoodsName: 商品名称  
      - ShortName: 商品简称
      - Count: 剩余数量
    
    支持的商品：
    - 飞龙雨 (ID: 1)
    - 小飞龙 (ID: 5)  
    - 山知水心 (ID: 3)
    """
    
    # API 接口地址
    api_url = "https://api.fly96089.com/api/LLM_Api/SearchTickets.php"
    
    # 构建请求参数
    params = {
        "ID": ID,
        "Mobile": Mobile,
        "action": Action
    }
    
    # 移除空值参数
    params = {k: v for k, v in params.items() if v != ""}
    
    try:
        # 发送 POST 请求
        response = requests.post(api_url, data=params, timeout=30)
        response.raise_for_status()  # 检查HTTP错误
        
        # 解析响应
        result = response.json()
        
        # 直接返回原始结果，让大模型自主组织语言
        return result
        
    except requests.exceptions.RequestException as e:
        return {"Result": 0, "Message": f"网络请求失败：{str(e)}", "Tickets": []}
    except json.JSONDecodeError as e:
        return {"Result": 0, "Message": f"响应解析失败：{str(e)}", "Tickets": []}
    except Exception as e:
        return {"Result": 0, "Message": f"未知错误：{str(e)}", "Tickets": []}

