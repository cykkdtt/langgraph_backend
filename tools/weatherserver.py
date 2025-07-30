from pydantic import BaseModel, Field
from langchain_core.tools import tool
import requests, json
class WeatherQuery(BaseModel):
    city: str = Field(description="The location name of the city")

@tool(args_schema=WeatherQuery)
def get_weather(city: str):
    """
    查询即时天气函数
    :param city: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
    注意，中国的城市不能转换为英文，例如如果需要查询北京市天气，则city参数需要输入'北京市'；
    :return：高德天气 API查询即时天气的结果，具体URL请求地址为：https://restapi.amap.com/v3/weather/weatherInfo\
    返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    url = "https://restapi.amap.com/v3/weather/weatherInfo"

    params = {
        "key": "d8c005390e4ce186cb1c190ccf84d26b",
        "city": city
    }

    response = requests.get(url, params=params)
    
    data = response.json()
    return json.dumps(data)

# print(f'''
# name: {get_weather.name}
# description: {get_weather.description}
# arguments: {get_weather.args}
# ''')