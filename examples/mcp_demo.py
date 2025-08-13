"""
MCP (Model Context Protocol) 集成示例

本示例展示如何在LangGraph项目中使用MCP工具：
1. 基本MCP工具使用
2. 图表生成工具
3. 文件系统访问
4. 数据库查询
5. 网络搜索
"""

import asyncio
import logging
from typing import Dict, Any

from core.tools.mcp_manager import get_mcp_manager, initialize_mcp_manager
from core.tools import get_tool_registry, ToolExecutionContext
from core.agents.base import BaseAgent, AgentState, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class MCPDemoAgent(BaseAgent):
    """MCP演示智能体"""
    
    def __init__(self, llm=None):
        # 提供BaseAgent需要的必需参数
        super().__init__(
            agent_id="mcp_demo_agent",
            name="MCP演示智能体",
            description="演示MCP工具使用的智能体",
            llm=llm  # 可以传入LLM，如果没有则使用默认响应
        )
    
    async def initialize(self):
        """初始化智能体"""
        await super().initialize()
        
        # 确保MCP管理器已初始化
        await initialize_mcp_manager()
        
        # 获取工具注册表并加载MCP工具
        tool_registry = get_tool_registry()
        await tool_registry.load_mcp_tools()
        
        logger.info("MCP演示智能体初始化完成")
    
    async def demo_chart_generation(self) -> Dict[str, Any]:
        """演示图表生成功能"""
        try:
            # 准备图表数据
            chart_data = {
                "type": "bar",
                "data": {
                    "labels": ["一月", "二月", "三月", "四月", "五月"],
                    "datasets": [{
                        "label": "销售额",
                        "data": [12, 19, 3, 5, 2],
                        "backgroundColor": "rgba(54, 162, 235, 0.2)",
                        "borderColor": "rgba(54, 162, 235, 1)",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": "月度销售报告"
                        }
                    }
                }
            }
            
            # 执行图表生成工具
            tool_registry = get_tool_registry()
            context = ToolExecutionContext(
                user_id="demo_user",
                session_id="demo_session",
                permissions=[],
                metadata={"demo": True}
            )
            
            # 查找图表工具
            chart_tools = [tool for tool in tool_registry.list_tools() 
                          if "chart" in tool.name.lower()]
            
            if chart_tools:
                tool_name = chart_tools[0].name
                result = await tool_registry.execute_tool(
                    tool_name, 
                    context,
                    config=chart_data
                )
                
                return {
                    "success": True,
                    "chart_url": result.result,
                    "tool_used": tool_name
                }
            else:
                return {
                    "success": False,
                    "error": "未找到图表生成工具"
                }
                
        except Exception as e:
            logger.error(f"图表生成演示失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def demo_mcp_tools_discovery(self) -> Dict[str, Any]:
        """演示MCP工具发现"""
        try:
            mcp_manager = get_mcp_manager()
            
            # 获取所有MCP工具
            all_tools = await mcp_manager.get_tools()
            
            # 按服务器分组
            tools_by_server = {}
            for server_name in mcp_manager.get_server_names():
                try:
                    server_tools = await mcp_manager.get_tools(server_name)
                    tools_by_server[server_name] = []
                    
                    for tool in server_tools:
                        tools_by_server[server_name].append({
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": getattr(tool, 'input_schema', {})
                        })
                except Exception as e:
                    logger.warning(f"获取服务器 {server_name} 工具失败: {e}")
                    tools_by_server[server_name] = []
            
            return {
                "success": True,
                "total_tools": len(all_tools),
                "tools_by_server": tools_by_server
            }
            
        except Exception as e:
            logger.error(f"MCP工具发现演示失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def demo_resource_access(self) -> Dict[str, Any]:
        """演示资源访问功能"""
        try:
            mcp_manager = get_mcp_manager()
            
            # 获取所有资源
            all_resources = []
            for server_name in mcp_manager.get_server_names():
                try:
                    resources = await mcp_manager.get_resources(server_name)
                    for resource in resources:
                        all_resources.append({
                            "source": resource.source,
                            "mimetype": resource.mimetype,
                            "server": server_name,
                            "size": len(resource.data) if hasattr(resource, 'data') else 0
                        })
                except Exception as e:
                    logger.warning(f"获取服务器 {server_name} 资源失败: {e}")
                    continue
            
            # 尝试读取第一个资源（如果存在）
            resource_content = None
            if all_resources:
                try:
                    first_resource = all_resources[0]
                    # 使用新的资源获取方法
                    resources = await mcp_manager.get_resources(
                        first_resource["server"], 
                        [first_resource["source"]]
                    )
                    if resources:
                        content = resources[0].as_string()
                        resource_content = {
                            "source": first_resource["source"],
                            "content_preview": content[:200] + "..." if len(content) > 200 else content
                        }
                except Exception as e:
                    logger.warning(f"读取资源失败: {e}")
            
            return {
                "success": True,
                "total_resources": len(all_resources),
                "resources": all_resources,
                "sample_content": resource_content
            }
            
        except Exception as e:
            logger.error(f"资源访问演示失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def demo_prompt_execution(self) -> Dict[str, Any]:
        """演示提示执行功能"""
        try:
            mcp_manager = get_mcp_manager()
            
            # 获取所有提示
            all_prompts = []
            for server_name in mcp_manager.clients.keys():
                try:
                    prompts = await mcp_manager.get_prompts(server_name)
                    for prompt in prompts:
                        all_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments,
                            "server": server_name
                        })
                except Exception as e:
                    logger.warning(f"获取服务器 {server_name} 提示失败: {e}")
                    continue
            
            # 尝试执行第一个提示（如果存在）
            prompt_result = None
            if all_prompts:
                try:
                    first_prompt = all_prompts[0]
                    # 构建简单的参数
                    args = {}
                    for arg in first_prompt["arguments"]:
                        if arg.get("required", False):
                            args[arg["name"]] = "demo_value"
                    
                    result = await mcp_manager.get_prompt(first_prompt["name"], args)
                    prompt_result = {
                        "prompt_name": first_prompt["name"],
                        "arguments_used": args,
                        "result_preview": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    }
                except Exception as e:
                    logger.warning(f"执行提示失败: {e}")
            
            return {
                "success": True,
                "total_prompts": len(all_prompts),
                "prompts": all_prompts,
                "sample_execution": prompt_result
            }
            
        except Exception as e:
            logger.error(f"提示执行演示失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def run_full_demo(self) -> Dict[str, Any]:
        """运行完整的MCP演示"""
        demo_results = {}
        
        # 1. 工具发现
        logger.info("开始MCP工具发现演示...")
        demo_results["tools_discovery"] = await self.demo_mcp_tools_discovery()
        
        # 2. 图表生成
        logger.info("开始图表生成演示...")
        demo_results["chart_generation"] = await self.demo_chart_generation()
        
        # 3. 资源访问
        logger.info("开始资源访问演示...")
        demo_results["resource_access"] = await self.demo_resource_access()
        
        # 4. 提示执行
        logger.info("开始提示执行演示...")
        demo_results["prompt_execution"] = await self.demo_prompt_execution()
        
        return {
            "demo_completed": True,
            "results": demo_results,
            "summary": {
                "total_demos": 4,
                "successful_demos": sum(1 for result in demo_results.values() if result.get("success", False)),
                "failed_demos": sum(1 for result in demo_results.values() if not result.get("success", False))
            }
        }


async def main():
    """主演示函数"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 创建演示智能体（不需要LLM，使用默认响应）
        demo_agent = MCPDemoAgent(llm=None)
        await demo_agent.initialize()
        
        # 运行完整演示
        results = await demo_agent.run_full_demo()
        
        print("=== MCP集成演示结果 ===")
        print(f"演示完成: {results['demo_completed']}")
        print(f"成功演示: {results['summary']['successful_demos']}")
        print(f"失败演示: {results['summary']['failed_demos']}")
        
        for demo_name, demo_result in results["results"].items():
            print(f"\n--- {demo_name} ---")
            if demo_result.get("success"):
                print("✅ 成功")
                # 打印关键信息
                if "total_tools" in demo_result:
                    print(f"发现工具数量: {demo_result['total_tools']}")
                if "chart_url" in demo_result:
                    print(f"图表URL: {demo_result['chart_url']}")
                if "total_resources" in demo_result:
                    print(f"资源数量: {demo_result['total_resources']}")
                if "total_prompts" in demo_result:
                    print(f"提示数量: {demo_result['total_prompts']}")
            else:
                print(f"❌ 失败: {demo_result.get('error', '未知错误')}")
        
    except Exception as e:
        logger.error(f"演示运行失败: {e}")
        print(f"演示运行失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())