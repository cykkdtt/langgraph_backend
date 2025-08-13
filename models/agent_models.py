"""
智能体相关数据模型

定义智能体管理API的请求和响应模型。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AgentType(str, Enum):
    """智能体类型"""
    CHAT = "chat"
    RAG = "rag"
    TOOL = "tool"
    WORKFLOW = "workflow"
    MULTI_AGENT = "multi_agent"


class AgentStatus(str, Enum):
    """智能体状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentCapability(BaseModel):
    """智能体能力"""
    name: str = Field(description="能力名称")
    description: str = Field(description="能力描述")
    enabled: bool = Field(description="是否启用")
    config: Optional[Dict[str, Any]] = Field(None, description="能力配置")


class AgentTool(BaseModel):
    """智能体工具"""
    name: str = Field(description="工具名称")
    description: str = Field(description="工具描述")
    enabled: bool = Field(description="是否启用")
    config: Optional[Dict[str, Any]] = Field(None, description="工具配置")


class AgentConfig(BaseModel):
    """智能体配置"""
    model: str = Field(description="使用的模型")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(None, ge=1, description="最大token数")
    system_prompt: Optional[str] = Field(None, description="系统提示")
    tools: List[AgentTool] = Field(default_factory=list, description="可用工具")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="智能体能力")
    memory_config: Optional[Dict[str, Any]] = Field(None, description="记忆配置")
    rag_config: Optional[Dict[str, Any]] = Field(None, description="RAG配置")
    workflow_config: Optional[Dict[str, Any]] = Field(None, description="工作流配置")


class AgentInfo(BaseModel):
    """智能体信息"""
    id: str = Field(description="智能体ID")
    name: str = Field(description="智能体名称")
    description: Optional[str] = Field(None, description="智能体描述")
    type: AgentType = Field(description="智能体类型")
    status: AgentStatus = Field(description="智能体状态")
    config: AgentConfig = Field(description="智能体配置")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    created_by: Optional[str] = Field(None, description="创建者")
    version: str = Field(description="版本号")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class CreateAgentRequest(BaseModel):
    """创建智能体请求"""
    name: str = Field(description="智能体名称")
    description: Optional[str] = Field(None, description="智能体描述")
    type: AgentType = Field(description="智能体类型")
    config: AgentConfig = Field(description="智能体配置")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class UpdateAgentRequest(BaseModel):
    """更新智能体请求"""
    name: Optional[str] = Field(None, description="智能体名称")
    description: Optional[str] = Field(None, description="智能体描述")
    status: Optional[AgentStatus] = Field(None, description="智能体状态")
    config: Optional[AgentConfig] = Field(None, description="智能体配置")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class AgentListRequest(BaseModel):
    """智能体列表请求"""
    type: Optional[AgentType] = Field(None, description="智能体类型")
    status: Optional[AgentStatus] = Field(None, description="智能体状态")
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页大小")
    search: Optional[str] = Field(None, description="搜索关键词")
    created_by: Optional[str] = Field(None, description="创建者")


class AgentInstanceRequest(BaseModel):
    """智能体实例请求"""
    agent_id: str = Field(description="智能体ID")
    instance_config: Optional[Dict[str, Any]] = Field(None, description="实例配置")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class AgentInstanceResponse(BaseModel):
    """智能体实例响应"""
    instance_id: str = Field(description="实例ID")
    agent_id: str = Field(description="智能体ID")
    status: str = Field(description="实例状态")
    created_at: datetime = Field(description="创建时间")
    config: Dict[str, Any] = Field(description="实例配置")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class AgentTypeInfo(BaseModel):
    """智能体类型信息"""
    type: AgentType = Field(description="智能体类型")
    name: str = Field(description="类型名称")
    description: str = Field(description="类型描述")
    capabilities: List[str] = Field(description="支持的能力")
    required_config: List[str] = Field(description="必需的配置项")
    optional_config: List[str] = Field(description="可选的配置项")


class AgentMetrics(BaseModel):
    """智能体指标"""
    agent_id: str = Field(description="智能体ID")
    total_conversations: int = Field(description="总对话数")
    total_messages: int = Field(description="总消息数")
    avg_response_time: float = Field(description="平均响应时间")
    success_rate: float = Field(description="成功率")
    error_count: int = Field(description="错误次数")
    last_active: datetime = Field(description="最后活跃时间")
    uptime: float = Field(description="运行时间")


class AgentHealthCheck(BaseModel):
    """智能体健康检查"""
    agent_id: str = Field(description="智能体ID")
    status: str = Field(description="健康状态")
    checks: Dict[str, Dict[str, Any]] = Field(description="检查项目")
    timestamp: datetime = Field(description="检查时间")