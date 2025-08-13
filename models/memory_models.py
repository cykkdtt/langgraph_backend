"""
记忆管理相关数据模型

定义记忆管理API的请求和响应模型。
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MemoryType(str, Enum):
    """记忆类型"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class MemoryImportance(str, Enum):
    """记忆重要性"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryStatus(str, Enum):
    """记忆状态"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    PENDING = "pending"


class MemoryItem(BaseModel):
    """记忆项"""
    id: str = Field(description="记忆ID")
    content: str = Field(description="记忆内容")
    type: MemoryType = Field(description="记忆类型")
    importance: MemoryImportance = Field(description="重要性")
    status: MemoryStatus = Field(description="状态")
    agent_id: str = Field(description="智能体ID")
    thread_id: Optional[str] = Field(None, description="会话线程ID")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    accessed_at: Optional[datetime] = Field(None, description="最后访问时间")
    access_count: int = Field(0, description="访问次数")
    tags: List[str] = Field(default_factory=list, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    embedding: Optional[List[float]] = Field(None, description="向量嵌入")
    related_memories: List[str] = Field(default_factory=list, description="相关记忆ID")


class MemoryCreateRequest(BaseModel):
    """创建记忆请求"""
    content: str = Field(description="记忆内容")
    type: MemoryType = Field(description="记忆类型")
    importance: MemoryImportance = Field(MemoryImportance.MEDIUM, description="重要性")
    agent_id: str = Field(description="智能体ID")
    thread_id: Optional[str] = Field(None, description="会话线程ID")
    tags: List[str] = Field(default_factory=list, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class MemoryUpdateRequest(BaseModel):
    """更新记忆请求"""
    content: Optional[str] = Field(None, description="记忆内容")
    type: Optional[MemoryType] = Field(None, description="记忆类型")
    importance: Optional[MemoryImportance] = Field(None, description="重要性")
    status: Optional[MemoryStatus] = Field(None, description="状态")
    tags: Optional[List[str]] = Field(None, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class MemorySearchRequest(BaseModel):
    """记忆搜索请求"""
    query: str = Field(description="搜索查询")
    agent_id: str = Field(description="智能体ID")
    thread_id: Optional[str] = Field(None, description="会话线程ID")
    memory_types: Optional[List[MemoryType]] = Field(None, description="记忆类型过滤")
    importance_levels: Optional[List[MemoryImportance]] = Field(None, description="重要性过滤")
    tags: Optional[List[str]] = Field(None, description="标签过滤")
    limit: int = Field(10, ge=1, le=100, description="返回数量限制")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="相似度阈值")
    include_archived: bool = Field(False, description="是否包含已归档记忆")


class MemorySearchResponse(BaseModel):
    """记忆搜索响应"""
    memories: List[MemoryItem] = Field(description="搜索结果")
    total: int = Field(description="总数量")
    query: str = Field(description="搜索查询")
    search_time: float = Field(description="搜索耗时")


class MemoryListRequest(BaseModel):
    """记忆列表请求"""
    agent_id: str = Field(description="智能体ID")
    thread_id: Optional[str] = Field(None, description="会话线程ID")
    memory_types: Optional[List[MemoryType]] = Field(None, description="记忆类型过滤")
    importance_levels: Optional[List[MemoryImportance]] = Field(None, description="重要性过滤")
    status: Optional[MemoryStatus] = Field(None, description="状态过滤")
    tags: Optional[List[str]] = Field(None, description="标签过滤")
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页大小")
    sort_by: str = Field("created_at", description="排序字段")
    sort_order: str = Field("desc", description="排序顺序")


class MemoryManageRequest(BaseModel):
    """记忆管理请求"""
    action: str = Field(description="管理操作")
    memory_ids: List[str] = Field(description="记忆ID列表")
    parameters: Optional[Dict[str, Any]] = Field(None, description="操作参数")


class MemoryStats(BaseModel):
    """记忆统计"""
    agent_id: str = Field(description="智能体ID")
    total_memories: int = Field(description="总记忆数")
    by_type: Dict[str, int] = Field(description="按类型统计")
    by_importance: Dict[str, int] = Field(description="按重要性统计")
    by_status: Dict[str, int] = Field(description="按状态统计")
    avg_access_count: float = Field(description="平均访问次数")
    last_created: Optional[datetime] = Field(None, description="最后创建时间")
    last_accessed: Optional[datetime] = Field(None, description="最后访问时间")


class MemoryConsolidationRequest(BaseModel):
    """记忆整合请求"""
    agent_id: str = Field(description="智能体ID")
    consolidation_type: str = Field(description="整合类型")
    parameters: Optional[Dict[str, Any]] = Field(None, description="整合参数")


class MemoryConsolidationResponse(BaseModel):
    """记忆整合响应"""
    task_id: str = Field(description="任务ID")
    status: str = Field(description="整合状态")
    created_at: datetime = Field(description="创建时间")
    estimated_duration: Optional[int] = Field(None, description="预估耗时（秒）")


class MemoryRelationship(BaseModel):
    """记忆关系"""
    source_id: str = Field(description="源记忆ID")
    target_id: str = Field(description="目标记忆ID")
    relationship_type: str = Field(description="关系类型")
    strength: float = Field(ge=0.0, le=1.0, description="关系强度")
    created_at: datetime = Field(description="创建时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class MemoryGraph(BaseModel):
    """记忆图"""
    nodes: List[MemoryItem] = Field(description="记忆节点")
    edges: List[MemoryRelationship] = Field(description="记忆关系")
    metadata: Optional[Dict[str, Any]] = Field(None, description="图元数据")