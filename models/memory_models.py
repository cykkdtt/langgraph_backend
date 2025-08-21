from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

# 记忆类型枚举
class MemoryType(str, Enum):
    SEMANTIC = "semantic"  # 语义记忆
    EPISODIC = "episodic"  # 情节记忆
    PROCEDURAL = "procedural"  # 程序记忆
    WORKING = "working"  # 工作记忆
    LONG_TERM = "long_term"  # 长期记忆
    SHORT_TERM = "short_term"  # 短期记忆

# 记忆状态枚举
class MemoryStatus(str, Enum):
    ACTIVE = "active"  # 活跃
    ARCHIVED = "archived"  # 归档
    DELETED = "deleted"  # 已删除
    CONSOLIDATED = "consolidated"  # 已巩固
    DECAYING = "decaying"  # 衰减中

# 记忆重要性级别
class MemoryImportance(str, Enum):
    CRITICAL = "critical"  # 关键
    HIGH = "high"  # 高
    MEDIUM = "medium"  # 中
    LOW = "low"  # 低
    MINIMAL = "minimal"  # 最低

# 记忆访问模式
class AccessPattern(str, Enum):
    FREQUENT = "frequent"  # 频繁访问
    OCCASIONAL = "occasional"  # 偶尔访问
    RARE = "rare"  # 很少访问
    NEVER = "never"  # 从未访问

# 记忆关联类型
class AssociationType(str, Enum):
    CAUSAL = "causal"  # 因果关系
    TEMPORAL = "temporal"  # 时间关系
    SPATIAL = "spatial"  # 空间关系
    SEMANTIC = "semantic"  # 语义关系
    EMOTIONAL = "emotional"  # 情感关系
    CONTEXTUAL = "contextual"  # 上下文关系

# 记忆向量模型
class MemoryVector(BaseModel):
    embedding: List[float] = Field(..., description="向量嵌入")
    dimension: int = Field(..., description="向量维度")
    model: str = Field(..., description="嵌入模型")
    version: str = Field(..., description="模型版本")

# 记忆关联模型
class MemoryAssociation(BaseModel):
    id: str = Field(..., description="关联ID")
    source_memory_id: str = Field(..., description="源记忆ID")
    target_memory_id: str = Field(..., description="目标记忆ID")
    association_type: AssociationType = Field(..., description="关联类型")
    strength: float = Field(..., ge=0.0, le=1.0, description="关联强度")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    context: Dict[str, Any] = Field(default_factory=dict, description="关联上下文")
    created_at: datetime = Field(..., description="创建时间")
    last_accessed: Optional[datetime] = Field(None, description="最后访问时间")

# 记忆元数据模型
class MemoryMetadata(BaseModel):
    source: str = Field(..., description="记忆来源")
    session_id: Optional[str] = Field(None, description="会话ID")
    thread_id: Optional[str] = Field(None, description="线程ID")
    agent_id: Optional[str] = Field(None, description="智能体ID")
    user_id: str = Field(..., description="用户ID")
    tags: List[str] = Field(default_factory=list, description="标签")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    location: Optional[str] = Field(None, description="地理位置")
    emotion: Optional[str] = Field(None, description="情感状态")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")

# 记忆创建请求
class MemoryCreateRequest(BaseModel):
    type: MemoryType = Field(..., description="记忆类型")
    content: str = Field(..., description="记忆内容")
    title: Optional[str] = Field(None, description="记忆标题")
    summary: Optional[str] = Field(None, description="记忆摘要")
    importance: MemoryImportance = Field(default=MemoryImportance.MEDIUM, description="重要性")
    metadata: MemoryMetadata = Field(..., description="元数据")
    vector: Optional[MemoryVector] = Field(None, description="向量嵌入")
    associations: List[str] = Field(default_factory=list, description="关联的记忆ID")
    expiry_date: Optional[datetime] = Field(None, description="过期时间")
    auto_consolidate: bool = Field(default=True, description="是否自动巩固")

# 记忆更新请求
class MemoryUpdateRequest(BaseModel):
    content: Optional[str] = Field(None, description="记忆内容")
    title: Optional[str] = Field(None, description="记忆标题")
    summary: Optional[str] = Field(None, description="记忆摘要")
    importance: Optional[MemoryImportance] = Field(None, description="重要性")
    status: Optional[MemoryStatus] = Field(None, description="记忆状态")
    metadata: Optional[MemoryMetadata] = Field(None, description="元数据")
    vector: Optional[MemoryVector] = Field(None, description="向量嵌入")
    expiry_date: Optional[datetime] = Field(None, description="过期时间")
    auto_consolidate: Optional[bool] = Field(None, description="是否自动巩固")

# 记忆搜索请求
class MemorySearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="搜索查询")
    type: Optional[MemoryType] = Field(None, description="记忆类型")
    status: Optional[MemoryStatus] = Field(None, description="记忆状态")
    importance: Optional[MemoryImportance] = Field(None, description="重要性")
    tags: Optional[List[str]] = Field(None, description="标签")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    thread_id: Optional[str] = Field(None, description="线程ID")
    agent_id: Optional[str] = Field(None, description="智能体ID")
    created_after: Optional[datetime] = Field(None, description="创建时间起")
    created_before: Optional[datetime] = Field(None, description="创建时间止")
    accessed_after: Optional[datetime] = Field(None, description="访问时间起")
    accessed_before: Optional[datetime] = Field(None, description="访问时间止")
    min_importance: Optional[MemoryImportance] = Field(None, description="最低重要性")
    vector_query: Optional[List[float]] = Field(None, description="向量查询")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")
    max_results: int = Field(default=50, ge=1, le=1000, description="最大结果数")
    include_associations: bool = Field(default=False, description="是否包含关联")
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")
    sort_by: str = Field(default="created_at", description="排序字段")
    sort_order: str = Field(default="desc", description="排序方向")

# 记忆相似性搜索请求
class MemorySimilarityRequest(BaseModel):
    query_vector: List[float] = Field(..., description="查询向量")
    memory_type: Optional[MemoryType] = Field(None, description="记忆类型")
    user_id: Optional[str] = Field(None, description="用户ID")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")
    top_k: int = Field(default=10, ge=1, le=100, description="返回前K个结果")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    filter_conditions: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")

# 记忆巩固请求
class MemoryConsolidationRequest(BaseModel):
    memory_ids: List[str] = Field(..., description="要巩固的记忆ID列表")
    consolidation_strategy: str = Field(default="importance", description="巩固策略")
    merge_similar: bool = Field(default=True, description="是否合并相似记忆")
    similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0, description="相似度阈值")
    preserve_original: bool = Field(default=False, description="是否保留原始记忆")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="巩固元数据")

# 记忆信息模型
class MemoryInfo(BaseModel):
    id: str = Field(..., description="记忆ID")
    type: MemoryType = Field(..., description="记忆类型")
    content: str = Field(..., description="记忆内容")
    title: Optional[str] = Field(None, description="记忆标题")
    summary: Optional[str] = Field(None, description="记忆摘要")
    importance: MemoryImportance = Field(..., description="重要性")
    status: MemoryStatus = Field(..., description="记忆状态")
    metadata: MemoryMetadata = Field(..., description="元数据")
    vector: Optional[MemoryVector] = Field(None, description="向量嵌入")
    associations: List[MemoryAssociation] = Field(default_factory=list, description="关联记忆")
    access_count: int = Field(default=0, description="访问次数")
    access_pattern: AccessPattern = Field(default=AccessPattern.NEVER, description="访问模式")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    last_accessed: Optional[datetime] = Field(None, description="最后访问时间")
    expiry_date: Optional[datetime] = Field(None, description="过期时间")
    consolidation_count: int = Field(default=0, description="巩固次数")
    decay_factor: float = Field(default=1.0, ge=0.0, le=1.0, description="衰减因子")
    retrieval_strength: float = Field(default=1.0, ge=0.0, le=1.0, description="检索强度")

# 记忆统计信息
class MemoryStatistics(BaseModel):
    total_memories: int = Field(default=0, description="总记忆数")
    by_type: Dict[str, int] = Field(default_factory=dict, description="按类型统计")
    by_status: Dict[str, int] = Field(default_factory=dict, description="按状态统计")
    by_importance: Dict[str, int] = Field(default_factory=dict, description="按重要性统计")
    average_access_count: float = Field(default=0.0, description="平均访问次数")
    most_accessed_memory: Optional[str] = Field(None, description="最常访问的记忆")
    recent_memories: int = Field(default=0, description="最近记忆数（24小时内）")
    consolidated_memories: int = Field(default=0, description="已巩固记忆数")
    expired_memories: int = Field(default=0, description="已过期记忆数")
    memory_size_mb: float = Field(default=0.0, description="记忆总大小（MB）")
    association_count: int = Field(default=0, description="关联总数")
    average_importance: float = Field(default=0.0, description="平均重要性")

# 记忆相似性结果
class MemorySimilarityResult(BaseModel):
    memory: MemoryInfo = Field(..., description="记忆信息")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="相似度分数")
    distance: float = Field(..., description="向量距离")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="相关性分数")

# 响应模型
class MemoryCreateResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    memory: Optional[MemoryInfo] = Field(None, description="记忆信息")

class MemoryUpdateResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    memory: Optional[MemoryInfo] = Field(None, description="记忆信息")

class MemoryDeleteResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")

class MemoryListResponse(BaseModel):
    memories: List[MemoryInfo] = Field(..., description="记忆列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页")
    page_size: int = Field(..., description="每页大小")
    has_next: bool = Field(..., description="是否有下一页")
    statistics: Optional[MemoryStatistics] = Field(None, description="统计信息")

class MemorySimilarityResponse(BaseModel):
    results: List[MemorySimilarityResult] = Field(..., description="相似性搜索结果")
    total: int = Field(..., description="总结果数")
    query_time_ms: float = Field(..., description="查询时间（毫秒）")
    threshold_used: float = Field(..., description="使用的阈值")

class MemoryConsolidationResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    consolidated_memory: Optional[MemoryInfo] = Field(None, description="巩固后的记忆")
    merged_count: int = Field(default=0, description="合并的记忆数量")
    original_ids: List[str] = Field(default_factory=list, description="原始记忆ID列表")

# 记忆导出请求
class MemoryExportRequest(BaseModel):
    memory_ids: Optional[List[str]] = Field(None, description="指定记忆ID列表")
    memory_type: Optional[MemoryType] = Field(None, description="记忆类型")
    user_id: Optional[str] = Field(None, description="用户ID")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="日期范围")
    format: str = Field(default="json", description="导出格式")
    include_vectors: bool = Field(default=False, description="是否包含向量")
    include_associations: bool = Field(default=True, description="是否包含关联")
    compress: bool = Field(default=True, description="是否压缩")

class MemoryExportResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    export_id: str = Field(..., description="导出ID")
    download_url: str = Field(..., description="下载链接")
    file_size: int = Field(..., description="文件大小（字节）")
    memory_count: int = Field(..., description="记忆数量")
    expires_at: datetime = Field(..., description="链接过期时间")

# 记忆分析请求
class MemoryAnalysisRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="用户ID")
    analysis_type: str = Field(..., description="分析类型")
    time_range: Optional[Dict[str, datetime]] = Field(None, description="时间范围")
    memory_types: Optional[List[MemoryType]] = Field(None, description="记忆类型")
    include_trends: bool = Field(default=True, description="是否包含趋势")
    include_patterns: bool = Field(default=True, description="是否包含模式")
    granularity: str = Field(default="day", description="时间粒度")

class MemoryAnalysisData(BaseModel):
    memory_growth: Dict[str, int] = Field(default_factory=dict, description="记忆增长")
    access_patterns: Dict[str, Any] = Field(default_factory=dict, description="访问模式")
    importance_distribution: Dict[str, int] = Field(default_factory=dict, description="重要性分布")
    type_distribution: Dict[str, int] = Field(default_factory=dict, description="类型分布")
    consolidation_trends: Dict[str, Any] = Field(default_factory=dict, description="巩固趋势")
    decay_analysis: Dict[str, Any] = Field(default_factory=dict, description="衰减分析")
    association_network: Dict[str, Any] = Field(default_factory=dict, description="关联网络")
    retrieval_efficiency: Dict[str, float] = Field(default_factory=dict, description="检索效率")

class MemoryAnalysisResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    analysis_data: MemoryAnalysisData = Field(..., description="分析数据")
    insights: List[str] = Field(default_factory=list, description="洞察")
    recommendations: List[str] = Field(default_factory=list, description="建议")
    generated_at: datetime = Field(..., description="生成时间")

# 批量操作请求
class MemoryBatchOperation(BaseModel):
    operation: str = Field(..., description="操作类型")
    memory_ids: List[str] = Field(..., description="记忆ID列表")
    params: Dict[str, Any] = Field(default_factory=dict, description="操作参数")

class MemoryBatchOperationResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="操作结果")
    failed_ids: List[str] = Field(default_factory=list, description="失败的ID")

class MemoryStatisticsResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    statistics: MemoryStatistics = Field(..., description="记忆统计信息")