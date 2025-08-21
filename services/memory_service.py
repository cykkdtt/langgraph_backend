"""记忆服务

提供记忆管理相关的业务逻辑，包括语义记忆、情节记忆、程序记忆的创建、检索、更新等功能。
"""

import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum
import math

from pydantic import BaseModel, validator
from sqlalchemy.orm import Session

from .base import BaseService, ServiceError, CacheConfig, publish_event
from ..models.database_models import Memory, Session as ChatSession, User
from ..models.response_models import MemoryResponse, BaseResponse
from ..database.repositories import MemoryRepository, SessionRepository, UserRepository
from ..utils.validation import (
    ValidationException, BusinessRuleException, 
    PermissionDeniedException, DataValidator
)
from ..utils.performance_monitoring import monitor_performance


class MemoryType(str, Enum):
    """记忆类型枚举"""
    SEMANTIC = "semantic"  # 语义记忆：事实、概念、知识
    EPISODIC = "episodic"  # 情节记忆：事件、经历、对话
    PROCEDURAL = "procedural"  # 程序记忆：技能、习惯、流程
    WORKING = "working"  # 工作记忆：临时、短期记忆


class MemoryImportance(str, Enum):
    """记忆重要性枚举"""
    CRITICAL = "critical"  # 关键记忆
    HIGH = "high"  # 高重要性
    MEDIUM = "medium"  # 中等重要性
    LOW = "low"  # 低重要性
    TRIVIAL = "trivial"  # 琐碎记忆


class MemoryCreateSchema(BaseModel):
    """记忆创建模式"""
    session_id: Optional[UUID] = None
    memory_type: MemoryType
    content: str
    metadata: Optional[Dict[str, Any]] = None
    importance: Optional[MemoryImportance] = MemoryImportance.MEDIUM
    tags: Optional[List[str]] = None
    related_memory_ids: Optional[List[UUID]] = None
    embedding_vector: Optional[List[float]] = None
    
    @validator('content')
    def validate_content(cls, v):
        if not DataValidator.validate_content_length(v, min_length=1, max_length=10000):
            raise ValueError("Content must be between 1 and 10000 characters")
        return v
    
    @validator('embedding_vector')
    def validate_embedding_vector(cls, v):
        if v is not None:
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError("Embedding vector must be a non-empty list")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Embedding vector must contain only numbers")
            if len(v) > 1536:  # OpenAI embedding dimension limit
                raise ValueError("Embedding vector dimension too large")
        return v


class MemoryUpdateSchema(BaseModel):
    """记忆更新模式"""
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    importance: Optional[MemoryImportance] = None
    tags: Optional[List[str]] = None
    related_memory_ids: Optional[List[UUID]] = None
    embedding_vector: Optional[List[float]] = None
    is_archived: Optional[bool] = None


class MemorySearchSchema(BaseModel):
    """记忆搜索模式"""
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    memory_type: Optional[MemoryType] = None
    content_query: Optional[str] = None
    importance: Optional[MemoryImportance] = None
    tags: Optional[List[str]] = None
    created_from: Optional[datetime] = None
    created_to: Optional[datetime] = None
    last_accessed_from: Optional[datetime] = None
    last_accessed_to: Optional[datetime] = None
    min_relevance_score: Optional[float] = None
    is_archived: Optional[bool] = None
    limit: Optional[int] = 50
    offset: Optional[int] = 0


class MemoryRetrievalSchema(BaseModel):
    """记忆检索模式"""
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    memory_types: Optional[List[MemoryType]] = None
    session_id: Optional[UUID] = None
    importance_threshold: Optional[MemoryImportance] = None
    similarity_threshold: Optional[float] = 0.7
    time_decay_factor: Optional[float] = 0.1
    max_results: Optional[int] = 20
    include_archived: Optional[bool] = False
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v
    
    @validator('time_decay_factor')
    def validate_time_decay_factor(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Time decay factor must be between 0.0 and 1.0")
        return v


class MemoryConsolidationSchema(BaseModel):
    """记忆整合模式"""
    session_id: Optional[UUID] = None
    memory_type: Optional[MemoryType] = None
    consolidation_strategy: str = "similarity"  # similarity, temporal, importance
    similarity_threshold: Optional[float] = 0.8
    time_window_hours: Optional[int] = 24
    min_importance: Optional[MemoryImportance] = MemoryImportance.LOW
    max_consolidated_memories: Optional[int] = 100
    
    @validator('consolidation_strategy')
    def validate_strategy(cls, v):
        valid_strategies = ["similarity", "temporal", "importance", "hybrid"]
        if v not in valid_strategies:
            raise ValueError(f"Strategy must be one of: {valid_strategies}")
        return v


class MemoryStatsResponse(BaseModel):
    """记忆统计响应"""
    total_memories: int
    memories_by_type: Dict[str, int]
    memories_by_importance: Dict[str, int]
    memories_by_session: Dict[str, int]
    archived_memories: int
    active_memories: int
    average_relevance_score: float
    memory_growth_rate: float
    most_accessed_memories: List[Dict[str, Any]]
    recent_memories: List[Dict[str, Any]]
    memory_consolidation_stats: Dict[str, Any]
    generated_at: datetime


class MemoryAnalyticsResponse(BaseModel):
    """记忆分析响应"""
    memory_distribution: Dict[str, Any]
    access_patterns: Dict[str, Any]
    consolidation_opportunities: List[Dict[str, Any]]
    memory_decay_analysis: Dict[str, Any]
    relationship_network: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime


class MemoryService(BaseService[Memory, MemoryCreateSchema, MemoryUpdateSchema, MemoryResponse]):
    """记忆服务"""
    
    def __init__(
        self, 
        repository: MemoryRepository,
        session_repository: SessionRepository,
        user_repository: UserRepository,
        session: Optional[Session] = None
    ):
        cache_config = CacheConfig(
            enabled=True,
            ttl=300,  # 5分钟
            key_prefix="memory_service",
            invalidate_on_update=True
        )
        super().__init__(repository, MemoryResponse, cache_config, session)
        self.memory_repository = repository
        self.session_repository = session_repository
        self.user_repository = user_repository
    
    def _calculate_relevance_score(
        self, 
        memory: Memory, 
        query_embedding: Optional[List[float]] = None,
        time_decay_factor: float = 0.1
    ) -> float:
        """计算记忆相关性评分"""
        base_score = 0.5  # 基础分数
        
        # 重要性权重
        importance_weights = {
            MemoryImportance.CRITICAL: 1.0,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.MEDIUM: 0.6,
            MemoryImportance.LOW: 0.4,
            MemoryImportance.TRIVIAL: 0.2
        }
        importance_score = importance_weights.get(memory.importance, 0.6)
        
        # 访问频率权重
        access_score = min(memory.access_count / 10, 1.0)  # 10次访问为满分
        
        # 时间衰减
        if memory.created_at:
            days_since_created = (datetime.utcnow() - memory.created_at).days
            time_score = math.exp(-time_decay_factor * days_since_created)
        else:
            time_score = 0.5
        
        # 语义相似性（如果提供了查询向量）
        similarity_score = 0.5
        if query_embedding and memory.embedding_vector:
            try:
                memory_embedding = json.loads(memory.embedding_vector)
                similarity_score = self._calculate_cosine_similarity(
                    query_embedding, memory_embedding
                )
            except (json.JSONDecodeError, ValueError):
                similarity_score = 0.5
        
        # 综合评分
        relevance_score = (
            base_score * 0.1 +
            importance_score * 0.3 +
            access_score * 0.2 +
            time_score * 0.2 +
            similarity_score * 0.2
        )
        
        return round(relevance_score, 3)
    
    def _calculate_cosine_similarity(
        self, 
        vector1: List[float], 
        vector2: List[float]
    ) -> float:
        """计算余弦相似度"""
        try:
            if len(vector1) != len(vector2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vector1, vector2))
            magnitude1 = math.sqrt(sum(a * a for a in vector1))
            magnitude2 = math.sqrt(sum(a * a for a in vector2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception:
            return 0.0
    
    def _apply_memory_decay(self, memory: Memory) -> float:
        """应用记忆衰减机制"""
        if not memory.last_accessed_at:
            return memory.relevance_score or 0.5
        
        # 计算自上次访问以来的天数
        days_since_access = (datetime.utcnow() - memory.last_accessed_at).days
        
        # 根据记忆类型设置不同的衰减率
        decay_rates = {
            MemoryType.WORKING: 0.5,  # 工作记忆衰减最快
            MemoryType.EPISODIC: 0.1,  # 情节记忆中等衰减
            MemoryType.SEMANTIC: 0.05,  # 语义记忆衰减较慢
            MemoryType.PROCEDURAL: 0.02  # 程序记忆衰减最慢
        }
        
        decay_rate = decay_rates.get(memory.memory_type, 0.1)
        decay_factor = math.exp(-decay_rate * days_since_access)
        
        # 应用衰减
        current_score = memory.relevance_score or 0.5
        decayed_score = current_score * decay_factor
        
        return max(decayed_score, 0.1)  # 最低保持0.1的相关性
    
    def _validate_business_rules(self, data: Any, action: str = "create"):
        """验证业务规则"""
        if action == "create":
            # 检查会话是否存在（如果指定了会话ID）
            if hasattr(data, 'session_id') and data.session_id:
                session = self.session_repository.get_by_id(data.session_id)
                if not session:
                    raise BusinessRuleException(f"Session {data.session_id} not found")
                
                # 检查会话是否属于当前用户
                if str(session.user_id) != str(self.current_user_id):
                    raise PermissionDeniedException("Cannot create memory for other user's session")
            
            # 检查相关记忆是否存在
            if hasattr(data, 'related_memory_ids') and data.related_memory_ids:
                for memory_id in data.related_memory_ids:
                    related_memory = self.memory_repository.get_by_id(memory_id)
                    if not related_memory:
                        raise BusinessRuleException(f"Related memory {memory_id} not found")
                    
                    # 检查相关记忆是否属于当前用户
                    if str(related_memory.user_id) != str(self.current_user_id):
                        raise PermissionDeniedException("Cannot reference other user's memory")
    
    def _check_permission(self, action: str, resource: Any = None):
        """检查权限"""
        if not self.current_user_id:
            raise PermissionDeniedException("Authentication required")
        
        if action in ["create", "update", "delete", "retrieve"] and resource:
            # 检查记忆是否属于当前用户
            if hasattr(resource, 'user_id'):
                if str(resource.user_id) != str(self.current_user_id):
                    # 检查是否是管理员
                    if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                        raise PermissionDeniedException("Can only access your own memories")
        
        if action in ["admin_stats", "admin_search", "consolidate_all"]:
            # 需要管理员权限
            if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                raise PermissionDeniedException("Admin privileges required")
    
    @monitor_performance
    @publish_event("memory_created", "memory")
    def create_memory(self, data: MemoryCreateSchema) -> BaseResponse[MemoryResponse]:
        """创建记忆"""
        try:
            # 验证业务规则
            self._validate_business_rules(data, "create")
            
            # 创建记忆数据
            memory_data = {
                "id": uuid4(),
                "user_id": UUID(self.current_user_id),
                "session_id": data.session_id,
                "memory_type": data.memory_type,
                "content": data.content,
                "metadata": json.dumps(data.metadata) if data.metadata else None,
                "importance": data.importance or MemoryImportance.MEDIUM,
                "tags": json.dumps(data.tags) if data.tags else None,
                "related_memory_ids": json.dumps([str(id) for id in data.related_memory_ids]) if data.related_memory_ids else None,
                "embedding_vector": json.dumps(data.embedding_vector) if data.embedding_vector else None,
                "relevance_score": 0.5,  # 初始相关性评分
                "access_count": 0,
                "is_archived": False,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # 创建记忆
            memory = self.memory_repository.create(memory_data)
            
            # 转换为响应模型
            response_data = self._transform_to_response(memory)
            
            return self._create_success_response(
                response_data,
                "Memory created successfully"
            )
            
        except (ValidationException, BusinessRuleException, PermissionDeniedException):
            raise
        except Exception as e:
            self.logger.error(f"Error creating memory: {e}")
            raise ServiceError("Failed to create memory")
    
    @monitor_performance
    def retrieve_memories(
        self, 
        retrieval_params: MemoryRetrievalSchema
    ) -> BaseResponse[List[MemoryResponse]]:
        """检索相关记忆"""
        try:
            # 获取候选记忆
            candidate_memories = self.memory_repository.search_memories(
                user_id=UUID(self.current_user_id),
                session_id=retrieval_params.session_id,
                memory_types=retrieval_params.memory_types,
                content_query=retrieval_params.query_text,
                importance_threshold=retrieval_params.importance_threshold,
                is_archived=retrieval_params.include_archived,
                limit=retrieval_params.max_results * 2  # 获取更多候选，后续过滤
            )
            
            # 计算相关性评分并排序
            scored_memories = []
            for memory in candidate_memories:
                relevance_score = self._calculate_relevance_score(
                    memory,
                    retrieval_params.query_embedding,
                    retrieval_params.time_decay_factor or 0.1
                )
                
                # 应用相似性阈值过滤
                if relevance_score >= (retrieval_params.similarity_threshold or 0.7):
                    scored_memories.append((memory, relevance_score))
            
            # 按相关性评分排序
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            
            # 限制结果数量
            max_results = retrieval_params.max_results or 20
            top_memories = scored_memories[:max_results]
            
            # 更新访问信息
            for memory, score in top_memories:
                self.memory_repository.update_access_info(memory.id)
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(memory) for memory, _ in top_memories
            ]
            
            return self._create_success_response(response_data)
            
        except Exception as e:
            self.logger.error(f"Error retrieving memories: {e}")
            raise ServiceError("Failed to retrieve memories")
    
    @monitor_performance
    def search_memories(
        self, 
        search_params: MemorySearchSchema
    ) -> BaseResponse[List[MemoryResponse]]:
        """搜索记忆"""
        try:
            # 如果指定了用户ID，检查权限
            if search_params.user_id:
                if str(search_params.user_id) != str(self.current_user_id):
                    self._check_permission("admin_search")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not search_params.user_id:
                search_params.user_id = UUID(self.current_user_id)
            
            # 执行搜索
            memories = self.memory_repository.search_memories(
                user_id=search_params.user_id,
                session_id=search_params.session_id,
                memory_type=search_params.memory_type,
                content_query=search_params.content_query,
                importance=search_params.importance,
                tags=search_params.tags,
                created_from=search_params.created_from,
                created_to=search_params.created_to,
                last_accessed_from=search_params.last_accessed_from,
                last_accessed_to=search_params.last_accessed_to,
                min_relevance_score=search_params.min_relevance_score,
                is_archived=search_params.is_archived,
                limit=search_params.limit or 50,
                offset=search_params.offset or 0
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(memory) for memory in memories
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
            raise ServiceError("Failed to search memories")
    
    @monitor_performance
    def get_session_memories(
        self, 
        session_id: UUID,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> BaseResponse[List[MemoryResponse]]:
        """获取会话记忆"""
        try:
            # 检查会话权限
            session = self.session_repository.get_or_404(session_id)
            if str(session.user_id) != str(self.current_user_id):
                if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                    raise PermissionDeniedException("Can only access your own session memories")
            
            # 获取会话记忆
            memories = self.memory_repository.get_session_memories(
                session_id=session_id,
                memory_type=memory_type,
                limit=limit,
                offset=offset
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(memory) for memory in memories
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting session memories: {e}")
            raise ServiceError("Failed to get session memories")
    
    @monitor_performance
    def get_related_memories(
        self, 
        memory_id: UUID,
        max_results: int = 10
    ) -> BaseResponse[List[MemoryResponse]]:
        """获取相关记忆"""
        try:
            # 获取原始记忆
            memory = self.memory_repository.get_or_404(memory_id)
            
            # 权限检查
            self._check_permission("retrieve", memory)
            
            # 获取相关记忆
            related_memories = self.memory_repository.get_related_memories(
                memory_id=memory_id,
                user_id=UUID(self.current_user_id),
                limit=max_results
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(related_memory) for related_memory in related_memories
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting related memories: {e}")
            raise ServiceError("Failed to get related memories")
    
    @monitor_performance
    @publish_event("memory_consolidated", "memory")
    def consolidate_memories(
        self, 
        consolidation_params: MemoryConsolidationSchema
    ) -> BaseResponse[Dict[str, Any]]:
        """整合记忆"""
        try:
            # 获取待整合的记忆
            memories_to_consolidate = self.memory_repository.get_memories_for_consolidation(
                user_id=UUID(self.current_user_id),
                session_id=consolidation_params.session_id,
                memory_type=consolidation_params.memory_type,
                similarity_threshold=consolidation_params.similarity_threshold or 0.8,
                time_window_hours=consolidation_params.time_window_hours or 24,
                min_importance=consolidation_params.min_importance,
                max_memories=consolidation_params.max_consolidated_memories or 100
            )
            
            consolidated_count = 0
            consolidated_groups = []
            
            # 根据策略进行整合
            if consolidation_params.consolidation_strategy == "similarity":
                consolidated_count, consolidated_groups = self._consolidate_by_similarity(
                    memories_to_consolidate,
                    consolidation_params.similarity_threshold or 0.8
                )
            elif consolidation_params.consolidation_strategy == "temporal":
                consolidated_count, consolidated_groups = self._consolidate_by_time(
                    memories_to_consolidate,
                    consolidation_params.time_window_hours or 24
                )
            elif consolidation_params.consolidation_strategy == "importance":
                consolidated_count, consolidated_groups = self._consolidate_by_importance(
                    memories_to_consolidate
                )
            elif consolidation_params.consolidation_strategy == "hybrid":
                consolidated_count, consolidated_groups = self._consolidate_hybrid(
                    memories_to_consolidate,
                    consolidation_params
                )
            
            result = {
                "consolidated_memories": consolidated_count,
                "consolidated_groups": len(consolidated_groups),
                "consolidation_strategy": consolidation_params.consolidation_strategy,
                "consolidation_details": consolidated_groups,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return self._create_success_response(
                result,
                f"Consolidated {consolidated_count} memories into {len(consolidated_groups)} groups"
            )
            
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")
            raise ServiceError("Failed to consolidate memories")
    
    def _consolidate_by_similarity(
        self, 
        memories: List[Memory], 
        threshold: float
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """基于相似性整合记忆"""
        consolidated_count = 0
        consolidated_groups = []
        processed_ids = set()
        
        for i, memory1 in enumerate(memories):
            if memory1.id in processed_ids:
                continue
            
            similar_memories = [memory1]
            processed_ids.add(memory1.id)
            
            # 查找相似记忆
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if memory2.id in processed_ids:
                    continue
                
                # 计算相似度
                similarity = self._calculate_memory_similarity(memory1, memory2)
                if similarity >= threshold:
                    similar_memories.append(memory2)
                    processed_ids.add(memory2.id)
            
            # 如果找到相似记忆，进行整合
            if len(similar_memories) > 1:
                consolidated_memory = self._merge_memories(similar_memories)
                consolidated_groups.append({
                    "consolidated_memory_id": str(consolidated_memory.id),
                    "source_memory_ids": [str(m.id) for m in similar_memories],
                    "similarity_scores": [self._calculate_memory_similarity(memory1, m) for m in similar_memories[1:]]
                })
                consolidated_count += len(similar_memories) - 1
        
        return consolidated_count, consolidated_groups
    
    def _consolidate_by_time(
        self, 
        memories: List[Memory], 
        time_window_hours: int
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """基于时间窗口整合记忆"""
        consolidated_count = 0
        consolidated_groups = []
        
        # 按时间排序
        sorted_memories = sorted(memories, key=lambda m: m.created_at or datetime.min)
        
        i = 0
        while i < len(sorted_memories):
            current_memory = sorted_memories[i]
            time_group = [current_memory]
            
            # 查找时间窗口内的记忆
            j = i + 1
            while j < len(sorted_memories):
                next_memory = sorted_memories[j]
                time_diff = (next_memory.created_at - current_memory.created_at).total_seconds() / 3600
                
                if time_diff <= time_window_hours:
                    time_group.append(next_memory)
                    j += 1
                else:
                    break
            
            # 如果时间组有多个记忆，进行整合
            if len(time_group) > 1:
                consolidated_memory = self._merge_memories(time_group)
                consolidated_groups.append({
                    "consolidated_memory_id": str(consolidated_memory.id),
                    "source_memory_ids": [str(m.id) for m in time_group],
                    "time_window_hours": time_window_hours
                })
                consolidated_count += len(time_group) - 1
            
            i = j if j > i + 1 else i + 1
        
        return consolidated_count, consolidated_groups
    
    def _consolidate_by_importance(
        self, 
        memories: List[Memory]
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """基于重要性整合记忆"""
        consolidated_count = 0
        consolidated_groups = []
        
        # 按重要性分组
        importance_groups = {}
        for memory in memories:
            importance = memory.importance or MemoryImportance.MEDIUM
            if importance not in importance_groups:
                importance_groups[importance] = []
            importance_groups[importance].append(memory)
        
        # 整合低重要性的记忆
        for importance, group_memories in importance_groups.items():
            if importance in [MemoryImportance.LOW, MemoryImportance.TRIVIAL] and len(group_memories) > 5:
                # 将多个低重要性记忆合并为一个
                consolidated_memory = self._merge_memories(group_memories)
                consolidated_groups.append({
                    "consolidated_memory_id": str(consolidated_memory.id),
                    "source_memory_ids": [str(m.id) for m in group_memories],
                    "consolidation_reason": f"Low importance ({importance}) batch consolidation"
                })
                consolidated_count += len(group_memories) - 1
        
        return consolidated_count, consolidated_groups
    
    def _consolidate_hybrid(
        self, 
        memories: List[Memory], 
        params: MemoryConsolidationSchema
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """混合策略整合记忆"""
        # 先按相似性整合
        sim_count, sim_groups = self._consolidate_by_similarity(
            memories, params.similarity_threshold or 0.8
        )
        
        # 再按时间整合
        time_count, time_groups = self._consolidate_by_time(
            memories, params.time_window_hours or 24
        )
        
        # 最后按重要性整合
        imp_count, imp_groups = self._consolidate_by_importance(memories)
        
        total_count = sim_count + time_count + imp_count
        all_groups = sim_groups + time_groups + imp_groups
        
        return total_count, all_groups
    
    def _calculate_memory_similarity(self, memory1: Memory, memory2: Memory) -> float:
        """计算两个记忆的相似度"""
        similarity_score = 0.0
        
        # 内容相似度（简单的词汇重叠）
        content1_words = set(memory1.content.lower().split())
        content2_words = set(memory2.content.lower().split())
        
        if content1_words and content2_words:
            intersection = len(content1_words.intersection(content2_words))
            union = len(content1_words.union(content2_words))
            content_similarity = intersection / union if union > 0 else 0.0
        else:
            content_similarity = 0.0
        
        # 向量相似度
        vector_similarity = 0.0
        if memory1.embedding_vector and memory2.embedding_vector:
            try:
                vector1 = json.loads(memory1.embedding_vector)
                vector2 = json.loads(memory2.embedding_vector)
                vector_similarity = self._calculate_cosine_similarity(vector1, vector2)
            except (json.JSONDecodeError, ValueError):
                vector_similarity = 0.0
        
        # 标签相似度
        tag_similarity = 0.0
        if memory1.tags and memory2.tags:
            try:
                tags1 = set(json.loads(memory1.tags))
                tags2 = set(json.loads(memory2.tags))
                if tags1 and tags2:
                    intersection = len(tags1.intersection(tags2))
                    union = len(tags1.union(tags2))
                    tag_similarity = intersection / union if union > 0 else 0.0
            except (json.JSONDecodeError, ValueError):
                tag_similarity = 0.0
        
        # 综合相似度
        similarity_score = (
            content_similarity * 0.4 +
            vector_similarity * 0.4 +
            tag_similarity * 0.2
        )
        
        return similarity_score
    
    def _merge_memories(self, memories: List[Memory]) -> Memory:
        """合并多个记忆为一个"""
        if not memories:
            raise ValueError("Cannot merge empty memory list")
        
        if len(memories) == 1:
            return memories[0]
        
        # 选择最重要的记忆作为基础
        importance_order = {
            MemoryImportance.CRITICAL: 5,
            MemoryImportance.HIGH: 4,
            MemoryImportance.MEDIUM: 3,
            MemoryImportance.LOW: 2,
            MemoryImportance.TRIVIAL: 1
        }
        
        base_memory = max(memories, key=lambda m: importance_order.get(m.importance, 3))
        
        # 合并内容
        merged_content = base_memory.content
        for memory in memories:
            if memory.id != base_memory.id:
                merged_content += f"\n\n[Merged from memory {memory.id}]: {memory.content}"
        
        # 合并标签
        all_tags = set()
        for memory in memories:
            if memory.tags:
                try:
                    tags = json.loads(memory.tags)
                    all_tags.update(tags)
                except (json.JSONDecodeError, ValueError):
                    pass
        
        # 合并元数据
        merged_metadata = {}
        for memory in memories:
            if memory.metadata:
                try:
                    metadata = json.loads(memory.metadata)
                    merged_metadata.update(metadata)
                except (json.JSONDecodeError, ValueError):
                    pass
        
        # 添加合并信息
        merged_metadata["consolidated_from"] = [str(m.id) for m in memories]
        merged_metadata["consolidation_timestamp"] = datetime.utcnow().isoformat()
        
        # 更新基础记忆
        update_data = {
            "content": merged_content,
            "tags": json.dumps(list(all_tags)) if all_tags else None,
            "metadata": json.dumps(merged_metadata),
            "access_count": sum(m.access_count or 0 for m in memories),
            "relevance_score": max(m.relevance_score or 0.5 for m in memories),
            "updated_at": datetime.utcnow()
        }
        
        updated_memory = self.memory_repository.update(base_memory.id, update_data)
        
        # 删除其他记忆
        for memory in memories:
            if memory.id != base_memory.id:
                self.memory_repository.delete(memory.id)
        
        return updated_memory
    
    @monitor_performance
    def apply_memory_decay(self, user_id: Optional[UUID] = None) -> BaseResponse[Dict[str, Any]]:
        """应用记忆衰减机制"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_stats")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取用户的所有记忆
            memories = self.memory_repository.get_user_memories(user_id)
            
            updated_count = 0
            archived_count = 0
            
            for memory in memories:
                # 计算衰减后的相关性评分
                decayed_score = self._apply_memory_decay(memory)
                
                # 如果相关性评分过低，归档记忆
                if decayed_score < 0.1:
                    self.memory_repository.update(memory.id, {
                        "is_archived": True,
                        "relevance_score": decayed_score,
                        "updated_at": datetime.utcnow()
                    })
                    archived_count += 1
                else:
                    # 更新相关性评分
                    self.memory_repository.update(memory.id, {
                        "relevance_score": decayed_score,
                        "updated_at": datetime.utcnow()
                    })
                
                updated_count += 1
            
            result = {
                "processed_memories": updated_count,
                "archived_memories": archived_count,
                "user_id": str(user_id),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return self._create_success_response(
                result,
                f"Applied decay to {updated_count} memories, archived {archived_count}"
            )
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error applying memory decay: {e}")
            raise ServiceError("Failed to apply memory decay")
    
    @monitor_performance
    def get_memory_statistics(
        self, 
        user_id: Optional[UUID] = None
    ) -> BaseResponse[MemoryStatsResponse]:
        """获取记忆统计信息"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_stats")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取统计数据
            stats = self.memory_repository.get_user_memory_statistics(user_id)
            
            # 获取最常访问的记忆
            most_accessed = self.memory_repository.get_most_accessed_memories(
                user_id, limit=10
            )
            
            # 获取最近的记忆
            recent_memories = self.memory_repository.get_recent_memories(
                user_id, limit=10
            )
            
            # 获取整合统计
            consolidation_stats = self.memory_repository.get_consolidation_statistics(user_id)
            
            memory_stats = MemoryStatsResponse(
                total_memories=stats.get("total_memories", 0),
                memories_by_type=stats.get("memories_by_type", {}),
                memories_by_importance=stats.get("memories_by_importance", {}),
                memories_by_session=stats.get("memories_by_session", {}),
                archived_memories=stats.get("archived_memories", 0),
                active_memories=stats.get("active_memories", 0),
                average_relevance_score=stats.get("avg_relevance_score", 0.0),
                memory_growth_rate=stats.get("growth_rate", 0.0),
                most_accessed_memories=most_accessed,
                recent_memories=recent_memories,
                memory_consolidation_stats=consolidation_stats,
                generated_at=datetime.utcnow()
            )
            
            return self._create_success_response(memory_stats)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting memory statistics: {e}")
            raise ServiceError("Failed to get memory statistics")
    
    @monitor_performance
    def get_memory_analytics(
        self, 
        user_id: Optional[UUID] = None
    ) -> BaseResponse[MemoryAnalyticsResponse]:
        """获取记忆分析"""
        try:
            # 如果指定了用户ID，检查权限
            if user_id and str(user_id) != str(self.current_user_id):
                self._check_permission("admin_stats")
            
            # 如果没有指定用户ID，默认使用当前用户
            if not user_id:
                user_id = UUID(self.current_user_id)
            
            # 获取分析数据
            analytics = self.memory_repository.get_memory_analytics(user_id)
            
            # 生成建议
            recommendations = self._generate_memory_recommendations(analytics)
            
            memory_analytics = MemoryAnalyticsResponse(
                memory_distribution=analytics.get("distribution", {}),
                access_patterns=analytics.get("access_patterns", {}),
                consolidation_opportunities=analytics.get("consolidation_opportunities", []),
                memory_decay_analysis=analytics.get("decay_analysis", {}),
                relationship_network=analytics.get("relationship_network", {}),
                performance_metrics=analytics.get("performance_metrics", {}),
                recommendations=recommendations,
                generated_at=datetime.utcnow()
            )
            
            return self._create_success_response(memory_analytics)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting memory analytics: {e}")
            raise ServiceError("Failed to get memory analytics")
    
    def _generate_memory_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """生成记忆管理建议"""
        recommendations = []
        
        # 基于记忆分布的建议
        distribution = analytics.get("distribution", {})
        total_memories = distribution.get("total", 0)
        
        if total_memories > 1000:
            recommendations.append("考虑定期整合记忆以提高检索效率")
        
        # 基于访问模式的建议
        access_patterns = analytics.get("access_patterns", {})
        low_access_ratio = access_patterns.get("low_access_ratio", 0)
        
        if low_access_ratio > 0.7:
            recommendations.append("有大量记忆很少被访问，建议归档或删除不重要的记忆")
        
        # 基于衰减分析的建议
        decay_analysis = analytics.get("decay_analysis", {})
        high_decay_ratio = decay_analysis.get("high_decay_ratio", 0)
        
        if high_decay_ratio > 0.5:
            recommendations.append("许多记忆的相关性正在快速衰减，建议增加重要记忆的访问频率")
        
        # 基于整合机会的建议
        consolidation_opportunities = analytics.get("consolidation_opportunities", [])
        
        if len(consolidation_opportunities) > 10:
            recommendations.append("发现多个记忆整合机会，建议运行记忆整合以优化存储")
        
        return recommendations


# 便捷函数
def create_memory_service(session: Optional[Session] = None) -> MemoryService:
    """创建记忆服务实例"""
    from ..database.repositories import get_repository_manager
    
    repo_manager = get_repository_manager()
    memory_repository = repo_manager.get_memory_repository(session)
    session_repository = repo_manager.get_session_repository(session)
    user_repository = repo_manager.get_user_repository(session)
    
    return MemoryService(memory_repository, session_repository, user_repository, session)