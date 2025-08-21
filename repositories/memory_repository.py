"""记忆仓储模块

提供记忆相关的数据访问接口和业务逻辑。
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, func, text
from sqlalchemy.orm import Session, selectinload

from ..models.database import Memory, MemoryType
from ..models.api import MemoryCreate, MemoryUpdate, MemoryStats
from ..database.repository import CRUDRepository, EntityNotFoundError
from ..database.query_builder import QueryFilter
from ..database.session import SessionManager

logger = logging.getLogger(__name__)


class MemoryRepository(CRUDRepository[Memory]):
    """记忆仓储类"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        super().__init__(Memory, session_manager)
        # 设置软删除字段
        self.set_soft_delete_field("deleted_at")
        # 设置默认加载选项
        self.set_default_load_options(
            selectinload(Memory.session),
            selectinload(Memory.thread)
        )
    
    def create_memory(
        self, 
        memory_create: MemoryCreate, 
        session_id: int,
        thread_id: Optional[int] = None,
        session: Optional[Session] = None
    ) -> Memory:
        """创建记忆"""
        memory_data = memory_create.model_dump()
        memory_data["session_id"] = session_id
        
        if thread_id:
            memory_data["thread_id"] = thread_id
        
        # 设置初始状态
        memory_data["is_active"] = True
        
        return self.create(memory_data, session)
    
    def get_session_memories(
        self, 
        session_id: int, 
        memory_type: Optional[MemoryType] = None,
        skip: int = 0, 
        limit: int = 100,
        include_inactive: bool = False,
        session: Optional[Session] = None
    ) -> List[Memory]:
        """获取会话的记忆列表"""
        filters = QueryFilter().eq("session_id", session_id)
        
        if memory_type:
            filters = filters.and_(QueryFilter().eq("memory_type", memory_type))
        
        if not include_inactive:
            filters = filters.and_(QueryFilter().eq("is_active", True))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("importance", "desc"), ("created_at", "desc")],
            session=session
        )
    
    def get_thread_memories(
        self, 
        thread_id: int, 
        memory_type: Optional[MemoryType] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Memory]:
        """获取线程的记忆列表"""
        filters = QueryFilter().eq("thread_id", thread_id)
        
        if memory_type:
            filters = filters.and_(QueryFilter().eq("memory_type", memory_type))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("importance", "desc"), ("created_at", "desc")],
            session=session
        )
    
    def get_memories_by_type(
        self, 
        memory_type: MemoryType,
        session_id: Optional[int] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Memory]:
        """根据类型获取记忆"""
        filters = QueryFilter().eq("memory_type", memory_type)
        
        if session_id:
            filters = filters.and_(QueryFilter().eq("session_id", session_id))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("importance", "desc"), ("created_at", "desc")],
            session=session
        )
    
    def get_important_memories(
        self, 
        session_id: int,
        min_importance: float = 0.7,
        memory_type: Optional[MemoryType] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Memory]:
        """获取重要记忆"""
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
            QueryFilter().gte("importance", min_importance),
            QueryFilter().eq("is_active", True)
        )
        
        if memory_type:
            filters = filters.and_(QueryFilter().eq("memory_type", memory_type))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("importance", "desc"), ("created_at", "desc")],
            session=session
        )
    
    def get_recent_memories(
        self, 
        session_id: int,
        hours: int = 24,
        memory_type: Optional[MemoryType] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Memory]:
        """获取最近的记忆"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
            QueryFilter().gte("created_at", cutoff_time),
            QueryFilter().eq("is_active", True)
        )
        
        if memory_type:
            filters = filters.and_(QueryFilter().eq("memory_type", memory_type))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def search_memories(
        self, 
        session_id: int,
        query: str, 
        memory_type: Optional[MemoryType] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Memory]:
        """搜索记忆内容"""
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
            QueryFilter().or_(
                QueryFilter().ilike("content", f"%{query}%"),
                QueryFilter().ilike("summary", f"%{query}%")
            ),
            QueryFilter().eq("is_active", True)
        )
        
        if memory_type:
            filters = filters.and_(QueryFilter().eq("memory_type", memory_type))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("importance", "desc"), ("created_at", "desc")],
            session=session
        )
    
    def search_memories_by_embedding(
        self, 
        session_id: int,
        query_embedding: List[float],
        similarity_threshold: float = 0.7,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        session: Optional[Session] = None
    ) -> List[Tuple[Memory, float]]:
        """基于向量相似度搜索记忆"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 构建基础查询
            query = db_session.query(Memory)
            query = query.filter(Memory.session_id == session_id)
            query = query.filter(Memory.is_active == True)
            query = query.filter(Memory.embedding.isnot(None))
            
            if memory_type:
                query = query.filter(Memory.memory_type == memory_type)
            
            # 计算余弦相似度（需要数据库支持向量操作）
            # 这里使用简化的实现，实际项目中可能需要使用专门的向量数据库
            memories = query.all()
            
            results = []
            for memory in memories:
                if memory.embedding:
                    # 计算余弦相似度
                    similarity = self._calculate_cosine_similarity(
                        query_embedding, 
                        memory.embedding
                    )
                    
                    if similarity >= similarity_threshold:
                        results.append((memory, similarity))
            
            # 按相似度排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching memories by embedding: {e}")
            return []
        finally:
            if not session:
                db_session.close()
    
    def _calculate_cosine_similarity(
        self, 
        vec1: List[float], 
        vec2: List[float]
    ) -> float:
        """计算余弦相似度"""
        try:
            import numpy as np
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def update_memory_importance(
        self, 
        memory_id: int,
        importance: float,
        session: Optional[Session] = None
    ) -> Optional[Memory]:
        """更新记忆重要性"""
        try:
            memory = self.get(memory_id, session)
            if not memory:
                raise EntityNotFoundError(f"Memory with ID {memory_id} not found")
            
            # 确保重要性在0-1之间
            importance = max(0, min(1, importance))
            
            update_data = {
                "importance": importance
            }
            
            return self.update(memory, update_data, session)
            
        except Exception as e:
            logger.error(f"Error updating memory importance for memory {memory_id}: {e}")
            return None
    
    def update_memory_embedding(
        self, 
        memory_id: int,
        embedding: List[float],
        session: Optional[Session] = None
    ) -> Optional[Memory]:
        """更新记忆向量"""
        try:
            memory = self.get(memory_id, session)
            if not memory:
                raise EntityNotFoundError(f"Memory with ID {memory_id} not found")
            
            update_data = {
                "embedding": embedding
            }
            
            return self.update(memory, update_data, session)
            
        except Exception as e:
            logger.error(f"Error updating memory embedding for memory {memory_id}: {e}")
            return None
    
    def update_memory_metadata(
        self, 
        memory_id: int,
        metadata: Dict[str, Any],
        session: Optional[Session] = None
    ) -> Optional[Memory]:
        """更新记忆元数据"""
        try:
            memory = self.get(memory_id, session)
            if not memory:
                raise EntityNotFoundError(f"Memory with ID {memory_id} not found")
            
            # 合并元数据
            current_metadata = memory.metadata or {}
            current_metadata.update(metadata)
            
            update_data = {
                "metadata": current_metadata
            }
            
            return self.update(memory, update_data, session)
            
        except Exception as e:
            logger.error(f"Error updating memory metadata for memory {memory_id}: {e}")
            return None
    
    def deactivate_memory(
        self, 
        memory_id: int,
        session: Optional[Session] = None
    ) -> Optional[Memory]:
        """停用记忆"""
        try:
            memory = self.get(memory_id, session)
            if not memory:
                raise EntityNotFoundError(f"Memory with ID {memory_id} not found")
            
            update_data = {
                "is_active": False
            }
            
            return self.update(memory, update_data, session)
            
        except Exception as e:
            logger.error(f"Error deactivating memory {memory_id}: {e}")
            return None
    
    def activate_memory(
        self, 
        memory_id: int,
        session: Optional[Session] = None
    ) -> Optional[Memory]:
        """激活记忆"""
        try:
            memory = self.get(memory_id, session)
            if not memory:
                raise EntityNotFoundError(f"Memory with ID {memory_id} not found")
            
            update_data = {
                "is_active": True
            }
            
            return self.update(memory, update_data, session)
            
        except Exception as e:
            logger.error(f"Error activating memory {memory_id}: {e}")
            return None
    
    def get_memories_by_date_range(
        self, 
        session_id: int,
        start_date: datetime,
        end_date: datetime,
        memory_type: Optional[MemoryType] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Memory]:
        """根据日期范围获取记忆"""
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
            QueryFilter().gte("created_at", start_date),
            QueryFilter().lte("created_at", end_date),
            QueryFilter().eq("is_active", True)
        )
        
        if memory_type:
            filters = filters.and_(QueryFilter().eq("memory_type", memory_type))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_memory_statistics(
        self, 
        session_id: Optional[int] = None,
        user_id: Optional[int] = None,
        session: Optional[Session] = None
    ) -> MemoryStats:
        """获取记忆统计信息"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            base_filters = QueryFilter().eq("is_active", True)
            
            if session_id:
                base_filters = base_filters.and_(QueryFilter().eq("session_id", session_id))
            elif user_id:
                # 通过session表连接查询用户的记忆
                query = db_session.query(Memory).join(Memory.session)
                query = query.filter(Memory.session.has(user_id=user_id))
                query = query.filter(Memory.is_active == True)
                
                total_memories = query.count()
                semantic_memories = query.filter(Memory.memory_type == MemoryType.SEMANTIC).count()
                episodic_memories = query.filter(Memory.memory_type == MemoryType.EPISODIC).count()
                procedural_memories = query.filter(Memory.memory_type == MemoryType.PROCEDURAL).count()
                
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                today_memories = query.filter(Memory.created_at >= today_start).count()
                
                # 计算平均重要性
                avg_importance_result = query.with_entities(func.avg(Memory.importance)).scalar()
                avg_importance = float(avg_importance_result) if avg_importance_result else 0.0
                
                return MemoryStats(
                    total_memories=total_memories,
                    semantic_memories=semantic_memories,
                    episodic_memories=episodic_memories,
                    procedural_memories=procedural_memories,
                    today_memories=today_memories,
                    avg_importance=round(avg_importance, 3)
                )
            
            # 总记忆数
            total_memories = self.count(filters=base_filters, session=db_session)
            
            # 按类型统计
            semantic_filters = base_filters.and_(QueryFilter().eq("memory_type", MemoryType.SEMANTIC))
            semantic_memories = self.count(filters=semantic_filters, session=db_session)
            
            episodic_filters = base_filters.and_(QueryFilter().eq("memory_type", MemoryType.EPISODIC))
            episodic_memories = self.count(filters=episodic_filters, session=db_session)
            
            procedural_filters = base_filters.and_(QueryFilter().eq("memory_type", MemoryType.PROCEDURAL))
            procedural_memories = self.count(filters=procedural_filters, session=db_session)
            
            # 今日记忆数
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_filters = base_filters.and_(QueryFilter().gte("created_at", today_start))
            today_memories = self.count(filters=today_filters, session=db_session)
            
            # 计算平均重要性
            query = db_session.query(Memory)
            if session_id:
                query = query.filter(Memory.session_id == session_id)
            query = query.filter(Memory.is_active == True)
            
            avg_importance_result = query.with_entities(func.avg(Memory.importance)).scalar()
            avg_importance = float(avg_importance_result) if avg_importance_result else 0.0
            
            return MemoryStats(
                total_memories=total_memories,
                semantic_memories=semantic_memories,
                episodic_memories=episodic_memories,
                procedural_memories=procedural_memories,
                today_memories=today_memories,
                avg_importance=round(avg_importance, 3)
            )
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return MemoryStats(
                total_memories=0,
                semantic_memories=0,
                episodic_memories=0,
                procedural_memories=0,
                today_memories=0,
                avg_importance=0.0
            )
        finally:
            if not session:
                db_session.close()
    
    def cleanup_low_importance_memories(
        self, 
        session_id: int,
        importance_threshold: float = 0.3,
        days: int = 30,
        session: Optional[Session] = None
    ) -> int:
        """清理低重要性的旧记忆"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # 查找低重要性的旧记忆
            filters = QueryFilter().and_(
                QueryFilter().eq("session_id", session_id),
                QueryFilter().lt("importance", importance_threshold),
                QueryFilter().lt("created_at", cutoff_date),
                QueryFilter().eq("is_active", True)
            )
            
            low_importance_memories = self.get_multi(
                filters=filters,
                limit=1000,  # 批量处理
                session=session
            )
            
            # 停用这些记忆
            deactivated_count = 0
            for memory in low_importance_memories:
                if self.deactivate_memory(memory.id, session):
                    deactivated_count += 1
            
            logger.info(f"Deactivated {deactivated_count} low importance memories")
            return deactivated_count
            
        except Exception as e:
            logger.error(f"Error cleaning up low importance memories: {e}")
            return 0
    
    def consolidate_similar_memories(
        self, 
        session_id: int,
        similarity_threshold: float = 0.9,
        session: Optional[Session] = None
    ) -> int:
        """合并相似记忆"""
        try:
            # 获取所有活跃记忆
            memories = self.get_session_memories(
                session_id=session_id,
                include_inactive=False,
                limit=1000,
                session=session
            )
            
            consolidated_count = 0
            processed_ids = set()
            
            for i, memory1 in enumerate(memories):
                if memory1.id in processed_ids or not memory1.embedding:
                    continue
                
                similar_memories = []
                
                for j, memory2 in enumerate(memories[i+1:], i+1):
                    if memory2.id in processed_ids or not memory2.embedding:
                        continue
                    
                    # 计算相似度
                    similarity = self._calculate_cosine_similarity(
                        memory1.embedding, 
                        memory2.embedding
                    )
                    
                    if similarity >= similarity_threshold:
                        similar_memories.append(memory2)
                
                # 如果找到相似记忆，进行合并
                if similar_memories:
                    # 合并内容和元数据
                    combined_content = memory1.content
                    combined_metadata = memory1.metadata or {}
                    max_importance = memory1.importance
                    
                    for similar_memory in similar_memories:
                        combined_content += f"\n\n{similar_memory.content}"
                        if similar_memory.metadata:
                            combined_metadata.update(similar_memory.metadata)
                        max_importance = max(max_importance, similar_memory.importance)
                        
                        # 停用相似记忆
                        self.deactivate_memory(similar_memory.id, session)
                        processed_ids.add(similar_memory.id)
                    
                    # 更新主记忆
                    self.update(memory1, {
                        "content": combined_content,
                        "metadata": combined_metadata,
                        "importance": max_importance
                    }, session)
                    
                    consolidated_count += len(similar_memories)
                    processed_ids.add(memory1.id)
            
            logger.info(f"Consolidated {consolidated_count} similar memories")
            return consolidated_count
            
        except Exception as e:
            logger.error(f"Error consolidating similar memories: {e}")
            return 0
    
    def get_memory_clusters(
        self, 
        session_id: int,
        num_clusters: int = 5,
        session: Optional[Session] = None
    ) -> Dict[int, List[Memory]]:
        """获取记忆聚类"""
        try:
            # 获取有向量的记忆
            memories = self.get_session_memories(
                session_id=session_id,
                include_inactive=False,
                limit=1000,
                session=session
            )
            
            memories_with_embedding = [
                memory for memory in memories 
                if memory.embedding
            ]
            
            if len(memories_with_embedding) < num_clusters:
                # 如果记忆数量少于聚类数，每个记忆一个聚类
                clusters = {}
                for i, memory in enumerate(memories_with_embedding):
                    clusters[i] = [memory]
                return clusters
            
            # 使用简单的K-means聚类（实际项目中可能需要更复杂的聚类算法）
            import numpy as np
            from sklearn.cluster import KMeans
            
            # 提取向量
            embeddings = np.array([memory.embedding for memory in memories_with_embedding])
            
            # 执行聚类
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # 组织结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(memories_with_embedding[i])
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error getting memory clusters: {e}")
            return {}
    
    def get_memory_timeline(
        self, 
        session_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """获取记忆时间线"""
        try:
            filters = QueryFilter().and_(
                QueryFilter().eq("session_id", session_id),
                QueryFilter().eq("is_active", True)
            )
            
            if start_date:
                filters = filters.and_(QueryFilter().gte("created_at", start_date))
            
            if end_date:
                filters = filters.and_(QueryFilter().lte("created_at", end_date))
            
            memories = self.get_multi(
                filters=filters,
                order_by=[("created_at", "asc")],
                limit=1000,
                session=session
            )
            
            timeline = []
            for memory in memories:
                timeline.append({
                    "id": memory.id,
                    "type": memory.memory_type.value,
                    "content": memory.content[:200] + "..." if len(memory.content) > 200 else memory.content,
                    "summary": memory.summary,
                    "importance": memory.importance,
                    "created_at": memory.created_at.isoformat(),
                    "metadata": memory.metadata
                })
            
            return timeline
            
        except Exception as e:
            logger.error(f"Error getting memory timeline: {e}")
            return []