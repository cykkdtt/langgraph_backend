from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import json
import asyncio
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, desc, asc, func

from models.memory_models import (
    MemoryCreateRequest, MemoryUpdateRequest, MemorySearchRequest,
    MemorySimilarityRequest, MemoryConsolidationRequest, MemoryExportRequest,
    MemoryAnalysisRequest, MemoryBatchOperation, MemoryInfo, MemoryStatistics,
    MemorySimilarityResult, MemoryAnalysisData, MemoryCreateResponse,
    MemoryUpdateResponse, MemoryDeleteResponse, MemoryListResponse,
    MemorySimilarityResponse, MemoryConsolidationResponse, MemoryExportResponse,
    MemoryAnalysisResponse, MemoryBatchOperationResponse, MemoryStatisticsResponse,
    MemoryType, MemoryStatus, MemoryImportance, AccessPattern, AssociationType,
    MemoryMetadata, MemoryVector
)
from models.database_models import Memory as DBMemory
from models.api_models import (
    BaseResponse, PaginatedResponse, ErrorDetail, ErrorCode
)
from models.auth_models import UserInfo
from core.database import get_async_session
from api.auth import get_current_user
# from core.memory.engine import MemoryEngine
# from core.memory.vector_store import VectorStore
from core.logging import get_logger

router = APIRouter(prefix="/memory", tags=["memory"])

# 导出文件存储目录
EXPORT_DIR = Path("exports/memory")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# 辅助函数
def get_memory_by_id(db: AsyncSession, memory_id: str, user_id: str) -> Optional[DBMemory]:
    """根据ID获取记忆"""
    return db.query(DBMemory).filter(
        and_(
            DBMemory.id == memory_id,
            DBMemory.user_id == user_id,
            DBMemory.status != MemoryStatus.DELETED
        )
    ).first()

def convert_db_memory_to_info(db_memory: DBMemory) -> MemoryInfo:
    """将数据库记忆转换为记忆信息"""
    # 解析元数据
    metadata_dict = db_memory.metadata or {}
    metadata = MemoryMetadata(
        source=metadata_dict.get("source", "unknown"),
        session_id=metadata_dict.get("session_id"),
        thread_id=metadata_dict.get("thread_id"),
        agent_id=metadata_dict.get("agent_id"),
        user_id=db_memory.user_id,
        tags=metadata_dict.get("tags", []),
        context=metadata_dict.get("context", {}),
        location=metadata_dict.get("location"),
        emotion=metadata_dict.get("emotion"),
        confidence=metadata_dict.get("confidence", 1.0)
    )
    
    # 解析向量
    vector = None
    if db_memory.vector_embedding:
        vector = MemoryVector(
            embedding=db_memory.vector_embedding,
            dimension=len(db_memory.vector_embedding),
            model=metadata_dict.get("vector_model", "unknown"),
            version=metadata_dict.get("vector_version", "1.0")
        )
    
    # 解析关联（简化版，实际应该从关联表查询）
    associations = []
    
    # 确定访问模式
    access_pattern = AccessPattern.NEVER
    if db_memory.access_count > 100:
        access_pattern = AccessPattern.FREQUENT
    elif db_memory.access_count > 10:
        access_pattern = AccessPattern.OCCASIONAL
    elif db_memory.access_count > 0:
        access_pattern = AccessPattern.RARE
    
    return MemoryInfo(
        id=db_memory.id,
        type=MemoryType(db_memory.type),
        content=db_memory.content,
        title=db_memory.title,
        summary=db_memory.summary,
        importance=MemoryImportance(db_memory.importance),
        status=MemoryStatus(db_memory.status),
        metadata=metadata,
        vector=vector,
        associations=associations,
        access_count=db_memory.access_count or 0,
        access_pattern=access_pattern,
        created_at=db_memory.created_at,
        updated_at=db_memory.updated_at,
        last_accessed=db_memory.last_accessed,
        expiry_date=db_memory.expiry_date,
        consolidation_count=db_memory.consolidation_count or 0,
        decay_factor=db_memory.decay_factor or 1.0,
        retrieval_strength=db_memory.retrieval_strength or 1.0
    )

def calculate_memory_statistics(db: AsyncSession, user_id: str) -> MemoryStatistics:
    """计算记忆统计信息"""
    # 基础统计
    total_memories = db.query(DBMemory).filter(
        and_(
            DBMemory.user_id == user_id,
            DBMemory.status != MemoryStatus.DELETED
        )
    ).count()
    
    # 按类型统计
    type_stats = db.query(
        DBMemory.type,
        func.count(DBMemory.id)
    ).filter(
        and_(
            DBMemory.user_id == user_id,
            DBMemory.status != MemoryStatus.DELETED
        )
    ).group_by(DBMemory.type).all()
    
    by_type = {memory_type: count for memory_type, count in type_stats}
    
    # 按状态统计
    status_stats = db.query(
        DBMemory.status,
        func.count(DBMemory.id)
    ).filter(
        DBMemory.user_id == user_id
    ).group_by(DBMemory.status).all()
    
    by_status = {status: count for status, count in status_stats}
    
    # 按重要性统计
    importance_stats = db.query(
        DBMemory.importance,
        func.count(DBMemory.id)
    ).filter(
        and_(
            DBMemory.user_id == user_id,
            DBMemory.status != MemoryStatus.DELETED
        )
    ).group_by(DBMemory.importance).all()
    
    by_importance = {importance: count for importance, count in importance_stats}
    
    # 平均访问次数
    avg_access = db.query(
        func.avg(DBMemory.access_count)
    ).filter(
        and_(
            DBMemory.user_id == user_id,
            DBMemory.status != MemoryStatus.DELETED
        )
    ).scalar() or 0.0
    
    # 最常访问的记忆
    most_accessed = db.query(DBMemory.id).filter(
        and_(
            DBMemory.user_id == user_id,
            DBMemory.status != MemoryStatus.DELETED
        )
    ).order_by(desc(DBMemory.access_count)).first()
    
    # 最近记忆数（24小时内）
    recent_cutoff = datetime.utcnow() - timedelta(hours=24)
    recent_memories = db.query(DBMemory).filter(
        and_(
            DBMemory.user_id == user_id,
            DBMemory.status != MemoryStatus.DELETED,
            DBMemory.created_at >= recent_cutoff
        )
    ).count()
    
    # 已巩固记忆数
    consolidated_memories = db.query(DBMemory).filter(
        and_(
            DBMemory.user_id == user_id,
            DBMemory.status == MemoryStatus.CONSOLIDATED
        )
    ).count()
    
    # 已过期记忆数
    now = datetime.utcnow()
    expired_memories = db.query(DBMemory).filter(
        and_(
            DBMemory.user_id == user_id,
            DBMemory.expiry_date.isnot(None),
            DBMemory.expiry_date < now
        )
    ).count()
    
    return MemoryStatistics(
        total_memories=total_memories,
        by_type=by_type,
        by_status=by_status,
        by_importance=by_importance,
        average_access_count=float(avg_access),
        most_accessed_memory=most_accessed[0] if most_accessed else None,
        recent_memories=recent_memories,
        consolidated_memories=consolidated_memories,
        expired_memories=expired_memories,
        memory_size_mb=0.0,  # 需要根据实际内容计算
        association_count=0,  # 需要从关联表查询
        average_importance=0.0  # 需要计算
    )

def update_memory_access(db: AsyncSession, memory_id: str):
    """更新记忆访问信息"""
    memory = db.query(DBMemory).filter(DBMemory.id == memory_id).first()
    if memory:
        memory.access_count = (memory.access_count or 0) + 1
        memory.last_accessed = datetime.utcnow()
        db.commit()

async def cleanup_export_file(file_path: str):
    """清理导出文件"""
    await asyncio.sleep(24 * 3600)  # 24小时后清理
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

def export_memories_to_json(memories: List[DBMemory], include_vectors: bool = False) -> str:
    """导出记忆为JSON格式"""
    export_data = []
    for memory in memories:
        data = {
            "id": memory.id,
            "type": memory.type,
            "content": memory.content,
            "title": memory.title,
            "summary": memory.summary,
            "importance": memory.importance,
            "status": memory.status,
            "metadata": memory.metadata,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
            "expiry_date": memory.expiry_date.isoformat() if memory.expiry_date else None,
            "access_count": memory.access_count,
            "consolidation_count": memory.consolidation_count,
            "decay_factor": memory.decay_factor,
            "retrieval_strength": memory.retrieval_strength
        }
        
        if include_vectors and memory.vector_embedding:
            data["vector_embedding"] = memory.vector_embedding
        
        export_data.append(data)
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def export_memories_to_csv(memories: List[DBMemory]) -> str:
    """导出记忆为CSV格式"""
    output = []
    fieldnames = [
        "id", "type", "title", "content", "summary", "importance", "status",
        "created_at", "updated_at", "last_accessed", "access_count"
    ]
    
    # 写入CSV头
    output.append(",".join(fieldnames))
    
    # 写入数据
    for memory in memories:
        row = [
            memory.id,
            memory.type,
            memory.title or "",
            memory.content.replace('"', '""') if memory.content else "",
            memory.summary or "",
            memory.importance,
            memory.status,
            memory.created_at.isoformat(),
            memory.updated_at.isoformat(),
            memory.last_accessed.isoformat() if memory.last_accessed else "",
            str(memory.access_count or 0)
        ]
        output.append(','.join(f'"{field}"' for field in row))
    
    return "\n".join(output)

# API路由
@router.post("/", response_model=BaseResponse[MemoryInfo])
async def create_memory(
    request: MemoryCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """创建新记忆"""
    try:
        # 验证记忆类型和内容
        if not request.content.strip():
            return BaseResponse.error(
                message="记忆内容不能为空",
                errors=[ErrorDetail(
                    code=ErrorCode.VALIDATION_ERROR,
                    message="记忆内容不能为空"
                )]
            )
        
        # 创建记忆对象
        memory_data = {
            "id": str(uuid.uuid4()),
            "user_id": current_user.id,
            "type": request.type,
            "content": request.content,
            "metadata": request.metadata or {},
            "importance": request.importance,
            "status": MemoryStatus.ACTIVE,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "access_count": 0,
            "last_accessed_at": None
        }
        
        # 保存到数据库
        db_memory = DBMemory(**memory_data)
        db.add(db_memory)
        db.commit()
        db.refresh(db_memory)
        
        # 异步处理向量化和关联分析
        background_tasks.add_task(
            process_memory_embedding,
            memory_data["id"],
            request.content,
            request.type
        )
        
        memory_info = MemoryInfo(
            id=db_memory.id,
            user_id=db_memory.user_id,
            type=db_memory.type,
            content=db_memory.content,
            metadata=db_memory.metadata,
            importance=db_memory.importance,
            status=db_memory.status,
            created_at=db_memory.created_at,
            updated_at=db_memory.updated_at,
            access_count=db_memory.access_count,
            last_accessed_at=db_memory.last_accessed_at,
            associations=[],
            similarity_score=None
        )
        
        return BaseResponse.success(
            data=memory_info,
            message="记忆创建成功"
        )
        
    except Exception as e:
        logger.error(f"创建记忆失败: {e}")
        db.rollback()
        return BaseResponse.error(
            message="创建记忆失败",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )

@router.get("/", response_model=PaginatedResponse[MemoryInfo])
async def get_memories(
    type: Optional[MemoryType] = Query(None, description="记忆类型过滤"),
    importance: Optional[MemoryImportance] = Query(None, description="重要性过滤"),
    status: Optional[MemoryStatus] = Query(None, description="状态过滤"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    tags: Optional[str] = Query(None, description="标签过滤，逗号分隔"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    sort_by: Optional[str] = Query("created_at", description="排序字段"),
    sort_order: Optional[str] = Query("desc", description="排序方向"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """获取记忆列表"""
    try:
        # 构建查询条件
        from sqlalchemy import select
        query = select(DBMemory).filter(DBMemory.user_id == current_user.id)
        
        # 应用过滤条件
        if type:
            query = query.filter(DBMemory.type == type.value)
        
        if importance:
            query = query.filter(DBMemory.importance == importance.value)
        
        if status:
            query = query.filter(DBMemory.status == status.value)
        
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    DBMemory.content.ilike(search_pattern),
                    DBMemory.title.ilike(search_pattern),
                    DBMemory.summary.ilike(search_pattern)
                )
            )
        
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            for tag in tag_list:
                query = query.filter(DBMemory.metadata["tags"].astext.contains(tag))
        
        if start_date:
            query = query.filter(DBMemory.created_at >= start_date)
        
        if end_date:
            query = query.filter(DBMemory.created_at <= end_date)
        
        # 排序
        if sort_order.lower() == "desc":
            query = query.order_by(desc(getattr(DBMemory, sort_by)))
        else:
            query = query.order_by(asc(getattr(DBMemory, sort_by)))
        
        # 分页
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        result = await db.execute(query)
        memories = result.scalars().all()
        
        # 转换为响应模型
        memory_list = [convert_db_memory_to_info(memory) for memory in memories]
        
        return PaginatedResponse.success(
            data=memory_list,
            pagination=PaginationInfo(
                page=page,
                page_size=page_size,
                total=total,
                total_pages=(total + page_size - 1) // page_size
            ),
            message="获取记忆列表成功"
        )
        
    except Exception as e:
        logger.error(f"获取记忆列表失败: {e}")
        return PaginatedResponse.error(
            message="获取记忆列表失败",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )

@router.get("/{memory_id}", response_model=MemoryInfo)
async def get_memory_detail(
    memory_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """获取记忆详情"""
    memory = get_memory_by_id(db, memory_id, current_user.id)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    # 更新访问信息
    update_memory_access(db, memory_id)
    
    return convert_db_memory_to_info(memory)

@router.put("/{memory_id}", response_model=MemoryUpdateResponse)
async def update_memory(
    memory_id: str,
    request: MemoryUpdateRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """更新记忆"""
    memory = get_memory_by_id(db, memory_id, current_user.id)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    try:
        # 更新字段
        if request.content is not None:
            memory.content = request.content
        
        if request.title is not None:
            memory.title = request.title
        
        if request.summary is not None:
            memory.summary = request.summary
        
        if request.importance is not None:
            memory.importance = request.importance.value
        
        if request.status is not None:
            memory.status = request.status.value
        
        if request.metadata is not None:
            metadata_dict = {
                "source": request.metadata.source,
                "session_id": request.metadata.session_id,
                "thread_id": request.metadata.thread_id,
                "agent_id": request.metadata.agent_id,
                "tags": request.metadata.tags,
                "context": request.metadata.context,
                "location": request.metadata.location,
                "emotion": request.metadata.emotion,
                "confidence": request.metadata.confidence
            }
            memory.metadata = metadata_dict
        
        if request.vector is not None:
            memory.vector_embedding = request.vector.embedding
            if memory.metadata:
                memory.metadata["vector_model"] = request.vector.model
                memory.metadata["vector_version"] = request.vector.version
        
        if request.expiry_date is not None:
            memory.expiry_date = request.expiry_date
        
        memory.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(memory)
        
        memory_info = convert_db_memory_to_info(memory)
        
        return MemoryUpdateResponse(
            success=True,
            message="记忆更新成功",
            memory=memory_info
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"更新记忆失败: {str(e)}")

@router.delete("/{memory_id}", response_model=MemoryDeleteResponse)
async def delete_memory(
    memory_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """删除记忆（软删除）"""
    memory = get_memory_by_id(db, memory_id, current_user.id)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    try:
        memory.status = MemoryStatus.DELETED.value
        memory.updated_at = datetime.utcnow()
        
        db.commit()
        
        return MemoryDeleteResponse(
            success=True,
            message="记忆删除成功"
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除记忆失败: {str(e)}")

@router.post("/similarity", response_model=MemorySimilarityResponse)
async def search_similar_memories(
    request: MemorySimilarityRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """相似性搜索"""
    try:
        # 这里需要实现向量相似性搜索
        # 由于示例中没有具体的向量数据库，这里返回模拟结果
        
        query = db.query(DBMemory).filter(
            and_(
                DBMemory.user_id == current_user.id,
                DBMemory.status != MemoryStatus.DELETED.value,
                DBMemory.vector_embedding.isnot(None)
            )
        )
        
        if request.memory_type:
            query = query.filter(DBMemory.type == request.memory_type.value)
        
        memories = query.limit(request.top_k).all()
        
        # 模拟相似性计算
        results = []
        for memory in memories:
            similarity_score = 0.8  # 模拟相似度分数
            if similarity_score >= request.threshold:
                result = MemorySimilarityResult(
                    memory=convert_db_memory_to_info(memory),
                    similarity_score=similarity_score,
                    distance=1.0 - similarity_score,
                    relevance_score=similarity_score
                )
                results.append(result)
        
        return MemorySimilarityResponse(
            results=results,
            total=len(results),
            query_time_ms=50.0,
            threshold_used=request.threshold
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"相似性搜索失败: {str(e)}")

@router.post("/consolidate", response_model=MemoryConsolidationResponse)
async def consolidate_memories(
    request: MemoryConsolidationRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """记忆巩固"""
    try:
        # 获取要巩固的记忆
        memories = db.query(DBMemory).filter(
            and_(
                DBMemory.id.in_(request.memory_ids),
                DBMemory.user_id == current_user.id,
                DBMemory.status != MemoryStatus.DELETED.value
            )
        ).all()
        
        if not memories:
            raise HTTPException(status_code=404, detail="未找到要巩固的记忆")
        
        # 创建巩固后的记忆
        consolidated_content = "\n\n".join([m.content for m in memories])
        consolidated_title = f"巩固记忆 - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        
        consolidated_memory = DBMemory(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            type=MemoryType.LONG_TERM.value,
            content=consolidated_content,
            title=consolidated_title,
            summary=f"由{len(memories)}个记忆巩固而成",
            importance=MemoryImportance.HIGH.value,
            status=MemoryStatus.CONSOLIDATED.value,
            metadata=request.metadata,
            consolidation_count=1,
            decay_factor=1.0,
            retrieval_strength=1.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(consolidated_memory)
        
        # 更新原始记忆状态
        original_ids = []
        for memory in memories:
            original_ids.append(memory.id)
            if not request.preserve_original:
                memory.status = MemoryStatus.ARCHIVED.value
            memory.consolidation_count = (memory.consolidation_count or 0) + 1
        
        db.commit()
        db.refresh(consolidated_memory)
        
        return MemoryConsolidationResponse(
            success=True,
            message="记忆巩固成功",
            consolidated_memory=convert_db_memory_to_info(consolidated_memory),
            merged_count=len(memories),
            original_ids=original_ids
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"记忆巩固失败: {str(e)}")

@router.post("/export", response_model=MemoryExportResponse)
async def export_memories(
    request: MemoryExportRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """导出记忆"""
    try:
        # 构建查询
        query = db.query(DBMemory).filter(
            and_(
                DBMemory.user_id == current_user.id,
                DBMemory.status != MemoryStatus.DELETED.value
            )
        )
        
        if request.memory_ids:
            query = query.filter(DBMemory.id.in_(request.memory_ids))
        
        if request.memory_type:
            query = query.filter(DBMemory.type == request.memory_type.value)
        
        if request.date_range:
            if "start" in request.date_range:
                query = query.filter(DBMemory.created_at >= request.date_range["start"])
            if "end" in request.date_range:
                query = query.filter(DBMemory.created_at <= request.date_range["end"])
        
        memories = query.all()
        
        if not memories:
            raise HTTPException(status_code=404, detail="没有找到要导出的记忆")
        
        # 生成导出文件
        export_id = str(uuid.uuid4())
        file_extension = "json" if request.format == "json" else "csv"
        filename = f"memory_export_{export_id}.{file_extension}"
        file_path = EXPORT_DIR / filename
        
        if request.format == "json":
            content = export_memories_to_json(memories, request.include_vectors)
        else:
            content = export_memories_to_csv(memories)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        file_size = os.path.getsize(file_path)
        
        # 设置清理任务
        background_tasks.add_task(cleanup_export_file, str(file_path))
        
        return MemoryExportResponse(
            success=True,
            message="导出成功",
            export_id=export_id,
            download_url=f"/api/v1/memory/export/{export_id}/download",
            file_size=file_size,
            memory_count=len(memories),
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")

@router.get("/export/{export_id}/download")
async def download_export(
    export_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """下载导出文件"""
    # 查找文件
    json_file = EXPORT_DIR / f"memory_export_{export_id}.json"
    csv_file = EXPORT_DIR / f"memory_export_{export_id}.csv"
    
    if json_file.exists():
        return FileResponse(
            path=str(json_file),
            filename=f"memory_export_{export_id}.json",
            media_type="application/json"
        )
    elif csv_file.exists():
        return FileResponse(
            path=str(csv_file),
            filename=f"memory_export_{export_id}.csv",
            media_type="text/csv"
        )
    else:
        raise HTTPException(status_code=404, detail="导出文件不存在或已过期")

@router.post("/analytics", response_model=MemoryAnalysisResponse)
async def get_memory_analytics(
    request: MemoryAnalysisRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """获取记忆分析"""
    try:
        # 基础查询
        query = db.query(DBMemory).filter(
            and_(
                DBMemory.user_id == current_user.id,
                DBMemory.status != MemoryStatus.DELETED.value
            )
        )
        
        if request.time_range:
            if "start" in request.time_range:
                query = query.filter(DBMemory.created_at >= request.time_range["start"])
            if "end" in request.time_range:
                query = query.filter(DBMemory.created_at <= request.time_range["end"])
        
        if request.memory_types:
            type_values = [t.value for t in request.memory_types]
            query = query.filter(DBMemory.type.in_(type_values))
        
        memories = query.all()
        
        # 分析数据
        analysis_data = MemoryAnalysisData(
            memory_growth={"total": len(memories)},
            access_patterns={"average_access": sum(m.access_count or 0 for m in memories) / len(memories) if memories else 0},
            importance_distribution={imp.value: len([m for m in memories if m.importance == imp.value]) for imp in MemoryImportance},
            type_distribution={t.value: len([m for m in memories if m.type == t.value]) for t in MemoryType},
            consolidation_trends={"consolidated_count": len([m for m in memories if m.status == MemoryStatus.CONSOLIDATED.value])},
            decay_analysis={"average_decay": sum(m.decay_factor or 1.0 for m in memories) / len(memories) if memories else 1.0},
            association_network={"total_associations": 0},
            retrieval_efficiency={"average_strength": sum(m.retrieval_strength or 1.0 for m in memories) / len(memories) if memories else 1.0}
        )
        
        # 生成洞察和建议
        insights = [
            f"您总共有 {len(memories)} 个记忆",
            f"平均访问次数为 {analysis_data.access_patterns['average_access']:.1f}",
            f"记忆衰减因子平均值为 {analysis_data.decay_analysis['average_decay']:.2f}"
        ]
        
        recommendations = [
            "建议定期巩固重要记忆以提高检索效率",
            "可以考虑为低访问频率的记忆添加更多标签",
            "建议定期清理过期或不重要的记忆"
        ]
        
        return MemoryAnalysisResponse(
            success=True,
            message="分析完成",
            analysis_data=analysis_data,
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

@router.post("/batch", response_model=MemoryBatchOperationResponse)
async def batch_memory_operations(
    request: MemoryBatchOperation,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """批量操作记忆"""
    try:
        results = []
        failed_ids = []
        
        for memory_id in request.memory_ids:
            try:
                memory = get_memory_by_id(db, memory_id, current_user.id)
                if not memory:
                    failed_ids.append(memory_id)
                    continue
                
                if request.operation == "delete":
                    memory.status = MemoryStatus.DELETED.value
                    memory.updated_at = datetime.utcnow()
                    results.append({"id": memory_id, "operation": "delete", "success": True})
                
                elif request.operation == "archive":
                    memory.status = MemoryStatus.ARCHIVED.value
                    memory.updated_at = datetime.utcnow()
                    results.append({"id": memory_id, "operation": "archive", "success": True})
                
                elif request.operation == "update_importance":
                    importance = request.params.get("importance")
                    if importance:
                        memory.importance = importance
                        memory.updated_at = datetime.utcnow()
                        results.append({"id": memory_id, "operation": "update_importance", "success": True})
                    else:
                        failed_ids.append(memory_id)
                
                else:
                    failed_ids.append(memory_id)
                    
            except Exception:
                failed_ids.append(memory_id)
        
        db.commit()
        
        return MemoryBatchOperationResponse(
            success=len(failed_ids) == 0,
            message=f"批量操作完成，成功 {len(results)} 个，失败 {len(failed_ids)} 个",
            results=results,
            failed_ids=failed_ids
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量操作失败: {str(e)}")

@router.get("/statistics", response_model=MemoryStatistics)
async def get_memory_statistics(
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """获取记忆统计信息"""
    return calculate_memory_statistics(db, current_user.id)