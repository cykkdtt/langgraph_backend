from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
import json
import hashlib
import gzip
import os
import tempfile
import asyncio
from pathlib import Path

from models.time_travel_models import (
    SnapshotType, SnapshotStatus, RollbackType, RollbackStatus,
    HistoryEventType, StateComponentType, StateComponent, SnapshotMetadata,
    SnapshotCreateRequest, SnapshotUpdateRequest, SnapshotSearchRequest,
    RollbackRequest, HistoryViewRequest, BranchRequest, CompareRequest,
    ExportRequest, SnapshotInfo, RollbackInfo, HistoryEvent, Timeline,
    BranchInfo, TimeTravelStatistics, SnapshotCreateResponse,
    SnapshotUpdateResponse, SnapshotDeleteResponse, SnapshotListResponse,
    RollbackResponse, HistoryViewResponse, BranchCreateResponse,
    BranchListResponse, CompareResponse, ExportResponse
)
from models.api_models import (
    BaseResponse, PaginatedResponse, ErrorDetail, ErrorCode
)
from api.auth import get_current_user
from core.database import get_async_session
from core.logging import get_logger

router = APIRouter(prefix="/time-travel", tags=["时间旅行"])
logger = get_logger(__name__)

# 模拟数据存储（实际应用中应使用数据库）
snapshots_db: Dict[str, SnapshotInfo] = {}
rollbacks_db: Dict[str, RollbackInfo] = {}
events_db: List[HistoryEvent] = []
branches_db: Dict[str, BranchInfo] = {}
export_files: Dict[str, str] = {}

# 辅助函数
def generate_id() -> str:
    """生成唯一ID"""
    import uuid
    return str(uuid.uuid4())

def calculate_checksum(data: Any) -> str:
    """计算数据校验和"""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()

def compress_data(data: bytes) -> bytes:
    """压缩数据"""
    return gzip.compress(data)

def decompress_data(data: bytes) -> bytes:
    """解压数据"""
    return gzip.decompress(data)

def get_snapshot(snapshot_id: str) -> Optional[SnapshotInfo]:
    """获取快照"""
    return snapshots_db.get(snapshot_id)

def save_snapshot(snapshot: SnapshotInfo) -> None:
    """保存快照"""
    snapshots_db[snapshot.id] = snapshot

def delete_snapshot(snapshot_id: str) -> bool:
    """删除快照"""
    if snapshot_id in snapshots_db:
        del snapshots_db[snapshot_id]
        return True
    return False

def get_rollback(rollback_id: str) -> Optional[RollbackInfo]:
    """获取回滚信息"""
    return rollbacks_db.get(rollback_id)

def save_rollback(rollback: RollbackInfo) -> None:
    """保存回滚信息"""
    rollbacks_db[rollback.id] = rollback

def add_history_event(event: HistoryEvent) -> None:
    """添加历史事件"""
    events_db.append(event)

def get_branch(branch_id: str) -> Optional[BranchInfo]:
    """获取分支信息"""
    return branches_db.get(branch_id)

def save_branch(branch: BranchInfo) -> None:
    """保存分支信息"""
    branches_db[branch.id] = branch

def search_snapshots(request: SnapshotSearchRequest) -> List[SnapshotInfo]:
    """搜索快照"""
    results = list(snapshots_db.values())
    
    # 应用过滤条件
    if request.query:
        results = [s for s in results if request.query.lower() in s.name.lower() or 
                  (s.description and request.query.lower() in s.description.lower())]
    
    if request.type:
        results = [s for s in results if s.type == request.type]
    
    if request.status:
        results = [s for s in results if s.status == request.status]
    
    if request.user_id:
        results = [s for s in results if s.metadata.user_id == request.user_id]
    
    if request.session_id:
        results = [s for s in results if s.metadata.session_id == request.session_id]
    
    if request.thread_id:
        results = [s for s in results if s.metadata.thread_id == request.thread_id]
    
    if request.workflow_id:
        results = [s for s in results if s.metadata.workflow_id == request.workflow_id]
    
    if request.agent_id:
        results = [s for s in results if s.metadata.agent_id == request.agent_id]
    
    if request.tags:
        results = [s for s in results if any(tag in s.tags for tag in request.tags)]
    
    if request.created_after:
        results = [s for s in results if s.created_at >= request.created_after]
    
    if request.created_before:
        results = [s for s in results if s.created_at <= request.created_before]
    
    if request.is_milestone is not None:
        results = [s for s in results if s.is_milestone == request.is_milestone]
    
    if request.branch_name:
        results = [s for s in results if s.branch_name == request.branch_name]
    
    if request.has_components:
        results = [s for s in results if any(comp.component_type in request.has_components for comp in s.components)]
    
    if request.min_size:
        results = [s for s in results if s.size_bytes >= request.min_size]
    
    if request.max_size:
        results = [s for s in results if s.size_bytes <= request.max_size]
    
    # 排序
    reverse = request.sort_order == "desc"
    if request.sort_by == "created_at":
        results.sort(key=lambda x: x.created_at, reverse=reverse)
    elif request.sort_by == "name":
        results.sort(key=lambda x: x.name, reverse=reverse)
    elif request.sort_by == "size":
        results.sort(key=lambda x: x.size_bytes, reverse=reverse)
    elif request.sort_by == "access_count":
        results.sort(key=lambda x: x.access_count, reverse=reverse)
    
    # 分页
    start = (request.page - 1) * request.page_size
    end = start + request.page_size
    return results[start:end]

def get_history_events(request: HistoryViewRequest) -> List[HistoryEvent]:
    """获取历史事件"""
    results = [e for e in events_db if e.entity_type == request.entity_type and e.entity_id == request.entity_id]
    
    if request.start_time:
        results = [e for e in results if e.timestamp >= request.start_time]
    
    if request.end_time:
        results = [e for e in results if e.timestamp <= request.end_time]
    
    if request.event_types:
        results = [e for e in results if e.event_type in request.event_types]
    
    # 排序和分页
    results.sort(key=lambda x: x.timestamp, reverse=True)
    start = (request.page - 1) * request.page_size
    end = start + request.page_size
    return results[start:end]

def calculate_statistics() -> TimeTravelStatistics:
    """计算统计信息"""
    snapshots = list(snapshots_db.values())
    rollbacks = list(rollbacks_db.values())
    branches = list(branches_db.values())
    
    total_snapshots = len(snapshots)
    total_rollbacks = len(rollbacks)
    total_branches = len(branches)
    total_events = len(events_db)
    
    # 按类型统计
    by_type = {}
    for snapshot in snapshots:
        by_type[snapshot.type] = by_type.get(snapshot.type, 0) + 1
    
    # 按状态统计
    by_status = {}
    for snapshot in snapshots:
        by_status[snapshot.status] = by_status.get(snapshot.status, 0) + 1
    
    # 存储使用量
    storage_used_mb = sum(s.size_bytes for s in snapshots) / (1024 * 1024)
    average_snapshot_size_mb = storage_used_mb / total_snapshots if total_snapshots > 0 else 0
    
    # 最活跃用户
    user_activity = {}
    for snapshot in snapshots:
        user_id = snapshot.metadata.user_id
        user_activity[user_id] = user_activity.get(user_id, 0) + 1
    most_active_user = max(user_activity.items(), key=lambda x: x[1])[0] if user_activity else None
    
    # 回滚成功率
    successful_rollbacks = len([r for r in rollbacks if r.status == RollbackStatus.COMPLETED])
    rollback_success_rate = successful_rollbacks / total_rollbacks if total_rollbacks > 0 else 0
    
    # 平均回滚时间
    completed_rollbacks = [r for r in rollbacks if r.status == RollbackStatus.COMPLETED and r.duration_seconds]
    average_rollback_time = sum(r.duration_seconds for r in completed_rollbacks) / len(completed_rollbacks) if completed_rollbacks else 0
    
    # 里程碑数量
    milestone_count = len([s for s in snapshots if s.is_milestone])
    
    # 过期快照
    now = datetime.utcnow()
    expired_snapshots = len([s for s in snapshots if s.expires_at and s.expires_at < now])
    
    return TimeTravelStatistics(
        total_snapshots=total_snapshots,
        total_rollbacks=total_rollbacks,
        total_branches=total_branches,
        total_events=total_events,
        by_type=by_type,
        by_status=by_status,
        storage_used_mb=storage_used_mb,
        average_snapshot_size_mb=average_snapshot_size_mb,
        most_active_user=most_active_user,
        rollback_success_rate=rollback_success_rate,
        average_rollback_time_seconds=average_rollback_time,
        milestone_count=milestone_count,
        expired_snapshots=expired_snapshots
    )

async def execute_rollback(rollback: RollbackInfo) -> None:
    """异步执行回滚"""
    try:
        rollback.status = RollbackStatus.IN_PROGRESS
        save_rollback(rollback)
        
        # 模拟回滚过程
        total_steps = len(rollback.components) * 10
        for i in range(total_steps):
            await asyncio.sleep(0.1)  # 模拟处理时间
            rollback.progress = (i + 1) / total_steps
            save_rollback(rollback)
        
        rollback.status = RollbackStatus.COMPLETED
        rollback.completed_at = datetime.utcnow()
        rollback.duration_seconds = (rollback.completed_at - rollback.started_at).total_seconds()
        save_rollback(rollback)
        
        # 添加历史事件
        event = HistoryEvent(
            id=generate_id(),
            event_type=HistoryEventType.ROLLBACK_COMPLETED,
            entity_type="rollback",
            entity_id=rollback.id,
            snapshot_id=rollback.snapshot_id,
            user_id=rollback.user_id,
            timestamp=datetime.utcnow(),
            data={"rollback_id": rollback.id, "snapshot_id": rollback.snapshot_id},
            metadata={"duration_seconds": rollback.duration_seconds},
            source="time_travel_api"
        )
        add_history_event(event)
        
    except Exception as e:
        rollback.status = RollbackStatus.FAILED
        rollback.error_message = str(e)
        rollback.completed_at = datetime.utcnow()
        save_rollback(rollback)
        logger.error(f"回滚失败: {e}")

def cleanup_export_file(export_id: str, file_path: str) -> None:
    """清理导出文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        if export_id in export_files:
            del export_files[export_id]
        logger.info(f"已清理导出文件: {export_id}")
    except Exception as e:
        logger.error(f"清理导出文件失败: {e}")

# API 路由

@router.get("/", response_model=BaseResponse[TimeTravelStatistics])
async def get_time_travel_status(
    current_user: dict = Depends(get_current_user)
):
    """获取时间旅行状态和统计信息"""
    try:
        statistics = calculate_statistics()
        return BaseResponse.success(
            data=statistics,
            message="获取时间旅行状态成功"
        )
    except Exception as e:
        logger.error(f"获取时间旅行状态失败: {e}")
        return BaseResponse.error(
            message="获取时间旅行状态失败",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )

@router.post("/snapshots", response_model=BaseResponse[SnapshotInfo])
async def create_snapshot(
    request: SnapshotCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """创建快照"""
    try:
        # 生成快照ID
        snapshot_id = generate_id()
        
        # 模拟收集状态组件数据
        components = []
        total_size = 0
        for component_type in request.components:
            component_data = {"type": component_type, "data": f"mock_data_for_{component_type}"}
            component = StateComponent(
                component_type=component_type,
                component_id=generate_id(),
                data=component_data,
                version="1.0.0",
                checksum=calculate_checksum(component_data),
                metadata={"collected_at": datetime.utcnow().isoformat()}
            )
            components.append(component)
            total_size += len(json.dumps(component_data))
        
        # 创建快照
        snapshot = SnapshotInfo(
            id=snapshot_id,
            name=request.name,
            description=request.description,
            type=request.type,
            status=SnapshotStatus.CREATING,
            metadata=request.metadata,
            components=components,
            size_bytes=total_size,
            compressed_size_bytes=int(total_size * 0.7) if request.compress else None,
            checksum=calculate_checksum({"components": [c.dict() for c in components]}),
            version="1.0.0",
            parent_snapshot_id=request.metadata.parent_snapshot_id,
            child_snapshot_ids=[],
            branch_name=request.metadata.branch_name,
            is_milestone=request.metadata.is_milestone,
            tags=request.tags,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=request.auto_cleanup_after) if request.auto_cleanup_after else None,
            access_count=0,
            last_accessed=None,
            rollback_count=0,
            is_encrypted=request.encrypt,
            compression_ratio=0.7 if request.compress else None
        )
        
        # 保存快照
        save_snapshot(snapshot)
        
        # 更新状态为活跃
        snapshot.status = SnapshotStatus.ACTIVE
        save_snapshot(snapshot)
        
        # 添加历史事件
        event = HistoryEvent(
            id=generate_id(),
            event_type=HistoryEventType.SNAPSHOT_CREATED,
            entity_type="snapshot",
            entity_id=snapshot_id,
            snapshot_id=snapshot_id,
            user_id=current_user["id"],
            timestamp=datetime.utcnow(),
            data={"snapshot_name": request.name, "type": request.type},
            metadata={"components_count": len(components), "size_bytes": total_size},
            source="time_travel_api"
        )
        add_history_event(event)
        
        return BaseResponse.success(
            data=snapshot,
            message="快照创建成功"
        )
        
    except Exception as e:
        logger.error(f"创建快照失败: {e}")
        return BaseResponse.error(
            message="创建快照失败",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )

@router.get("/snapshots", response_model=SnapshotListResponse)
async def list_snapshots(
    query: Optional[str] = Query(None, description="搜索查询"),
    type: Optional[SnapshotType] = Query(None, description="快照类型"),
    status: Optional[SnapshotStatus] = Query(None, description="快照状态"),
    user_id: Optional[str] = Query(None, description="用户ID"),
    session_id: Optional[str] = Query(None, description="会话ID"),
    thread_id: Optional[str] = Query(None, description="线程ID"),
    workflow_id: Optional[str] = Query(None, description="工作流ID"),
    agent_id: Optional[str] = Query(None, description="智能体ID"),
    is_milestone: Optional[bool] = Query(None, description="是否为里程碑"),
    branch_name: Optional[str] = Query(None, description="分支名称"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    sort_by: str = Query("created_at", description="排序字段"),
    sort_order: str = Query("desc", description="排序方向"),
    include_statistics: bool = Query(False, description="是否包含统计信息"),
    current_user: dict = Depends(get_current_user)
):
    """获取快照列表"""
    try:
        # 构建搜索请求
        search_request = SnapshotSearchRequest(
            query=query,
            type=type,
            status=status,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            workflow_id=workflow_id,
            agent_id=agent_id,
            is_milestone=is_milestone,
            branch_name=branch_name,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # 搜索快照
        snapshots = search_snapshots(search_request)
        total = len(list(snapshots_db.values()))
        has_next = page * page_size < total
        
        # 获取统计信息
        statistics = calculate_statistics() if include_statistics else None
        
        return SnapshotListResponse(
            snapshots=snapshots,
            total=total,
            page=page,
            page_size=page_size,
            has_next=has_next,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"获取快照列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取快照列表失败: {str(e)}")

@router.get("/snapshots/{snapshot_id}", response_model=SnapshotInfo)
async def get_snapshot_detail(
    snapshot_id: str,
    current_user: dict = Depends(get_current_user)
):
    """获取快照详情"""
    try:
        snapshot = get_snapshot(snapshot_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="快照不存在")
        
        # 更新访问统计
        snapshot.access_count += 1
        snapshot.last_accessed = datetime.utcnow()
        save_snapshot(snapshot)
        
        return snapshot
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取快照详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取快照详情失败: {str(e)}")

@router.put("/snapshots/{snapshot_id}", response_model=SnapshotUpdateResponse)
async def update_snapshot(
    snapshot_id: str,
    request: SnapshotUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """更新快照"""
    try:
        snapshot = get_snapshot(snapshot_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="快照不存在")
        
        # 更新字段
        if request.name is not None:
            snapshot.name = request.name
        if request.description is not None:
            snapshot.description = request.description
        if request.status is not None:
            snapshot.status = request.status
        if request.tags is not None:
            snapshot.tags = request.tags
        if request.metadata is not None:
            for key, value in request.metadata.items():
                setattr(snapshot.metadata, key, value)
        if request.retention_days is not None:
            snapshot.expires_at = datetime.utcnow() + timedelta(days=request.retention_days)
        if request.is_milestone is not None:
            snapshot.is_milestone = request.is_milestone
        
        snapshot.updated_at = datetime.utcnow()
        save_snapshot(snapshot)
        
        return SnapshotUpdateResponse(
            success=True,
            message="快照更新成功",
            snapshot=snapshot
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新快照失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新快照失败: {str(e)}")

@router.delete("/snapshots/{snapshot_id}", response_model=SnapshotDeleteResponse)
async def delete_snapshot_endpoint(
    snapshot_id: str,
    force: bool = Query(False, description="是否强制删除"),
    current_user: dict = Depends(get_current_user)
):
    """删除快照"""
    try:
        snapshot = get_snapshot(snapshot_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="快照不存在")
        
        # 检查是否有子快照
        if snapshot.child_snapshot_ids and not force:
            raise HTTPException(status_code=400, detail="快照有子快照，请先删除子快照或使用强制删除")
        
        # 软删除
        snapshot.status = SnapshotStatus.DELETED
        snapshot.updated_at = datetime.utcnow()
        save_snapshot(snapshot)
        
        # 添加历史事件
        event = HistoryEvent(
            id=generate_id(),
            event_type=HistoryEventType.SNAPSHOT_DELETED,
            entity_type="snapshot",
            entity_id=snapshot_id,
            snapshot_id=snapshot_id,
            user_id=current_user["id"],
            timestamp=datetime.utcnow(),
            data={"snapshot_name": snapshot.name, "force": force},
            source="time_travel_api"
        )
        add_history_event(event)
        
        return SnapshotDeleteResponse(
            success=True,
            message="快照删除成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除快照失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除快照失败: {str(e)}")

@router.post("/rollback", response_model=RollbackResponse)
async def rollback_to_snapshot(
    request: RollbackRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """回滚到快照"""
    try:
        # 检查目标快照是否存在
        target_snapshot = get_snapshot(request.snapshot_id)
        if not target_snapshot:
            raise HTTPException(status_code=404, detail="目标快照不存在")
        
        if target_snapshot.status != SnapshotStatus.ACTIVE:
            raise HTTPException(status_code=400, detail="目标快照不可用")
        
        # 创建备份快照
        backup_snapshot_id = None
        if request.create_backup:
            backup_request = SnapshotCreateRequest(
                type=SnapshotType.BACKUP,
                name=request.backup_name or f"回滚前备份_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                description=f"回滚到快照 {request.snapshot_id} 前的备份",
                metadata=SnapshotMetadata(
                    user_id=current_user["id"],
                    description="自动创建的回滚备份",
                    trigger_event="rollback_backup",
                    context={"target_snapshot_id": request.snapshot_id}
                ),
                components=request.components or list(StateComponentType),
                include_full_state=True,
                compress=True
            )
            # 这里应该调用创建快照的逻辑，简化处理
            backup_snapshot_id = generate_id()
        
        # 创建回滚记录
        rollback_id = generate_id()
        rollback = RollbackInfo(
            id=rollback_id,
            snapshot_id=request.snapshot_id,
            rollback_type=request.rollback_type,
            status=RollbackStatus.PENDING,
            components=request.components or list(StateComponentType),
            backup_snapshot_id=backup_snapshot_id,
            progress=0.0,
            metadata=request.metadata,
            started_at=datetime.utcnow(),
            user_id=current_user["id"],
            dry_run=request.dry_run
        )
        
        save_rollback(rollback)
        
        # 添加历史事件
        event = HistoryEvent(
            id=generate_id(),
            event_type=HistoryEventType.ROLLBACK_INITIATED,
            entity_type="rollback",
            entity_id=rollback_id,
            snapshot_id=request.snapshot_id,
            user_id=current_user["id"],
            timestamp=datetime.utcnow(),
            data={
                "rollback_type": request.rollback_type,
                "target_snapshot_id": request.snapshot_id,
                "backup_snapshot_id": backup_snapshot_id,
                "dry_run": request.dry_run
            },
            source="time_travel_api"
        )
        add_history_event(event)
        
        # 异步执行回滚
        if not request.dry_run:
            background_tasks.add_task(execute_rollback, rollback)
        else:
            # 试运行模式，直接标记为完成
            rollback.status = RollbackStatus.COMPLETED
            rollback.completed_at = datetime.utcnow()
            rollback.duration_seconds = 0.1
            save_rollback(rollback)
        
        # 更新目标快照的回滚计数
        target_snapshot.rollback_count += 1
        save_snapshot(target_snapshot)
        
        return RollbackResponse(
            success=True,
            message="回滚已启动" if not request.dry_run else "试运行完成",
            rollback=rollback,
            backup_snapshot_id=backup_snapshot_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"回滚失败: {e}")
        raise HTTPException(status_code=500, detail=f"回滚失败: {str(e)}")

@router.get("/rollbacks/{rollback_id}", response_model=RollbackInfo)
async def get_rollback_status(
    rollback_id: str,
    current_user: dict = Depends(get_current_user)
):
    """获取回滚状态"""
    try:
        rollback = get_rollback(rollback_id)
        if not rollback:
            raise HTTPException(status_code=404, detail="回滚记录不存在")
        
        return rollback
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取回滚状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取回滚状态失败: {str(e)}")

@router.post("/rollbacks/{rollback_id}/cancel")
async def cancel_rollback(
    rollback_id: str,
    current_user: dict = Depends(get_current_user)
):
    """取消回滚"""
    try:
        rollback = get_rollback(rollback_id)
        if not rollback:
            raise HTTPException(status_code=404, detail="回滚记录不存在")
        
        if rollback.status not in [RollbackStatus.PENDING, RollbackStatus.IN_PROGRESS]:
            raise HTTPException(status_code=400, detail="回滚无法取消")
        
        rollback.status = RollbackStatus.CANCELLED
        rollback.completed_at = datetime.utcnow()
        save_rollback(rollback)
        
        return {"success": True, "message": "回滚已取消"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消回滚失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消回滚失败: {str(e)}")

@router.post("/history", response_model=HistoryViewResponse)
async def view_history(
    request: HistoryViewRequest,
    current_user: dict = Depends(get_current_user)
):
    """查看历史"""
    try:
        # 获取历史事件
        events = get_history_events(request)
        
        # 获取相关快照
        snapshots = []
        if request.include_snapshots:
            snapshot_ids = {e.snapshot_id for e in events if e.snapshot_id}
            snapshots = [get_snapshot(sid) for sid in snapshot_ids if get_snapshot(sid)]
        
        # 构建时间线
        timeline = Timeline(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            events=events,
            snapshots=snapshots,
            start_time=request.start_time or (min(e.timestamp for e in events) if events else datetime.utcnow()),
            end_time=request.end_time or (max(e.timestamp for e in events) if events else datetime.utcnow()),
            total_events=len(events),
            total_snapshots=len(snapshots),
            summary={
                "event_types": list(set(e.event_type for e in events)),
                "date_range": {
                    "start": min(e.timestamp for e in events).isoformat() if events else None,
                    "end": max(e.timestamp for e in events).isoformat() if events else None
                },
                "most_frequent_event": max(set(e.event_type for e in events), key=lambda x: sum(1 for e in events if e.event_type == x)) if events else None
            }
        )
        
        total_events = len([e for e in events_db if e.entity_type == request.entity_type and e.entity_id == request.entity_id])
        has_next = request.page * request.page_size < total_events
        
        return HistoryViewResponse(
            timeline=timeline,
            total_events=total_events,
            page=request.page,
            page_size=request.page_size,
            has_next=has_next
        )
        
    except Exception as e:
        logger.error(f"查看历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"查看历史失败: {str(e)}")

@router.post("/branches", response_model=BranchCreateResponse)
async def create_branch(
    request: BranchRequest,
    current_user: dict = Depends(get_current_user)
):
    """创建分支"""
    try:
        # 检查源快照是否存在
        source_snapshot = get_snapshot(request.source_snapshot_id)
        if not source_snapshot:
            raise HTTPException(status_code=404, detail="源快照不存在")
        
        # 检查分支名称是否已存在
        existing_branch = next((b for b in branches_db.values() if b.name == request.branch_name), None)
        if existing_branch:
            raise HTTPException(status_code=400, detail="分支名称已存在")
        
        # 创建分支
        branch_id = generate_id()
        branch = BranchInfo(
            id=branch_id,
            name=request.branch_name,
            description=request.description,
            source_snapshot_id=request.source_snapshot_id,
            current_snapshot_id=request.source_snapshot_id,
            snapshots=[request.source_snapshot_id],
            is_active=True,
            is_merged=False,
            metadata=request.metadata,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            user_id=current_user["id"]
        )
        
        save_branch(branch)
        
        # 添加历史事件
        event = HistoryEvent(
            id=generate_id(),
            event_type=HistoryEventType.BRANCH_CREATED,
            entity_type="branch",
            entity_id=branch_id,
            snapshot_id=request.source_snapshot_id,
            user_id=current_user["id"],
            timestamp=datetime.utcnow(),
            data={
                "branch_name": request.branch_name,
                "source_snapshot_id": request.source_snapshot_id
            },
            source="time_travel_api"
        )
        add_history_event(event)
        
        return BranchCreateResponse(
            success=True,
            message="分支创建成功",
            branch=branch
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建分支失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建分支失败: {str(e)}")

@router.get("/branches", response_model=BranchListResponse)
async def list_branches(
    active_only: bool = Query(False, description="仅显示活跃分支"),
    current_user: dict = Depends(get_current_user)
):
    """获取分支列表"""
    try:
        branches = list(branches_db.values())
        
        if active_only:
            branches = [b for b in branches if b.is_active]
        
        # 排序
        branches.sort(key=lambda x: x.created_at, reverse=True)
        
        active_branches = len([b for b in branches if b.is_active])
        merged_branches = len([b for b in branches if b.is_merged])
        
        return BranchListResponse(
            branches=branches,
            total=len(branches),
            active_branches=active_branches,
            merged_branches=merged_branches
        )
        
    except Exception as e:
        logger.error(f"获取分支列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取分支列表失败: {str(e)}")

@router.post("/compare", response_model=CompareResponse)
async def compare_snapshots(
    request: CompareRequest,
    current_user: dict = Depends(get_current_user)
):
    """比较快照"""
    try:
        # 获取快照
        source_snapshot = get_snapshot(request.source_snapshot_id)
        target_snapshot = get_snapshot(request.target_snapshot_id)
        
        if not source_snapshot:
            raise HTTPException(status_code=404, detail="源快照不存在")
        if not target_snapshot:
            raise HTTPException(status_code=404, detail="目标快照不存在")
        
        # 比较组件
        differences = []
        source_components = {c.component_type: c for c in source_snapshot.components}
        target_components = {c.component_type: c for c in target_snapshot.components}
        
        # 检查要比较的组件
        components_to_compare = request.components or list(source_components.keys())
        
        for component_type in components_to_compare:
            source_comp = source_components.get(component_type)
            target_comp = target_components.get(component_type)
            
            if not source_comp and target_comp:
                differences.append({
                    "type": "added",
                    "component_type": component_type,
                    "description": f"组件 {component_type} 在目标快照中新增"
                })
            elif source_comp and not target_comp:
                differences.append({
                    "type": "removed",
                    "component_type": component_type,
                    "description": f"组件 {component_type} 在目标快照中被移除"
                })
            elif source_comp and target_comp:
                if source_comp.checksum != target_comp.checksum:
                    differences.append({
                        "type": "modified",
                        "component_type": component_type,
                        "description": f"组件 {component_type} 在两个快照间有差异",
                        "source_version": source_comp.version,
                        "target_version": target_comp.version
                    })
        
        # 计算相似度
        total_components = len(set(list(source_components.keys()) + list(target_components.keys())))
        unchanged_components = total_components - len(differences)
        similarity_score = unchanged_components / total_components if total_components > 0 else 1.0
        
        # 生成摘要
        summary = {
            "total_differences": len(differences),
            "added_components": len([d for d in differences if d["type"] == "added"]),
            "removed_components": len([d for d in differences if d["type"] == "removed"]),
            "modified_components": len([d for d in differences if d["type"] == "modified"]),
            "unchanged_components": unchanged_components,
            "similarity_percentage": round(similarity_score * 100, 2)
        }
        
        return CompareResponse(
            source_snapshot=source_snapshot,
            target_snapshot=target_snapshot,
            differences=differences,
            summary=summary,
            similarity_score=similarity_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"比较快照失败: {e}")
        raise HTTPException(status_code=500, detail=f"比较快照失败: {str(e)}")

@router.post("/export", response_model=ExportResponse)
async def export_snapshots(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """导出快照"""
    try:
        # 获取要导出的快照
        snapshots_to_export = []
        if request.snapshot_ids:
            for snapshot_id in request.snapshot_ids:
                snapshot = get_snapshot(snapshot_id)
                if snapshot:
                    snapshots_to_export.append(snapshot)
        else:
            # 根据日期范围导出
            snapshots_to_export = list(snapshots_db.values())
            if request.date_range:
                start_date = request.date_range.get("start")
                end_date = request.date_range.get("end")
                if start_date:
                    snapshots_to_export = [s for s in snapshots_to_export if s.created_at >= start_date]
                if end_date:
                    snapshots_to_export = [s for s in snapshots_to_export if s.created_at <= end_date]
        
        if not snapshots_to_export:
            raise HTTPException(status_code=400, detail="没有找到要导出的快照")
        
        # 生成导出ID
        export_id = generate_id()
        
        # 准备导出数据
        export_data = {
            "export_id": export_id,
            "exported_at": datetime.utcnow().isoformat(),
            "exported_by": current_user["id"],
            "format": request.format,
            "snapshot_count": len(snapshots_to_export),
            "snapshots": []
        }
        
        for snapshot in snapshots_to_export:
            snapshot_data = snapshot.dict()
            if not request.include_data:
                # 移除组件数据
                for component in snapshot_data.get("components", []):
                    component["data"] = "<data_excluded>"
            export_data["snapshots"].append(snapshot_data)
        
        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        file_extension = "json" if request.format == "json" else "txt"
        file_name = f"snapshots_export_{export_id}.{file_extension}"
        file_path = os.path.join(temp_dir, file_name)
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            if request.format == "json":
                json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            else:
                # 简单文本格式
                f.write(f"快照导出报告\n")
                f.write(f"导出时间: {export_data['exported_at']}\n")
                f.write(f"导出用户: {export_data['exported_by']}\n")
                f.write(f"快照数量: {export_data['snapshot_count']}\n\n")
                for snapshot in snapshots_to_export:
                    f.write(f"快照ID: {snapshot.id}\n")
                    f.write(f"名称: {snapshot.name}\n")
                    f.write(f"类型: {snapshot.type}\n")
                    f.write(f"状态: {snapshot.status}\n")
                    f.write(f"创建时间: {snapshot.created_at}\n")
                    f.write(f"大小: {snapshot.size_bytes} 字节\n")
                    f.write("\n" + "-"*50 + "\n\n")
        
        # 压缩文件
        if request.compress:
            compressed_path = file_path + ".gz"
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(file_path)
            file_path = compressed_path
            file_name = file_name + ".gz"
        
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        
        # 保存导出信息
        export_files[export_id] = file_path
        
        # 设置24小时后清理文件
        cleanup_time = datetime.utcnow() + timedelta(hours=24)
        background_tasks.add_task(cleanup_export_file, export_id, file_path)
        
        return ExportResponse(
            success=True,
            message="导出成功",
            export_id=export_id,
            download_url=f"/api/v1/time-travel/export/{export_id}/download",
            file_size=file_size,
            snapshot_count=len(snapshots_to_export),
            expires_at=cleanup_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出快照失败: {e}")
        raise HTTPException(status_code=500, detail=f"导出快照失败: {str(e)}")

@router.get("/export/{export_id}/download")
async def download_export(
    export_id: str,
    current_user: dict = Depends(get_current_user)
):
    """下载导出文件"""
    try:
        if export_id not in export_files:
            raise HTTPException(status_code=404, detail="导出文件不存在或已过期")
        
        file_path = export_files[export_id]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="导出文件不存在")
        
        file_name = os.path.basename(file_path)
        return FileResponse(
            path=file_path,
            filename=file_name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载导出文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载导出文件失败: {str(e)}")

@router.get("/statistics", response_model=TimeTravelStatistics)
async def get_statistics(
    current_user: dict = Depends(get_current_user)
):
    """获取统计信息"""
    try:
        return calculate_statistics()
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")