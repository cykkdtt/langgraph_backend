"""
时间旅行API路由

提供时间旅行相关的REST API接口，包括快照管理、检查点管理、
回滚操作、分支管理和执行历史查询等功能。
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

# 导入时间旅行相关类型和管理器
from .time_travel_manager import TimeTravelManager
from .time_travel_types import (
    StateSnapshot, Checkpoint, RollbackPoint, TimeTravelConfig, HistoryQuery,
    SnapshotType, CheckpointType, RollbackStrategy, BranchInfo
)

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/time-travel", tags=["时间旅行"])


# ===== API请求/响应模型 =====

class SnapshotRequest(BaseModel):
    """创建快照请求"""
    thread_id: str = Field(..., description="会话ID")
    snapshot_type: SnapshotType = Field(SnapshotType.MANUAL, description="快照类型")
    description: Optional[str] = Field(None, description="快照描述")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class CheckpointRequest(BaseModel):
    """创建检查点请求"""
    thread_id: str = Field(..., description="会话ID")
    checkpoint_name: str = Field(..., description="检查点名称")
    checkpoint_type: CheckpointType = Field(CheckpointType.USER, description="检查点类型")
    description: Optional[str] = Field(None, description="检查点描述")


class RollbackRequest(BaseModel):
    """回滚请求"""
    thread_id: str = Field(..., description="会话ID")
    target_snapshot_id: Optional[str] = Field(None, description="目标快照ID")
    target_checkpoint_id: Optional[str] = Field(None, description="目标检查点ID")
    rollback_strategy: RollbackStrategy = Field(RollbackStrategy.SOFT, description="回滚策略")
    rollback_reason: Optional[str] = Field(None, description="回滚原因")


class BranchRequest(BaseModel):
    """创建分支请求"""
    thread_id: str = Field(..., description="源会话ID")
    branch_name: str = Field(..., description="分支名称")
    from_snapshot_id: Optional[str] = Field(None, description="分支起点快照ID")
    description: Optional[str] = Field(None, description="分支描述")


# ===== 依赖注入 =====

def get_time_travel_manager() -> TimeTravelManager:
    """获取时间旅行管理器实例"""
    try:
        from core.time_travel import get_time_travel_manager as _get_manager
        return _get_manager()
    except ImportError:
        # 如果没有全局管理器，创建一个新实例
        return TimeTravelManager()


# ===== API端点实现 =====

@router.get("/config", response_model=TimeTravelConfig)
async def get_time_travel_config(
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> TimeTravelConfig:
    """获取时间旅行配置"""
    try:
        config = await time_travel_manager.get_config()
        return config
    except Exception as e:
        logger.error(f"获取时间旅行配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")


@router.put("/config", response_model=TimeTravelConfig)
async def update_time_travel_config(
    config: TimeTravelConfig,
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> TimeTravelConfig:
    """更新时间旅行配置"""
    try:
        updated_config = await time_travel_manager.update_config(config)
        return updated_config
    except Exception as e:
        logger.error(f"更新时间旅行配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


@router.post("/snapshots", response_model=StateSnapshot)
async def create_snapshot(
    request: SnapshotRequest,
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> StateSnapshot:
    """创建快照"""
    try:
        snapshot = await time_travel_manager.create_snapshot(
            thread_id=request.thread_id,
            snapshot_type=request.snapshot_type,
            description=request.description,
            metadata=request.metadata
        )
        return snapshot
    except Exception as e:
        logger.error(f"创建快照失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建快照失败: {str(e)}")


@router.get("/snapshots/{thread_id}", response_model=List[StateSnapshot])
async def list_snapshots(
    thread_id: str,
    snapshot_type: Optional[SnapshotType] = Query(None, description="快照类型过滤"),
    limit: int = Query(50, description="限制数量"),
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> List[StateSnapshot]:
    """获取快照列表"""
    try:
        snapshots = await time_travel_manager.list_snapshots(
            thread_id=thread_id,
            snapshot_type=snapshot_type,
            limit=limit
        )
        return snapshots
    except Exception as e:
        logger.error(f"获取快照列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取快照列表失败: {str(e)}")


@router.get("/snapshots/{thread_id}/{snapshot_id}", response_model=StateSnapshot)
async def get_snapshot(
    thread_id: str,
    snapshot_id: str,
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> StateSnapshot:
    """获取特定快照"""
    try:
        snapshot = await time_travel_manager.get_snapshot(thread_id, snapshot_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="快照不存在")
        return snapshot
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取快照失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取快照失败: {str(e)}")


@router.delete("/snapshots/{thread_id}/{snapshot_id}")
async def delete_snapshot(
    thread_id: str,
    snapshot_id: str,
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> Dict[str, str]:
    """删除快照"""
    try:
        await time_travel_manager.delete_snapshot(thread_id, snapshot_id)
        return {"message": f"快照 {snapshot_id} 已删除"}
    except Exception as e:
        logger.error(f"删除快照失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除快照失败: {str(e)}")


@router.post("/checkpoints", response_model=Checkpoint)
async def create_checkpoint(
    request: CheckpointRequest,
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> Checkpoint:
    """创建检查点"""
    try:
        checkpoint = await time_travel_manager.create_checkpoint(
            thread_id=request.thread_id,
            checkpoint_name=request.checkpoint_name,
            checkpoint_type=request.checkpoint_type,
            description=request.description
        )
        return checkpoint
    except Exception as e:
        logger.error(f"创建检查点失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建检查点失败: {str(e)}")


@router.get("/checkpoints/{thread_id}", response_model=List[Checkpoint])
async def list_checkpoints(
    thread_id: str,
    checkpoint_type: Optional[CheckpointType] = Query(None, description="检查点类型过滤"),
    limit: int = Query(50, description="限制数量"),
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> List[Checkpoint]:
    """获取检查点列表"""
    try:
        checkpoints = await time_travel_manager.list_checkpoints(
            thread_id=thread_id,
            checkpoint_type=checkpoint_type,
            limit=limit
        )
        return checkpoints
    except Exception as e:
        logger.error(f"获取检查点列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取检查点列表失败: {str(e)}")


@router.get("/checkpoints/{thread_id}/{checkpoint_id}", response_model=Checkpoint)
async def get_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> Checkpoint:
    """获取特定检查点"""
    try:
        checkpoint = await time_travel_manager.get_checkpoint(thread_id, checkpoint_id)
        if not checkpoint:
            raise HTTPException(status_code=404, detail="检查点不存在")
        return checkpoint
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取检查点失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取检查点失败: {str(e)}")


@router.delete("/checkpoints/{thread_id}/{checkpoint_id}")
async def delete_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> Dict[str, str]:
    """删除检查点"""
    try:
        await time_travel_manager.delete_checkpoint(thread_id, checkpoint_id)
        return {"message": f"检查点 {checkpoint_id} 已删除"}
    except Exception as e:
        logger.error(f"删除检查点失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除检查点失败: {str(e)}")


@router.post("/rollback", response_model=RollbackPoint)
async def rollback_to_point(
    request: RollbackRequest,
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> RollbackPoint:
    """执行回滚操作"""
    try:
        rollback_point = await time_travel_manager.rollback(
            thread_id=request.thread_id,
            target_snapshot_id=request.target_snapshot_id,
            target_checkpoint_id=request.target_checkpoint_id,
            rollback_strategy=request.rollback_strategy,
            rollback_reason=request.rollback_reason
        )
        return rollback_point
    except Exception as e:
        logger.error(f"回滚操作失败: {e}")
        raise HTTPException(status_code=500, detail=f"回滚操作失败: {str(e)}")


@router.get("/rollback/{thread_id}", response_model=List[RollbackPoint])
async def list_rollback_points(
    thread_id: str,
    limit: int = Query(50, description="限制数量"),
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> List[RollbackPoint]:
    """获取回滚点列表"""
    try:
        rollback_points = await time_travel_manager.list_rollback_points(
            thread_id=thread_id,
            limit=limit
        )
        return rollback_points
    except Exception as e:
        logger.error(f"获取回滚点列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取回滚点列表失败: {str(e)}")


@router.post("/branches", response_model=BranchInfo)
async def create_branch(
    request: BranchRequest,
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> BranchInfo:
    """创建分支"""
    try:
        branch = await time_travel_manager.create_branch(
            thread_id=request.thread_id,
            branch_name=request.branch_name,
            from_snapshot_id=request.from_snapshot_id,
            description=request.description
        )
        return branch
    except Exception as e:
        logger.error(f"创建分支失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建分支失败: {str(e)}")


@router.get("/branches/{thread_id}", response_model=List[BranchInfo])
async def list_branches(
    thread_id: str,
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> List[BranchInfo]:
    """获取分支列表"""
    try:
        branches = await time_travel_manager.list_branches(thread_id)
        return branches
    except Exception as e:
        logger.error(f"获取分支列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取分支列表失败: {str(e)}")


@router.post("/history/query", response_model=List[StateSnapshot])
async def query_history(
    query: HistoryQuery,
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> List[StateSnapshot]:
    """查询执行历史"""
    try:
        history = await time_travel_manager.query_history(query)
        return history
    except Exception as e:
        logger.error(f"查询历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询历史失败: {str(e)}")


@router.get("/history/{thread_id}", response_model=List[StateSnapshot])
async def get_execution_history(
    thread_id: str,
    limit: int = Query(100, description="限制数量"),
    include_state_data: bool = Query(False, description="包含状态数据"),
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> List[StateSnapshot]:
    """获取执行历史"""
    try:
        history = await time_travel_manager.get_execution_history(
            thread_id=thread_id,
            limit=limit,
            include_state_data=include_state_data
        )
        return history
    except Exception as e:
        logger.error(f"获取执行历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取执行历史失败: {str(e)}")


@router.delete("/cleanup/{thread_id}")
async def cleanup_thread_data(
    thread_id: str,
    keep_checkpoints: bool = Query(True, description="保留检查点"),
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> Dict[str, str]:
    """清理会话数据"""
    try:
        await time_travel_manager.cleanup_thread_data(
            thread_id=thread_id,
            keep_checkpoints=keep_checkpoints
        )
        return {"message": f"会话 {thread_id} 的数据已清理"}
    except Exception as e:
        logger.error(f"清理数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理数据失败: {str(e)}")


@router.get("/status")
async def get_time_travel_status(
    time_travel_manager: TimeTravelManager = Depends(get_time_travel_manager)
) -> Dict[str, Any]:
    """获取时间旅行系统状态"""
    try:
        status = await time_travel_manager.get_system_status()
        return status
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")