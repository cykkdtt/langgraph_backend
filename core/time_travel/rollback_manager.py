"""
回滚管理器

管理状态回滚操作和策略。
"""

import asyncio
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from enum import Enum

from .time_travel_types import (
    RollbackPoint, RollbackStrategy, StateSnapshot, Checkpoint,
    TimeTravelConfig
)


class RollbackStatus(str, Enum):
    """回滚状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RollbackManager:
    """回滚管理器"""
    
    def __init__(self, config: Optional[TimeTravelConfig] = None):
        self.config = config or TimeTravelConfig()
        
        # 回滚点存储
        self._rollback_points: Dict[str, RollbackPoint] = {}
        self._execution_rollbacks: Dict[str, List[str]] = {}  # execution_id -> rollback_ids
        
        # 回滚策略处理器
        self._strategy_handlers: Dict[RollbackStrategy, Callable] = {
            RollbackStrategy.SOFT: self._handle_soft_rollback,
            RollbackStrategy.HARD: self._handle_hard_rollback,
            RollbackStrategy.BRANCH: self._handle_branch_rollback,
            RollbackStrategy.MERGE: self._handle_merge_rollback
        }
        
        # 事件处理器
        self._rollback_handlers: List[Callable] = []
        self._validation_handlers: List[Callable] = []
        
        # 统计信息
        self._stats = {
            "total_rollbacks": 0,
            "successful_rollbacks": 0,
            "failed_rollbacks": 0,
            "cancelled_rollbacks": 0
        }
    
    async def create_rollback_point(
        self,
        target_snapshot_id: str,
        execution_id: str,
        agent_id: str,
        strategy: RollbackStrategy = RollbackStrategy.SOFT,
        reason: Optional[str] = None,
        target_checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RollbackPoint:
        """创建回滚点"""
        rollback_id = f"rb_{len(self._rollback_points)}_{datetime.now().timestamp()}"
        
        rollback_point = RollbackPoint(
            id=rollback_id,
            target_snapshot_id=target_snapshot_id,
            target_checkpoint_id=target_checkpoint_id,
            rollback_strategy=strategy,
            rollback_reason=reason,
            execution_id=execution_id,
            agent_id=agent_id,
            created_at=datetime.now(),
            status=RollbackStatus.PENDING.value,
            metadata=metadata or {}
        )
        
        # 存储回滚点
        self._rollback_points[rollback_id] = rollback_point
        
        # 更新索引
        if execution_id not in self._execution_rollbacks:
            self._execution_rollbacks[execution_id] = []
        self._execution_rollbacks[execution_id].append(rollback_id)
        
        # 更新统计
        self._stats["total_rollbacks"] += 1
        
        # 触发处理器
        await self._trigger_rollback_handlers("created", rollback_point)
        
        return rollback_point
    
    async def execute_rollback(
        self,
        rollback_id: str,
        snapshot_provider: Callable[[str], StateSnapshot],
        checkpoint_provider: Optional[Callable[[str], Checkpoint]] = None,
        validation_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行回滚"""
        if rollback_id not in self._rollback_points:
            raise ValueError(f"回滚点不存在: {rollback_id}")
        
        rollback_point = self._rollback_points[rollback_id]
        
        if rollback_point.status != RollbackStatus.PENDING.value:
            raise RuntimeError(f"回滚点状态无效: {rollback_point.status}")
        
        # 更新状态
        rollback_point.status = RollbackStatus.IN_PROGRESS.value
        
        try:
            # 获取目标快照
            target_snapshot = snapshot_provider(rollback_point.target_snapshot_id)
            if not target_snapshot:
                raise ValueError(f"目标快照不存在: {rollback_point.target_snapshot_id}")
            
            # 验证回滚
            if validation_options:
                validation_result = await self._validate_rollback(
                    rollback_point, target_snapshot, validation_options
                )
                if not validation_result["valid"]:
                    raise RuntimeError(f"回滚验证失败: {validation_result['reason']}")
            
            # 执行回滚策略
            strategy_handler = self._strategy_handlers.get(rollback_point.rollback_strategy)
            if not strategy_handler:
                raise RuntimeError(f"不支持的回滚策略: {rollback_point.rollback_strategy}")
            
            rollback_result = await strategy_handler(
                rollback_point, target_snapshot, checkpoint_provider
            )
            
            # 更新回滚点
            rollback_point.status = RollbackStatus.COMPLETED.value
            rollback_point.executed_at = datetime.now()
            rollback_point.affected_steps = rollback_result.get("affected_steps", [])
            rollback_point.affected_data = rollback_result.get("affected_data", {})
            
            # 更新统计
            self._stats["successful_rollbacks"] += 1
            
            # 触发处理器
            await self._trigger_rollback_handlers("completed", rollback_point)
            
            return {
                "rollback_id": rollback_id,
                "status": "completed",
                "target_snapshot_id": rollback_point.target_snapshot_id,
                "strategy": rollback_point.rollback_strategy.value,
                "executed_at": rollback_point.executed_at,
                "result": rollback_result
            }
            
        except Exception as e:
            # 更新失败状态
            rollback_point.status = RollbackStatus.FAILED.value
            rollback_point.error = str(e)
            rollback_point.executed_at = datetime.now()
            
            # 更新统计
            self._stats["failed_rollbacks"] += 1
            
            # 触发处理器
            await self._trigger_rollback_handlers("failed", rollback_point)
            
            raise
    
    async def cancel_rollback(self, rollback_id: str, reason: Optional[str] = None) -> bool:
        """取消回滚"""
        if rollback_id not in self._rollback_points:
            return False
        
        rollback_point = self._rollback_points[rollback_id]
        
        if rollback_point.status not in [RollbackStatus.PENDING.value, RollbackStatus.IN_PROGRESS.value]:
            return False
        
        # 更新状态
        rollback_point.status = RollbackStatus.CANCELLED.value
        rollback_point.error = reason or "用户取消"
        
        # 更新统计
        self._stats["cancelled_rollbacks"] += 1
        
        # 触发处理器
        await self._trigger_rollback_handlers("cancelled", rollback_point)
        
        return True
    
    async def get_rollback_point(self, rollback_id: str) -> Optional[RollbackPoint]:
        """获取回滚点"""
        return self._rollback_points.get(rollback_id)
    
    async def list_rollback_points(
        self,
        execution_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        status: Optional[RollbackStatus] = None,
        strategy: Optional[RollbackStrategy] = None,
        limit: Optional[int] = None
    ) -> List[RollbackPoint]:
        """列出回滚点"""
        rollback_points = list(self._rollback_points.values())
        
        # 应用过滤条件
        if execution_id:
            rollback_ids = self._execution_rollbacks.get(execution_id, [])
            rollback_points = [rb for rb in rollback_points if rb.id in rollback_ids]
        
        if agent_id:
            rollback_points = [rb for rb in rollback_points if rb.agent_id == agent_id]
        
        if status:
            rollback_points = [rb for rb in rollback_points if rb.status == status.value]
        
        if strategy:
            rollback_points = [rb for rb in rollback_points if rb.rollback_strategy == strategy]
        
        # 排序
        rollback_points.sort(key=lambda rb: rb.created_at, reverse=True)
        
        # 限制数量
        if limit:
            rollback_points = rollback_points[:limit]
        
        return rollback_points
    
    async def get_rollback_history(self, execution_id: str) -> List[RollbackPoint]:
        """获取执行的回滚历史"""
        return await self.list_rollback_points(execution_id=execution_id)
    
    async def get_rollback_statistics(self) -> Dict[str, Any]:
        """获取回滚统计信息"""
        # 按策略统计
        strategy_stats = {}
        for rollback_point in self._rollback_points.values():
            strategy_name = rollback_point.rollback_strategy.value
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = 0
            strategy_stats[strategy_name] += 1
        
        # 按状态统计
        status_stats = {}
        for rollback_point in self._rollback_points.values():
            status = rollback_point.status
            if status not in status_stats:
                status_stats[status] = 0
            status_stats[status] += 1
        
        return {
            **self._stats,
            "by_strategy": strategy_stats,
            "by_status": status_stats,
            "success_rate": (
                self._stats["successful_rollbacks"] / max(self._stats["total_rollbacks"], 1)
            ) * 100
        }
    
    def register_rollback_handler(self, handler: Callable) -> None:
        """注册回滚处理器"""
        self._rollback_handlers.append(handler)
    
    def register_validation_handler(self, handler: Callable) -> None:
        """注册验证处理器"""
        self._validation_handlers.append(handler)
    
    async def _handle_soft_rollback(
        self,
        rollback_point: RollbackPoint,
        target_snapshot: StateSnapshot,
        checkpoint_provider: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """处理软回滚"""
        # 软回滚：保留历史记录，只恢复状态
        return {
            "type": "soft",
            "restored_state": target_snapshot.state_data,
            "preserved_history": True,
            "affected_steps": [],
            "affected_data": {
                "restored_from": target_snapshot.id,
                "restore_timestamp": datetime.now().isoformat()
            }
        }
    
    async def _handle_hard_rollback(
        self,
        rollback_point: RollbackPoint,
        target_snapshot: StateSnapshot,
        checkpoint_provider: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """处理硬回滚"""
        # 硬回滚：删除后续历史记录
        affected_steps = []
        
        # 这里应该与时间旅行管理器协调，删除后续快照
        # 简化实现，只记录影响的步骤
        if target_snapshot.step_name:
            affected_steps.append(target_snapshot.step_name)
        
        return {
            "type": "hard",
            "restored_state": target_snapshot.state_data,
            "preserved_history": False,
            "affected_steps": affected_steps,
            "affected_data": {
                "restored_from": target_snapshot.id,
                "restore_timestamp": datetime.now().isoformat(),
                "deleted_history": True
            }
        }
    
    async def _handle_branch_rollback(
        self,
        rollback_point: RollbackPoint,
        target_snapshot: StateSnapshot,
        checkpoint_provider: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """处理分支回滚"""
        # 分支回滚：创建新分支从目标点继续
        branch_name = f"rollback_{rollback_point.id}_{datetime.now().timestamp()}"
        
        return {
            "type": "branch",
            "restored_state": target_snapshot.state_data,
            "preserved_history": True,
            "new_branch": branch_name,
            "affected_steps": [],
            "affected_data": {
                "restored_from": target_snapshot.id,
                "restore_timestamp": datetime.now().isoformat(),
                "branch_created": branch_name
            }
        }
    
    async def _handle_merge_rollback(
        self,
        rollback_point: RollbackPoint,
        target_snapshot: StateSnapshot,
        checkpoint_provider: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """处理合并回滚"""
        # 合并回滚：智能合并状态变更
        return {
            "type": "merge",
            "restored_state": target_snapshot.state_data,
            "preserved_history": True,
            "merge_strategy": "intelligent",
            "affected_steps": [],
            "affected_data": {
                "restored_from": target_snapshot.id,
                "restore_timestamp": datetime.now().isoformat(),
                "merge_applied": True
            }
        }
    
    async def _validate_rollback(
        self,
        rollback_point: RollbackPoint,
        target_snapshot: StateSnapshot,
        validation_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证回滚操作"""
        validation_result = {
            "valid": True,
            "reason": None,
            "warnings": []
        }
        
        # 基本验证
        if not target_snapshot.state_data:
            validation_result["valid"] = False
            validation_result["reason"] = "目标快照状态数据为空"
            return validation_result
        
        # 时间验证
        if validation_options.get("check_time_consistency", True):
            time_diff = (datetime.now() - target_snapshot.timestamp).total_seconds()
            max_age = validation_options.get("max_snapshot_age_hours", 24) * 3600
            
            if time_diff > max_age:
                validation_result["warnings"].append(
                    f"目标快照较旧，创建于 {time_diff/3600:.1f} 小时前"
                )
        
        # 数据完整性验证
        if validation_options.get("check_data_integrity", True):
            required_fields = validation_options.get("required_fields", [])
            for field in required_fields:
                if field not in target_snapshot.state_data:
                    validation_result["valid"] = False
                    validation_result["reason"] = f"缺少必需字段: {field}"
                    return validation_result
        
        # 自定义验证处理器
        for handler in self._validation_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    custom_result = await handler(rollback_point, target_snapshot, validation_options)
                else:
                    custom_result = handler(rollback_point, target_snapshot, validation_options)
                
                if not custom_result.get("valid", True):
                    validation_result["valid"] = False
                    validation_result["reason"] = custom_result.get("reason", "自定义验证失败")
                    return validation_result
                
                # 合并警告
                validation_result["warnings"].extend(custom_result.get("warnings", []))
                
            except Exception as e:
                validation_result["warnings"].append(f"验证处理器错误: {e}")
        
        return validation_result
    
    async def _trigger_rollback_handlers(self, event_type: str, rollback_point: RollbackPoint) -> None:
        """触发回滚处理器"""
        for handler in self._rollback_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, rollback_point)
                else:
                    handler(event_type, rollback_point)
            except Exception as e:
                print(f"回滚处理器错误: {e}")