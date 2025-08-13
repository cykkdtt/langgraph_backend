"""
时间旅行管理器

提供时间旅行功能的核心管理器，支持状态快照、回滚和分支管理。
"""

import asyncio
import json
import gzip
import hashlib
from typing import Optional, List, Dict, Any, Callable, Set
from datetime import datetime, timedelta
from pathlib import Path

from .time_travel_types import (
    StateSnapshot, Checkpoint, RollbackPoint, TimeTravelConfig,
    HistoryQuery, BranchInfo, MergeRequest, StateVersion,
    SnapshotType, CheckpointType, RollbackStrategy
)


class TimeTravelManager:
    """时间旅行管理器"""
    
    def __init__(self, config: Optional[TimeTravelConfig] = None):
        self.config = config or TimeTravelConfig()
        
        # 存储
        self._snapshots: Dict[str, StateSnapshot] = {}
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._rollback_points: Dict[str, RollbackPoint] = {}
        self._branches: Dict[str, BranchInfo] = {}
        self._merge_requests: Dict[str, MergeRequest] = {}
        
        # 版本管理
        self._current_version: int = 0
        self._current_branch: str = "main"
        self._version_tree: Dict[str, List[StateVersion]] = {"main": []}
        
        # 事件处理器
        self._snapshot_handlers: List[Callable] = []
        self._checkpoint_handlers: List[Callable] = []
        self._rollback_handlers: List[Callable] = []
        
        # 状态
        self._is_recording: bool = True
        self._last_snapshot_step: int = 0
        
        # 初始化主分支
        self._branches["main"] = BranchInfo(
            name="main",
            description="主分支",
            created_at=datetime.now()
        )
    
    async def create_snapshot(
        self,
        state_data: Dict[str, Any],
        snapshot_type: SnapshotType = SnapshotType.MANUAL,
        execution_id: Optional[str] = None,
        step_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> StateSnapshot:
        """创建状态快照"""
        if not self._is_recording:
            raise RuntimeError("时间旅行记录已暂停")
        
        # 生成版本信息
        self._current_version += 1
        version = StateVersion(
            version_id=f"v{self._current_version}",
            branch_name=self._current_branch,
            version_number=self._current_version,
            created_at=datetime.now(),
            description=description
        )
        
        # 计算变更和差异
        changes = None
        diff_from_parent = None
        if self._snapshots:
            # 获取最新快照
            latest_snapshot = max(
                self._snapshots.values(),
                key=lambda s: s.timestamp
            )
            changes = self._calculate_changes(latest_snapshot.state_data, state_data)
            diff_from_parent = self._calculate_diff(latest_snapshot.state_data, state_data)
        
        # 创建快照
        snapshot_id = f"snap_{self._current_version}_{datetime.now().timestamp()}"
        snapshot = StateSnapshot(
            id=snapshot_id,
            version=version,
            type=snapshot_type,
            state_data=state_data,
            context_data=context_data,
            execution_id=execution_id,
            step_name=step_name,
            agent_id=agent_id,
            timestamp=datetime.now(),
            changes=changes,
            diff_from_parent=diff_from_parent
        )
        
        # 压缩和存储
        if self.config.compression_enabled:
            snapshot = await self._compress_snapshot(snapshot)
        
        # 存储快照
        self._snapshots[snapshot_id] = snapshot
        self._version_tree[self._current_branch].append(version)
        
        # 更新分支信息
        branch = self._branches[self._current_branch]
        branch.snapshot_count += 1
        branch.last_activity = datetime.now()
        
        # 触发处理器
        await self._trigger_snapshot_handlers(snapshot)
        
        # 自动清理
        if self.config.auto_cleanup:
            await self._cleanup_old_snapshots()
        
        return snapshot
    
    async def create_checkpoint(
        self,
        name: str,
        checkpoint_type: CheckpointType,
        snapshot_id: str,
        execution_id: str,
        agent_id: str,
        description: Optional[str] = None,
        step_name: Optional[str] = None,
        restore_config: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """创建检查点"""
        if snapshot_id not in self._snapshots:
            raise ValueError(f"快照不存在: {snapshot_id}")
        
        checkpoint_id = f"cp_{len(self._checkpoints)}_{datetime.now().timestamp()}"
        checkpoint = Checkpoint(
            id=checkpoint_id,
            name=name,
            type=checkpoint_type,
            description=description,
            snapshot_id=snapshot_id,
            execution_id=execution_id,
            step_name=step_name,
            agent_id=agent_id,
            created_at=datetime.now(),
            restore_config=restore_config
        )
        
        self._checkpoints[checkpoint_id] = checkpoint
        
        # 更新分支信息
        branch = self._branches[self._current_branch]
        branch.checkpoint_count += 1
        branch.last_activity = datetime.now()
        
        # 触发处理器
        await self._trigger_checkpoint_handlers(checkpoint)
        
        return checkpoint
    
    async def rollback_to_snapshot(
        self,
        snapshot_id: str,
        strategy: RollbackStrategy = RollbackStrategy.SOFT,
        reason: Optional[str] = None
    ) -> RollbackPoint:
        """回滚到指定快照"""
        if snapshot_id not in self._snapshots:
            raise ValueError(f"快照不存在: {snapshot_id}")
        
        snapshot = self._snapshots[snapshot_id]
        
        # 创建回滚点
        rollback_id = f"rb_{len(self._rollback_points)}_{datetime.now().timestamp()}"
        rollback_point = RollbackPoint(
            id=rollback_id,
            target_snapshot_id=snapshot_id,
            rollback_strategy=strategy,
            rollback_reason=reason,
            execution_id=snapshot.execution_id or "unknown",
            agent_id=snapshot.agent_id or "unknown",
            created_at=datetime.now()
        )
        
        # 执行回滚
        try:
            await self._execute_rollback(rollback_point, snapshot)
            rollback_point.status = "completed"
            rollback_point.executed_at = datetime.now()
        except Exception as e:
            rollback_point.status = "failed"
            rollback_point.error = str(e)
            raise
        finally:
            self._rollback_points[rollback_id] = rollback_point
        
        # 触发处理器
        await self._trigger_rollback_handlers(rollback_point)
        
        return rollback_point
    
    async def rollback_to_checkpoint(
        self,
        checkpoint_id: str,
        strategy: RollbackStrategy = RollbackStrategy.SOFT,
        reason: Optional[str] = None
    ) -> RollbackPoint:
        """回滚到指定检查点"""
        if checkpoint_id not in self._checkpoints:
            raise ValueError(f"检查点不存在: {checkpoint_id}")
        
        checkpoint = self._checkpoints[checkpoint_id]
        return await self.rollback_to_snapshot(
            checkpoint.snapshot_id,
            strategy,
            reason
        )
    
    async def create_branch(
        self,
        branch_name: str,
        source_snapshot_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> BranchInfo:
        """创建分支"""
        if not self.config.enable_branching:
            raise RuntimeError("分支功能未启用")
        
        if branch_name in self._branches:
            raise ValueError(f"分支已存在: {branch_name}")
        
        if len(self._branches) >= self.config.max_branches:
            raise RuntimeError(f"分支数量已达上限: {self.config.max_branches}")
        
        # 创建分支信息
        branch = BranchInfo(
            name=branch_name,
            description=description,
            parent_branch=self._current_branch,
            created_at=datetime.now()
        )
        
        self._branches[branch_name] = branch
        self._version_tree[branch_name] = []
        
        # 如果指定了源快照，复制版本历史
        if source_snapshot_id and source_snapshot_id in self._snapshots:
            source_snapshot = self._snapshots[source_snapshot_id]
            # 复制到分支点的版本历史
            for version in self._version_tree[self._current_branch]:
                if version.version_number <= source_snapshot.version.version_number:
                    self._version_tree[branch_name].append(version)
        
        return branch
    
    async def switch_branch(self, branch_name: str) -> None:
        """切换分支"""
        if branch_name not in self._branches:
            raise ValueError(f"分支不存在: {branch_name}")
        
        self._current_branch = branch_name
        
        # 更新当前版本号
        if self._version_tree[branch_name]:
            self._current_version = max(
                v.version_number for v in self._version_tree[branch_name]
            )
        else:
            self._current_version = 0
    
    async def merge_branch(
        self,
        source_branch: str,
        target_branch: str,
        merge_strategy: str = "auto"
    ) -> MergeRequest:
        """合并分支"""
        if source_branch not in self._branches:
            raise ValueError(f"源分支不存在: {source_branch}")
        
        if target_branch not in self._branches:
            raise ValueError(f"目标分支不存在: {target_branch}")
        
        # 创建合并请求
        merge_id = f"merge_{len(self._merge_requests)}_{datetime.now().timestamp()}"
        merge_request = MergeRequest(
            id=merge_id,
            source_branch=source_branch,
            target_branch=target_branch,
            title=f"合并 {source_branch} 到 {target_branch}",
            merge_strategy=merge_strategy,
            created_at=datetime.now()
        )
        
        # 检测冲突
        conflicts = await self._detect_conflicts(source_branch, target_branch)
        merge_request.conflicts = conflicts
        
        if not conflicts and merge_strategy == "auto":
            # 自动合并
            await self._execute_merge(merge_request)
            merge_request.status = "merged"
        else:
            merge_request.status = "pending"
        
        self._merge_requests[merge_id] = merge_request
        return merge_request
    
    async def query_history(self, query: HistoryQuery) -> List[StateSnapshot]:
        """查询历史记录"""
        snapshots = list(self._snapshots.values())
        
        # 应用过滤条件
        if query.execution_id:
            snapshots = [s for s in snapshots if s.execution_id == query.execution_id]
        
        if query.agent_id:
            snapshots = [s for s in snapshots if s.agent_id == query.agent_id]
        
        if query.step_name:
            snapshots = [s for s in snapshots if s.step_name == query.step_name]
        
        if query.start_time:
            snapshots = [s for s in snapshots if s.timestamp >= query.start_time]
        
        if query.end_time:
            snapshots = [s for s in snapshots if s.timestamp <= query.end_time]
        
        if query.start_version:
            snapshots = [s for s in snapshots if s.version.version_number >= query.start_version]
        
        if query.end_version:
            snapshots = [s for s in snapshots if s.version.version_number <= query.end_version]
        
        if query.snapshot_types:
            snapshots = [s for s in snapshots if s.type in query.snapshot_types]
        
        if query.branch_name:
            snapshots = [s for s in snapshots if s.version.branch_name == query.branch_name]
        elif not query.include_branches:
            snapshots = [s for s in snapshots if s.version.branch_name == self._current_branch]
        
        # 排序
        reverse = query.sort_order == "desc"
        if query.sort_by == "timestamp":
            snapshots.sort(key=lambda s: s.timestamp, reverse=reverse)
        elif query.sort_by == "version":
            snapshots.sort(key=lambda s: s.version.version_number, reverse=reverse)
        
        # 分页
        if query.offset:
            snapshots = snapshots[query.offset:]
        
        if query.limit:
            snapshots = snapshots[:query.limit]
        
        # 处理数据包含选项
        if not query.include_state_data:
            for snapshot in snapshots:
                snapshot.state_data = {}
        
        if not query.include_context_data:
            for snapshot in snapshots:
                snapshot.context_data = None
        
        if not query.include_changes:
            for snapshot in snapshots:
                snapshot.changes = None
                snapshot.diff_from_parent = None
        
        return snapshots
    
    def get_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """获取快照"""
        return self._snapshots.get(snapshot_id)
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """获取检查点"""
        return self._checkpoints.get(checkpoint_id)
    
    def get_current_branch(self) -> str:
        """获取当前分支"""
        return self._current_branch
    
    def get_branches(self) -> List[BranchInfo]:
        """获取所有分支"""
        return list(self._branches.values())
    
    def get_current_version(self) -> int:
        """获取当前版本号"""
        return self._current_version
    
    def is_recording(self) -> bool:
        """是否正在记录"""
        return self._is_recording
    
    def pause_recording(self) -> None:
        """暂停记录"""
        self._is_recording = False
    
    def resume_recording(self) -> None:
        """恢复记录"""
        self._is_recording = True
    
    def register_snapshot_handler(self, handler: Callable) -> None:
        """注册快照处理器"""
        self._snapshot_handlers.append(handler)
    
    def register_checkpoint_handler(self, handler: Callable) -> None:
        """注册检查点处理器"""
        self._checkpoint_handlers.append(handler)
    
    def register_rollback_handler(self, handler: Callable) -> None:
        """注册回滚处理器"""
        self._rollback_handlers.append(handler)
    
    async def _compress_snapshot(self, snapshot: StateSnapshot) -> StateSnapshot:
        """压缩快照"""
        if self.config.compression_algorithm == "gzip":
            # 压缩状态数据
            state_json = json.dumps(snapshot.state_data).encode('utf-8')
            compressed_data = gzip.compress(state_json)
            
            # 计算压缩信息
            snapshot.size_bytes = len(compressed_data)
            snapshot.compression = "gzip"
            snapshot.checksum = hashlib.md5(compressed_data).hexdigest()
        
        return snapshot
    
    def _calculate_changes(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算变更"""
        changes = {
            "added": {},
            "modified": {},
            "removed": {}
        }
        
        # 检查新增和修改
        for key, value in new_data.items():
            if key not in old_data:
                changes["added"][key] = value
            elif old_data[key] != value:
                changes["modified"][key] = {
                    "old": old_data[key],
                    "new": value
                }
        
        # 检查删除
        for key in old_data:
            if key not in new_data:
                changes["removed"][key] = old_data[key]
        
        return changes
    
    def _calculate_diff(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算差异"""
        # 简化的差异计算
        return {
            "old_keys": set(old_data.keys()),
            "new_keys": set(new_data.keys()),
            "changed_keys": [
                key for key in old_data
                if key in new_data and old_data[key] != new_data[key]
            ]
        }
    
    async def _execute_rollback(self, rollback_point: RollbackPoint, snapshot: StateSnapshot) -> None:
        """执行回滚"""
        if rollback_point.rollback_strategy == RollbackStrategy.HARD:
            # 硬回滚：删除后续快照
            snapshots_to_remove = [
                sid for sid, s in self._snapshots.items()
                if s.version.version_number > snapshot.version.version_number
                and s.version.branch_name == snapshot.version.branch_name
            ]
            
            for sid in snapshots_to_remove:
                del self._snapshots[sid]
            
            # 更新版本号
            self._current_version = snapshot.version.version_number
        
        elif rollback_point.rollback_strategy == RollbackStrategy.BRANCH:
            # 分支回滚：创建新分支
            branch_name = f"rollback_{datetime.now().timestamp()}"
            await self.create_branch(branch_name, snapshot.id)
            await self.switch_branch(branch_name)
    
    async def _detect_conflicts(self, source_branch: str, target_branch: str) -> List[Dict[str, Any]]:
        """检测合并冲突"""
        conflicts = []
        
        # 获取两个分支的最新快照
        source_snapshots = [
            s for s in self._snapshots.values()
            if s.version.branch_name == source_branch
        ]
        target_snapshots = [
            s for s in self._snapshots.values()
            if s.version.branch_name == target_branch
        ]
        
        if not source_snapshots or not target_snapshots:
            return conflicts
        
        source_latest = max(source_snapshots, key=lambda s: s.version.version_number)
        target_latest = max(target_snapshots, key=lambda s: s.version.version_number)
        
        # 检查数据冲突
        for key in source_latest.state_data:
            if (key in target_latest.state_data and
                source_latest.state_data[key] != target_latest.state_data[key]):
                conflicts.append({
                    "type": "data_conflict",
                    "key": key,
                    "source_value": source_latest.state_data[key],
                    "target_value": target_latest.state_data[key]
                })
        
        return conflicts
    
    async def _execute_merge(self, merge_request: MergeRequest) -> None:
        """执行合并"""
        # 简化的合并实现
        source_branch = merge_request.source_branch
        target_branch = merge_request.target_branch
        
        # 将源分支的版本历史合并到目标分支
        source_versions = self._version_tree[source_branch]
        target_versions = self._version_tree[target_branch]
        
        # 更新目标分支
        self._version_tree[target_branch].extend(source_versions)
        
        # 标记源分支为已合并
        self._branches[source_branch].is_merged = True
        self._branches[source_branch].merge_target = target_branch
    
    async def _cleanup_old_snapshots(self) -> None:
        """清理旧快照"""
        if not self.config.auto_cleanup:
            return
        
        cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
        
        # 找到需要删除的快照
        snapshots_to_remove = [
            sid for sid, snapshot in self._snapshots.items()
            if snapshot.timestamp < cutoff_time
        ]
        
        # 保留最近的快照
        if len(self._snapshots) - len(snapshots_to_remove) < 10:
            snapshots_to_remove = snapshots_to_remove[:-10]
        
        # 删除快照
        for sid in snapshots_to_remove:
            del self._snapshots[sid]
        
        # 清理相关检查点
        checkpoints_to_remove = [
            cid for cid, checkpoint in self._checkpoints.items()
            if checkpoint.snapshot_id in snapshots_to_remove
        ]
        
        for cid in checkpoints_to_remove:
            del self._checkpoints[cid]
    
    async def _trigger_snapshot_handlers(self, snapshot: StateSnapshot) -> None:
        """触发快照处理器"""
        for handler in self._snapshot_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(snapshot)
                else:
                    handler(snapshot)
            except Exception as e:
                # 记录错误但不中断流程
                print(f"快照处理器错误: {e}")
    
    async def _trigger_checkpoint_handlers(self, checkpoint: Checkpoint) -> None:
        """触发检查点处理器"""
        for handler in self._checkpoint_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(checkpoint)
                else:
                    handler(checkpoint)
            except Exception as e:
                print(f"检查点处理器错误: {e}")
    
    async def _trigger_rollback_handlers(self, rollback_point: RollbackPoint) -> None:
        """触发回滚处理器"""
        for handler in self._rollback_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(rollback_point)
                else:
                    handler(rollback_point)
            except Exception as e:
                print(f"回滚处理器错误: {e}")