"""
状态历史管理器

管理执行状态的历史记录和版本控制。
"""

import asyncio
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict

from .time_travel_types import (
    StateSnapshot, StateVersion, HistoryQuery, TimeTravelConfig,
    SnapshotType
)


class StateHistoryManager:
    """状态历史管理器"""
    
    def __init__(self, config: Optional[TimeTravelConfig] = None):
        self.config = config or TimeTravelConfig()
        
        # 历史存储
        self._history: Dict[str, List[StateSnapshot]] = defaultdict(list)  # execution_id -> snapshots
        self._agent_history: Dict[str, List[StateSnapshot]] = defaultdict(list)  # agent_id -> snapshots
        self._step_history: Dict[str, List[StateSnapshot]] = defaultdict(list)  # step_name -> snapshots
        
        # 索引
        self._timestamp_index: List[tuple] = []  # (timestamp, snapshot_id)
        self._version_index: Dict[int, str] = {}  # version_number -> snapshot_id
        
        # 统计信息
        self._stats = {
            "total_snapshots": 0,
            "total_executions": 0,
            "total_agents": 0,
            "storage_size": 0
        }
        
        # 事件处理器
        self._history_handlers: List[Callable] = []
    
    async def add_snapshot(self, snapshot: StateSnapshot) -> None:
        """添加快照到历史记录"""
        # 添加到主历史
        if snapshot.execution_id:
            self._history[snapshot.execution_id].append(snapshot)
        
        # 添加到智能体历史
        if snapshot.agent_id:
            self._agent_history[snapshot.agent_id].append(snapshot)
        
        # 添加到步骤历史
        if snapshot.step_name:
            self._step_history[snapshot.step_name].append(snapshot)
        
        # 更新索引
        self._timestamp_index.append((snapshot.timestamp, snapshot.id))
        self._timestamp_index.sort(key=lambda x: x[0])
        
        if snapshot.version:
            self._version_index[snapshot.version.version_number] = snapshot.id
        
        # 更新统计
        self._stats["total_snapshots"] += 1
        if snapshot.execution_id and len(self._history[snapshot.execution_id]) == 1:
            self._stats["total_executions"] += 1
        if snapshot.agent_id and len(self._agent_history[snapshot.agent_id]) == 1:
            self._stats["total_agents"] += 1
        
        # 触发处理器
        await self._trigger_history_handlers("snapshot_added", snapshot)
        
        # 自动清理
        if self.config.auto_cleanup:
            await self._cleanup_old_history()
    
    async def get_execution_history(
        self,
        execution_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[StateSnapshot]:
        """获取执行历史"""
        snapshots = self._history.get(execution_id, [])
        
        # 时间过滤
        if start_time:
            snapshots = [s for s in snapshots if s.timestamp >= start_time]
        if end_time:
            snapshots = [s for s in snapshots if s.timestamp <= end_time]
        
        # 排序和限制
        snapshots.sort(key=lambda s: s.timestamp)
        if limit:
            snapshots = snapshots[:limit]
        
        return snapshots
    
    async def get_agent_history(
        self,
        agent_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[StateSnapshot]:
        """获取智能体历史"""
        snapshots = self._agent_history.get(agent_id, [])
        
        # 时间过滤
        if start_time:
            snapshots = [s for s in snapshots if s.timestamp >= start_time]
        if end_time:
            snapshots = [s for s in snapshots if s.timestamp <= end_time]
        
        # 排序和限制
        snapshots.sort(key=lambda s: s.timestamp)
        if limit:
            snapshots = snapshots[:limit]
        
        return snapshots
    
    async def get_step_history(
        self,
        step_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[StateSnapshot]:
        """获取步骤历史"""
        snapshots = self._step_history.get(step_name, [])
        
        # 时间过滤
        if start_time:
            snapshots = [s for s in snapshots if s.timestamp >= start_time]
        if end_time:
            snapshots = [s for s in snapshots if s.timestamp <= end_time]
        
        # 排序和限制
        snapshots.sort(key=lambda s: s.timestamp)
        if limit:
            snapshots = snapshots[:limit]
        
        return snapshots
    
    async def search_history(self, query: HistoryQuery) -> List[StateSnapshot]:
        """搜索历史记录"""
        all_snapshots = []
        
        # 收集所有相关快照
        if query.execution_id:
            all_snapshots.extend(self._history.get(query.execution_id, []))
        elif query.agent_id:
            all_snapshots.extend(self._agent_history.get(query.agent_id, []))
        elif query.step_name:
            all_snapshots.extend(self._step_history.get(query.step_name, []))
        else:
            # 收集所有快照
            for snapshots in self._history.values():
                all_snapshots.extend(snapshots)
        
        # 去重
        seen_ids = set()
        unique_snapshots = []
        for snapshot in all_snapshots:
            if snapshot.id not in seen_ids:
                unique_snapshots.append(snapshot)
                seen_ids.add(snapshot.id)
        
        # 应用过滤条件
        filtered_snapshots = []
        for snapshot in unique_snapshots:
            # 时间过滤
            if query.start_time and snapshot.timestamp < query.start_time:
                continue
            if query.end_time and snapshot.timestamp > query.end_time:
                continue
            
            # 版本过滤
            if query.start_version and snapshot.version and snapshot.version.version_number < query.start_version:
                continue
            if query.end_version and snapshot.version and snapshot.version.version_number > query.end_version:
                continue
            
            # 类型过滤
            if query.snapshot_types and snapshot.type not in query.snapshot_types:
                continue
            
            # 分支过滤
            if query.branch_name and snapshot.version and snapshot.version.branch_name != query.branch_name:
                continue
            
            # 其他条件过滤
            if query.execution_id and snapshot.execution_id != query.execution_id:
                continue
            if query.agent_id and snapshot.agent_id != query.agent_id:
                continue
            if query.step_name and snapshot.step_name != query.step_name:
                continue
            
            filtered_snapshots.append(snapshot)
        
        # 排序
        reverse = query.sort_order == "desc"
        if query.sort_by == "timestamp":
            filtered_snapshots.sort(key=lambda s: s.timestamp, reverse=reverse)
        elif query.sort_by == "version":
            filtered_snapshots.sort(
                key=lambda s: s.version.version_number if s.version else 0,
                reverse=reverse
            )
        
        # 分页
        if query.offset:
            filtered_snapshots = filtered_snapshots[query.offset:]
        if query.limit:
            filtered_snapshots = filtered_snapshots[:query.limit]
        
        return filtered_snapshots
    
    async def get_timeline(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        execution_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None
    ) -> List[StateSnapshot]:
        """获取时间线"""
        # 从时间索引中获取快照
        timeline_snapshots = []
        
        for timestamp, snapshot_id in self._timestamp_index:
            # 时间过滤
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                break
            
            # 查找快照
            snapshot = None
            for snapshots in self._history.values():
                for s in snapshots:
                    if s.id == snapshot_id:
                        snapshot = s
                        break
                if snapshot:
                    break
            
            if not snapshot:
                continue
            
            # 执行ID过滤
            if execution_ids and snapshot.execution_id not in execution_ids:
                continue
            
            # 智能体ID过滤
            if agent_ids and snapshot.agent_id not in agent_ids:
                continue
            
            timeline_snapshots.append(snapshot)
        
        return timeline_snapshots
    
    async def get_state_evolution(
        self,
        execution_id: str,
        key_path: str
    ) -> List[Dict[str, Any]]:
        """获取状态演化"""
        snapshots = await self.get_execution_history(execution_id)
        evolution = []
        
        for snapshot in snapshots:
            # 提取指定路径的值
            value = self._extract_value_by_path(snapshot.state_data, key_path)
            evolution.append({
                "timestamp": snapshot.timestamp,
                "version": snapshot.version.version_number if snapshot.version else None,
                "step_name": snapshot.step_name,
                "value": value
            })
        
        return evolution
    
    async def compare_snapshots(
        self,
        snapshot_id1: str,
        snapshot_id2: str
    ) -> Dict[str, Any]:
        """比较快照"""
        snapshot1 = await self._find_snapshot_by_id(snapshot_id1)
        snapshot2 = await self._find_snapshot_by_id(snapshot_id2)
        
        if not snapshot1 or not snapshot2:
            raise ValueError("快照不存在")
        
        # 计算差异
        diff = {
            "snapshot1": {
                "id": snapshot1.id,
                "timestamp": snapshot1.timestamp,
                "version": snapshot1.version.version_number if snapshot1.version else None
            },
            "snapshot2": {
                "id": snapshot2.id,
                "timestamp": snapshot2.timestamp,
                "version": snapshot2.version.version_number if snapshot2.version else None
            },
            "time_diff": (snapshot2.timestamp - snapshot1.timestamp).total_seconds(),
            "state_diff": self._calculate_state_diff(
                snapshot1.state_data,
                snapshot2.state_data
            )
        }
        
        return diff
    
    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 计算存储大小
        total_size = 0
        for snapshots in self._history.values():
            for snapshot in snapshots:
                if snapshot.size_bytes:
                    total_size += snapshot.size_bytes
        
        self._stats["storage_size"] = total_size
        
        # 计算时间范围
        if self._timestamp_index:
            earliest = self._timestamp_index[0][0]
            latest = self._timestamp_index[-1][0]
            time_span = (latest - earliest).total_seconds()
        else:
            earliest = latest = None
            time_span = 0
        
        return {
            **self._stats,
            "earliest_snapshot": earliest,
            "latest_snapshot": latest,
            "time_span_seconds": time_span,
            "average_snapshots_per_execution": (
                self._stats["total_snapshots"] / max(self._stats["total_executions"], 1)
            )
        }
    
    async def cleanup_execution(self, execution_id: str) -> int:
        """清理执行历史"""
        if execution_id not in self._history:
            return 0
        
        snapshots = self._history[execution_id]
        count = len(snapshots)
        
        # 从各个索引中移除
        for snapshot in snapshots:
            # 从时间索引移除
            self._timestamp_index = [
                (ts, sid) for ts, sid in self._timestamp_index
                if sid != snapshot.id
            ]
            
            # 从版本索引移除
            if snapshot.version:
                self._version_index.pop(snapshot.version.version_number, None)
            
            # 从智能体历史移除
            if snapshot.agent_id:
                self._agent_history[snapshot.agent_id] = [
                    s for s in self._agent_history[snapshot.agent_id]
                    if s.id != snapshot.id
                ]
            
            # 从步骤历史移除
            if snapshot.step_name:
                self._step_history[snapshot.step_name] = [
                    s for s in self._step_history[snapshot.step_name]
                    if s.id != snapshot.id
                ]
        
        # 删除执行历史
        del self._history[execution_id]
        
        # 更新统计
        self._stats["total_snapshots"] -= count
        self._stats["total_executions"] -= 1
        
        # 触发处理器
        await self._trigger_history_handlers("execution_cleaned", {"execution_id": execution_id, "count": count})
        
        return count
    
    def register_history_handler(self, handler: Callable) -> None:
        """注册历史处理器"""
        self._history_handlers.append(handler)
    
    def _extract_value_by_path(self, data: Dict[str, Any], path: str) -> Any:
        """根据路径提取值"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _calculate_state_diff(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> Dict[str, Any]:
        """计算状态差异"""
        diff = {
            "added": {},
            "modified": {},
            "removed": {}
        }
        
        # 检查新增和修改
        for key, value in state2.items():
            if key not in state1:
                diff["added"][key] = value
            elif state1[key] != value:
                diff["modified"][key] = {
                    "old": state1[key],
                    "new": value
                }
        
        # 检查删除
        for key in state1:
            if key not in state2:
                diff["removed"][key] = state1[key]
        
        return diff
    
    async def _find_snapshot_by_id(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """根据ID查找快照"""
        for snapshots in self._history.values():
            for snapshot in snapshots:
                if snapshot.id == snapshot_id:
                    return snapshot
        return None
    
    async def _cleanup_old_history(self) -> None:
        """清理旧历史记录"""
        if not self.config.auto_cleanup:
            return
        
        cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
        
        # 清理旧快照
        executions_to_clean = []
        for execution_id, snapshots in self._history.items():
            # 过滤旧快照
            new_snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
            
            if len(new_snapshots) != len(snapshots):
                if new_snapshots:
                    self._history[execution_id] = new_snapshots
                else:
                    executions_to_clean.append(execution_id)
        
        # 清理空的执行记录
        for execution_id in executions_to_clean:
            await self.cleanup_execution(execution_id)
    
    async def _trigger_history_handlers(self, event_type: str, data: Any) -> None:
        """触发历史处理器"""
        for handler in self._history_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data)
                else:
                    handler(event_type, data)
            except Exception as e:
                print(f"历史处理器错误: {e}")