"""
检查点管理器

管理执行检查点的创建、存储和恢复。
"""

import asyncio
import json
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path

from .time_travel_types import (
    Checkpoint, CheckpointType, StateSnapshot, TimeTravelConfig
)


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, config: Optional[TimeTravelConfig] = None):
        self.config = config or TimeTravelConfig()
        
        # 检查点存储
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._execution_checkpoints: Dict[str, List[str]] = {}  # execution_id -> checkpoint_ids
        self._agent_checkpoints: Dict[str, List[str]] = {}  # agent_id -> checkpoint_ids
        
        # 自动检查点配置
        self._auto_checkpoint_rules: List[Dict[str, Any]] = []
        
        # 事件处理器
        self._checkpoint_handlers: List[Callable] = []
        self._restore_handlers: List[Callable] = []
        
        # 统计信息
        self._stats = {
            "total_checkpoints": 0,
            "auto_checkpoints": 0,
            "manual_checkpoints": 0,
            "restored_checkpoints": 0
        }
    
    async def create_checkpoint(
        self,
        name: str,
        checkpoint_type: CheckpointType,
        snapshot_id: str,
        execution_id: str,
        agent_id: str,
        description: Optional[str] = None,
        step_name: Optional[str] = None,
        restore_config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """创建检查点"""
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
            restore_config=restore_config or {},
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # 存储检查点
        self._checkpoints[checkpoint_id] = checkpoint
        
        # 更新索引
        if execution_id not in self._execution_checkpoints:
            self._execution_checkpoints[execution_id] = []
        self._execution_checkpoints[execution_id].append(checkpoint_id)
        
        if agent_id not in self._agent_checkpoints:
            self._agent_checkpoints[agent_id] = []
        self._agent_checkpoints[agent_id].append(checkpoint_id)
        
        # 更新统计
        self._stats["total_checkpoints"] += 1
        if checkpoint_type == CheckpointType.USER:
            self._stats["manual_checkpoints"] += 1
        else:
            self._stats["auto_checkpoints"] += 1
        
        # 触发处理器
        await self._trigger_checkpoint_handlers("created", checkpoint)
        
        # 自动清理
        if self.config.auto_cleanup:
            await self._cleanup_old_checkpoints()
        
        return checkpoint
    
    async def create_auto_checkpoint(
        self,
        execution_id: str,
        agent_id: str,
        snapshot_id: str,
        step_name: str,
        trigger_reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """创建自动检查点"""
        if not self.config.auto_checkpoint:
            raise RuntimeError("自动检查点功能未启用")
        
        # 确定检查点类型
        checkpoint_type = CheckpointType.STEP
        if "error" in trigger_reason.lower():
            checkpoint_type = CheckpointType.ERROR
        elif "milestone" in trigger_reason.lower():
            checkpoint_type = CheckpointType.MILESTONE
        
        name = f"Auto-{checkpoint_type.value}-{step_name}"
        description = f"自动创建的检查点: {trigger_reason}"
        
        return await self.create_checkpoint(
            name=name,
            checkpoint_type=checkpoint_type,
            snapshot_id=snapshot_id,
            execution_id=execution_id,
            agent_id=agent_id,
            description=description,
            step_name=step_name,
            tags=["auto", trigger_reason],
            metadata=metadata
        )
    
    async def create_milestone_checkpoint(
        self,
        execution_id: str,
        agent_id: str,
        snapshot_id: str,
        milestone_name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """创建里程碑检查点"""
        name = f"Milestone-{milestone_name}"
        
        return await self.create_checkpoint(
            name=name,
            checkpoint_type=CheckpointType.MILESTONE,
            snapshot_id=snapshot_id,
            execution_id=execution_id,
            agent_id=agent_id,
            description=description or f"里程碑: {milestone_name}",
            tags=["milestone", milestone_name],
            metadata=metadata
        )
    
    async def create_error_checkpoint(
        self,
        execution_id: str,
        agent_id: str,
        snapshot_id: str,
        error_info: Dict[str, Any],
        step_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """创建错误检查点"""
        if not self.config.checkpoint_on_error:
            return None
        
        error_type = error_info.get("type", "unknown")
        name = f"Error-{error_type}"
        description = f"错误检查点: {error_info.get('message', 'Unknown error')}"
        
        checkpoint_metadata = {
            "error_info": error_info,
            **(metadata or {})
        }
        
        return await self.create_checkpoint(
            name=name,
            checkpoint_type=CheckpointType.ERROR,
            snapshot_id=snapshot_id,
            execution_id=execution_id,
            agent_id=agent_id,
            description=description,
            step_name=step_name,
            tags=["error", error_type],
            metadata=checkpoint_metadata
        )
    
    async def restore_checkpoint(
        self,
        checkpoint_id: str,
        restore_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """恢复检查点"""
        if checkpoint_id not in self._checkpoints:
            raise ValueError(f"检查点不存在: {checkpoint_id}")
        
        checkpoint = self._checkpoints[checkpoint_id]
        
        # 合并恢复配置
        restore_config = {
            **checkpoint.restore_config,
            **(restore_options or {})
        }
        
        # 执行恢复
        restore_result = {
            "checkpoint_id": checkpoint_id,
            "snapshot_id": checkpoint.snapshot_id,
            "execution_id": checkpoint.execution_id,
            "agent_id": checkpoint.agent_id,
            "restored_at": datetime.now(),
            "restore_config": restore_config
        }
        
        # 更新统计
        self._stats["restored_checkpoints"] += 1
        
        # 触发处理器
        await self._trigger_restore_handlers("restored", checkpoint, restore_result)
        
        return restore_result
    
    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """获取检查点"""
        return self._checkpoints.get(checkpoint_id)
    
    async def list_checkpoints(
        self,
        execution_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        checkpoint_type: Optional[CheckpointType] = None,
        tags: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Checkpoint]:
        """列出检查点"""
        checkpoints = list(self._checkpoints.values())
        
        # 应用过滤条件
        if execution_id:
            checkpoint_ids = self._execution_checkpoints.get(execution_id, [])
            checkpoints = [cp for cp in checkpoints if cp.id in checkpoint_ids]
        
        if agent_id:
            checkpoint_ids = self._agent_checkpoints.get(agent_id, [])
            checkpoints = [cp for cp in checkpoints if cp.id in checkpoint_ids]
        
        if checkpoint_type:
            checkpoints = [cp for cp in checkpoints if cp.type == checkpoint_type]
        
        if tags:
            checkpoints = [
                cp for cp in checkpoints
                if any(tag in cp.tags for tag in tags)
            ]
        
        if start_time:
            checkpoints = [cp for cp in checkpoints if cp.created_at >= start_time]
        
        if end_time:
            checkpoints = [cp for cp in checkpoints if cp.created_at <= end_time]
        
        # 排序
        checkpoints.sort(key=lambda cp: cp.created_at, reverse=True)
        
        # 限制数量
        if limit:
            checkpoints = checkpoints[:limit]
        
        return checkpoints
    
    async def get_execution_checkpoints(self, execution_id: str) -> List[Checkpoint]:
        """获取执行的所有检查点"""
        return await self.list_checkpoints(execution_id=execution_id)
    
    async def get_agent_checkpoints(self, agent_id: str) -> List[Checkpoint]:
        """获取智能体的所有检查点"""
        return await self.list_checkpoints(agent_id=agent_id)
    
    async def get_latest_checkpoint(
        self,
        execution_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        checkpoint_type: Optional[CheckpointType] = None
    ) -> Optional[Checkpoint]:
        """获取最新检查点"""
        checkpoints = await self.list_checkpoints(
            execution_id=execution_id,
            agent_id=agent_id,
            checkpoint_type=checkpoint_type,
            limit=1
        )
        
        return checkpoints[0] if checkpoints else None
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        if checkpoint_id not in self._checkpoints:
            return False
        
        checkpoint = self._checkpoints[checkpoint_id]
        
        # 从索引中移除
        if checkpoint.execution_id in self._execution_checkpoints:
            self._execution_checkpoints[checkpoint.execution_id].remove(checkpoint_id)
        
        if checkpoint.agent_id in self._agent_checkpoints:
            self._agent_checkpoints[checkpoint.agent_id].remove(checkpoint_id)
        
        # 删除检查点
        del self._checkpoints[checkpoint_id]
        
        # 更新统计
        self._stats["total_checkpoints"] -= 1
        
        # 触发处理器
        await self._trigger_checkpoint_handlers("deleted", checkpoint)
        
        return True
    
    async def add_auto_checkpoint_rule(
        self,
        rule_name: str,
        condition: Dict[str, Any],
        checkpoint_config: Dict[str, Any]
    ) -> None:
        """添加自动检查点规则"""
        rule = {
            "name": rule_name,
            "condition": condition,
            "config": checkpoint_config,
            "created_at": datetime.now()
        }
        
        self._auto_checkpoint_rules.append(rule)
    
    async def evaluate_auto_checkpoint_rules(
        self,
        execution_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """评估自动检查点规则"""
        triggered_rules = []
        
        for rule in self._auto_checkpoint_rules:
            if await self._evaluate_rule_condition(rule["condition"], execution_context):
                triggered_rules.append(rule)
        
        return triggered_rules
    
    async def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """获取检查点统计信息"""
        # 按类型统计
        type_stats = {}
        for checkpoint in self._checkpoints.values():
            type_name = checkpoint.type.value
            if type_name not in type_stats:
                type_stats[type_name] = 0
            type_stats[type_name] += 1
        
        # 按时间统计
        now = datetime.now()
        recent_checkpoints = [
            cp for cp in self._checkpoints.values()
            if (now - cp.created_at).days <= 7
        ]
        
        return {
            **self._stats,
            "by_type": type_stats,
            "recent_count": len(recent_checkpoints),
            "total_executions": len(self._execution_checkpoints),
            "total_agents": len(self._agent_checkpoints)
        }
    
    def register_checkpoint_handler(self, handler: Callable) -> None:
        """注册检查点处理器"""
        self._checkpoint_handlers.append(handler)
    
    def register_restore_handler(self, handler: Callable) -> None:
        """注册恢复处理器"""
        self._restore_handlers.append(handler)
    
    async def _evaluate_rule_condition(
        self,
        condition: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """评估规则条件"""
        condition_type = condition.get("type", "simple")
        
        if condition_type == "simple":
            # 简单条件匹配
            for key, expected_value in condition.get("match", {}).items():
                if context.get(key) != expected_value:
                    return False
            return True
        
        elif condition_type == "expression":
            # 表达式条件（简化实现）
            expression = condition.get("expression", "")
            try:
                # 注意：实际应用中应该使用安全的表达式评估器
                return eval(expression, {"context": context})
            except:
                return False
        
        elif condition_type == "step_count":
            # 步骤数量条件
            step_count = context.get("step_count", 0)
            interval = condition.get("interval", 10)
            return step_count > 0 and step_count % interval == 0
        
        elif condition_type == "time_interval":
            # 时间间隔条件
            last_checkpoint_time = context.get("last_checkpoint_time")
            if not last_checkpoint_time:
                return True
            
            interval_minutes = condition.get("interval_minutes", 30)
            time_diff = (datetime.now() - last_checkpoint_time).total_seconds() / 60
            return time_diff >= interval_minutes
        
        return False
    
    async def _cleanup_old_checkpoints(self) -> None:
        """清理旧检查点"""
        if not self.config.auto_cleanup:
            return
        
        cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
        
        # 找到需要删除的检查点
        checkpoints_to_delete = [
            cp_id for cp_id, cp in self._checkpoints.items()
            if cp.created_at < cutoff_time and cp.type != CheckpointType.MILESTONE
        ]
        
        # 保留每个执行的最新检查点
        for execution_id, checkpoint_ids in self._execution_checkpoints.items():
            if checkpoint_ids:
                # 按时间排序，保留最新的
                execution_checkpoints = [
                    self._checkpoints[cp_id] for cp_id in checkpoint_ids
                    if cp_id in self._checkpoints
                ]
                execution_checkpoints.sort(key=lambda cp: cp.created_at, reverse=True)
                
                # 保留最新的检查点
                if execution_checkpoints:
                    latest_id = execution_checkpoints[0].id
                    if latest_id in checkpoints_to_delete:
                        checkpoints_to_delete.remove(latest_id)
        
        # 删除检查点
        for cp_id in checkpoints_to_delete:
            await self.delete_checkpoint(cp_id)
    
    async def _trigger_checkpoint_handlers(self, event_type: str, checkpoint: Checkpoint) -> None:
        """触发检查点处理器"""
        for handler in self._checkpoint_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, checkpoint)
                else:
                    handler(event_type, checkpoint)
            except Exception as e:
                print(f"检查点处理器错误: {e}")
    
    async def _trigger_restore_handlers(
        self,
        event_type: str,
        checkpoint: Checkpoint,
        restore_result: Dict[str, Any]
    ) -> None:
        """触发恢复处理器"""
        for handler in self._restore_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, checkpoint, restore_result)
                else:
                    handler(event_type, checkpoint, restore_result)
            except Exception as e:
                print(f"恢复处理器错误: {e}")