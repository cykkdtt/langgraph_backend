"""时间旅行仓储模块

提供时间旅行相关的数据访问接口和业务逻辑。
"""

import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, func, desc
from sqlalchemy.orm import Session, selectinload

from ..models.database import TimeTravel, EntityType
from ..models.api import TimeTravelCreate, TimeTravelUpdate, TimeTravelStats
from ..database.repository import CRUDRepository, EntityNotFoundError
from ..database.query_builder import QueryFilter
from ..database.session import SessionManager

logger = logging.getLogger(__name__)


class TimeTravelRepository(CRUDRepository[TimeTravel]):
    """时间旅行仓储类"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        super().__init__(TimeTravel, session_manager)
        # 设置软删除字段
        self.set_soft_delete_field("deleted_at")
        # 设置默认加载选项
        self.set_default_load_options(
            selectinload(TimeTravel.session)
        )
    
    def create_snapshot(
        self, 
        entity_type: EntityType,
        entity_id: int,
        session_id: int,
        state_data: Dict[str, Any],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None
    ) -> TimeTravel:
        """创建状态快照"""
        snapshot_data = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "session_id": session_id,
            "state_data": state_data,
            "description": description,
            "metadata": metadata or {},
            "is_active": True
        }
        
        return self.create(snapshot_data, session)
    
    def get_entity_snapshots(
        self, 
        entity_type: EntityType,
        entity_id: int,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[TimeTravel]:
        """获取实体的所有快照"""
        filters = QueryFilter().and_(
            QueryFilter().eq("entity_type", entity_type),
            QueryFilter().eq("entity_id", entity_id),
            QueryFilter().eq("is_active", True)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_session_snapshots(
        self, 
        session_id: int,
        entity_type: Optional[EntityType] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[TimeTravel]:
        """获取会话的所有快照"""
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
            QueryFilter().eq("is_active", True)
        )
        
        if entity_type:
            filters = filters.and_(QueryFilter().eq("entity_type", entity_type))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_snapshot_at_time(
        self, 
        entity_type: EntityType,
        entity_id: int,
        target_time: datetime,
        session: Optional[Session] = None
    ) -> Optional[TimeTravel]:
        """获取指定时间点的快照"""
        filters = QueryFilter().and_(
            QueryFilter().eq("entity_type", entity_type),
            QueryFilter().eq("entity_id", entity_id),
            QueryFilter().lte("created_at", target_time),
            QueryFilter().eq("is_active", True)
        )
        
        snapshots = self.get_multi(
            filters=filters,
            order_by=[("created_at", "desc")],
            limit=1,
            session=session
        )
        
        return snapshots[0] if snapshots else None
    
    def get_latest_snapshot(
        self, 
        entity_type: EntityType,
        entity_id: int,
        session: Optional[Session] = None
    ) -> Optional[TimeTravel]:
        """获取最新快照"""
        filters = QueryFilter().and_(
            QueryFilter().eq("entity_type", entity_type),
            QueryFilter().eq("entity_id", entity_id),
            QueryFilter().eq("is_active", True)
        )
        
        snapshots = self.get_multi(
            filters=filters,
            order_by=[("created_at", "desc")],
            limit=1,
            session=session
        )
        
        return snapshots[0] if snapshots else None
    
    def get_snapshots_between(
        self, 
        entity_type: EntityType,
        entity_id: int,
        start_time: datetime,
        end_time: datetime,
        session: Optional[Session] = None
    ) -> List[TimeTravel]:
        """获取时间范围内的快照"""
        filters = QueryFilter().and_(
            QueryFilter().eq("entity_type", entity_type),
            QueryFilter().eq("entity_id", entity_id),
            QueryFilter().gte("created_at", start_time),
            QueryFilter().lte("created_at", end_time),
            QueryFilter().eq("is_active", True)
        )
        
        return self.get_multi(
            filters=filters,
            order_by=[("created_at", "asc")],
            session=session
        )
    
    def get_snapshot_history(
        self, 
        entity_type: EntityType,
        entity_id: int,
        days: int = 30,
        session: Optional[Session] = None
    ) -> List[TimeTravel]:
        """获取实体的历史快照"""
        start_time = datetime.utcnow() - timedelta(days=days)
        
        return self.get_snapshots_between(
            entity_type=entity_type,
            entity_id=entity_id,
            start_time=start_time,
            end_time=datetime.utcnow(),
            session=session
        )
    
    def compare_snapshots(
        self, 
        snapshot1_id: int,
        snapshot2_id: int,
        session: Optional[Session] = None
    ) -> Optional[Dict[str, Any]]:
        """比较两个快照的差异"""
        try:
            snapshot1 = self.get(snapshot1_id, session)
            snapshot2 = self.get(snapshot2_id, session)
            
            if not snapshot1 or not snapshot2:
                return None
            
            if (snapshot1.entity_type != snapshot2.entity_type or 
                snapshot1.entity_id != snapshot2.entity_id):
                logger.warning("Comparing snapshots of different entities")
                return None
            
            # 计算状态差异
            differences = self._calculate_state_differences(
                snapshot1.state_data, 
                snapshot2.state_data
            )
            
            return {
                "snapshot1": {
                    "id": snapshot1.id,
                    "created_at": snapshot1.created_at.isoformat(),
                    "description": snapshot1.description
                },
                "snapshot2": {
                    "id": snapshot2.id,
                    "created_at": snapshot2.created_at.isoformat(),
                    "description": snapshot2.description
                },
                "differences": differences,
                "entity_type": snapshot1.entity_type.value,
                "entity_id": snapshot1.entity_id
            }
            
        except Exception as e:
            logger.error(f"Error comparing snapshots {snapshot1_id} and {snapshot2_id}: {e}")
            return None
    
    def _calculate_state_differences(
        self, 
        state1: Dict[str, Any], 
        state2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算状态差异"""
        differences = {
            "added": {},
            "removed": {},
            "modified": {},
            "unchanged": {}
        }
        
        all_keys = set(state1.keys()) | set(state2.keys())
        
        for key in all_keys:
            if key in state1 and key in state2:
                if state1[key] != state2[key]:
                    differences["modified"][key] = {
                        "old_value": state1[key],
                        "new_value": state2[key]
                    }
                else:
                    differences["unchanged"][key] = state1[key]
            elif key in state1:
                differences["removed"][key] = state1[key]
            else:
                differences["added"][key] = state2[key]
        
        return differences
    
    def rollback_to_snapshot(
        self, 
        snapshot_id: int,
        session: Optional[Session] = None
    ) -> Optional[Dict[str, Any]]:
        """回滚到指定快照"""
        try:
            snapshot = self.get(snapshot_id, session)
            if not snapshot:
                raise EntityNotFoundError(f"Snapshot with ID {snapshot_id} not found")
            
            # 创建回滚记录
            rollback_data = {
                "entity_type": snapshot.entity_type,
                "entity_id": snapshot.entity_id,
                "session_id": snapshot.session_id,
                "state_data": snapshot.state_data,
                "description": f"Rollback to snapshot {snapshot_id}",
                "metadata": {
                    "rollback_from_snapshot": snapshot_id,
                    "rollback_time": datetime.utcnow().isoformat()
                },
                "is_active": True
            }
            
            rollback_snapshot = self.create(rollback_data, session)
            
            return {
                "rollback_snapshot_id": rollback_snapshot.id,
                "original_snapshot_id": snapshot_id,
                "entity_type": snapshot.entity_type.value,
                "entity_id": snapshot.entity_id,
                "state_data": snapshot.state_data,
                "rollback_time": rollback_snapshot.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error rolling back to snapshot {snapshot_id}: {e}")
            return None
    
    def create_checkpoint(
        self, 
        session_id: int,
        checkpoint_name: str,
        description: Optional[str] = None,
        session: Optional[Session] = None
    ) -> List[TimeTravel]:
        """创建会话检查点（为会话中的所有实体创建快照）"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 这里需要根据实际业务逻辑获取会话中的所有实体
            # 示例：获取会话中的线程、消息、工作流等
            from ..models.database import Thread, Message, Workflow, Memory
            
            checkpoints = []
            checkpoint_time = datetime.utcnow()
            
            # 为线程创建快照
            threads = db_session.query(Thread).filter(
                Thread.session_id == session_id,
                Thread.is_active == True
            ).all()
            
            for thread in threads:
                thread_state = {
                    "title": thread.title,
                    "description": thread.description,
                    "is_active": thread.is_active,
                    "message_count": thread.message_count,
                    "last_message_at": thread.last_message_at.isoformat() if thread.last_message_at else None,
                    "metadata": thread.metadata
                }
                
                checkpoint = self.create_snapshot(
                    entity_type=EntityType.THREAD,
                    entity_id=thread.id,
                    session_id=session_id,
                    state_data=thread_state,
                    description=f"Checkpoint: {checkpoint_name} - Thread {thread.title}",
                    metadata={
                        "checkpoint_name": checkpoint_name,
                        "checkpoint_time": checkpoint_time.isoformat(),
                        "description": description
                    },
                    session=db_session
                )
                checkpoints.append(checkpoint)
            
            # 为工作流创建快照
            workflows = db_session.query(Workflow).filter(
                Workflow.session_id == session_id,
                Workflow.is_active == True
            ).all()
            
            for workflow in workflows:
                workflow_state = {
                    "name": workflow.name,
                    "description": workflow.description,
                    "workflow_type": workflow.workflow_type.value,
                    "status": workflow.status.value,
                    "config": workflow.config,
                    "result": workflow.result,
                    "progress": workflow.progress,
                    "current_step": workflow.current_step,
                    "is_active": workflow.is_active
                }
                
                checkpoint = self.create_snapshot(
                    entity_type=EntityType.WORKFLOW,
                    entity_id=workflow.id,
                    session_id=session_id,
                    state_data=workflow_state,
                    description=f"Checkpoint: {checkpoint_name} - Workflow {workflow.name}",
                    metadata={
                        "checkpoint_name": checkpoint_name,
                        "checkpoint_time": checkpoint_time.isoformat(),
                        "description": description
                    },
                    session=db_session
                )
                checkpoints.append(checkpoint)
            
            logger.info(f"Created checkpoint '{checkpoint_name}' with {len(checkpoints)} snapshots")
            return checkpoints
            
        except Exception as e:
            logger.error(f"Error creating checkpoint '{checkpoint_name}': {e}")
            return []
        finally:
            if not session:
                db_session.close()
    
    def get_checkpoints(
        self, 
        session_id: int,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """获取会话的检查点列表"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 查询包含检查点名称的快照
            query = db_session.query(TimeTravel)
            query = query.filter(
                TimeTravel.session_id == session_id,
                TimeTravel.metadata.op('->>')('checkpoint_name').isnot(None),
                TimeTravel.is_active == True
            )
            
            # 按检查点名称和时间分组
            snapshots = query.order_by(desc(TimeTravel.created_at)).all()
            
            checkpoints = {}
            for snapshot in snapshots:
                checkpoint_name = snapshot.metadata.get('checkpoint_name')
                checkpoint_time = snapshot.metadata.get('checkpoint_time')
                
                if checkpoint_name not in checkpoints:
                    checkpoints[checkpoint_name] = {
                        "name": checkpoint_name,
                        "description": snapshot.metadata.get('description'),
                        "created_at": checkpoint_time,
                        "snapshot_count": 0,
                        "snapshots": []
                    }
                
                checkpoints[checkpoint_name]["snapshot_count"] += 1
                checkpoints[checkpoint_name]["snapshots"].append({
                    "id": snapshot.id,
                    "entity_type": snapshot.entity_type.value,
                    "entity_id": snapshot.entity_id,
                    "description": snapshot.description
                })
            
            # 转换为列表并排序
            checkpoint_list = list(checkpoints.values())
            checkpoint_list.sort(key=lambda x: x['created_at'], reverse=True)
            
            # 应用分页
            return checkpoint_list[skip:skip + limit]
            
        except Exception as e:
            logger.error(f"Error getting checkpoints for session {session_id}: {e}")
            return []
        finally:
            if not session:
                db_session.close()
    
    def restore_checkpoint(
        self, 
        session_id: int,
        checkpoint_name: str,
        session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """恢复到指定检查点"""
        try:
            # 获取检查点的所有快照
            filters = QueryFilter().and_(
                QueryFilter().eq("session_id", session_id),
                QueryFilter().eq("is_active", True)
            )
            
            snapshots = self.get_multi(
                filters=filters,
                session=session
            )
            
            checkpoint_snapshots = [
                snapshot for snapshot in snapshots
                if (snapshot.metadata and 
                    snapshot.metadata.get('checkpoint_name') == checkpoint_name)
            ]
            
            if not checkpoint_snapshots:
                logger.warning(f"No snapshots found for checkpoint '{checkpoint_name}'")
                return []
            
            # 为每个快照执行回滚
            rollback_results = []
            for snapshot in checkpoint_snapshots:
                rollback_result = self.rollback_to_snapshot(snapshot.id, session)
                if rollback_result:
                    rollback_results.append(rollback_result)
            
            logger.info(f"Restored checkpoint '{checkpoint_name}' with {len(rollback_results)} rollbacks")
            return rollback_results
            
        except Exception as e:
            logger.error(f"Error restoring checkpoint '{checkpoint_name}': {e}")
            return []
    
    def get_time_travel_statistics(
        self, 
        session_id: Optional[int] = None,
        entity_type: Optional[EntityType] = None,
        days: int = 30,
        session: Optional[Session] = None
    ) -> TimeTravelStats:
        """获取时间旅行统计信息"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            base_filters = QueryFilter().and_(
                QueryFilter().gte("created_at", start_date),
                QueryFilter().eq("is_active", True)
            )
            
            if session_id:
                base_filters = base_filters.and_(QueryFilter().eq("session_id", session_id))
            
            if entity_type:
                base_filters = base_filters.and_(QueryFilter().eq("entity_type", entity_type))
            
            # 总快照数
            total_snapshots = self.count(filters=base_filters, session=session)
            
            # 今日快照数
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_filters = base_filters.and_(QueryFilter().gte("created_at", today_start))
            today_snapshots = self.count(filters=today_filters, session=session)
            
            # 按实体类型统计
            thread_filters = base_filters.and_(QueryFilter().eq("entity_type", EntityType.THREAD))
            thread_snapshots = self.count(filters=thread_filters, session=session)
            
            message_filters = base_filters.and_(QueryFilter().eq("entity_type", EntityType.MESSAGE))
            message_snapshots = self.count(filters=message_filters, session=session)
            
            workflow_filters = base_filters.and_(QueryFilter().eq("entity_type", EntityType.WORKFLOW))
            workflow_snapshots = self.count(filters=workflow_filters, session=session)
            
            memory_filters = base_filters.and_(QueryFilter().eq("entity_type", EntityType.MEMORY))
            memory_snapshots = self.count(filters=memory_filters, session=session)
            
            # 检查点数量（通过元数据中的检查点名称统计）
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            try:
                query = db_session.query(TimeTravel)
                query = query.filter(
                    TimeTravel.created_at >= start_date,
                    TimeTravel.metadata.op('->>')('checkpoint_name').isnot(None),
                    TimeTravel.is_active == True
                )
                
                if session_id:
                    query = query.filter(TimeTravel.session_id == session_id)
                
                checkpoint_snapshots = query.all()
                unique_checkpoints = set(
                    snapshot.metadata.get('checkpoint_name') 
                    for snapshot in checkpoint_snapshots
                    if snapshot.metadata and snapshot.metadata.get('checkpoint_name')
                )
                checkpoint_count = len(unique_checkpoints)
                
            except Exception as e:
                logger.error(f"Error counting checkpoints: {e}")
                checkpoint_count = 0
            finally:
                if not session:
                    db_session.close()
            
            return TimeTravelStats(
                total_snapshots=total_snapshots,
                today_snapshots=today_snapshots,
                thread_snapshots=thread_snapshots,
                message_snapshots=message_snapshots,
                workflow_snapshots=workflow_snapshots,
                memory_snapshots=memory_snapshots,
                checkpoint_count=checkpoint_count
            )
            
        except Exception as e:
            logger.error(f"Error getting time travel statistics: {e}")
            return TimeTravelStats(
                total_snapshots=0,
                today_snapshots=0,
                thread_snapshots=0,
                message_snapshots=0,
                workflow_snapshots=0,
                memory_snapshots=0,
                checkpoint_count=0
            )
    
    def cleanup_old_snapshots(
        self, 
        days: int = 90,
        keep_checkpoints: bool = True,
        session: Optional[Session] = None
    ) -> int:
        """清理旧快照"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            base_filters = QueryFilter().and_(
                QueryFilter().lt("created_at", cutoff_date),
                QueryFilter().eq("is_active", True)
            )
            
            # 如果保留检查点，排除检查点快照
            if keep_checkpoints:
                # 这里需要根据实际数据库实现调整查询
                old_snapshots = self.get_multi(
                    filters=base_filters,
                    limit=1000,
                    session=session
                )
                
                # 过滤掉检查点快照
                non_checkpoint_snapshots = [
                    snapshot for snapshot in old_snapshots
                    if not (snapshot.metadata and snapshot.metadata.get('checkpoint_name'))
                ]
            else:
                non_checkpoint_snapshots = self.get_multi(
                    filters=base_filters,
                    limit=1000,
                    session=session
                )
            
            # 软删除这些快照
            deleted_count = 0
            for snapshot in non_checkpoint_snapshots:
                if self.soft_delete(snapshot.id, session):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old snapshots")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old snapshots: {e}")
            return 0
    
    def get_entity_timeline(
        self, 
        entity_type: EntityType,
        entity_id: int,
        days: int = 30,
        session: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """获取实体的时间线"""
        try:
            snapshots = self.get_snapshot_history(
                entity_type=entity_type,
                entity_id=entity_id,
                days=days,
                session=session
            )
            
            timeline = []
            for i, snapshot in enumerate(snapshots):
                timeline_item = {
                    "id": snapshot.id,
                    "created_at": snapshot.created_at.isoformat(),
                    "description": snapshot.description,
                    "metadata": snapshot.metadata,
                    "state_summary": self._create_state_summary(snapshot.state_data)
                }
                
                # 如果不是第一个快照，计算与前一个快照的差异
                if i > 0:
                    prev_snapshot = snapshots[i - 1]
                    differences = self._calculate_state_differences(
                        prev_snapshot.state_data,
                        snapshot.state_data
                    )
                    timeline_item["changes"] = differences
                
                timeline.append(timeline_item)
            
            return timeline
            
        except Exception as e:
            logger.error(f"Error getting entity timeline: {e}")
            return []
    
    def _create_state_summary(
        self, 
        state_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建状态摘要"""
        summary = {}
        
        # 提取关键字段
        key_fields = ['title', 'name', 'status', 'is_active', 'description']
        
        for field in key_fields:
            if field in state_data:
                summary[field] = state_data[field]
        
        # 添加字段数量
        summary['field_count'] = len(state_data)
        
        return summary