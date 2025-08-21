"""工作流仓储模块

提供工作流相关的数据访问接口和业务逻辑。
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session, selectinload

from ..models.database import Workflow, WorkflowStatus, WorkflowType
from ..models.api import WorkflowCreate, WorkflowUpdate, WorkflowStats
from ..database.repository import CRUDRepository, EntityNotFoundError
from ..database.query_builder import QueryFilter
from ..database.session import SessionManager

logger = logging.getLogger(__name__)


class WorkflowRepository(CRUDRepository[Workflow]):
    """工作流仓储类"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        super().__init__(Workflow, session_manager)
        # 设置软删除字段
        self.set_soft_delete_field("deleted_at")
        # 设置默认加载选项
        self.set_default_load_options(
            selectinload(Workflow.session),
            selectinload(Workflow.thread)
        )
    
    def create_workflow(
        self, 
        workflow_create: WorkflowCreate, 
        session_id: int,
        thread_id: Optional[int] = None,
        session: Optional[Session] = None
    ) -> Workflow:
        """创建工作流"""
        workflow_data = workflow_create.model_dump()
        workflow_data["session_id"] = session_id
        
        if thread_id:
            workflow_data["thread_id"] = thread_id
        
        # 设置初始状态
        workflow_data["status"] = WorkflowStatus.PENDING
        workflow_data["is_active"] = True
        
        return self.create(workflow_data, session)
    
    def get_session_workflows(
        self, 
        session_id: int, 
        skip: int = 0, 
        limit: int = 100,
        include_inactive: bool = False,
        session: Optional[Session] = None
    ) -> List[Workflow]:
        """获取会话的工作流列表"""
        filters = QueryFilter().eq("session_id", session_id)
        
        if not include_inactive:
            filters = filters.and_(QueryFilter().eq("is_active", True))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_thread_workflows(
        self, 
        thread_id: int, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Workflow]:
        """获取线程的工作流列表"""
        filters = QueryFilter().eq("thread_id", thread_id)
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_workflows_by_status(
        self, 
        status: WorkflowStatus,
        session_id: Optional[int] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Workflow]:
        """根据状态获取工作流"""
        filters = QueryFilter().eq("status", status)
        
        if session_id:
            filters = filters.and_(QueryFilter().eq("session_id", session_id))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("updated_at", "desc")],
            session=session
        )
    
    def get_workflows_by_type(
        self, 
        workflow_type: WorkflowType,
        session_id: Optional[int] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Workflow]:
        """根据类型获取工作流"""
        filters = QueryFilter().eq("workflow_type", workflow_type)
        
        if session_id:
            filters = filters.and_(QueryFilter().eq("session_id", session_id))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_running_workflows(
        self, 
        session_id: Optional[int] = None,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Workflow]:
        """获取正在运行的工作流"""
        filters = QueryFilter().eq("status", WorkflowStatus.RUNNING)
        
        if session_id:
            filters = filters.and_(QueryFilter().eq("session_id", session_id))
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("started_at", "desc")],
            session=session
        )
    
    def start_workflow(
        self, 
        workflow_id: int,
        session: Optional[Session] = None
    ) -> Optional[Workflow]:
        """启动工作流"""
        try:
            workflow = self.get(workflow_id, session)
            if not workflow:
                raise EntityNotFoundError(f"Workflow with ID {workflow_id} not found")
            
            if workflow.status != WorkflowStatus.PENDING:
                logger.warning(f"Workflow {workflow_id} is not in pending status")
                return None
            
            update_data = {
                "status": WorkflowStatus.RUNNING,
                "started_at": datetime.utcnow()
            }
            
            return self.update(workflow, update_data, session)
            
        except Exception as e:
            logger.error(f"Error starting workflow {workflow_id}: {e}")
            return None
    
    def complete_workflow(
        self, 
        workflow_id: int,
        result: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None
    ) -> Optional[Workflow]:
        """完成工作流"""
        try:
            workflow = self.get(workflow_id, session)
            if not workflow:
                raise EntityNotFoundError(f"Workflow with ID {workflow_id} not found")
            
            if workflow.status not in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]:
                logger.warning(f"Workflow {workflow_id} is not in running or paused status")
                return None
            
            update_data = {
                "status": WorkflowStatus.COMPLETED,
                "completed_at": datetime.utcnow()
            }
            
            if result:
                current_result = workflow.result or {}
                current_result.update(result)
                update_data["result"] = current_result
            
            return self.update(workflow, update_data, session)
            
        except Exception as e:
            logger.error(f"Error completing workflow {workflow_id}: {e}")
            return None
    
    def fail_workflow(
        self, 
        workflow_id: int,
        error_message: str,
        session: Optional[Session] = None
    ) -> Optional[Workflow]:
        """标记工作流失败"""
        try:
            workflow = self.get(workflow_id, session)
            if not workflow:
                raise EntityNotFoundError(f"Workflow with ID {workflow_id} not found")
            
            update_data = {
                "status": WorkflowStatus.FAILED,
                "error_message": error_message,
                "completed_at": datetime.utcnow()
            }
            
            return self.update(workflow, update_data, session)
            
        except Exception as e:
            logger.error(f"Error failing workflow {workflow_id}: {e}")
            return None
    
    def pause_workflow(
        self, 
        workflow_id: int,
        session: Optional[Session] = None
    ) -> Optional[Workflow]:
        """暂停工作流"""
        try:
            workflow = self.get(workflow_id, session)
            if not workflow:
                raise EntityNotFoundError(f"Workflow with ID {workflow_id} not found")
            
            if workflow.status != WorkflowStatus.RUNNING:
                logger.warning(f"Workflow {workflow_id} is not in running status")
                return None
            
            update_data = {
                "status": WorkflowStatus.PAUSED
            }
            
            return self.update(workflow, update_data, session)
            
        except Exception as e:
            logger.error(f"Error pausing workflow {workflow_id}: {e}")
            return None
    
    def resume_workflow(
        self, 
        workflow_id: int,
        session: Optional[Session] = None
    ) -> Optional[Workflow]:
        """恢复工作流"""
        try:
            workflow = self.get(workflow_id, session)
            if not workflow:
                raise EntityNotFoundError(f"Workflow with ID {workflow_id} not found")
            
            if workflow.status != WorkflowStatus.PAUSED:
                logger.warning(f"Workflow {workflow_id} is not in paused status")
                return None
            
            update_data = {
                "status": WorkflowStatus.RUNNING
            }
            
            return self.update(workflow, update_data, session)
            
        except Exception as e:
            logger.error(f"Error resuming workflow {workflow_id}: {e}")
            return None
    
    def cancel_workflow(
        self, 
        workflow_id: int,
        session: Optional[Session] = None
    ) -> Optional[Workflow]:
        """取消工作流"""
        try:
            workflow = self.get(workflow_id, session)
            if not workflow:
                raise EntityNotFoundError(f"Workflow with ID {workflow_id} not found")
            
            if workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                logger.warning(f"Workflow {workflow_id} is already finished")
                return None
            
            update_data = {
                "status": WorkflowStatus.CANCELLED,
                "completed_at": datetime.utcnow()
            }
            
            return self.update(workflow, update_data, session)
            
        except Exception as e:
            logger.error(f"Error cancelling workflow {workflow_id}: {e}")
            return None
    
    def update_workflow_progress(
        self, 
        workflow_id: int,
        progress: float,
        current_step: Optional[str] = None,
        session: Optional[Session] = None
    ) -> Optional[Workflow]:
        """更新工作流进度"""
        try:
            workflow = self.get(workflow_id, session)
            if not workflow:
                raise EntityNotFoundError(f"Workflow with ID {workflow_id} not found")
            
            # 确保进度在0-100之间
            progress = max(0, min(100, progress))
            
            update_data = {
                "progress": progress
            }
            
            if current_step:
                update_data["current_step"] = current_step
            
            return self.update(workflow, update_data, session)
            
        except Exception as e:
            logger.error(f"Error updating workflow progress for workflow {workflow_id}: {e}")
            return None
    
    def update_workflow_config(
        self, 
        workflow_id: int,
        config: Dict[str, Any],
        session: Optional[Session] = None
    ) -> Optional[Workflow]:
        """更新工作流配置"""
        try:
            workflow = self.get(workflow_id, session)
            if not workflow:
                raise EntityNotFoundError(f"Workflow with ID {workflow_id} not found")
            
            # 合并配置
            current_config = workflow.config or {}
            current_config.update(config)
            
            update_data = {
                "config": current_config
            }
            
            return self.update(workflow, update_data, session)
            
        except Exception as e:
            logger.error(f"Error updating workflow config for workflow {workflow_id}: {e}")
            return None
    
    def search_workflows(
        self, 
        session_id: int,
        query: str, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Workflow]:
        """搜索工作流"""
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
            QueryFilter().or_(
                QueryFilter().ilike("name", f"%{query}%"),
                QueryFilter().ilike("description", f"%{query}%")
            )
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("updated_at", "desc")],
            session=session
        )
    
    def get_workflows_by_date_range(
        self, 
        session_id: int,
        start_date: datetime,
        end_date: datetime,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Workflow]:
        """根据日期范围获取工作流"""
        filters = QueryFilter().and_(
            QueryFilter().eq("session_id", session_id),
            QueryFilter().gte("created_at", start_date),
            QueryFilter().lte("created_at", end_date)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_workflow_statistics(
        self, 
        session_id: Optional[int] = None,
        user_id: Optional[int] = None,
        session: Optional[Session] = None
    ) -> WorkflowStats:
        """获取工作流统计信息"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            base_filters = QueryFilter()
            
            if session_id:
                base_filters = base_filters.eq("session_id", session_id)
            elif user_id:
                # 通过session表连接查询用户的工作流
                query = db_session.query(Workflow).join(Workflow.session)
                query = query.filter(Workflow.session.has(user_id=user_id))
                
                total_workflows = query.count()
                running_workflows = query.filter(Workflow.status == WorkflowStatus.RUNNING).count()
                completed_workflows = query.filter(Workflow.status == WorkflowStatus.COMPLETED).count()
                failed_workflows = query.filter(Workflow.status == WorkflowStatus.FAILED).count()
                
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                today_workflows = query.filter(Workflow.created_at >= today_start).count()
                
                return WorkflowStats(
                    total_workflows=total_workflows,
                    running_workflows=running_workflows,
                    completed_workflows=completed_workflows,
                    failed_workflows=failed_workflows,
                    today_workflows=today_workflows
                )
            
            # 总工作流数
            total_workflows = self.count(filters=base_filters, session=db_session)
            
            # 按状态统计
            running_filters = base_filters.and_(QueryFilter().eq("status", WorkflowStatus.RUNNING))
            running_workflows = self.count(filters=running_filters, session=db_session)
            
            completed_filters = base_filters.and_(QueryFilter().eq("status", WorkflowStatus.COMPLETED))
            completed_workflows = self.count(filters=completed_filters, session=db_session)
            
            failed_filters = base_filters.and_(QueryFilter().eq("status", WorkflowStatus.FAILED))
            failed_workflows = self.count(filters=failed_filters, session=db_session)
            
            # 今日工作流数
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_filters = base_filters.and_(QueryFilter().gte("created_at", today_start))
            today_workflows = self.count(filters=today_filters, session=db_session)
            
            return WorkflowStats(
                total_workflows=total_workflows,
                running_workflows=running_workflows,
                completed_workflows=completed_workflows,
                failed_workflows=failed_workflows,
                today_workflows=today_workflows
            )
            
        except Exception as e:
            logger.error(f"Error getting workflow statistics: {e}")
            return WorkflowStats(
                total_workflows=0,
                running_workflows=0,
                completed_workflows=0,
                failed_workflows=0,
                today_workflows=0
            )
        finally:
            if not session:
                db_session.close()
    
    def get_long_running_workflows(
        self, 
        hours: int = 24,
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Workflow]:
        """获取长时间运行的工作流"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        filters = QueryFilter().and_(
            QueryFilter().eq("status", WorkflowStatus.RUNNING),
            QueryFilter().lt("started_at", cutoff_time)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("started_at", "asc")],
            session=session
        )
    
    def cleanup_old_workflows(
        self, 
        days: int = 90,
        session: Optional[Session] = None
    ) -> int:
        """清理旧的已完成工作流"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # 查找旧的已完成工作流
            filters = QueryFilter().and_(
                QueryFilter().in_("status", [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]),
                QueryFilter().lt("completed_at", cutoff_date)
            )
            
            old_workflows = self.get_multi(
                filters=filters,
                limit=1000,  # 批量处理
                session=session
            )
            
            # 软删除这些工作流
            deleted_count = 0
            for workflow in old_workflows:
                if self.soft_delete(workflow.id, session):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old workflows")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old workflows: {e}")
            return 0
    
    def get_workflow_execution_time(
        self, 
        workflow_id: int,
        session: Optional[Session] = None
    ) -> Optional[float]:
        """获取工作流执行时间（分钟）"""
        try:
            workflow = self.get(workflow_id, session)
            if not workflow or not workflow.started_at:
                return None
            
            end_time = workflow.completed_at or datetime.utcnow()
            execution_time = (end_time - workflow.started_at).total_seconds() / 60
            
            return round(execution_time, 2)
            
        except Exception as e:
            logger.error(f"Error getting execution time for workflow {workflow_id}: {e}")
            return None
    
    def get_average_execution_time(
        self, 
        workflow_type: Optional[WorkflowType] = None,
        session_id: Optional[int] = None,
        session: Optional[Session] = None
    ) -> Optional[float]:
        """获取平均执行时间（分钟）"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 查询已完成的工作流
            query = db_session.query(Workflow)
            query = query.filter(Workflow.status == WorkflowStatus.COMPLETED)
            query = query.filter(Workflow.started_at.isnot(None))
            query = query.filter(Workflow.completed_at.isnot(None))
            
            if workflow_type:
                query = query.filter(Workflow.workflow_type == workflow_type)
            
            if session_id:
                query = query.filter(Workflow.session_id == session_id)
            
            workflows = query.all()
            
            if not workflows:
                return None
            
            total_time = 0
            for workflow in workflows:
                execution_time = (workflow.completed_at - workflow.started_at).total_seconds() / 60
                total_time += execution_time
            
            average_time = total_time / len(workflows)
            return round(average_time, 2)
            
        except Exception as e:
            logger.error(f"Error getting average execution time: {e}")
            return None
        finally:
            if not session:
                db_session.close()
    
    def get_workflow_success_rate(
        self, 
        workflow_type: Optional[WorkflowType] = None,
        session_id: Optional[int] = None,
        days: int = 30,
        session: Optional[Session] = None
    ) -> float:
        """获取工作流成功率"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            base_filters = QueryFilter().and_(
                QueryFilter().gte("created_at", cutoff_date),
                QueryFilter().in_("status", [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED])
            )
            
            if workflow_type:
                base_filters = base_filters.and_(QueryFilter().eq("workflow_type", workflow_type))
            
            if session_id:
                base_filters = base_filters.and_(QueryFilter().eq("session_id", session_id))
            
            # 总完成数（成功+失败）
            total_completed = self.count(filters=base_filters, session=session)
            
            if total_completed == 0:
                return 0.0
            
            # 成功数
            success_filters = base_filters.and_(QueryFilter().eq("status", WorkflowStatus.COMPLETED))
            successful = self.count(filters=success_filters, session=session)
            
            success_rate = (successful / total_completed) * 100
            return round(success_rate, 2)
            
        except Exception as e:
            logger.error(f"Error getting workflow success rate: {e}")
            return 0.0