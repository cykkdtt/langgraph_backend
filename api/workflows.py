from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import json
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from models.workflow_models import (
    WorkflowCreateRequest, WorkflowUpdateRequest, WorkflowExecuteRequest,
    WorkflowSearchRequest, WorkflowInfo, WorkflowExecutionInfo,
    WorkflowCreateResponse, WorkflowUpdateResponse, WorkflowDeleteResponse,
    WorkflowListResponse, WorkflowExecuteResponse, WorkflowExecutionListResponse,
    WorkflowStatistics, WorkflowTemplate, WorkflowBatchOperation,
    WorkflowBatchOperationResponse, WorkflowStatus, ExecutionStatus,
    WorkflowType, ExecutionNodeStatus
)
from models.api_models import (
    BaseResponse, PaginatedResponse, ErrorDetail, ErrorCode
)
from models.auth_models import UserInfo
from core.database import get_async_session
# from core.workflows.engine import WorkflowEngine
# from core.workflows.executor import WorkflowExecutor
# from core.workflows.monitor import WorkflowMonitor
from core.logging import get_logger

router = APIRouter(prefix="/workflows", tags=["workflows"])
logger = get_logger(__name__)

# 依赖注入
def get_current_user() -> UserInfo:
    """获取当前用户信息"""
    # 这里应该从JWT token中解析用户信息
    from models.auth_models import UserRole, UserStatus
    return UserInfo(
        id=123,
        username="test_user",
        email="test@example.com",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
        is_active=True,
        is_admin=False,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

# 辅助函数
async def get_workflow_by_id(workflow_id: str, db: Session, user_id: str) -> Optional[Dict[str, Any]]:
    """根据ID获取工作流"""
    try:
        # 这里应该查询数据库获取工作流信息
        # 模拟数据库查询
        workflow_data = {
            "id": workflow_id,
            "name": "示例工作流",
            "description": "这是一个示例工作流",
            "type": WorkflowType.SEQUENTIAL,
            "status": WorkflowStatus.ACTIVE,
            "nodes": [],
            "edges": [],
            "triggers": [],
            "config": {},
            "tags": ["示例"],
            "is_public": False,
            "timeout": 3600,
            "max_retries": 3,
            "created_by": user_id,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "version": 1,
            "execution_count": 0,
            "success_count": 0,
            "last_execution": None
        }
        return workflow_data
    except Exception as e:
        logger.error(f"获取工作流失败: {e}")
        return None

async def save_workflow(workflow_data: Dict[str, Any], db: Session) -> bool:
    """保存工作流到数据库"""
    try:
        # 这里应该保存到数据库
        logger.info(f"保存工作流: {workflow_data['name']}")
        return True
    except Exception as e:
        logger.error(f"保存工作流失败: {e}")
        return False

async def delete_workflow_from_db(workflow_id: str, db: Session) -> bool:
    """从数据库删除工作流"""
    try:
        # 这里应该从数据库删除工作流（软删除）
        logger.info(f"删除工作流: {workflow_id}")
        return True
    except Exception as e:
        logger.error(f"删除工作流失败: {e}")
        return False

async def search_workflows(request: WorkflowSearchRequest, db: AsyncSession, user_id: str) -> tuple[List[Dict[str, Any]], int]:
    """搜索工作流"""
    try:
        # 这里应该根据搜索条件查询数据库
        # 模拟搜索结果
        workflows = [
            {
                "id": f"workflow_{i}",
                "name": f"工作流 {i}",
                "description": f"这是工作流 {i} 的描述",
                "type": WorkflowType.SEQUENTIAL,
                "status": WorkflowStatus.ACTIVE,
                "nodes": [],
                "edges": [],
                "triggers": [],
                "config": {},
                "tags": ["示例"],
                "is_public": False,
                "timeout": 3600,
                "max_retries": 3,
                "created_by": user_id,
                "created_at": datetime.now() - timedelta(days=i),
                "updated_at": datetime.now() - timedelta(days=i),
                "version": 1,
                "execution_count": i * 10,
                "success_count": i * 8,
                "last_execution": datetime.now() - timedelta(hours=i)
            }
            for i in range(1, min(request.page_size + 1, 6))
        ]
        total = 25  # 模拟总数
        return workflows, total
    except Exception as e:
        logger.error(f"搜索工作流失败: {e}")
        return [], 0

async def execute_workflow_async(workflow_id: str, request: WorkflowExecuteRequest, user_id: str, db: AsyncSession) -> str:
    """异步执行工作流"""
    execution_id = str(uuid.uuid4())
    try:
        # 获取工作流定义
        workflow = await get_workflow_by_id(workflow_id, db, user_id)
        if not workflow:
            raise Exception("工作流不存在")
        
        # 创建执行记录
        execution_data = {
            "id": execution_id,
            "workflow_id": workflow_id,
            "workflow_name": workflow["name"],
            "status": ExecutionStatus.RUNNING,
            "start_time": datetime.now(),
            "input_data": request.input_data,
            "created_by": user_id,
            "priority": request.priority,
            "metadata": request.metadata
        }
        
        # 这里应该保存执行记录到数据库
        logger.info(f"开始执行工作流: {workflow_id}, 执行ID: {execution_id}")
        
        # 模拟工作流执行
        await asyncio.sleep(2)  # 模拟执行时间
        
        # 更新执行状态
        execution_data.update({
            "status": ExecutionStatus.COMPLETED,
            "end_time": datetime.now(),
            "duration": 2.0,
            "output_data": {"result": "执行成功"},
            "progress": 100.0
        })
        
        logger.info(f"工作流执行完成: {execution_id}")
        return execution_id
        
    except Exception as e:
        logger.error(f"工作流执行失败: {e}")
        # 更新执行状态为失败
        return execution_id

async def get_execution_by_id(execution_id: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
    """根据ID获取执行信息"""
    try:
        # 这里应该查询数据库获取执行信息
        # 模拟执行信息
        execution_data = {
            "id": execution_id,
            "workflow_id": "workflow_123",
            "workflow_name": "示例工作流",
            "status": ExecutionStatus.COMPLETED,
            "start_time": datetime.now() - timedelta(minutes=5),
            "end_time": datetime.now(),
            "duration": 300.0,
            "input_data": {"input": "test"},
            "output_data": {"result": "success"},
            "error_message": None,
            "progress": 100.0,
            "node_statuses": [],
            "created_by": "user_123",
            "priority": 5,
            "metadata": {}
        }
        return execution_data
    except Exception as e:
        logger.error(f"获取执行信息失败: {e}")
        return None

async def get_workflow_statistics(workflow_id: str, db: AsyncSession) -> WorkflowStatistics:
    """获取工作流统计信息"""
    try:
        # 这里应该查询数据库计算统计信息
        # 模拟统计数据
        stats = WorkflowStatistics(
            total_executions=100,
            successful_executions=85,
            failed_executions=15,
            average_duration=120.5,
            success_rate=85.0,
            last_24h_executions=10,
            peak_execution_time="14:00-15:00",
            most_failed_node="node_3"
        )
        return stats
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return WorkflowStatistics()

# API路由
@router.post("/", response_model=BaseResponse[WorkflowInfo])
async def create_workflow(
    request: WorkflowCreateRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """创建工作流"""
    try:
        workflow_id = str(uuid.uuid4())
        workflow_data = {
            "id": workflow_id,
            "name": request.name,
            "description": request.description,
            "type": request.type,
            "status": WorkflowStatus.DRAFT,
            "nodes": [node.dict() for node in request.nodes],
            "edges": [edge.dict() for edge in request.edges],
            "triggers": [trigger.dict() for trigger in request.triggers],
            "config": request.config,
            "tags": request.tags,
            "is_public": request.is_public,
            "timeout": request.timeout,
            "max_retries": request.max_retries,
            "created_by": current_user.id,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "version": 1,
            "execution_count": 0,
            "success_count": 0,
            "last_execution": None
        }
        
        success = await save_workflow(workflow_data, db)
        if not success:
            return BaseResponse.error(
                message="创建工作流失败",
                errors=[ErrorDetail(
                    code=ErrorCode.INTERNAL_SERVER_ERROR,
                    message="保存工作流到数据库失败"
                )]
            )
        
        workflow_info = WorkflowInfo(**workflow_data)
        return BaseResponse.success(
            data=workflow_info,
            message="工作流创建成功"
        )
        
    except Exception as e:
        logger.error(f"创建工作流失败: {e}")
        return BaseResponse.error(
            message="创建工作流失败",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )

@router.get("/", response_model=PaginatedResponse[WorkflowInfo])
async def list_workflows(
    query: Optional[str] = Query(None, description="搜索关键词"),
    type: Optional[WorkflowType] = Query(None, description="工作流类型"),
    status: Optional[WorkflowStatus] = Query(None, description="工作流状态"),
    tags: Optional[str] = Query(None, description="标签（逗号分隔）"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    sort_by: str = Query("created_at", description="排序字段"),
    sort_order: str = Query("desc", description="排序方向"),
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """获取工作流列表"""
    try:
        search_request = WorkflowSearchRequest(
            query=query,
            type=type,
            status=status,
            tags=tags.split(",") if tags else None,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        workflows_data, total = await search_workflows(search_request, db, current_user.id)
        workflows = [WorkflowInfo(**data) for data in workflows_data]
        
        from models.api_models import PaginationInfo
        pagination = PaginationInfo.create(page, page_size, total)
        
        return PaginatedResponse.success(
            data=workflows,
            pagination=pagination,
            message="获取工作流列表成功"
        )
        
    except Exception as e:
        logger.error(f"获取工作流列表失败: {e}")
        from models.api_models import PaginationInfo
        pagination = PaginationInfo.create(page, page_size, 0)
        return PaginatedResponse(
            status="error",
            message="获取工作流列表失败",
            data=[],
            pagination=pagination,
            total_count=0,
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message=str(e)
            )]
        )

@router.get("/{workflow_id}", response_model=BaseResponse[WorkflowInfo])
async def get_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """获取工作流详情"""
    try:
        workflow_data = await get_workflow_by_id(workflow_id, db, current_user.id)
        if not workflow_data:
            return BaseResponse.error(
                message="工作流不存在",
                errors=[ErrorDetail(
                    code=ErrorCode.RESOURCE_NOT_FOUND,
                    message="指定的工作流不存在或您没有访问权限"
                )]
            )
        
        workflow_info = WorkflowInfo(**workflow_data)
        return BaseResponse.success(
            data=workflow_info,
            message="获取工作流详情成功"
        )
        
    except Exception as e:
        logger.error(f"获取工作流详情失败: {e}")
        return BaseResponse.error(
            message="获取工作流详情失败",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )

@router.put("/{workflow_id}", response_model=WorkflowUpdateResponse)
async def update_workflow(
    workflow_id: str,
    request: WorkflowUpdateRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """更新工作流"""
    try:
        workflow_data = await get_workflow_by_id(workflow_id, db, current_user.id)
        if not workflow_data:
            raise HTTPException(status_code=404, detail="工作流不存在")
        
        # 更新字段
        update_data = request.dict(exclude_unset=True)
        workflow_data.update(update_data)
        workflow_data["updated_at"] = datetime.now()
        workflow_data["version"] += 1
        
        success = await save_workflow(workflow_data, db)
        if not success:
            raise HTTPException(status_code=500, detail="更新工作流失败")
        
        workflow_info = WorkflowInfo(**workflow_data)
        return WorkflowUpdateResponse(
            success=True,
            message="工作流更新成功",
            workflow=workflow_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新工作流失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{workflow_id}", response_model=WorkflowDeleteResponse)
async def delete_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """删除工作流"""
    try:
        workflow_data = await get_workflow_by_id(workflow_id, db, current_user.id)
        if not workflow_data:
            raise HTTPException(status_code=404, detail="工作流不存在")
        
        success = await delete_workflow_from_db(workflow_id, db)
        if not success:
            raise HTTPException(status_code=500, detail="删除工作流失败")
        
        return WorkflowDeleteResponse(
            success=True,
            message="工作流删除成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除工作流失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{workflow_id}/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(
    workflow_id: str,
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """执行工作流"""
    try:
        workflow_data = await get_workflow_by_id(workflow_id, db, current_user.id)
        if not workflow_data:
            raise HTTPException(status_code=404, detail="工作流不存在")
        
        if workflow_data["status"] != WorkflowStatus.ACTIVE:
            raise HTTPException(status_code=400, detail="工作流未激活")
        
        # 启动后台任务执行工作流
        execution_id = str(uuid.uuid4())
        background_tasks.add_task(
            execute_workflow_async,
            workflow_id,
            request,
            current_user.id,
            db
        )
        
        # 创建初始执行信息
        execution_info = WorkflowExecutionInfo(
            id=execution_id,
            workflow_id=workflow_id,
            workflow_name=workflow_data["name"],
            status=ExecutionStatus.PENDING,
            start_time=datetime.now(),
            input_data=request.input_data,
            created_by=current_user.id,
            priority=request.priority,
            metadata=request.metadata
        )
        
        return WorkflowExecuteResponse(
            success=True,
            message="工作流执行已启动",
            execution=execution_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"执行工作流失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{workflow_id}/executions", response_model=WorkflowExecutionListResponse)
async def list_workflow_executions(
    workflow_id: str,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    status: Optional[ExecutionStatus] = Query(None, description="执行状态"),
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """获取工作流执行列表"""
    try:
        # 这里应该查询数据库获取执行列表
        # 模拟执行列表
        executions_data = [
            {
                "id": f"exec_{i}",
                "workflow_id": workflow_id,
                "workflow_name": "示例工作流",
                "status": ExecutionStatus.COMPLETED if i % 2 == 0 else ExecutionStatus.FAILED,
                "start_time": datetime.now() - timedelta(hours=i),
                "end_time": datetime.now() - timedelta(hours=i-1),
                "duration": 3600.0,
                "input_data": {"input": f"test_{i}"},
                "output_data": {"result": f"result_{i}"},
                "error_message": None if i % 2 == 0 else "执行失败",
                "progress": 100.0,
                "node_statuses": [],
                "created_by": current_user.id,
                "priority": 5,
                "metadata": {}
            }
            for i in range(1, min(page_size + 1, 6))
        ]
        
        executions = [WorkflowExecutionInfo(**data) for data in executions_data]
        total = 25  # 模拟总数
        
        return WorkflowExecutionListResponse(
            executions=executions,
            total=total,
            page=page,
            page_size=page_size,
            has_next=page * page_size < total
        )
        
    except Exception as e:
        logger.error(f"获取执行列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/executions/{execution_id}", response_model=WorkflowExecutionInfo)
async def get_execution(
    execution_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """获取执行详情"""
    try:
        execution_data = await get_execution_by_id(execution_id, db)
        if not execution_data:
            raise HTTPException(status_code=404, detail="执行记录不存在")
        
        return WorkflowExecutionInfo(**execution_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取执行详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(
    execution_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """取消执行"""
    try:
        execution_data = await get_execution_by_id(execution_id, db)
        if not execution_data:
            raise HTTPException(status_code=404, detail="执行记录不存在")
        
        if execution_data["status"] not in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
            raise HTTPException(status_code=400, detail="执行已完成，无法取消")
        
        # 这里应该取消执行
        logger.info(f"取消执行: {execution_id}")
        
        return {"success": True, "message": "执行已取消"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{workflow_id}/statistics", response_model=WorkflowStatistics)
async def get_workflow_stats(
    workflow_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """获取工作流统计信息"""
    try:
        workflow_data = await get_workflow_by_id(workflow_id, db, current_user.id)
        if not workflow_data:
            raise HTTPException(status_code=404, detail="工作流不存在")
        
        stats = await get_workflow_statistics(workflow_id, db)
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/", response_model=List[WorkflowTemplate])
async def list_workflow_templates(
    category: Optional[str] = Query(None, description="模板分类"),
    featured: Optional[bool] = Query(None, description="是否推荐"),
    db: AsyncSession = Depends(get_async_session)
):
    """获取工作流模板列表"""
    try:
        # 这里应该查询数据库获取模板列表
        # 模拟模板数据
        templates = [
            WorkflowTemplate(
                id=f"template_{i}",
                name=f"模板 {i}",
                description=f"这是模板 {i} 的描述",
                category="通用",
                type=WorkflowType.SEQUENTIAL,
                nodes=[],
                edges=[],
                config={},
                tags=["模板"],
                is_featured=i <= 2,
                usage_count=i * 10,
                rating=4.5,
                created_by="system",
                created_at=datetime.now() - timedelta(days=i)
            )
            for i in range(1, 6)
        ]
        
        return templates
        
    except Exception as e:
        logger.error(f"获取模板列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=WorkflowBatchOperationResponse)
async def batch_operation(
    request: WorkflowBatchOperation,
    db: AsyncSession = Depends(get_async_session),
    current_user: UserInfo = Depends(get_current_user)
):
    """批量操作工作流"""
    try:
        results = []
        failed_ids = []
        
        for workflow_id in request.workflow_ids:
            try:
                workflow_data = await get_workflow_by_id(workflow_id, db, current_user.id)
                if not workflow_data:
                    failed_ids.append(workflow_id)
                    continue
                
                # 根据操作类型执行相应操作
                if request.operation == "delete":
                    success = await delete_workflow_from_db(workflow_id, db)
                elif request.operation == "activate":
                    workflow_data["status"] = WorkflowStatus.ACTIVE
                    success = await save_workflow(workflow_data, db)
                elif request.operation == "deactivate":
                    workflow_data["status"] = WorkflowStatus.PAUSED
                    success = await save_workflow(workflow_data, db)
                else:
                    success = False
                
                if success:
                    results.append({"id": workflow_id, "status": "success"})
                else:
                    failed_ids.append(workflow_id)
                    
            except Exception as e:
                logger.error(f"批量操作失败 {workflow_id}: {e}")
                failed_ids.append(workflow_id)
        
        return WorkflowBatchOperationResponse(
            success=len(failed_ids) == 0,
            message=f"批量操作完成，成功: {len(results)}, 失败: {len(failed_ids)}",
            results=results,
            failed_ids=failed_ids
        )
        
    except Exception as e:
        logger.error(f"批量操作失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))