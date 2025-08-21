"""对话线程管理API实现

本模块实现了LangGraph多智能体系统的对话线程管理功能，包括：
- 线程创建、更新、删除
- 线程搜索和过滤
- 线程归档和恢复
- 批量操作
- 线程分析和导出
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_, text
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta
import uuid
import json
import csv
import io
import os
from pathlib import Path

from models.thread_models import (
    ThreadCreateRequest, ThreadUpdateRequest, ThreadSearchRequest,
    ThreadInfo, ThreadListResponse, ThreadCreateResponse, ThreadUpdateResponse,
    ThreadDeleteResponse, ThreadArchiveRequest, ThreadArchiveResponse,
    ThreadRestoreRequest, ThreadRestoreResponse, ThreadBatchOperation,
    ThreadBatchOperationResponse, ThreadExportRequest, ThreadExportResponse,
    ThreadAnalyticsRequest, ThreadAnalyticsResponse, ThreadAnalyticsData,
    ThreadStatus, ThreadPriority, ThreadType, ThreadSortBy, SortOrder,
    ThreadMessageSummary, ThreadParticipant, ThreadStatistics
)
from models.database_models import Session as DBSession, Message, User
from models.auth_models import UserInfo
from models.chat_models import MessageRole, MessageType
from api.auth import get_current_user
from core.database import get_async_session
from core.logging import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/threads", tags=["Thread Management"])


# 辅助函数
async def get_thread_by_id(
    db: AsyncSession, 
    thread_id: str, 
    user_id: str,
    check_access: bool = True
) -> DBSession:
    """根据ID获取线程"""
    query = select(DBSession).where(DBSession.id == thread_id)
    
    if check_access:
        query = query.where(DBSession.user_id == user_id)
    
    result = await db.execute(query)
    thread = result.scalar_one_or_none()
    
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="线程不存在或无访问权限"
        )
    
    return thread


async def calculate_thread_statistics(
    db: AsyncSession, 
    thread_id: str
) -> ThreadStatistics:
    """计算线程统计信息"""
    # 获取消息统计
    message_stats_query = select(
        func.count(Message.id).label('total_messages'),
        func.sum(func.case((Message.role == MessageRole.USER, 1), else_=0)).label('user_messages'),
        func.sum(func.case((Message.role == MessageRole.ASSISTANT, 1), else_=0)).label('agent_messages'),
        func.sum(func.case((Message.message_type == MessageType.TOOL_CALL, 1), else_=0)).label('tool_calls'),
        func.min(Message.created_at).label('first_message_at'),
        func.max(Message.created_at).label('last_message_at')
    ).where(Message.session_id == thread_id)
    
    result = await db.execute(message_stats_query)
    stats = result.first()
    
    # 获取参与者数量
    participants_query = select(func.count(func.distinct(Message.user_id))).where(
        Message.session_id == thread_id
    )
    participants_result = await db.execute(participants_query)
    participants_count = participants_result.scalar() or 0
    
    # 计算活跃时长
    active_duration = None
    if stats.first_message_at and stats.last_message_at:
        active_duration = int((stats.last_message_at - stats.first_message_at).total_seconds())
    
    return ThreadStatistics(
        total_messages=stats.total_messages or 0,
        user_messages=stats.user_messages or 0,
        agent_messages=stats.agent_messages or 0,
        tool_calls=stats.tool_calls or 0,
        participants_count=participants_count,
        first_message_at=stats.first_message_at,
        last_message_at=stats.last_message_at,
        active_duration=active_duration
    )


async def get_thread_participants(
    db: AsyncSession, 
    thread_id: str
) -> List[ThreadParticipant]:
    """获取线程参与者"""
    query = select(
        User.id,
        User.username,
        func.min(Message.created_at).label('joined_at'),
        func.max(Message.created_at).label('last_activity'),
        func.count(Message.id).label('message_count')
    ).select_from(
        Message
    ).join(
        User, Message.user_id == User.id
    ).where(
        Message.session_id == thread_id
    ).group_by(
        User.id, User.username
    )
    
    result = await db.execute(query)
    participants = []
    
    for row in result:
        participants.append(ThreadParticipant(
            user_id=row.id,
            username=row.username,
            role="participant",  # 简化处理，实际可以根据业务需求设置
            joined_at=row.joined_at,
            last_activity=row.last_activity,
            message_count=row.message_count
        ))
    
    return participants


async def get_recent_messages(
    db: AsyncSession, 
    thread_id: str, 
    limit: int = 5
) -> List[ThreadMessageSummary]:
    """获取最近消息"""
    query = select(Message).where(
        Message.session_id == thread_id
    ).order_by(
        Message.created_at.desc()
    ).limit(limit)
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    return [
        ThreadMessageSummary(
            message_id=str(msg.id),
            role=msg.role,
            content=msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
            message_type=msg.message_type,
            created_at=msg.created_at,
            agent_type=msg.metadata.get('agent_type') if msg.metadata else None
        )
        for msg in messages
    ]


async def convert_db_session_to_thread_info(
    db: AsyncSession, 
    session: DBSession
) -> ThreadInfo:
    """将数据库会话转换为线程信息"""
    # 获取创建者信息
    user_query = select(User).where(User.id == session.user_id)
    user_result = await db.execute(user_query)
    user = user_result.scalar_one_or_none()
    
    # 计算统计信息
    statistics = await calculate_thread_statistics(db, str(session.id))
    
    # 获取参与者
    participants = await get_thread_participants(db, str(session.id))
    
    # 获取最近消息
    recent_messages = await get_recent_messages(db, str(session.id))
    
    return ThreadInfo(
        thread_id=str(session.id),
        title=session.title or f"会话 {session.id}",
        description=session.metadata.get('description') if session.metadata else None,
        thread_type=ThreadType(session.metadata.get('thread_type', 'chat')) if session.metadata else ThreadType.CHAT,
        status=ThreadStatus(session.status) if hasattr(ThreadStatus, session.status.upper()) else ThreadStatus.ACTIVE,
        priority=ThreadPriority(session.metadata.get('priority', 'normal')) if session.metadata else ThreadPriority.NORMAL,
        mode=session.mode,
        owner_id=str(session.user_id),
        owner_username=user.username if user else "Unknown",
        tags=session.metadata.get('tags', []) if session.metadata else [],
        metadata=session.metadata or {},
        created_at=session.created_at,
        updated_at=session.updated_at,
        last_activity=statistics.last_message_at,
        statistics=statistics,
        participants=participants,
        recent_messages=recent_messages
    )


# API端点
@router.post("/", response_model=ThreadCreateResponse)
async def create_thread(
    request: ThreadCreateRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """创建新线程"""
    try:
        # 创建会话记录
        session_data = {
            "id": uuid.uuid4(),
            "user_id": current_user.user_id,
            "title": request.title,
            "mode": request.mode,
            "status": "active",
            "metadata": {
                "description": request.description,
                "thread_type": request.thread_type,
                "priority": request.priority,
                "tags": request.tags,
                "agent_type": request.agent_type,
                **request.metadata
            },
            "context": request.context,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        new_session = DBSession(**session_data)
        db.add(new_session)
        
        # 如果有初始消息，创建消息记录
        if request.initial_message:
            initial_message = Message(
                id=uuid.uuid4(),
                session_id=new_session.id,
                user_id=current_user.user_id,
                role=MessageRole.USER,
                content=request.initial_message,
                message_type=MessageType.TEXT,
                created_at=datetime.utcnow()
            )
            db.add(initial_message)
        
        await db.commit()
        await db.refresh(new_session)
        
        logger.info(f"线程创建成功: {new_session.id}, 用户: {current_user.user_id}")
        
        return ThreadCreateResponse(
            thread_id=str(new_session.id),
            title=new_session.title,
            status=ThreadStatus.ACTIVE,
            created_at=new_session.created_at
        )
        
    except Exception as e:
        await db.rollback()
        logger.error(f"创建线程失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建线程失败: {str(e)}"
        )


@router.get("/", response_model=ThreadListResponse)
async def list_threads(
    search: Optional[str] = Query(None, description="搜索关键词"),
    status_filter: Optional[List[ThreadStatus]] = Query(None, description="状态过滤"),
    thread_type: Optional[List[ThreadType]] = Query(None, description="类型过滤"),
    priority: Optional[List[ThreadPriority]] = Query(None, description="优先级过滤"),
    tags: Optional[List[str]] = Query(None, description="标签过滤"),
    sort_by: ThreadSortBy = Query(ThreadSortBy.UPDATED_AT, description="排序字段"),
    sort_order: SortOrder = Query(SortOrder.DESC, description="排序方式"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """获取线程列表"""
    try:
        # 构建查询
        query = select(DBSession).where(DBSession.user_id == current_user.user_id)
        
        # 搜索过滤
        if search:
            search_pattern = f"%{search}%"
            query = query.where(
                or_(
                    DBSession.title.ilike(search_pattern),
                    func.jsonb_extract_path_text(DBSession.metadata, 'description').ilike(search_pattern)
                )
            )
        
        # 状态过滤
        if status_filter:
            status_values = [s.value for s in status_filter]
            query = query.where(DBSession.status.in_(status_values))
        
        # 类型过滤
        if thread_type:
            type_values = [t.value for t in thread_type]
            query = query.where(
                func.jsonb_extract_path_text(DBSession.metadata, 'thread_type').in_(type_values)
            )
        
        # 优先级过滤
        if priority:
            priority_values = [p.value for p in priority]
            query = query.where(
                func.jsonb_extract_path_text(DBSession.metadata, 'priority').in_(priority_values)
            )
        
        # 标签过滤
        if tags:
            for tag in tags:
                query = query.where(
                    func.jsonb_path_exists(
                        DBSession.metadata,
                        f'$.tags[*] ? (@ == "{tag}")'
                    )
                )
        
        # 排序
        if sort_by == ThreadSortBy.CREATED_AT:
            order_column = DBSession.created_at
        elif sort_by == ThreadSortBy.UPDATED_AT:
            order_column = DBSession.updated_at
        elif sort_by == ThreadSortBy.TITLE:
            order_column = DBSession.title
        else:
            order_column = DBSession.updated_at
        
        if sort_order == SortOrder.DESC:
            query = query.order_by(order_column.desc())
        else:
            query = query.order_by(order_column.asc())
        
        # 计算总数
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # 分页
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        # 执行查询
        result = await db.execute(query)
        sessions = result.scalars().all()
        
        # 转换为线程信息
        threads = []
        for session in sessions:
            thread_info = await convert_db_session_to_thread_info(db, session)
            threads.append(thread_info)
        
        # 计算分页信息
        total_pages = (total + page_size - 1) // page_size
        has_next = page < total_pages
        has_prev = page > 1
        
        return ThreadListResponse(
            threads=threads,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev
        )
        
    except Exception as e:
        logger.error(f"获取线程列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取线程列表失败: {str(e)}"
        )


@router.get("/{thread_id}", response_model=ThreadInfo)
async def get_thread(
    thread_id: str,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """获取线程详情"""
    try:
        session = await get_thread_by_id(db, thread_id, current_user.user_id)
        thread_info = await convert_db_session_to_thread_info(db, session)
        return thread_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取线程详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取线程详情失败: {str(e)}"
        )


@router.put("/{thread_id}", response_model=ThreadUpdateResponse)
async def update_thread(
    thread_id: str,
    request: ThreadUpdateRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """更新线程"""
    try:
        session = await get_thread_by_id(db, thread_id, current_user.user_id)
        
        updated_fields = []
        
        # 更新基本字段
        if request.title is not None:
            session.title = request.title
            updated_fields.append("title")
        
        if request.status is not None:
            session.status = request.status.value
            updated_fields.append("status")
        
        # 更新元数据
        if session.metadata is None:
            session.metadata = {}
        
        if request.description is not None:
            session.metadata["description"] = request.description
            updated_fields.append("description")
        
        if request.priority is not None:
            session.metadata["priority"] = request.priority.value
            updated_fields.append("priority")
        
        if request.tags is not None:
            session.metadata["tags"] = request.tags
            updated_fields.append("tags")
        
        if request.metadata is not None:
            session.metadata.update(request.metadata)
            updated_fields.append("metadata")
        
        session.updated_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info(f"线程更新成功: {thread_id}, 字段: {updated_fields}")
        
        return ThreadUpdateResponse(
            thread_id=thread_id,
            updated_fields=updated_fields,
            updated_at=session.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"更新线程失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新线程失败: {str(e)}"
        )


@router.delete("/{thread_id}", response_model=ThreadDeleteResponse)
async def delete_thread(
    thread_id: str,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """删除线程"""
    try:
        session = await get_thread_by_id(db, thread_id, current_user.user_id)
        
        # 软删除：更新状态为已删除
        session.status = ThreadStatus.DELETED.value
        session.updated_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info(f"线程删除成功: {thread_id}")
        
        return ThreadDeleteResponse(
            thread_id=thread_id,
            deleted_at=session.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"删除线程失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除线程失败: {str(e)}"
        )


@router.post("/{thread_id}/archive", response_model=ThreadArchiveResponse)
async def archive_thread(
    thread_id: str,
    request: ThreadArchiveRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """归档线程"""
    try:
        session = await get_thread_by_id(db, thread_id, current_user.user_id)
        
        # 更新状态为已归档
        session.status = ThreadStatus.ARCHIVED.value
        session.updated_at = datetime.utcnow()
        
        # 添加归档信息到元数据
        if session.metadata is None:
            session.metadata = {}
        
        session.metadata["archived_at"] = session.updated_at.isoformat()
        session.metadata["archive_reason"] = request.reason
        session.metadata["archived_by"] = current_user.user_id
        
        # 计算归档的消息数量
        if request.archive_messages:
            message_count_query = select(func.count(Message.id)).where(
                Message.session_id == thread_id
            )
            result = await db.execute(message_count_query)
            archived_messages_count = result.scalar() or 0
        else:
            archived_messages_count = 0
        
        await db.commit()
        
        logger.info(f"线程归档成功: {thread_id}")
        
        return ThreadArchiveResponse(
            thread_id=thread_id,
            archived_at=session.updated_at,
            archived_messages_count=archived_messages_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"归档线程失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"归档线程失败: {str(e)}"
        )


@router.post("/{thread_id}/restore", response_model=ThreadRestoreResponse)
async def restore_thread(
    thread_id: str,
    request: ThreadRestoreRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """恢复线程"""
    try:
        session = await get_thread_by_id(db, thread_id, current_user.user_id, check_access=False)
        
        # 检查是否可以恢复
        if session.status not in [ThreadStatus.ARCHIVED.value, ThreadStatus.DELETED.value]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="只能恢复已归档或已删除的线程"
            )
        
        # 恢复状态为活跃
        session.status = ThreadStatus.ACTIVE.value
        session.updated_at = datetime.utcnow()
        
        # 添加恢复信息到元数据
        if session.metadata is None:
            session.metadata = {}
        
        session.metadata["restored_at"] = session.updated_at.isoformat()
        session.metadata["restore_reason"] = request.reason
        session.metadata["restored_by"] = current_user.user_id
        
        # 计算恢复的消息数量
        if request.restore_messages:
            message_count_query = select(func.count(Message.id)).where(
                Message.session_id == thread_id
            )
            result = await db.execute(message_count_query)
            restored_messages_count = result.scalar() or 0
        else:
            restored_messages_count = 0
        
        await db.commit()
        
        logger.info(f"线程恢复成功: {thread_id}")
        
        return ThreadRestoreResponse(
            thread_id=thread_id,
            restored_at=session.updated_at,
            restored_messages_count=restored_messages_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"恢复线程失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"恢复线程失败: {str(e)}"
        )


@router.post("/batch", response_model=ThreadBatchOperationResponse)
async def batch_operation(
    request: ThreadBatchOperation,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """批量操作线程"""
    try:
        successful = 0
        failed = 0
        failed_thread_ids = []
        errors = []
        
        for thread_id in request.thread_ids:
            try:
                session = await get_thread_by_id(db, thread_id, current_user.user_id, check_access=False)
                
                if request.operation == "delete":
                    session.status = ThreadStatus.DELETED.value
                elif request.operation == "archive":
                    session.status = ThreadStatus.ARCHIVED.value
                elif request.operation == "restore":
                    session.status = ThreadStatus.ACTIVE.value
                elif request.operation == "update_status":
                    new_status = request.parameters.get("status")
                    if new_status:
                        session.status = new_status
                
                session.updated_at = datetime.utcnow()
                successful += 1
                
            except Exception as e:
                failed += 1
                failed_thread_ids.append(thread_id)
                errors.append(f"线程 {thread_id}: {str(e)}")
        
        await db.commit()
        
        logger.info(f"批量操作完成: {request.operation}, 成功: {successful}, 失败: {failed}")
        
        return ThreadBatchOperationResponse(
            operation=request.operation,
            total_requested=len(request.thread_ids),
            successful=successful,
            failed=failed,
            failed_thread_ids=failed_thread_ids,
            errors=errors,
            processed_at=datetime.utcnow()
        )
        
    except Exception as e:
        await db.rollback()
        logger.error(f"批量操作失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量操作失败: {str(e)}"
        )


@router.post("/export", response_model=ThreadExportResponse)
async def export_threads(
    request: ThreadExportRequest,
    background_tasks: BackgroundTasks,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """导出线程数据"""
    try:
        # 构建查询
        query = select(DBSession).where(DBSession.user_id == current_user.user_id)
        
        # 应用过滤条件
        if request.thread_ids:
            query = query.where(DBSession.id.in_(request.thread_ids))
        
        if request.date_range:
            if request.date_range.start_date:
                query = query.where(DBSession.created_at >= request.date_range.start_date)
            if request.date_range.end_date:
                query = query.where(DBSession.created_at <= request.date_range.end_date)
        
        if request.status_filter:
            status_values = [s.value for s in request.status_filter]
            query = query.where(DBSession.status.in_(status_values))
        
        # 执行查询
        result = await db.execute(query)
        sessions = result.scalars().all()
        
        # 创建导出文件
        export_id = str(uuid.uuid4())
        export_dir = Path(settings.EXPORT_DIR) if hasattr(settings, 'EXPORT_DIR') else Path("/tmp/exports")
        export_dir.mkdir(exist_ok=True)
        
        if request.format == "json":
            file_path = export_dir / f"threads_export_{export_id}.json"
            await export_to_json(db, sessions, file_path, request.include_messages)
        elif request.format == "csv":
            file_path = export_dir / f"threads_export_{export_id}.csv"
            await export_to_csv(db, sessions, file_path, request.include_messages)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不支持的导出格式"
            )
        
        # 设置文件清理任务
        background_tasks.add_task(cleanup_export_file, file_path, delay_hours=24)
        
        logger.info(f"线程导出成功: {export_id}, 格式: {request.format}, 数量: {len(sessions)}")
        
        return ThreadExportResponse(
            export_id=export_id,
            file_path=str(file_path),
            format=request.format,
            thread_count=len(sessions),
            file_size=file_path.stat().st_size,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        
    except Exception as e:
        logger.error(f"导出线程失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出线程失败: {str(e)}"
        )


@router.get("/export/{export_id}/download")
async def download_export(
    export_id: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """下载导出文件"""
    try:
        export_dir = Path(settings.EXPORT_DIR) if hasattr(settings, 'EXPORT_DIR') else Path("/tmp/exports")
        
        # 查找匹配的文件
        json_file = export_dir / f"threads_export_{export_id}.json"
        csv_file = export_dir / f"threads_export_{export_id}.csv"
        
        if json_file.exists():
            return FileResponse(
                path=str(json_file),
                filename=f"threads_export_{export_id}.json",
                media_type="application/json"
            )
        elif csv_file.exists():
            return FileResponse(
                path=str(csv_file),
                filename=f"threads_export_{export_id}.csv",
                media_type="text/csv"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="导出文件不存在或已过期"
            )
            
    except Exception as e:
        logger.error(f"下载导出文件失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"下载导出文件失败: {str(e)}"
        )


@router.post("/analytics", response_model=ThreadAnalyticsResponse)
async def get_thread_analytics(
    request: ThreadAnalyticsRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """获取线程分析数据"""
    try:
        # 构建基础查询
        base_query = select(DBSession).where(DBSession.user_id == current_user.user_id)
        
        # 应用时间范围过滤
        if request.date_range:
            if request.date_range.start_date:
                base_query = base_query.where(DBSession.created_at >= request.date_range.start_date)
            if request.date_range.end_date:
                base_query = base_query.where(DBSession.created_at <= request.date_range.end_date)
        
        # 线程总数统计
        total_threads_query = select(func.count(DBSession.id)).select_from(base_query.subquery())
        total_threads_result = await db.execute(total_threads_query)
        total_threads = total_threads_result.scalar() or 0
        
        # 按状态统计
        status_stats_query = select(
            DBSession.status,
            func.count(DBSession.id).label('count')
        ).select_from(
            base_query.subquery()
        ).group_by(DBSession.status)
        
        status_stats_result = await db.execute(status_stats_query)
        status_distribution = {row.status: row.count for row in status_stats_result}
        
        # 按类型统计
        type_stats_query = select(
            func.jsonb_extract_path_text(DBSession.metadata, 'thread_type').label('thread_type'),
            func.count(DBSession.id).label('count')
        ).select_from(
            base_query.subquery()
        ).group_by(
            func.jsonb_extract_path_text(DBSession.metadata, 'thread_type')
        )
        
        type_stats_result = await db.execute(type_stats_query)
        type_distribution = {row.thread_type or 'chat': row.count for row in type_stats_result}
        
        # 消息统计
        message_stats_query = select(
            func.count(Message.id).label('total_messages'),
            func.avg(func.count(Message.id)).over().label('avg_messages_per_thread')
        ).select_from(
            Message
        ).join(
            DBSession, Message.session_id == DBSession.id
        ).where(
            DBSession.user_id == current_user.user_id
        )
        
        if request.date_range:
            if request.date_range.start_date:
                message_stats_query = message_stats_query.where(DBSession.created_at >= request.date_range.start_date)
            if request.date_range.end_date:
                message_stats_query = message_stats_query.where(DBSession.created_at <= request.date_range.end_date)
        
        message_stats_result = await db.execute(message_stats_query)
        message_stats = message_stats_result.first()
        
        total_messages = message_stats.total_messages if message_stats else 0
        avg_messages_per_thread = float(message_stats.avg_messages_per_thread) if message_stats and message_stats.avg_messages_per_thread else 0.0
        
        # 活跃度统计（按天）
        if request.metrics and 'daily_activity' in request.metrics:
            daily_activity_query = select(
                func.date(DBSession.created_at).label('date'),
                func.count(DBSession.id).label('threads_created')
            ).select_from(
                base_query.subquery()
            ).group_by(
                func.date(DBSession.created_at)
            ).order_by(
                func.date(DBSession.created_at)
            )
            
            daily_activity_result = await db.execute(daily_activity_query)
            daily_activity = [
                {"date": row.date.isoformat(), "threads_created": row.threads_created}
                for row in daily_activity_result
            ]
        else:
            daily_activity = []
        
        # 热门标签统计
        if request.metrics and 'popular_tags' in request.metrics:
            # 这里需要使用PostgreSQL的JSON函数来展开标签数组
            popular_tags_query = text("""
                SELECT tag, COUNT(*) as usage_count
                FROM (
                    SELECT jsonb_array_elements_text(metadata->'tags') as tag
                    FROM sessions
                    WHERE user_id = :user_id
                    AND metadata ? 'tags'
                ) tags_expanded
                GROUP BY tag
                ORDER BY usage_count DESC
                LIMIT 10
            """)
            
            popular_tags_result = await db.execute(popular_tags_query, {"user_id": current_user.user_id})
            popular_tags = [
                {"tag": row.tag, "usage_count": row.usage_count}
                for row in popular_tags_result
            ]
        else:
            popular_tags = []
        
        analytics_data = ThreadAnalyticsData(
            total_threads=total_threads,
            total_messages=total_messages,
            avg_messages_per_thread=avg_messages_per_thread,
            status_distribution=status_distribution,
            type_distribution=type_distribution,
            daily_activity=daily_activity,
            popular_tags=popular_tags
        )
        
        logger.info(f"线程分析数据生成成功，用户: {current_user.user_id}")
        
        return ThreadAnalyticsResponse(
            analytics=analytics_data,
            generated_at=datetime.utcnow(),
            date_range=request.date_range
        )
        
    except Exception as e:
        logger.error(f"获取线程分析数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取线程分析数据失败: {str(e)}"
        )


# 辅助函数：导出功能
async def export_to_json(
    db: AsyncSession,
    sessions: List[DBSession],
    file_path: Path,
    include_messages: bool = False
):
    """导出为JSON格式"""
    export_data = {
        "export_info": {
            "created_at": datetime.utcnow().isoformat(),
            "thread_count": len(sessions),
            "include_messages": include_messages
        },
        "threads": []
    }
    
    for session in sessions:
        thread_data = {
            "thread_id": str(session.id),
            "title": session.title,
            "mode": session.mode,
            "status": session.status,
            "metadata": session.metadata,
            "context": session.context,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        }
        
        if include_messages:
            # 获取消息
            messages_query = select(Message).where(
                Message.session_id == session.id
            ).order_by(Message.created_at)
            
            messages_result = await db.execute(messages_query)
            messages = messages_result.scalars().all()
            
            thread_data["messages"] = [
                {
                    "message_id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "message_type": msg.message_type,
                    "metadata": msg.metadata,
                    "created_at": msg.created_at.isoformat()
                }
                for msg in messages
            ]
        
        export_data["threads"].append(thread_data)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)


async def export_to_csv(
    db: AsyncSession,
    sessions: List[DBSession],
    file_path: Path,
    include_messages: bool = False
):
    """导出为CSV格式"""
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        if include_messages:
            fieldnames = [
                'thread_id', 'thread_title', 'thread_status', 'thread_created_at',
                'message_id', 'message_role', 'message_content', 'message_type', 'message_created_at'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for session in sessions:
                # 获取消息
                messages_query = select(Message).where(
                    Message.session_id == session.id
                ).order_by(Message.created_at)
                
                messages_result = await db.execute(messages_query)
                messages = messages_result.scalars().all()
                
                if messages:
                    for msg in messages:
                        writer.writerow({
                            'thread_id': str(session.id),
                            'thread_title': session.title,
                            'thread_status': session.status,
                            'thread_created_at': session.created_at.isoformat(),
                            'message_id': str(msg.id),
                            'message_role': msg.role,
                            'message_content': msg.content,
                            'message_type': msg.message_type,
                            'message_created_at': msg.created_at.isoformat()
                        })
                else:
                    # 没有消息的线程也要导出
                    writer.writerow({
                        'thread_id': str(session.id),
                        'thread_title': session.title,
                        'thread_status': session.status,
                        'thread_created_at': session.created_at.isoformat(),
                        'message_id': '',
                        'message_role': '',
                        'message_content': '',
                        'message_type': '',
                        'message_created_at': ''
                    })
        else:
            fieldnames = [
                'thread_id', 'title', 'status', 'mode', 'created_at', 'updated_at',
                'description', 'thread_type', 'priority', 'tags'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for session in sessions:
                metadata = session.metadata or {}
                writer.writerow({
                    'thread_id': str(session.id),
                    'title': session.title,
                    'status': session.status,
                    'mode': session.mode,
                    'created_at': session.created_at.isoformat(),
                    'updated_at': session.updated_at.isoformat(),
                    'description': metadata.get('description', ''),
                    'thread_type': metadata.get('thread_type', ''),
                    'priority': metadata.get('priority', ''),
                    'tags': ','.join(metadata.get('tags', []))
                })


async def cleanup_export_file(file_path: Path, delay_hours: int = 24):
    """清理导出文件"""
    import asyncio
    await asyncio.sleep(delay_hours * 3600)  # 转换为秒
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"导出文件已清理: {file_path}")
    except Exception as e:
        logger.error(f"清理导出文件失败: {e}")