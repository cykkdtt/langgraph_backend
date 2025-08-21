"""聊天API路由实现

本模块实现了LangGraph多智能体系统的聊天API，包括：
- 单次聊天API
- 流式聊天API
- 批量聊天API
- 聊天历史管理
- 会话管理
"""

from typing import List, Optional, Dict, Any, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
import json
import asyncio
from datetime import datetime, timedelta
import uuid

from models.chat_models import (
    ChatRequest, ChatResponse, StreamChunk, ChatHistory, 
    BatchChatRequest, BatchChatResponse, SessionInfo,
    ChatMessage, MessageRole, MessageType, ChatMode,
    StreamEventType, AgentStatus, ChatError, MessageInfo, ChatMessageRequest, ChatMessageResponse
)
from models.database_models import User, Session, Message, AgentState
from models.api_models import BaseResponse, PaginatedResponse, ErrorDetail, ErrorCode, WebSocketMessage
from models.auth_models import UserInfo
from api.auth import get_current_user
from core.database import get_async_session
from core.agents import AgentRegistry, AgentManager
from core.logging import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/chat", tags=["聊天"])

# 智能体管理器实例
agent_manager = AgentManager()


@router.post("/send", response_model=ChatResponse, summary="发送聊天消息")
async def send_chat_message(
    request: ChatRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> ChatResponse:
    """发送聊天消息并获取智能体响应"""
    try:
        start_time = datetime.utcnow()
        
        # 获取或创建会话
        session = await get_or_create_session(
            db, request.session_id, current_user.user_id, request.mode
        )
        
        # 保存用户消息
        user_message = await save_message(
            db, session.id, current_user.user_id, 
            MessageRole.USER, request.message, MessageType.TEXT
        )
        
        # 选择合适的智能体
        agent = await select_agent(request, session)
        
        # 获取聊天历史
        history = await get_chat_history_for_agent(
            db, session.id, request.max_history
        )
        
        # 调用智能体处理消息
        agent_response = await agent.chat(
            message=request.message,
            history=history,
            context=request.context,
            tools_enabled=request.include_tools
        )
        
        # 保存智能体响应
        assistant_message = await save_message(
            db, session.id, current_user.user_id,
            MessageRole.ASSISTANT, agent_response.content, MessageType.TEXT,
            metadata={
                "agent_type": agent.agent_type,
                "agent_name": agent.name,
                "tool_calls": [call.dict() for call in agent_response.tool_calls],
                "usage": agent_response.usage
            }
        )
        
        # 更新会话信息
        await update_session_activity(db, session.id)
        
        # 计算处理时间
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 后台任务：更新智能体状态和指标
        background_tasks.add_task(
            update_agent_metrics, agent.agent_type, processing_time, True
        )
        
        return ChatResponse(
            message_id=assistant_message.id,
            session_id=session.id,
            content=agent_response.content,
            agent_type=agent.agent_type,
            agent_name=agent.name,
            tool_calls=agent_response.tool_calls,
            metadata=agent_response.metadata,
            usage=agent_response.usage,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"聊天消息处理失败: {e}")
        # 后台任务：记录错误指标
        if 'agent' in locals():
            background_tasks.add_task(
                update_agent_metrics, agent.agent_type, 0, False
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"聊天处理失败: {str(e)}"
        )


@router.post("/stream", summary="流式聊天")
async def stream_chat_message(
    request: ChatRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """流式聊天API"""
    async def generate_stream():
        try:
            # 获取或创建会话
            session = await get_or_create_session(
                db, request.session_id, current_user.user_id, request.mode
            )
            
            # 保存用户消息
            user_message = await save_message(
                db, session.id, current_user.user_id,
                MessageRole.USER, request.message, MessageType.TEXT
            )
            
            # 选择智能体
            agent = await select_agent(request, session)
            
            # 获取聊天历史
            history = await get_chat_history_for_agent(
                db, session.id, request.max_history
            )
            
            # 发送开始事件
            yield create_stream_chunk(
                StreamEventType.MESSAGE_START,
                {"session_id": session.id, "agent_type": agent.agent_type},
                session.id, agent.agent_type
            )
            
            # 流式处理消息
            full_content = ""
            async for chunk in agent.stream_chat(
                message=request.message,
                history=history,
                context=request.context,
                tools_enabled=request.include_tools
            ):
                if chunk.event_type == StreamEventType.MESSAGE_DELTA:
                    full_content += chunk.data.get("content", "")
                
                yield create_stream_chunk(
                    chunk.event_type,
                    chunk.data,
                    session.id,
                    agent.agent_type
                )
            
            # 保存完整响应
            assistant_message = await save_message(
                db, session.id, current_user.user_id,
                MessageRole.ASSISTANT, full_content, MessageType.TEXT,
                metadata={"agent_type": agent.agent_type, "agent_name": agent.name}
            )
            
            # 发送结束事件
            yield create_stream_chunk(
                StreamEventType.MESSAGE_END,
                {"message_id": assistant_message.id, "total_content": full_content},
                session.id, agent.agent_type
            )
            
        except Exception as e:
            logger.error(f"流式聊天失败: {e}")
            yield create_stream_chunk(
                StreamEventType.ERROR,
                {"error": str(e), "error_type": type(e).__name__},
                session.id if 'session' in locals() else None
            )
    
    return StreamingResponse(
        generate_stream_response(generate_stream()),
        media_type="text/plain"
    )


@router.post("/batch", response_model=BatchChatResponse, summary="批量聊天")
async def batch_chat(
    request: BatchChatRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
) -> BatchChatResponse:
    """批量处理聊天请求"""
    try:
        start_time = datetime.utcnow()
        responses = []
        success_count = 0
        error_count = 0
        
        if request.parallel:
            # 并行处理
            semaphore = asyncio.Semaphore(request.max_concurrent)
            tasks = []
            
            for chat_request in request.requests:
                task = process_single_chat_with_semaphore(
                    semaphore, chat_request, current_user, db
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                    responses.append(create_error_response(str(result)))
                else:
                    success_count += 1
                    responses.append(result)
        else:
            # 串行处理
            for chat_request in request.requests:
                try:
                    response = await process_single_chat(chat_request, current_user, db)
                    responses.append(response)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    responses.append(create_error_response(str(e)))
        
        total_time = (datetime.utcnow() - start_time).total_seconds()
        
        return BatchChatResponse(
            responses=responses,
            success_count=success_count,
            error_count=error_count,
            total_time=total_time
        )
        
    except Exception as e:
        logger.error(f"批量聊天处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量聊天处理失败: {str(e)}"
        )


@router.get("/history/{session_id}", response_model=ChatHistory, summary="获取聊天历史")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
) -> ChatHistory:
    """获取指定会话的聊天历史"""
    try:
        # 验证会话权限
        session = await verify_session_access(db, session_id, current_user.user_id)
        
        # 查询消息
        query = select(Message).where(
            Message.session_id == session_id
        ).order_by(desc(Message.created_at)).offset(offset).limit(limit + 1)
        
        result = await db.execute(query)
        messages_data = result.scalars().all()
        
        # 检查是否有更多消息
        has_more = len(messages_data) > limit
        if has_more:
            messages_data = messages_data[:-1]
        
        # 转换为ChatMessage模型
        messages = []
        for msg in messages_data:
            chat_msg = ChatMessage(
                id=msg.id,
                role=MessageRole(msg.role),
                content=msg.content,
                message_type=MessageType(msg.message_type),
                metadata=msg.metadata or {},
                created_at=msg.created_at,
                updated_at=msg.updated_at
            )
            messages.append(chat_msg)
        
        # 获取总消息数
        count_query = select(func.count(Message.id)).where(
            Message.session_id == session_id
        )
        total_result = await db.execute(count_query)
        total_count = total_result.scalar()
        
        return ChatHistory(
            session_id=session_id,
            messages=messages,
            total_count=total_count,
            has_more=has_more,
            next_cursor=str(offset + limit) if has_more else None
        )
        
    except Exception as e:
        logger.error(f"获取聊天历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取聊天历史失败: {str(e)}"
        )


@router.get("/sessions", response_model=PaginatedResponse[SessionInfo], summary="获取用户会话列表")
async def get_user_sessions(
    limit: int = 20,
    offset: int = 0,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
) -> PaginatedResponse[SessionInfo]:
    """获取当前用户的会话列表"""
    try:
        query = select(Session).where(
            Session.user_id == current_user.user_id
        ).order_by(desc(Session.updated_at)).offset(offset).limit(limit)
        
        result = await db.execute(query)
        sessions = result.scalars().all()
        
        session_infos = []
        for session in sessions:
            # 获取消息数量
            count_query = select(func.count(Message.id)).where(
                Message.session_id == session.id
            )
            count_result = await db.execute(count_query)
            message_count = count_result.scalar()
            
            session_info = SessionInfo(
                session_id=session.id,
                user_id=session.user_id,
                title=session.title,
                description=session.description,
                mode=ChatMode(session.mode),
                active_agents=session.metadata.get("active_agents", []),
                message_count=message_count,
                created_at=session.created_at,
                updated_at=session.updated_at,
                metadata=session.metadata or {}
            )
            session_infos.append(session_info)
        
        # 获取总数
        total_query = select(func.count(Session.id)).where(
            Session.user_id == current_user.user_id
        )
        total_result = await db.execute(total_query)
        total = total_result.scalar()
        
        from models.api_models import PaginationInfo
        pagination = PaginationInfo.create(offset // limit + 1, limit, total)
        
        return PaginatedResponse.success(
            data=session_infos,
            pagination=pagination,
            message="获取会话列表成功"
        )
        
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        from models.api_models import PaginationInfo
        pagination = PaginationInfo.create(offset // limit + 1, limit, 0)
        return PaginatedResponse(
            status="error",
            message="获取会话列表失败",
            data=[],
            pagination=pagination,
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )


@router.delete("/sessions/{session_id}", summary="删除会话")
async def delete_session(
    session_id: str,
    current_user: UserInfo = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """删除指定会话及其所有消息"""
    try:
        # 验证会话权限
        session = await verify_session_access(db, session_id, current_user.user_id)
        
        # 删除会话相关的消息
        await db.execute(
            Message.__table__.delete().where(Message.session_id == session_id)
        )
        
        # 删除会话
        await db.delete(session)
        await db.commit()
        
        return {"message": "会话删除成功"}
        
    except Exception as e:
        logger.error(f"删除会话失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除会话失败: {str(e)}"
        )


# 辅助函数

async def get_or_create_session(
    db: AsyncSession, session_id: Optional[str], user_id: str, mode: ChatMode
) -> Session:
    """获取或创建会话"""
    if session_id:
        # 查找现有会话
        query = select(Session).where(
            and_(Session.id == session_id, Session.user_id == user_id)
        )
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if session:
            return session
    
    # 创建新会话
    session = Session(
        id=str(uuid.uuid4()),
        user_id=user_id,
        title=f"会话 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        mode=mode.value,
        status="active",
        metadata={"created_by": "chat_api"}
    )
    
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    return session


async def save_message(
    db: AsyncSession, session_id: str, user_id: str,
    role: MessageRole, content: str, message_type: MessageType,
    metadata: Optional[Dict[str, Any]] = None
) -> Message:
    """保存消息到数据库"""
    message = Message(
        id=str(uuid.uuid4()),
        session_id=session_id,
        user_id=user_id,
        role=role.value,
        content=content,
        message_type=message_type.value,
        metadata=metadata or {}
    )
    
    db.add(message)
    await db.commit()
    await db.refresh(message)
    
    return message


async def select_agent(request: ChatRequest, session: Session):
    """选择合适的智能体"""
    if request.agent_config and request.agent_config.agent_type:
        # 使用指定的智能体
        agent_type = request.agent_config.agent_type
    elif request.mode == ChatMode.AUTO_SELECT:
        # 自动选择智能体
        agent_type = await auto_select_agent(request.message, session)
    else:
        # 使用默认智能体
        agent_type = "general"
    
    # 获取智能体实例
    agent = await agent_manager.get_agent(agent_type)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"智能体类型 '{agent_type}' 不存在"
        )
    
    return agent


async def auto_select_agent(message: str, session: Session) -> str:
    """自动选择最适合的智能体"""
    # 这里可以实现智能体选择逻辑
    # 例如基于消息内容、历史对话等
    
    # 简单的关键词匹配示例
    message_lower = message.lower()
    
    if any(keyword in message_lower for keyword in ["图表", "数据可视化", "chart"]):
        return "chart"
    elif any(keyword in message_lower for keyword in ["研究", "分析", "调研"]):
        return "research"
    elif any(keyword in message_lower for keyword in ["代码", "编程", "开发"]):
        return "coding"
    else:
        return "general"


async def get_chat_history_for_agent(
    db: AsyncSession, session_id: str, max_history: int
) -> List[Dict[str, Any]]:
    """获取智能体所需的聊天历史格式"""
    query = select(Message).where(
        Message.session_id == session_id
    ).order_by(desc(Message.created_at)).limit(max_history)
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    # 转换为智能体所需的格式
    history = []
    for msg in reversed(messages):  # 按时间正序
        history.append({
            "role": msg.role,
            "content": msg.content,
            "metadata": msg.metadata or {}
        })
    
    return history


async def update_session_activity(db: AsyncSession, session_id: str):
    """更新会话活动时间"""
    query = select(Session).where(Session.id == session_id)
    result = await db.execute(query)
    session = result.scalar_one_or_none()
    
    if session:
        session.updated_at = datetime.utcnow()
        await db.commit()


async def verify_session_access(db: AsyncSession, session_id: str, user_id: str) -> Session:
    """验证用户对会话的访问权限"""
    query = select(Session).where(
        and_(Session.id == session_id, Session.user_id == user_id)
    )
    result = await db.execute(query)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在或无访问权限"
        )
    
    return session


def create_stream_chunk(
    event_type: StreamEventType, data: Any, session_id: str, agent_type: str = None
) -> str:
    """创建流式响应块"""
    chunk = StreamChunk(
        event_type=event_type,
        data=data,
        session_id=session_id,
        agent_type=agent_type
    )
    return f"data: {chunk.json()}\n\n"


async def generate_stream_response(generator: AsyncGenerator) -> AsyncGenerator[str, None]:
    """生成流式响应"""
    async for chunk in generator:
        yield chunk


async def process_single_chat_with_semaphore(
    semaphore: asyncio.Semaphore, request: ChatRequest, 
    current_user: UserInfo, db: AsyncSession
) -> ChatResponse:
    """使用信号量控制的单次聊天处理"""
    async with semaphore:
        return await process_single_chat(request, current_user, db)


@router.post("/sessions/{session_id}/messages", response_model=BaseResponse[ChatMessageResponse])
async def send_message(
    session_id: int,
    message_data: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    db = Depends(get_async_session)
):
    """发送聊天消息"""
    try:
        # 验证会话权限
        session = await get_session_by_id(session_id)
        if not session or session.user_id != current_user.id:
            return BaseResponse.error(
                message="会话不存在或无权限访问",
                errors=[ErrorDetail(
                    code=ErrorCode.RESOURCE_NOT_FOUND,
                    message="指定的会话不存在或您没有访问权限"
                )]
            )
        
        # 创建用户消息
        user_message_id = await create_message(
            session_id=session_id,
            role="user",
            content=message_data.content,
            metadata=message_data.metadata
        )
        
        # 处理消息并生成AI回复
        ai_response = await process_chat_message(
            session_id=session_id,
            user_message=message_data.content,
            agents=session.agents,
            config=session.config
        )
        
        # 创建AI回复消息
        ai_message_id = await create_message(
            session_id=session_id,
            role="assistant",
            content=ai_response["content"],
            metadata=ai_response.get("metadata", {})
        )
        
        # 获取消息详情
        user_message = await get_message_by_id(user_message_id)
        ai_message = await get_message_by_id(ai_message_id)
        
        message_response = ChatMessageResponse(
            user_message=MessageInfo(
                id=user_message.id,
                role=user_message.role,
                content=user_message.content,
                metadata=user_message.metadata,
                created_at=user_message.created_at
            ),
            ai_message=MessageInfo(
                id=ai_message.id,
                role=ai_message.role,
                content=ai_message.content,
                metadata=ai_message.metadata,
                created_at=ai_message.created_at
            )
        )
        
        return BaseResponse.success(
            data=message_response,
            message="消息发送成功"
        )
    
    except Exception as e:
        return BaseResponse.error(
            message="发送消息失败",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )


@router.get("/sessions/{session_id}/messages", response_model=PaginatedResponse[MessageInfo])
async def get_session_messages(
    session_id: int,
    page: int = 1,
    page_size: int = 50,
    current_user: User = Depends(get_current_user),
    db = Depends(get_async_session)
):
    """获取会话消息列表"""
    try:
        # 验证会话权限
        session = await get_session_by_id(session_id)
        if not session or session.user_id != current_user.id:
            from models.api_models import PaginationInfo
            pagination = PaginationInfo.create(page, page_size, 0)
            return PaginatedResponse(
                status="error",
                message="会话不存在或无权限访问",
                data=[],
                pagination=pagination,
                errors=[ErrorDetail(
                    code=ErrorCode.RESOURCE_NOT_FOUND,
                    message="指定的会话不存在或您没有访问权限"
                )]
            )
        
        # 获取消息列表
        messages, total = await get_session_messages_paginated(
            session_id=session_id,
            page=page,
            page_size=page_size
        )
        
        message_list = []
        for message in messages:
            message_info = MessageInfo(
                id=message.id,
                role=message.role,
                content=message.content,
                metadata=message.metadata,
                created_at=message.created_at
            )
            message_list.append(message_info)
        
        from models.api_models import PaginationInfo
        pagination = PaginationInfo.create(page, page_size, total)
        
        return PaginatedResponse.success(
            data=message_list,
            pagination=pagination,
            message="获取消息列表成功"
        )
    
    except Exception as e:
        from models.api_models import PaginationInfo
        pagination = PaginationInfo.create(page, page_size, 0)
        return PaginatedResponse(
            status="error",
            message="获取消息列表失败",
            data=[],
            pagination=pagination,
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )


async def process_single_chat(
    request: ChatRequest, current_user: UserInfo, db: AsyncSession
) -> ChatResponse:
    """处理单次聊天请求"""
    # 这里复用send_chat_message的逻辑
    # 为了简化，直接调用主要逻辑
    pass  # 实际实现中应该提取公共逻辑


def create_error_response(error_message: str) -> ChatResponse:
    """创建错误响应"""
    return ChatResponse(
        message_id=str(uuid.uuid4()),
        session_id="error",
        content=f"处理失败: {error_message}",
        agent_type="error",
        metadata={"error": True, "error_message": error_message}
    )


async def update_agent_metrics(
    agent_type: str, processing_time: float, success: bool
):
    """更新智能体指标（后台任务）"""
    try:
        # 这里可以实现指标更新逻辑
        # 例如更新Redis缓存或数据库中的指标
        logger.info(
            f"智能体指标更新: {agent_type}, 处理时间: {processing_time}s, 成功: {success}"
        )
    except Exception as e:
        logger.error(f"更新智能体指标失败: {e}")