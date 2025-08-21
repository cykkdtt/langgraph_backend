#!/usr/bin/env python3
"""数据库初始化脚本

本脚本用于初始化数据库，创建表结构并插入初始数据。
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database import db_manager, get_db_session
from models.database_models import (
    User, Session, Message, ToolCall, AgentState, 
    SystemLog, Workflow, WorkflowExecution, Memory
)
from werkzeug.security import generate_password_hash
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_admin_user():
    """创建管理员用户"""
    with get_db_session() as db:
        # 检查是否已存在管理员用户
        admin_user = db.query(User).filter(User.username == "admin").first()
        if admin_user:
            logger.info("管理员用户已存在")
            return admin_user
        
        # 创建管理员用户
        admin_user = User(
            username="admin",
            email="admin@langgraph.com",
            password_hash=generate_password_hash("admin123"),
            full_name="系统管理员",
            role="admin",
            status="active",
            is_active=True,
            is_admin=True,
            preferences={
                "theme": "dark",
                "language": "zh-CN",
                "notifications": True
            }
        )
        
        db.add(admin_user)
        db.flush()  # 获取 ID
        logger.info(f"创建管理员用户: {admin_user.username} (ID: {admin_user.id})")
        return admin_user


def create_demo_user():
    """创建演示用户"""
    with get_db_session() as db:
        # 检查是否已存在演示用户
        demo_user = db.query(User).filter(User.username == "demo").first()
        if demo_user:
            logger.info("演示用户已存在")
            return demo_user
        
        # 创建演示用户
        demo_user = User(
            username="demo",
            email="demo@langgraph.com",
            password_hash=generate_password_hash("demo123"),
            full_name="演示用户",
            role="user",
            status="active",
            is_active=True,
            is_admin=False,
            preferences={
                "theme": "light",
                "language": "zh-CN",
                "notifications": True
            }
        )
        
        db.add(demo_user)
        db.flush()  # 获取 ID
        logger.info(f"创建演示用户: {demo_user.username} (ID: {demo_user.id})")
        return demo_user


def create_demo_session(user: User):
    """创建演示会话"""
    with get_db_session() as db:
        session = Session(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="演示对话",
            description="这是一个演示对话会话",
            metadata={
                "agent_count": 2,
                "workflow_enabled": True,
                "memory_enabled": True
            },
            is_active=True
        )
        
        db.add(session)
        db.flush()
        logger.info(f"创建演示会话: {session.title} (ID: {session.id})")
        return session


def create_demo_messages(session: Session, user: User):
    """创建演示消息"""
    with get_db_session() as db:
        messages = [
            {
                "role": "user",
                "content": "你好，我想了解一下LangGraph多智能体系统的功能。",
                "message_type": "text"
            },
            {
                "role": "assistant",
                "content": "您好！LangGraph多智能体系统是一个强大的AI协作平台，主要功能包括：\n\n1. **多智能体协作**：支持多个AI智能体同时工作\n2. **工作流管理**：可以创建和执行复杂的工作流\n3. **记忆管理**：具备语义记忆、情节记忆和程序记忆\n4. **时间旅行**：支持状态回滚和历史查看\n5. **实时通信**：WebSocket实时消息传递\n\n您想了解哪个方面的详细信息？",
                "message_type": "text",
                "agent_id": "assistant_001",
                "agent_name": "主助手"
            },
            {
                "role": "user",
                "content": "能演示一下多智能体协作吗？",
                "message_type": "text"
            },
            {
                "role": "assistant",
                "content": "当然可以！让我启动一个多智能体协作演示。",
                "message_type": "text",
                "agent_id": "coordinator_001",
                "agent_name": "协调器",
                "tool_calls": [
                    {
                        "tool_name": "start_multi_agent_demo",
                        "tool_input": {"agents": ["analyst", "researcher", "writer"]}
                    }
                ]
            }
        ]
        
        for i, msg_data in enumerate(messages):
            message = Message(
                id=str(uuid.uuid4()),
                session_id=session.id,
                user_id=user.id,
                role=msg_data["role"],
                message_type=msg_data["message_type"],
                content=msg_data["content"],
                agent_id=msg_data.get("agent_id"),
                agent_name=msg_data.get("agent_name"),
                tool_calls=msg_data.get("tool_calls"),
                tokens_used=len(msg_data["content"]) // 4,  # 估算token数
                processing_time=0.5 + i * 0.2,
                created_at=datetime.utcnow() + timedelta(minutes=i)
            )
            db.add(message)
        
        db.flush()
        logger.info(f"创建了 {len(messages)} 条演示消息")


def create_demo_workflow(user: User):
    """创建演示工作流"""
    with get_db_session() as db:
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name="文档分析工作流",
            description="自动分析文档内容并生成摘要的工作流",
            user_id=user.id,
            definition={
                "nodes": [
                    {
                        "id": "input",
                        "type": "input",
                        "name": "文档输入",
                        "config": {"accept_types": ["text", "pdf", "docx"]}
                    },
                    {
                        "id": "analyzer",
                        "type": "agent",
                        "name": "文档分析器",
                        "config": {"agent_type": "document_analyzer"}
                    },
                    {
                        "id": "summarizer",
                        "type": "agent",
                        "name": "摘要生成器",
                        "config": {"agent_type": "summarizer"}
                    },
                    {
                        "id": "output",
                        "type": "output",
                        "name": "结果输出",
                        "config": {"format": "json"}
                    }
                ],
                "edges": [
                    {"from": "input", "to": "analyzer"},
                    {"from": "analyzer", "to": "summarizer"},
                    {"from": "summarizer", "to": "output"}
                ]
            },
            version="1.0.0",
            status="active",
            is_public=True,
            tags=["文档处理", "AI分析", "自动化"],
            metadata={
                "estimated_time": "2-5分钟",
                "complexity": "medium",
                "category": "document_processing"
            }
        )
        
        db.add(workflow)
        db.flush()
        logger.info(f"创建演示工作流: {workflow.name} (ID: {workflow.id})")
        return workflow


def create_demo_memories(user: User, session: Session):
    """创建演示记忆"""
    with get_db_session() as db:
        memories = [
            {
                "memory_type": "semantic",
                "content": "LangGraph是一个用于构建多智能体系统的框架",
                "importance_score": 0.9,
                "metadata": {"category": "技术知识", "source": "用户对话"}
            },
            {
                "memory_type": "episodic",
                "content": "用户询问了多智能体协作功能的演示",
                "importance_score": 0.7,
                "metadata": {"event_type": "用户询问", "context": "功能演示"}
            },
            {
                "memory_type": "procedural",
                "content": "启动多智能体演示的步骤：1.选择智能体类型 2.配置协作模式 3.开始执行",
                "importance_score": 0.8,
                "metadata": {"procedure_type": "系统操作", "steps": 3}
            }
        ]
        
        for memory_data in memories:
            memory = Memory(
                id=str(uuid.uuid4()),
                user_id=user.id,
                session_id=session.id,
                memory_type=memory_data["memory_type"],
                content=memory_data["content"],
                importance_score=memory_data["importance_score"],
                metadata=memory_data["metadata"],
                access_count=1,
                last_accessed_at=datetime.utcnow()
            )
            db.add(memory)
        
        db.flush()
        logger.info(f"创建了 {len(memories)} 条演示记忆")


def create_system_logs():
    """创建系统日志"""
    with get_db_session() as db:
        logs = [
            {
                "level": "INFO",
                "logger_name": "system.startup",
                "message": "LangGraph多智能体系统启动成功",
                "module": "main",
                "function": "startup"
            },
            {
                "level": "INFO",
                "logger_name": "database.init",
                "message": "数据库初始化完成",
                "module": "database",
                "function": "init_database"
            },
            {
                "level": "INFO",
                "logger_name": "api.auth",
                "message": "用户认证模块加载成功",
                "module": "auth",
                "function": "load_auth_module"
            }
        ]
        
        for log_data in logs:
            log = SystemLog(
                level=log_data["level"],
                logger_name=log_data["logger_name"],
                message=log_data["message"],
                module=log_data["module"],
                function=log_data["function"],
                created_at=datetime.utcnow()
            )
            db.add(log)
        
        db.flush()
        logger.info(f"创建了 {len(logs)} 条系统日志")


def main():
    """主函数"""
    logger.info("开始初始化数据库...")
    
    try:
        # 初始化数据库表
        logger.info("创建数据库表...")
        db_manager.init_db()
        
        # 创建用户
        logger.info("创建用户...")
        admin_user = create_admin_user()
        demo_user = create_demo_user()
        
        # 创建演示数据
        logger.info("创建演示数据...")
        demo_session = create_demo_session(demo_user)
        create_demo_messages(demo_session, demo_user)
        create_demo_workflow(demo_user)
        create_demo_memories(demo_user, demo_session)
        
        # 创建系统日志
        logger.info("创建系统日志...")
        create_system_logs()
        
        # 检查数据库连接
        if db_manager.check_connection():
            logger.info("数据库连接正常")
        else:
            logger.error("数据库连接失败")
            return False
        
        # 显示统计信息
        stats = db_manager.get_database_stats()
        logger.info("数据库统计信息:")
        for table, count in stats.get("table_stats", {}).items():
            logger.info(f"  {table}: {count} 条记录")
        logger.info(f"总记录数: {stats.get('total_records', 0)}")
        
        logger.info("数据库初始化完成！")
        
        # 输出登录信息
        print("\n" + "="*50)
        print("数据库初始化成功！")
        print("\n默认用户账号:")
        print(f"管理员 - 用户名: admin, 密码: admin123")
        print(f"演示用户 - 用户名: demo, 密码: demo123")
        print("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)