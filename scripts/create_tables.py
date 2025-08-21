#!/usr/bin/env python3
"""数据库表创建脚本

本脚本用于创建LangGraph多智能体系统所需的数据库表结构，包括：
- 用户表
- 会话表
- 消息表
- 智能体状态表
- 工具调用表
- 系统日志表
- 工作流表
- 记忆表
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from models.database_models import Base
from config.settings import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


async def create_database_extensions(engine):
    """创建数据库扩展"""
    async with engine.begin() as conn:
        # 创建UUID扩展
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        logger.info("已创建uuid-ossp扩展")
        
        # 创建向量扩展（如果需要）
        try:
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            logger.info("已创建vector扩展")
        except Exception as e:
            logger.warning(f"创建vector扩展失败（可选）: {e}")
        
        # 创建全文搜索扩展
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS pg_trgm'))
        logger.info("已创建pg_trgm扩展")


async def create_tables():
    """创建所有数据库表"""
    try:
        # 创建异步引擎
        engine = create_async_engine(
            settings.database.postgres_uri,
            echo=True,  # 打印SQL语句
            future=True
        )
        
        logger.info("开始创建数据库表...")
        
        # 创建扩展
        await create_database_extensions(engine)
        
        # 创建所有表
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("所有数据库表创建完成")
        
        # 创建额外的索引和约束
        await create_additional_indexes(engine)
        
        await engine.dispose()
        
    except Exception as e:
        logger.error(f"创建数据库表失败: {e}")
        raise


async def create_additional_indexes(engine):
    """创建额外的索引和约束"""
    async with engine.begin() as conn:
        # 创建全文搜索索引
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_messages_content_gin 
            ON messages USING gin(to_tsvector('english', content))
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_memories_content_gin 
            ON memories USING gin(to_tsvector('english', content))
        """))
        
        # 创建复合索引
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_role_created 
            ON messages (session_id, role, created_at DESC)
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_states_session_type_updated 
            ON agent_states (session_id, agent_type, updated_at DESC)
        """))
        
        # 创建分区表（如果需要）
        await create_partitioned_tables(conn)
        
        logger.info("额外索引创建完成")


async def create_partitioned_tables(conn):
    """创建分区表（用于大数据量场景）"""
    try:
        # 为系统日志表创建按月分区
        await conn.execute(text("""
            -- 创建分区表的示例（可选）
            -- ALTER TABLE system_logs PARTITION BY RANGE (created_at);
        """))
        
        logger.info("分区表配置完成")
    except Exception as e:
        logger.warning(f"分区表创建失败（可选功能）: {e}")


async def insert_default_data():
    """插入默认数据"""
    try:
        engine = create_async_engine(
            settings.database.postgres_uri,
            echo=False
        )
        
        async with engine.begin() as conn:
            # 检查是否已有管理员用户
            result = await conn.execute(text(
                "SELECT COUNT(*) as count FROM users WHERE role = 'admin'"
            ))
            admin_count = result.fetchone()[0]
            
            if admin_count == 0:
                # 创建默认管理员用户
                from passlib.context import CryptContext
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                
                admin_password = "admin123456"  # 默认密码，生产环境需要修改
                hashed_password = pwd_context.hash(admin_password)
                
                await conn.execute(text("""
                    INSERT INTO users (username, email, password_hash, full_name, role, status, is_active, is_admin, created_at, updated_at)
                    VALUES ('admin', 'admin@langgraph.com', :password_hash, 'System Administrator', 'admin', 'active', true, true, NOW(), NOW())
                """), {"password_hash": hashed_password})
                
                logger.info("已创建默认管理员用户: admin / admin123456")
            
            # 插入其他默认数据...
            
        await engine.dispose()
        
    except Exception as e:
        logger.error(f"插入默认数据失败: {e}")
        raise


async def drop_tables():
    """删除所有表（谨慎使用）"""
    try:
        engine = create_async_engine(
            settings.database.postgres_uri,
            echo=True
        )
        
        logger.warning("开始删除所有数据库表...")
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        logger.warning("所有数据库表已删除")
        
        await engine.dispose()
        
    except Exception as e:
        logger.error(f"删除数据库表失败: {e}")
        raise


async def check_tables():
    """检查表是否存在"""
    try:
        engine = create_async_engine(
            settings.database.postgres_uri,
            echo=False
        )
        
        async with engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result.fetchall()]
            
            logger.info(f"数据库中的表: {tables}")
            
            # 检查必要的表是否存在
            required_tables = [
                'users', 'sessions', 'messages', 'tool_calls', 
                'agent_states', 'system_logs', 'workflows', 
                'workflow_executions', 'memories'
            ]
            
            missing_tables = [table for table in required_tables if table not in tables]
            
            if missing_tables:
                logger.warning(f"缺少的表: {missing_tables}")
            else:
                logger.info("所有必要的表都已存在")
        
        await engine.dispose()
        
    except Exception as e:
        logger.error(f"检查表失败: {e}")
        raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库表管理脚本")
    parser.add_argument("action", choices=["create", "drop", "check", "init"], 
                       help="操作类型: create(创建表), drop(删除表), check(检查表), init(初始化数据)")
    parser.add_argument("--force", action="store_true", help="强制执行（用于删除操作）")
    
    args = parser.parse_args()
    
    if args.action == "create":
        asyncio.run(create_tables())
    elif args.action == "drop":
        if args.force:
            asyncio.run(drop_tables())
        else:
            print("删除表需要使用 --force 参数")
    elif args.action == "check":
        asyncio.run(check_tables())
    elif args.action == "init":
        asyncio.run(create_tables())
        asyncio.run(insert_default_data())


if __name__ == "__main__":
    main()