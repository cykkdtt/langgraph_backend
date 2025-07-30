#!/usr/bin/env python3
"""
简化的数据库初始化脚本
只使用PostgreSQL连接，不依赖Redis
"""

import asyncio
import asyncpg
import argparse
import logging
import sys
import os
from dotenv import load_dotenv
from typing import List, Dict
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleDatabaseInitializer:
    """简化的数据库初始化器"""
    
    def __init__(self):
        self.settings = Settings()
        self.conn = None
    
    async def connect(self) -> None:
        """连接数据库"""
        logger.info("连接PostgreSQL数据库...")
        
        # 从URI中提取连接参数
        uri = self.settings.database.postgres_url
        self.conn = await asyncpg.connect(uri)
        
        # 测试连接
        version = await self.conn.fetchval("SELECT version()")
        logger.info(f"PostgreSQL连接成功: {version}")
    
    async def disconnect(self) -> None:
        """断开数据库连接"""
        if self.conn:
            await self.conn.close()
            logger.info("数据库连接已关闭")
    
    async def get_existing_tables(self) -> List[str]:
        """获取现有表列表"""
        logger.info("获取现有表列表...")
        
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        
        rows = await self.conn.fetch(query)
        tables = [row['table_name'] for row in rows]
        
        logger.info(f"现有表: {tables}")
        return tables
    
    async def install_extensions(self) -> None:
        """安装PostgreSQL扩展"""
        logger.info("安装PostgreSQL扩展...")
        
        # 安装uuid-ossp扩展
        try:
            await self.conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
            logger.info("uuid-ossp扩展安装成功")
        except Exception as e:
            logger.warning(f"uuid-ossp扩展安装失败: {e}")
        
        # 安装pgvector扩展
        try:
            await self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("pgvector扩展安装成功")
        except Exception as e:
            logger.error(f"pgvector扩展安装失败: {e}")
            logger.error("请确保已正确安装pgvector扩展")
            raise Exception("pgvector扩展安装失败，这是LangMem的必需依赖")
    
    async def create_langgraph_tables(self) -> None:
        """创建LangGraph相关表"""
        logger.info("创建LangGraph相关表...")
        
        try:
            # 创建检查点表
            logger.info("创建检查点表...")
            async with AsyncPostgresSaver.from_conn_string(
                self.settings.database.postgres_url
            ) as checkpoint_saver:
                await checkpoint_saver.setup()
            logger.info("检查点表创建完成")
            
            # 创建存储表（包含LangMem所需的store和store_vectors表）
            logger.info("创建存储表...")
            
            # 配置向量索引以支持阿里text-embedding-v4模型
            # 使用1024维度（阿里text-embedding-v4的默认维度）
            index_config = {
                "dims": self.settings.llm.embedding_dimensions,  # 1024维度
                "embed": self.settings.llm.embedding_model,      # dashscope:text-embedding-v4
                "fields": ["content", "summary", "description"]  # 需要向量化的字段
            }
            
            async with AsyncPostgresStore.from_conn_string(
                self.settings.database.postgres_url,
                index=index_config
            ) as store:
                await store.setup()
            logger.info(f"存储表创建完成（包含LangMem所需的store和store_vectors表，使用{self.settings.llm.embedding_model}模型，维度：{self.settings.llm.embedding_dimensions}）")
            
        except Exception as e:
            logger.error(f"创建LangGraph表失败: {e}")
            raise
    
    async def create_custom_tables(self) -> None:
        """创建自定义表"""
        logger.info("创建自定义表...")
        
        # 用户表
        user_table_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            is_admin BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # 会话表
        session_table_sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            title VARCHAR(200),
            description TEXT,
            metadata JSONB DEFAULT '{}',
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # 消息表
        message_table_sql = """
        CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # 工具调用记录表
        tool_calls_table_sql = """
        CREATE TABLE IF NOT EXISTS tool_calls (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
            message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
            tool_name VARCHAR(100) NOT NULL,
            tool_input JSONB NOT NULL,
            tool_output JSONB,
            status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'success', 'error')),
            error_message TEXT,
            execution_time FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP WITH TIME ZONE
        );
        """
        
        # 智能体状态表
        agent_states_table_sql = """
        CREATE TABLE IF NOT EXISTS agent_states (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
            agent_name VARCHAR(100) NOT NULL,
            state_data JSONB NOT NULL,
            version INTEGER DEFAULT 1,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # 系统日志表
        system_logs_table_sql = """
        CREATE TABLE IF NOT EXISTS system_logs (
            id BIGSERIAL PRIMARY KEY,
            level VARCHAR(20) NOT NULL,
            logger_name VARCHAR(100) NOT NULL,
            message TEXT NOT NULL,
            module VARCHAR(100),
            function_name VARCHAR(100),
            line_number INTEGER,
            exception TEXT,
            extra_data JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # 执行SQL语句
        tables_sql = [
            user_table_sql,
            session_table_sql,
            message_table_sql,
            tool_calls_table_sql,
            agent_states_table_sql,
            system_logs_table_sql
        ]
        
        for sql in tables_sql:
            await self.conn.execute(sql)
            logger.debug(f"执行SQL: {sql[:50]}...")
        
        logger.info("自定义表创建完成")
    
    async def create_indexes(self) -> None:
        """创建索引"""
        logger.info("创建索引...")
        
        # 用户表索引
        user_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);",
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
            "CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);",
        ]
        
        # 会话表索引
        session_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);",
            "CREATE INDEX IF NOT EXISTS idx_sessions_is_active ON sessions(is_active);",
            "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);",
        ]
        
        # 消息表索引
        message_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);",
            "CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);",
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);",
        ]
        
        # 工具调用表索引
        tool_call_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_tool_calls_session_id ON tool_calls(session_id);",
            "CREATE INDEX IF NOT EXISTS idx_tool_calls_message_id ON tool_calls(message_id);",
            "CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name ON tool_calls(tool_name);",
            "CREATE INDEX IF NOT EXISTS idx_tool_calls_status ON tool_calls(status);",
        ]
        
        # 智能体状态表索引
        agent_state_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_agent_states_session_id ON agent_states(session_id);",
            "CREATE INDEX IF NOT EXISTS idx_agent_states_agent_name ON agent_states(agent_name);",
        ]
        
        # 系统日志表索引
        log_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_logger_name ON system_logs(logger_name);",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs(created_at);",
        ]
        
        all_indexes = (user_indexes + session_indexes + message_indexes + 
                      tool_call_indexes + agent_state_indexes + log_indexes)
        
        for index_sql in all_indexes:
            await self.conn.execute(index_sql)
            logger.debug(f"创建索引: {index_sql[:50]}...")
        
        logger.info("索引创建完成")
    
    async def create_triggers(self) -> None:
        """创建触发器"""
        logger.info("创建触发器...")
        
        # 更新时间戳触发器函数
        trigger_function_sql = """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """
        
        # 用户表更新触发器
        user_trigger_sql = """
        DROP TRIGGER IF EXISTS update_users_updated_at ON users;
        CREATE TRIGGER update_users_updated_at
            BEFORE UPDATE ON users
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """
        
        # 会话表更新触发器
        session_trigger_sql = """
        DROP TRIGGER IF EXISTS update_sessions_updated_at ON sessions;
        CREATE TRIGGER update_sessions_updated_at
            BEFORE UPDATE ON sessions
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """
        
        await self.conn.execute(trigger_function_sql)
        await self.conn.execute(user_trigger_sql)
        await self.conn.execute(session_trigger_sql)
        
        logger.info("触发器创建完成")
    
    async def insert_initial_data(self) -> None:
        """插入初始数据"""
        logger.info("插入初始数据...")
        
        # 创建默认管理员用户
        admin_user_sql = """
        INSERT INTO users (username, email, password_hash, is_admin)
        VALUES ('admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq/3Hm.', TRUE)
        ON CONFLICT (username) DO NOTHING;
        """
        
        await self.conn.execute(admin_user_sql)
        logger.info("初始数据插入完成")
    
    async def verify_tables(self) -> Dict[str, bool]:
        """验证表是否创建成功"""
        logger.info("验证表创建...")
        
        # 自定义表
        expected_custom_tables = [
            'users', 'sessions', 'messages', 'tool_calls', 
            'agent_states', 'system_logs'
        ]
        
        # LangGraph相关表
        expected_langgraph_tables = [
            'checkpoints', 'checkpoint_blobs', 'checkpoint_writes',  # 检查点表
            'store', 'store_vectors'  # LangMem存储表
        ]
        
        expected_tables = expected_custom_tables + expected_langgraph_tables
        existing_tables = await self.get_existing_tables()
        
        verification_result = {}
        for table in expected_tables:
            exists = table in existing_tables
            verification_result[table] = exists
            if exists:
                logger.info(f"表 {table} 创建成功")
            else:
                logger.error(f"表 {table} 创建失败")
        
        # 验证PostgreSQL扩展
        logger.info("验证PostgreSQL扩展...")
        
        # 检查pgvector扩展
        vector_exists = await self.conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
        )
        verification_result['pgvector_extension'] = vector_exists
        
        if vector_exists:
            logger.info("pgvector扩展安装成功")
        else:
            logger.error("pgvector扩展安装失败")
        
        # 检查uuid-ossp扩展
        uuid_exists = await self.conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp');"
        )
        verification_result['uuid_ossp_extension'] = uuid_exists
        
        if uuid_exists:
            logger.info("uuid-ossp扩展安装成功")
        else:
            logger.error("uuid-ossp扩展安装失败")
        
        return verification_result
    
    async def run_full_initialization(self, force: bool = False) -> bool:
        """运行完整的数据库初始化"""
        logger.info("开始数据库初始化...")
        
        try:
            # 连接数据库
            await self.connect()
            
            # 获取现有表
            existing_tables = await self.get_existing_tables()
            
            # 如果表已存在且不强制重建，则跳过
            if existing_tables and not force:
                logger.info("数据库表已存在，跳过初始化（使用 --force 强制重建）")
                return True
            
            # 安装扩展
            await self.install_extensions()
            
            # 创建LangGraph表
            await self.create_langgraph_tables()
            
            # 创建自定义表
            await self.create_custom_tables()
            
            # 创建索引
            await self.create_indexes()
            
            # 创建触发器
            await self.create_triggers()
            
            # 插入初始数据
            await self.insert_initial_data()
            
            # 验证表创建
            verification_result = await self.verify_tables()
            
            # 检查是否所有表都创建成功
            all_created = all(verification_result.values())
            
            if all_created:
                logger.info("数据库初始化完成")
                return True
            else:
                failed_tables = [table for table, created in verification_result.items() if not created]
                logger.error(f"以下表创建失败: {failed_tables}")
                return False
        
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            return False
        finally:
            # 断开连接
            await self.disconnect()


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化的数据库初始化脚本")
    parser.add_argument("--force", action="store_true", help="强制重建数据库表")
    parser.add_argument("--check-only", action="store_true", help="仅检查数据库状态")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建数据库初始化器
    initializer = SimpleDatabaseInitializer()
    
    if args.check_only:
        # 仅检查数据库状态
        logger.info("检查数据库状态...")
        
        try:
            await initializer.connect()
            existing_tables = await initializer.get_existing_tables()
            logger.info(f"数据库连接正常，现有表: {existing_tables}")
            return 0
        except Exception as e:
            logger.error(f"数据库检查失败: {e}")
            return 1
        finally:
            await initializer.disconnect()
    else:
        # 运行数据库初始化
        success = await initializer.run_full_initialization(force=args.force)
        return 0 if success else 1


if __name__ == "__main__":
    # 运行主函数
    exit_code = asyncio.run(main())
    sys.exit(exit_code)