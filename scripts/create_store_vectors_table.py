#!/usr/bin/env python3
"""
创建 store_vectors 表的脚本
使用 LangGraph 的 AsyncPostgresStore 和 DashScopeEmbeddings
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
load_dotenv(project_root / ".env")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def create_store_vectors_table():
    """创建 store_vectors 表"""
    try:
        from langgraph.store.postgres import AsyncPostgresStore
        from langchain_community.embeddings import DashScopeEmbeddings
        
        # 获取数据库连接字符串
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            logger.error("未找到 POSTGRES_URI 环境变量")
            return False
        
        logger.info(f"连接数据库: {postgres_uri}")
        
        # 创建嵌入模型
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.error("未找到 DASHSCOPE_API_KEY 环境变量")
            return False
        
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=api_key
        )
        
        logger.info(f"嵌入模型: {embeddings}")
        
        # 配置索引
        index_config = {
            "embed": embeddings,
            "dims": 1024,
            "fields": ["$"]  # 索引所有字段
        }
        
        logger.info(f"索引配置: {index_config}")
        
        # 使用 from_conn_string 创建存储实例
        async with AsyncPostgresStore.from_conn_string(
            postgres_uri,
            index=index_config
        ) as store:
            logger.info("AsyncPostgresStore 创建成功")
            
            # 设置存储（这会创建必要的表）
            await store.setup()
            
            logger.info("✅ store_vectors 表创建/验证成功！")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建 store_vectors 表失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("创建 store_vectors 表")
    logger.info("=" * 50)
    
    success = asyncio.run(create_store_vectors_table())
    
    if success:
        logger.info("🎉 表创建成功！")
        sys.exit(0)
    else:
        logger.error("💥 表创建失败")
        sys.exit(1)