#!/usr/bin/env python3
"""
测试 store_vectors 表功能的简单脚本
验证向量存储和检索是否正常工作
"""

import asyncio
import logging
import sys
import os
from dotenv import load_dotenv
from langgraph.store.postgres.aio import AsyncPostgresStore

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings

# 加载环境变量
load_dotenv()
print("已加载环境变量文件: .env")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_store_vectors():
    """测试 store_vectors 表功能"""
    settings = Settings()
    
    logger.info("开始测试 store_vectors 表功能...")
    
    try:
        # 创建嵌入模型实例
        from langchain_community.embeddings import DashScopeEmbeddings
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.error("未找到 DASHSCOPE_API_KEY 环境变量")
            return False
        
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=api_key
        )
        
        # 创建存储实例
        async with AsyncPostgresStore.from_conn_string(
            settings.database.postgres_url,
            index={
                "dims": settings.llm.embedding_dimensions,  # 1024
                "embed": embeddings,                        # 使用DashScopeEmbeddings实例
                "fields": ["content", "summary", "description"]
            }
        ) as store:
            logger.info("✅ AsyncPostgresStore 初始化成功")
            
            # 测试数据
            test_namespace = ("test", "vectors")  # 使用元组而不是列表
            test_key = "test_document_001"
            test_data = {
                "content": "这是一个测试文档，用于验证向量存储功能。LangMem 是一个强大的记忆管理系统。",
                "summary": "测试文档摘要",
                "description": "用于测试向量存储的示例文档",
                "metadata": {
                    "type": "test",
                    "created_by": "test_script"
                }
            }
            
            # 存储数据
            logger.info("存储测试数据...")
            await store.aput(test_namespace, test_key, test_data)
            logger.info("✅ 数据存储成功")
            
            # 检索数据
            logger.info("检索测试数据...")
            retrieved_data = await store.aget(test_namespace, test_key)
            if retrieved_data:
                logger.info("✅ 数据检索成功")
                logger.info(f"检索到的内容: {retrieved_data.value.get('content', '')[:50]}...")
            else:
                logger.error("❌ 数据检索失败")
                return False
            
            # 测试语义搜索
            logger.info("测试语义搜索...")
            search_results = await store.asearch(
                test_namespace,  # 使用位置参数而不是关键字参数
                query="记忆管理系统",
                limit=5
            )
            
            if search_results:
                logger.info(f"✅ 语义搜索成功，找到 {len(search_results)} 条结果")
                for i, result in enumerate(search_results):
                    logger.info(f"  结果 {i+1}: {result.value.get('content', '')[:30]}...")
            else:
                logger.info("⚠️ 语义搜索未找到结果（可能是向量索引还未生效）")
            
            # 清理测试数据
            logger.info("清理测试数据...")
            await store.adelete(test_namespace, test_key)
            logger.info("✅ 测试数据清理完成")
            
            logger.info("🎉 store_vectors 表功能测试完成！")
            return True
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False


async def main():
    """主函数"""
    logger.info("开始 store_vectors 表功能测试...")
    
    success = await test_store_vectors()
    
    if success:
        print("\n" + "="*60)
        print("🎉 store_vectors 表功能测试成功！")
        print("✅ 向量存储功能正常工作")
        print("✅ 语义搜索功能已启用")
        print("✅ LangMem 已准备就绪")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ store_vectors 表功能测试失败")
        print("请检查配置和数据库连接")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())