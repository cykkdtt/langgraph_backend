#!/usr/bin/env python
"""
简化的 pgvector 扩展安装脚本

此脚本直接使用 asyncpg 连接数据库，避免复杂的依赖
"""

import os
import sys
import asyncio
import logging
import platform
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg
from core.logging import get_logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化日志
logger = get_logger("pgvector.simple_setup")


class SimplePgVectorSetup:
    """简化的 pgvector 扩展设置器"""
    
    def __init__(self):
        self.database_url = os.getenv("POSTGRES_URI")
        if not self.database_url:
            raise ValueError("未找到 POSTGRES_URI 环境变量")
        
        logger.info(f"连接到数据库: {self.database_url.split('@')[1] if '@' in self.database_url else self.database_url}")
    
    async def check_pgvector_installed(self) -> Dict[str, Any]:
        """检查 pgvector 扩展是否已安装"""
        logger.info("检查 pgvector 扩展...")
        
        try:
            conn = await asyncpg.connect(self.database_url)
            
            try:
                # 检查扩展是否已安装
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
                )
                
                version = None
                if result:
                    # 获取版本信息
                    version = await conn.fetchval(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
                    )
                
                return {
                    "installed": result,
                    "version": version,
                    "status": "success"
                }
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"检查 pgvector 扩展失败: {e}")
            return {
                "installed": False,
                "version": None,
                "status": "error",
                "error": str(e)
            }
    
    async def install_pgvector_extension(self) -> Dict[str, Any]:
        """安装 pgvector 扩展"""
        logger.info("尝试安装 pgvector 扩展...")
        
        try:
            conn = await asyncpg.connect(self.database_url)
            
            try:
                # 尝试安装扩展
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # 验证安装
                is_installed = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
                )
                
                if is_installed:
                    # 获取版本信息
                    version = await conn.fetchval(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
                    )
                    
                    logger.info(f"pgvector 扩展安装成功，版本: {version}")
                    return {
                        "installed": True,
                        "version": version,
                        "status": "success"
                    }
                else:
                    logger.error("pgvector 扩展安装失败")
                    return {
                        "installed": False,
                        "status": "failed"
                    }
                    
            finally:
                await conn.close()
                    
        except Exception as e:
            logger.error(f"安装 pgvector 扩展失败: {e}")
            return {
                "installed": False,
                "status": "error",
                "error": str(e)
            }
    
    async def test_vector_operations(self) -> Dict[str, Any]:
        """测试向量操作"""
        logger.info("测试向量操作...")
        
        try:
            conn = await asyncpg.connect(self.database_url)
            
            try:
                # 创建测试表
                await conn.execute("""
                    DROP TABLE IF EXISTS test_vectors;
                    CREATE TABLE test_vectors (
                        id SERIAL PRIMARY KEY,
                        embedding vector(3)
                    );
                """)
                
                # 插入测试数据
                await conn.execute("""
                    INSERT INTO test_vectors (embedding) VALUES 
                    ('[1,2,3]'),
                    ('[4,5,6]'),
                    ('[7,8,9]');
                """)
                
                # 测试相似性搜索
                results = await conn.fetch("""
                    SELECT id, embedding, embedding <-> '[1,2,3]' AS distance
                    FROM test_vectors
                    ORDER BY distance
                    LIMIT 2;
                """)
                
                # 清理测试表
                await conn.execute("DROP TABLE test_vectors;")
                
                logger.info("向量操作测试成功")
                return {
                    "status": "success",
                    "test_results": [
                        {"id": row["id"], "embedding": str(row["embedding"]), "distance": float(row["distance"])}
                        for row in results
                    ]
                }
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"向量操作测试失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_installation_instructions(self) -> Dict[str, str]:
        """获取不同平台的安装指导"""
        system = platform.system().lower()
        
        instructions = {
            "linux": """
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-contrib
sudo apt install postgresql-14-pgvector

# CentOS/RHEL
sudo yum install postgresql-contrib
sudo yum install pgvector

# 或者从源码编译
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
            """,
            "darwin": """
# macOS with Homebrew
brew install pgvector

# 或者从源码编译
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
            """,
            "windows": """
# Windows
# 1. 下载预编译的二进制文件
# 2. 或者使用 WSL 安装 Linux 版本
# 3. 或者使用 Docker

# Docker 方式（推荐）
docker run -d \\
  --name postgres-pgvector \\
  -e POSTGRES_PASSWORD=password \\
  -p 5432:5432 \\
  pgvector/pgvector:pg15
            """
        }
        
        return {
            "current_system": system,
            "instructions": instructions.get(system, instructions["linux"]),
            "docker_option": """
# 使用 Docker（适用于所有平台）
docker run -d \\
  --name postgres-pgvector \\
  -e POSTGRES_PASSWORD=password \\
  -p 5432:5432 \\
  pgvector/pgvector:pg15
            """
        }
    
    async def setup_vector_indexes(self) -> Dict[str, Any]:
        """设置向量索引"""
        logger.info("设置向量索引...")
        
        try:
            conn = await asyncpg.connect(self.database_url)
            
            try:
                # 检查 store_vectors 表是否存在
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'store_vectors'
                    );
                """)
                
                if not table_exists:
                    logger.warning("store_vectors 表不存在，跳过索引创建")
                    return {
                        "status": "skipped",
                        "reason": "store_vectors table does not exist"
                    }
                
                # 创建向量索引
                indexes_created = []
                
                # HNSW 余弦相似度索引
                try:
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_store_vectors_embedding_cosine 
                        ON store_vectors USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64);
                    """)
                    indexes_created.append("hnsw_cosine")
                except Exception as e:
                    logger.warning(f"创建 HNSW 余弦索引失败: {e}")
                
                # HNSW L2 距离索引
                try:
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_store_vectors_embedding_l2 
                        ON store_vectors USING hnsw (embedding vector_l2_ops)
                        WITH (m = 16, ef_construction = 64);
                    """)
                    indexes_created.append("hnsw_l2")
                except Exception as e:
                    logger.warning(f"创建 HNSW L2 索引失败: {e}")
                
                logger.info(f"向量索引设置完成，创建了 {len(indexes_created)} 个索引")
                return {
                    "status": "success",
                    "indexes_created": indexes_created
                }
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"设置向量索引失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run_full_setup(self) -> Dict[str, Any]:
        """运行完整的 pgvector 设置"""
        logger.info("开始 pgvector 完整设置...")
        
        results = {}
        
        try:
            # 1. 检查当前状态
            check_result = await self.check_pgvector_installed()
            results["check"] = check_result
            
            if not check_result["installed"]:
                logger.info("pgvector 扩展未安装，尝试安装...")
                
                # 2. 尝试安装扩展
                install_result = await self.install_pgvector_extension()
                results["install"] = install_result
                
                if not install_result["installed"]:
                    # 提供安装指导
                    instructions = self.get_installation_instructions()
                    results["instructions"] = instructions
                    
                    logger.error("pgvector 扩展安装失败，需要在数据库服务器上手动安装")
                    print("\n" + "="*60)
                    print("❌ pgvector 扩展安装失败")
                    print("="*60)
                    print("⚠️  pgvector 扩展需要在数据库服务器上安装，不是在本地客户端！")
                    print()
                    print("请联系数据库管理员在服务器上执行以下操作：")
                    print()
                    print("1. 安装 pgvector 扩展包：")
                    print(instructions["instructions"])
                    print()
                    print("2. 在数据库中启用扩展：")
                    print("   psql -d langgraph -c \"CREATE EXTENSION IF NOT EXISTS vector;\"")
                    print()
                    print("3. 验证安装：")
                    print("   psql -d langgraph -c \"SELECT * FROM pg_extension WHERE extname = 'vector';\"")
                    print("="*60)
                    
                    return results
            
            # 3. 测试向量操作
            test_result = await self.test_vector_operations()
            results["test"] = test_result
            
            if test_result["status"] == "success":
                logger.info("向量操作测试通过")
                
                # 4. 设置向量索引
                index_result = await self.setup_vector_indexes()
                results["indexes"] = index_result
                
                logger.info("pgvector 设置完成")
                print("\n" + "="*60)
                print("🎉 pgvector 扩展设置成功！")
                print("="*60)
                print(f"版本: {check_result.get('version', 'unknown')}")
                print("功能: 向量存储和相似性搜索已就绪")
                if index_result.get("indexes_created"):
                    print(f"索引: {', '.join(index_result['indexes_created'])}")
                print("="*60)
            else:
                logger.error("向量操作测试失败")
            
            return results
            
        except Exception as e:
            logger.error(f"pgvector 设置失败: {e}")
            results["error"] = str(e)
            return results


async def main():
    """主函数"""
    try:
        setup = SimplePgVectorSetup()
        results = await setup.run_full_setup()
        
        # 检查整体状态
        success = (
            results.get("check", {}).get("installed", False) and
            results.get("test", {}).get("status") == "success"
        )
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
        print(f"\n❌ 错误: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)