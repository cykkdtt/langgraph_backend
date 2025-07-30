#!/usr/bin/env python
"""
LangMem 数据库要求验证脚本

此脚本用于验证数据库是否满足LangMem的要求，包括：
1. PostgreSQL版本检查
2. 必需扩展检查
3. 表结构验证
4. 索引验证
5. 性能测试
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from core.database import get_database_manager
from core.logging import get_logger
from config.settings import get_settings
from config.memory_config import memory_config

# 初始化日志
logger = get_logger("langmem.validator")


class LangMemValidator:
    """LangMem数据库要求验证器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = get_database_manager()
        self.engine: Optional[AsyncEngine] = None
        self.validation_results: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """初始化数据库连接"""
        await self.db_manager.initialize()
        self.engine = self.db_manager.engine
    
    async def validate_postgresql_version(self) -> Dict[str, Any]:
        """验证PostgreSQL版本"""
        logger.info("验证PostgreSQL版本...")
        
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT version();"))
                version_info = (await result.fetchone())[0]
                
                # 提取版本号
                version_parts = version_info.split()
                version_str = version_parts[1] if len(version_parts) > 1 else "unknown"
                
                # 检查是否满足最低版本要求（13+）
                try:
                    major_version = int(version_str.split('.')[0])
                    meets_requirement = major_version >= 13
                except (ValueError, IndexError):
                    major_version = 0
                    meets_requirement = False
                
                result = {
                    "status": "pass" if meets_requirement else "fail",
                    "version_info": version_info,
                    "major_version": major_version,
                    "requirement": "PostgreSQL 13+",
                    "meets_requirement": meets_requirement
                }
                
                if meets_requirement:
                    logger.info(f"PostgreSQL版本检查通过: {version_str}")
                else:
                    logger.error(f"PostgreSQL版本不满足要求: {version_str} < 13")
                
                return result
                
        except Exception as e:
            logger.error(f"PostgreSQL版本检查失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "meets_requirement": False
            }
    
    async def validate_extensions(self) -> Dict[str, Any]:
        """验证必需的PostgreSQL扩展"""
        logger.info("验证PostgreSQL扩展...")
        
        required_extensions = {
            "vector": "pgvector扩展（向量存储）",
            "uuid-ossp": "UUID生成扩展"
        }
        
        results = {}
        
        try:
            async with self.engine.begin() as conn:
                for ext_name, description in required_extensions.items():
                    # 检查扩展是否已安装
                    result = await conn.execute(text(
                        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = :ext_name);"
                    ), {"ext_name": ext_name})
                    
                    is_installed = (await result.fetchone())[0]
                    
                    # 如果已安装，获取版本信息
                    version = None
                    if is_installed:
                        version_result = await conn.execute(text(
                            "SELECT extversion FROM pg_extension WHERE extname = :ext_name;"
                        ), {"ext_name": ext_name})
                        version = (await version_result.fetchone())[0]
                    
                    results[ext_name] = {
                        "description": description,
                        "installed": is_installed,
                        "version": version,
                        "status": "pass" if is_installed else "fail"
                    }
                    
                    if is_installed:
                        logger.info(f"{description} 已安装，版本: {version}")
                    else:
                        logger.error(f"{description} 未安装")
            
            # 检查整体状态
            all_installed = all(ext["installed"] for ext in results.values())
            
            return {
                "status": "pass" if all_installed else "fail",
                "extensions": results,
                "all_installed": all_installed
            }
            
        except Exception as e:
            logger.error(f"扩展验证失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "all_installed": False
            }
    
    async def validate_tables(self) -> Dict[str, Any]:
        """验证LangMem所需的表"""
        logger.info("验证LangMem所需的表...")
        
        required_tables = {
            "store": "主存储表",
            "store_vectors": "向量存储表",
            "checkpoints": "检查点表",
            "checkpoint_blobs": "检查点二进制数据表",
            "checkpoint_writes": "检查点写入记录表"
        }
        
        results = {}
        
        try:
            async with self.engine.begin() as conn:
                for table_name, description in required_tables.items():
                    # 检查表是否存在
                    result = await conn.execute(text(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = :table_name
                        );
                        """
                    ), {"table_name": table_name})
                    
                    exists = (await result.fetchone())[0]
                    
                    # 如果表存在，获取行数
                    row_count = None
                    if exists:
                        count_result = await conn.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
                        row_count = (await count_result.fetchone())[0]
                    
                    results[table_name] = {
                        "description": description,
                        "exists": exists,
                        "row_count": row_count,
                        "status": "pass" if exists else "fail"
                    }
                    
                    if exists:
                        logger.info(f"{description} ({table_name}) 存在，行数: {row_count}")
                    else:
                        logger.error(f"{description} ({table_name}) 不存在")
            
            # 检查整体状态
            all_exist = all(table["exists"] for table in results.values())
            
            return {
                "status": "pass" if all_exist else "fail",
                "tables": results,
                "all_exist": all_exist
            }
            
        except Exception as e:
            logger.error(f"表验证失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "all_exist": False
            }
    
    async def validate_indexes(self) -> Dict[str, Any]:
        """验证向量索引"""
        logger.info("验证向量索引...")
        
        try:
            async with self.engine.begin() as conn:
                # 检查store_vectors表的向量索引
                result = await conn.execute(text(
                    """
                    SELECT indexname, indexdef 
                    FROM pg_indexes 
                    WHERE tablename = 'store_vectors' 
                    AND indexdef LIKE '%vector%';
                    """
                ))
                
                indexes = await result.fetchall()
                
                vector_indexes = []
                for index in indexes:
                    index_name, index_def = index
                    vector_indexes.append({
                        "name": index_name,
                        "definition": index_def
                    })
                
                has_vector_indexes = len(vector_indexes) > 0
                
                if has_vector_indexes:
                    logger.info(f"找到 {len(vector_indexes)} 个向量索引")
                    for idx in vector_indexes:
                        logger.info(f"  - {idx['name']}")
                else:
                    logger.warning("未找到向量索引")
                
                return {
                    "status": "pass" if has_vector_indexes else "warning",
                    "vector_indexes": vector_indexes,
                    "has_vector_indexes": has_vector_indexes
                }
                
        except Exception as e:
            logger.error(f"索引验证失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "has_vector_indexes": False
            }
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """验证配置"""
        logger.info("验证LangMem配置...")
        
        config_checks = {
            "store_type": {
                "value": memory_config.store_type,
                "expected": "postgres",
                "description": "存储类型"
            },
            "embedding_dims": {
                "value": memory_config.embedding_dims,
                "expected": [384, 512, 768, 1024, 1536, 3072],
                "description": "嵌入向量维度"
            },
            "require_pgvector": {
                "value": memory_config.require_pgvector,
                "expected": True,
                "description": "是否需要pgvector扩展"
            }
        }
        
        results = {}
        all_valid = True
        
        for check_name, check_info in config_checks.items():
            value = check_info["value"]
            expected = check_info["expected"]
            description = check_info["description"]
            
            if isinstance(expected, list):
                is_valid = value in expected
                status_msg = f"值 {value} 在允许范围内: {expected}" if is_valid else f"值 {value} 不在允许范围内: {expected}"
            else:
                is_valid = value == expected
                status_msg = f"值 {value} 符合预期: {expected}" if is_valid else f"值 {value} 不符合预期: {expected}"
            
            results[check_name] = {
                "description": description,
                "value": value,
                "expected": expected,
                "is_valid": is_valid,
                "status": "pass" if is_valid else "fail",
                "message": status_msg
            }
            
            if is_valid:
                logger.info(f"{description}: {status_msg}")
            else:
                logger.error(f"{description}: {status_msg}")
                all_valid = False
        
        return {
            "status": "pass" if all_valid else "fail",
            "checks": results,
            "all_valid": all_valid
        }
    
    async def run_performance_test(self) -> Dict[str, Any]:
        """运行简单的性能测试"""
        logger.info("运行性能测试...")
        
        try:
            import time
            
            async with self.engine.begin() as conn:
                # 测试基本查询性能
                start_time = time.time()
                await conn.execute(text("SELECT 1;"))
                basic_query_time = time.time() - start_time
                
                # 测试store表查询（如果存在）
                store_query_time = None
                try:
                    start_time = time.time()
                    result = await conn.execute(text("SELECT COUNT(*) FROM store;"))
                    await result.fetchone()
                    store_query_time = time.time() - start_time
                except:
                    pass
                
                # 测试向量表查询（如果存在）
                vector_query_time = None
                try:
                    start_time = time.time()
                    result = await conn.execute(text("SELECT COUNT(*) FROM store_vectors;"))
                    await result.fetchone()
                    vector_query_time = time.time() - start_time
                except:
                    pass
            
            return {
                "status": "pass",
                "basic_query_time": basic_query_time,
                "store_query_time": store_query_time,
                "vector_query_time": vector_query_time
            }
            
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """运行完整验证"""
        logger.info("开始LangMem数据库要求验证...")
        
        try:
            await self.initialize()
            
            # 运行各项验证
            self.validation_results = {
                "postgresql_version": await self.validate_postgresql_version(),
                "extensions": await self.validate_extensions(),
                "tables": await self.validate_tables(),
                "indexes": await self.validate_indexes(),
                "configuration": await self.validate_configuration(),
                "performance": await self.run_performance_test()
            }
            
            # 计算整体状态
            overall_status = "pass"
            critical_failures = []
            
            for check_name, result in self.validation_results.items():
                if result.get("status") == "fail":
                    overall_status = "fail"
                    critical_failures.append(check_name)
                elif result.get("status") == "error":
                    overall_status = "error"
                    critical_failures.append(check_name)
            
            self.validation_results["overall"] = {
                "status": overall_status,
                "critical_failures": critical_failures,
                "ready_for_langmem": overall_status == "pass"
            }
            
            # 输出验证结果摘要
            self._print_validation_summary()
            
            return self.validation_results
            
        except Exception as e:
            logger.error(f"验证过程失败: {e}")
            return {
                "overall": {
                    "status": "error",
                    "error": str(e),
                    "ready_for_langmem": False
                }
            }
        finally:
            await self.db_manager.cleanup()
    
    def _print_validation_summary(self) -> None:
        """打印验证结果摘要"""
        print("\n" + "="*60)
        print("LangMem 数据库要求验证结果")
        print("="*60)
        
        for check_name, result in self.validation_results.items():
            if check_name == "overall":
                continue
                
            status = result.get("status", "unknown")
            status_symbol = {
                "pass": "✅",
                "fail": "❌",
                "warning": "⚠️",
                "error": "💥"
            }.get(status, "❓")
            
            print(f"{status_symbol} {check_name.replace('_', ' ').title()}: {status.upper()}")
        
        print("-"*60)
        overall = self.validation_results.get("overall", {})
        overall_status = overall.get("status", "unknown")
        ready = overall.get("ready_for_langmem", False)
        
        if ready:
            print("🎉 数据库已准备好使用LangMem！")
        else:
            print("🚫 数据库尚未准备好使用LangMem")
            failures = overall.get("critical_failures", [])
            if failures:
                print(f"   需要解决的问题: {', '.join(failures)}")
        
        print("="*60)


async def main():
    """主函数"""
    validator = LangMemValidator()
    results = await validator.run_full_validation()
    
    # 返回适当的退出码
    overall_status = results.get("overall", {}).get("status", "error")
    return 0 if overall_status == "pass" else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)