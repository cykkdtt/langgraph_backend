#!/usr/bin/env python
"""
LangMem 数据库要求快速检查脚本

此脚本快速检查数据库是否满足 LangMem 的基本要求：
1. PostgreSQL 版本
2. pgvector 扩展
3. 必需的表结构
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from core.database import get_database_manager
from core.logging import get_logger
from config.settings import get_settings

# 初始化日志
logger = get_logger("langmem.check")


async def quick_check() -> Dict[str, Any]:
    """快速检查 LangMem 数据库要求"""
    
    settings = get_settings()
    db_manager = await get_database_manager()
    
    results = {
        "postgresql_version": {"status": "unknown"},
        "pgvector_extension": {"status": "unknown"},
        "required_tables": {"status": "unknown"},
        "overall_status": "unknown"
    }
    
    try:
        # 数据库管理器已经在get_database_manager中初始化了
        # 使用异步引擎进行查询
        engine = db_manager.async_engine
        
        async with engine.begin() as conn:
            # 1. 检查 PostgreSQL 版本
            try:
                result = await conn.execute(text("SELECT version();"))
                version_info = (await result.fetchone())[0]
                
                # 提取版本号
                import re
                version_match = re.search(r'PostgreSQL (\d+)\.(\d+)', version_info)
                if version_match:
                    major, minor = int(version_match.group(1)), int(version_match.group(2))
                    version_str = f"{major}.{minor}"
                    
                    if major >= 13:
                        results["postgresql_version"] = {
                            "status": "ok",
                            "version": version_str,
                            "message": f"PostgreSQL {version_str} (满足要求 >= 13)"
                        }
                    else:
                        results["postgresql_version"] = {
                            "status": "error",
                            "version": version_str,
                            "message": f"PostgreSQL {version_str} (不满足要求 >= 13)"
                        }
                else:
                    results["postgresql_version"] = {
                        "status": "warning",
                        "message": "无法解析 PostgreSQL 版本"
                    }
                    
            except Exception as e:
                results["postgresql_version"] = {
                    "status": "error",
                    "message": f"检查 PostgreSQL 版本失败: {e}"
                }
            
            # 2. 检查 pgvector 扩展
            try:
                result = await conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
                ))
                is_installed = (await result.fetchone())[0]
                
                if is_installed:
                    # 获取版本
                    version_result = await conn.execute(text(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
                    ))
                    version = (await version_result.fetchone())[0]
                    
                    results["pgvector_extension"] = {
                        "status": "ok",
                        "version": version,
                        "message": f"pgvector {version} 已安装"
                    }
                else:
                    results["pgvector_extension"] = {
                        "status": "error",
                        "message": "pgvector 扩展未安装"
                    }
                    
            except Exception as e:
                results["pgvector_extension"] = {
                    "status": "error",
                    "message": f"检查 pgvector 扩展失败: {e}"
                }
            
            # 3. 检查必需的表
            required_tables = ["store", "store_vectors", "checkpoints"]
            table_status = {}
            
            for table in required_tables:
                try:
                    result = await conn.execute(text(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = '{table}'
                        );
                    """))
                    exists = (await result.fetchone())[0]
                    table_status[table] = "存在" if exists else "不存在"
                except Exception as e:
                    table_status[table] = f"检查失败: {e}"
            
            missing_tables = [table for table, status in table_status.items() if status != "存在"]
            
            if not missing_tables:
                results["required_tables"] = {
                    "status": "ok",
                    "message": "所有必需表都存在",
                    "tables": table_status
                }
            else:
                results["required_tables"] = {
                    "status": "error",
                    "message": f"缺少表: {', '.join(missing_tables)}",
                    "tables": table_status
                }
        
        # 4. 计算整体状态
        all_ok = all(
            result["status"] == "ok" 
            for result in [
                results["postgresql_version"],
                results["pgvector_extension"],
                results["required_tables"]
            ]
        )
        
        results["overall_status"] = "ready" if all_ok else "not_ready"
        
    except Exception as e:
        results["overall_status"] = "error"
        results["error"] = str(e)
        logger.error(f"数据库检查失败: {e}")
    
    finally:
        if 'db_manager' in locals():
            await db_manager.cleanup()
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """打印检查结果"""
    
    print("\n" + "="*60)
    print("🔍 LangMem 数据库要求检查结果")
    print("="*60)
    
    # PostgreSQL 版本
    pg_result = results["postgresql_version"]
    status_icon = "✅" if pg_result["status"] == "ok" else "❌" if pg_result["status"] == "error" else "⚠️"
    print(f"{status_icon} PostgreSQL 版本: {pg_result['message']}")
    
    # pgvector 扩展
    pv_result = results["pgvector_extension"]
    status_icon = "✅" if pv_result["status"] == "ok" else "❌" if pv_result["status"] == "error" else "⚠️"
    print(f"{status_icon} pgvector 扩展: {pv_result['message']}")
    
    # 必需表
    tb_result = results["required_tables"]
    status_icon = "✅" if tb_result["status"] == "ok" else "❌" if tb_result["status"] == "error" else "⚠️"
    print(f"{status_icon} 必需表: {tb_result['message']}")
    
    if "tables" in tb_result:
        for table, status in tb_result["tables"].items():
            table_icon = "✅" if status == "存在" else "❌"
            print(f"    {table_icon} {table}: {status}")
    
    print("-"*60)
    
    # 整体状态
    overall = results["overall_status"]
    if overall == "ready":
        print("🎉 数据库已准备就绪，可以使用 LangMem！")
    elif overall == "not_ready":
        print("⚠️  数据库未完全准备就绪")
        print("\n建议操作：")
        
        if results["pgvector_extension"]["status"] != "ok":
            print("1. 安装 pgvector 扩展:")
            print("   python scripts/setup_pgvector.py")
        
        if results["required_tables"]["status"] != "ok":
            print("2. 初始化数据库表:")
            print("   python scripts/initialize_database.py")
        
        print("3. 重新运行检查:")
        print("   python scripts/quick_check_langmem.py")
    else:
        print("❌ 数据库检查失败")
        if "error" in results:
            print(f"错误: {results['error']}")
    
    print("="*60)


async def main():
    """主函数"""
    results = await quick_check()
    print_results(results)
    
    # 返回适当的退出码
    return 0 if results["overall_status"] == "ready" else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)