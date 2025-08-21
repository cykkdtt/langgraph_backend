#!/usr/bin/env python
"""
简化的 LangMem 数据库检查脚本

专门检查 PostgreSQL 和 pgvector 扩展，不依赖复杂的数据库管理器
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg
from config.settings import get_settings
from core.logging import get_logger

# 初始化日志
logger = get_logger("langmem.simple_check")


async def simple_check() -> Dict[str, Any]:
    """简化的 LangMem 数据库要求检查"""
    
    settings = get_settings()
    
    results = {
        "postgresql_version": {"status": "unknown"},
        "pgvector_extension": {"status": "unknown"},
        "required_tables": {"status": "unknown"},
        "overall_status": "unknown"
    }
    
    conn = None
    try:
        # 直接连接到PostgreSQL
        postgres_url = settings.database.url
        logger.info(f"连接到数据库: {postgres_url.split('@')[1] if '@' in postgres_url else 'localhost'}")
        
        conn = await asyncpg.connect(postgres_url)
        
        # 1. 检查 PostgreSQL 版本
        try:
            version_info = await conn.fetchval("SELECT version();")
            
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
            is_installed = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
            )
            
            if is_installed:
                # 获取版本
                version = await conn.fetchval(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
                )
                
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
                exists = await conn.fetchval(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = '{table}'
                    );
                """)
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
        if conn:
            await conn.close()
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """打印检查结果"""
    
    print("\n" + "="*60)
    print("🔍 LangMem 数据库要求检查结果")
    print("="*60)
    
    # PostgreSQL 版本
    pg_result = results["postgresql_version"]
    status_icon = "✅" if pg_result["status"] == "ok" else "❌" if pg_result["status"] == "error" else "⚠️"
    message = pg_result.get("message", "检查失败")
    print(f"{status_icon} PostgreSQL 版本: {message}")
    
    # pgvector 扩展
    pv_result = results["pgvector_extension"]
    status_icon = "✅" if pv_result["status"] == "ok" else "❌" if pv_result["status"] == "error" else "⚠️"
    message = pv_result.get("message", "检查失败")
    print(f"{status_icon} pgvector 扩展: {message}")
    
    # 必需表
    tb_result = results["required_tables"]
    status_icon = "✅" if tb_result["status"] == "ok" else "❌" if tb_result["status"] == "error" else "⚠️"
    message = tb_result.get("message", "检查失败")
    print(f"{status_icon} 必需表: {message}")
    
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
        print("   python scripts/simple_check_langmem.py")
    else:
        print("❌ 数据库检查失败")
        if "error" in results:
            print(f"错误: {results['error']}")
    
    print("="*60)


async def main():
    """主函数"""
    results = await simple_check()
    print_results(results)
    
    # 返回适当的退出码
    return 0 if results["overall_status"] == "ready" else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)