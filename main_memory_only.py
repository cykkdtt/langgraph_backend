#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时启动脚本 - 仅使用内存存储
跳过PostgreSQL连接问题
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 临时禁用PostgreSQL连接
os.environ['DISABLE_POSTGRES'] = 'true'
os.environ['USE_MEMORY_ONLY'] = 'true'

# 导入并运行主应用
if __name__ == "__main__":
    try:
        from main import app
        import uvicorn
        
        print("🚀 启动LangGraph服务 (仅内存模式)")
        print("📝 注意: PostgreSQL连接已禁用，使用内存存储")
        print("🌐 服务地址: http://0.0.0.0:8000")
        print("📊 前端地址: http://localhost:5173")
        print()
        
        # 启动服务
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)