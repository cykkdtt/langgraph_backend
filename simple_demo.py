#!/usr/bin/env python3
"""
LangGraph多智能体系统 - 简化演示脚本

这个脚本演示系统的基本功能，不依赖复杂的数据库配置。
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查.env文件
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  .env文件不存在，从模板创建...")
        template_file = project_root / ".env.template"
        if template_file.exists():
            import shutil
            shutil.copy(template_file, env_file)
            print("✅ 已从.env.template创建.env文件")
        else:
            print("❌ .env.template文件也不存在")
            return False
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
        print("   需要Python 3.9或更高版本")
        return False
    
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    return True

def demo_project_structure():
    """演示项目结构"""
    print("\n" + "="*50)
    print("📁 项目结构演示")
    print("="*50)
    
    def show_tree(path, prefix="", max_depth=2, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = []
        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith('.') and item.name not in ['.env', '.env.template']:
                    continue
                items.append(item)
        except PermissionError:
            return
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}{'/' if item.is_dir() else ''}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                show_tree(item, next_prefix, max_depth, current_depth + 1)
    
    print("langgraph_study/")
    show_tree(project_root, max_depth=3)

def demo_configuration():
    """演示配置管理"""
    print("\n" + "="*50)
    print("⚙️  配置管理演示")
    print("="*50)
    
    try:
        # 尝试导入配置
        from config.settings import Settings
        
        print("✅ 配置模块导入成功")
        
        # 显示配置信息
        settings = Settings()
        print(f"📊 应用配置:")
        print(f"  - 调试模式: {settings.app.debug}")
        print(f"  - 主机地址: {settings.app.host}")
        print(f"  - 端口: {settings.app.port}")
        print(f"  - 环境: {settings.app.environment}")
        
        print(f"🗄️  数据库配置:")
        print(f"  - 类型: {settings.database.type}")
        print(f"  - 主机: {settings.database.postgres.host}")
        print(f"  - 端口: {settings.database.postgres.port}")
        
        print(f"🤖 LLM配置:")
        print(f"  - 默认提供商: {settings.llm.default_provider}")
        print(f"  - 默认模型: {settings.llm.default_model}")
        
    except Exception as e:
        print(f"❌ 配置导入失败: {e}")

def demo_core_modules():
    """演示核心模块"""
    print("\n" + "="*50)
    print("🧩 核心模块演示")
    print("="*50)
    
    modules = [
        ("agents", "智能体模块"),
        ("memory", "记忆管理模块"),
        ("tools", "工具集成模块"),
        ("database", "数据库管理模块"),
        ("logging", "日志系统模块"),
        ("error", "错误处理模块"),
        ("env", "环境管理模块"),
        ("checkpoint", "检查点管理模块")
    ]
    
    for module_name, description in modules:
        module_path = project_root / "core" / module_name
        if module_path.exists():
            print(f"✅ {description}: core/{module_name}/")
            
            # 显示模块文件
            try:
                files = [f for f in module_path.iterdir() if f.is_file() and f.suffix == '.py']
                for file in sorted(files)[:3]:  # 只显示前3个文件
                    print(f"    📄 {file.name}")
                if len(files) > 3:
                    print(f"    ... 还有 {len(files) - 3} 个文件")
            except:
                pass
        else:
            print(f"❌ {description}: 模块不存在")

def demo_scripts():
    """演示管理脚本"""
    print("\n" + "="*50)
    print("📜 管理脚本演示")
    print("="*50)
    
    scripts = [
        ("setup_project.py", "项目设置脚本"),
        ("initialize_database.py", "数据库初始化脚本"),
        ("initialize_system.py", "系统初始化脚本")
    ]
    
    scripts_dir = project_root / "scripts"
    if scripts_dir.exists():
        for script_name, description in scripts:
            script_path = scripts_dir / script_name
            if script_path.exists():
                print(f"✅ {description}: scripts/{script_name}")
                
                # 显示脚本大小
                size = script_path.stat().st_size
                print(f"    📏 文件大小: {size} 字节")
            else:
                print(f"❌ {description}: 脚本不存在")
    else:
        print("❌ scripts目录不存在")

def demo_documentation():
    """演示文档系统"""
    print("\n" + "="*50)
    print("📚 文档系统演示")
    print("="*50)
    
    docs = [
        ("README.md", "项目说明文档"),
        ("spec/01_core_architecture.md", "核心架构设计"),
        ("spec/02_api_design.md", "API设计规范"),
        ("spec/03_agent_implementation.md", "智能体实现指南"),
        ("spec/04_deployment_ops.md", "部署运维指南"),
        ("spec/05_langmem_integration.md", "LangMem集成说明"),
        ("spec/FAQ.md", "常见问题解答")
    ]
    
    for doc_path, description in docs:
        full_path = project_root / doc_path
        if full_path.exists():
            print(f"✅ {description}: {doc_path}")
            
            # 显示文档大小
            size = full_path.stat().st_size
            print(f"    📏 文件大小: {size} 字节")
        else:
            print(f"❌ {description}: 文档不存在")

def demo_usage_examples():
    """演示使用示例"""
    print("\n" + "="*50)
    print("💡 使用示例演示")
    print("="*50)
    
    print("🚀 快速启动命令:")
    print("  python start.py                    # 一键启动系统")
    print("  python main.py                     # 启动Web服务")
    print("  python demo.py                     # 运行功能演示")
    
    print("\n🔧 管理命令:")
    print("  python scripts/setup_project.py    # 项目设置")
    print("  python scripts/initialize_database.py  # 数据库初始化")
    print("  python scripts/initialize_system.py    # 系统初始化")
    
    print("\n🌐 Web访问:")
    print("  http://localhost:8000              # 主页")
    print("  http://localhost:8000/docs         # API文档")
    print("  http://localhost:8000/health       # 健康检查")
    print("  http://localhost:8000/status       # 系统状态")
    
    print("\n📡 API示例:")
    print("  curl -X POST http://localhost:8000/chat \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"content\": \"你好\", \"user_id\": \"user123\"}'")

async def main():
    """主演示函数"""
    print("🎭 LangGraph多智能体系统 - 项目演示")
    print("=" * 60)
    
    # 环境检查
    if not check_environment():
        print("\n❌ 环境检查失败，请检查配置")
        return
    
    # 运行各项演示
    demo_project_structure()
    demo_configuration()
    demo_core_modules()
    demo_scripts()
    demo_documentation()
    demo_usage_examples()
    
    print("\n" + "="*60)
    print("🎉 项目演示完成！")
    print("💡 提示：")
    print("  1. 配置.env文件中的API密钥")
    print("  2. 运行 python start.py 启动系统")
    print("  3. 访问 http://localhost:8000 使用Web界面")
    print("  4. 查看 spec/FAQ.md 获取更多帮助")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())