"""
多智能体LangGraph项目 - 配置管理模块

本模块提供统一的配置管理功能，包括：
- 应用配置
- 数据库配置  
- LLM配置
- 记忆管理配置
- 环境变量管理
"""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field
from typing import Optional, Dict, Any, List
import os
from pathlib import Path


class AppConfig(BaseSettings):
    """应用基础配置"""
    app_name: str = "LangGraph Multi-Agent System"
    version: str = "1.0.0"
    debug: bool = Field(default=False, description="调试模式")
    api_version: str = "v1"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # CORS配置
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    class Config:
        env_prefix = "APP_"
        case_sensitive = False


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    # PostgreSQL配置
    postgres_host: str = Field(default="47.107.169.40", description="PostgreSQL主机")
    postgres_port: int = Field(default=5432, description="PostgreSQL端口")
    postgres_user: str = Field(default="postgres", description="PostgreSQL用户名")
    postgres_password: str = Field(default="postgres123", description="PostgreSQL密码")
    postgres_db: str = Field(default="langgraph", description="PostgreSQL数据库名")
    postgres_ssl_mode: str = Field(default="disable", description="SSL模式")
    
    # Redis配置
    redis_uri: Optional[str] = Field(default=None, description="Redis连接URI")
    redis_host: str = Field(default="47.107.169.40", description="Redis主机")
    redis_port: int = Field(default=3306, description="Redis端口")
    redis_password: Optional[str] = Field(default="cyk085959", description="Redis密码")
    redis_db: int = Field(default=0, description="Redis数据库编号")
    
    # 向量数据库配置
    vector_db_type: str = Field(default="chroma", description="向量数据库类型")
    chroma_host: str = Field(default="localhost", description="Chroma主机")
    chroma_port: int = Field(default=8000, description="Chroma端口")
    
    @property
    def postgres_url(self) -> str:
        """构建PostgreSQL连接URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?sslmode={self.postgres_ssl_mode}"
        )
    
    @property
    def redis_url(self) -> str:
        """构建Redis连接URL"""
        # 优先使用REDIS_URI环境变量
        if self.redis_uri:
            return self.redis_uri
        
        # 否则使用单独的配置项构建URL
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False


class LLMConfig(BaseSettings):
    """LLM模型配置"""
    # OpenAI配置
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API密钥")
    openai_model: str = Field(default="gpt-4", description="OpenAI模型名称")
    openai_temperature: float = Field(default=0.7, description="OpenAI温度参数")
    openai_max_tokens: Optional[int] = Field(default=None, description="最大token数")
    
    # DeepSeek配置
    deepseek_api_key: Optional[str] = Field(default=None, description="DeepSeek API密钥")
    deepseek_model: str = Field(default="deepseek-chat", description="DeepSeek模型名称")
    deepseek_temperature: float = Field(default=0.7, description="DeepSeek温度参数")
    
    # 通义千问配置
    tongyi_api_key: Optional[str] = Field(default=None, description="通义千问API密钥")
    tongyi_model: str = Field(default="qwen-plus", description="通义千问模型名称")
    tongyi_temperature: float = Field(default=0.7, description="通义千问温度参数")
    
    # 阿里DashScope配置
    dashscope_api_key: Optional[str] = Field(default=None, description="阿里DashScope API密钥")
    dashscope_embedding_model: str = Field(default="text-embedding-v4", description="阿里嵌入模型名称")
    
    # 嵌入模型配置（使用OpenAI兼容格式调用阿里云模型）
    embedding_model: str = Field(default="openai:text-embedding-v4", description="嵌入模型")
    embedding_dimensions: int = Field(default=1024, description="嵌入向量维度")
    
    # 默认模型选择
    default_chat_model: str = Field(default="deepseek", description="默认聊天模型")
    default_supervisor_model: str = Field(default="deepseek", description="默认supervisor模型")
    default_research_model: str = Field(default="deepseek", description="默认research模型")
    default_chart_model: str = Field(default="tongyi", description="默认chart模型")
    
    class Config:
        env_prefix = "LLM_"
        case_sensitive = False


class MemoryConfig(BaseSettings):
    """记忆管理配置"""
    # LangMem配置
    memory_store_type: str = Field(default="postgres", description="记忆存储类型")
    memory_namespace_pattern: str = Field(default="memories.{user_id}", description="记忆命名空间模式")
    
    # 记忆管理参数
    semantic_memory_importance_threshold: float = Field(default=0.5, description="语义记忆重要性阈值")
    episodic_memory_max_items: int = Field(default=1000, description="情节记忆最大条目数")
    procedural_memory_confidence_threshold: float = Field(default=0.7, description="程序记忆置信度阈值")
    
    # 记忆搜索配置
    memory_search_limit: int = Field(default=10, description="记忆搜索结果限制")
    memory_search_threshold: float = Field(default=0.7, description="记忆搜索相似度阈值")
    
    class Config:
        env_prefix = "MEMORY_"
        case_sensitive = False


class ToolConfig(BaseSettings):
    """工具配置"""
    # 搜索工具配置
    serper_api_key: Optional[str] = Field(default=None, description="Serper API密钥")
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API密钥")
    
    # MCP服务器配置
    mcp_servers_config_path: str = Field(default="servers_config.json", description="MCP服务器配置文件路径")
    
    # 工具执行配置
    tool_timeout: int = Field(default=30, description="工具执行超时时间(秒)")
    tool_max_retries: int = Field(default=3, description="工具执行最大重试次数")
    
    class Config:
        env_prefix = "TOOL_"
        case_sensitive = False


class SecurityConfig(BaseSettings):
    """安全配置"""
    # JWT配置
    jwt_secret_key: str = Field(default="your-secret-key-change-in-production", description="JWT密钥")
    jwt_algorithm: str = Field(default="HS256", description="JWT算法")
    jwt_expire_minutes: int = Field(default=1440, description="JWT过期时间(分钟)")
    
    # API密钥管理
    api_key_header: str = Field(default="X-API-Key", description="API密钥头部名称")
    
    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


class LoggingConfig(BaseSettings):
    """日志配置"""
    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="日志格式")
    log_file: Optional[str] = Field(default=None, description="日志文件路径")
    log_max_size: int = Field(default=10485760, description="日志文件最大大小(字节)")
    log_backup_count: int = Field(default=5, description="日志文件备份数量")
    
    # 结构化日志配置
    structured_logging: bool = Field(default=True, description="启用结构化日志")
    log_json_format: bool = Field(default=False, description="使用JSON格式日志")
    
    class Config:
        env_prefix = "LOG_"
        case_sensitive = False


class Settings:
    """统一配置管理类"""
    
    def __init__(self):
        self.app = AppConfig()
        self.database = DatabaseConfig()
        self.llm = LLMConfig()
        self.memory = MemoryConfig()
        self.tool = ToolConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
    
    def get_config_dict(self) -> Dict[str, Any]:
        """获取所有配置的字典表示"""
        return {
            "app": self.app.dict(),
            "database": self.database.dict(),
            "llm": self.llm.dict(),
            "memory": self.memory.dict(),
            "tool": self.tool.dict(),
            "security": self.security.dict(),
            "logging": self.logging.dict(),
        }
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 验证必要的API密钥
            if not any([
                self.llm.openai_api_key,
                self.llm.deepseek_api_key,
                self.llm.tongyi_api_key
            ]):
                raise ValueError("至少需要配置一个LLM API密钥")
            
            # 验证数据库连接参数
            if not all([
                self.database.postgres_host,
                self.database.postgres_user,
                self.database.postgres_password,
                self.database.postgres_db
            ]):
                raise ValueError("PostgreSQL连接参数不完整")
            
            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取全局配置实例"""
    return settings


def load_env_file(env_file: str = ".env") -> None:
    """加载环境变量文件"""
    from dotenv import load_dotenv
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"已加载环境变量文件: {env_path}")
    else:
        print(f"环境变量文件不存在: {env_path}")


# 自动加载.env文件
load_env_file()