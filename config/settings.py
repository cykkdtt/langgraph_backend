"""应用配置管理模块

统一管理应用配置、环境变量、数据库连接、缓存设置等配置信息。
支持多环境配置和动态配置更新。
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from functools import lru_cache
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


class Environment(str, Enum):
    """环境类型枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    
    # 支持直接使用完整的PostgreSQL URI
    postgres_uri: Optional[str] = Field(default=None, env="DB_POSTGRES_URI")
    
    # 兼容性属性：postgres_url (指向postgres_uri)
    @property
    def postgres_url(self) -> str:
        """兼容性属性，返回postgres_uri或构建的URL"""
        if self.postgres_uri:
            return self.postgres_uri
        return self.url
    
    # 或者使用单独的连接参数
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    username: str = Field(default="postgres", env="DB_USERNAME")
    password: str = Field(default="postgres", env="DB_PASSWORD")
    database: str = Field(default="langgraph", env="DB_DATABASE")
    
    # 连接池配置
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    
    # SSL配置
    ssl_mode: str = Field(default="prefer", env="DB_SSL_MODE")
    ssl_cert: Optional[str] = Field(default=None, env="DB_SSL_CERT")
    ssl_key: Optional[str] = Field(default=None, env="DB_SSL_KEY")
    ssl_ca: Optional[str] = Field(default=None, env="DB_SSL_CA")
    
    # 性能配置
    echo: bool = Field(default=False, env="DB_ECHO")
    echo_pool: bool = Field(default=False, env="DB_ECHO_POOL")
    
    @property
    def url(self) -> str:
        """构建数据库连接URL"""
        if self.postgres_uri:
            # 如果提供了完整的URI，直接使用
            return self.postgres_uri.replace("postgresql+asyncpg://", "postgresql://")
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
    
    @property
    def async_url(self) -> str:
        """构建异步数据库连接URL"""
        if self.postgres_uri:
            # AsyncPostgresSaver期望标准的postgresql://格式
            if self.postgres_uri.startswith("postgresql+asyncpg://"):
                return self.postgres_uri.replace("postgresql+asyncpg://", "postgresql://")
            elif self.postgres_uri.startswith("postgresql://"):
                return self.postgres_uri
            else:
                return f"postgresql://{self.postgres_uri}"
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
    
    model_config = SettingsConfigDict(env_prefix="DB_", case_sensitive=False)


class RedisSettings(BaseSettings):
    """Redis配置"""
    
    # Redis连接URI
    redis_uri: Optional[str] = Field(default=None, env="REDIS_URI")
    
    # 兼容性属性：redis_url (指向redis_uri)
    @property
    def redis_url(self) -> str:
        """兼容性属性，返回redis_uri或构建的URL"""
        if self.redis_uri:
            return self.redis_uri
        return self.url
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    database: int = Field(default=0, env="REDIS_DATABASE")
    
    # 连接池配置
    max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    
    # SSL配置
    ssl: bool = Field(default=False, env="REDIS_SSL")
    ssl_cert_reqs: Optional[str] = Field(default=None, env="REDIS_SSL_CERT_REQS")
    ssl_ca_certs: Optional[str] = Field(default=None, env="REDIS_SSL_CA_CERTS")
    ssl_certfile: Optional[str] = Field(default=None, env="REDIS_SSL_CERTFILE")
    ssl_keyfile: Optional[str] = Field(default=None, env="REDIS_SSL_KEYFILE")
    
    @property
    def url(self) -> str:
        """构建Redis连接URL"""
        auth_part = f":{self.password}@" if self.password else ""
        protocol = "rediss" if self.ssl else "redis"
        return f"{protocol}://{auth_part}{self.host}:{self.port}/{self.database}"
    
    model_config = SettingsConfigDict(env_prefix="REDIS_", case_sensitive=False)


class JWTSettings(BaseSettings):
    """JWT配置"""
    
    secret_key: str = Field(env="JWT_SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    
    # 令牌配置
    issuer: str = Field(default="langgraph-backend", env="JWT_ISSUER")
    audience: str = Field(default="langgraph-users", env="JWT_AUDIENCE")
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if not v or len(v) < 32:
            raise ValueError('JWT secret key must be at least 32 characters long')
        return v
    
    model_config = SettingsConfigDict(env_prefix="JWT_", case_sensitive=False)


class CORSSettings(BaseSettings):
    """CORS配置"""
    
    allow_origins: List[str] = Field(default=["*"], env="CORS_ALLOW_ORIGINS")
    allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    expose_headers: List[str] = Field(default=[], env="CORS_EXPOSE_HEADERS")
    max_age: int = Field(default=600, env="CORS_MAX_AGE")
    
    @validator('allow_origins', pre=True)
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('allow_methods', pre=True)
    def parse_methods(cls, v):
        if isinstance(v, str):
            return [method.strip() for method in v.split(',')]
        return v
    
    @validator('allow_headers', pre=True)
    def parse_headers(cls, v):
        if isinstance(v, str):
            return [header.strip() for header in v.split(',')]
        return v
    
    model_config = SettingsConfigDict(env_prefix="CORS_", case_sensitive=False)


class SecuritySettings(BaseSettings):
    """安全配置"""
    
    # 密码策略
    password_min_length: int = Field(default=8, env="SECURITY_PASSWORD_MIN_LENGTH")
    password_require_uppercase: bool = Field(default=True, env="SECURITY_PASSWORD_REQUIRE_UPPERCASE")
    password_require_lowercase: bool = Field(default=True, env="SECURITY_PASSWORD_REQUIRE_LOWERCASE")
    password_require_numbers: bool = Field(default=True, env="SECURITY_PASSWORD_REQUIRE_NUMBERS")
    password_require_symbols: bool = Field(default=True, env="SECURITY_PASSWORD_REQUIRE_SYMBOLS")
    
    # 会话配置
    session_timeout_minutes: int = Field(default=60, env="SECURITY_SESSION_TIMEOUT_MINUTES")
    max_login_attempts: int = Field(default=5, env="SECURITY_MAX_LOGIN_ATTEMPTS")
    lockout_duration_minutes: int = Field(default=15, env="SECURITY_LOCKOUT_DURATION_MINUTES")
    
    # API限流
    rate_limit_requests: int = Field(default=100, env="SECURITY_RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, env="SECURITY_RATE_LIMIT_WINDOW_SECONDS")
    
    # 加密配置
    encryption_key: Optional[str] = Field(default=None, env="SECURITY_ENCRYPTION_KEY")
    
    model_config = SettingsConfigDict(env_prefix="SECURITY_", case_sensitive=False)


class LangGraphSettings(BaseSettings):
    """LangGraph特定配置"""
    
    # 工作流配置
    max_workflow_nodes: int = Field(default=100, env="LANGGRAPH_MAX_WORKFLOW_NODES")
    max_workflow_execution_time: int = Field(default=3600, env="LANGGRAPH_MAX_WORKFLOW_EXECUTION_TIME")
    workflow_checkpoint_interval: int = Field(default=10, env="LANGGRAPH_WORKFLOW_CHECKPOINT_INTERVAL")
    
    # 记忆配置
    memory_vector_dimension: int = Field(default=1536, env="LANGGRAPH_MEMORY_VECTOR_DIMENSION")
    memory_similarity_threshold: float = Field(default=0.8, env="LANGGRAPH_MEMORY_SIMILARITY_THRESHOLD")
    memory_max_entries: int = Field(default=10000, env="LANGGRAPH_MEMORY_MAX_ENTRIES")
    
    # 智能体配置
    agent_max_iterations: int = Field(default=50, env="LANGGRAPH_AGENT_MAX_ITERATIONS")
    agent_timeout_seconds: int = Field(default=300, env="LANGGRAPH_AGENT_TIMEOUT_SECONDS")
    
    # 工具配置
    tool_execution_timeout: int = Field(default=60, env="LANGGRAPH_TOOL_EXECUTION_TIMEOUT")
    max_tool_calls_per_turn: int = Field(default=10, env="LANGGRAPH_MAX_TOOL_CALLS_PER_TURN")
    
    model_config = SettingsConfigDict(env_prefix="LANGGRAPH_", case_sensitive=False)


class LLMSettings(BaseSettings):
    """LLM配置"""
    
    # 默认模型配置
    default_chat_model: str = Field(default="deepseek-chat", env="LLM_DEFAULT_CHAT_MODEL")
    default_supervisor_model: str = Field(default="deepseek-chat", env="LLM_DEFAULT_SUPERVISOR_MODEL")
    default_research_model: str = Field(default="deepseek-chat", env="LLM_DEFAULT_RESEARCH_MODEL")
    default_chart_model: str = Field(default="deepseek-chat", env="LLM_DEFAULT_CHART_MODEL")
    
    # 模型参数
    default_temperature: float = Field(default=0.7, env="LLM_DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=2000, env="LLM_DEFAULT_MAX_TOKENS")
    
    # API配置
    request_timeout: int = Field(default=60, env="LLM_REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, env="LLM_MAX_RETRIES")
    
    model_config = SettingsConfigDict(env_prefix="LLM_", case_sensitive=False)


class MonitoringSettings(BaseSettings):
    """监控配置"""
    
    # Prometheus配置
    enable_metrics: bool = Field(default=True, env="MONITORING_ENABLE_METRICS")
    metrics_port: int = Field(default=8001, env="MONITORING_METRICS_PORT")
    metrics_path: str = Field(default="/metrics", env="MONITORING_METRICS_PATH")
    
    # 日志配置
    log_level: LogLevel = Field(default=LogLevel.INFO, env="MONITORING_LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="MONITORING_LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default=None, env="MONITORING_LOG_FILE")
    log_max_size: int = Field(default=10485760, env="MONITORING_LOG_MAX_SIZE")  # 10MB
    log_backup_count: int = Field(default=5, env="MONITORING_LOG_BACKUP_COUNT")
    
    # 健康检查配置
    health_check_interval: int = Field(default=30, env="MONITORING_HEALTH_CHECK_INTERVAL")
    
    # 告警配置
    enable_alerts: bool = Field(default=True, env="MONITORING_ENABLE_ALERTS")
    alert_webhook_url: Optional[str] = Field(default=None, env="MONITORING_ALERT_WEBHOOK_URL")
    
    model_config = SettingsConfigDict(env_prefix="MONITORING_", case_sensitive=False)


class Settings(BaseSettings):
    """主配置类"""
    
    # 基础配置
    app_name: str = Field(default="LangGraph Backend", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # 服务器配置
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # API配置
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    docs_url: Optional[str] = Field(default="/docs", env="DOCS_URL")
    redoc_url: Optional[str] = Field(default="/redoc", env="REDOC_URL")
    openapi_url: Optional[str] = Field(default="/openapi.json", env="OPENAPI_URL")
    
    # 子配置
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    langgraph: LangGraphSettings = Field(default_factory=LangGraphSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    # 外部服务配置
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    
    @validator('environment', pre=True)
    def parse_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator('debug')
    def set_debug_based_on_env(cls, v, values):
        environment = values.get('environment')
        if environment == Environment.DEVELOPMENT:
            return True
        return v
    
    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """是否为测试环境"""
        return self.environment == Environment.TESTING
    
    def get_database_url(self, async_driver: bool = False) -> str:
        """获取数据库连接URL"""
        return self.database.async_url if async_driver else self.database.url
    
    def get_redis_url(self) -> str:
        """获取Redis连接URL"""
        return self.redis.url
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._settings: Optional[Settings] = None
        self._config_file_path: Optional[Path] = None
        self.logger = logging.getLogger(__name__)
    
    @lru_cache(maxsize=1)
    def get_settings(self) -> Settings:
        """获取配置实例（缓存）"""
        if self._settings is None:
            self._settings = Settings()
            self._validate_settings()
        return self._settings
    
    def _validate_settings(self):
        """验证配置"""
        settings = self._settings
        
        # 验证必需的配置
        if not settings.jwt.secret_key:
            raise ValueError("JWT_SECRET_KEY is required")
        
        # 生产环境特殊验证
        if settings.is_production:
            if settings.debug:
                self.logger.warning("Debug mode is enabled in production")
            
            if not settings.database.password:
                raise ValueError("Database password is required in production")
            
            if settings.cors.allow_origins == ["*"]:
                self.logger.warning("CORS allows all origins in production")
    
    def reload_settings(self):
        """重新加载配置"""
        self.get_settings.cache_clear()
        self._settings = None
        return self.get_settings()
    
    def update_setting(self, key: str, value: Any):
        """更新单个配置项"""
        settings = self.get_settings()
        
        # 支持嵌套键，如 "database.host"
        keys = key.split('.')
        obj = settings
        
        for k in keys[:-1]:
            obj = getattr(obj, k)
        
        setattr(obj, keys[-1], value)
        
        # 清除缓存以强制重新验证
        self.get_settings.cache_clear()
    
    def export_config(self, file_path: str = None, format: str = "json") -> Union[str, Dict[str, Any]]:
        """导出配置"""
        settings = self.get_settings()
        config_dict = settings.dict()
        
        if format.lower() == "json":
            config_json = json.dumps(config_dict, indent=2, default=str)
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(config_json)
                return file_path
            
            return config_json
        
        elif format.lower() == "env":
            env_lines = []
            
            def flatten_dict(d, prefix=""):
                for key, value in d.items():
                    if isinstance(value, dict):
                        flatten_dict(value, f"{prefix}{key.upper()}_")
                    else:
                        env_key = f"{prefix}{key.upper()}"
                        env_lines.append(f"{env_key}={value}")
            
            flatten_dict(config_dict)
            env_content = "\n".join(env_lines)
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(env_content)
                return file_path
            
            return env_content
        
        return config_dict
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        settings = self.get_settings()
        
        return {
            "app_info": {
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment.value,
                "debug": settings.debug
            },
            "server": {
                "host": settings.host,
                "port": settings.port,
                "workers": settings.workers
            },
            "database": {
                "host": settings.database.host,
                "port": settings.database.port,
                "database": settings.database.database,
                "pool_size": settings.database.pool_size
            },
            "redis": {
                "host": settings.redis.host,
                "port": settings.redis.port,
                "database": settings.redis.database
            },
            "security": {
                "jwt_algorithm": settings.jwt.algorithm,
                "session_timeout": settings.security.session_timeout_minutes,
                "rate_limit": f"{settings.security.rate_limit_requests}/{settings.security.rate_limit_window_seconds}s"
            },
            "monitoring": {
                "metrics_enabled": settings.monitoring.enable_metrics,
                "log_level": settings.monitoring.log_level.value,
                "alerts_enabled": settings.monitoring.enable_alerts
            }
        }


# 全局配置管理器实例
config_manager = ConfigManager()


# 便捷函数
def get_settings() -> Settings:
    """获取应用配置"""
    return config_manager.get_settings()


def get_database_url(async_driver: bool = False) -> str:
    """获取数据库连接URL"""
    return get_settings().get_database_url(async_driver)


def get_redis_url() -> str:
    """获取Redis连接URL"""
    return get_settings().get_redis_url()


def is_development() -> bool:
    """是否为开发环境"""
    return get_settings().is_development


def is_production() -> bool:
    """是否为生产环境"""
    return get_settings().is_production


def is_testing() -> bool:
    """是否为测试环境"""
    return get_settings().is_testing