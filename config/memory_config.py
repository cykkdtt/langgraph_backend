"""
LangMem 记忆管理配置模块

提供记忆系统的配置管理，包括存储配置、嵌入模型配置、
记忆管理配置、缓存配置和清理配置等。
"""

import os
from typing import Optional, Literal
from pydantic import Field

# 兼容不同版本的 Pydantic
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class MemoryConfig(BaseSettings):
    """LangMem 记忆管理配置类"""
    
    # 存储配置
    store_type: Literal["postgres", "memory"] = Field(
        default="postgres", 
        description="存储类型：postgres 或 memory"
    )
    postgres_url: str = Field(
        default_factory=lambda: os.getenv(
            "LANGMEM_POSTGRES_URL", 
            "postgresql://user:pass@localhost:5432/langmem"
        ),
        description="PostgreSQL 连接URL"
    )
    
    # 数据库表配置
    store_table_name: str = Field(
        default="store",
        description="主存储表名称"
    )
    store_vectors_table_name: str = Field(
        default="store_vectors",
        description="向量存储表名称"
    )
    checkpoints_table_name: str = Field(
        default="checkpoints",
        description="检查点表名称"
    )
    
    # PostgreSQL扩展配置
    require_pgvector: bool = Field(
        default=True,
        description="是否需要pgvector扩展"
    )
    pgvector_version: str = Field(
        default="0.5.0",
        description="所需的pgvector版本"
    )
    
    # 向量索引配置
    vector_index_type: Literal["hnsw", "ivfflat"] = Field(
        default="hnsw",
        description="向量索引类型"
    )
    hnsw_m: int = Field(
        default=16,
        description="HNSW索引的m参数"
    )
    hnsw_ef_construction: int = Field(
        default=64,
        description="HNSW索引的ef_construction参数"
    )
    
    # 嵌入模型配置
    embedding_model: str = Field(
        default="openai:text-embedding-3-small",
        description="嵌入模型名称"
    )
    embedding_dims: int = Field(
        default=1536,
        description="嵌入向量维度"
    )
    
    # 记忆管理配置
    max_memories_per_namespace: int = Field(
        default=10000,
        description="每个命名空间最大记忆数量"
    )
    auto_consolidate: bool = Field(
        default=True,
        description="是否自动整合记忆"
    )
    consolidate_threshold: int = Field(
        default=1000,
        description="记忆整合阈值"
    )
    
    # 缓存配置
    redis_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        description="Redis 连接URL"
    )
    cache_ttl: int = Field(
        default=3600,
        description="缓存TTL（秒）"
    )
    
    # 清理配置
    cleanup_interval: int = Field(
        default=86400,  # 24小时
        description="清理间隔（秒）"
    )
    backup_enabled: bool = Field(
        default=True,
        description="是否启用备份"
    )
    backup_interval: int = Field(
        default=604800,  # 7天
        description="备份间隔（秒）"
    )
    
    # 命名空间配置
    namespace_prefix: str = Field(
        default="langgraph_agents",
        description="命名空间前缀"
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "LANGMEM_"
        extra = "ignore"  # 忽略额外的环境变量


# 全局配置实例
memory_config = MemoryConfig()