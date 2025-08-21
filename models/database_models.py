"""数据库模型定义

本模块定义了LangGraph多智能体系统的数据库模型，包括：
- 用户表模型
- 会话表模型
- 消息表模型
- 智能体状态表模型
- 工具调用表模型
- 系统日志表模型
- 工作流表模型
- 工作流执行表模型
- 记忆表模型

优化特性：
- 高性能索引策略
- 数据分区支持
- 完整的数据验证约束
- 软删除和归档机制
- 向量搜索优化
- 时间序列数据优化
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    TIMESTAMP, BigInteger, Float, event, DDL, desc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from sqlalchemy.sql import func
import uuid
import hashlib

Base = declarative_base()

# 数据库分区策略和优化触发器

# 用户表分区策略（按创建时间分区）
user_partition_ddl = DDL("""
-- 创建用户表分区（按年分区）
CREATE TABLE IF NOT EXISTS users_y2024 PARTITION OF users 
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS users_y2025 PARTITION OF users 
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- 用户搜索向量更新触发器
CREATE OR REPLACE FUNCTION update_user_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', 
        COALESCE(NEW.username, '') || ' ' ||
        COALESCE(NEW.email, '') || ' ' ||
        COALESCE(NEW.full_name, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_user_search_vector ON users;
CREATE TRIGGER trigger_update_user_search_vector
    BEFORE INSERT OR UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_user_search_vector();
""")

# 会话表分区策略（按创建时间分区）
session_partition_ddl = DDL("""
-- 创建会话表分区（按月分区）
CREATE TABLE IF NOT EXISTS sessions_y2024m01 PARTITION OF sessions 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE IF NOT EXISTS sessions_y2024m02 PARTITION OF sessions 
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- 会话统计更新触发器
CREATE OR REPLACE FUNCTION update_session_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- 更新用户的会话统计
    UPDATE users SET 
        session_count = session_count + 1,
        updated_at = NOW()
    WHERE id = NEW.user_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_session_stats ON sessions;
CREATE TRIGGER trigger_update_session_stats
    AFTER INSERT ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_session_stats();
""")

# 消息表分区策略（按创建时间分区）
message_partition_ddl = DDL("""
-- 创建消息表分区（按周分区，高频数据）
CREATE TABLE IF NOT EXISTS messages_y2024w01 PARTITION OF messages 
FOR VALUES FROM ('2024-01-01') TO ('2024-01-08');

CREATE TABLE IF NOT EXISTS messages_y2024w02 PARTITION OF messages 
FOR VALUES FROM ('2024-01-08') TO ('2024-01-15');

-- 消息搜索向量更新触发器
CREATE OR REPLACE FUNCTION update_message_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.content, ''));
    NEW.content_hash := encode(sha256(COALESCE(NEW.content, '')::bytea), 'hex');
    NEW.content_length := length(COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_message_search_vector ON messages;
CREATE TRIGGER trigger_update_message_search_vector
    BEFORE INSERT OR UPDATE ON messages
    FOR EACH ROW EXECUTE FUNCTION update_message_search_vector();

-- 消息统计更新触发器
CREATE OR REPLACE FUNCTION update_message_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- 更新会话统计
    UPDATE sessions SET 
        message_count = message_count + 1,
        total_tokens = total_tokens + COALESCE(NEW.tokens_used, 0),
        total_cost = total_cost + COALESCE(NEW.cost_estimate, 0),
        last_activity_at = NOW(),
        updated_at = NOW()
    WHERE id = NEW.session_id;
    
    -- 更新用户统计
    UPDATE users SET 
        message_count = message_count + 1,
        total_tokens_used = total_tokens_used + COALESCE(NEW.tokens_used, 0),
        total_cost = total_cost + COALESCE(NEW.cost_estimate, 0),
        updated_at = NOW()
    WHERE id = NEW.user_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_message_stats ON messages;
CREATE TRIGGER trigger_update_message_stats
    AFTER INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION update_message_stats();
""")

# 记忆表分区策略（按创建时间和重要性分区）
memory_partition_ddl = DDL("""
-- 创建记忆表分区（按重要性和时间分区）
CREATE TABLE IF NOT EXISTS memories_high_importance PARTITION OF memories 
FOR VALUES FROM (0.8) TO (1.0);

CREATE TABLE IF NOT EXISTS memories_medium_importance PARTITION OF memories 
FOR VALUES FROM (0.5) TO (0.8);

CREATE TABLE IF NOT EXISTS memories_low_importance PARTITION OF memories 
FOR VALUES FROM (0.0) TO (0.5);

-- 记忆向量更新触发器
CREATE OR REPLACE FUNCTION update_memory_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_hash := encode(sha256(COALESCE(NEW.content, '')::bytea), 'hex');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_memory_vector ON memories;
CREATE TRIGGER trigger_update_memory_vector
    BEFORE INSERT OR UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION update_memory_vector();
""")


class User(Base):
    """用户表模型
    
    优化点：
    - 添加了更多复合索引以提升查询性能
    - 增加了数据验证约束
    - 优化了字段长度和类型
    - 添加了软删除支持
    - 支持全文搜索
    - 添加了用户行为分析字段
    - 优化了安全性和隐私保护
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    role = Column(String(20), nullable=False, default="user", index=True)
    status = Column(String(20), nullable=False, default="active", index=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    is_admin = Column(Boolean, nullable=False, default=False)
    is_deleted = Column(Boolean, nullable=False, default=False, index=True)  # 软删除标记
    preferences = Column(JSONB, nullable=True, default={})
    api_key_hash = Column(String(255), nullable=True, unique=True)  # API密钥哈希
    rate_limit_tier = Column(String(20), nullable=False, default="basic")  # 限流等级
    # 全文搜索支持
    search_vector = Column(TSVECTOR, nullable=True)  # 全文搜索向量
    # 用户行为分析
    login_count = Column(Integer, nullable=False, default=0)  # 登录次数
    session_count = Column(Integer, nullable=False, default=0)  # 会话数量
    message_count = Column(Integer, nullable=False, default=0)  # 消息数量
    total_tokens_used = Column(BigInteger, nullable=False, default=0)  # 总令牌使用量
    total_cost = Column(Float, nullable=False, default=0.0)  # 总成本
    # 安全和隐私
    failed_login_attempts = Column(Integer, nullable=False, default=0)  # 失败登录次数
    account_locked_until = Column(DateTime, nullable=True)  # 账户锁定到期时间
    password_changed_at = Column(DateTime, nullable=True)  # 密码修改时间
    two_factor_enabled = Column(Boolean, nullable=False, default=False)  # 双因子认证
    privacy_settings = Column(JSONB, nullable=True, default={})  # 隐私设置
    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True, index=True)
    deleted_at = Column(DateTime, nullable=True)  # 删除时间
    
    # 关系
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    agent_states = relationship("AgentState", back_populates="user", cascade="all, delete-orphan")
    workflows = relationship("Workflow", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="user", cascade="all, delete-orphan")
    
    # 索引优化
    __table_args__ = (
        # 基础索引
        Index('idx_user_email', 'email'),
        Index('idx_user_username', 'username'),
        Index('idx_user_role', 'role'),
        Index('idx_user_status', 'status'),
        Index('idx_user_created_at', 'created_at'),
        Index('idx_user_last_login', 'last_login_at'),
        Index('idx_user_api_key', 'api_key_hash'),
        # 全文搜索索引
        Index('idx_user_search_vector', 'search_vector', postgresql_using='gin'),
        # 性能分析索引
        Index('idx_user_tokens_used', 'total_tokens_used'),
        Index('idx_user_cost', 'total_cost'),
        Index('idx_user_login_count', 'login_count'),
        # 安全相关索引
        Index('idx_user_failed_attempts', 'failed_login_attempts'),
        Index('idx_user_locked_until', 'account_locked_until'),
        Index('idx_user_2fa', 'two_factor_enabled'),
        # 复合索引用于常见查询模式
        Index("idx_users_email_active_deleted", "email", "is_active", "is_deleted"),
        Index("idx_users_username_active_deleted", "username", "is_active", "is_deleted"),
        Index("idx_users_role_status_active", "role", "status", "is_active"),
        Index("idx_users_created_status", "created_at", "status"),
        Index("idx_users_last_login_active", "last_login_at", "is_active"),
        Index("idx_users_api_key_active", "api_key_hash", "is_active"),
        Index('idx_user_role_status', 'role', 'status'),
        Index('idx_user_status_created', 'status', 'created_at'),
        Index('idx_user_active_login', 'is_active', 'last_login_at'),
        Index('idx_user_active_deleted', 'is_active', 'deleted_at'),
        Index('idx_user_role_active', 'role', 'is_active'),
        Index('idx_user_tier_cost', 'rate_limit_tier', 'total_cost'),
        # 分区键索引（按创建时间分区）
        Index('idx_user_partition_key', 'created_at', 'id'),
        # 数据验证约束
        CheckConstraint("role IN ('admin', 'user', 'guest', 'service')", name="check_user_role"),
        CheckConstraint("status IN ('active', 'inactive', 'suspended', 'deleted', 'pending')", name="check_user_status"),
        CheckConstraint("rate_limit_tier IN ('basic', 'premium', 'enterprise', 'unlimited')", name="check_rate_limit_tier"),
        CheckConstraint(r"email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'", name="check_email_format"),
        CheckConstraint("length(username) >= 3", name="check_username_length"),
        CheckConstraint('length(password_hash) >= 8', name='check_password_length'),
        CheckConstraint('login_count >= 0', name='check_login_count_positive'),
        CheckConstraint('session_count >= 0', name='check_session_count_positive'),
        CheckConstraint('message_count >= 0', name='check_message_count_positive'),
        CheckConstraint('total_tokens_used >= 0', name='check_tokens_positive'),
        CheckConstraint('total_cost >= 0', name='check_cost_positive'),
        CheckConstraint('failed_login_attempts >= 0', name='check_failed_attempts_positive'),
    )


class Session(Base):
    """会话表模型
    
    优化点：
    - 添加了会话元数据支持
    - 优化了索引结构
    - 增加了会话状态管理
    - 添加了最后活动时间跟踪
    - 支持会话分析和统计
    - 添加了会话质量评估
    - 优化了时间序列查询
    """
    __tablename__ = "sessions"
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(200), nullable=True)  # 会话标题
    description = Column(Text, nullable=True)  # 会话描述
    session_metadata = Column(JSONB, nullable=True, default={})  # 会话元数据
    is_active = Column(Boolean, nullable=False, default=True)  # 是否活跃
    # 会话统计
    message_count = Column(Integer, nullable=False, default=0)  # 消息数量
    total_tokens = Column(BigInteger, nullable=False, default=0)  # 总令牌数
    total_cost = Column(Float, nullable=False, default=0.0)  # 总成本
    average_response_time = Column(Float, nullable=True)  # 平均响应时间
    # 会话质量
    quality_score = Column(Float, nullable=True)  # 质量评分
    satisfaction_rating = Column(Integer, nullable=True)  # 满意度评分(1-5)
    completion_status = Column(String(20), nullable=False, default='ongoing')  # 完成状态
    # 会话分类
    session_type = Column(String(50), nullable=False, default='chat')  # 会话类型
    category = Column(String(100), nullable=True)  # 会话分类
    tags = Column(JSONB, nullable=True, default=[])  # 标签
    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_activity_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)  # 最后活动时间
    ended_at = Column(DateTime, nullable=True)  # 结束时间
    
    # 关系
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    agent_states = relationship("AgentState", back_populates="session", cascade="all, delete-orphan")
    
    # 索引优化
    __table_args__ = (
        # 基础索引
        Index('idx_session_user_id', 'user_id'),
        Index('idx_session_created_at', 'created_at'),
        Index('idx_session_last_activity', 'last_activity_at'),
        Index('idx_session_ended_at', 'ended_at'),
        # 统计分析索引
        Index('idx_session_message_count', 'message_count'),
        Index('idx_session_total_tokens', 'total_tokens'),
        Index('idx_session_total_cost', 'total_cost'),
        Index('idx_session_quality_score', 'quality_score'),
        Index('idx_session_satisfaction', 'satisfaction_rating'),
        # 分类索引
        Index('idx_session_type', 'session_type'),
        Index('idx_session_category', 'category'),
        Index('idx_session_completion', 'completion_status'),
        Index('idx_session_tags', 'tags', postgresql_using='gin'),
        # 复合索引
        Index('idx_session_user_created', 'user_id', 'created_at'),
        Index('idx_session_user_activity', 'user_id', 'last_activity_at'),
        Index('idx_session_active_created', 'is_active', 'created_at'),
        Index('idx_session_user_active', 'user_id', 'is_active'),
        Index('idx_session_type_created', 'session_type', 'created_at'),
        Index('idx_session_status_ended', 'completion_status', 'ended_at'),
        Index('idx_session_user_type', 'user_id', 'session_type'),
        # 性能分析复合索引
        Index('idx_session_cost_tokens', 'total_cost', 'total_tokens'),
        Index('idx_session_quality_satisfaction', 'quality_score', 'satisfaction_rating'),
        # 分区键索引（按创建时间分区）
        Index('idx_session_partition_key', 'created_at', 'user_id'),
        # 约束
        CheckConstraint('message_count >= 0', name='check_message_count_positive'),
        CheckConstraint('total_tokens >= 0', name='check_tokens_positive'),
        CheckConstraint('total_cost >= 0', name='check_cost_positive'),
        CheckConstraint('average_response_time >= 0', name='check_response_time_positive'),
        CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='check_quality_score_range'),
        CheckConstraint('satisfaction_rating >= 1 AND satisfaction_rating <= 5', name='check_satisfaction_range'),
        CheckConstraint("completion_status IN ('ongoing', 'completed', 'abandoned', 'timeout')", name='check_completion_status'),
        CheckConstraint("session_type IN ('chat', 'workflow', 'analysis', 'training')", name='check_session_type'),
        CheckConstraint('ended_at IS NULL OR ended_at >= created_at', name='check_end_after_start'),
    )


class Message(Base):
    """消息表模型
    
    优化点：
    - 添加了消息状态和优先级字段
    - 优化了索引结构以支持复杂查询
    - 增加了内容长度和性能指标
    - 添加了消息版本控制
    - 支持全文搜索和向量搜索
    - 添加了消息分析和统计
    - 优化了大数据量查询性能
    """
    __tablename__ = "messages"
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(255), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    parent_id = Column(String(255), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True)
    thread_id = Column(String(255), nullable=True, index=True)  # 对话线程ID
    role = Column(String(20), nullable=False, index=True)  # user, assistant, system, tool
    message_type = Column(String(20), nullable=False, default="text", index=True)  # text, image, audio, video, file
    status = Column(String(20), nullable=False, default="sent", index=True)  # sent, delivered, read, processing, failed
    priority = Column(Integer, nullable=False, default=5)  # 1-10, 10为最高优先级
    content = Column(Text, nullable=True)
    content_length = Column(Integer, nullable=True, default=0)  # 内容长度
    content_data = Column(JSONB, nullable=True)  # 结构化内容数据
    message_metadata = Column(JSONB, nullable=True, default={})
    
    # 搜索优化
    content_vector = Column(JSONB, nullable=True)  # 内容向量（用于语义搜索）
    search_vector = Column(TSVECTOR, nullable=True)  # 全文搜索向量
    content_hash = Column(String(64), nullable=True, index=True)  # 内容哈希（去重）
    
    agent_id = Column(String(100), nullable=True, index=True)
    agent_name = Column(String(100), nullable=True)
    agent_version = Column(String(20), nullable=True)  # 智能体版本
    tool_calls = Column(JSONB, nullable=True)  # 工具调用信息
    tool_results = Column(JSONB, nullable=True)  # 工具执行结果
    tokens_used = Column(Integer, nullable=True, default=0)
    cost_estimate = Column(Float, nullable=True, default=0.0)  # 成本估算
    processing_time = Column(Float, nullable=True)  # 处理时间（秒）
    quality_score = Column(Float, nullable=True)  # 质量评分 0.0-1.0
    
    # 消息分析
    sentiment_score = Column(Float, nullable=True)  # 情感分析评分
    complexity_score = Column(Float, nullable=True)  # 复杂度评分
    relevance_score = Column(Float, nullable=True)  # 相关性评分
    engagement_score = Column(Float, nullable=True)  # 参与度评分
    
    # 消息统计
    view_count = Column(Integer, nullable=False, default=0)  # 查看次数
    reaction_count = Column(Integer, nullable=False, default=0)  # 反应次数
    share_count = Column(Integer, nullable=False, default=0)  # 分享次数
    
    is_edited = Column(Boolean, nullable=False, default=False)  # 是否已编辑
    edit_count = Column(Integer, nullable=False, default=0)  # 编辑次数
    version = Column(Integer, nullable=False, default=1)  # 消息版本
    
    # 消息生命周期
    expires_at = Column(DateTime, nullable=True)  # 过期时间
    archived_at = Column(DateTime, nullable=True)  # 归档时间
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)  # 送达时间
    read_at = Column(DateTime, nullable=True)  # 阅读时间
    
    # 关系
    session = relationship("Session", back_populates="messages")
    user = relationship("User", back_populates="messages")
    parent = relationship("Message", remote_side=[id], backref="children")
    tool_call_records = relationship("ToolCall", back_populates="message", cascade="all, delete-orphan")
    
    # 优化索引结构
    __table_args__ = (
        # 基础索引
        Index('idx_message_session_id', 'session_id'),
        Index('idx_message_user_id', 'user_id'),
        Index('idx_message_thread_id', 'thread_id'),
        Index('idx_message_role', 'role'),
        Index('idx_message_type', 'message_type'),
        Index('idx_message_status', 'status'),
        Index('idx_message_priority', 'priority'),
        Index('idx_message_agent_id', 'agent_id'),
        Index('idx_message_parent_id', 'parent_id'),
        Index('idx_message_created_at', 'created_at'),
        # 搜索索引
        Index('idx_message_content_hash', 'content_hash'),
        Index('idx_message_search_vector', 'search_vector', postgresql_using='gin'),
        # 性能指标索引
        Index('idx_message_tokens', 'tokens_used'),
        Index('idx_message_cost', 'cost_estimate'),
        Index('idx_message_processing_time', 'processing_time'),
        Index('idx_message_quality', 'quality_score'),
        # 分析指标索引
        Index('idx_message_sentiment', 'sentiment_score'),
        Index('idx_message_complexity', 'complexity_score'),
        Index('idx_message_relevance', 'relevance_score'),
        Index('idx_message_engagement', 'engagement_score'),
        # 统计索引
        Index('idx_message_view_count', 'view_count'),
        Index('idx_message_reaction_count', 'reaction_count'),
        Index('idx_message_share_count', 'share_count'),
        # 生命周期索引
        Index('idx_message_expires_at', 'expires_at'),
        Index('idx_message_archived_at', 'archived_at'),
        Index('idx_message_delivered_at', 'delivered_at'),
        Index('idx_message_read_at', 'read_at'),
        # 时间序列查询优化
        Index("idx_messages_session_created_desc", "session_id", desc("created_at")),
        Index("idx_messages_user_created_desc", "user_id", desc("created_at")),
        Index("idx_messages_thread_created", "thread_id", "created_at"),
        # 状态和类型查询优化
        Index("idx_messages_role_type_status", "role", "message_type", "status"),
        Index("idx_messages_agent_status", "agent_id", "status"),
        Index("idx_messages_priority_created", "priority", "created_at"),
        # 性能分析查询优化
        Index("idx_messages_tokens_cost", "tokens_used", "cost_estimate"),
        Index("idx_messages_processing_time", "processing_time"),
        Index("idx_messages_quality_score", "quality_score"),
        # 父子关系查询优化
        Index("idx_messages_parent_created", "parent_id", "created_at"),
        # 复合索引 - 基础查询
        Index('idx_message_session_created', 'session_id', 'created_at'),
        Index('idx_message_user_created', 'user_id', 'created_at'),
        Index('idx_message_thread_created', 'thread_id', 'created_at'),
        Index('idx_message_role_created', 'role', 'created_at'),
        Index('idx_message_type_status', 'message_type', 'status'),
        Index('idx_message_agent_created', 'agent_id', 'created_at'),
        Index('idx_message_session_role', 'session_id', 'role'),
        Index('idx_message_user_role', 'user_id', 'role'),
        Index('idx_message_parent_thread', 'parent_id', 'thread_id'),
        # 复合索引 - 性能分析
        Index('idx_message_cost_tokens', 'cost_estimate', 'tokens_used'),
        Index('idx_message_quality_time', 'quality_score', 'processing_time'),
        Index('idx_message_session_cost', 'session_id', 'cost_estimate'),
        Index('idx_message_user_tokens', 'user_id', 'tokens_used'),
        Index('idx_message_agent_quality', 'agent_id', 'quality_score'),
        # 复合索引 - 分析查询
        Index('idx_message_sentiment_engagement', 'sentiment_score', 'engagement_score'),
        Index('idx_message_complexity_quality', 'complexity_score', 'quality_score'),
        Index('idx_message_session_sentiment', 'session_id', 'sentiment_score'),
        # 复合索引 - 统计查询
        Index('idx_message_view_reaction', 'view_count', 'reaction_count'),
        Index('idx_message_session_views', 'session_id', 'view_count'),
        Index('idx_message_user_engagement', 'user_id', 'engagement_score'),
        # 分区键索引（按创建时间分区）
        Index('idx_message_partition_key', 'created_at', 'session_id'),
        # 数据验证约束
        CheckConstraint("role IN ('user', 'assistant', 'system', 'tool', 'function')", name="check_message_role"),
        CheckConstraint("message_type IN ('text', 'image', 'audio', 'video', 'file', 'code', 'markdown')", name="check_message_type"),
        CheckConstraint("status IN ('sent', 'delivered', 'read', 'processing', 'failed', 'cancelled')", name="check_message_status"),
        CheckConstraint("priority >= 1 AND priority <= 10", name="check_message_priority"),
        CheckConstraint("tokens_used >= 0", name="check_tokens_used"),
        CheckConstraint("cost_estimate >= 0.0", name="check_cost_estimate"),
        CheckConstraint("processing_time >= 0.0", name="check_processing_time"),
        CheckConstraint("quality_score IS NULL OR (quality_score >= 0.0 AND quality_score <= 1.0)", name="check_quality_score"),
        CheckConstraint("sentiment_score IS NULL OR (sentiment_score >= -1.0 AND sentiment_score <= 1.0)", name="check_sentiment_score"),
        CheckConstraint("complexity_score IS NULL OR (complexity_score >= 0.0 AND complexity_score <= 1.0)", name="check_complexity_score"),
        CheckConstraint("relevance_score IS NULL OR (relevance_score >= 0.0 AND relevance_score <= 1.0)", name="check_relevance_score"),
        CheckConstraint("engagement_score IS NULL OR (engagement_score >= 0.0 AND engagement_score <= 1.0)", name="check_engagement_score"),
        CheckConstraint("view_count >= 0", name="check_view_count"),
        CheckConstraint("reaction_count >= 0", name="check_reaction_count"),
        CheckConstraint("share_count >= 0", name="check_share_count"),
        CheckConstraint("content_length >= 0", name="check_content_length"),
        CheckConstraint("edit_count >= 0", name="check_edit_count"),
        CheckConstraint("version >= 1", name="check_version"),
        CheckConstraint("expires_at IS NULL OR expires_at > created_at", name="check_expires_after_created"),
        CheckConstraint("archived_at IS NULL OR archived_at >= created_at", name="check_archived_after_created"),
        CheckConstraint("delivered_at IS NULL OR delivered_at >= created_at", name="check_delivered_after_created"),
        CheckConstraint("read_at IS NULL OR read_at >= delivered_at", name="check_read_after_delivered"),
    )


class ToolCall(Base):
    """工具调用表模型"""
    __tablename__ = "tool_calls"
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(String(255), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(String(255), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    tool_name = Column(String(100), nullable=False, index=True)
    tool_input = Column(JSONB, nullable=True)
    tool_output = Column(JSONB, nullable=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, running, success, error
    error_message = Column(Text, nullable=True)
    execution_time = Column(Float, nullable=True)  # 执行时间（秒）
    agent_id = Column(String(100), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # 关系
    message = relationship("Message", back_populates="tool_call_records")
    
    # 索引
    __table_args__ = (
        Index("idx_tool_calls_message_id", "message_id"),
        Index("idx_tool_calls_session_created", "session_id", "created_at"),
        Index("idx_tool_calls_tool_name_status", "tool_name", "status"),
        Index("idx_tool_calls_agent_id", "agent_id"),
        CheckConstraint("status IN ('pending', 'running', 'success', 'error')", name="check_tool_call_status"),
    )


class AgentState(Base):
    """智能体状态表模型"""
    __tablename__ = "agent_states"
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(255), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_id = Column(String(100), nullable=False, index=True)
    agent_name = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default="idle")  # idle, thinking, working, waiting, error
    current_task = Column(Text, nullable=True)
    state_data = Column(JSONB, nullable=True, default={})
    memory_data = Column(JSONB, nullable=True, default={})
    context_data = Column(JSONB, nullable=True, default={})
    performance_metrics = Column(JSONB, nullable=True, default={})
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_activity_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # 关系
    session = relationship("Session", back_populates="agent_states")
    user = relationship("User", back_populates="agent_states")
    
    # 索引
    __table_args__ = (
        Index("idx_agent_states_session_agent", "session_id", "agent_id"),
        Index("idx_agent_states_user_agent", "user_id", "agent_id"),
        Index("idx_agent_states_agent_type_status", "agent_type", "status"),
        Index("idx_agent_states_last_activity", "last_activity_at"),
        CheckConstraint("status IN ('idle', 'thinking', 'working', 'waiting', 'error')", name="check_agent_status"),
    )


class SystemLog(Base):
    """系统日志表模型"""
    __tablename__ = "system_logs"
    
    id = Column(BigInteger, primary_key=True, index=True)
    level = Column(String(10), nullable=False, index=True)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    logger_name = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    module = Column(String(100), nullable=True, index=True)
    function = Column(String(100), nullable=True)
    line_number = Column(Integer, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    session_id = Column(String(255), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    request_id = Column(String(255), nullable=True, index=True)
    extra_data = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # 索引
    __table_args__ = (
        Index("idx_system_logs_level_created", "level", "created_at"),
        Index("idx_system_logs_logger_created", "logger_name", "created_at"),
        Index("idx_system_logs_module_created", "module", "created_at"),
        Index("idx_system_logs_user_created", "user_id", "created_at"),
        Index("idx_system_logs_session_created", "session_id", "created_at"),
        Index("idx_system_logs_request_id", "request_id"),
        CheckConstraint("level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')", name="check_log_level"),
    )


class Workflow(Base):
    """工作流表模型
    
    优化点：
    - 添加了工作流分类和复杂度评估
    - 增加了性能指标和统计信息
    - 优化了版本管理和发布控制
    - 添加了访问控制和权限管理
    """
    __tablename__ = "workflows"
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=True, index=True)  # 工作流分类
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    definition = Column(JSONB, nullable=False)  # 工作流定义
    version = Column(String(20), nullable=False, default="1.0.0", index=True)
    major_version = Column(Integer, nullable=False, default=1)  # 主版本号
    minor_version = Column(Integer, nullable=False, default=0)  # 次版本号
    patch_version = Column(Integer, nullable=False, default=0)  # 补丁版本号
    status = Column(String(20), nullable=False, default="draft", index=True)  # draft, active, inactive, archived, deprecated
    is_public = Column(Boolean, nullable=False, default=False, index=True)
    is_featured = Column(Boolean, nullable=False, default=False)  # 是否推荐
    is_template = Column(Boolean, nullable=False, default=False)  # 是否为模板
    complexity_score = Column(Float, nullable=False, default=1.0)  # 复杂度评分 1.0-10.0
    estimated_duration = Column(Integer, nullable=True)  # 预估执行时间（秒）
    max_concurrent_executions = Column(Integer, nullable=False, default=1)  # 最大并发执行数
    tags = Column(JSONB, nullable=True, default=[])
    workflow_metadata = Column(JSONB, nullable=True, default={})
    permissions = Column(JSONB, nullable=True, default={})  # 权限配置
    # 统计信息
    execution_count = Column(Integer, nullable=False, default=0)  # 执行次数
    success_count = Column(Integer, nullable=False, default=0)  # 成功次数
    failure_count = Column(Integer, nullable=False, default=0)  # 失败次数
    average_duration = Column(Float, nullable=True)  # 平均执行时间
    last_execution_at = Column(DateTime, nullable=True)  # 最后执行时间
    usage_count = Column(Integer, nullable=False, default=0)  # 使用次数
    rating = Column(Float, nullable=True)  # 用户评分 1.0-5.0
    rating_count = Column(Integer, nullable=False, default=0)  # 评分次数
    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)  # 发布时间
    archived_at = Column(DateTime, nullable=True)  # 归档时间
    
    # 关系
    user = relationship("User", back_populates="workflows")
    executions = relationship("WorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")
    
    # 优化索引结构
    __table_args__ = (
        # 用户和状态查询优化
        Index("idx_workflows_user_status_public", "user_id", "status", "is_public"),
        Index("idx_workflows_category_status", "category", "status"),
        Index("idx_workflows_public_featured", "is_public", "is_featured"),
        Index("idx_workflows_template_category", "is_template", "category"),
        # 版本管理优化
        Index("idx_workflows_name_version", "name", "major_version", "minor_version", "patch_version"),
        Index("idx_workflows_version_status", "version", "status"),
        # 性能和统计查询优化
        Index("idx_workflows_complexity_duration", "complexity_score", "estimated_duration"),
        Index("idx_workflows_execution_stats", "execution_count", "success_count"),
        Index("idx_workflows_rating_usage", "rating", "usage_count"),
        Index("idx_workflows_last_execution", "last_execution_at"),
        # 时间序列查询优化
        Index("idx_workflows_created_status", "created_at", "status"),
        Index("idx_workflows_published_rating", "published_at", "rating"),
        # 数据验证约束
        CheckConstraint("status IN ('draft', 'active', 'inactive', 'archived', 'deprecated', 'testing')", name="check_workflow_status"),
        CheckConstraint("complexity_score >= 1.0 AND complexity_score <= 10.0", name="check_complexity_score"),
        CheckConstraint("estimated_duration IS NULL OR estimated_duration > 0", name="check_estimated_duration"),
        CheckConstraint("max_concurrent_executions >= 1", name="check_max_concurrent_executions"),
        CheckConstraint("execution_count >= 0", name="check_execution_count"),
        CheckConstraint("success_count >= 0", name="check_success_count"),
        CheckConstraint("failure_count >= 0", name="check_failure_count"),
        CheckConstraint("success_count + failure_count <= execution_count", name="check_execution_consistency"),
        CheckConstraint("average_duration IS NULL OR average_duration >= 0.0", name="check_average_duration"),
        CheckConstraint("usage_count >= 0", name="check_usage_count"),
        CheckConstraint("rating IS NULL OR (rating >= 1.0 AND rating <= 5.0)", name="check_rating"),
        CheckConstraint("rating_count >= 0", name="check_rating_count"),
        CheckConstraint("major_version >= 1", name="check_major_version"),
        CheckConstraint("minor_version >= 0", name="check_minor_version"),
        CheckConstraint("patch_version >= 0", name="check_patch_version"),
    )


class WorkflowExecution(Base):
    """工作流执行表模型"""
    __tablename__ = "workflow_executions"
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String(255), ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(String(255), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed, cancelled
    input_data = Column(JSONB, nullable=True)
    output_data = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    execution_log = Column(JSONB, nullable=True, default=[])
    current_step = Column(String(100), nullable=True)
    progress = Column(Float, nullable=False, default=0.0)  # 0.0 - 1.0
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    workflow = relationship("Workflow", back_populates="executions")
    
    # 索引
    __table_args__ = (
        Index("idx_workflow_executions_workflow_status", "workflow_id", "status"),
        Index("idx_workflow_executions_user_created", "user_id", "created_at"),
        Index("idx_workflow_executions_session_id", "session_id"),
        Index("idx_workflow_executions_status_created", "status", "created_at"),
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", name="check_execution_status"),
        CheckConstraint("progress >= 0.0 AND progress <= 1.0", name="check_execution_progress"),
    )


class Memory(Base):
    """记忆表模型
    
    优化点：
    - 添加了记忆关联和层次结构
    - 增加了向量搜索优化字段
    - 优化了记忆衰减和巩固机制
    - 添加了记忆质量评估
    """
    __tablename__ = "memories"
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(String(255), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    parent_memory_id = Column(String(255), ForeignKey("memories.id", ondelete="SET NULL"), nullable=True, index=True)
    memory_type = Column(String(20), nullable=False, index=True)  # semantic, episodic, procedural, working
    memory_subtype = Column(String(30), nullable=True, index=True)  # 记忆子类型
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=True, index=True)  # 内容哈希，用于去重
    content_vector = Column("content_vector", String, nullable=True)  # 向量表示
    vector_model = Column(String(50), nullable=True)  # 向量模型名称
    vector_dimension = Column(Integer, nullable=True)  # 向量维度
    memory_metadata = Column(JSONB, nullable=True, default={})
    # 重要性和质量评估
    importance_score = Column(Float, nullable=False, default=0.5, index=True)  # 重要性评分 0.0-1.0
    confidence_score = Column(Float, nullable=False, default=0.5)  # 置信度评分 0.0-1.0
    relevance_score = Column(Float, nullable=False, default=0.5)  # 相关性评分 0.0-1.0
    quality_score = Column(Float, nullable=False, default=0.5)  # 质量评分 0.0-1.0
    # 访问和使用统计
    access_count = Column(Integer, nullable=False, default=0, index=True)
    retrieval_count = Column(Integer, nullable=False, default=0)  # 检索次数
    consolidation_count = Column(Integer, nullable=False, default=0)  # 巩固次数
    last_accessed_at = Column(DateTime, nullable=True, index=True)
    last_retrieved_at = Column(DateTime, nullable=True)
    last_consolidated_at = Column(DateTime, nullable=True)
    # 记忆衰减和生命周期
    decay_rate = Column(Float, nullable=False, default=0.1)  # 衰减率
    strength = Column(Float, nullable=False, default=1.0)  # 记忆强度
    is_consolidated = Column(Boolean, nullable=False, default=False)  # 是否已巩固
    is_active = Column(Boolean, nullable=False, default=True, index=True)  # 是否活跃
    # 关联和标签
    tags = Column(JSONB, nullable=True, default=[])  # 标签
    associations = Column(JSONB, nullable=True, default=[])  # 关联记忆ID列表
    context_data = Column(JSONB, nullable=True, default={})  # 上下文数据
    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True, index=True)  # 过期时间
    archived_at = Column(DateTime, nullable=True)  # 归档时间
    
    # 关系
    user = relationship("User", back_populates="memories")
    parent_memory = relationship("Memory", remote_side=[id], backref="child_memories")
    
    # 优化索引结构
    __table_args__ = (
        # 基础查询优化
        Index("idx_memories_user_type_active", "user_id", "memory_type", "is_active"),
        Index("idx_memories_session_type_active", "session_id", "memory_type", "is_active"),
        Index("idx_memories_parent_type", "parent_memory_id", "memory_type"),
        # 评分和质量查询优化
        Index("idx_memories_importance_quality", "importance_score", "quality_score"),
        Index("idx_memories_confidence_relevance", "confidence_score", "relevance_score"),
        Index("idx_memories_strength_consolidated", "strength", "is_consolidated"),
        # 访问模式优化
        Index("idx_memories_access_count_last", "access_count", "last_accessed_at"),
        Index("idx_memories_retrieval_count", "retrieval_count"),
        Index("idx_memories_last_retrieved", "last_retrieved_at"),
        # 内容和向量查询优化
        Index("idx_memories_content_hash", "content_hash"),
        Index("idx_memories_vector_model_dim", "vector_model", "vector_dimension"),
        Index("idx_memories_subtype_active", "memory_subtype", "is_active"),
        # 生命周期管理优化
        Index("idx_memories_expires_active", "expires_at", "is_active"),
        Index("idx_memories_created_strength", "created_at", "strength"),
        Index("idx_memories_decay_rate", "decay_rate"),
        # 数据验证约束
        CheckConstraint("memory_type IN ('semantic', 'episodic', 'procedural', 'working', 'meta')", name="check_memory_type"),
        CheckConstraint("importance_score >= 0.0 AND importance_score <= 1.0", name="check_importance_score"),
        CheckConstraint("confidence_score >= 0.0 AND confidence_score <= 1.0", name="check_confidence_score"),
        CheckConstraint("relevance_score >= 0.0 AND relevance_score <= 1.0", name="check_relevance_score"),
        CheckConstraint("quality_score >= 0.0 AND quality_score <= 1.0", name="check_quality_score"),
        CheckConstraint("decay_rate >= 0.0 AND decay_rate <= 1.0", name="check_decay_rate"),
        CheckConstraint("strength >= 0.0 AND strength <= 1.0", name="check_strength"),
        CheckConstraint("access_count >= 0", name="check_access_count"),
        CheckConstraint("retrieval_count >= 0", name="check_retrieval_count"),
        CheckConstraint("consolidation_count >= 0", name="check_consolidation_count"),
        CheckConstraint("vector_dimension IS NULL OR vector_dimension > 0", name="check_vector_dimension"),
        CheckConstraint("length(content) > 0", name="check_content_not_empty"),
    )


# 事件监听器：绑定DDL语句到表创建事件
# 这些监听器会在表创建后自动执行分区和触发器创建

@event.listens_for(User.__table__, 'after_create')
def create_user_partitions_and_triggers(target, connection, **kw):
    """用户表创建后执行分区和触发器创建"""
    connection.execute(user_partition_ddl)

@event.listens_for(Session.__table__, 'after_create')
def create_session_partitions_and_triggers(target, connection, **kw):
    """会话表创建后执行分区和触发器创建"""
    connection.execute(session_partition_ddl)

@event.listens_for(Message.__table__, 'after_create')
def create_message_partitions_and_triggers(target, connection, **kw):
    """消息表创建后执行分区和触发器创建"""
    connection.execute(message_partition_ddl)

@event.listens_for(Memory.__table__, 'after_create')
def create_memory_partitions_and_triggers(target, connection, **kw):
    """记忆表创建后执行分区和触发器创建"""
    connection.execute(memory_partition_ddl)


# 数据库优化工具函数

def create_database_indexes(engine):
    """创建所有优化索引"""
    with engine.connect() as connection:
        # 创建全文搜索索引
        connection.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_search_gin ON users USING gin(search_vector);")
        connection.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_message_search_gin ON messages USING gin(search_vector);")
        
        # 创建向量搜索索引（如果使用pgvector扩展）
        try:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            connection.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_message_content_vector ON messages USING ivfflat (content_vector vector_cosine_ops);")
            connection.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_content_vector ON memories USING ivfflat (content_vector vector_cosine_ops);")
        except Exception:
            # 如果pgvector不可用，跳过向量索引创建
            pass

def optimize_database_settings(engine):
    """优化数据库设置"""
    with engine.connect() as connection:
        # 设置PostgreSQL优化参数
        optimization_settings = [
            "SET shared_preload_libraries = 'pg_stat_statements';",
            "SET max_connections = 200;",
            "SET shared_buffers = '256MB';",
            "SET effective_cache_size = '1GB';",
            "SET maintenance_work_mem = '64MB';",
            "SET checkpoint_completion_target = 0.9;",
            "SET wal_buffers = '16MB';",
            "SET default_statistics_target = 100;",
            "SET random_page_cost = 1.1;",
            "SET effective_io_concurrency = 200;",
        ]
        
        for setting in optimization_settings:
            try:
                connection.execute(setting)
            except Exception:
                # 某些设置可能需要重启数据库，跳过错误
                pass

def create_maintenance_procedures(engine):
    """创建数据库维护存储过程"""
    with engine.connect() as connection:
        # 创建数据清理存储过程
        cleanup_procedure = """
        CREATE OR REPLACE FUNCTION cleanup_old_data()
        RETURNS void AS $$
        BEGIN
            -- 清理过期的消息
            DELETE FROM messages WHERE expires_at < NOW();
            
            -- 清理过期的记忆
            DELETE FROM memories WHERE expires_at < NOW();
            
            -- 归档旧的会话
            UPDATE sessions SET archived_at = NOW() 
            WHERE created_at < NOW() - INTERVAL '1 year' AND archived_at IS NULL;
            
            -- 更新统计信息
            ANALYZE users;
            ANALYZE sessions;
            ANALYZE messages;
            ANALYZE memories;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        connection.execute(cleanup_procedure)
        
        # 创建性能监控存储过程
        monitoring_procedure = """
        CREATE OR REPLACE FUNCTION get_performance_stats()
        RETURNS TABLE(
            table_name text,
            total_size text,
            index_size text,
            row_count bigint
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                schemaname||'.'||tablename as table_name,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
                pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as index_size,
                n_tup_ins + n_tup_upd + n_tup_del as row_count
            FROM pg_stat_user_tables 
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        connection.execute(monitoring_procedure)