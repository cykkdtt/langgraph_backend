# 数据库表结构和关系参考文档

## 📋 概述

本文档详细描述了LangGraph多智能体系统的数据库表结构、字段作用和表间关系，为开发人员提供快速参考。

## 🗂️ 表分类

### 1. 自定义应用表 (6个)
- `users` - 用户管理
- `sessions` - 会话管理  
- `messages` - 消息记录
- `tool_calls` - 工具调用记录
- `agent_states` - 智能体状态
- `system_logs` - 系统日志

### 2. LangGraph核心表 (5个)
- `checkpoints` - 检查点数据
- `checkpoint_blobs` - 检查点数据块
- `checkpoint_writes` - 检查点写入记录
- `store` - LangMem存储
- `store_vectors` - LangMem向量数据

### 3. 迁移管理表 (3个)
- `checkpoint_migrations` - 检查点迁移记录
- `store_migrations` - 存储迁移记录
- `vector_migrations` - 向量迁移记录

---

## 🔗 表关系图

```
users (用户)
  └── sessions (会话) [user_id → users.id]
      ├── messages (消息) [session_id → sessions.id]
      ├── agent_states (智能体状态) [session_id → sessions.id]
      └── tool_calls (工具调用) [session_id → sessions.id]
              └── messages (消息) [message_id → messages.id]

store (存储)
  └── store_vectors (向量) [prefix+key → store.prefix+key]
```

---

## 📊 详细表结构

### 1. users (用户表)
**作用**: 管理系统用户账户信息

| 字段名 | 类型 | 约束 | 作用 |
|--------|------|------|------|
| `id` | SERIAL | PRIMARY KEY | 用户唯一标识 |
| `username` | VARCHAR(50) | UNIQUE NOT NULL | 用户名，登录凭证 |
| `email` | VARCHAR(100) | UNIQUE NOT NULL | 邮箱地址，联系方式 |
| `password_hash` | VARCHAR(255) | NOT NULL | 密码哈希值，安全存储 |
| `is_active` | BOOLEAN | DEFAULT TRUE | 账户是否激活 |
| `is_admin` | BOOLEAN | DEFAULT FALSE | 是否为管理员 |
| `created_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 创建时间 |
| `updated_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 更新时间 |

**索引**:
- `idx_users_username` - 用户名查询
- `idx_users_email` - 邮箱查询  
- `idx_users_is_active` - 活跃用户筛选

**触发器**: `update_users_updated_at` - 自动更新updated_at字段

---

### 2. sessions (会话表)
**作用**: 管理用户与智能体的对话会话

| 字段名 | 类型 | 约束 | 作用 |
|--------|------|------|------|
| `id` | UUID | PRIMARY KEY | 会话唯一标识 |
| `user_id` | INTEGER | FK → users.id | 所属用户 |
| `title` | VARCHAR(200) | | 会话标题 |
| `description` | TEXT | | 会话描述 |
| `metadata` | JSONB | DEFAULT '{}' | 扩展元数据 |
| `is_active` | BOOLEAN | DEFAULT TRUE | 会话是否活跃 |
| `created_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 创建时间 |
| `updated_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 更新时间 |

**外键关系**:
- `user_id` → `users.id` (CASCADE删除)

**索引**:
- `idx_sessions_user_id` - 用户会话查询
- `idx_sessions_is_active` - 活跃会话筛选
- `idx_sessions_created_at` - 时间排序

**触发器**: `update_sessions_updated_at` - 自动更新updated_at字段

---

### 3. messages (消息表)
**作用**: 存储会话中的所有消息记录

| 字段名 | 类型 | 约束 | 作用 |
|--------|------|------|------|
| `id` | UUID | PRIMARY KEY | 消息唯一标识 |
| `session_id` | UUID | FK → sessions.id | 所属会话 |
| `role` | VARCHAR(20) | CHECK IN ('user', 'assistant', 'system') | 消息角色 |
| `content` | TEXT | NOT NULL | 消息内容 |
| `metadata` | JSONB | DEFAULT '{}' | 消息元数据 |
| `created_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 创建时间 |

**外键关系**:
- `session_id` → `sessions.id` (CASCADE删除)

**索引**:
- `idx_messages_session_id` - 会话消息查询
- `idx_messages_role` - 角色筛选
- `idx_messages_created_at` - 时间排序

---

### 4. tool_calls (工具调用表)
**作用**: 记录智能体的工具调用历史和结果

| 字段名 | 类型 | 约束 | 作用 |
|--------|------|------|------|
| `id` | UUID | PRIMARY KEY | 调用记录唯一标识 |
| `session_id` | UUID | FK → sessions.id | 所属会话 |
| `message_id` | UUID | FK → messages.id | 关联消息 |
| `tool_name` | VARCHAR(100) | NOT NULL | 工具名称 |
| `tool_input` | JSONB | NOT NULL | 工具输入参数 |
| `tool_output` | JSONB | | 工具输出结果 |
| `status` | VARCHAR(20) | CHECK IN ('pending', 'success', 'error') | 执行状态 |
| `error_message` | TEXT | | 错误信息 |
| `execution_time` | FLOAT | | 执行耗时(秒) |
| `created_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 创建时间 |
| `completed_at` | TIMESTAMP WITH TIME ZONE | | 完成时间 |

**外键关系**:
- `session_id` → `sessions.id` (CASCADE删除)
- `message_id` → `messages.id` (CASCADE删除)

**索引**:
- `idx_tool_calls_session_id` - 会话工具调用查询
- `idx_tool_calls_message_id` - 消息关联查询
- `idx_tool_calls_tool_name` - 工具名称筛选
- `idx_tool_calls_status` - 状态筛选

---

### 5. agent_states (智能体状态表)
**作用**: 保存智能体在会话中的状态数据

| 字段名 | 类型 | 约束 | 作用 |
|--------|------|------|------|
| `id` | UUID | PRIMARY KEY | 状态记录唯一标识 |
| `session_id` | UUID | FK → sessions.id | 所属会话 |
| `agent_name` | VARCHAR(100) | NOT NULL | 智能体名称 |
| `state_data` | JSONB | NOT NULL | 状态数据 |
| `version` | INTEGER | DEFAULT 1 | 状态版本号 |
| `created_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 创建时间 |

**外键关系**:
- `session_id` → `sessions.id` (CASCADE删除)

**索引**:
- `idx_agent_states_session_id` - 会话状态查询
- `idx_agent_states_agent_name` - 智能体筛选

---

### 6. system_logs (系统日志表)
**作用**: 记录系统运行日志，用于调试和监控

| 字段名 | 类型 | 约束 | 作用 |
|--------|------|------|------|
| `id` | BIGSERIAL | PRIMARY KEY | 日志记录唯一标识 |
| `level` | VARCHAR(20) | NOT NULL | 日志级别 |
| `logger_name` | VARCHAR(100) | NOT NULL | 记录器名称 |
| `message` | TEXT | NOT NULL | 日志消息 |
| `module` | VARCHAR(100) | | 模块名称 |
| `function_name` | VARCHAR(100) | | 函数名称 |
| `line_number` | INTEGER | | 行号 |
| `exception` | TEXT | | 异常信息 |
| `extra_data` | JSONB | DEFAULT '{}' | 额外数据 |
| `created_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 创建时间 |

**索引**:
- `idx_system_logs_level` - 日志级别筛选
- `idx_system_logs_logger_name` - 记录器筛选
- `idx_system_logs_created_at` - 时间排序

---

## 🔧 LangGraph核心表

### 7. checkpoints (检查点表)
**作用**: LangGraph工作流检查点数据，支持状态恢复

| 字段名 | 类型 | 约束 | 作用 |
|--------|------|------|------|
| `thread_id` | TEXT | NOT NULL | 线程标识 |
| `checkpoint_ns` | TEXT | NOT NULL | 检查点命名空间 |
| `checkpoint_id` | TEXT | NOT NULL | 检查点标识 |
| `parent_checkpoint_id` | TEXT | | 父检查点标识 |
| `type` | TEXT | | 检查点类型 |
| `checkpoint` | JSONB | | 检查点数据 |
| `metadata` | JSONB | DEFAULT '{}' | 元数据 |
| `created_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 创建时间 |

---

### 8. checkpoint_blobs (检查点数据块表)
**作用**: 存储大型检查点数据块

| 字段名 | 类型 | 约束 | 作用 |
|--------|------|------|------|
| `thread_id` | TEXT | NOT NULL | 线程标识 |
| `checkpoint_ns` | TEXT | NOT NULL | 检查点命名空间 |
| `channel` | TEXT | NOT NULL | 通道名称 |
| `version` | TEXT | NOT NULL | 版本号 |
| `type` | TEXT | NOT NULL | 数据类型 |
| `blob` | BYTEA | | 二进制数据 |
| `created_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 创建时间 |

---

### 9. checkpoint_writes (检查点写入表)
**作用**: 记录检查点写入操作

| 字段名 | 类型 | 约束 | 作用 |
|--------|------|------|------|
| `thread_id` | TEXT | NOT NULL | 线程标识 |
| `checkpoint_ns` | TEXT | NOT NULL | 检查点命名空间 |
| `checkpoint_id` | TEXT | NOT NULL | 检查点标识 |
| `task_id` | TEXT | NOT NULL | 任务标识 |
| `idx` | INTEGER | NOT NULL | 索引 |
| `channel` | TEXT | NOT NULL | 通道名称 |
| `type` | TEXT | | 写入类型 |
| `value` | JSONB | | 写入值 |
| `created_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 创建时间 |

---

### 10. store (LangMem存储表)
**作用**: LangMem记忆存储的主表，存储结构化数据

| 字段名 | 类型 | 约束 | 作用 |
|--------|------|------|------|
| `prefix` | TEXT | NOT NULL | 命名空间前缀 |
| `key` | TEXT | NOT NULL | 存储键 |
| `value` | JSONB | NOT NULL | 存储值 |
| `created_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 创建时间 |
| `updated_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 更新时间 |
| `expires_at` | TIMESTAMP WITH TIME ZONE | | 过期时间 |
| `ttl_minutes` | INTEGER | | 生存时间(分钟) |

**主键**: `(prefix, key)`

---

### 11. store_vectors (LangMem向量表)
**作用**: 存储向量嵌入数据，支持语义搜索

| 字段名 | 类型 | 约束 | 作用 |
|--------|------|------|------|
| `prefix` | TEXT | NOT NULL | 命名空间前缀 |
| `key` | TEXT | NOT NULL | 存储键 |
| `field_name` | TEXT | NOT NULL | 字段名称 |
| `embedding` | VECTOR(1024) | | 向量嵌入 |
| `created_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 创建时间 |
| `updated_at` | TIMESTAMP WITH TIME ZONE | DEFAULT NOW() | 更新时间 |

**外键关系**:
- `(prefix, key)` → `store.(prefix, key)` (CASCADE删除)

**主键**: `(prefix, key, field_name)`

---

## 🔒 数据完整性规则

### 外键约束和删除规则

1. **用户级联删除**:
   - 删除用户 → 自动删除所有相关会话、消息、工具调用、智能体状态

2. **会话级联删除**:
   - 删除会话 → 自动删除该会话的所有消息、工具调用、智能体状态

3. **消息级联删除**:
   - 删除消息 → 自动删除关联的工具调用记录

4. **存储级联删除**:
   - 删除存储记录 → 自动删除对应的向量数据

### 数据约束

- **角色约束**: messages.role 只能是 'user', 'assistant', 'system'
- **状态约束**: tool_calls.status 只能是 'pending', 'success', 'error'
- **唯一性约束**: 用户名和邮箱必须唯一
- **非空约束**: 关键字段如密码、消息内容等不能为空

---

## 📈 性能优化

### 索引策略

1. **查询优化索引**:
   - 用户查询: username, email, is_active
   - 会话查询: user_id, is_active, created_at
   - 消息查询: session_id, role, created_at
   - 工具调用: session_id, message_id, tool_name, status
   - 日志查询: level, logger_name, created_at

2. **向量搜索优化**:
   - store_vectors表使用pgvector扩展
   - 支持高效的向量相似度搜索

### 分区策略建议

- **system_logs**: 按时间分区，定期清理旧日志
- **tool_calls**: 按创建时间分区，提高查询性能
- **messages**: 考虑按会话或时间分区

---

## 🛠️ 开发使用指南

### 常用查询模式

1. **获取用户所有会话**:
```sql
SELECT s.* FROM sessions s 
WHERE s.user_id = ? AND s.is_active = true 
ORDER BY s.updated_at DESC;
```

2. **获取会话消息历史**:
```sql
SELECT m.* FROM messages m 
WHERE m.session_id = ? 
ORDER BY m.created_at ASC;
```

3. **获取工具调用记录**:
```sql
SELECT tc.* FROM tool_calls tc 
WHERE tc.session_id = ? 
ORDER BY tc.created_at DESC;
```

4. **语义搜索记忆**:
```sql
SELECT s.*, sv.embedding <=> ? as distance 
FROM store s 
JOIN store_vectors sv ON s.prefix = sv.prefix AND s.key = sv.key 
WHERE s.prefix = ? 
ORDER BY distance 
LIMIT 10;
```

### 事务处理建议

- 创建会话时同时创建初始消息使用事务
- 工具调用记录的创建和更新使用事务
- 批量插入消息时使用事务优化性能

---

## 📝 维护说明

### 定期维护任务

1. **日志清理**: 定期清理过期的system_logs记录
2. **检查点清理**: 清理过期的checkpoint相关数据
3. **向量索引维护**: 定期重建向量索引以保持性能
4. **统计信息更新**: 更新表统计信息以优化查询计划

### 监控指标

- 表大小增长趋势
- 查询性能指标
- 外键约束违反情况
- 向量搜索性能

---

*最后更新: 2025-01-30*
*版本: 1.0*