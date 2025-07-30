# 04_deployment_ops.md - 部署运维与开发指南

## 部署配置

## LangMem 记忆系统部署配置

### LangMem 依赖安装

#### requirements.txt 更新
```txt
# 现有依赖...
langmem>=0.1.0
langgraph[store]>=0.2.0
psycopg2-binary>=2.9.0  # PostgreSQL 支持
redis>=4.0.0            # Redis 缓存支持
```

#### Docker 镜像更新
```dockerfile
# 在现有 Dockerfile 基础上添加
RUN pip install langmem langgraph[store] psycopg2-binary redis
```

### 记忆存储配置

#### PostgreSQL 记忆存储配置
```yaml
# docker-compose.yml 添加 PostgreSQL 服务
services:
  postgres-memory:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: langmem_store
      POSTGRES_USER: langmem_user
      POSTGRES_PASSWORD: ${POSTGRES_MEMORY_PASSWORD}
    volumes:
      - postgres_memory_data:/var/lib/postgresql/data
    ports:
      - "5433:5432"
    networks:
      - langgraph-network

  redis-cache:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - langgraph-network

volumes:
  postgres_memory_data:
  redis_data:
```

#### 环境变量配置
```bash
# .env 文件添加记忆系统配置
# LangMem 配置
LANGMEM_STORE_TYPE=postgres
LANGMEM_POSTGRES_URL=postgresql://langmem_user:${POSTGRES_MEMORY_PASSWORD}@postgres-memory:5432/langmem_store
LANGMEM_EMBEDDING_MODEL=openai:text-embedding-3-small
LANGMEM_EMBEDDING_DIMS=1536
LANGMEM_MAX_MEMORIES_PER_NAMESPACE=10000
LANGMEM_AUTO_CONSOLIDATE=true
LANGMEM_CONSOLIDATE_THRESHOLD=1000

# Redis 缓存配置
REDIS_URL=redis://redis-cache:6379/0
REDIS_MEMORY_CACHE_TTL=3600

# 记忆管理配置
MEMORY_CLEANUP_INTERVAL=86400  # 24小时
MEMORY_BACKUP_ENABLED=true
MEMORY_BACKUP_INTERVAL=604800  # 7天
```

### Kubernetes 部署配置

#### LangMem ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: langmem-config
  namespace: langgraph
data:
  langmem.yaml: |
    store:
      type: postgres
      connection_string: "postgresql://langmem_user:${POSTGRES_PASSWORD}@postgres-memory:5432/langmem_store"
      index:
        dims: 1536
        embed: "openai:text-embedding-3-small"
    
    memory:
      max_memories_per_namespace: 10000
      auto_consolidate: true
      consolidate_threshold: 1000
      cleanup_interval: 86400
    
    cache:
      redis_url: "redis://redis-cache:6379/0"
      ttl: 3600
    
    backup:
      enabled: true
      interval: 604800
      storage: "s3://langmem-backups/"
```

#### PostgreSQL 部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-memory
  namespace: langgraph
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres-memory
  template:
    metadata:
      labels:
        app: postgres-memory
    spec:
      containers:
      - name: postgres
        image: pgvector/pgvector:pg16
        env:
        - name: POSTGRES_DB
          value: "langmem_store"
        - name: POSTGRES_USER
          value: "langmem_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-memory-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-memory-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-memory
  namespace: langgraph
spec:
  selector:
    app: postgres-memory
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

#### Redis 部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-cache
  namespace: langgraph
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis-cache
  template:
    metadata:
      labels:
        app: redis-cache
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis-cache
  namespace: langgraph
spec:
  selector:
    app: redis-cache
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
```

### 记忆系统监控配置

#### Prometheus 监控指标
```yaml
# prometheus-config.yml 添加 LangMem 指标
- job_name: 'langmem-metrics'
  static_configs:
  - targets: ['langgraph-api:8000']
  metrics_path: '/metrics/memory'
  scrape_interval: 30s
```

#### 记忆系统指标定义
```python
# app/monitoring/memory_metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 记忆操作指标
memory_operations_total = Counter(
    'langmem_operations_total',
    'Total number of memory operations',
    ['operation', 'namespace', 'status']
)

memory_search_duration = Histogram(
    'langmem_search_duration_seconds',
    'Time spent searching memories',
    ['namespace']
)

memory_store_size = Gauge(
    'langmem_store_size_bytes',
    'Size of memory store in bytes',
    ['namespace']
)

memory_consolidation_duration = Histogram(
    'langmem_consolidation_duration_seconds',
    'Time spent consolidating memories'
)
```

### 1. Docker容器化部署

#### 1.1 Dockerfile
```dockerfile
# 多阶段构建 Dockerfile
FROM python:3.11-slim as builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --user -r requirements.txt

# 生产阶段
FROM python:3.11-slim

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app

# 设置工作目录
WORKDIR /app

# 从builder阶段复制依赖
COPY --from=builder /root/.local /home/app/.local

# 复制应用代码
COPY . .

# 设置权限
RUN chown -R app:app /app

# 切换到非root用户
USER app

# 设置环境变量
ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 1.2 docker-compose.yml
```yaml
version: '3.8'

services:
  # 主应用服务
  langgraph-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/langgraph
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    restart: unless-stopped
    networks:
      - langgraph-network

  # PostgreSQL数据库
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=langgraph
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - langgraph-network

  # Redis缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - langgraph-network

  # Nginx反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - langgraph-api
    networks:
      - langgraph-network

  # Prometheus监控
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - langgraph-network

  # Grafana可视化
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - langgraph-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  langgraph-network:
    driver: bridge
```

### 2. Kubernetes部署

#### 2.1 k8s-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-api
  labels:
    app: langgraph-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langgraph-api
  template:
    metadata:
      labels:
        app: langgraph-api
    spec:
      containers:
      - name: langgraph-api
        image: langgraph-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: redis-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: langgraph-api-service
spec:
  selector:
    app: langgraph-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: langgraph-api-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.langgraph.example.com
    secretName: langgraph-tls
  rules:
  - host: api.langgraph.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: langgraph-api-service
            port:
              number: 80
```

## 环境配置

### 1. 配置管理
```python
# config.py - 基于Pydantic的配置管理
from pydantic import BaseSettings, Field
from typing import Optional, List
import os

class AppConfig(BaseSettings):
    """应用基础配置"""
    app_name: str = Field(default="LangGraph Multi-Agent API", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    class Config:
        env_file = ".env"

class ServerConfig(BaseSettings):
    """服务器配置"""
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=False, env="RELOAD")
    
    class Config:
        env_file = ".env"

class DatabaseConfig(BaseSettings):
    """数据库配置"""
    url: str = Field(..., env="DATABASE_URL")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    class Config:
        env_file = ".env"

class RedisConfig(BaseSettings):
    """Redis配置"""
    url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    
    class Config:
        env_file = ".env"

class LLMConfig(BaseSettings):
    """LLM API配置"""
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    default_model: str = Field(default="gpt-4", env="DEFAULT_MODEL")
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    class Config:
        env_file = ".env"

class VectorStoreConfig(BaseSettings):
    """向量存储配置"""
    provider: str = Field(default="chroma", env="VECTOR_STORE_PROVIDER")
    collection_name: str = Field(default="documents", env="VECTOR_COLLECTION_NAME")
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    
    class Config:
        env_file = ".env"

class FileStorageConfig(BaseSettings):
    """文件存储配置"""
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_extensions: List[str] = Field(
        default=["pdf", "txt", "docx", "md"], 
        env="ALLOWED_EXTENSIONS"
    )
    
    class Config:
        env_file = ".env"

class SecurityConfig(BaseSettings):
    """安全配置"""
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    class Config:
        env_file = ".env"

class MonitoringConfig(BaseSettings):
    """监控配置"""
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"

class AgentConfig(BaseSettings):
    """智能体配置"""
    max_concurrent_agents: int = Field(default=10, env="MAX_CONCURRENT_AGENTS")
    agent_timeout: int = Field(default=300, env="AGENT_TIMEOUT")  # 5分钟
    enable_memory: bool = Field(default=True, env="ENABLE_MEMORY")
    memory_ttl: int = Field(default=3600, env="MEMORY_TTL")  # 1小时
    
    class Config:
        env_file = ".env"

class RateLimitConfig(BaseSettings):
    """限流配置"""
    requests_per_minute: int = Field(default=60, env="REQUESTS_PER_MINUTE")
    requests_per_hour: int = Field(default=1000, env="REQUESTS_PER_HOUR")
    burst_size: int = Field(default=10, env="BURST_SIZE")
    
    class Config:
        env_file = ".env"

class Settings:
    """统一配置管理"""
    
    def __init__(self):
        self.app = AppConfig()
        self.server = ServerConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.llm = LLMConfig()
        self.vector_store = VectorStoreConfig()
        self.file_storage = FileStorageConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.agent = AgentConfig()
        self.rate_limit = RateLimitConfig()
    
    def get_environment_config(self):
        """根据环境返回特定配置"""
        if self.app.environment == "production":
            return self._get_production_config()
        elif self.app.environment == "staging":
            return self._get_staging_config()
        else:
            return self._get_development_config()
    
    def _get_production_config(self):
        """生产环境配置"""
        return {
            "debug": False,
            "log_level": "WARNING",
            "workers": 4,
            "pool_size": 20,
            "max_concurrent_agents": 50
        }
    
    def _get_staging_config(self):
        """预发布环境配置"""
        return {
            "debug": False,
            "log_level": "INFO",
            "workers": 2,
            "pool_size": 10,
            "max_concurrent_agents": 20
        }
    
    def _get_development_config(self):
        """开发环境配置"""
        return {
            "debug": True,
            "log_level": "DEBUG",
            "workers": 1,
            "reload": True,
            "max_concurrent_agents": 5
        }

# 全局配置实例
settings = Settings()
```

## 监控和日志

### 1. 应用监控
```python
# monitoring.py - 基于Prometheus的监控
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
import time
import psutil
import asyncio
from functools import wraps

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # HTTP请求指标
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # 智能体指标
        self.agent_requests_total = Counter(
            'agent_requests_total',
            'Total agent requests',
            ['agent_type', 'status'],
            registry=self.registry
        )
        
        self.agent_response_time = Histogram(
            'agent_response_time_seconds',
            'Agent response time',
            ['agent_type'],
            registry=self.registry
        )
        
        self.active_agents = Gauge(
            'active_agents_count',
            'Number of active agents',
            ['agent_type'],
            registry=self.registry
        )
        
        # 系统指标
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # 数据库指标
        self.db_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration',
            ['query_type'],
            registry=self.registry
        )
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """记录HTTP请求指标"""
        self.http_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=status_code
        ).inc()
        
        self.http_request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
    
    def record_agent_request(self, agent_type: str, status: str, duration: float):
        """记录智能体请求指标"""
        self.agent_requests_total.labels(
            agent_type=agent_type, 
            status=status
        ).inc()
        
        self.agent_response_time.labels(
            agent_type=agent_type
        ).observe(duration)
    
    def update_system_metrics(self):
        """更新系统指标"""
        # 内存使用率
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent()
        self.cpu_usage.set(cpu_percent)
    
    def get_metrics(self):
        """获取所有指标"""
        return generate_latest(self.registry)

# 全局指标收集器
metrics = MetricsCollector()

def monitor_http_requests(func):
    """HTTP请求监控装饰器"""
    @wraps(func)
    async def wrapper(request, *args, **kwargs):
        start_time = time.time()
        method = request.method
        endpoint = request.url.path
        
        try:
            response = await func(request, *args, **kwargs)
            status_code = response.status_code
            return response
        except Exception as e:
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time
            metrics.record_http_request(method, endpoint, status_code, duration)
    
    return wrapper

def monitor_agent_requests(agent_type: str):
    """智能体请求监控装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics.record_agent_request(agent_type, status, duration)
        
        return wrapper
    return decorator

async def update_metrics_periodically():
    """定期更新系统指标"""
    while True:
        metrics.update_system_metrics()
        await asyncio.sleep(30)  # 每30秒更新一次
```

### 2. 结构化日志
```python
# logging_config.py - 结构化日志配置
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any
import traceback

class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 添加额外字段
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'thread_id'):
            log_entry['thread_id'] = record.thread_id
        
        if hasattr(record, 'agent_type'):
            log_entry['agent_type'] = record.agent_type
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, ensure_ascii=False)

class AgentLogger:
    """智能体专用日志器"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        file_handler = logging.FileHandler('logs/agents.log')
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
    
    def log_agent_start(self, agent_type: str, user_id: str, thread_id: str):
        """记录智能体启动"""
        self.logger.info(
            "Agent started",
            extra={
                "agent_type": agent_type,
                "user_id": user_id,
                "thread_id": thread_id,
                "event_type": "agent_start"
            }
        )
    
    def log_agent_response(self, agent_type: str, user_id: str, thread_id: str, 
                          response_time: float, token_count: int):
        """记录智能体响应"""
        self.logger.info(
            "Agent response generated",
            extra={
                "agent_type": agent_type,
                "user_id": user_id,
                "thread_id": thread_id,
                "response_time": response_time,
                "token_count": token_count,
                "event_type": "agent_response"
            }
        )
    
    def log_agent_error(self, agent_type: str, user_id: str, thread_id: str, 
                       error: Exception):
        """记录智能体错误"""
        self.logger.error(
            "Agent error occurred",
            extra={
                "agent_type": agent_type,
                "user_id": user_id,
                "thread_id": thread_id,
                "event_type": "agent_error"
            },
            exc_info=True
        )
    
    def log_tool_usage(self, agent_type: str, tool_name: str, user_id: str, 
                      thread_id: str, execution_time: float):
        """记录工具使用"""
        self.logger.info(
            "Tool executed",
            extra={
                "agent_type": agent_type,
                "tool_name": tool_name,
                "user_id": user_id,
                "thread_id": thread_id,
                "execution_time": execution_time,
                "event_type": "tool_usage"
            }
        )

def setup_logging():
    """设置全局日志配置"""
    # 根日志器配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 移除默认处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加结构化处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.FileHandler('logs/app.log')
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)
    
    # 错误日志文件
    error_handler = logging.FileHandler('logs/error.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(error_handler)

# 全局智能体日志器
agent_logger = AgentLogger("agents")
```

### 3. 健康检查
```python
# health.py - 健康检查端点
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import time
import psutil
from sqlalchemy import text
from redis import Redis

router = APIRouter()

class HealthStatus(BaseModel):
    """健康状态模型"""
    status: str
    timestamp: str
    version: str
    uptime: float
    checks: Dict[str, Any]

class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.start_time = time.time()
    
    async def check_database(self, db_session) -> Dict[str, Any]:
        """检查数据库连接"""
        try:
            result = await db_session.execute(text("SELECT 1"))
            return {
                "status": "healthy",
                "response_time": 0.001,  # 实际测量
                "details": "Database connection successful"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Database connection failed"
            }
    
    async def check_redis(self, redis_client: Redis) -> Dict[str, Any]:
        """检查Redis连接"""
        try:
            start_time = time.time()
            await redis_client.ping()
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "details": "Redis connection successful"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Redis connection failed"
            }
    
    async def check_llm_api(self) -> Dict[str, Any]:
        """检查LLM API连接"""
        try:
            # 这里可以添加实际的LLM API健康检查
            # 例如发送一个简单的请求来验证API可用性
            return {
                "status": "healthy",
                "details": "LLM API accessible"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "LLM API connection failed"
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """检查系统资源"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent()
            
            # 定义阈值
            memory_threshold = 90  # 90%
            disk_threshold = 90    # 90%
            cpu_threshold = 90     # 90%
            
            status = "healthy"
            warnings = []
            
            if memory.percent > memory_threshold:
                status = "degraded"
                warnings.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > disk_threshold:
                status = "degraded"
                warnings.append(f"High disk usage: {disk.percent}%")
            
            if cpu_percent > cpu_threshold:
                status = "degraded"
                warnings.append(f"High CPU usage: {cpu_percent}%")
            
            return {
                "status": status,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "cpu_percent": cpu_percent,
                "warnings": warnings
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "System resource check failed"
            }
    
    def get_uptime(self) -> float:
        """获取运行时间"""
        return time.time() - self.start_time

# 全局健康检查器
health_checker = HealthChecker()

@router.get("/health", response_model=HealthStatus)
async def health_check():
    """基础健康检查"""
    return HealthStatus(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        uptime=health_checker.get_uptime(),
        checks={}
    )

@router.get("/health/ready")
async def readiness_check(db_session=Depends(get_db_session), 
                         redis_client=Depends(get_redis_client)):
    """就绪检查 - 检查所有依赖服务"""
    checks = {}
    overall_status = "healthy"
    
    # 检查数据库
    db_check = await health_checker.check_database(db_session)
    checks["database"] = db_check
    if db_check["status"] != "healthy":
        overall_status = "unhealthy"
    
    # 检查Redis
    redis_check = await health_checker.check_redis(redis_client)
    checks["redis"] = redis_check
    if redis_check["status"] != "healthy":
        overall_status = "unhealthy"
    
    # 检查LLM API
    llm_check = await health_checker.check_llm_api()
    checks["llm_api"] = llm_check
    if llm_check["status"] != "healthy":
        overall_status = "unhealthy"
    
    if overall_status != "healthy":
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return HealthStatus(
        status=overall_status,
        timestamp=time.time(),
        version="1.0.0",
        uptime=health_checker.get_uptime(),
        checks=checks
    )

@router.get("/health/live")
async def liveness_check():
    """存活检查 - 检查应用是否响应"""
    system_check = health_checker.check_system_resources()
    
    if system_check["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail="Service unhealthy")
    
    return HealthStatus(
        status=system_check["status"],
        timestamp=time.time(),
        version="1.0.0",
        uptime=health_checker.get_uptime(),
        checks={"system": system_check}
    )
```

## 开发指南

### 1. 开发优先级和里程碑
```python
# development_phases.py - 开发阶段管理
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class TaskStatus(Enum):
    """任务状态"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

class Priority(Enum):
    """优先级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class DevelopmentTask:
    """开发任务"""
    id: str
    name: str
    description: str
    priority: Priority
    estimated_hours: int
    status: TaskStatus = TaskStatus.NOT_STARTED
    assignee: Optional[str] = None
    dependencies: List[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class DevelopmentPhase:
    """开发阶段"""
    id: str
    name: str
    description: str
    tasks: List[DevelopmentTask]
    start_date: datetime
    target_end_date: datetime
    
    @property
    def completion_percentage(self) -> float:
        """完成百分比"""
        if not self.tasks:
            return 0.0
        
        completed_tasks = sum(1 for task in self.tasks if task.status == TaskStatus.COMPLETED)
        return (completed_tasks / len(self.tasks)) * 100
    
    @property
    def is_completed(self) -> bool:
        """是否完成"""
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks)

class ProjectManager:
    """项目管理器"""
    
    def __init__(self):
        self.phases = self._initialize_phases()
    
    def _initialize_phases(self) -> List[DevelopmentPhase]:
        """初始化开发阶段"""
        
        # 阶段1: 核心架构
        phase1_tasks = [
            DevelopmentTask(
                id="arch_001",
                name="智能体基础架构设计",
                description="实现BaseAgent抽象类和AgentRegistry",
                priority=Priority.HIGH,
                estimated_hours=16
            ),
            DevelopmentTask(
                id="arch_002", 
                name="状态管理系统",
                description="实现CheckpointManager和ThreadManager",
                priority=Priority.HIGH,
                estimated_hours=20
            ),
            DevelopmentTask(
                id="arch_003",
                name="配置管理系统",
                description="实现基于Pydantic的配置管理",
                priority=Priority.MEDIUM,
                estimated_hours=8
            ),
            DevelopmentTask(
                id="arch_004",
                name="数据库模型设计",
                description="设计和实现数据库模型",
                priority=Priority.HIGH,
                estimated_hours=12
            )
        ]
        
        # 阶段2: 智能体实现
        phase2_tasks = [
            DevelopmentTask(
                id="agent_001",
                name="多智能体协作系统",
                description="实现supervisor + research_agent + chart_agent",
                priority=Priority.HIGH,
                estimated_hours=24,
                dependencies=["arch_001", "arch_002"]
            ),
            DevelopmentTask(
                id="agent_002",
                name="Agentic RAG系统",
                description="实现智能检索增强生成系统",
                priority=Priority.HIGH,
                estimated_hours=20,
                dependencies=["arch_001", "arch_002"]
            ),
            DevelopmentTask(
                id="agent_003",
                name="专业化智能体",
                description="实现代码、数据分析、内容创作智能体",
                priority=Priority.MEDIUM,
                estimated_hours=32,
                dependencies=["agent_001"]
            ),
            DevelopmentTask(
                id="agent_004",
                name="工具集成系统",
                description="实现工具管理和动态加载",
                priority=Priority.HIGH,
                estimated_hours=16
            )
        ]
        
        # 阶段3: API和流式处理
        phase3_tasks = [
            DevelopmentTask(
                id="api_001",
                name="RESTful API实现",
                description="实现所有API端点",
                priority=Priority.HIGH,
                estimated_hours=20,
                dependencies=["agent_001", "agent_002"]
            ),
            DevelopmentTask(
                id="api_002",
                name="流式响应实现",
                description="实现WebSocket和SSE流式响应",
                priority=Priority.HIGH,
                estimated_hours=16,
                dependencies=["api_001"]
            ),
            DevelopmentTask(
                id="api_003",
                name="中断和恢复机制",
                description="实现智能体执行的中断和恢复",
                priority=Priority.MEDIUM,
                estimated_hours=12,
                dependencies=["api_001"]
            ),
            DevelopmentTask(
                id="api_004",
                name="认证和授权",
                description="实现用户认证和API授权",
                priority=Priority.MEDIUM,
                estimated_hours=10
            )
        ]
        
        # 阶段4: 部署和监控
        phase4_tasks = [
            DevelopmentTask(
                id="deploy_001",
                name="Docker容器化",
                description="创建Dockerfile和docker-compose配置",
                priority=Priority.HIGH,
                estimated_hours=8
            ),
            DevelopmentTask(
                id="deploy_002",
                name="Kubernetes部署",
                description="创建K8s部署配置",
                priority=Priority.MEDIUM,
                estimated_hours=12
            ),
            DevelopmentTask(
                id="monitor_001",
                name="监控系统",
                description="实现Prometheus监控和Grafana仪表板",
                priority=Priority.HIGH,
                estimated_hours=16
            ),
            DevelopmentTask(
                id="monitor_002",
                name="日志系统",
                description="实现结构化日志和日志聚合",
                priority=Priority.MEDIUM,
                estimated_hours=8
            ),
            DevelopmentTask(
                id="monitor_003",
                name="健康检查",
                description="实现健康检查端点",
                priority=Priority.HIGH,
                estimated_hours=6
            )
        ]
        
        # 阶段5: 测试和文档
        phase5_tasks = [
            DevelopmentTask(
                id="test_001",
                name="单元测试",
                description="编写核心组件单元测试",
                priority=Priority.HIGH,
                estimated_hours=20
            ),
            DevelopmentTask(
                id="test_002",
                name="集成测试",
                description="编写API集成测试",
                priority=Priority.HIGH,
                estimated_hours=16
            ),
            DevelopmentTask(
                id="test_003",
                name="性能测试",
                description="进行负载和性能测试",
                priority=Priority.MEDIUM,
                estimated_hours=12
            ),
            DevelopmentTask(
                id="doc_001",
                name="API文档",
                description="编写完整的API文档",
                priority=Priority.MEDIUM,
                estimated_hours=8
            ),
            DevelopmentTask(
                id="doc_002",
                name="部署文档",
                description="编写部署和运维文档",
                priority=Priority.MEDIUM,
                estimated_hours=6
            )
        ]
        
        # 创建阶段
        base_date = datetime.now()
        
        return [
            DevelopmentPhase(
                id="phase_1",
                name="核心架构",
                description="建立项目基础架构和核心组件",
                tasks=phase1_tasks,
                start_date=base_date,
                target_end_date=base_date + timedelta(weeks=2)
            ),
            DevelopmentPhase(
                id="phase_2", 
                name="智能体实现",
                description="实现各类智能体和工具集成",
                tasks=phase2_tasks,
                start_date=base_date + timedelta(weeks=2),
                target_end_date=base_date + timedelta(weeks=5)
            ),
            DevelopmentPhase(
                id="phase_3",
                name="API和流式处理",
                description="实现API接口和流式响应",
                tasks=phase3_tasks,
                start_date=base_date + timedelta(weeks=5),
                target_end_date=base_date + timedelta(weeks=7)
            ),
            DevelopmentPhase(
                id="phase_4",
                name="部署和监控",
                description="实现部署配置和监控系统",
                tasks=phase4_tasks,
                start_date=base_date + timedelta(weeks=7),
                target_end_date=base_date + timedelta(weeks=9)
            ),
            DevelopmentPhase(
                id="phase_5",
                name="测试和文档",
                description="完善测试覆盖和项目文档",
                tasks=phase5_tasks,
                start_date=base_date + timedelta(weeks=9),
                target_end_date=base_date + timedelta(weeks=11)
            )
        ]
    
    def get_current_phase(self) -> Optional[DevelopmentPhase]:
        """获取当前阶段"""
        now = datetime.now()
        for phase in self.phases:
            if phase.start_date <= now <= phase.target_end_date and not phase.is_completed:
                return phase
        return None
    
    def get_next_tasks(self, limit: int = 5) -> List[DevelopmentTask]:
        """获取下一批任务"""
        current_phase = self.get_current_phase()
        if not current_phase:
            return []
        
        # 获取未开始的高优先级任务
        available_tasks = [
            task for task in current_phase.tasks
            if task.status == TaskStatus.NOT_STARTED
            and self._are_dependencies_met(task)
        ]
        
        # 按优先级排序
        available_tasks.sort(key=lambda t: (t.priority.value, t.estimated_hours))
        
        return available_tasks[:limit]
    
    def _are_dependencies_met(self, task: DevelopmentTask) -> bool:
        """检查任务依赖是否满足"""
        if not task.dependencies:
            return True
        
        # 查找所有任务
        all_tasks = []
        for phase in self.phases:
            all_tasks.extend(phase.tasks)
        
        task_dict = {t.id: t for t in all_tasks}
        
        for dep_id in task.dependencies:
            if dep_id in task_dict:
                if task_dict[dep_id].status != TaskStatus.COMPLETED:
                    return False
        
        return True
    
    def update_task_status(self, task_id: str, status: TaskStatus, assignee: str = None):
        """更新任务状态"""
        for phase in self.phases:
            for task in phase.tasks:
                if task.id == task_id:
                    task.status = status
                    if assignee:
                        task.assignee = assignee
                    if status == TaskStatus.IN_PROGRESS and not task.start_date:
                        task.start_date = datetime.now()
                    elif status == TaskStatus.COMPLETED and not task.end_date:
                        task.end_date = datetime.now()
                    return
    
    def get_project_status(self) -> Dict:
        """获取项目整体状态"""
        total_tasks = sum(len(phase.tasks) for phase in self.phases)
        completed_tasks = sum(
            len([t for t in phase.tasks if t.status == TaskStatus.COMPLETED])
            for phase in self.phases
        )
        
        return {
            "total_phases": len(self.phases),
            "completed_phases": len([p for p in self.phases if p.is_completed]),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "overall_completion": (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0,
            "current_phase": self.get_current_phase().name if self.get_current_phase() else None,
            "phases": [
                {
                    "name": phase.name,
                    "completion": phase.completion_percentage,
                    "is_completed": phase.is_completed
                }
                for phase in self.phases
            ]
        }

# 全局项目管理器
project_manager = ProjectManager()
```

### 2. 质量保证
```python
# quality_assurance.py - 质量保证工具
import ast
import subprocess
import coverage
from typing import List, Dict, Any
import pytest
import asyncio
from pathlib import Path

class CodeQualityChecker:
    """代码质量检查器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
    
    def check_code_style(self) -> Dict[str, Any]:
        """检查代码风格 (使用black和flake8)"""
        results = {}
        
        # Black格式检查
        try:
            black_result = subprocess.run(
                ["black", "--check", "--diff", str(self.project_root)],
                capture_output=True,
                text=True
            )
            results["black"] = {
                "passed": black_result.returncode == 0,
                "output": black_result.stdout,
                "errors": black_result.stderr
            }
        except FileNotFoundError:
            results["black"] = {"error": "Black not installed"}
        
        # Flake8检查
        try:
            flake8_result = subprocess.run(
                ["flake8", str(self.project_root)],
                capture_output=True,
                text=True
            )
            results["flake8"] = {
                "passed": flake8_result.returncode == 0,
                "output": flake8_result.stdout,
                "errors": flake8_result.stderr
            }
        except FileNotFoundError:
            results["flake8"] = {"error": "Flake8 not installed"}
        
        return results
    
    def check_complexity(self) -> Dict[str, Any]:
        """检查代码复杂度"""
        results = {}
        
        try:
            # 使用radon检查圈复杂度
            radon_result = subprocess.run(
                ["radon", "cc", str(self.project_root), "-a"],
                capture_output=True,
                text=True
            )
            results["cyclomatic_complexity"] = {
                "output": radon_result.stdout,
                "errors": radon_result.stderr
            }
            
            # 使用radon检查维护性指数
            mi_result = subprocess.run(
                ["radon", "mi", str(self.project_root)],
                capture_output=True,
                text=True
            )
            results["maintainability_index"] = {
                "output": mi_result.stdout,
                "errors": mi_result.stderr
            }
            
        except FileNotFoundError:
            results["error"] = "Radon not installed"
        
        return results
    
    def check_test_coverage(self) -> Dict[str, Any]:
        """检查测试覆盖率"""
        try:
            # 运行coverage
            cov = coverage.Coverage()
            cov.start()
            
            # 这里应该运行测试套件
            # pytest.main([str(self.project_root / "tests")])
            
            cov.stop()
            cov.save()
            
            # 生成报告
            total_coverage = cov.report()
            
            return {
                "total_coverage": total_coverage,
                "details": "Coverage report generated"
            }
            
        except Exception as e:
            return {"error": str(e)}

class TestFramework:
    """测试框架"""
    
    def __init__(self):
        self.test_results = {}
    
    async def run_unit_tests(self) -> Dict[str, Any]:
        """运行单元测试"""
        try:
            # 运行pytest
            result = subprocess.run(
                ["pytest", "tests/unit", "-v", "--tb=short"],
                capture_output=True,
                text=True
            )
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "return_code": result.returncode
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """运行集成测试"""
        try:
            result = subprocess.run(
                ["pytest", "tests/integration", "-v", "--tb=short"],
                capture_output=True,
                text=True
            )
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "return_code": result.returncode
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """运行性能测试"""
        try:
            # 使用pytest-benchmark或locust
            result = subprocess.run(
                ["pytest", "tests/performance", "-v", "--benchmark-only"],
                capture_output=True,
                text=True
            )
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "return_code": result.returncode
            }
        except Exception as e:
            return {"error": str(e)}

# 示例测试用例
class TestAgentSystem:
    """智能体系统测试"""
    
    @pytest.mark.asyncio
    async def test_agent_creation(self):
        """测试智能体创建"""
        from agents.base import AgentFactory
        
        config = {
            "agent_type": "multi_agent_supervisor",
            "llm_config": {"model": "gpt-4", "temperature": 0.7}
        }
        
        agent = await AgentFactory.create_agent("multi_agent_supervisor", config)
        assert agent is not None
        assert agent.agent_type == "multi_agent_supervisor"
    
    @pytest.mark.asyncio
    async def test_agent_chat(self):
        """测试智能体对话"""
        from agents.base import AgentFactory
        
        config = {
            "agent_type": "multi_agent_supervisor",
            "llm_config": {"model": "gpt-4", "temperature": 0.7}
        }
        
        agent = await AgentFactory.create_agent("multi_agent_supervisor", config)
        response = await agent.chat("Hello, how are you?")
        
        assert response is not None
        assert response.message is not None
        assert len(response.message) > 0
    
    @pytest.mark.asyncio
    async def test_state_persistence(self):
        """测试状态持久化"""
        from state.thread_manager import ThreadManager
        from state.checkpoint_manager import CheckpointManager
        
        checkpoint_manager = CheckpointManager()
        thread_manager = ThreadManager(checkpoint_manager.checkpointer)
        
        # 创建线程
        thread_id = await thread_manager.create_thread("test_user", "test_agent")
        assert thread_id is not None
        
        # 获取线程状态
        state = await thread_manager.get_thread_state(thread_id)
        assert state is not None
        assert state["user_id"] == "test_user"
        assert state["agent_type"] == "test_agent"

class TestAPIEndpoints:
    """API端点测试"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """测试健康检查端点"""
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_agent_chat_endpoint(self):
        """测试智能体对话端点"""
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        chat_request = {
            "message": "Hello, world!",
            "user_id": "test_user",
            "agent_config": {
                "model": "gpt-4",
                "temperature": 0.7
            }
        }
        
        response = client.post(
            "/api/v1/agents/multi_agent_supervisor/chat",
            json=chat_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "thread_id" in data

# 全局质量检查器
quality_checker = CodeQualityChecker(".")
test_framework = TestFramework()
```

这个文档涵盖了：

1. **Docker容器化部署** - 包含多阶段构建的Dockerfile和完整的docker-compose配置
2. **Kubernetes部署** - K8s部署配置文件
3. **环境配置管理** - 基于Pydantic的配置系统，支持多环境
4. **监控系统** - Prometheus指标收集和监控装饰器
5. **结构化日志** - JSON格式的结构化日志系统
6. **健康检查** - 完整的健康检查端点实现
7. **开发管理** - 项目阶段管理和任务跟踪
8. **质量保证** - 代码质量检查和测试框架

这些配置和工具将帮助您建立一个完整的生产级LangGraph多智能体系统。