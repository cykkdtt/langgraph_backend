"""
MCP连接管理器

负责管理MCP服务器连接的生命周期，包括：
- 连接建立和维护
- 健康检查和监控
- 自动重连机制
- 连接池管理
- 性能指标收集
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import json
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field


class ConnectionStatus(Enum):
    """连接状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class ConnectionMetrics:
    """连接指标"""
    connection_count: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency: float = 0.0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    reconnect_count: int = 0
    uptime_start: Optional[float] = None
    
    @property
    def average_latency(self) -> float:
        """平均延迟"""
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency / self.successful_calls
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        total_calls = self.successful_calls + self.failed_calls
        if total_calls == 0:
            return 0.0
        return self.successful_calls / total_calls
    
    @property
    def uptime(self) -> float:
        """运行时间（秒）"""
        if self.uptime_start is None:
            return 0.0
        return time.time() - self.uptime_start


class MCPConnectionConfig(BaseModel):
    """MCP连接配置"""
    name: str = Field(description="连接名称")
    command: Optional[str] = Field(default=None, description="启动命令")
    args: List[str] = Field(default_factory=list, description="命令参数")
    url: Optional[str] = Field(default=None, description="服务器URL")
    transport: str = Field(default="stdio", description="传输协议")
    env: Dict[str, str] = Field(default_factory=dict, description="环境变量")
    enabled: bool = Field(default=True, description="是否启用")
    
    # 连接管理配置
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟（秒）")
    health_check_interval: float = Field(default=30.0, description="健康检查间隔（秒）")
    connection_timeout: float = Field(default=10.0, description="连接超时（秒）")
    call_timeout: float = Field(default=30.0, description="调用超时（秒）")
    
    # 性能配置
    max_concurrent_calls: int = Field(default=10, description="最大并发调用数")
    rate_limit_per_second: Optional[float] = Field(default=None, description="每秒速率限制")


class MCPConnection:
    """MCP连接实例"""
    
    def __init__(self, config: MCPConnectionConfig):
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.metrics = ConnectionMetrics()
        self.client: Optional[MultiServerMCPClient] = None
        self.logger = logging.getLogger(f"mcp.connection.{config.name}")
        
        # 控制变量
        self._stop_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        self._connection_lock = asyncio.Lock()
        self._call_semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        self._rate_limiter: Optional[asyncio.Semaphore] = None
        
        # 回调函数
        self._status_callbacks: Set[Callable[[str, ConnectionStatus], None]] = set()
        
        # 速率限制
        if config.rate_limit_per_second:
            self._rate_limiter = asyncio.Semaphore(int(config.rate_limit_per_second))
            self._rate_reset_task: Optional[asyncio.Task] = None
    
    def add_status_callback(self, callback: Callable[[str, ConnectionStatus], None]):
        """添加状态变化回调"""
        self._status_callbacks.add(callback)
    
    def remove_status_callback(self, callback: Callable[[str, ConnectionStatus], None]):
        """移除状态变化回调"""
        self._status_callbacks.discard(callback)
    
    def _notify_status_change(self, old_status: ConnectionStatus, new_status: ConnectionStatus):
        """通知状态变化"""
        for callback in self._status_callbacks:
            try:
                callback(self.config.name, new_status)
            except Exception as e:
                self.logger.error(f"状态回调执行失败: {e}")
    
    def _set_status(self, new_status: ConnectionStatus):
        """设置连接状态"""
        old_status = self.status
        self.status = new_status
        
        if old_status != new_status:
            self.logger.info(f"连接状态变化: {old_status.value} -> {new_status.value}")
            self._notify_status_change(old_status, new_status)
            
            # 更新指标
            if new_status == ConnectionStatus.CONNECTED:
                self.metrics.uptime_start = time.time()
            elif new_status in [ConnectionStatus.DISCONNECTED, ConnectionStatus.FAILED]:
                self.metrics.uptime_start = None
    
    async def connect(self) -> bool:
        """建立连接"""
        if not self.config.enabled:
            self._set_status(ConnectionStatus.DISABLED)
            return False
        
        async with self._connection_lock:
            if self.status == ConnectionStatus.CONNECTED:
                return True
            
            self._set_status(ConnectionStatus.CONNECTING)
            
            try:
                # 构建连接配置
                connection_config = {
                    "transport": self.config.transport
                }
                
                if self.config.command:
                    connection_config["command"] = self.config.command
                    connection_config["args"] = self.config.args
                    if self.config.env:
                        connection_config["env"] = self.config.env
                elif self.config.url:
                    connection_config["url"] = self.config.url
                else:
                    raise ValueError("缺少command或url配置")
                
                # 创建客户端
                connections = {self.config.name: connection_config}
                self.client = MultiServerMCPClient(connections)
                
                # 测试连接
                await asyncio.wait_for(
                    self._test_connection(),
                    timeout=self.config.connection_timeout
                )
                
                self._set_status(ConnectionStatus.CONNECTED)
                self.metrics.connection_count += 1
                
                # 启动健康检查
                await self._start_health_check()
                
                # 启动速率限制重置任务
                if self._rate_limiter:
                    await self._start_rate_reset_task()
                
                self.logger.info("MCP连接建立成功")
                return True
                
            except Exception as e:
                self.logger.error(f"MCP连接建立失败: {e}")
                self._set_status(ConnectionStatus.FAILED)
                return False
    
    async def _test_connection(self):
        """测试连接"""
        if not self.client:
            raise RuntimeError("客户端未初始化")
        
        # 尝试获取工具列表来测试连接
        await self.client.get_tools(self.config.name)
    
    async def disconnect(self):
        """断开连接"""
        async with self._connection_lock:
            self._stop_event.set()
            
            # 停止健康检查
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None
            
            # 停止速率限制重置任务
            if hasattr(self, '_rate_reset_task') and self._rate_reset_task:
                self._rate_reset_task.cancel()
                try:
                    await self._rate_reset_task
                except asyncio.CancelledError:
                    pass
                self._rate_reset_task = None
            
            # 清理客户端
            if self.client:
                self.client = None
            
            self._set_status(ConnectionStatus.DISCONNECTED)
            self.logger.info("MCP连接已断开")
    
    async def _start_health_check(self):
        """启动健康检查任务"""
        if self._health_check_task:
            return
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if self.status == ConnectionStatus.CONNECTED:
                    await self._perform_health_check()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"健康检查失败: {e}")
                await self._handle_connection_failure()
    
    async def _perform_health_check(self):
        """执行健康检查"""
        try:
            start_time = time.time()
            await asyncio.wait_for(
                self._test_connection(),
                timeout=self.config.call_timeout
            )
            
            # 更新指标
            latency = time.time() - start_time
            self.metrics.successful_calls += 1
            self.metrics.total_latency += latency
            self.metrics.last_success_time = time.time()
            
        except Exception as e:
            self.logger.warning(f"健康检查失败: {e}")
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = time.time()
            await self._handle_connection_failure()
    
    async def _handle_connection_failure(self):
        """处理连接失败"""
        if self.status == ConnectionStatus.CONNECTED:
            self._set_status(ConnectionStatus.RECONNECTING)
            await self._attempt_reconnect()
    
    async def _attempt_reconnect(self):
        """尝试重连"""
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"尝试重连 (第{attempt + 1}次)")
                
                # 清理旧连接
                if self.client:
                    self.client = None
                
                # 等待重试延迟
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                
                # 尝试重新连接
                if await self.connect():
                    self.metrics.reconnect_count += 1
                    self.logger.info("重连成功")
                    return
                
            except Exception as e:
                self.logger.error(f"重连失败 (第{attempt + 1}次): {e}")
        
        self.logger.error("重连失败，达到最大重试次数")
        self._set_status(ConnectionStatus.FAILED)
    
    async def _start_rate_reset_task(self):
        """启动速率限制重置任务"""
        if not self._rate_limiter:
            return
        
        self._rate_reset_task = asyncio.create_task(self._rate_reset_loop())
    
    async def _rate_reset_loop(self):
        """速率限制重置循环"""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(1.0)  # 每秒重置
                
                # 重置速率限制信号量
                if self._rate_limiter and self.config.rate_limit_per_second:
                    current_value = self._rate_limiter._value
                    max_value = int(self.config.rate_limit_per_second)
                    
                    # 释放信号量到最大值
                    for _ in range(max_value - current_value):
                        self._rate_limiter.release()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"速率限制重置失败: {e}")
    
    @asynccontextmanager
    async def call_context(self):
        """调用上下文管理器"""
        if self.status != ConnectionStatus.CONNECTED:
            raise RuntimeError(f"连接未就绪: {self.status.value}")
        
        # 获取并发控制信号量
        async with self._call_semaphore:
            # 获取速率限制信号量
            if self._rate_limiter:
                await self._rate_limiter.acquire()
            
            try:
                yield self.client
            except Exception as e:
                self.metrics.failed_calls += 1
                self.metrics.last_failure_time = time.time()
                raise
            else:
                self.metrics.successful_calls += 1
                self.metrics.last_success_time = time.time()
    
    def get_status_info(self) -> Dict[str, Any]:
        """获取连接状态信息"""
        return {
            "name": self.config.name,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "metrics": {
                "connection_count": self.metrics.connection_count,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "average_latency": self.metrics.average_latency,
                "success_rate": self.metrics.success_rate,
                "uptime": self.metrics.uptime,
                "reconnect_count": self.metrics.reconnect_count,
                "last_success_time": self.metrics.last_success_time,
                "last_failure_time": self.metrics.last_failure_time
            },
            "config": {
                "transport": self.config.transport,
                "max_retries": self.config.max_retries,
                "health_check_interval": self.config.health_check_interval,
                "max_concurrent_calls": self.config.max_concurrent_calls,
                "rate_limit_per_second": self.config.rate_limit_per_second
            }
        }


class MCPConnectionManager:
    """MCP连接管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "servers_config.json"
        self.logger = logging.getLogger("mcp.connection_manager")
        
        self.connections: Dict[str, MCPConnection] = {}
        self._global_metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "total_calls": 0,
            "total_errors": 0,
            "start_time": time.time()
        }
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载连接配置"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self.logger.warning(f"配置文件不存在: {self.config_path}")
                return
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            mcp_servers = config_data.get("mcpServers", {})
            
            for server_name, server_config in mcp_servers.items():
                try:
                    config = MCPConnectionConfig(
                        name=server_name,
                        **server_config
                    )
                    
                    connection = MCPConnection(config)
                    connection.add_status_callback(self._on_connection_status_change)
                    
                    self.connections[server_name] = connection
                    self.logger.info(f"加载连接配置: {server_name}")
                    
                except Exception as e:
                    self.logger.error(f"解析连接配置失败 {server_name}: {e}")
            
            self.logger.info(f"成功加载 {len(self.connections)} 个连接配置")
            
        except Exception as e:
            self.logger.error(f"加载连接配置失败: {e}")
    
    def _on_connection_status_change(self, connection_name: str, status: ConnectionStatus):
        """连接状态变化回调"""
        self.logger.info(f"连接 {connection_name} 状态变化: {status.value}")
        
        # 更新全局指标
        active_count = sum(
            1 for conn in self.connections.values()
            if conn.status == ConnectionStatus.CONNECTED
        )
        self._global_metrics["active_connections"] = active_count
    
    async def connect_all(self) -> Dict[str, bool]:
        """连接所有服务器"""
        results = {}
        
        tasks = []
        for name, connection in self.connections.items():
            task = asyncio.create_task(connection.connect())
            tasks.append((name, task))
        
        for name, task in tasks:
            try:
                success = await task
                results[name] = success
                if success:
                    self._global_metrics["total_connections"] += 1
            except Exception as e:
                self.logger.error(f"连接 {name} 失败: {e}")
                results[name] = False
        
        return results
    
    async def disconnect_all(self):
        """断开所有连接"""
        tasks = []
        for connection in self.connections.values():
            task = asyncio.create_task(connection.disconnect())
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        self.logger.info("所有连接已断开")
    
    def get_connection(self, name: str) -> Optional[MCPConnection]:
        """获取指定连接"""
        return self.connections.get(name)
    
    def get_active_connections(self) -> List[MCPConnection]:
        """获取活跃连接列表"""
        return [
            conn for conn in self.connections.values()
            if conn.status == ConnectionStatus.CONNECTED
        ]
    
    def get_status_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        status_counts = {}
        for connection in self.connections.values():
            status = connection.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_calls = sum(conn.metrics.successful_calls + conn.metrics.failed_calls 
                         for conn in self.connections.values())
        total_errors = sum(conn.metrics.failed_calls 
                          for conn in self.connections.values())
        
        return {
            "total_connections": len(self.connections),
            "status_counts": status_counts,
            "global_metrics": {
                **self._global_metrics,
                "total_calls": total_calls,
                "total_errors": total_errors,
                "uptime": time.time() - self._global_metrics["start_time"]
            },
            "connections": {
                name: conn.get_status_info()
                for name, conn in self.connections.items()
            }
        }
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """对所有连接执行健康检查"""
        results = {}
        
        for name, connection in self.connections.items():
            try:
                if connection.status == ConnectionStatus.CONNECTED:
                    start_time = time.time()
                    await connection._test_connection()
                    latency = time.time() - start_time
                    
                    results[name] = {
                        "status": "healthy",
                        "latency": latency,
                        "error": None
                    }
                else:
                    results[name] = {
                        "status": "unhealthy",
                        "latency": None,
                        "error": f"连接状态: {connection.status.value}"
                    }
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "latency": None,
                    "error": str(e)
                }
        
        return results
    
    async def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            # 断开所有现有连接
            await self.disconnect_all()
            
            # 清空连接
            self.connections.clear()
            
            # 重新加载配置
            self._load_config()
            
            # 重新连接
            await self.connect_all()
            
            self.logger.info("配置重新加载完成")
            return True
            
        except Exception as e:
            self.logger.error(f"重新加载配置失败: {e}")
            return False


# 全局连接管理器实例
_connection_manager: Optional[MCPConnectionManager] = None


def get_connection_manager() -> MCPConnectionManager:
    """获取全局连接管理器实例"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = MCPConnectionManager()
    return _connection_manager


async def initialize_connection_manager() -> bool:
    """初始化全局连接管理器"""
    manager = get_connection_manager()
    results = await manager.connect_all()
    return any(results.values())