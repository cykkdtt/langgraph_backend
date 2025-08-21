"""
多智能体LangGraph项目 - 日志系统

本模块提供统一的日志管理功能，包括：
- 结构化日志配置
- 日志格式化
- 日志轮转管理
- 不同模块的日志配置
"""

import os
import sys
import logging
import logging.handlers
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from config.settings import get_settings


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def __init__(self, json_format: bool = False):
        self.json_format = json_format
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 基础日志信息
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        if hasattr(record, 'agent_id'):
            log_data["agent_id"] = record.agent_id
        if hasattr(record, 'session_id'):
            log_data["session_id"] = record.session_id
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        
        if self.json_format:
            return json.dumps(log_data, ensure_ascii=False)
        else:
            # 格式化为可读文本
            base_msg = f"{log_data['timestamp']} - {log_data['logger']} - {log_data['level']} - {log_data['message']}"
            if log_data.get('exception'):
                base_msg += f"\n{log_data['exception']}"
            return base_msg


class LoggerManager:
    """日志管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.is_initialized = False
        self._loggers: Dict[str, logging.Logger] = {}
    
    def initialize(self) -> None:
        """初始化日志系统"""
        if self.is_initialized:
            return
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.settings.monitoring.log_level.upper()))
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 创建格式化器
        formatter = StructuredFormatter(
            json_format=getattr(self.settings.monitoring, 'log_json_format', False)
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 文件处理器（如果配置了日志文件）
        if self.settings.monitoring.log_file:
            self._setup_file_handler(root_logger, formatter)
        
        # 设置第三方库日志级别
        self._configure_third_party_loggers()
        
        self.is_initialized = True
        logging.info("日志系统初始化完成")
    
    def _setup_file_handler(self, logger: logging.Logger, formatter: logging.Formatter) -> None:
        """设置文件处理器"""
        log_file = Path(self.settings.monitoring.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用轮转文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=self.settings.monitoring.log_max_size,
            backupCount=self.settings.monitoring.log_backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def _configure_third_party_loggers(self) -> None:
        """配置第三方库日志级别"""
        # 设置第三方库日志级别，避免过多日志
        third_party_loggers = {
            'httpx': 'WARNING',
            'httpcore': 'WARNING',
            'urllib3': 'WARNING',
            'requests': 'WARNING',
            'asyncio': 'WARNING',
            'langchain': 'INFO',
            'langgraph': 'INFO',
            'openai': 'WARNING',
            'chromadb': 'WARNING',
            'sqlalchemy': 'WARNING',
        }
        
        for logger_name, level in third_party_loggers.items():
            logging.getLogger(logger_name).setLevel(getattr(logging, level))
    
    def get_logger(self, name: str, **extra_fields) -> logging.Logger:
        """获取指定名称的日志器
        
        Args:
            name: 日志器名称
            **extra_fields: 额外的上下文字段
            
        Returns:
            logging.Logger: 日志器实例
        """
        if not self.is_initialized:
            self.initialize()
        
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        
        logger = self._loggers[name]
        
        # 添加额外字段到日志器
        if extra_fields:
            for key, value in extra_fields.items():
                setattr(logger, key, value)
        
        return logger
    
    def create_context_logger(
        self, 
        name: str, 
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> logging.Logger:
        """创建带上下文的日志器
        
        Args:
            name: 日志器名称
            user_id: 用户ID
            agent_id: 智能体ID
            session_id: 会话ID
            request_id: 请求ID
            
        Returns:
            logging.Logger: 带上下文的日志器
        """
        context = {}
        if user_id:
            context['user_id'] = user_id
        if agent_id:
            context['agent_id'] = agent_id
        if session_id:
            context['session_id'] = session_id
        if request_id:
            context['request_id'] = request_id
        
        return self.get_logger(name, **context)


# 全局日志管理器实例
logger_manager = LoggerManager()


def get_logger(name: str, **extra_fields) -> logging.Logger:
    """获取日志器的便捷函数"""
    return logger_manager.get_logger(name, **extra_fields)


def get_context_logger(
    name: str,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> logging.Logger:
    """获取带上下文日志器的便捷函数"""
    return logger_manager.create_context_logger(
        name, user_id, agent_id, session_id, request_id
    )


def initialize_logging() -> None:
    """初始化日志系统的便捷函数"""
    logger_manager.initialize()


# 导出列表
__all__ = [
    "StructuredFormatter",
    "LoggerManager",
    "logger_manager",
    "get_logger",
    "get_context_logger",
    "initialize_logging"
]