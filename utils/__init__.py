"""工具和实用程序模块

提供各种工具函数和实用程序类。
"""

from .validation import (
    ValidationException,
    BusinessRuleException,
    PermissionDeniedException,
    validate_email,
    validate_password,
    validate_uuid,
    validate_json_schema
)

from .performance_monitoring import (
    monitor_performance,
    PerformanceMetrics,
    get_performance_stats
)

from .cache import (
    CacheManager,
    cache_result,
    invalidate_cache,
    get_cache_stats
)

from .security import (
    hash_password,
    verify_password,
    generate_token,
    verify_token,
    generate_api_key,
    SecurityManager
)

from .helpers import (
    generate_uuid,
    format_datetime,
    parse_datetime,
    sanitize_string,
    truncate_text,
    calculate_hash,
    deep_merge_dict
)

__all__ = [
    # 验证相关
    "ValidationException",
    "BusinessRuleException", 
    "PermissionDeniedException",
    "validate_email",
    "validate_password",
    "validate_uuid",
    "validate_json_schema",
    
    # 性能监控
    "monitor_performance",
    "PerformanceMetrics",
    "get_performance_stats",
    
    # 缓存管理
    "CacheManager",
    "cache_result",
    "invalidate_cache",
    "get_cache_stats",
    
    # 安全相关
    "hash_password",
    "verify_password",
    "generate_token",
    "verify_token",
    "generate_api_key",
    "SecurityManager",
    
    # 辅助函数
    "generate_uuid",
    "format_datetime",
    "parse_datetime",
    "sanitize_string",
    "truncate_text",
    "calculate_hash",
    "deep_merge_dict"
]