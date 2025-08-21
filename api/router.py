"""API路由管理模块

统一管理所有API路由、中间件、依赖注入和路由配置。
提供路由注册、权限控制、限流等功能。
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict

from ..config.settings import get_settings
from ..utils.validation import APIException, ValidationException, handle_exceptions
from ..utils.performance_monitoring import get_performance_monitor, PerformanceMiddleware
from ..models.response_models import BaseResponse, ErrorCode, ResponseStatus
from ..models.auth_models import UserInfo, UserRole, UserStatus


class RateLimitMiddleware(BaseHTTPMiddleware):
    """API限流中间件"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next):
        # 获取客户端IP
        client_ip = request.client.host
        current_time = datetime.now()
        
        # 清理过期的请求记录
        cutoff_time = current_time - timedelta(minutes=1)
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > cutoff_time
        ]
        
        # 检查是否超过限制
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            self.logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": 60
                }
            )
        
        # 记录当前请求
        self.requests[client_ip].append(current_time)
        
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """安全头中间件"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # 添加安全头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger("api.requests")
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 记录请求信息
        self.logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"from {request.client.host}"
        )
        
        try:
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录响应信息
            self.logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Time: {process_time:.3f}s"
            )
            
            # 添加处理时间头
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            self.logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- Error: {str(e)} - Time: {process_time:.3f}s"
            )
            raise


class RoutePermission:
    """路由权限定义"""
    
    def __init__(
        self,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
        require_auth: bool = True,
        rate_limit: Optional[int] = None
    ):
        self.roles = roles or []
        self.permissions = permissions or []
        self.require_auth = require_auth
        self.rate_limit = rate_limit


class RouterManager:
    """路由管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.security = HTTPBearer(auto_error=False)
        self.routers: Dict[str, APIRouter] = {}
        self.route_permissions: Dict[str, RoutePermission] = {}
        
        # 创建主路由器
        self.main_router = APIRouter(prefix=self.settings.api_prefix)
        
        # 性能监控
        self.performance_monitor = get_performance_monitor()
    
    def create_router(
        self,
        prefix: str,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[Depends]] = None
    ) -> APIRouter:
        """创建子路由器"""
        router = APIRouter(
            prefix=prefix,
            tags=tags or [],
            dependencies=dependencies or []
        )
        
        self.routers[prefix] = router
        return router
    
    def register_router(
        self,
        router: APIRouter,
        prefix: str = "",
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[Depends]] = None
    ):
        """注册路由器到主路由器"""
        self.main_router.include_router(
            router,
            prefix=prefix,
            tags=tags or [],
            dependencies=dependencies or []
        )
        
        self.logger.info(f"Registered router with prefix: {prefix}")
    
    def set_route_permission(
        self,
        path: str,
        permission: RoutePermission
    ):
        """设置路由权限"""
        self.route_permissions[path] = permission
    
    def require_permission(
        self,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
        require_auth: bool = True
    ):
        """权限装饰器"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 这里可以添加权限检查逻辑
                # 暂时跳过具体实现，等待认证模块完成
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def rate_limit(self, requests_per_minute: int):
        """限流装饰器"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                # 简单的内存限流实现
                # 生产环境建议使用Redis
                client_ip = request.client.host
                current_time = datetime.now()
                
                # 这里可以添加更复杂的限流逻辑
                return await func(request, *args, **kwargs)
            return wrapper
        return decorator
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """获取当前用户依赖"""
        # 这里应该验证JWT令牌并返回用户信息
        # 暂时返回模拟用户，等待认证模块完成
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # TODO: 实现JWT令牌验证
        return UserInfo(
            id=1,
            username="mock_user",
            email="mock@example.com",
            full_name="Mock User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            is_active=True,
            is_admin=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    def get_optional_user(self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
        """获取可选用户依赖"""
        if not credentials:
            return None
        
        try:
            return self.get_current_user(credentials)
        except HTTPException:
            return None
    
    def create_response_handler(self):
        """创建统一响应处理器"""
        def response_handler(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    
                    # 如果结果已经是BaseResponse，直接返回
                    if isinstance(result, BaseResponse):
                        return result
                    
                    # 否则包装成标准响应
                    return BaseResponse(
                        status=ResponseStatus.SUCCESS,
                        data=result,
                        message="Operation completed successfully"
                    )
                    
                except APIException as e:
                    return BaseResponse(
                        status=ResponseStatus.ERROR,
                        error_code=e.error_code,
                        message=e.message,
                        details=e.details
                    )
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                    return BaseResponse(
                        status=ResponseStatus.ERROR,
                        error_code=ErrorCode.INTERNAL_ERROR,
                        message="An unexpected error occurred"
                    )
            
            return wrapper
        return response_handler
    
    def add_middleware(self, app):
        """添加中间件"""
        # 性能监控中间件
        app.add_middleware(PerformanceMiddleware)
        
        # 请求日志中间件
        app.add_middleware(RequestLoggingMiddleware)
        
        # 安全头中间件
        app.add_middleware(SecurityHeadersMiddleware)
        
        # 限流中间件
        if not self.settings.is_development:
            app.add_middleware(
                RateLimitMiddleware,
                requests_per_minute=self.settings.security.rate_limit_requests
            )
        
        # GZIP压缩中间件
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # CORS中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.cors.allow_origins,
            allow_credentials=self.settings.cors.allow_credentials,
            allow_methods=self.settings.cors.allow_methods,
            allow_headers=self.settings.cors.allow_headers,
            expose_headers=self.settings.cors.expose_headers,
            max_age=self.settings.cors.max_age
        )
        
        # 可信主机中间件（生产环境）
        if self.settings.is_production:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=["*"]  # 应该配置具体的主机名
            )
        
        self.logger.info("All middleware added successfully")
    
    def get_router(self) -> APIRouter:
        """获取主路由器"""
        return self.main_router
    
    def get_route_summary(self) -> Dict[str, Any]:
        """获取路由摘要"""
        routes = []
        
        for route in self.main_router.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                routes.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": getattr(route, 'name', 'Unknown'),
                    "tags": getattr(route, 'tags', [])
                })
        
        return {
            "total_routes": len(routes),
            "routes": routes,
            "registered_routers": list(self.routers.keys()),
            "route_permissions": {
                path: {
                    "roles": perm.roles,
                    "permissions": perm.permissions,
                    "require_auth": perm.require_auth,
                    "rate_limit": perm.rate_limit
                }
                for path, perm in self.route_permissions.items()
            }
        }


# 全局路由管理器实例
router_manager = RouterManager()


# 便捷函数和装饰器
def get_router_manager() -> RouterManager:
    """获取路由管理器"""
    return router_manager


def create_router(
    prefix: str,
    tags: Optional[List[str]] = None,
    dependencies: Optional[List[Depends]] = None
) -> APIRouter:
    """创建路由器"""
    return router_manager.create_router(prefix, tags, dependencies)


def require_auth(func: Callable) -> Callable:
    """需要认证装饰器"""
    return router_manager.require_permission(require_auth=True)(func)


def require_roles(*roles: str):
    """需要角色装饰器"""
    def decorator(func: Callable) -> Callable:
        return router_manager.require_permission(roles=list(roles))(func)
    return decorator


def require_permissions(*permissions: str):
    """需要权限装饰器"""
    def decorator(func: Callable) -> Callable:
        return router_manager.require_permission(permissions=list(permissions))(func)
    return decorator


def rate_limit(requests_per_minute: int):
    """限流装饰器"""
    return router_manager.rate_limit(requests_per_minute)


def get_current_user():
    """获取当前用户依赖"""
    return Depends(router_manager.get_current_user)


def get_optional_user():
    """获取可选用户依赖"""
    return Depends(router_manager.get_optional_user)


def handle_response(func: Callable) -> Callable:
    """响应处理装饰器"""
    return router_manager.create_response_handler()(func)


# 常用依赖组合
class CommonDependencies:
    """常用依赖组合"""
    
    @staticmethod
    def authenticated_user():
        """需要认证的用户"""
        return get_current_user()
    
    @staticmethod
    def optional_user():
        """可选用户"""
        return get_optional_user()
    
    @staticmethod
    def admin_user():
        """管理员用户"""
        # TODO: 实现管理员权限检查
        return get_current_user()
    
    @staticmethod
    def pagination_params(
        page: int = 1,
        size: int = 20,
        max_size: int = 100
    ):
        """分页参数"""
        if page < 1:
            raise ValidationException("Page must be greater than 0")
        
        if size < 1 or size > max_size:
            raise ValidationException(f"Size must be between 1 and {max_size}")
        
        return {
            "page": page,
            "size": size,
            "offset": (page - 1) * size
        }