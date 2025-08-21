"""用户服务

提供用户管理相关的业务逻辑，包括用户注册、认证、权限管理等功能。
"""

import hashlib
import secrets
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID

from pydantic import BaseModel, EmailStr, validator
from sqlalchemy.orm import Session

from .base import BaseService, ServiceError, CacheConfig, publish_event
from ..models.database_models import User
from ..models.response_models import UserResponse, BaseResponse
from ..database.repositories import UserRepository
from ..utils.validation import (
    ValidationException, BusinessRuleException, 
    PermissionDeniedException, DataValidator
)
from ..utils.performance_monitoring import monitor_performance


class UserCreateSchema(BaseModel):
    """用户创建模式"""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if not DataValidator.validate_username(v):
            raise ValueError("Invalid username format")
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if not DataValidator.validate_password(v):
            raise ValueError("Password does not meet security requirements")
        return v


class UserUpdateSchema(BaseModel):
    """用户更新模式"""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    is_active: Optional[bool] = None
    rate_limit_tier: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if v is not None and not DataValidator.validate_username(v):
            raise ValueError("Invalid username format")
        return v


class PasswordChangeSchema(BaseModel):
    """密码修改模式"""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if not DataValidator.validate_password(v):
            raise ValueError("New password does not meet security requirements")
        return v


class UserLoginSchema(BaseModel):
    """用户登录模式"""
    username_or_email: str
    password: str


class UserStatsResponse(BaseModel):
    """用户统计响应"""
    total_users: int
    active_users: int
    new_users_today: int
    new_users_this_week: int
    new_users_this_month: int
    users_by_tier: Dict[str, int]
    generated_at: datetime


class UserService(BaseService[User, UserCreateSchema, UserUpdateSchema, UserResponse]):
    """用户服务"""
    
    def __init__(self, repository: UserRepository, session: Optional[Session] = None):
        cache_config = CacheConfig(
            enabled=True,
            ttl=600,  # 10分钟
            key_prefix="user_service",
            invalidate_on_update=True
        )
        super().__init__(repository, UserResponse, cache_config, session)
        self.user_repository = repository
    
    def _hash_password(self, password: str) -> str:
        """哈希密码"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """验证密码"""
        try:
            salt, stored_hash = hashed_password.split(':')
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return password_hash.hex() == stored_hash
        except ValueError:
            return False
    
    def _generate_api_key(self) -> str:
        """生成API密钥"""
        return f"lgs_{secrets.token_urlsafe(32)}"
    
    def _hash_api_key(self, api_key: str) -> str:
        """哈希API密钥"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def _validate_business_rules(self, data: Any, action: str = "create"):
        """验证业务规则"""
        if action == "create":
            # 检查用户名是否已存在
            if self.user_repository.get_by_username(data.username):
                raise BusinessRuleException("Username already exists")
            
            # 检查邮箱是否已存在
            if self.user_repository.get_by_email(data.email):
                raise BusinessRuleException("Email already exists")
        
        elif action == "update":
            if hasattr(data, 'username') and data.username:
                existing_user = self.user_repository.get_by_username(data.username)
                if existing_user and str(existing_user.id) != str(self.current_user_id):
                    raise BusinessRuleException("Username already exists")
            
            if hasattr(data, 'email') and data.email:
                existing_user = self.user_repository.get_by_email(data.email)
                if existing_user and str(existing_user.id) != str(self.current_user_id):
                    raise BusinessRuleException("Email already exists")
    
    def _check_permission(self, action: str, resource: Any = None):
        """检查权限"""
        if action in ["create"]:
            # 注册不需要认证
            return
        
        if not self.current_user_id:
            raise PermissionDeniedException("Authentication required")
        
        if action in ["update", "delete", "change_password", "generate_api_key"]:
            # 只能操作自己的账户，除非是管理员
            if resource and str(resource.id) != str(self.current_user_id):
                if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                    raise PermissionDeniedException("Can only modify your own account")
        
        if action in ["list", "statistics", "admin_update"]:
            # 需要管理员权限
            if not (self.current_user and getattr(self.current_user, 'is_admin', False)):
                raise PermissionDeniedException("Admin privileges required")
    
    @monitor_performance
    @publish_event("user_registered", "user")
    def register(self, data: UserCreateSchema) -> BaseResponse[UserResponse]:
        """用户注册"""
        try:
            # 验证业务规则
            self._validate_business_rules(data, "create")
            
            # 哈希密码
            hashed_password = self._hash_password(data.password)
            
            # 创建用户数据
            user_data = {
                "username": data.username,
                "email": data.email,
                "password_hash": hashed_password,
                "full_name": data.full_name,
                "avatar_url": data.avatar_url,
                "is_active": True,
                "rate_limit_tier": "basic",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # 创建用户
            user = self.user_repository.create(user_data)
            
            # 转换为响应模型
            response_data = self._transform_to_response(user)
            
            return self._create_success_response(
                response_data,
                "User registered successfully"
            )
            
        except (ValidationException, BusinessRuleException):
            raise
        except Exception as e:
            self.logger.error(f"Error registering user: {e}")
            raise ServiceError("Failed to register user")
    
    @monitor_performance
    def authenticate(self, data: UserLoginSchema) -> BaseResponse[Dict[str, Any]]:
        """用户认证"""
        try:
            # 根据用户名或邮箱查找用户
            user = None
            if "@" in data.username_or_email:
                user = self.user_repository.get_by_email(data.username_or_email)
            else:
                user = self.user_repository.get_by_username(data.username_or_email)
            
            if not user:
                raise BusinessRuleException("Invalid credentials")
            
            # 检查用户是否激活
            if not user.is_active:
                raise BusinessRuleException("Account is deactivated")
            
            # 验证密码
            if not self._verify_password(data.password, user.password_hash):
                raise BusinessRuleException("Invalid credentials")
            
            # 更新最后登录时间
            self.user_repository.update_last_login(user.id)
            
            # 返回用户信息（不包含密码）
            user_response = self._transform_to_response(user)
            
            response_data = {
                "user": user_response,
                "message": "Authentication successful"
            }
            
            return self._create_success_response(
                response_data,
                "User authenticated successfully"
            )
            
        except BusinessRuleException:
            raise
        except Exception as e:
            self.logger.error(f"Error authenticating user: {e}")
            raise ServiceError("Authentication failed")
    
    @monitor_performance
    def change_password(
        self, 
        user_id: UUID, 
        data: PasswordChangeSchema
    ) -> BaseResponse[None]:
        """修改密码"""
        try:
            # 获取用户
            user = self.user_repository.get_or_404(user_id)
            
            # 权限检查
            self._check_permission("change_password", user)
            
            # 验证当前密码
            if not self._verify_password(data.current_password, user.password_hash):
                raise BusinessRuleException("Current password is incorrect")
            
            # 哈希新密码
            new_password_hash = self._hash_password(data.new_password)
            
            # 更新密码
            self.user_repository.update(user_id, {
                "password_hash": new_password_hash,
                "updated_at": datetime.utcnow()
            })
            
            return self._create_success_response(
                None,
                "Password changed successfully"
            )
            
        except (BusinessRuleException, PermissionDeniedException):
            raise
        except Exception as e:
            self.logger.error(f"Error changing password for user {user_id}: {e}")
            raise ServiceError("Failed to change password")
    
    @monitor_performance
    def generate_api_key(self, user_id: UUID) -> BaseResponse[Dict[str, str]]:
        """生成API密钥"""
        try:
            # 获取用户
            user = self.user_repository.get_or_404(user_id)
            
            # 权限检查
            self._check_permission("generate_api_key", user)
            
            # 生成新的API密钥
            api_key = self._generate_api_key()
            api_key_hash = self._hash_api_key(api_key)
            
            # 更新用户的API密钥哈希
            self.user_repository.update(user_id, {
                "api_key_hash": api_key_hash,
                "updated_at": datetime.utcnow()
            })
            
            response_data = {
                "api_key": api_key,
                "message": "API key generated successfully. Please store it securely as it won't be shown again."
            }
            
            return self._create_success_response(
                response_data,
                "API key generated successfully"
            )
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error generating API key for user {user_id}: {e}")
            raise ServiceError("Failed to generate API key")
    
    @monitor_performance
    def verify_api_key(self, api_key: str) -> Optional[User]:
        """验证API密钥"""
        try:
            api_key_hash = self._hash_api_key(api_key)
            return self.user_repository.get_by_api_key_hash(api_key_hash)
        except Exception as e:
            self.logger.error(f"Error verifying API key: {e}")
            return None
    
    @monitor_performance
    def deactivate_user(self, user_id: UUID) -> BaseResponse[None]:
        """停用用户"""
        try:
            # 获取用户
            user = self.user_repository.get_or_404(user_id)
            
            # 权限检查
            self._check_permission("update", user)
            
            # 停用用户
            self.user_repository.deactivate_user(user_id)
            
            return self._create_success_response(
                None,
                "User deactivated successfully"
            )
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error deactivating user {user_id}: {e}")
            raise ServiceError("Failed to deactivate user")
    
    @monitor_performance
    def activate_user(self, user_id: UUID) -> BaseResponse[None]:
        """激活用户"""
        try:
            # 获取用户
            user = self.user_repository.get_or_404(user_id)
            
            # 权限检查（需要管理员权限）
            self._check_permission("admin_update", user)
            
            # 激活用户
            self.user_repository.update(user_id, {
                "is_active": True,
                "updated_at": datetime.utcnow()
            })
            
            return self._create_success_response(
                None,
                "User activated successfully"
            )
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error activating user {user_id}: {e}")
            raise ServiceError("Failed to activate user")
    
    @monitor_performance
    def update_rate_limit_tier(
        self, 
        user_id: UUID, 
        tier: str
    ) -> BaseResponse[UserResponse]:
        """更新用户限流等级"""
        try:
            # 获取用户
            user = self.user_repository.get_or_404(user_id)
            
            # 权限检查（需要管理员权限）
            self._check_permission("admin_update", user)
            
            # 验证限流等级
            valid_tiers = ["basic", "premium", "enterprise", "unlimited"]
            if tier not in valid_tiers:
                raise ValidationException(f"Invalid rate limit tier. Must be one of: {valid_tiers}")
            
            # 更新限流等级
            updated_user = self.user_repository.update_rate_limit_tier(user_id, tier)
            
            # 转换为响应模型
            response_data = self._transform_to_response(updated_user)
            
            return self._create_success_response(
                response_data,
                f"Rate limit tier updated to {tier}"
            )
            
        except (ValidationException, PermissionDeniedException):
            raise
        except Exception as e:
            self.logger.error(f"Error updating rate limit tier for user {user_id}: {e}")
            raise ServiceError("Failed to update rate limit tier")
    
    @monitor_performance
    def get_active_users(self, limit: int = 100) -> BaseResponse[List[UserResponse]]:
        """获取活跃用户列表"""
        try:
            # 权限检查
            self._check_permission("list")
            
            # 获取活跃用户
            active_users = self.user_repository.get_active_users(limit)
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(user) for user in active_users
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting active users: {e}")
            raise ServiceError("Failed to get active users")
    
    @monitor_performance
    def get_users_by_tier(self, tier: str) -> BaseResponse[List[UserResponse]]:
        """根据限流等级获取用户"""
        try:
            # 权限检查
            self._check_permission("list")
            
            # 获取指定等级的用户
            users = self.user_repository.get_users_by_rate_limit_tier(tier)
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(user) for user in users
            ]
            
            return self._create_success_response(response_data)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting users by tier {tier}: {e}")
            raise ServiceError(f"Failed to get users by tier {tier}")
    
    @monitor_performance
    def get_user_statistics(self) -> BaseResponse[UserStatsResponse]:
        """获取用户统计信息"""
        try:
            # 权限检查
            self._check_permission("statistics")
            
            # 获取统计数据
            total_users = self.user_repository.count()
            active_users = len(self.user_repository.get_active_users())
            
            # 计算时间范围
            now = datetime.utcnow()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = today_start - timedelta(days=now.weekday())
            month_start = today_start.replace(day=1)
            
            # 获取新用户统计
            new_users_today = self.user_repository.count_users_created_after(today_start)
            new_users_this_week = self.user_repository.count_users_created_after(week_start)
            new_users_this_month = self.user_repository.count_users_created_after(month_start)
            
            # 获取各等级用户数量
            users_by_tier = {}
            for tier in ["basic", "premium", "enterprise", "unlimited"]:
                users_by_tier[tier] = len(self.user_repository.get_users_by_rate_limit_tier(tier))
            
            stats = UserStatsResponse(
                total_users=total_users,
                active_users=active_users,
                new_users_today=new_users_today,
                new_users_this_week=new_users_this_week,
                new_users_this_month=new_users_this_month,
                users_by_tier=users_by_tier,
                generated_at=now
            )
            
            return self._create_success_response(stats)
            
        except PermissionDeniedException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting user statistics: {e}")
            raise ServiceError("Failed to get user statistics")


# 便捷函数
def create_user_service(session: Optional[Session] = None) -> UserService:
    """创建用户服务实例"""
    from ..database.repositories import get_repository_manager
    
    repo_manager = get_repository_manager()
    user_repository = repo_manager.get_user_repository(session)
    
    return UserService(user_repository, session)