"""认证相关数据模型

本模块定义了用户认证、授权相关的数据模型，包括：
- 用户注册和登录请求/响应
- JWT令牌管理
- 用户权限和角色
- 会话管理
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, EmailStr, validator
from models.base_models import BaseResponse


class UserRole(str, Enum):
    """用户角色枚举"""
    ADMIN = "admin"  # 管理员
    USER = "user"    # 普通用户
    GUEST = "guest"  # 访客


class UserStatus(str, Enum):
    """用户状态枚举"""
    ACTIVE = "active"      # 活跃
    INACTIVE = "inactive"  # 非活跃
    SUSPENDED = "suspended"  # 暂停
    DELETED = "deleted"    # 已删除


class TokenType(str, Enum):
    """令牌类型枚举"""
    ACCESS = "access"    # 访问令牌
    REFRESH = "refresh"  # 刷新令牌
    RESET = "reset"      # 重置密码令牌


# 用户注册请求
class UserRegisterRequest(BaseModel):
    """用户注册请求"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱地址")
    password: str = Field(..., min_length=8, max_length=128, description="密码")
    confirm_password: str = Field(..., description="确认密码")
    full_name: Optional[str] = Field(None, max_length=100, description="全名")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('密码不匹配')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('密码必须包含至少一个大写字母')
        if not any(c.islower() for c in v):
            raise ValueError('密码必须包含至少一个小写字母')
        if not any(c.isdigit() for c in v):
            raise ValueError('密码必须包含至少一个数字')
        return v


# 用户登录请求
class UserLoginRequest(BaseModel):
    """用户登录请求"""
    username: str = Field(..., description="用户名或邮箱")
    password: str = Field(..., description="密码")
    remember_me: bool = Field(False, description="记住我")


# 用户信息模型
class UserInfo(BaseModel):
    """用户信息"""
    id: int = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    email: str = Field(..., description="邮箱地址")
    full_name: Optional[str] = Field(None, description="全名")
    role: UserRole = Field(..., description="用户角色")
    status: UserStatus = Field(..., description="用户状态")
    is_active: bool = Field(..., description="是否活跃")
    is_admin: bool = Field(..., description="是否为管理员")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    last_login_at: Optional[datetime] = Field(None, description="最后登录时间")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="用户偏好设置")


# JWT令牌信息
class TokenInfo(BaseModel):
    """JWT令牌信息"""
    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field("bearer", description="令牌类型")
    expires_in: int = Field(..., description="过期时间（秒）")
    expires_at: datetime = Field(..., description="过期时间戳")
    scope: List[str] = Field(default_factory=list, description="权限范围")


# 令牌刷新请求
class TokenRefreshRequest(BaseModel):
    """令牌刷新请求"""
    refresh_token: str = Field(..., description="刷新令牌")


# 密码重置请求
class PasswordResetRequest(BaseModel):
    """密码重置请求"""
    email: EmailStr = Field(..., description="邮箱地址")


# 密码重置确认
class PasswordResetConfirm(BaseModel):
    """密码重置确认"""
    token: str = Field(..., description="重置令牌")
    new_password: str = Field(..., min_length=8, max_length=128, description="新密码")
    confirm_password: str = Field(..., description="确认新密码")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('密码不匹配')
        return v


# 密码修改请求
class PasswordChangeRequest(BaseModel):
    """密码修改请求"""
    current_password: str = Field(..., description="当前密码")
    new_password: str = Field(..., min_length=8, max_length=128, description="新密码")
    confirm_password: str = Field(..., description="确认新密码")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('密码不匹配')
        return v


# 用户资料更新请求
class UserProfileUpdateRequest(BaseModel):
    """用户资料更新请求"""
    full_name: Optional[str] = Field(None, max_length=100, description="全名")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    preferences: Optional[Dict[str, Any]] = Field(None, description="用户偏好设置")


# 用户权限信息
class UserPermissions(BaseModel):
    """用户权限信息"""
    user_id: int = Field(..., description="用户ID")
    permissions: List[str] = Field(..., description="权限列表")
    roles: List[str] = Field(..., description="角色列表")
    is_admin: bool = Field(..., description="是否为管理员")


# 会话信息
class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str = Field(..., description="会话ID")
    user_id: int = Field(..., description="用户ID")
    created_at: datetime = Field(..., description="创建时间")
    last_activity: datetime = Field(..., description="最后活动时间")
    ip_address: Optional[str] = Field(None, description="IP地址")
    user_agent: Optional[str] = Field(None, description="用户代理")
    is_active: bool = Field(..., description="是否活跃")


# API响应模型
class UserRegisterResponse(BaseResponse):
    """用户注册响应"""
    data: UserInfo = Field(..., description="用户信息")


class UserLoginResponse(BaseResponse):
    """用户登录响应"""
    data: Dict[str, Any] = Field(..., description="登录数据")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "登录成功",
                "data": {
                    "user": {
                        "id": 1,
                        "username": "testuser",
                        "email": "test@example.com",
                        "role": "user"
                    },
                    "token": {
                        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                        "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                        "token_type": "bearer",
                        "expires_in": 3600
                    }
                }
            }
        }


class TokenRefreshResponse(BaseResponse):
    """令牌刷新响应"""
    data: TokenInfo = Field(..., description="新令牌信息")


class UserProfileResponse(BaseResponse):
    """用户资料响应"""
    data: UserInfo = Field(..., description="用户信息")


class UserPermissionsResponse(BaseResponse):
    """用户权限响应"""
    data: UserPermissions = Field(..., description="用户权限信息")


class SessionListResponse(BaseResponse):
    """会话列表响应"""
    data: List[SessionInfo] = Field(..., description="会话列表")


# JWT载荷模型
class JWTPayload(BaseModel):
    """JWT载荷"""
    sub: str = Field(..., description="主题（用户ID）")
    username: str = Field(..., description="用户名")
    email: str = Field(..., description="邮箱")
    role: str = Field(..., description="角色")
    iat: int = Field(..., description="签发时间")
    exp: int = Field(..., description="过期时间")
    jti: str = Field(..., description="JWT ID")
    token_type: TokenType = Field(..., description="令牌类型")
    scope: List[str] = Field(default_factory=list, description="权限范围")