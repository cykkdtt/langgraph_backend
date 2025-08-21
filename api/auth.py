"""用户认证和授权API路由

本模块实现了用户认证和授权相关的API端点，包括：
- 用户注册和登录
- JWT令牌管理
- 密码重置和修改
- 用户资料管理
- 会话管理
- 权限验证
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from passlib.context import CryptContext
import jwt
from email_validator import validate_email, EmailNotValidError

from models.auth_models import (
    UserRegisterRequest, UserLoginRequest, UserInfo, TokenInfo,
    TokenRefreshRequest, PasswordResetRequest, PasswordResetConfirm,
    PasswordChangeRequest, UserProfileUpdateRequest, UserPermissions,
    SessionInfo, UserRegisterResponse, UserLoginResponse,
    TokenRefreshResponse, UserProfileResponse, UserPermissionsResponse,
    SessionListResponse, JWTPayload, UserRole, UserStatus, TokenType
)
from models.base_models import BaseResponse, ErrorResponse
from models.api_models import ErrorDetail, ErrorCode
from core.database import get_async_session
from config.settings import get_settings
from core.logging import get_logger

# 初始化
router = APIRouter(prefix="/auth", tags=["认证"])
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
logger = get_logger(__name__)
settings = get_settings()

# JWT配置
JWT_SECRET_KEY = settings.jwt.secret_key or "your-secret-key-here"
JWT_ALGORITHM = settings.jwt.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.jwt.access_token_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = 7


# 密码工具函数
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """获取密码哈希"""
    return pwd_context.hash(password)


# JWT工具函数
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": secrets.token_urlsafe(32),
        "token_type": TokenType.ACCESS
    })
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """创建刷新令牌"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": secrets.token_urlsafe(32),
        "token_type": TokenType.REFRESH
    })
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: TokenType = TokenType.ACCESS) -> Optional[JWTPayload]:
    """验证令牌"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # 检查令牌类型
        if payload.get("token_type") != token_type:
            return None
            
        # 检查过期时间
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            return None
            
        return JWTPayload(**payload)
    except jwt.PyJWTError:
        return None


# 依赖函数
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security),
                          db: AsyncSession = Depends(get_async_session)) -> UserInfo:
    """获取当前用户"""
    token = credentials.credentials
    payload = verify_token(token, TokenType.ACCESS)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 从数据库获取用户信息
    user_id = int(payload.sub)
    result = await db.execute(
        select("*").select_from("users").where("id = :user_id"),
        {"user_id": user_id}
    )
    user_data = result.fetchone()
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在"
        )
    
    return UserInfo(
        id=user_data["id"],
        username=user_data["username"],
        email=user_data["email"],
        full_name=user_data["full_name"],
        role=UserRole(user_data["role"]),
        status=UserStatus(user_data["status"]),
        is_active=user_data["is_active"],
        is_admin=user_data["role"] == UserRole.ADMIN,
        created_at=user_data["created_at"],
        updated_at=user_data["updated_at"],
        last_login_at=user_data["last_login_at"],
        avatar_url=user_data["avatar_url"],
        preferences=user_data["preferences"] or {}
    )


async def get_current_active_user(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
    """获取当前活跃用户"""
    if not current_user.is_active or current_user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户账户已被禁用"
        )
    return current_user


async def get_current_admin_user(current_user: UserInfo = Depends(get_current_active_user)) -> UserInfo:
    """获取当前管理员用户"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    return current_user


async def get_current_user_websocket(token: str, db: AsyncSession) -> Optional[UserInfo]:
    """WebSocket连接的用户认证"""
    try:
        payload = verify_token(token, TokenType.ACCESS)
        
        if not payload:
            return None
        
        # 从数据库获取用户信息
        user_id = int(payload.sub)
        result = await db.execute(
            select("*").select_from("users").where("id = :user_id"),
            {"user_id": user_id}
        )
        user_data = result.fetchone()
        
        if not user_data:
            return None
        
        user_info = UserInfo(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            full_name=user_data["full_name"],
            role=UserRole(user_data["role"]),
            status=UserStatus(user_data["status"]),
            is_active=user_data["is_active"],
            is_admin=user_data["role"] == UserRole.ADMIN,
            created_at=user_data["created_at"],
            updated_at=user_data["updated_at"],
            last_login_at=user_data["last_login_at"],
            avatar_url=user_data["avatar_url"],
            preferences=user_data["preferences"] or {}
        )
        
        # 检查用户是否活跃
        if not user_info.is_active or user_info.status != UserStatus.ACTIVE:
            return None
            
        return user_info
        
    except Exception as e:
        logger.error(f"WebSocket用户认证失败: {str(e)}")
        return None


# API路由
@router.post("/register", response_model=BaseResponse[UserRegisterResponse], status_code=status.HTTP_201_CREATED)
async def register_user(request: UserRegisterRequest, db: AsyncSession = Depends(get_async_session)):
    """用户注册"""
    try:
        # 检查用户名是否已存在
        result = await db.execute(
            select("id").select_from("users").where("username = :username"),
            {"username": request.username}
        )
        if result.fetchone():
            return BaseResponse.error(
                message="用户名已存在",
                errors=[ErrorDetail(
                    code=ErrorCode.RESOURCE_ALREADY_EXISTS,
                    message="该用户名已被注册",
                    field="username"
                )]
            )
        
        # 检查邮箱是否已存在
        result = await db.execute(
            select("id").select_from("users").where("email = :email"),
            {"email": request.email}
        )
        if result.fetchone():
            return BaseResponse.error(
                message="邮箱已被注册",
                errors=[ErrorDetail(
                    code=ErrorCode.RESOURCE_ALREADY_EXISTS,
                    message="该邮箱已被注册",
                    field="email"
                )]
            )
        
        # 创建用户
        hashed_password = get_password_hash(request.password)
        now = datetime.utcnow()
        
        user_data = {
            "username": request.username,
            "email": request.email,
            "password_hash": hashed_password,
            "full_name": request.full_name,
            "role": UserRole.USER,
            "status": UserStatus.ACTIVE,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "preferences": {}
        }
        
        result = await db.execute(
            "INSERT INTO users (username, email, password_hash, full_name, role, status, is_active, created_at, updated_at, preferences) "
            "VALUES (:username, :email, :password_hash, :full_name, :role, :status, :is_active, :created_at, :updated_at, :preferences) "
            "RETURNING *",
            user_data
        )
        
        user_row = result.fetchone()
        await db.commit()
        
        user_info = UserInfo(
            id=user_row["id"],
            username=user_row["username"],
            email=user_row["email"],
            full_name=user_row["full_name"],
            role=UserRole(user_row["role"]),
            status=UserStatus(user_row["status"]),
            is_active=user_row["is_active"],
            is_admin=False,
            created_at=user_row["created_at"],
            updated_at=user_row["updated_at"],
            last_login_at=None,
            avatar_url=None,
            preferences={}
        )
        
        logger.info(f"用户注册成功: {request.username}")
        
        return BaseResponse.success(
            data=UserRegisterResponse(
                success=True,
                message="注册成功",
                data=user_info
            ),
            message="注册成功"
        )
        
    except Exception as e:
        logger.error(f"用户注册失败: {str(e)}")
        await db.rollback()
        return BaseResponse.error(
            message="注册失败，请稍后重试",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )


@router.post("/login", response_model=BaseResponse[UserLoginResponse])
async def login_user(request: UserLoginRequest, db: AsyncSession = Depends(get_async_session)):
    """用户登录"""
    try:
        # 查找用户（支持用户名或邮箱登录）
        result = await db.execute(
            "SELECT * FROM users WHERE username = :username OR email = :username",
            {"username": request.username}
        )
        user_data = result.fetchone()
        
        if not user_data or not verify_password(request.password, user_data["password_hash"]):
            return BaseResponse.error(
                message="登录失败",
                errors=[ErrorDetail(
                    code=ErrorCode.AUTHENTICATION_FAILED,
                    message="用户名或密码错误"
                )]
            )
        
        # 检查用户状态
        if not user_data["is_active"] or user_data["status"] != UserStatus.ACTIVE:
            return BaseResponse.error(
                message="账户已被禁用",
                errors=[ErrorDetail(
                    code=ErrorCode.PERMISSION_DENIED,
                    message="账户已被禁用，请联系管理员"
                )]
            )
        
        # 更新最后登录时间
        now = datetime.utcnow()
        await db.execute(
            "UPDATE users SET last_login_at = :now WHERE id = :user_id",
            {"now": now, "user_id": user_data["id"]}
        )
        await db.commit()
        
        # 创建令牌
        token_data = {
            "sub": str(user_data["id"]),
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data["role"],
            "scope": ["read", "write"] if user_data["role"] == UserRole.ADMIN else ["read"]
        }
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        if request.remember_me:
            access_token_expires = timedelta(days=1)  # 记住我时延长到1天
        
        access_token = create_access_token(token_data, access_token_expires)
        refresh_token = create_refresh_token(token_data)
        
        user_info = UserInfo(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            full_name=user_data["full_name"],
            role=UserRole(user_data["role"]),
            status=UserStatus(user_data["status"]),
            is_active=user_data["is_active"],
            is_admin=user_data["role"] == UserRole.ADMIN,
            created_at=user_data["created_at"],
            updated_at=user_data["updated_at"],
            last_login_at=now,
            avatar_url=user_data["avatar_url"],
            preferences=user_data["preferences"] or {}
        )
        
        token_info = TokenInfo(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            expires_at=datetime.utcnow() + access_token_expires,
            scope=token_data["scope"]
        )
        
        logger.info(f"用户登录成功: {user_data['username']}")
        
        return BaseResponse.success(
            data=UserLoginResponse(
                success=True,
                message="登录成功",
                data={
                    "user": user_info.dict(),
                    "token": token_info.dict()
                }
            ),
            message="登录成功"
        )
        
    except Exception as e:
        logger.error(f"用户登录失败: {str(e)}")
        return BaseResponse.error(
            message="登录失败，请稍后重试",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )


@router.post("/refresh", response_model=BaseResponse[TokenRefreshResponse])
async def refresh_token(request: TokenRefreshRequest, db: AsyncSession = Depends(get_async_session)):
    """刷新访问令牌"""
    try:
        # 验证刷新令牌
        payload = verify_token(request.refresh_token, TokenType.REFRESH)
        if not payload:
            return BaseResponse.error(
                message="无效的刷新令牌",
                errors=[ErrorDetail(
                    code=ErrorCode.TOKEN_INVALID,
                    message="令牌格式错误或已被篡改"
                )]
            )
        
        # 获取用户信息
        user_id = int(payload.sub)
        result = await db.execute(
            "SELECT * FROM users WHERE id = :user_id AND is_active = true",
            {"user_id": user_id}
        )
        user_data = result.fetchone()
        
        if not user_data:
            return BaseResponse.error(
                message="无效的刷新令牌",
                errors=[ErrorDetail(
                    code=ErrorCode.TOKEN_INVALID,
                    message="用户不存在或已被禁用"
                )]
            )
        
        # 创建新的访问令牌
        token_data = {
            "sub": str(user_data["id"]),
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data["role"],
            "scope": payload.scope
        }
        
        access_token = create_access_token(token_data)
        new_refresh_token = create_refresh_token(token_data)
        
        token_info = TokenInfo(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            expires_at=datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            scope=token_data["scope"]
        )
        
        return BaseResponse.success(
            data=TokenRefreshResponse(
                success=True,
                message="令牌刷新成功",
                data=token_info
            ),
            message="令牌刷新成功"
        )
        
    except Exception as e:
        logger.error(f"令牌刷新失败: {str(e)}")
        return BaseResponse.error(
            message="令牌刷新失败",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )


@router.post("/logout", response_model=BaseResponse[Dict[str, str]])
async def logout_user(current_user: UserInfo = Depends(get_current_active_user)):
    """用户登出"""
    try:
        # 在实际应用中，这里可以将令牌加入黑名单
        # 目前只是返回成功响应
        logger.info(f"用户登出: {current_user.username}")
        
        return BaseResponse.success(
            data={"message": "登出成功"},
            message="登出成功"
        )
    except Exception as e:
        return BaseResponse.error(
            message="登出失败",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )


@router.get("/me", response_model=BaseResponse[UserInfo])
async def get_current_user_profile(current_user: UserInfo = Depends(get_current_active_user)):
    """获取当前用户资料"""
    try:
        return BaseResponse.success(
            data=current_user,
            message="获取用户资料成功"
        )
    except Exception as e:
        return BaseResponse.error(
            message="获取用户资料失败",
            errors=[ErrorDetail(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(e)
            )]
        )


@router.put("/me", response_model=UserProfileResponse)
async def update_current_user_profile(
    request: UserProfileUpdateRequest,
    current_user: UserInfo = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """更新当前用户资料"""
    try:
        update_data = {}
        if request.full_name is not None:
            update_data["full_name"] = request.full_name
        if request.avatar_url is not None:
            update_data["avatar_url"] = request.avatar_url
        if request.preferences is not None:
            update_data["preferences"] = request.preferences
        
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            
            # 构建更新查询
            set_clause = ", ".join([f"{key} = :{key}" for key in update_data.keys()])
            query = f"UPDATE users SET {set_clause} WHERE id = :user_id RETURNING *"
            update_data["user_id"] = current_user.id
            
            result = await db.execute(query, update_data)
            user_row = result.fetchone()
            await db.commit()
            
            updated_user = UserInfo(
                id=user_row["id"],
                username=user_row["username"],
                email=user_row["email"],
                full_name=user_row["full_name"],
                role=UserRole(user_row["role"]),
                status=UserStatus(user_row["status"]),
                is_active=user_row["is_active"],
                is_admin=user_row["role"] == UserRole.ADMIN,
                created_at=user_row["created_at"],
                updated_at=user_row["updated_at"],
                last_login_at=user_row["last_login_at"],
                avatar_url=user_row["avatar_url"],
                preferences=user_row["preferences"] or {}
            )
            
            logger.info(f"用户资料更新成功: {current_user.username}")
            
            return UserProfileResponse(
                success=True,
                message="用户资料更新成功",
                data=updated_user
            )
        else:
            return UserProfileResponse(
                success=True,
                message="没有需要更新的内容",
                data=current_user
            )
            
    except Exception as e:
        logger.error(f"用户资料更新失败: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="用户资料更新失败"
        )


@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    current_user: UserInfo = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """修改密码"""
    try:
        # 获取当前密码哈希
        result = await db.execute(
            "SELECT password_hash FROM users WHERE id = :user_id",
            {"user_id": current_user.id}
        )
        user_data = result.fetchone()
        
        if not user_data or not verify_password(request.current_password, user_data["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="当前密码错误"
            )
        
        # 更新密码
        new_password_hash = get_password_hash(request.new_password)
        await db.execute(
            "UPDATE users SET password_hash = :password_hash, updated_at = :updated_at WHERE id = :user_id",
            {
                "password_hash": new_password_hash,
                "updated_at": datetime.utcnow(),
                "user_id": current_user.id
            }
        )
        await db.commit()
        
        logger.info(f"用户密码修改成功: {current_user.username}")
        
        return BaseResponse(
            success=True,
            message="密码修改成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"密码修改失败: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="密码修改失败"
        )


@router.get("/permissions", response_model=UserPermissionsResponse)
async def get_current_user_permissions(current_user: UserInfo = Depends(get_current_active_user)):
    """获取当前用户权限"""
    permissions = ["read"]
    if current_user.is_admin:
        permissions.extend(["write", "admin", "delete"])
    elif current_user.role == UserRole.USER:
        permissions.append("write")
    
    user_permissions = UserPermissions(
        user_id=current_user.id,
        permissions=permissions,
        roles=[current_user.role.value],
        is_admin=current_user.is_admin
    )
    
    return UserPermissionsResponse(
        success=True,
        message="获取用户权限成功",
        data=user_permissions
    )