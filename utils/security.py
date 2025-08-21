"""安全工具模块

提供密码哈希、令牌生成验证、API密钥管理、权限控制等安全功能。
"""

import secrets
import hashlib
import hmac
import base64
import jwt
import bcrypt
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """权限级别枚举"""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class TokenPayload:
    """JWT令牌载荷"""
    user_id: str
    username: str
    email: str
    permissions: List[str]
    issued_at: datetime
    expires_at: datetime
    token_type: str = "access"
    session_id: Optional[str] = None


@dataclass
class APIKeyInfo:
    """API密钥信息"""
    key_id: str
    user_id: str
    name: str
    permissions: List[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    usage_count: int = 0
    rate_limit: Optional[int] = None


class PasswordHasher:
    """密码哈希器"""
    
    def __init__(self, rounds: int = 12):
        self.rounds = rounds
    
    def hash_password(self, password: str) -> str:
        """哈希密码"""
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """验证密码"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def is_password_strong(self, password: str) -> tuple[bool, List[str]]:
        """检查密码强度"""
        issues = []
        
        if len(password) < 8:
            issues.append("密码长度至少8位")
        
        if not re.search(r'[a-z]', password):
            issues.append("密码必须包含小写字母")
        
        if not re.search(r'[A-Z]', password):
            issues.append("密码必须包含大写字母")
        
        if not re.search(r'\d', password):
            issues.append("密码必须包含数字")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("密码必须包含特殊字符")
        
        # 检查常见弱密码
        weak_passwords = [
            'password', '123456', 'qwerty', 'abc123', 
            'password123', '12345678', 'admin'
        ]
        
        if password.lower() in weak_passwords:
            issues.append("密码过于简单，请使用更复杂的密码")
        
        return len(issues) == 0, issues


class TokenManager:
    """JWT令牌管理器"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=7)
    
    def create_access_token(self, payload: TokenPayload) -> str:
        """创建访问令牌"""
        now = datetime.utcnow()
        payload.issued_at = now
        payload.expires_at = now + self.access_token_expire
        payload.token_type = "access"
        
        jwt_payload = {
            "user_id": payload.user_id,
            "username": payload.username,
            "email": payload.email,
            "permissions": payload.permissions,
            "iat": payload.issued_at.timestamp(),
            "exp": payload.expires_at.timestamp(),
            "type": payload.token_type,
            "session_id": payload.session_id
        }
        
        return jwt.encode(jwt_payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, payload: TokenPayload) -> str:
        """创建刷新令牌"""
        now = datetime.utcnow()
        payload.issued_at = now
        payload.expires_at = now + self.refresh_token_expire
        payload.token_type = "refresh"
        
        jwt_payload = {
            "user_id": payload.user_id,
            "username": payload.username,
            "email": payload.email,
            "permissions": payload.permissions,
            "iat": payload.issued_at.timestamp(),
            "exp": payload.expires_at.timestamp(),
            "type": payload.token_type,
            "session_id": payload.session_id
        }
        
        return jwt.encode(jwt_payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """验证令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            return TokenPayload(
                user_id=payload["user_id"],
                username=payload["username"],
                email=payload["email"],
                permissions=payload["permissions"],
                issued_at=datetime.fromtimestamp(payload["iat"]),
                expires_at=datetime.fromtimestamp(payload["exp"]),
                token_type=payload.get("type", "access"),
                session_id=payload.get("session_id")
            )
        
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """使用刷新令牌创建新的访问令牌"""
        payload = self.verify_token(refresh_token)
        
        if not payload or payload.token_type != "refresh":
            return None
        
        # 创建新的访问令牌
        new_payload = TokenPayload(
            user_id=payload.user_id,
            username=payload.username,
            email=payload.email,
            permissions=payload.permissions,
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.access_token_expire,
            session_id=payload.session_id
        )
        
        return self.create_access_token(new_payload)


class APIKeyManager:
    """API密钥管理器"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = None
    
    def generate_api_key(self, prefix: str = "lgs") -> str:
        """生成API密钥"""
        # 生成32字节的随机数据
        random_bytes = secrets.token_bytes(32)
        # 转换为base64编码
        key_data = base64.urlsafe_b64encode(random_bytes).decode('utf-8').rstrip('=')
        # 添加前缀
        return f"{prefix}_{key_data}"
    
    def hash_api_key(self, api_key: str) -> str:
        """哈希API密钥用于存储"""
        return hashlib.sha256(api_key.encode('utf-8')).hexdigest()
    
    def verify_api_key(self, api_key: str, hashed_key: str) -> bool:
        """验证API密钥"""
        return hmac.compare_digest(self.hash_api_key(api_key), hashed_key)
    
    def encrypt_sensitive_data(self, data: str) -> Optional[str]:
        """加密敏感数据"""
        if not self.cipher:
            return data
        
        try:
            encrypted = self.cipher.encrypt(data.encode('utf-8'))
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return None
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Optional[str]:
        """解密敏感数据"""
        if not self.cipher:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return None


class PermissionManager:
    """权限管理器"""
    
    def __init__(self):
        self.permission_hierarchy = {
            PermissionLevel.SUPER_ADMIN: [
                PermissionLevel.ADMIN,
                PermissionLevel.WRITE,
                PermissionLevel.READ,
                PermissionLevel.NONE
            ],
            PermissionLevel.ADMIN: [
                PermissionLevel.WRITE,
                PermissionLevel.READ,
                PermissionLevel.NONE
            ],
            PermissionLevel.WRITE: [
                PermissionLevel.READ,
                PermissionLevel.NONE
            ],
            PermissionLevel.READ: [
                PermissionLevel.NONE
            ],
            PermissionLevel.NONE: []
        }
    
    def has_permission(self, user_permissions: List[str], 
                      required_permission: str) -> bool:
        """检查用户是否有指定权限"""
        # 检查直接权限
        if required_permission in user_permissions:
            return True
        
        # 检查权限层级
        try:
            required_level = PermissionLevel(required_permission)
            
            for user_perm in user_permissions:
                try:
                    user_level = PermissionLevel(user_perm)
                    if required_level in self.permission_hierarchy.get(user_level, []):
                        return True
                except ValueError:
                    # 不是标准权限级别，跳过
                    continue
        
        except ValueError:
            # 不是标准权限级别，只检查直接匹配
            pass
        
        return False
    
    def get_effective_permissions(self, user_permissions: List[str]) -> List[str]:
        """获取用户的有效权限列表"""
        effective_permissions = set(user_permissions)
        
        for user_perm in user_permissions:
            try:
                user_level = PermissionLevel(user_perm)
                inherited_perms = self.permission_hierarchy.get(user_level, [])
                effective_permissions.update([p.value for p in inherited_perms])
            except ValueError:
                # 不是标准权限级别，保持原样
                continue
        
        return list(effective_permissions)
    
    def can_access_resource(self, user_permissions: List[str], 
                          resource_permissions: List[str]) -> bool:
        """检查用户是否可以访问资源"""
        if not resource_permissions:
            return True  # 无权限要求的资源
        
        return any(
            self.has_permission(user_permissions, req_perm)
            for req_perm in resource_permissions
        )


class RateLimiter:
    """速率限制器"""
    
    def __init__(self):
        self.requests = {}  # {key: [(timestamp, count), ...]}
        self.cleanup_interval = 3600  # 1小时清理一次
        self.last_cleanup = datetime.now()
    
    def is_allowed(self, key: str, limit: int, window: int) -> tuple[bool, Dict[str, Any]]:
        """检查是否允许请求"""
        now = datetime.now()
        
        # 定期清理过期记录
        if (now - self.last_cleanup).total_seconds() > self.cleanup_interval:
            self._cleanup_expired_records()
            self.last_cleanup = now
        
        # 获取或创建请求记录
        if key not in self.requests:
            self.requests[key] = []
        
        requests = self.requests[key]
        
        # 移除过期的请求记录
        cutoff_time = now - timedelta(seconds=window)
        requests[:] = [(timestamp, count) for timestamp, count in requests 
                      if timestamp > cutoff_time]
        
        # 计算当前窗口内的请求数
        current_count = sum(count for _, count in requests)
        
        # 检查是否超过限制
        if current_count >= limit:
            return False, {
                "allowed": False,
                "current_count": current_count,
                "limit": limit,
                "window": window,
                "reset_time": (requests[0][0] + timedelta(seconds=window)).isoformat() if requests else None
            }
        
        # 添加当前请求
        requests.append((now, 1))
        
        return True, {
            "allowed": True,
            "current_count": current_count + 1,
            "limit": limit,
            "window": window,
            "remaining": limit - current_count - 1
        }
    
    def _cleanup_expired_records(self):
        """清理过期的请求记录"""
        now = datetime.now()
        keys_to_remove = []
        
        for key, requests in self.requests.items():
            # 移除1小时前的记录
            cutoff_time = now - timedelta(hours=1)
            requests[:] = [(timestamp, count) for timestamp, count in requests 
                          if timestamp > cutoff_time]
            
            # 如果没有记录了，标记删除
            if not requests:
                keys_to_remove.append(key)
        
        # 删除空的记录
        for key in keys_to_remove:
            del self.requests[key]


class SecurityManager:
    """安全管理器 - 整合所有安全功能"""
    
    def __init__(self, secret_key: str, encryption_key: Optional[bytes] = None):
        self.password_hasher = PasswordHasher()
        self.token_manager = TokenManager(secret_key)
        self.api_key_manager = APIKeyManager(encryption_key)
        self.permission_manager = PermissionManager()
        self.rate_limiter = RateLimiter()
    
    def authenticate_user(self, username: str, password: str, 
                         stored_hash: str) -> bool:
        """用户认证"""
        return self.password_hasher.verify_password(password, stored_hash)
    
    def create_user_session(self, user_id: str, username: str, email: str,
                          permissions: List[str]) -> Dict[str, str]:
        """创建用户会话"""
        session_id = secrets.token_urlsafe(32)
        
        payload = TokenPayload(
            user_id=user_id,
            username=username,
            email=email,
            permissions=permissions,
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow(),
            session_id=session_id
        )
        
        access_token = self.token_manager.create_access_token(payload)
        refresh_token = self.token_manager.create_refresh_token(payload)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "session_id": session_id,
            "token_type": "bearer"
        }
    
    def validate_request(self, token: str, required_permissions: List[str] = None,
                        rate_limit_key: str = None, rate_limit: int = None,
                        rate_window: int = 3600) -> Dict[str, Any]:
        """验证请求"""
        result = {
            "valid": False,
            "user": None,
            "permissions": [],
            "rate_limit": None,
            "errors": []
        }
        
        # 验证令牌
        payload = self.token_manager.verify_token(token)
        if not payload:
            result["errors"].append("Invalid or expired token")
            return result
        
        result["user"] = {
            "user_id": payload.user_id,
            "username": payload.username,
            "email": payload.email,
            "session_id": payload.session_id
        }
        result["permissions"] = payload.permissions
        
        # 检查权限
        if required_permissions:
            if not self.permission_manager.can_access_resource(
                payload.permissions, required_permissions
            ):
                result["errors"].append("Insufficient permissions")
                return result
        
        # 检查速率限制
        if rate_limit_key and rate_limit:
            allowed, rate_info = self.rate_limiter.is_allowed(
                rate_limit_key, rate_limit, rate_window
            )
            result["rate_limit"] = rate_info
            
            if not allowed:
                result["errors"].append("Rate limit exceeded")
                return result
        
        result["valid"] = True
        return result


# 全局安全管理器实例
_security_manager = None


def get_security_manager(secret_key: str = None, 
                        encryption_key: bytes = None) -> SecurityManager:
    """获取全局安全管理器"""
    global _security_manager
    
    if _security_manager is None:
        if not secret_key:
            secret_key = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
        
        _security_manager = SecurityManager(secret_key, encryption_key)
    
    return _security_manager


def generate_encryption_key() -> bytes:
    """生成加密密钥"""
    return Fernet.generate_key()


def derive_key_from_password(password: str, salt: bytes = None) -> bytes:
    """从密码派生加密密钥"""
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key