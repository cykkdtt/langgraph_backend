"""模型安全系统模块

本模块提供数据加密、访问控制和安全审计功能。
"""

from typing import (
    Dict, Any, Optional, List, Union, Callable, Type, 
    TypeVar, Generic, Tuple, Set
)
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps
import hashlib
import hmac
import secrets
import base64
import json
import logging
import threading
from contextlib import contextmanager
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from sqlalchemy.orm import Session
from sqlalchemy import event

# 导入项目模型
from .models import (
    User, Session as ChatSession, Thread, Message, Workflow, 
    WorkflowExecution, WorkflowStep, Memory, MemoryVector, 
    TimeTravel, Attachment, UserPreference, SystemConfig
)
from .events import EventType, emit_business_event


logger = logging.getLogger(__name__)


T = TypeVar('T')


class SecurityLevel(Enum):
    """安全级别枚举"""
    PUBLIC = "public"              # 公开
    INTERNAL = "internal"          # 内部
    CONFIDENTIAL = "confidential"  # 机密
    SECRET = "secret"              # 秘密
    TOP_SECRET = "top_secret"      # 绝密


class PermissionType(Enum):
    """权限类型枚举"""
    READ = "read"                  # 读取
    WRITE = "write"                # 写入
    UPDATE = "update"              # 更新
    DELETE = "delete"              # 删除
    EXECUTE = "execute"            # 执行
    ADMIN = "admin"                # 管理


class EncryptionAlgorithm(Enum):
    """加密算法枚举"""
    AES_256_GCM = "aes_256_gcm"    # AES-256-GCM
    AES_256_CBC = "aes_256_cbc"    # AES-256-CBC
    RSA_2048 = "rsa_2048"          # RSA-2048
    RSA_4096 = "rsa_4096"          # RSA-4096
    FERNET = "fernet"              # Fernet
    CHACHA20 = "chacha20"          # ChaCha20


class AuditAction(Enum):
    """审计动作枚举"""
    LOGIN = "login"                # 登录
    LOGOUT = "logout"              # 登出
    ACCESS = "access"              # 访问
    CREATE = "create"              # 创建
    READ = "read"                  # 读取
    UPDATE = "update"              # 更新
    DELETE = "delete"              # 删除
    PERMISSION_CHANGE = "permission_change"  # 权限变更
    SECURITY_VIOLATION = "security_violation"  # 安全违规


@dataclass
class SecurityContext:
    """安全上下文"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    authenticated: bool = False
    mfa_verified: bool = False
    session_expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: str) -> bool:
        """检查是否有权限"""
        return permission in self.permissions or "admin" in self.roles
    
    def has_role(self, role: str) -> bool:
        """检查是否有角色"""
        return role in self.roles
    
    def is_session_valid(self) -> bool:
        """检查会话是否有效"""
        if not self.authenticated:
            return False
        if self.session_expires_at and datetime.now() > self.session_expires_at:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'user_id': self.user_id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'permissions': list(self.permissions),
            'roles': list(self.roles),
            'security_level': self.security_level.value,
            'authenticated': self.authenticated,
            'mfa_verified': self.mfa_verified,
            'session_expires_at': self.session_expires_at.isoformat() if self.session_expires_at else None,
            'metadata': self.metadata
        }


@dataclass
class AuditLog:
    """审计日志"""
    id: str
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    action: AuditAction
    resource_type: str
    resource_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    success: bool
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'action': self.action.value,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'success': self.success,
            'error_message': self.error_message,
            'details': self.details,
            'security_level': self.security_level.value
        }


class EncryptionProvider(ABC):
    """加密提供者基类"""
    
    @abstractmethod
    def encrypt(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """加密数据"""
        pass
    
    @abstractmethod
    def decrypt(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """解密数据"""
        pass
    
    @abstractmethod
    def generate_key(self) -> bytes:
        """生成密钥"""
        pass


class FernetEncryption(EncryptionProvider):
    """Fernet加密"""
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or self.generate_key()
        self.fernet = Fernet(self.key)
    
    def encrypt(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """加密数据"""
        if key:
            fernet = Fernet(key)
            return fernet.encrypt(data)
        return self.fernet.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """解密数据"""
        if key:
            fernet = Fernet(key)
            return fernet.decrypt(encrypted_data)
        return self.fernet.decrypt(encrypted_data)
    
    def generate_key(self) -> bytes:
        """生成密钥"""
        return Fernet.generate_key()


class AESEncryption(EncryptionProvider):
    """AES加密"""
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or self.generate_key()
    
    def encrypt(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """加密数据"""
        encryption_key = key or self.key
        
        # 生成随机IV
        iv = secrets.token_bytes(16)
        
        # 创建加密器
        cipher = Cipher(algorithms.AES(encryption_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # 填充数据
        padded_data = self._pad_data(data)
        
        # 加密
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # 返回IV + 加密数据
        return iv + encrypted_data
    
    def decrypt(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """解密数据"""
        encryption_key = key or self.key
        
        # 提取IV和加密数据
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # 创建解密器
        cipher = Cipher(algorithms.AES(encryption_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        # 解密
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # 去除填充
        return self._unpad_data(padded_data)
    
    def generate_key(self) -> bytes:
        """生成密钥"""
        return secrets.token_bytes(32)  # 256位密钥
    
    def _pad_data(self, data: bytes) -> bytes:
        """PKCS7填充"""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """去除PKCS7填充"""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]


class RSAEncryption(EncryptionProvider):
    """RSA加密"""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.private_key = self._generate_private_key()
        self.public_key = self.private_key.public_key()
    
    def encrypt(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """加密数据（使用公钥）"""
        if key:
            # 从字节加载公钥
            public_key = serialization.load_pem_public_key(key)
        else:
            public_key = self.public_key
        
        # RSA加密有长度限制，需要分块加密
        max_chunk_size = (self.key_size // 8) - 2 * (hashes.SHA256().digest_size) - 2
        
        encrypted_chunks = []
        for i in range(0, len(data), max_chunk_size):
            chunk = data[i:i + max_chunk_size]
            encrypted_chunk = public_key.encrypt(
                chunk,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            encrypted_chunks.append(encrypted_chunk)
        
        return b''.join(encrypted_chunks)
    
    def decrypt(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """解密数据（使用私钥）"""
        if key:
            # 从字节加载私钥
            private_key = serialization.load_pem_private_key(key, password=None)
        else:
            private_key = self.private_key
        
        # 分块解密
        chunk_size = self.key_size // 8
        decrypted_chunks = []
        
        for i in range(0, len(encrypted_data), chunk_size):
            chunk = encrypted_data[i:i + chunk_size]
            decrypted_chunk = private_key.decrypt(
                chunk,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            decrypted_chunks.append(decrypted_chunk)
        
        return b''.join(decrypted_chunks)
    
    def generate_key(self) -> bytes:
        """生成密钥对（返回私钥）"""
        private_key = self._generate_private_key()
        return private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
    
    def get_public_key_bytes(self) -> bytes:
        """获取公钥字节"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def _generate_private_key(self):
        """生成私钥"""
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )


class PasswordHasher:
    """密码哈希器"""
    
    def __init__(self, algorithm: str = "pbkdf2_sha256", iterations: int = 100000):
        self.algorithm = algorithm
        self.iterations = iterations
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> str:
        """哈希密码"""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        if self.algorithm == "pbkdf2_sha256":
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.iterations,
            )
            key = kdf.derive(password.encode('utf-8'))
            
            # 返回算法:迭代次数:盐:哈希值
            return f"{self.algorithm}:{self.iterations}:{base64.b64encode(salt).decode()}:{base64.b64encode(key).decode()}"
        
        elif self.algorithm == "sha256":
            # 简单SHA256（不推荐用于生产）
            hash_obj = hashlib.sha256()
            hash_obj.update(salt + password.encode('utf-8'))
            key = hash_obj.digest()
            
            return f"{self.algorithm}:1:{base64.b64encode(salt).decode()}:{base64.b64encode(key).decode()}"
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """验证密码"""
        try:
            parts = hashed_password.split(':')
            if len(parts) != 4:
                return False
            
            algorithm, iterations_str, salt_b64, hash_b64 = parts
            iterations = int(iterations_str)
            salt = base64.b64decode(salt_b64)
            expected_hash = base64.b64decode(hash_b64)
            
            if algorithm == "pbkdf2_sha256":
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=iterations,
                )
                key = kdf.derive(password.encode('utf-8'))
                return hmac.compare_digest(key, expected_hash)
            
            elif algorithm == "sha256":
                hash_obj = hashlib.sha256()
                hash_obj.update(salt + password.encode('utf-8'))
                key = hash_obj.digest()
                return hmac.compare_digest(key, expected_hash)
            
            else:
                return False
        
        except Exception:
            return False


class JWTManager:
    """JWT管理器"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def generate_token(self, payload: Dict[str, Any], 
                      expires_in: int = 3600) -> str:
        """生成JWT令牌"""
        now = datetime.utcnow()
        payload.update({
            'iat': now,
            'exp': now + timedelta(seconds=expires_in),
            'jti': secrets.token_urlsafe(16)  # JWT ID
        })
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def refresh_token(self, token: str, expires_in: int = 3600) -> Optional[str]:
        """刷新JWT令牌"""
        payload = self.verify_token(token)
        if not payload:
            return None
        
        # 移除时间相关字段
        payload.pop('iat', None)
        payload.pop('exp', None)
        payload.pop('jti', None)
        
        return self.generate_token(payload, expires_in)


class AccessControlManager:
    """访问控制管理器"""
    
    def __init__(self):
        self._permissions: Dict[str, Set[str]] = defaultdict(set)
        self._roles: Dict[str, Set[str]] = defaultdict(set)
        self._user_roles: Dict[str, Set[str]] = defaultdict(set)
        self._resource_permissions: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._lock = threading.RLock()
    
    def add_permission(self, role: str, permission: str) -> None:
        """添加角色权限"""
        with self._lock:
            self._permissions[role].add(permission)
    
    def remove_permission(self, role: str, permission: str) -> None:
        """移除角色权限"""
        with self._lock:
            self._permissions[role].discard(permission)
    
    def add_role(self, user_id: str, role: str) -> None:
        """添加用户角色"""
        with self._lock:
            self._user_roles[user_id].add(role)
    
    def remove_role(self, user_id: str, role: str) -> None:
        """移除用户角色"""
        with self._lock:
            self._user_roles[user_id].discard(role)
    
    def grant_resource_permission(self, user_id: str, resource_type: str, 
                                 resource_id: str, permission: str) -> None:
        """授予资源权限"""
        with self._lock:
            resource_key = f"{resource_type}:{resource_id}"
            self._resource_permissions[user_id][resource_key].add(permission)
    
    def revoke_resource_permission(self, user_id: str, resource_type: str, 
                                  resource_id: str, permission: str) -> None:
        """撤销资源权限"""
        with self._lock:
            resource_key = f"{resource_type}:{resource_id}"
            self._resource_permissions[user_id][resource_key].discard(permission)
    
    def check_permission(self, user_id: str, permission: str, 
                        resource_type: Optional[str] = None, 
                        resource_id: Optional[str] = None) -> bool:
        """检查权限"""
        with self._lock:
            # 检查角色权限
            user_roles = self._user_roles.get(user_id, set())
            for role in user_roles:
                if permission in self._permissions.get(role, set()):
                    return True
                # 检查管理员权限
                if "admin" in self._permissions.get(role, set()):
                    return True
            
            # 检查资源特定权限
            if resource_type and resource_id:
                resource_key = f"{resource_type}:{resource_id}"
                user_resource_permissions = self._resource_permissions.get(user_id, {})
                if permission in user_resource_permissions.get(resource_key, set()):
                    return True
            
            return False
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """获取用户所有权限"""
        with self._lock:
            permissions = set()
            
            # 从角色获取权限
            user_roles = self._user_roles.get(user_id, set())
            for role in user_roles:
                permissions.update(self._permissions.get(role, set()))
            
            # 从资源权限获取
            user_resource_permissions = self._resource_permissions.get(user_id, {})
            for resource_permissions in user_resource_permissions.values():
                permissions.update(resource_permissions)
            
            return permissions
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """获取用户角色"""
        return self._user_roles.get(user_id, set()).copy()


class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self):
        self._logs: List[AuditLog] = []
        self._lock = threading.RLock()
        self._handlers: List[Callable[[AuditLog], None]] = []
    
    def add_handler(self, handler: Callable[[AuditLog], None]) -> None:
        """添加日志处理器"""
        self._handlers.append(handler)
    
    def log(self, action: AuditAction, resource_type: str, 
           context: Optional[SecurityContext] = None,
           resource_id: Optional[str] = None,
           success: bool = True,
           error_message: Optional[str] = None,
           details: Optional[Dict[str, Any]] = None) -> None:
        """记录审计日志"""
        audit_log = AuditLog(
            id=secrets.token_urlsafe(16),
            timestamp=datetime.now(),
            user_id=context.user_id if context else None,
            session_id=context.session_id if context else None,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=context.ip_address if context else None,
            user_agent=context.user_agent if context else None,
            success=success,
            error_message=error_message,
            details=details or {},
            security_level=context.security_level if context else SecurityLevel.INTERNAL
        )
        
        with self._lock:
            self._logs.append(audit_log)
        
        # 调用处理器
        for handler in self._handlers:
            try:
                handler(audit_log)
            except Exception as e:
                logger.error(f"Audit handler error: {e}")
        
        # 发布事件
        emit_business_event(
            EventType.SYSTEM_STARTUP,  # 使用系统事件类型
            "security_audit",
            data=audit_log.to_dict()
        )
    
    def get_logs(self, user_id: Optional[str] = None,
                action: Optional[AuditAction] = None,
                resource_type: Optional[str] = None,
                start_time: Optional[datetime] = None,
                end_time: Optional[datetime] = None,
                limit: int = 100) -> List[AuditLog]:
        """获取审计日志"""
        with self._lock:
            filtered_logs = self._logs.copy()
        
        # 应用过滤器
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
        
        if action:
            filtered_logs = [log for log in filtered_logs if log.action == action]
        
        if resource_type:
            filtered_logs = [log for log in filtered_logs if log.resource_type == resource_type]
        
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
        
        # 按时间倒序排序并限制数量
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_logs[:limit]


class SecurityManager:
    """安全管理器"""
    
    def __init__(self, secret_key: str):
        self.encryption_providers: Dict[EncryptionAlgorithm, EncryptionProvider] = {
            EncryptionAlgorithm.FERNET: FernetEncryption(),
            EncryptionAlgorithm.AES_256_CBC: AESEncryption(),
            EncryptionAlgorithm.RSA_2048: RSAEncryption(2048),
            EncryptionAlgorithm.RSA_4096: RSAEncryption(4096)
        }
        
        self.password_hasher = PasswordHasher()
        self.jwt_manager = JWTManager(secret_key)
        self.access_control = AccessControlManager()
        self.audit_logger = AuditLogger()
        
        self._current_context = threading.local()
    
    def encrypt_data(self, data: Union[str, bytes], 
                    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET,
                    key: Optional[bytes] = None) -> bytes:
        """加密数据"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        provider = self.encryption_providers.get(algorithm)
        if not provider:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
        
        return provider.encrypt(data, key)
    
    def decrypt_data(self, encrypted_data: bytes, 
                    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET,
                    key: Optional[bytes] = None) -> bytes:
        """解密数据"""
        provider = self.encryption_providers.get(algorithm)
        if not provider:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
        
        return provider.decrypt(encrypted_data, key)
    
    def hash_password(self, password: str) -> str:
        """哈希密码"""
        return self.password_hasher.hash_password(password)
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """验证密码"""
        return self.password_hasher.verify_password(password, hashed_password)
    
    def generate_token(self, user_id: str, permissions: List[str], 
                      roles: List[str], expires_in: int = 3600) -> str:
        """生成访问令牌"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'roles': roles
        }
        return self.jwt_manager.generate_token(payload, expires_in)
    
    def verify_token(self, token: str) -> Optional[SecurityContext]:
        """验证访问令牌"""
        payload = self.jwt_manager.verify_token(token)
        if not payload:
            return None
        
        context = SecurityContext(
            user_id=payload.get('user_id'),
            permissions=set(payload.get('permissions', [])),
            roles=set(payload.get('roles', [])),
            authenticated=True,
            session_expires_at=datetime.fromtimestamp(payload.get('exp', 0))
        )
        
        return context
    
    def set_current_context(self, context: SecurityContext) -> None:
        """设置当前安全上下文"""
        self._current_context.value = context
    
    def get_current_context(self) -> Optional[SecurityContext]:
        """获取当前安全上下文"""
        return getattr(self._current_context, 'value', None)
    
    @contextmanager
    def security_context(self, context: SecurityContext):
        """安全上下文管理器"""
        old_context = self.get_current_context()
        self.set_current_context(context)
        try:
            yield context
        finally:
            self.set_current_context(old_context)
    
    def check_permission(self, permission: str, resource_type: Optional[str] = None,
                        resource_id: Optional[str] = None) -> bool:
        """检查当前用户权限"""
        context = self.get_current_context()
        if not context or not context.authenticated:
            return False
        
        # 检查会话有效性
        if not context.is_session_valid():
            return False
        
        # 检查权限
        if context.has_permission(permission):
            return True
        
        # 检查访问控制
        return self.access_control.check_permission(
            context.user_id, permission, resource_type, resource_id
        )
    
    def audit_log(self, action: AuditAction, resource_type: str,
                 resource_id: Optional[str] = None,
                 success: bool = True,
                 error_message: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        """记录审计日志"""
        context = self.get_current_context()
        self.audit_logger.log(
            action=action,
            resource_type=resource_type,
            context=context,
            resource_id=resource_id,
            success=success,
            error_message=error_message,
            details=details
        )


# 安全装饰器
def require_permission(permission: str, resource_type: Optional[str] = None):
    """权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 尝试从参数中获取资源ID
            resource_id = kwargs.get('id') or kwargs.get('resource_id')
            
            if not security_manager.check_permission(permission, resource_type, resource_id):
                security_manager.audit_log(
                    AuditAction.SECURITY_VIOLATION,
                    resource_type or "unknown",
                    resource_id,
                    success=False,
                    error_message=f"Permission denied: {permission}"
                )
                raise PermissionError(f"Permission denied: {permission}")
            
            # 记录访问日志
            security_manager.audit_log(
                AuditAction.ACCESS,
                resource_type or "function",
                resource_id or func.__name__,
                success=True
            )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_authentication(func):
    """认证检查装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        context = security_manager.get_current_context()
        if not context or not context.authenticated or not context.is_session_valid():
            security_manager.audit_log(
                AuditAction.SECURITY_VIOLATION,
                "authentication",
                success=False,
                error_message="Authentication required"
            )
            raise PermissionError("Authentication required")
        
        return func(*args, **kwargs)
    
    return wrapper


def encrypt_field(algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET):
    """字段加密装饰器"""
    def decorator(cls):
        original_setattr = cls.__setattr__
        original_getattribute = cls.__getattribute__
        
        def encrypted_setattr(self, name, value):
            if hasattr(cls, '_encrypted_fields') and name in cls._encrypted_fields:
                if isinstance(value, str):
                    value = security_manager.encrypt_data(value, algorithm)
            original_setattr(self, name, value)
        
        def encrypted_getattribute(self, name):
            value = original_getattribute(self, name)
            if hasattr(cls, '_encrypted_fields') and name in cls._encrypted_fields:
                if isinstance(value, bytes):
                    try:
                        value = security_manager.decrypt_data(value, algorithm).decode('utf-8')
                    except Exception:
                        pass  # 如果解密失败，返回原值
            return value
        
        cls.__setattr__ = encrypted_setattr
        cls.__getattribute__ = encrypted_getattribute
        
        return cls
    
    return decorator


# 全局安全管理器（需要在应用启动时初始化）
security_manager: Optional[SecurityManager] = None


def initialize_security(secret_key: str) -> SecurityManager:
    """初始化安全管理器"""
    global security_manager
    security_manager = SecurityManager(secret_key)
    return security_manager


# 便捷函数
def get_current_user() -> Optional[str]:
    """获取当前用户ID"""
    if security_manager:
        context = security_manager.get_current_context()
        return context.user_id if context else None
    return None


def is_authenticated() -> bool:
    """检查是否已认证"""
    if security_manager:
        context = security_manager.get_current_context()
        return context.authenticated if context else False
    return False


def has_permission(permission: str) -> bool:
    """检查是否有权限"""
    if security_manager:
        return security_manager.check_permission(permission)
    return False


# 导出所有类和函数
__all__ = [
    "SecurityLevel",
    "PermissionType",
    "EncryptionAlgorithm",
    "AuditAction",
    "SecurityContext",
    "AuditLog",
    "EncryptionProvider",
    "FernetEncryption",
    "AESEncryption",
    "RSAEncryption",
    "PasswordHasher",
    "JWTManager",
    "AccessControlManager",
    "AuditLogger",
    "SecurityManager",
    "require_permission",
    "require_authentication",
    "encrypt_field",
    "initialize_security",
    "get_current_user",
    "is_authenticated",
    "has_permission"
]