"""用户仓储模块

提供用户相关的数据访问接口和业务逻辑。
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session, selectinload
from passlib.context import CryptContext

from ..models.database import User, UserStatus
from ..models.api import UserCreate, UserUpdate, UserStats
from ..database.repository import CRUDRepository, EntityNotFoundError, EntityAlreadyExistsError
from ..database.query_builder import QueryFilter
from ..database.session import SessionManager

logger = logging.getLogger(__name__)

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRepository(CRUDRepository[User]):
    """用户仓储类"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        super().__init__(User, session_manager)
        # 设置软删除字段
        self.set_soft_delete_field("deleted_at")
        # 设置默认加载选项
        self.set_default_load_options(
            selectinload(User.sessions),
            selectinload(User.preferences)
        )
    
    def _hash_password(self, password: str) -> str:
        """加密密码"""
        return pwd_context.hash(password)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_user(self, user_create: UserCreate, session: Optional[Session] = None) -> User:
        """创建用户"""
        # 检查用户名是否已存在
        existing_user = self.get_by_username(user_create.username, session)
        if existing_user:
            raise EntityAlreadyExistsError(f"Username '{user_create.username}' already exists")
        
        # 检查邮箱是否已存在
        existing_email = self.get_by_email(user_create.email, session)
        if existing_email:
            raise EntityAlreadyExistsError(f"Email '{user_create.email}' already exists")
        
        # 创建用户数据
        user_data = user_create.model_dump(exclude={"password", "confirm_password"})
        user_data["password_hash"] = self._hash_password(user_create.password)
        user_data["status"] = UserStatus.ACTIVE
        
        return self.create(user_data, session)
    
    def get_by_username(self, username: str, session: Optional[Session] = None) -> Optional[User]:
        """根据用户名获取用户"""
        return self.get_by_field("username", username, session)
    
    def get_by_email(self, email: str, session: Optional[Session] = None) -> Optional[User]:
        """根据邮箱获取用户"""
        return self.get_by_field("email", email, session)
    
    def authenticate(self, username: str, password: str, session: Optional[Session] = None) -> Optional[User]:
        """用户认证"""
        try:
            # 支持用户名或邮箱登录
            user = self.get_by_username(username, session)
            if not user:
                user = self.get_by_email(username, session)
            
            if not user:
                logger.warning(f"Authentication failed: user '{username}' not found")
                return None
            
            # 检查用户状态
            if user.status != UserStatus.ACTIVE:
                logger.warning(f"Authentication failed: user '{username}' is not active")
                return None
            
            # 验证密码
            if not self._verify_password(password, user.password_hash):
                logger.warning(f"Authentication failed: invalid password for user '{username}'")
                # 更新失败登录次数
                self._update_failed_login_attempts(user, session)
                return None
            
            # 更新最后登录时间
            self._update_last_login(user, session)
            
            logger.info(f"User '{username}' authenticated successfully")
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user '{username}': {e}")
            return None
    
    def _update_last_login(self, user: User, session: Optional[Session] = None) -> None:
        """更新最后登录时间"""
        try:
            update_data = {
                "last_login_at": datetime.utcnow(),
                "failed_login_attempts": 0  # 重置失败次数
            }
            self.update(user, update_data, session)
        except Exception as e:
            logger.error(f"Error updating last login for user {user.id}: {e}")
    
    def _update_failed_login_attempts(self, user: User, session: Optional[Session] = None) -> None:
        """更新失败登录次数"""
        try:
            failed_attempts = (user.failed_login_attempts or 0) + 1
            update_data = {"failed_login_attempts": failed_attempts}
            
            # 如果失败次数过多，锁定账户
            if failed_attempts >= 5:
                update_data["status"] = UserStatus.SUSPENDED
                update_data["suspended_at"] = datetime.utcnow()
                logger.warning(f"User {user.id} suspended due to too many failed login attempts")
            
            self.update(user, update_data, session)
        except Exception as e:
            logger.error(f"Error updating failed login attempts for user {user.id}: {e}")
    
    def change_password(
        self, 
        user_id: int, 
        old_password: str, 
        new_password: str,
        session: Optional[Session] = None
    ) -> bool:
        """修改密码"""
        try:
            user = self.get(user_id, session)
            if not user:
                raise EntityNotFoundError(f"User with ID {user_id} not found")
            
            # 验证旧密码
            if not self._verify_password(old_password, user.password_hash):
                logger.warning(f"Password change failed: invalid old password for user {user_id}")
                return False
            
            # 更新密码
            update_data = {
                "password_hash": self._hash_password(new_password),
                "password_changed_at": datetime.utcnow()
            }
            self.update(user, update_data, session)
            
            logger.info(f"Password changed successfully for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error changing password for user {user_id}: {e}")
            return False
    
    def reset_password(self, email: str, new_password: str, session: Optional[Session] = None) -> bool:
        """重置密码"""
        try:
            user = self.get_by_email(email, session)
            if not user:
                raise EntityNotFoundError(f"User with email '{email}' not found")
            
            # 更新密码
            update_data = {
                "password_hash": self._hash_password(new_password),
                "password_changed_at": datetime.utcnow(),
                "failed_login_attempts": 0  # 重置失败次数
            }
            
            # 如果用户被暂停，恢复状态
            if user.status == UserStatus.SUSPENDED:
                update_data["status"] = UserStatus.ACTIVE
                update_data["suspended_at"] = None
            
            self.update(user, update_data, session)
            
            logger.info(f"Password reset successfully for user {user.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting password for email '{email}': {e}")
            return False
    
    def update_profile(
        self, 
        user_id: int, 
        user_update: UserUpdate,
        session: Optional[Session] = None
    ) -> Optional[User]:
        """更新用户资料"""
        try:
            user = self.get(user_id, session)
            if not user:
                raise EntityNotFoundError(f"User with ID {user_id} not found")
            
            # 检查用户名是否已被其他用户使用
            if user_update.username and user_update.username != user.username:
                existing_user = self.get_by_username(user_update.username, session)
                if existing_user and existing_user.id != user_id:
                    raise EntityAlreadyExistsError(f"Username '{user_update.username}' already exists")
            
            # 检查邮箱是否已被其他用户使用
            if user_update.email and user_update.email != user.email:
                existing_email = self.get_by_email(user_update.email, session)
                if existing_email and existing_email.id != user_id:
                    raise EntityAlreadyExistsError(f"Email '{user_update.email}' already exists")
            
            return self.update(user, user_update, session)
            
        except Exception as e:
            logger.error(f"Error updating profile for user {user_id}: {e}")
            raise
    
    def suspend_user(self, user_id: int, reason: str = None, session: Optional[Session] = None) -> bool:
        """暂停用户"""
        try:
            user = self.get(user_id, session)
            if not user:
                raise EntityNotFoundError(f"User with ID {user_id} not found")
            
            update_data = {
                "status": UserStatus.SUSPENDED,
                "suspended_at": datetime.utcnow()
            }
            
            if reason:
                update_data["suspension_reason"] = reason
            
            self.update(user, update_data, session)
            
            logger.info(f"User {user_id} suspended. Reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error suspending user {user_id}: {e}")
            return False
    
    def activate_user(self, user_id: int, session: Optional[Session] = None) -> bool:
        """激活用户"""
        try:
            user = self.get(user_id, session)
            if not user:
                raise EntityNotFoundError(f"User with ID {user_id} not found")
            
            update_data = {
                "status": UserStatus.ACTIVE,
                "suspended_at": None,
                "suspension_reason": None,
                "failed_login_attempts": 0
            }
            
            self.update(user, update_data, session)
            
            logger.info(f"User {user_id} activated")
            return True
            
        except Exception as e:
            logger.error(f"Error activating user {user_id}: {e}")
            return False
    
    def get_active_users(
        self, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[User]:
        """获取活跃用户列表"""
        filters = QueryFilter().eq("status", UserStatus.ACTIVE)
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def search_users(
        self, 
        query: str, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[User]:
        """搜索用户"""
        filters = QueryFilter().or_(
            QueryFilter().ilike("username", f"%{query}%"),
            QueryFilter().ilike("email", f"%{query}%"),
            QueryFilter().ilike("full_name", f"%{query}%")
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_users_by_status(
        self, 
        status: UserStatus, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[User]:
        """根据状态获取用户"""
        filters = QueryFilter().eq("status", status)
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("created_at", "desc")],
            session=session
        )
    
    def get_recently_active_users(
        self, 
        days: int = 7, 
        skip: int = 0, 
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[User]:
        """获取最近活跃的用户"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        filters = QueryFilter().and_(
            QueryFilter().eq("status", UserStatus.ACTIVE),
            QueryFilter().gte("last_login_at", cutoff_date)
        )
        
        return self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            order_by=[("last_login_at", "desc")],
            session=session
        )
    
    def get_user_stats(self, session: Optional[Session] = None) -> UserStats:
        """获取用户统计信息"""
        try:
            if session:
                db_session = session
            else:
                db_session = next(self.session_manager.get_session())
            
            # 总用户数
            total_users = self.count(session=db_session)
            
            # 活跃用户数
            active_filters = QueryFilter().eq("status", UserStatus.ACTIVE)
            active_users = self.count(filters=active_filters, session=db_session)
            
            # 暂停用户数
            suspended_filters = QueryFilter().eq("status", UserStatus.SUSPENDED)
            suspended_users = self.count(filters=suspended_filters, session=db_session)
            
            # 最近7天活跃用户数
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            recent_active_filters = QueryFilter().and_(
                QueryFilter().eq("status", UserStatus.ACTIVE),
                QueryFilter().gte("last_login_at", cutoff_date)
            )
            recent_active_users = self.count(filters=recent_active_filters, session=db_session)
            
            # 今日新注册用户数
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_new_filters = QueryFilter().gte("created_at", today_start)
            today_new_users = self.count(filters=today_new_filters, session=db_session)
            
            return UserStats(
                total_users=total_users,
                active_users=active_users,
                suspended_users=suspended_users,
                recent_active_users=recent_active_users,
                today_new_users=today_new_users
            )
            
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return UserStats(
                total_users=0,
                active_users=0,
                suspended_users=0,
                recent_active_users=0,
                today_new_users=0
            )
        finally:
            if not session:
                db_session.close()
    
    def cleanup_inactive_users(self, days: int = 365, session: Optional[Session] = None) -> int:
        """清理长期不活跃的用户"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # 查找长期不活跃的用户
            filters = QueryFilter().and_(
                QueryFilter().eq("status", UserStatus.ACTIVE),
                QueryFilter().or_(
                    QueryFilter().is_null("last_login_at"),
                    QueryFilter().lt("last_login_at", cutoff_date)
                ),
                QueryFilter().lt("created_at", cutoff_date)
            )
            
            inactive_users = self.get_multi(
                filters=filters,
                limit=1000,  # 批量处理
                session=session
            )
            
            # 软删除这些用户
            deleted_count = 0
            for user in inactive_users:
                if self.soft_delete(user.id, session):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} inactive users")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up inactive users: {e}")
            return 0