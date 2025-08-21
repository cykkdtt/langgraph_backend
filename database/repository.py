"""数据库仓储模式模块

提供统一的数据访问接口、CRUD操作和查询优化。
"""

import logging
from typing import (
    Type, TypeVar, Generic, Optional, List, Dict, Any, Union,
    Callable, Sequence, Set
)
from abc import ABC, abstractmethod
from datetime import datetime

from sqlalchemy import select, insert, update, delete, func, and_, or_
from sqlalchemy.orm import Session, selectinload, joinedload
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.sql import Select

from ..models.database import Base
from ..models.api import PaginationParams, SortOrder, FilterParams
from .query_builder import QueryBuilder, QueryFilter, create_query_builder
from .session import get_session_manager, SessionManager

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T', bound=Base)
CreateSchemaType = TypeVar('CreateSchemaType')
UpdateSchemaType = TypeVar('UpdateSchemaType')


class RepositoryError(Exception):
    """仓储操作异常"""
    pass


class EntityNotFoundError(RepositoryError):
    """实体未找到异常"""
    pass


class EntityAlreadyExistsError(RepositoryError):
    """实体已存在异常"""
    pass


class ValidationError(RepositoryError):
    """数据验证异常"""
    pass


class BaseRepository(Generic[T], ABC):
    """基础仓储抽象类"""
    
    def __init__(
        self, 
        model: Type[T], 
        session_manager: Optional[SessionManager] = None
    ):
        self.model = model
        self.session_manager = session_manager or get_session_manager()
    
    @abstractmethod
    def create(self, obj_in: CreateSchemaType, **kwargs) -> T:
        """创建实体"""
        pass
    
    @abstractmethod
    def get(self, id: Any, **kwargs) -> Optional[T]:
        """根据ID获取实体"""
        pass
    
    @abstractmethod
    def get_multi(
        self, 
        skip: int = 0, 
        limit: int = 100, 
        **kwargs
    ) -> List[T]:
        """获取多个实体"""
        pass
    
    @abstractmethod
    def update(self, db_obj: T, obj_in: UpdateSchemaType, **kwargs) -> T:
        """更新实体"""
        pass
    
    @abstractmethod
    def delete(self, id: Any, **kwargs) -> bool:
        """删除实体"""
        pass


class CRUDRepository(BaseRepository[T]):
    """CRUD仓储实现类"""
    
    def __init__(
        self, 
        model: Type[T], 
        session_manager: Optional[SessionManager] = None
    ):
        super().__init__(model, session_manager)
        self._default_load_options: List[Any] = []
        self._soft_delete_field: Optional[str] = None
        self._created_at_field: str = "created_at"
        self._updated_at_field: str = "updated_at"
    
    def set_default_load_options(self, *options) -> 'CRUDRepository[T]':
        """设置默认加载选项"""
        self._default_load_options = list(options)
        return self
    
    def set_soft_delete_field(self, field_name: str) -> 'CRUDRepository[T]':
        """设置软删除字段"""
        self._soft_delete_field = field_name
        return self
    
    def set_timestamp_fields(
        self, 
        created_at: str = "created_at", 
        updated_at: str = "updated_at"
    ) -> 'CRUDRepository[T]':
        """设置时间戳字段"""
        self._created_at_field = created_at
        self._updated_at_field = updated_at
        return self
    
    def _apply_soft_delete_filter(self, query_builder: QueryBuilder[T]) -> QueryBuilder[T]:
        """应用软删除过滤"""
        if self._soft_delete_field and hasattr(self.model, self._soft_delete_field):
            filter_obj = QueryFilter().is_null(self._soft_delete_field)
            query_builder.filter(filter_obj)
        return query_builder
    
    def _set_timestamps(self, obj_data: Dict[str, Any], is_update: bool = False) -> Dict[str, Any]:
        """设置时间戳"""
        now = datetime.utcnow()
        
        if not is_update and self._created_at_field:
            obj_data[self._created_at_field] = now
        
        if self._updated_at_field:
            obj_data[self._updated_at_field] = now
        
        return obj_data
    
    def create(self, obj_in: Union[Dict[str, Any], Any], session: Optional[Session] = None) -> T:
        """创建实体"""
        try:
            # 转换输入数据
            if hasattr(obj_in, 'dict'):
                obj_data = obj_in.dict(exclude_unset=True)
            elif hasattr(obj_in, 'model_dump'):
                obj_data = obj_in.model_dump(exclude_unset=True)
            elif isinstance(obj_in, dict):
                obj_data = obj_in.copy()
            else:
                raise ValidationError(f"Invalid input type: {type(obj_in)}")
            
            # 设置时间戳
            obj_data = self._set_timestamps(obj_data)
            
            # 创建实体
            db_obj = self.model(**obj_data)
            
            # 保存到数据库
            if session:
                session.add(db_obj)
                session.flush()
                session.refresh(db_obj)
            else:
                with self.session_manager.session_scope() as db_session:
                    db_session.add(db_obj)
                    db_session.flush()
                    db_session.refresh(db_obj)
            
            logger.info(f"Created {self.model.__name__} with ID: {db_obj.id}")
            return db_obj
            
        except IntegrityError as e:
            logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            raise EntityAlreadyExistsError(f"Entity already exists: {e}")
        except Exception as e:
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise RepositoryError(f"Failed to create entity: {e}")
    
    def get(
        self, 
        id: Any, 
        session: Optional[Session] = None,
        load_options: Optional[List[Any]] = None
    ) -> Optional[T]:
        """根据ID获取实体"""
        try:
            query_builder = create_query_builder(self.model, session)
            query_builder.filter_by(id=id)
            
            # 应用软删除过滤
            query_builder = self._apply_soft_delete_filter(query_builder)
            
            # 应用加载选项
            options = load_options or self._default_load_options
            if options:
                query_builder.options(*options)
            
            if session:
                query_builder.session = session
                return query_builder.first()
            else:
                with self.session_manager.session_scope() as db_session:
                    query_builder.session = db_session
                    return query_builder.first()
                    
        except Exception as e:
            logger.error(f"Error getting {self.model.__name__} with ID {id}: {e}")
            return None
    
    def get_by_field(
        self, 
        field_name: str, 
        value: Any, 
        session: Optional[Session] = None,
        load_options: Optional[List[Any]] = None
    ) -> Optional[T]:
        """根据字段值获取实体"""
        try:
            query_builder = create_query_builder(self.model, session)
            filter_obj = QueryFilter().eq(field_name, value)
            query_builder.filter(filter_obj)
            
            # 应用软删除过滤
            query_builder = self._apply_soft_delete_filter(query_builder)
            
            # 应用加载选项
            options = load_options or self._default_load_options
            if options:
                query_builder.options(*options)
            
            if session:
                query_builder.session = session
                return query_builder.first()
            else:
                with self.session_manager.session_scope() as db_session:
                    query_builder.session = db_session
                    return query_builder.first()
                    
        except Exception as e:
            logger.error(f"Error getting {self.model.__name__} by {field_name}={value}: {e}")
            return None
    
    def get_multi(
        self, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[QueryFilter] = None,
        order_by: Optional[List[tuple]] = None,
        session: Optional[Session] = None,
        load_options: Optional[List[Any]] = None
    ) -> List[T]:
        """获取多个实体"""
        try:
            query_builder = create_query_builder(self.model, session)
            
            # 应用软删除过滤
            query_builder = self._apply_soft_delete_filter(query_builder)
            
            # 应用自定义过滤
            if filters:
                query_builder.filter(filters)
            
            # 应用排序
            if order_by:
                for field, order in order_by:
                    query_builder.order_by(field, order)
            
            # 应用分页
            query_builder.offset(skip).limit(limit)
            
            # 应用加载选项
            options = load_options or self._default_load_options
            if options:
                query_builder.options(*options)
            
            if session:
                query_builder.session = session
                return query_builder.all()
            else:
                with self.session_manager.session_scope() as db_session:
                    query_builder.session = db_session
                    return query_builder.all()
                    
        except Exception as e:
            logger.error(f"Error getting multiple {self.model.__name__}: {e}")
            return []
    
    def count(
        self, 
        filters: Optional[QueryFilter] = None,
        session: Optional[Session] = None
    ) -> int:
        """获取实体数量"""
        try:
            query_builder = create_query_builder(self.model, session)
            
            # 应用软删除过滤
            query_builder = self._apply_soft_delete_filter(query_builder)
            
            # 应用自定义过滤
            if filters:
                query_builder.filter(filters)
            
            if session:
                query_builder.session = session
                return query_builder.count()
            else:
                with self.session_manager.session_scope() as db_session:
                    query_builder.session = db_session
                    return query_builder.count()
                    
        except Exception as e:
            logger.error(f"Error counting {self.model.__name__}: {e}")
            return 0
    
    def update(
        self, 
        db_obj: T, 
        obj_in: Union[Dict[str, Any], Any],
        session: Optional[Session] = None
    ) -> T:
        """更新实体"""
        try:
            # 转换输入数据
            if hasattr(obj_in, 'dict'):
                obj_data = obj_in.dict(exclude_unset=True)
            elif hasattr(obj_in, 'model_dump'):
                obj_data = obj_in.model_dump(exclude_unset=True)
            elif isinstance(obj_in, dict):
                obj_data = obj_in.copy()
            else:
                raise ValidationError(f"Invalid input type: {type(obj_in)}")
            
            # 设置时间戳
            obj_data = self._set_timestamps(obj_data, is_update=True)
            
            # 更新字段
            for field, value in obj_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            
            # 保存到数据库
            if session:
                session.flush()
                session.refresh(db_obj)
            else:
                with self.session_manager.session_scope() as db_session:
                    db_session.merge(db_obj)
                    db_session.flush()
                    db_session.refresh(db_obj)
            
            logger.info(f"Updated {self.model.__name__} with ID: {db_obj.id}")
            return db_obj
            
        except Exception as e:
            logger.error(f"Error updating {self.model.__name__}: {e}")
            raise RepositoryError(f"Failed to update entity: {e}")
    
    def update_by_id(
        self, 
        id: Any, 
        obj_in: Union[Dict[str, Any], Any],
        session: Optional[Session] = None
    ) -> Optional[T]:
        """根据ID更新实体"""
        db_obj = self.get(id, session)
        if not db_obj:
            raise EntityNotFoundError(f"{self.model.__name__} with ID {id} not found")
        
        return self.update(db_obj, obj_in, session)
    
    def delete(self, id: Any, session: Optional[Session] = None) -> bool:
        """删除实体"""
        try:
            db_obj = self.get(id, session)
            if not db_obj:
                return False
            
            if session:
                session.delete(db_obj)
                session.flush()
            else:
                with self.session_manager.session_scope() as db_session:
                    db_session.delete(db_obj)
                    db_session.flush()
            
            logger.info(f"Deleted {self.model.__name__} with ID: {id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting {self.model.__name__} with ID {id}: {e}")
            return False
    
    def soft_delete(self, id: Any, session: Optional[Session] = None) -> bool:
        """软删除实体"""
        if not self._soft_delete_field:
            raise RepositoryError("Soft delete field not configured")
        
        try:
            db_obj = self.get(id, session)
            if not db_obj:
                return False
            
            # 设置软删除标记
            setattr(db_obj, self._soft_delete_field, datetime.utcnow())
            
            if session:
                session.flush()
            else:
                with self.session_manager.session_scope() as db_session:
                    db_session.merge(db_obj)
                    db_session.flush()
            
            logger.info(f"Soft deleted {self.model.__name__} with ID: {id}")
            return True
            
        except Exception as e:
            logger.error(f"Error soft deleting {self.model.__name__} with ID {id}: {e}")
            return False
    
    def restore(self, id: Any, session: Optional[Session] = None) -> bool:
        """恢复软删除的实体"""
        if not self._soft_delete_field:
            raise RepositoryError("Soft delete field not configured")
        
        try:
            # 查询包括软删除的实体
            query_builder = create_query_builder(self.model, session)
            query_builder.filter_by(id=id)
            
            if session:
                query_builder.session = session
                db_obj = query_builder.first()
            else:
                with self.session_manager.session_scope() as db_session:
                    query_builder.session = db_session
                    db_obj = query_builder.first()
            
            if not db_obj:
                return False
            
            # 清除软删除标记
            setattr(db_obj, self._soft_delete_field, None)
            
            if session:
                session.flush()
            else:
                with self.session_manager.session_scope() as db_session:
                    db_session.merge(db_obj)
                    db_session.flush()
            
            logger.info(f"Restored {self.model.__name__} with ID: {id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring {self.model.__name__} with ID {id}: {e}")
            return False
    
    def bulk_create(
        self, 
        objs_in: List[Union[Dict[str, Any], Any]],
        session: Optional[Session] = None
    ) -> List[T]:
        """批量创建实体"""
        try:
            db_objs = []
            
            for obj_in in objs_in:
                # 转换输入数据
                if hasattr(obj_in, 'dict'):
                    obj_data = obj_in.dict(exclude_unset=True)
                elif hasattr(obj_in, 'model_dump'):
                    obj_data = obj_in.model_dump(exclude_unset=True)
                elif isinstance(obj_in, dict):
                    obj_data = obj_in.copy()
                else:
                    raise ValidationError(f"Invalid input type: {type(obj_in)}")
                
                # 设置时间戳
                obj_data = self._set_timestamps(obj_data)
                
                # 创建实体
                db_obj = self.model(**obj_data)
                db_objs.append(db_obj)
            
            # 批量保存
            if session:
                session.add_all(db_objs)
                session.flush()
                for db_obj in db_objs:
                    session.refresh(db_obj)
            else:
                with self.session_manager.session_scope() as db_session:
                    db_session.add_all(db_objs)
                    db_session.flush()
                    for db_obj in db_objs:
                        db_session.refresh(db_obj)
            
            logger.info(f"Bulk created {len(db_objs)} {self.model.__name__} entities")
            return db_objs
            
        except Exception as e:
            logger.error(f"Error bulk creating {self.model.__name__}: {e}")
            raise RepositoryError(f"Failed to bulk create entities: {e}")
    
    def bulk_update(
        self, 
        updates: List[Dict[str, Any]],
        session: Optional[Session] = None
    ) -> int:
        """批量更新实体"""
        try:
            updated_count = 0
            
            if session:
                for update_data in updates:
                    if 'id' not in update_data:
                        continue
                    
                    # 设置时间戳
                    update_data = self._set_timestamps(update_data, is_update=True)
                    
                    result = session.execute(
                        update(self.model)
                        .where(self.model.id == update_data['id'])
                        .values(**{k: v for k, v in update_data.items() if k != 'id'})
                    )
                    updated_count += result.rowcount
                
                session.flush()
            else:
                with self.session_manager.session_scope() as db_session:
                    for update_data in updates:
                        if 'id' not in update_data:
                            continue
                        
                        # 设置时间戳
                        update_data = self._set_timestamps(update_data, is_update=True)
                        
                        result = db_session.execute(
                            update(self.model)
                            .where(self.model.id == update_data['id'])
                            .values(**{k: v for k, v in update_data.items() if k != 'id'})
                        )
                        updated_count += result.rowcount
                    
                    db_session.flush()
            
            logger.info(f"Bulk updated {updated_count} {self.model.__name__} entities")
            return updated_count
            
        except Exception as e:
            logger.error(f"Error bulk updating {self.model.__name__}: {e}")
            raise RepositoryError(f"Failed to bulk update entities: {e}")
    
    def bulk_delete(
        self, 
        ids: List[Any],
        session: Optional[Session] = None
    ) -> int:
        """批量删除实体"""
        try:
            if session:
                result = session.execute(
                    delete(self.model).where(self.model.id.in_(ids))
                )
                deleted_count = result.rowcount
                session.flush()
            else:
                with self.session_manager.session_scope() as db_session:
                    result = db_session.execute(
                        delete(self.model).where(self.model.id.in_(ids))
                    )
                    deleted_count = result.rowcount
                    db_session.flush()
            
            logger.info(f"Bulk deleted {deleted_count} {self.model.__name__} entities")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error bulk deleting {self.model.__name__}: {e}")
            raise RepositoryError(f"Failed to bulk delete entities: {e}")
    
    def paginate(
        self, 
        page: int = 1, 
        per_page: int = 20,
        filters: Optional[QueryFilter] = None,
        order_by: Optional[List[tuple]] = None,
        session: Optional[Session] = None,
        load_options: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """分页查询"""
        try:
            query_builder = create_query_builder(self.model, session)
            
            # 应用软删除过滤
            query_builder = self._apply_soft_delete_filter(query_builder)
            
            # 应用自定义过滤
            if filters:
                query_builder.filter(filters)
            
            # 应用排序
            if order_by:
                for field, order in order_by:
                    query_builder.order_by(field, order)
            
            # 应用加载选项
            options = load_options or self._default_load_options
            if options:
                query_builder.options(*options)
            
            if session:
                query_builder.session = session
                return query_builder.paginate_result(page, per_page)
            else:
                with self.session_manager.session_scope() as db_session:
                    query_builder.session = db_session
                    return query_builder.paginate_result(page, per_page)
                    
        except Exception as e:
            logger.error(f"Error paginating {self.model.__name__}: {e}")
            return {
                "items": [],
                "total": 0,
                "page": page,
                "per_page": per_page,
                "total_pages": 0,
                "has_prev": False,
                "has_next": False,
                "prev_page": None,
                "next_page": None
            }


def create_repository(model: Type[T], session_manager: Optional[SessionManager] = None) -> CRUDRepository[T]:
    """创建CRUD仓储"""
    return CRUDRepository(model, session_manager)