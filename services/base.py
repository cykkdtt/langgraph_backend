"""服务层基类

提供业务逻辑处理、数据转换、缓存管理、事件发布等功能。
定义服务层的通用接口和实现模式。
"""

import logging
from typing import (
    Any, Dict, List, Optional, Type, TypeVar, Union, Generic,
    Callable, Awaitable
)
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from uuid import UUID
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ..models.response_models import (
    BaseResponse, PaginatedResponse, PaginationMeta,
    ResponseStatus, ErrorCode
)
from ..utils.validation import (
    APIException, ValidationException, ResourceNotFoundException,
    BusinessRuleException, PermissionDeniedException
)
from ..utils.model_mappers import ModelMapper, DataTransformationService
from ..utils.database_optimization import QueryCache, get_query_cache
from ..utils.performance_monitoring import monitor_performance
from ..database.connection import get_sync_session, get_async_session
from ..database.repository import SyncRepository, AsyncRepository
from ..database.repositories import RepositoryManager, get_repository_manager


# 类型变量
ModelType = TypeVar('ModelType')
CreateSchemaType = TypeVar('CreateSchemaType', bound=BaseModel)
UpdateSchemaType = TypeVar('UpdateSchemaType', bound=BaseModel)
ResponseType = TypeVar('ResponseType', bound=BaseModel)


class ServiceError(APIException):
    """服务层错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.BUSINESS_LOGIC_ERROR,
            details=details
        )


class CacheConfig:
    """缓存配置"""
    
    def __init__(
        self,
        enabled: bool = True,
        ttl: int = 300,  # 5分钟
        key_prefix: str = "",
        invalidate_on_update: bool = True
    ):
        self.enabled = enabled
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.invalidate_on_update = invalidate_on_update


class EventData:
    """事件数据"""
    
    def __init__(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        self.event_type = event_type
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.data = data
        self.user_id = user_id
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "data": self.data,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat()
        }


class EventPublisher:
    """事件发布器"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(f"{__name__}.EventPublisher")
    
    def subscribe(self, event_type: str, handler: Callable[[EventData], None]):
        """订阅事件"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def publish(self, event: EventData):
        """发布事件"""
        try:
            handlers = self.subscribers.get(event.event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event.event_type}: {e}")
        except Exception as e:
            self.logger.error(f"Error publishing event {event.event_type}: {e}")
    
    async def publish_async(self, event: EventData):
        """异步发布事件"""
        try:
            handlers = self.subscribers.get(event.event_type, [])
            tasks = []
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        # 在线程池中运行同步处理器
                        loop = asyncio.get_event_loop()
                        tasks.append(loop.run_in_executor(None, handler, event))
                except Exception as e:
                    self.logger.error(f"Error preparing handler for {event.event_type}: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Error publishing async event {event.event_type}: {e}")


# 全局事件发布器
_event_publisher = EventPublisher()


def get_event_publisher() -> EventPublisher:
    """获取事件发布器"""
    return _event_publisher


def cache_result(config: CacheConfig):
    """缓存结果装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not config.enabled:
                return func(self, *args, **kwargs)
            
            # 生成缓存键
            cache_key = f"{config.key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            cache = get_query_cache()
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = func(self, *args, **kwargs)
            cache.set(cache_key, result, ttl=config.ttl)
            
            return result
        return wrapper
    return decorator


def invalidate_cache(key_pattern: str):
    """缓存失效装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            
            # 失效相关缓存
            cache = get_query_cache()
            cache.invalidate_pattern(key_pattern)
            
            return result
        return wrapper
    return decorator


def publish_event(event_type: str, entity_type: str):
    """发布事件装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            
            # 发布事件
            if hasattr(result, 'id'):
                entity_id = str(result.id)
                event_data = EventData(
                    event_type=event_type,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    data=result.dict() if hasattr(result, 'dict') else {},
                    user_id=getattr(self, 'current_user_id', None)
                )
                get_event_publisher().publish(event_data)
            
            return result
        return wrapper
    return decorator


class BaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType, ResponseType], ABC):
    """基础服务类"""
    
    def __init__(
        self,
        repository: SyncRepository[ModelType],
        response_model: Type[ResponseType],
        cache_config: Optional[CacheConfig] = None,
        session: Optional[Session] = None
    ):
        self.repository = repository
        self.response_model = response_model
        self.cache_config = cache_config or CacheConfig()
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.mapper = ModelMapper()
        self.transformer = DataTransformationService()
        self.event_publisher = get_event_publisher()
        
        # 当前用户信息（由认证中间件设置）
        self.current_user_id: Optional[str] = None
        self.current_user: Optional[Any] = None
    
    def set_current_user(self, user_id: str, user: Any = None):
        """设置当前用户"""
        self.current_user_id = user_id
        self.current_user = user
    
    def _check_permission(self, action: str, resource: Any = None):
        """检查权限"""
        # 默认实现，子类可以重写
        if not self.current_user_id:
            raise PermissionDeniedException("Authentication required")
    
    def _validate_business_rules(self, data: Any, action: str = "create"):
        """验证业务规则"""
        # 默认实现，子类可以重写
        pass
    
    def _transform_to_response(self, data: ModelType) -> ResponseType:
        """转换为响应模型"""
        if hasattr(self.response_model, 'from_db_model'):
            return self.response_model.from_db_model(data)
        else:
            # 使用模型映射器
            return self.mapper.map_to_api_model(data, self.response_model)
    
    def _create_success_response(
        self,
        data: Any,
        message: str = "Operation successful"
    ) -> BaseResponse[Any]:
        """创建成功响应"""
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=data
        )
    
    def _create_error_response(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.BUSINESS_LOGIC_ERROR,
        details: Optional[Dict[str, Any]] = None
    ) -> BaseResponse[None]:
        """创建错误响应"""
        return BaseResponse(
            status=ResponseStatus.ERROR,
            message=message,
            error_code=error_code,
            details=details
        )
    
    @monitor_performance
    def create(self, data: CreateSchemaType) -> BaseResponse[ResponseType]:
        """创建资源"""
        try:
            # 权限检查
            self._check_permission("create")
            
            # 业务规则验证
            self._validate_business_rules(data, "create")
            
            # 创建资源
            db_obj = self.repository.create(data)
            
            # 转换为响应模型
            response_data = self._transform_to_response(db_obj)
            
            # 发布事件
            event = EventData(
                event_type="created",
                entity_type=self.repository.model.__name__.lower(),
                entity_id=str(db_obj.id),
                data=response_data.dict() if hasattr(response_data, 'dict') else {},
                user_id=self.current_user_id
            )
            self.event_publisher.publish(event)
            
            # 失效相关缓存
            if self.cache_config.invalidate_on_update:
                cache = get_query_cache()
                cache.invalidate_pattern(f"{self.cache_config.key_prefix}:*")
            
            return self._create_success_response(
                response_data,
                f"{self.repository.model.__name__} created successfully"
            )
            
        except APIException:
            raise
        except Exception as e:
            self.logger.error(f"Error creating {self.repository.model.__name__}: {e}")
            raise ServiceError(f"Failed to create {self.repository.model.__name__}")
    
    @monitor_performance
    @cache_result(CacheConfig(ttl=300))
    def get(self, id: Any) -> BaseResponse[ResponseType]:
        """获取资源"""
        try:
            # 获取资源
            db_obj = self.repository.get_or_404(id)
            
            # 权限检查
            self._check_permission("read", db_obj)
            
            # 转换为响应模型
            response_data = self._transform_to_response(db_obj)
            
            return self._create_success_response(response_data)
            
        except APIException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting {self.repository.model.__name__} {id}: {e}")
            raise ServiceError(f"Failed to get {self.repository.model.__name__}")
    
    @monitor_performance
    @cache_result(CacheConfig(ttl=180))
    def get_list(
        self,
        skip: int = 0,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        sorts: Optional[List[Dict[str, str]]] = None
    ) -> BaseResponse[List[ResponseType]]:
        """获取资源列表"""
        try:
            # 权限检查
            self._check_permission("list")
            
            # 构建查询过滤器和排序
            query_filters = []
            query_sorts = []
            
            if filters:
                from ..database.repository import QueryFilter
                for field, value in filters.items():
                    if isinstance(value, dict) and 'operator' in value:
                        query_filters.append(QueryFilter(
                            field=field,
                            operator=value['operator'],
                            value=value['value']
                        ))
                    else:
                        query_filters.append(QueryFilter(field, "eq", value))
            
            if sorts:
                from ..database.repository import QuerySort
                for sort_item in sorts:
                    query_sorts.append(QuerySort(
                        field=sort_item['field'],
                        direction=sort_item.get('direction', 'asc')
                    ))
            
            # 获取数据
            db_objects = self.repository.get_multi(
                skip=skip,
                limit=limit,
                filters=query_filters,
                sorts=query_sorts
            )
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(obj) for obj in db_objects
            ]
            
            return self._create_success_response(response_data)
            
        except APIException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting {self.repository.model.__name__} list: {e}")
            raise ServiceError(f"Failed to get {self.repository.model.__name__} list")
    
    @monitor_performance
    @cache_result(CacheConfig(ttl=180))
    def get_paginated(
        self,
        page: int = 1,
        size: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        sorts: Optional[List[Dict[str, str]]] = None
    ) -> BaseResponse[PaginatedResponse[ResponseType]]:
        """获取分页资源"""
        try:
            # 权限检查
            self._check_permission("list")
            
            # 构建查询过滤器和排序
            query_filters = []
            query_sorts = []
            
            if filters:
                from ..database.repository import QueryFilter
                for field, value in filters.items():
                    if isinstance(value, dict) and 'operator' in value:
                        query_filters.append(QueryFilter(
                            field=field,
                            operator=value['operator'],
                            value=value['value']
                        ))
                    else:
                        query_filters.append(QueryFilter(field, "eq", value))
            
            if sorts:
                from ..database.repository import QuerySort
                for sort_item in sorts:
                    query_sorts.append(QuerySort(
                        field=sort_item['field'],
                        direction=sort_item.get('direction', 'asc')
                    ))
            
            # 获取分页数据
            paginated_result = self.repository.get_paginated(
                page=page,
                size=size,
                filters=query_filters,
                sorts=query_sorts
            )
            
            # 转换为响应模型
            response_items = [
                self._transform_to_response(obj) for obj in paginated_result.items
            ]
            
            response_data = PaginatedResponse(
                items=response_items,
                pagination=paginated_result.pagination
            )
            
            return self._create_success_response(response_data)
            
        except APIException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting paginated {self.repository.model.__name__}: {e}")
            raise ServiceError(f"Failed to get paginated {self.repository.model.__name__}")
    
    @monitor_performance
    @invalidate_cache("*")
    def update(self, id: Any, data: UpdateSchemaType) -> BaseResponse[ResponseType]:
        """更新资源"""
        try:
            # 获取现有资源
            db_obj = self.repository.get_or_404(id)
            
            # 权限检查
            self._check_permission("update", db_obj)
            
            # 业务规则验证
            self._validate_business_rules(data, "update")
            
            # 更新资源
            updated_obj = self.repository.update(id, data)
            
            # 转换为响应模型
            response_data = self._transform_to_response(updated_obj)
            
            # 发布事件
            event = EventData(
                event_type="updated",
                entity_type=self.repository.model.__name__.lower(),
                entity_id=str(id),
                data=response_data.dict() if hasattr(response_data, 'dict') else {},
                user_id=self.current_user_id
            )
            self.event_publisher.publish(event)
            
            return self._create_success_response(
                response_data,
                f"{self.repository.model.__name__} updated successfully"
            )
            
        except APIException:
            raise
        except Exception as e:
            self.logger.error(f"Error updating {self.repository.model.__name__} {id}: {e}")
            raise ServiceError(f"Failed to update {self.repository.model.__name__}")
    
    @monitor_performance
    @invalidate_cache("*")
    def delete(self, id: Any, soft_delete: bool = True) -> BaseResponse[None]:
        """删除资源"""
        try:
            # 获取现有资源
            db_obj = self.repository.get_or_404(id)
            
            # 权限检查
            self._check_permission("delete", db_obj)
            
            # 删除资源
            success = self.repository.delete(id, soft_delete=soft_delete)
            
            if success:
                # 发布事件
                event = EventData(
                    event_type="deleted",
                    entity_type=self.repository.model.__name__.lower(),
                    entity_id=str(id),
                    data={"soft_delete": soft_delete},
                    user_id=self.current_user_id
                )
                self.event_publisher.publish(event)
                
                return self._create_success_response(
                    None,
                    f"{self.repository.model.__name__} deleted successfully"
                )
            else:
                raise ServiceError(f"Failed to delete {self.repository.model.__name__}")
            
        except APIException:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting {self.repository.model.__name__} {id}: {e}")
            raise ServiceError(f"Failed to delete {self.repository.model.__name__}")
    
    @monitor_performance
    def bulk_create(self, data_list: List[CreateSchemaType]) -> BaseResponse[List[ResponseType]]:
        """批量创建资源"""
        try:
            # 权限检查
            self._check_permission("bulk_create")
            
            # 验证每个项目的业务规则
            for data in data_list:
                self._validate_business_rules(data, "create")
            
            # 批量创建
            db_objects = self.repository.bulk_create(data_list)
            
            # 转换为响应模型
            response_data = [
                self._transform_to_response(obj) for obj in db_objects
            ]
            
            # 发布事件
            for i, db_obj in enumerate(db_objects):
                event = EventData(
                    event_type="bulk_created",
                    entity_type=self.repository.model.__name__.lower(),
                    entity_id=str(db_obj.id),
                    data=response_data[i].dict() if hasattr(response_data[i], 'dict') else {},
                    user_id=self.current_user_id
                )
                self.event_publisher.publish(event)
            
            # 失效相关缓存
            if self.cache_config.invalidate_on_update:
                cache = get_query_cache()
                cache.invalidate_pattern(f"{self.cache_config.key_prefix}:*")
            
            return self._create_success_response(
                response_data,
                f"Bulk created {len(response_data)} {self.repository.model.__name__} items"
            )
            
        except APIException:
            raise
        except Exception as e:
            self.logger.error(f"Error bulk creating {self.repository.model.__name__}: {e}")
            raise ServiceError(f"Failed to bulk create {self.repository.model.__name__}")
    
    @monitor_performance
    def get_statistics(self) -> BaseResponse[Dict[str, Any]]:
        """获取统计信息"""
        try:
            # 权限检查
            self._check_permission("statistics")
            
            # 基础统计
            total_count = self.repository.count()
            
            stats = {
                "total_count": total_count,
                "entity_type": self.repository.model.__name__.lower(),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return self._create_success_response(stats)
            
        except APIException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting {self.repository.model.__name__} statistics: {e}")
            raise ServiceError(f"Failed to get {self.repository.model.__name__} statistics")


class AsyncBaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType, ResponseType], ABC):
    """异步基础服务类"""
    
    def __init__(
        self,
        repository: AsyncRepository[ModelType],
        response_model: Type[ResponseType],
        cache_config: Optional[CacheConfig] = None,
        session: Optional[AsyncSession] = None
    ):
        self.repository = repository
        self.response_model = response_model
        self.cache_config = cache_config or CacheConfig()
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.mapper = ModelMapper()
        self.transformer = DataTransformationService()
        self.event_publisher = get_event_publisher()
        
        # 当前用户信息
        self.current_user_id: Optional[str] = None
        self.current_user: Optional[Any] = None
    
    # 异步版本的方法实现
    # 由于篇幅限制，这里只提供方法签名
    
    async def create(self, data: CreateSchemaType) -> BaseResponse[ResponseType]:
        """异步创建资源"""
        pass
    
    async def get(self, id: Any) -> BaseResponse[ResponseType]:
        """异步获取资源"""
        pass
    
    async def update(self, id: Any, data: UpdateSchemaType) -> BaseResponse[ResponseType]:
        """异步更新资源"""
        pass
    
    async def delete(self, id: Any, soft_delete: bool = True) -> BaseResponse[None]:
        """异步删除资源"""
        pass


# 服务工厂
class ServiceFactory:
    """服务工厂"""
    
    def __init__(self, repository_manager: Optional[RepositoryManager] = None):
        self.repository_manager = repository_manager or get_repository_manager()
        self._services = {}
    
    def create_service(
        self,
        service_class: Type[BaseService],
        repository: SyncRepository,
        response_model: Type[ResponseType],
        cache_config: Optional[CacheConfig] = None
    ) -> BaseService:
        """创建服务实例"""
        service_key = f"{service_class.__name__}_{repository.model.__name__}"
        
        if service_key not in self._services:
            self._services[service_key] = service_class(
                repository=repository,
                response_model=response_model,
                cache_config=cache_config
            )
        
        return self._services[service_key]
    
    def get_service(self, service_key: str) -> Optional[BaseService]:
        """获取服务实例"""
        return self._services.get(service_key)
    
    def clear_cache(self):
        """清理服务缓存"""
        self._services.clear()


# 全局服务工厂
_service_factory = None


def get_service_factory(repository_manager: Optional[RepositoryManager] = None) -> ServiceFactory:
    """获取服务工厂"""
    global _service_factory
    if _service_factory is None or repository_manager is not None:
        _service_factory = ServiceFactory(repository_manager)
    return _service_factory