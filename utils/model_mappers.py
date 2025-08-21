"""模型映射工具

提供数据库模型与API模型之间的转换方法，确保数据一致性和类型安全。
包含批量转换、字段映射、数据验证等功能。
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import inspect

# 泛型类型变量
DBModel = TypeVar('DBModel')
APIModel = TypeVar('APIModel', bound=BaseModel)


class ModelMapper:
    """模型映射器基类
    
    提供数据库模型与API模型之间的双向转换功能。
    """
    
    def __init__(self):
        self._field_mappings: Dict[str, Dict[str, str]] = {}
        self._custom_converters: Dict[str, Dict[str, Callable]] = {}
        self._validation_rules: Dict[str, List[Callable]] = {}
    
    def register_field_mapping(self, model_name: str, db_field: str, api_field: str):
        """注册字段映射"""
        if model_name not in self._field_mappings:
            self._field_mappings[model_name] = {}
        self._field_mappings[model_name][db_field] = api_field
    
    def register_converter(self, model_name: str, field_name: str, converter: Callable):
        """注册自定义转换器"""
        if model_name not in self._custom_converters:
            self._custom_converters[model_name] = {}
        self._custom_converters[model_name][field_name] = converter
    
    def register_validation_rule(self, model_name: str, validator: Callable):
        """注册验证规则"""
        if model_name not in self._validation_rules:
            self._validation_rules[model_name] = []
        self._validation_rules[model_name].append(validator)
    
    def db_to_api(self, db_model: DBModel, api_model_class: Type[APIModel], 
                  exclude_fields: Optional[List[str]] = None,
                  include_fields: Optional[List[str]] = None) -> APIModel:
        """数据库模型转API模型"""
        if db_model is None:
            return None
        
        model_name = db_model.__class__.__name__
        exclude_fields = exclude_fields or []
        
        # 获取数据库模型的所有字段
        db_data = self._extract_db_fields(db_model, exclude_fields, include_fields)
        
        # 应用字段映射
        api_data = self._apply_field_mappings(db_data, model_name, 'db_to_api')
        
        # 应用自定义转换器
        api_data = self._apply_converters(api_data, model_name, 'db_to_api')
        
        # 创建API模型实例
        try:
            api_instance = api_model_class(**api_data)
            
            # 应用验证规则
            self._apply_validation_rules(api_instance, model_name)
            
            return api_instance
        except Exception as e:
            raise ValueError(f"转换失败: {str(e)}, 数据: {api_data}")
    
    def api_to_db(self, api_model: APIModel, db_model_class: Type[DBModel],
                  exclude_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """API模型转数据库模型数据"""
        if api_model is None:
            return None
        
        model_name = api_model.__class__.__name__
        exclude_fields = exclude_fields or []
        
        # 获取API模型数据
        api_data = api_model.dict(exclude=set(exclude_fields))
        
        # 应用字段映射（反向）
        db_data = self._apply_field_mappings(api_data, model_name, 'api_to_db')
        
        # 应用自定义转换器（反向）
        db_data = self._apply_converters(db_data, model_name, 'api_to_db')
        
        # 过滤数据库模型不存在的字段
        db_data = self._filter_db_fields(db_data, db_model_class)
        
        return db_data
    
    def batch_db_to_api(self, db_models: List[DBModel], api_model_class: Type[APIModel],
                        **kwargs) -> List[APIModel]:
        """批量转换数据库模型到API模型"""
        if not db_models:
            return []
        
        return [self.db_to_api(db_model, api_model_class, **kwargs) 
                for db_model in db_models if db_model is not None]
    
    def _extract_db_fields(self, db_model: DBModel, exclude_fields: List[str],
                          include_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """提取数据库模型字段"""
        data = {}
        
        # 获取SQLAlchemy模型的所有列
        mapper = inspect(db_model.__class__)
        
        for column in mapper.columns:
            field_name = column.name
            
            # 检查包含/排除字段
            if include_fields and field_name not in include_fields:
                continue
            if field_name in exclude_fields:
                continue
            
            value = getattr(db_model, field_name, None)
            
            # 处理特殊类型
            if isinstance(value, datetime):
                data[field_name] = value
            elif hasattr(value, '__dict__'):  # 关联对象
                continue  # 跳过关联对象，需要单独处理
            else:
                data[field_name] = value
        
        # 处理关系字段
        for relationship in mapper.relationships:
            rel_name = relationship.key
            
            if include_fields and rel_name not in include_fields:
                continue
            if rel_name in exclude_fields:
                continue
            
            rel_value = getattr(db_model, rel_name, None)
            if rel_value is not None:
                if relationship.uselist:  # 一对多关系
                    data[rel_name] = [item.id if hasattr(item, 'id') else str(item) 
                                     for item in rel_value]
                else:  # 一对一关系
                    data[rel_name] = rel_value.id if hasattr(rel_value, 'id') else str(rel_value)
        
        return data
    
    def _apply_field_mappings(self, data: Dict[str, Any], model_name: str, 
                             direction: str) -> Dict[str, Any]:
        """应用字段映射"""
        if model_name not in self._field_mappings:
            return data
        
        mappings = self._field_mappings[model_name]
        new_data = data.copy()
        
        if direction == 'db_to_api':
            for db_field, api_field in mappings.items():
                if db_field in new_data:
                    new_data[api_field] = new_data.pop(db_field)
        else:  # api_to_db
            reverse_mappings = {v: k for k, v in mappings.items()}
            for api_field, db_field in reverse_mappings.items():
                if api_field in new_data:
                    new_data[db_field] = new_data.pop(api_field)
        
        return new_data
    
    def _apply_converters(self, data: Dict[str, Any], model_name: str,
                         direction: str) -> Dict[str, Any]:
        """应用自定义转换器"""
        if model_name not in self._custom_converters:
            return data
        
        converters = self._custom_converters[model_name]
        new_data = data.copy()
        
        for field_name, converter in converters.items():
            if field_name in new_data:
                try:
                    new_data[field_name] = converter(new_data[field_name], direction)
                except Exception as e:
                    # 转换失败时保持原值
                    pass
        
        return new_data
    
    def _apply_validation_rules(self, api_instance: APIModel, model_name: str):
        """应用验证规则"""
        if model_name not in self._validation_rules:
            return
        
        validators = self._validation_rules[model_name]
        for validator in validators:
            try:
                validator(api_instance)
            except Exception as e:
                raise ValueError(f"验证失败: {str(e)}")
    
    def _filter_db_fields(self, data: Dict[str, Any], db_model_class: Type[DBModel]) -> Dict[str, Any]:
        """过滤数据库模型不存在的字段"""
        mapper = inspect(db_model_class)
        valid_fields = set()
        
        # 添加列字段
        for column in mapper.columns:
            valid_fields.add(column.name)
        
        # 添加关系字段
        for relationship in mapper.relationships:
            valid_fields.add(relationship.key)
        
        return {k: v for k, v in data.items() if k in valid_fields}


# 全局映射器实例
mapper = ModelMapper()


# 预定义转换器
def json_converter(value: Any, direction: str) -> Any:
    """JSON字段转换器"""
    if direction == 'db_to_api':
        return value if isinstance(value, (dict, list)) else {}
    else:
        return value


def datetime_converter(value: Any, direction: str) -> Any:
    """日期时间转换器"""
    if direction == 'db_to_api':
        return value.isoformat() if isinstance(value, datetime) else value
    else:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except:
                return value
        return value


def list_converter(value: Any, direction: str) -> List[Any]:
    """列表转换器"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            import json
            return json.loads(value)
        except:
            return [value]
    return [value]


# 注册常用转换器
mapper.register_converter('User', 'preferences', json_converter)
mapper.register_converter('Message', 'content_data', json_converter)
mapper.register_converter('Message', 'message_metadata', json_converter)
mapper.register_converter('Message', 'tool_calls', list_converter)
mapper.register_converter('Message', 'tool_results', list_converter)
mapper.register_converter('Workflow', 'definition', json_converter)
mapper.register_converter('Workflow', 'workflow_metadata', json_converter)
mapper.register_converter('Workflow', 'permissions', json_converter)
mapper.register_converter('Workflow', 'tags', list_converter)
mapper.register_converter('Memory', 'memory_metadata', json_converter)
mapper.register_converter('Memory', 'tags', list_converter)
mapper.register_converter('Memory', 'associations', list_converter)
mapper.register_converter('Memory', 'context_data', json_converter)


# 验证规则
def validate_user_response(user: 'UserResponse'):
    """验证用户响应"""
    if not user.username or len(user.username) < 3:
        raise ValueError("用户名长度不能少于3个字符")
    if not user.email or '@' not in user.email:
        raise ValueError("邮箱格式不正确")


def validate_message_response(message: 'MessageResponse'):
    """验证消息响应"""
    if message.content and len(message.content) > 10000:
        raise ValueError("消息内容过长")
    if message.quality_score and (message.quality_score < 0 or message.quality_score > 1):
        raise ValueError("质量评分必须在0-1之间")


def validate_workflow_response(workflow: 'WorkflowResponse'):
    """验证工作流响应"""
    if not workflow.name or len(workflow.name) < 2:
        raise ValueError("工作流名称长度不能少于2个字符")
    if workflow.complexity_score < 0 or workflow.complexity_score > 10:
        raise ValueError("复杂度评分必须在0-10之间")


def validate_memory_response(memory: 'MemoryResponse'):
    """验证记忆响应"""
    if not memory.content or len(memory.content.strip()) == 0:
        raise ValueError("记忆内容不能为空")
    if memory.importance_score < 0 or memory.importance_score > 1:
        raise ValueError("重要性评分必须在0-1之间")


# 注册验证规则
mapper.register_validation_rule('UserResponse', validate_user_response)
mapper.register_validation_rule('MessageResponse', validate_message_response)
mapper.register_validation_rule('WorkflowResponse', validate_workflow_response)
mapper.register_validation_rule('MemoryResponse', validate_memory_response)


# 便捷函数
def convert_user_to_response(user_model):
    """转换用户模型到响应"""
    from models.response_models import UserResponse
    return mapper.db_to_api(user_model, UserResponse)


def convert_message_to_response(message_model):
    """转换消息模型到响应"""
    from models.response_models import MessageResponse
    return mapper.db_to_api(message_model, MessageResponse)


def convert_workflow_to_response(workflow_model):
    """转换工作流模型到响应"""
    from models.response_models import WorkflowResponse
    return mapper.db_to_api(workflow_model, WorkflowResponse)


def convert_memory_to_response(memory_model):
    """转换记忆模型到响应"""
    from models.response_models import MemoryResponse
    return mapper.db_to_api(memory_model, MemoryResponse)


def batch_convert_to_responses(models: List[Any], response_class: Type[APIModel]) -> List[APIModel]:
    """批量转换模型到响应"""
    return mapper.batch_db_to_api(models, response_class)


class DataTransformationService:
    """数据转换服务
    
    提供高级数据转换功能，包括关联数据处理、缓存优化等。
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self._cache = {}
    
    def transform_with_relations(self, db_model: DBModel, api_model_class: Type[APIModel],
                                include_relations: List[str] = None) -> APIModel:
        """转换模型并包含关联数据"""
        # 基础转换
        api_model = mapper.db_to_api(db_model, api_model_class)
        
        # 处理关联数据
        if include_relations:
            for relation_name in include_relations:
                if hasattr(db_model, relation_name):
                    relation_data = getattr(db_model, relation_name)
                    if relation_data:
                        # 根据关联类型进行转换
                        transformed_relation = self._transform_relation(relation_data, relation_name)
                        setattr(api_model, relation_name, transformed_relation)
        
        return api_model
    
    def _transform_relation(self, relation_data: Any, relation_name: str) -> Any:
        """转换关联数据"""
        # 这里可以根据关联类型进行不同的转换
        if isinstance(relation_data, list):
            return [self._transform_single_relation(item, relation_name) for item in relation_data]
        else:
            return self._transform_single_relation(relation_data, relation_name)
    
    def _transform_single_relation(self, item: Any, relation_name: str) -> Dict[str, Any]:
        """转换单个关联项"""
        if hasattr(item, 'id'):
            return {
                'id': item.id,
                'name': getattr(item, 'name', None),
                'type': item.__class__.__name__.lower()
            }
        return str(item)
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()