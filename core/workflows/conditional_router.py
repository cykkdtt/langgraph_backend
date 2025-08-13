"""
条件路由器

提供基于条件的路由和分支执行功能。
"""

import asyncio
from typing import Optional, List, Dict, Any, Callable, Union, Tuple
from datetime import datetime
import re
import json

from .workflow_types import (
    Condition, ConditionType, ConditionalBranch,
    WorkflowStep, WorkflowExecution
)


class ConditionEvaluator:
    """条件评估器"""
    
    def __init__(self):
        self._custom_functions: Dict[str, Callable] = {}
        self._variables: Dict[str, Any] = {}
        
        # 注册内置函数
        self._register_builtin_functions()
    
    def register_function(self, name: str, func: Callable) -> None:
        """注册自定义函数"""
        self._custom_functions[name] = func
    
    def set_variable(self, name: str, value: Any) -> None:
        """设置变量"""
        self._variables[name] = value
    
    def set_variables(self, variables: Dict[str, Any]) -> None:
        """批量设置变量"""
        self._variables.update(variables)
    
    def evaluate(self, condition: Condition, context: Dict[str, Any]) -> bool:
        """评估条件"""
        try:
            # 合并上下文和变量
            eval_context = {**self._variables, **context}
            
            if condition.type == ConditionType.SIMPLE:
                return self._evaluate_simple(condition.expression, eval_context)
            elif condition.type == ConditionType.COMPLEX:
                return self._evaluate_complex(condition.expression, eval_context)
            elif condition.type == ConditionType.SCRIPT:
                return self._evaluate_script(condition.expression, eval_context)
            elif condition.type == ConditionType.REGEX:
                return self._evaluate_regex(condition.expression, eval_context)
            else:
                raise ValueError(f"不支持的条件类型: {condition.type}")
        
        except Exception as e:
            # 记录错误但不抛出，返回 False
            print(f"条件评估失败: {e}")
            return False
    
    def _evaluate_simple(self, expression: str, context: Dict[str, Any]) -> bool:
        """评估简单表达式"""
        # 替换变量
        eval_expr = self._replace_variables(expression, context)
        
        # 安全的表达式评估
        try:
            # 只允许安全的操作符
            allowed_names = {
                "__builtins__": {},
                "True": True,
                "False": False,
                "None": None,
                **self._custom_functions,
                **context
            }
            
            result = eval(eval_expr, allowed_names)
            return bool(result)
        
        except Exception as e:
            print(f"简单表达式评估失败: {e}")
            return False
    
    def _evaluate_complex(self, expression: str, context: Dict[str, Any]) -> bool:
        """评估复杂表达式"""
        # 解析复杂表达式（支持 AND, OR, NOT 等逻辑操作）
        try:
            # 简化实现，实际应用中可能需要更复杂的解析器
            expr = expression.upper()
            
            # 替换逻辑操作符
            expr = expr.replace(" AND ", " and ")
            expr = expr.replace(" OR ", " or ")
            expr = expr.replace(" NOT ", " not ")
            
            # 替换变量
            expr = self._replace_variables(expr, context)
            
            # 评估表达式
            allowed_names = {
                "__builtins__": {},
                "True": True,
                "False": False,
                "None": None,
                "and": lambda x, y: x and y,
                "or": lambda x, y: x or y,
                "not": lambda x: not x,
                **self._custom_functions,
                **context
            }
            
            result = eval(expr, allowed_names)
            return bool(result)
        
        except Exception as e:
            print(f"复杂表达式评估失败: {e}")
            return False
    
    def _evaluate_script(self, script: str, context: Dict[str, Any]) -> bool:
        """评估脚本"""
        try:
            # 创建安全的执行环境
            local_vars = {**context, **self._custom_functions}
            global_vars = {"__builtins__": {}}
            
            # 执行脚本
            exec(script, global_vars, local_vars)
            
            # 脚本应该设置 result 变量
            return bool(local_vars.get("result", False))
        
        except Exception as e:
            print(f"脚本评估失败: {e}")
            return False
    
    def _evaluate_regex(self, pattern: str, context: Dict[str, Any]) -> bool:
        """评估正则表达式"""
        try:
            # 解析正则表达式配置
            # 格式: "field_name:pattern" 或 JSON 配置
            if ":" in pattern and not pattern.startswith("{"):
                field_name, regex_pattern = pattern.split(":", 1)
                field_value = str(context.get(field_name, ""))
                return bool(re.search(regex_pattern, field_value))
            else:
                # JSON 配置格式
                config = json.loads(pattern)
                field_name = config["field"]
                regex_pattern = config["pattern"]
                flags = config.get("flags", 0)
                
                field_value = str(context.get(field_name, ""))
                return bool(re.search(regex_pattern, field_value, flags))
        
        except Exception as e:
            print(f"正则表达式评估失败: {e}")
            return False
    
    def _replace_variables(self, expression: str, context: Dict[str, Any]) -> str:
        """替换表达式中的变量"""
        # 简单的变量替换实现
        result = expression
        
        # 替换 ${variable} 格式的变量
        import re
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_name = match.group(1)
            if var_name in context:
                value = context[var_name]
                if isinstance(value, str):
                    return f"'{value}'"
                else:
                    return str(value)
            return match.group(0)
        
        result = re.sub(pattern, replace_var, result)
        
        return result
    
    def _register_builtin_functions(self) -> None:
        """注册内置函数"""
        self._custom_functions.update({
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "any": any,
            "all": all,
            "contains": lambda container, item: item in container,
            "startswith": lambda s, prefix: str(s).startswith(prefix),
            "endswith": lambda s, suffix: str(s).endswith(suffix),
            "is_empty": lambda x: not x if x is not None else True,
            "is_null": lambda x: x is None,
            "equals": lambda x, y: x == y,
            "greater_than": lambda x, y: x > y,
            "less_than": lambda x, y: x < y,
            "between": lambda x, min_val, max_val: min_val <= x <= max_val
        })


class ConditionalRouter:
    """条件路由器"""
    
    def __init__(self):
        self.evaluator = ConditionEvaluator()
        self._route_handlers: Dict[str, Callable] = {}
        self._default_handler: Optional[Callable] = None
    
    def register_route_handler(self, route_name: str, handler: Callable) -> None:
        """注册路由处理器"""
        self._route_handlers[route_name] = handler
    
    def set_default_handler(self, handler: Callable) -> None:
        """设置默认处理器"""
        self._default_handler = handler
    
    async def route(
        self,
        branches: List[ConditionalBranch],
        conditions: List[Condition],
        context: Dict[str, Any]
    ) -> Tuple[List[str], Optional[str]]:
        """执行条件路由"""
        # 创建条件映射
        condition_map = {cond.id: cond for cond in conditions}
        
        # 评估每个分支
        for branch in branches:
            if branch.condition_id in condition_map:
                condition = condition_map[branch.condition_id]
                
                if self.evaluator.evaluate(condition, context):
                    # 条件为真，返回真分支步骤
                    return branch.true_steps, branch.id
                else:
                    # 条件为假，检查是否有假分支
                    if branch.false_steps:
                        return branch.false_steps, branch.id
        
        # 没有匹配的分支，返回空列表
        return [], None
    
    async def execute_route(
        self,
        route_name: str,
        steps: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行路由"""
        if route_name in self._route_handlers:
            handler = self._route_handlers[route_name]
            return await self._call_handler(handler, steps, context)
        elif self._default_handler:
            return await self._call_handler(self._default_handler, steps, context)
        else:
            # 默认执行：简单返回步骤列表
            return {
                "executed_steps": steps,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
    
    async def evaluate_conditions(
        self,
        conditions: List[Condition],
        context: Dict[str, Any]
    ) -> Dict[str, bool]:
        """批量评估条件"""
        results = {}
        
        for condition in conditions:
            try:
                result = self.evaluator.evaluate(condition, context)
                results[condition.id] = result
            except Exception as e:
                print(f"条件 {condition.id} 评估失败: {e}")
                results[condition.id] = False
        
        return results
    
    async def find_matching_branches(
        self,
        branches: List[ConditionalBranch],
        conditions: List[Condition],
        context: Dict[str, Any]
    ) -> List[Tuple[ConditionalBranch, bool]]:
        """查找匹配的分支"""
        condition_map = {cond.id: cond for cond in conditions}
        matching_branches = []
        
        for branch in branches:
            if branch.condition_id in condition_map:
                condition = condition_map[branch.condition_id]
                result = self.evaluator.evaluate(condition, context)
                matching_branches.append((branch, result))
        
        return matching_branches
    
    async def _call_handler(
        self,
        handler: Callable,
        steps: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """调用处理器"""
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(steps, context)
            else:
                return handler(steps, context)
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
    
    def create_condition_chain(
        self,
        conditions: List[Condition],
        operator: str = "AND"
    ) -> Condition:
        """创建条件链"""
        if not conditions:
            raise ValueError("条件列表不能为空")
        
        if len(conditions) == 1:
            return conditions[0]
        
        # 构建复合条件表达式
        expressions = []
        for cond in conditions:
            expressions.append(f"({cond.expression})")
        
        if operator.upper() == "AND":
            combined_expr = " and ".join(expressions)
        elif operator.upper() == "OR":
            combined_expr = " or ".join(expressions)
        else:
            raise ValueError(f"不支持的操作符: {operator}")
        
        return Condition(
            id=f"chain_{datetime.now().timestamp()}",
            name=f"条件链_{operator}",
            type=ConditionType.COMPLEX,
            expression=combined_expr,
            description=f"由 {len(conditions)} 个条件组成的 {operator} 链"
        )
    
    def create_negation_condition(self, condition: Condition) -> Condition:
        """创建否定条件"""
        return Condition(
            id=f"not_{condition.id}",
            name=f"NOT {condition.name}",
            type=condition.type,
            expression=f"not ({condition.expression})",
            description=f"条件 {condition.name} 的否定"
        )