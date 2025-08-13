"""
增强版中断管理器

集成LangGraph官方的interrupt()函数和Command原语，
提供与现有架构兼容的人工干预功能。
包含高级审批工作流和复杂审批逻辑。
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
import logging
from enum import Enum
from pydantic import BaseModel, Field

from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from .interrupt_types import (
    InterruptRequest, InterruptResponse, InterruptType, InterruptStatus,
    InterruptPriority, InterruptContext, InterruptNotification,
    ApprovalRequest, ApprovalResponse, HumanInputRequest, HumanInputResponse
)


class ApprovalWorkflowType(str, Enum):
    """审批工作流类型"""
    SIMPLE = "simple"  # 简单审批（单人）
    SEQUENTIAL = "sequential"  # 顺序审批
    PARALLEL = "parallel"  # 并行审批
    MAJORITY = "majority"  # 多数决
    UNANIMOUS = "unanimous"  # 一致同意


class ApprovalRule(BaseModel):
    """审批规则"""
    name: str
    description: str
    workflow_type: ApprovalWorkflowType
    required_approvers: List[str]
    minimum_approvals: Optional[int] = None
    timeout_seconds: Optional[int] = None
    auto_approve_conditions: Optional[Dict[str, Any]] = None
    escalation_rules: Optional[Dict[str, Any]] = None


class ApprovalWorkflow(BaseModel):
    """审批工作流"""
    id: str
    request_id: str
    rule: ApprovalRule
    current_step: int = 0
    completed_steps: List[str] = []
    pending_approvers: List[str] = []
    responses: List[ApprovalResponse] = []
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class EnhancedInterruptManager:
    """增强版中断管理器 - 集成LangGraph官方API和高级审批工作流"""
    
    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        self.checkpointer = checkpointer
        self.active_interrupts: Dict[str, InterruptRequest] = {}
        self.interrupt_responses: Dict[str, List[InterruptResponse]] = {}
        self.interrupt_handlers: Dict[InterruptType, List[Callable]] = {}
        self.notification_handlers: List[Callable] = []
        
        # 高级审批工作流
        self.approval_rules: Dict[str, ApprovalRule] = {}
        self.active_workflows: Dict[str, ApprovalWorkflow] = {}
        self.approval_handlers: List[Callable] = []
        self.escalation_handlers: List[Callable] = []
        
        self.logger = logging.getLogger("enhanced.interrupt.manager")
        
        # 设置默认审批规则
        self._setup_default_approval_rules()
        
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_interrupts())
    
    def _setup_default_approval_rules(self):
        """设置默认审批规则"""
        # 简单审批规则
        simple_rule = ApprovalRule(
            name="simple_approval",
            description="简单审批，需要一个审批者",
            workflow_type=ApprovalWorkflowType.SIMPLE,
            required_approvers=["admin"],
            timeout_seconds=3600
        )
        self.approval_rules["simple"] = simple_rule
        
        # 高优先级审批规则
        high_priority_rule = ApprovalRule(
            name="high_priority_approval",
            description="高优先级审批，需要管理员和主管同意",
            workflow_type=ApprovalWorkflowType.PARALLEL,
            required_approvers=["admin", "supervisor"],
            minimum_approvals=2,
            timeout_seconds=1800
        )
        self.approval_rules["high_priority"] = high_priority_rule
        
        # 关键操作审批规则
        critical_rule = ApprovalRule(
            name="critical_approval",
            description="关键操作审批，需要所有管理员一致同意",
            workflow_type=ApprovalWorkflowType.UNANIMOUS,
            required_approvers=["admin", "supervisor", "security_officer"],
            timeout_seconds=7200
        )
        self.approval_rules["critical"] = critical_rule
    
    def create_approval_interrupt(
        self,
        title: str,
        description: str,
        context: Dict[str, Any],
        priority: InterruptPriority = InterruptPriority.MEDIUM,
        required_approvers: Optional[List[str]] = None,
        timeout_seconds: Optional[int] = None,
        options: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """创建审批中断 - 使用LangGraph官方interrupt()函数
        
        这个方法应该在智能体节点内部调用，会触发LangGraph的中断机制。
        
        Args:
            title: 审批标题
            description: 审批描述
            context: 上下文信息
            priority: 优先级
            required_approvers: 需要的审批者
            timeout_seconds: 超时时间
            options: 可选选项
            
        Returns:
            Dict[str, Any]: 中断数据，传递给interrupt()函数
        """
        interrupt_id = str(uuid.uuid4())
        
        # 创建中断请求记录
        interrupt_request = InterruptRequest(
            interrupt_id=interrupt_id,
            run_id=context.get("run_id", ""),
            node_id=context.get("node_id", ""),
            interrupt_type=InterruptType.APPROVAL,
            priority=priority,
            title=title,
            message=description,
            context=context,
            options=options or [],
            timeout=timeout_seconds,
            required_approvers=required_approvers or []
        )
        
        self.active_interrupts[interrupt_id] = interrupt_request
        self.interrupt_responses[interrupt_id] = []
        
        self.logger.info(f"创建审批中断: {interrupt_id}")
        
        # 返回传递给interrupt()的数据
        return {
            "type": "approval_request",
            "interrupt_id": interrupt_id,
            "title": title,
            "description": description,
            "priority": priority.value,
            "context": context,
            "options": options or [
                {"value": "approve", "label": "批准"},
                {"value": "reject", "label": "拒绝"}
            ],
            "required_approvers": required_approvers or [],
            "timeout_seconds": timeout_seconds,
            "created_at": datetime.utcnow().isoformat()
        }
    
    def create_human_input_interrupt(
        self,
        prompt: str,
        input_type: str,
        context: Dict[str, Any],
        validation_rules: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
        default_value: Optional[Any] = None,
        options: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """创建人工输入中断
        
        Args:
            prompt: 输入提示
            input_type: 输入类型 (text, choice, file, form等)
            context: 上下文信息
            validation_rules: 验证规则
            timeout_seconds: 超时时间
            default_value: 默认值
            options: 选项列表
            
        Returns:
            Dict[str, Any]: 中断数据
        """
        interrupt_id = str(uuid.uuid4())
        
        interrupt_request = InterruptRequest(
            interrupt_id=interrupt_id,
            run_id=context.get("run_id", ""),
            node_id=context.get("node_id", ""),
            interrupt_type=InterruptType.HUMAN_INPUT,
            priority=InterruptPriority.MEDIUM,
            title="人工输入请求",
            message=prompt,
            context=context,
            timeout=timeout_seconds
        )
        
        self.active_interrupts[interrupt_id] = interrupt_request
        self.interrupt_responses[interrupt_id] = []
        
        self.logger.info(f"创建人工输入中断: {interrupt_id}")
        
        return {
            "type": "human_input_request",
            "interrupt_id": interrupt_id,
            "prompt": prompt,
            "input_type": input_type,
            "context": context,
            "validation_rules": validation_rules or {},
            "default_value": default_value,
            "options": options or [],
            "timeout_seconds": timeout_seconds,
            "created_at": datetime.utcnow().isoformat()
        }
    
    def create_tool_review_interrupt(
        self,
        proposed_tools: List[Dict[str, Any]],
        context: Dict[str, Any],
        allow_modifications: bool = True
    ) -> Dict[str, Any]:
        """创建工具审查中断
        
        Args:
            proposed_tools: 提议的工具调用列表
            context: 上下文信息
            allow_modifications: 是否允许修改
            
        Returns:
            Dict[str, Any]: 中断数据
        """
        interrupt_id = str(uuid.uuid4())
        
        interrupt_request = InterruptRequest(
            interrupt_id=interrupt_id,
            run_id=context.get("run_id", ""),
            node_id=context.get("node_id", ""),
            interrupt_type=InterruptType.REVIEW,
            priority=InterruptPriority.MEDIUM,
            title="工具调用审查",
            message=f"请审查 {len(proposed_tools)} 个工具调用",
            context=context
        )
        
        self.active_interrupts[interrupt_id] = interrupt_request
        self.interrupt_responses[interrupt_id] = []
        
        self.logger.info(f"创建工具审查中断: {interrupt_id}")
        
        return {
            "type": "tool_review_request",
            "interrupt_id": interrupt_id,
            "title": "工具调用审查",
            "description": f"请审查以下 {len(proposed_tools)} 个工具调用",
            "proposed_tools": proposed_tools,
            "context": context,
            "allow_modifications": allow_modifications,
            "actions": [
                {"value": "approve_all", "label": "批准所有工具"},
                {"value": "modify", "label": "修改工具调用"},
                {"value": "reject_all", "label": "拒绝所有工具"}
            ] if allow_modifications else [
                {"value": "approve_all", "label": "批准所有工具"},
                {"value": "reject_all", "label": "拒绝所有工具"}
            ],
            "created_at": datetime.utcnow().isoformat()
        }
    
    def create_state_edit_interrupt(
        self,
        current_state: Dict[str, Any],
        editable_fields: List[str],
        context: Dict[str, Any],
        validation_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建状态编辑中断
        
        Args:
            current_state: 当前状态
            editable_fields: 可编辑字段列表
            context: 上下文信息
            validation_schema: 验证模式
            
        Returns:
            Dict[str, Any]: 中断数据
        """
        interrupt_id = str(uuid.uuid4())
        
        interrupt_request = InterruptRequest(
            interrupt_id=interrupt_id,
            run_id=context.get("run_id", ""),
            node_id=context.get("node_id", ""),
            interrupt_type=InterruptType.REVIEW,
            priority=InterruptPriority.MEDIUM,
            title="状态编辑",
            message="请审查和编辑当前状态",
            context=context
        )
        
        self.active_interrupts[interrupt_id] = interrupt_request
        self.interrupt_responses[interrupt_id] = []
        
        self.logger.info(f"创建状态编辑中断: {interrupt_id}")
        
        return {
            "type": "state_edit_request",
            "interrupt_id": interrupt_id,
            "title": "状态编辑",
            "description": "请审查和编辑当前状态",
            "current_state": current_state,
            "editable_fields": editable_fields,
            "context": context,
            "validation_schema": validation_schema or {},
            "actions": [
                {"value": "approve", "label": "保持当前状态"},
                {"value": "edit", "label": "编辑状态"},
                {"value": "reset", "label": "重置状态"}
            ],
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def process_interrupt_response(
        self,
        interrupt_id: str,
        response_data: Dict[str, Any],
        responder_id: str
    ) -> bool:
        """处理中断响应
        
        Args:
            interrupt_id: 中断ID
            response_data: 响应数据
            responder_id: 响应者ID
            
        Returns:
            bool: 是否成功处理
        """
        if interrupt_id not in self.active_interrupts:
            self.logger.warning(f"中断请求不存在: {interrupt_id}")
            return False
        
        interrupt_request = self.active_interrupts[interrupt_id]
        
        # 检查是否已过期
        if self._is_expired(interrupt_request):
            self.logger.warning(f"中断请求已过期: {interrupt_id}")
            return False
        
        # 创建响应记录
        response = InterruptResponse(
            interrupt_id=interrupt_id,
            responder_id=responder_id,
            response_data=response_data,
            approved=response_data.get("approved", True),
            message=response_data.get("message")
        )
        
        self.interrupt_responses[interrupt_id].append(response)
        
        self.logger.info(f"处理中断响应: {interrupt_id} (响应者: {responder_id})")
        
        # 检查是否满足完成条件
        if await self._check_response_completion(interrupt_request):
            await self._complete_interrupt(interrupt_id)
        
        return True
    
    def get_interrupt_status(self, interrupt_id: str) -> Optional[InterruptStatus]:
        """获取中断状态"""
        if interrupt_id not in self.active_interrupts:
            return None
        
        interrupt_request = self.active_interrupts[interrupt_id]
        responses = self.interrupt_responses.get(interrupt_id, [])
        
        if self._is_expired(interrupt_request):
            return InterruptStatus.TIMEOUT
        
        if not responses:
            return InterruptStatus.PENDING
        
        # 检查是否完成
        if len(responses) >= len(interrupt_request.required_approvers or [1]):
            approved_responses = [r for r in responses if r.approved]
            return InterruptStatus.APPROVED if approved_responses else InterruptStatus.REJECTED
        
        return InterruptStatus.IN_PROGRESS
    
    def _is_expired(self, interrupt_request: InterruptRequest) -> bool:
        """检查中断是否已过期"""
        if not interrupt_request.expires_at:
            return False
        return datetime.utcnow() > interrupt_request.expires_at
    
    async def _check_response_completion(self, interrupt_request: InterruptRequest) -> bool:
        """检查响应是否完成"""
        responses = self.interrupt_responses.get(interrupt_request.interrupt_id, [])
        
        # 如果没有指定审批者，一个响应就足够
        if not interrupt_request.required_approvers:
            return len(responses) > 0
        
        # 检查是否所有必需的审批者都已响应
        responder_ids = {r.responder_id for r in responses}
        required_approvers = set(interrupt_request.required_approvers)
        
        return required_approvers.issubset(responder_ids)
    
    async def _complete_interrupt(self, interrupt_id: str):
        """完成中断处理"""
        if interrupt_id in self.active_interrupts:
            interrupt_request = self.active_interrupts[interrupt_id]
            del self.active_interrupts[interrupt_id]
            
            self.logger.info(f"完成中断处理: {interrupt_id}")
            
            # 触发完成事件处理器
            await self._trigger_completion_handlers(interrupt_request)
    
    async def _cleanup_expired_interrupts(self):
        """清理过期的中断"""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_ids = []
                
                for interrupt_id, interrupt_request in self.active_interrupts.items():
                    if self._is_expired(interrupt_request):
                        expired_ids.append(interrupt_id)
                
                for interrupt_id in expired_ids:
                    await self._handle_expired_interrupt(interrupt_id)
                
                # 清理过期的审批工作流
                expired_workflows = []
                for workflow_id, workflow in self.active_workflows.items():
                    if workflow.rule.timeout_seconds:
                        timeout_time = workflow.created_at + timedelta(seconds=workflow.rule.timeout_seconds)
                        if current_time > timeout_time:
                            expired_workflows.append(workflow_id)
                
                for workflow_id in expired_workflows:
                    await self._handle_expired_workflow(workflow_id)
                
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                self.logger.error(f"清理过期中断时出错: {e}")
                await asyncio.sleep(60)
    
    async def _handle_expired_interrupt(self, interrupt_id: str):
        """处理过期的中断"""
        if interrupt_id in self.active_interrupts:
            request = self.active_interrupts[interrupt_id]
            
            # 创建超时响应
            timeout_response = InterruptResponse(
                request_id=interrupt_id,
                response_type=request.interrupt_type,
                status=InterruptStatus.TIMEOUT,
                message="中断请求已超时",
                timestamp=datetime.utcnow()
            )
            
            # 添加到响应列表
            if interrupt_id not in self.interrupt_responses:
                self.interrupt_responses[interrupt_id] = []
            self.interrupt_responses[interrupt_id].append(timeout_response)
            
            # 从活跃中断中移除
            del self.active_interrupts[interrupt_id]
            
            # 发送通知
            await self._send_notification(
                InterruptNotification(
                    interrupt_id=interrupt_id,
                    message="中断请求已超时",
                    notification_type="timeout",
                    timestamp=datetime.utcnow()
                )
            )
            
            self.logger.warning(f"中断 {interrupt_id} 已超时")
    
    async def _handle_expired_workflow(self, workflow_id: str):
        """处理过期的审批工作流"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = "timeout"
            workflow.completed_at = datetime.utcnow()
            
            # 发送超时通知
            await self._send_workflow_timeout_notification(workflow)
            
            # 移除活跃工作流
            del self.active_workflows[workflow_id]
            
            self.logger.warning(f"审批工作流 {workflow_id} 已超时")
    
    # ==================== 高级审批工作流方法 ====================
    
    def add_approval_rule(self, rule_name: str, rule: ApprovalRule):
        """添加审批规则"""
        self.approval_rules[rule_name] = rule
        self.logger.info(f"添加审批规则: {rule_name}")
    
    def register_approval_handler(self, handler: Callable):
        """注册审批处理器"""
        self.approval_handlers.append(handler)
    
    def register_escalation_handler(self, handler: Callable):
        """注册升级处理器"""
        self.escalation_handlers.append(handler)
    
    async def start_approval_workflow(
        self,
        request: ApprovalRequest,
        rule_name: str = "simple"
    ) -> str:
        """启动审批工作流"""
        if rule_name not in self.approval_rules:
            raise ValueError(f"审批规则 {rule_name} 不存在")
        
        rule = self.approval_rules[rule_name]
        workflow_id = str(uuid.uuid4())
        
        workflow = ApprovalWorkflow(
            id=workflow_id,
            request_id=request.request_id,
            rule=rule,
            pending_approvers=rule.required_approvers.copy()
        )
        
        self.active_workflows[workflow_id] = workflow
        
        # 检查自动审批条件
        if await self._check_auto_approval_conditions(request, rule):
            await self._auto_approve_request(workflow_id, request)
            return workflow_id
        
        # 发送审批通知
        await self._send_approval_notifications(workflow, request)
        
        self.logger.info(f"启动审批工作流: {workflow_id}, 规则: {rule_name}")
        return workflow_id
    
    async def _check_auto_approval_conditions(
        self,
        request: ApprovalRequest,
        rule: ApprovalRule
    ) -> bool:
        """检查自动审批条件"""
        if not rule.auto_approve_conditions:
            return False
        
        conditions = rule.auto_approve_conditions
        
        # 检查请求者权限
        if "allowed_requesters" in conditions:
            if request.requester_id in conditions["allowed_requesters"]:
                return True
        
        # 检查优先级
        if "max_priority" in conditions:
            if request.priority.value <= conditions["max_priority"]:
                return True
        
        # 检查金额限制（如果适用）
        if "amount_limit" in conditions and "amount" in request.metadata:
            if float(request.metadata["amount"]) <= conditions["amount_limit"]:
                return True
        
        return False
    
    async def _auto_approve_request(self, workflow_id: str, request: ApprovalRequest):
        """自动审批请求"""
        workflow = self.active_workflows[workflow_id]
        
        # 创建自动审批响应
        auto_response = ApprovalResponse(
            request_id=request.request_id,
            approver_id="system",
            approved=True,
            reason="满足自动审批条件",
            timestamp=datetime.utcnow()
        )
        
        workflow.responses.append(auto_response)
        workflow.status = "approved"
        workflow.completed_at = datetime.utcnow()
        
        # 发送自动审批通知
        await self._send_auto_approval_notification(workflow, request)
        
        # 移除活跃工作流
        del self.active_workflows[workflow_id]
        
        self.logger.info(f"自动审批工作流: {workflow_id}")
    
    async def process_approval_response(
        self,
        workflow_id: str,
        response: ApprovalResponse
    ) -> bool:
        """处理审批响应"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"审批工作流 {workflow_id} 不存在")
        
        workflow = self.active_workflows[workflow_id]
        
        # 验证审批者权限
        if response.approver_id not in workflow.rule.required_approvers:
            raise ValueError(f"审批者 {response.approver_id} 无权限审批此请求")
        
        # 检查是否已经审批过
        for existing_response in workflow.responses:
            if existing_response.approver_id == response.approver_id:
                raise ValueError(f"审批者 {response.approver_id} 已经审批过此请求")
        
        # 添加响应
        workflow.responses.append(response)
        
        # 从待审批列表中移除
        if response.approver_id in workflow.pending_approvers:
            workflow.pending_approvers.remove(response.approver_id)
        
        # 检查工作流是否完成
        is_completed = await self._check_workflow_completion(workflow)
        
        if is_completed:
            await self._complete_workflow(workflow_id)
        
        return is_completed
    
    async def _check_workflow_completion(self, workflow: ApprovalWorkflow) -> bool:
        """检查工作流是否完成"""
        rule = workflow.rule
        responses = workflow.responses
        
        if rule.workflow_type == ApprovalWorkflowType.SIMPLE:
            # 简单审批：只需要一个审批
            return len(responses) > 0
        
        elif rule.workflow_type == ApprovalWorkflowType.SEQUENTIAL:
            # 顺序审批：按顺序完成所有审批
            return len(responses) == len(rule.required_approvers)
        
        elif rule.workflow_type == ApprovalWorkflowType.PARALLEL:
            # 并行审批：达到最小审批数量
            approved_count = sum(1 for r in responses if r.approved)
            min_approvals = rule.minimum_approvals or len(rule.required_approvers)
            return approved_count >= min_approvals
        
        elif rule.workflow_type == ApprovalWorkflowType.MAJORITY:
            # 多数决：超过一半同意
            approved_count = sum(1 for r in responses if r.approved)
            total_approvers = len(rule.required_approvers)
            return approved_count > total_approvers // 2
        
        elif rule.workflow_type == ApprovalWorkflowType.UNANIMOUS:
            # 一致同意：所有人都同意
            if len(responses) < len(rule.required_approvers):
                return False
            return all(r.approved for r in responses)
        
        return False
    
    async def _complete_workflow(self, workflow_id: str):
        """完成工作流"""
        workflow = self.active_workflows[workflow_id]
        
        # 确定最终状态
        approved_count = sum(1 for r in workflow.responses if r.approved)
        rejected_count = len(workflow.responses) - approved_count
        
        if approved_count > 0 and rejected_count == 0:
            workflow.status = "approved"
        elif rejected_count > 0:
            workflow.status = "rejected"
        else:
            workflow.status = "completed"
        
        workflow.completed_at = datetime.utcnow()
        
        # 发送完成通知
        await self._send_workflow_completion_notification(workflow)
        
        # 移除活跃工作流
        del self.active_workflows[workflow_id]
        
        self.logger.info(f"完成审批工作流: {workflow_id}, 状态: {workflow.status}")
    
    async def cancel_approval_workflow(self, workflow_id: str, reason: str = "用户取消"):
        """取消审批工作流"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"审批工作流 {workflow_id} 不存在")
        
        workflow = self.active_workflows[workflow_id]
        workflow.status = "cancelled"
        workflow.completed_at = datetime.utcnow()
        
        # 发送取消通知
        await self._send_workflow_cancellation_notification(workflow, reason)
        
        # 移除活跃工作流
        del self.active_workflows[workflow_id]
        
        self.logger.info(f"取消审批工作流: {workflow_id}, 原因: {reason}")
    
    def get_pending_approvals(self, approver_id: str) -> List[ApprovalWorkflow]:
        """获取待审批的工作流"""
        pending_workflows = []
        
        for workflow in self.active_workflows.values():
            if (workflow.status == "pending" and 
                approver_id in workflow.pending_approvers):
                pending_workflows.append(workflow)
        
        return pending_workflows
    
    def is_workflow_complete(self, workflow_id: str) -> bool:
        """检查工作流是否完成"""
        if workflow_id not in self.active_workflows:
            return True  # 不在活跃列表中说明已完成
        
        workflow = self.active_workflows[workflow_id]
        return workflow.status in ["approved", "rejected", "cancelled", "timeout"]
    
    # ==================== 通知方法 ====================
    
    async def _send_approval_notifications(self, workflow: ApprovalWorkflow, request: ApprovalRequest):
        """发送审批通知"""
        for approver_id in workflow.rule.required_approvers:
            notification = InterruptNotification(
                interrupt_id=workflow.request_id,
                message=f"新的审批请求: {request.title}",
                notification_type="approval_request",
                timestamp=datetime.utcnow(),
                metadata={
                    "workflow_id": workflow.id,
                    "approver_id": approver_id,
                    "request_title": request.title,
                    "priority": request.priority.value
                }
            )
            await self._send_notification(notification)
    
    async def _send_workflow_completion_notification(self, workflow: ApprovalWorkflow):
        """发送工作流完成通知"""
        notification = InterruptNotification(
            interrupt_id=workflow.request_id,
            message=f"审批工作流已完成，状态: {workflow.status}",
            notification_type="workflow_completed",
            timestamp=datetime.utcnow(),
            metadata={
                "workflow_id": workflow.id,
                "status": workflow.status,
                "completion_time": workflow.completed_at.isoformat() if workflow.completed_at else None
            }
        )
        await self._send_notification(notification)
    
    async def _send_workflow_cancellation_notification(self, workflow: ApprovalWorkflow, reason: str):
        """发送工作流取消通知"""
        notification = InterruptNotification(
            interrupt_id=workflow.request_id,
            message=f"审批工作流已取消: {reason}",
            notification_type="workflow_cancelled",
            timestamp=datetime.utcnow(),
            metadata={
                "workflow_id": workflow.id,
                "reason": reason
            }
        )
        await self._send_notification(notification)
    
    async def _send_auto_approval_notification(self, workflow: ApprovalWorkflow, request: ApprovalRequest):
        """发送自动审批通知"""
        notification = InterruptNotification(
            interrupt_id=workflow.request_id,
            message=f"请求已自动审批: {request.title}",
            notification_type="auto_approved",
            timestamp=datetime.utcnow(),
            metadata={
                "workflow_id": workflow.id,
                "request_title": request.title
            }
        )
        await self._send_notification(notification)
    
    async def _send_workflow_timeout_notification(self, workflow: ApprovalWorkflow):
        """发送工作流超时通知"""
        notification = InterruptNotification(
            interrupt_id=workflow.request_id,
            message=f"审批工作流已超时",
            notification_type="workflow_timeout",
            timestamp=datetime.utcnow(),
            metadata={
                "workflow_id": workflow.id,
                "timeout_duration": workflow.rule.timeout_seconds
            }
        )
        await self._send_notification(notification)
    
    async def _trigger_completion_handlers(self, interrupt_request: InterruptRequest):
        """触发完成事件处理器"""
        # 这里可以添加完成事件的处理逻辑
        pass
    
    async def _send_notification(self, notification: InterruptNotification):
        """发送通知"""
        for handler in self.notification_handlers:
            try:
                await handler(notification)
            except Exception as e:
                self.logger.error(f"通知处理器执行失败: {e}")
    
    def register_notification_handler(self, handler: Callable):
        """注册通知处理器"""
        self.notification_handlers.append(handler)


# 便利函数，用于在智能体节点中使用
def create_approval_node_interrupt(
    manager: EnhancedInterruptManager,
    title: str,
    description: str,
    context: Dict[str, Any],
    **kwargs
) -> Any:
    """在节点中创建审批中断的便利函数
    
    使用方法:
    def my_node(state):
        # 需要审批时
        approval_result = create_approval_node_interrupt(
            manager, 
            "请审批此操作", 
            "这是一个重要操作",
            {"user_id": state["user_id"]}
        )
        # approval_result 包含审批结果
        return state
    """
    interrupt_data = manager.create_approval_interrupt(
        title, description, context, **kwargs
    )
    
    # 使用LangGraph官方interrupt()函数
    return interrupt(interrupt_data)


def create_human_input_node_interrupt(
    manager: EnhancedInterruptManager,
    prompt: str,
    input_type: str,
    context: Dict[str, Any],
    **kwargs
) -> Any:
    """在节点中创建人工输入中断的便利函数"""
    interrupt_data = manager.create_human_input_interrupt(
        prompt, input_type, context, **kwargs
    )
    
    return interrupt(interrupt_data)


def create_tool_review_node_interrupt(
    manager: EnhancedInterruptManager,
    proposed_tools: List[Dict[str, Any]],
    context: Dict[str, Any],
    **kwargs
) -> Any:
    """在节点中创建工具审查中断的便利函数"""
    interrupt_data = manager.create_tool_review_interrupt(
        proposed_tools, context, **kwargs
    )
    
    return interrupt(interrupt_data)


def create_state_edit_node_interrupt(
    manager: EnhancedInterruptManager,
    current_state: Dict[str, Any],
    editable_fields: List[str],
    context: Dict[str, Any],
    **kwargs
) -> Any:
    """在节点中创建状态编辑中断的便利函数"""
    interrupt_data = manager.create_state_edit_interrupt(
        current_state, editable_fields, context, **kwargs
    )
    
    return interrupt(interrupt_data)