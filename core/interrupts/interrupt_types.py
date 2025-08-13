"""
中断类型定义

定义中断处理中使用的数据结构和枚举类型。
"""

from enum import Enum
from typing import Any, Dict, Optional, List
from pydantic import BaseModel
from datetime import datetime


class InterruptType(str, Enum):
    """中断类型枚举"""
    HUMAN_INPUT = "human_input"           # 需要人工输入
    APPROVAL = "approval"                 # 需要审批
    CONFIRMATION = "confirmation"         # 需要确认
    REVIEW = "review"                     # 需要审查
    DECISION = "decision"                 # 需要决策
    ERROR_HANDLING = "error_handling"     # 错误处理
    TIMEOUT = "timeout"                   # 超时处理
    CUSTOM = "custom"                     # 自定义中断


class InterruptStatus(str, Enum):
    """中断状态枚举"""
    PENDING = "pending"                   # 待处理
    IN_PROGRESS = "in_progress"           # 处理中
    APPROVED = "approved"                 # 已批准
    REJECTED = "rejected"                 # 已拒绝
    TIMEOUT = "timeout"                   # 已超时
    CANCELLED = "cancelled"               # 已取消
    COMPLETED = "completed"               # 已完成


class InterruptPriority(str, Enum):
    """中断优先级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class InterruptRequest(BaseModel):
    """中断请求"""
    interrupt_id: str
    run_id: str
    node_id: str
    interrupt_type: InterruptType
    priority: InterruptPriority = InterruptPriority.MEDIUM
    title: str
    message: str
    context: Dict[str, Any] = {}
    options: List[Dict[str, Any]] = []  # 可选的选项列表
    timeout: Optional[int] = None       # 超时时间（秒）
    required_approvers: List[str] = []  # 需要的审批者
    metadata: Dict[str, Any] = {}
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    
    def __init__(self, **data):
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        if data.get('timeout') and 'expires_at' not in data:
            from datetime import timedelta
            data['expires_at'] = data['created_at'] + timedelta(seconds=data['timeout'])
        super().__init__(**data)


class InterruptResponse(BaseModel):
    """中断响应"""
    interrupt_id: str
    responder_id: str
    response_type: str = "approval"  # approval, input, decision等
    response_data: Dict[str, Any] = {}
    approved: bool = True
    message: Optional[str] = None
    metadata: Dict[str, Any] = {}
    responded_at: datetime = None
    
    def __init__(self, **data):
        if 'responded_at' not in data:
            data['responded_at'] = datetime.utcnow()
        super().__init__(**data)


class ApprovalRequest(BaseModel):
    """审批请求"""
    approval_id: str
    interrupt_id: str
    title: str
    description: str
    requester_id: str
    approver_ids: List[str]
    approval_type: str = "single"  # single, multiple, unanimous
    context: Dict[str, Any] = {}
    attachments: List[Dict[str, Any]] = []
    deadline: Optional[datetime] = None
    created_at: datetime = None
    
    def __init__(self, **data):
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        super().__init__(**data)


class ApprovalResponse(BaseModel):
    """审批响应"""
    approval_id: str
    approver_id: str
    decision: str  # approved, rejected, delegated
    comments: Optional[str] = None
    conditions: List[str] = []  # 批准条件
    metadata: Dict[str, Any] = {}
    responded_at: datetime = None
    
    def __init__(self, **data):
        if 'responded_at' not in data:
            data['responded_at'] = datetime.utcnow()
        super().__init__(**data)


class HumanInputRequest(BaseModel):
    """人工输入请求"""
    request_id: str
    interrupt_id: str
    input_type: str  # text, choice, file, form等
    prompt: str
    validation_rules: Dict[str, Any] = {}
    default_value: Optional[Any] = None
    options: List[Dict[str, Any]] = []  # 对于选择类型
    required: bool = True
    metadata: Dict[str, Any] = {}
    created_at: datetime = None
    
    def __init__(self, **data):
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        super().__init__(**data)


class HumanInputResponse(BaseModel):
    """人工输入响应"""
    request_id: str
    user_id: str
    input_value: Any
    validation_passed: bool = True
    validation_errors: List[str] = []
    metadata: Dict[str, Any] = {}
    submitted_at: datetime = None
    
    def __init__(self, **data):
        if 'submitted_at' not in data:
            data['submitted_at'] = datetime.utcnow()
        super().__init__(**data)


class InterruptContext(BaseModel):
    """中断上下文"""
    run_id: str
    node_id: str
    agent_id: str
    user_id: str
    session_id: str
    current_state: Dict[str, Any] = {}
    execution_history: List[Dict[str, Any]] = []
    available_actions: List[str] = []
    metadata: Dict[str, Any] = {}


class InterruptNotification(BaseModel):
    """中断通知"""
    notification_id: str
    interrupt_id: str
    recipient_id: str
    notification_type: str  # email, sms, push, webhook等
    title: str
    message: str
    priority: InterruptPriority
    channels: List[str] = []  # 通知渠道
    metadata: Dict[str, Any] = {}
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    
    def __init__(self, **data):
        if 'sent_at' not in data:
            data['sent_at'] = datetime.utcnow()
        super().__init__(**data)