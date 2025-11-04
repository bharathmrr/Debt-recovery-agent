from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class ConversationChannel(str, Enum):
    CHAT = "chat"
    SMS = "sms"
    EMAIL = "email"
    VOICE = "voice"


class ActionType(str, Enum):
    INFORM = "inform"
    COLLECT_PAYMENT = "collect_payment"
    PROPOSE_PLAN = "propose_plan"
    ACKNOWLEDGE = "acknowledge"
    REQUEST_INFO = "request_info"
    ESCALATE = "escalate"
    CLOSE = "close"
    VERIFY_IDENTITY = "verify_identity"


class PlanType(str, Enum):
    INSTALLMENT = "installment"
    SETTLEMENT = "settlement"
    ONE_TIME = "one_time"


# Request Models
class UserMessage(BaseModel):
    loan_id: int
    user_text: str
    session_id: str
    channel: ConversationChannel = ConversationChannel.CHAT
    metadata: Optional[Dict[str, Any]] = None


class IdentityVerificationRequest(BaseModel):
    conversation_id: str
    verification_data: Dict[str, str]  # e.g., {"last_four_ssn": "1234", "last_payment_amount": "150.00"}


class PaymentRequest(BaseModel):
    loan_id: int
    amount: float
    payment_method: str
    conversation_id: str
    metadata: Optional[Dict[str, Any]] = None


class EscalationRequest(BaseModel):
    conversation_id: str
    reason: str
    priority: str = "normal"  # low, normal, high, urgent
    notes: Optional[str] = None


# Response Models
class StructuredPlan(BaseModel):
    type: PlanType
    amount: float
    installments: Optional[int] = None
    first_due_date: Optional[str] = None
    frequency: Optional[str] = "monthly"
    
    @validator('first_due_date')
    def validate_date_format(cls, v):
        if v:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('Date must be in YYYY-MM-DD format')
        return v


class AssistantResponse(BaseModel):
    action: ActionType
    message_to_user: str
    structured_plan: Optional[StructuredPlan] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    escalation: bool = False
    compliance_checks: List[str] = []
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    ok: bool
    conversation_id: str
    assistant: AssistantResponse
    session_data: Optional[Dict[str, Any]] = None
    requires_action: bool = False
    next_steps: Optional[List[str]] = None


class BorrowerInfo(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    consent_status: str
    preferred_contact_method: str


class LoanInfo(BaseModel):
    id: int
    account_number: str
    principal_amount: float
    current_balance: float
    due_date: datetime
    status: str
    days_overdue: int
    last_payment_date: Optional[datetime] = None
    last_payment_amount: float


class ConversationInfo(BaseModel):
    id: int
    conversation_id: str
    state: str
    channel: str
    identity_verified: bool
    last_activity: datetime
    assigned_agent_id: Optional[str] = None


class MessageInfo(BaseModel):
    id: int
    message_type: str
    content: str
    confidence_score: Optional[float] = None
    created_at: datetime


class PaymentPlanInfo(BaseModel):
    id: int
    plan_type: str
    total_amount: float
    installment_amount: Optional[float] = None
    number_of_installments: Optional[int] = None
    first_payment_date: Optional[datetime] = None
    status: str


# RAG Models
class RAGQuery(BaseModel):
    query: str
    loan_id: Optional[int] = None
    borrower_id: Optional[int] = None
    document_types: Optional[List[str]] = None
    top_k: int = 5


class RAGResult(BaseModel):
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    document_type: str


class RAGResponse(BaseModel):
    results: List[RAGResult]
    total_results: int
    query_time: float


# Compliance Models
class ComplianceCheck(BaseModel):
    check_name: str
    passed: bool
    details: Optional[str] = None
    severity: str = "info"  # info, warning, error, critical


class ComplianceReport(BaseModel):
    conversation_id: str
    checks: List[ComplianceCheck]
    overall_status: str  # passed, warning, failed
    requires_human_review: bool = False
    timestamp: datetime


# Analytics Models
class ConversationMetrics(BaseModel):
    total_conversations: int
    active_conversations: int
    escalated_conversations: int
    resolution_rate: float
    average_resolution_time: float
    channel_breakdown: Dict[str, int]


class CollectionMetrics(BaseModel):
    total_amount_collected: float
    number_of_payments: int
    payment_plan_acceptance_rate: float
    settlement_rate: float
    average_days_to_resolution: float


class LLMMetrics(BaseModel):
    total_api_calls: int
    total_tokens_used: int
    average_response_time: float
    total_cost: float
    confidence_score_distribution: Dict[str, int]


# Webhook Models
class TwilioWebhook(BaseModel):
    MessageSid: str
    AccountSid: str
    From: str
    To: str
    Body: str
    NumMedia: Optional[str] = "0"


class StripeWebhook(BaseModel):
    id: str
    object: str
    type: str
    data: Dict[str, Any]


# Configuration Models
class ComplianceConfig(BaseModel):
    max_settlement_percentage: float = 0.70
    max_installment_months: int = 12
    contact_hours_start: str = "08:00"
    contact_hours_end: str = "21:00"
    max_daily_contact_attempts: int = 3
    required_verification_fields: List[str] = ["last_four_ssn", "last_payment_amount"]
    prohibited_contact_days: List[str] = []  # e.g., ["sunday"]


class LLMConfig(BaseModel):
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.45
    max_tokens: int = 1000
    timeout: int = 30
    retry_attempts: int = 3


class SystemConfig(BaseModel):
    compliance: ComplianceConfig
    llm: LLMConfig
    log_level: str = "INFO"
    enable_audit_logging: bool = True
    enable_pii_masking: bool = True


# Error Models
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationError(BaseModel):
    field: str
    message: str
    invalid_value: Any


# Health Check Models
class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str
    dependencies: Dict[str, str]  # service_name: status


# Export all models for easy importing
__all__ = [
    "UserMessage", "IdentityVerificationRequest", "PaymentRequest", "EscalationRequest",
    "StructuredPlan", "AssistantResponse", "ConversationResponse",
    "BorrowerInfo", "LoanInfo", "ConversationInfo", "MessageInfo", "PaymentPlanInfo",
    "RAGQuery", "RAGResult", "RAGResponse",
    "ComplianceCheck", "ComplianceReport",
    "ConversationMetrics", "CollectionMetrics", "LLMMetrics",
    "TwilioWebhook", "StripeWebhook",
    "ComplianceConfig", "LLMConfig", "SystemConfig",
    "ErrorResponse", "ValidationError", "HealthCheck"
]
