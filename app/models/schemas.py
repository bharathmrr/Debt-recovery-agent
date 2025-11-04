from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum
from .database import Base


class ConsentStatus(enum.Enum):
    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"
    EXPIRED = "expired"


class LoanStatus(enum.Enum):
    ACTIVE = "active"
    OVERDUE = "overdue"
    SETTLED = "settled"
    PAID = "paid"
    WRITTEN_OFF = "written_off"
    IN_LITIGATION = "in_litigation"


class TransactionType(enum.Enum):
    PAYMENT = "payment"
    CHARGE = "charge"
    FEE = "fee"
    INTEREST = "interest"
    ADJUSTMENT = "adjustment"
    REFUND = "refund"


class ConversationState(enum.Enum):
    INITIATED = "initiated"
    IDENTITY_VERIFICATION = "identity_verification"
    ACTIVE_NEGOTIATION = "active_negotiation"
    PAYMENT_PROCESSING = "payment_processing"
    ESCALATED = "escalated"
    CLOSED = "closed"
    OPTED_OUT = "opted_out"


class MessageType(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class PaymentPlanType(enum.Enum):
    FULL_PAYMENT = "full_payment"
    INSTALLMENT = "installment"
    SETTLEMENT = "settlement"


class Borrower(Base):
    __tablename__ = "borrowers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True)
    phone = Column(String(20), index=True)
    address = Column(Text)
    date_of_birth = Column(DateTime)
    ssn_last_four = Column(String(4))  # Only store last 4 digits for verification
    consent_status = Column(Enum(ConsentStatus), default=ConsentStatus.PENDING)
    consent_date = Column(DateTime)
    opt_out_date = Column(DateTime)
    preferred_contact_method = Column(String(20), default="email")  # email, sms, phone
    contact_hours_start = Column(String(5), default="09:00")  # HH:MM format
    contact_hours_end = Column(String(5), default="18:00")
    timezone = Column(String(50), default="UTC")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    loans = relationship("Loan", back_populates="borrower")
    conversations = relationship("Conversation", back_populates="borrower")


class Loan(Base):
    __tablename__ = "loans"
    
    id = Column(Integer, primary_key=True, index=True)
    borrower_id = Column(Integer, ForeignKey("borrowers.id"), nullable=False)
    account_number = Column(String(50), unique=True, index=True)
    principal_amount = Column(Float, nullable=False)
    current_balance = Column(Float, nullable=False)
    interest_rate = Column(Float, default=0.0)
    currency = Column(String(3), default="USD")
    origination_date = Column(DateTime, nullable=False)
    due_date = Column(DateTime, nullable=False)
    last_payment_date = Column(DateTime)
    last_payment_amount = Column(Float, default=0.0)
    status = Column(Enum(LoanStatus), default=LoanStatus.ACTIVE)
    days_overdue = Column(Integer, default=0)
    total_fees = Column(Float, default=0.0)
    settlement_offer_percentage = Column(Float)  # If settlement offered
    settlement_expiry_date = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    borrower = relationship("Borrower", back_populates="loans")
    transactions = relationship("Transaction", back_populates="loan")
    conversations = relationship("Conversation", back_populates="loan")
    payment_plans = relationship("PaymentPlan", back_populates="loan")


class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    loan_id = Column(Integer, ForeignKey("loans.id"), nullable=False)
    transaction_id = Column(String(100), unique=True, index=True)
    amount = Column(Float, nullable=False)
    transaction_type = Column(Enum(TransactionType), nullable=False)
    description = Column(Text)
    payment_method = Column(String(50))  # credit_card, bank_transfer, check, etc.
    external_reference = Column(String(100))  # Stripe transaction ID, etc.
    processed_at = Column(DateTime, server_default=func.now())
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    loan = relationship("Loan", back_populates="transactions")


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(100), unique=True, index=True)
    borrower_id = Column(Integer, ForeignKey("borrowers.id"), nullable=False)
    loan_id = Column(Integer, ForeignKey("loans.id"), nullable=False)
    state = Column(Enum(ConversationState), default=ConversationState.INITIATED)
    channel = Column(String(20), nullable=False)  # chat, sms, email, voice
    session_data = Column(JSON)  # Store session context and variables
    identity_verified = Column(Boolean, default=False)
    verification_attempts = Column(Integer, default=0)
    last_activity = Column(DateTime, server_default=func.now())
    assigned_agent_id = Column(String(100))  # Human agent if escalated
    escalation_reason = Column(Text)
    escalation_date = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    borrower = relationship("Borrower", back_populates="conversations")
    loan = relationship("Loan", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    message_type = Column(Enum(MessageType), nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSON)  # Store LLM response data, confidence scores, etc.
    tokens_used = Column(Integer)
    processing_time = Column(Float)  # Response time in seconds
    confidence_score = Column(Float)
    compliance_flags = Column(JSON)  # Store compliance check results
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class PaymentPlan(Base):
    __tablename__ = "payment_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    loan_id = Column(Integer, ForeignKey("loans.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    plan_type = Column(Enum(PaymentPlanType), nullable=False)
    total_amount = Column(Float, nullable=False)
    installment_amount = Column(Float)
    number_of_installments = Column(Integer)
    first_payment_date = Column(DateTime)
    payment_frequency = Column(String(20), default="monthly")  # weekly, monthly, bi-weekly
    status = Column(String(20), default="proposed")  # proposed, accepted, active, completed, defaulted
    acceptance_date = Column(DateTime)
    completion_date = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    loan = relationship("Loan", back_populates="payment_plans")
    scheduled_payments = relationship("ScheduledPayment", back_populates="payment_plan")


class ScheduledPayment(Base):
    __tablename__ = "scheduled_payments"
    
    id = Column(Integer, primary_key=True, index=True)
    payment_plan_id = Column(Integer, ForeignKey("payment_plans.id"), nullable=False)
    installment_number = Column(Integer, nullable=False)
    due_date = Column(DateTime, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String(20), default="pending")  # pending, paid, overdue, skipped
    paid_date = Column(DateTime)
    paid_amount = Column(Float)
    transaction_id = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    payment_plan = relationship("PaymentPlan", back_populates="scheduled_payments")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(100), index=True)
    loan_id = Column(Integer, index=True)
    borrower_id = Column(Integer, index=True)
    action = Column(String(100), nullable=False)
    actor = Column(String(100), nullable=False)  # system, agent_id, user
    details = Column(JSON)
    compliance_data = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    timestamp = Column(DateTime, server_default=func.now())


class ComplianceEvent(Base):
    __tablename__ = "compliance_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(100), nullable=False)
    conversation_id = Column(String(100), index=True)
    loan_id = Column(Integer, index=True)
    borrower_id = Column(Integer, index=True)
    severity = Column(String(20), default="info")  # info, warning, error, critical
    description = Column(Text)
    automated_action = Column(String(100))  # Action taken by system
    requires_review = Column(Boolean, default=False)
    reviewed_by = Column(String(100))
    reviewed_at = Column(DateTime)
    metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())


class VectorDocument(Base):
    __tablename__ = "vector_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(100), unique=True, index=True)
    document_type = Column(String(50), nullable=False)  # policy, borrower_profile, communication, regulation
    source = Column(String(100))
    title = Column(String(255))
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, default=0)
    metadata = Column(JSON)
    loan_id = Column(Integer, ForeignKey("loans.id"))
    borrower_id = Column(Integer, ForeignKey("borrowers.id"))
    embedding_model = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
