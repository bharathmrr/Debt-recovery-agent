import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from ..models.schemas import (
    Conversation, Message, Borrower, Loan, PaymentPlan, ScheduledPayment,
    Transaction, AuditLog, ComplianceEvent, ConversationState, MessageType,
    PaymentPlanType, TransactionType
)
from ..models.pydantic_models import (
    ConversationResponse, AssistantResponse, ConversationChannel,
    ActionType, BorrowerInfo, LoanInfo, ConversationInfo, MessageInfo
)
from ..services.llm_service import llm_service
from ..services.compliance_service import compliance_service
from ..utils.logging_config import (
    get_logger, log_compliance_event, log_conversation_event,
    log_payment_event, log_escalation_event
)

logger = get_logger(__name__)


class ConversationService:
    """
    Core conversation orchestration service
    
    Handles:
    - Conversation lifecycle management
    - Message processing and routing
    - Identity verification
    - Payment processing coordination
    - Escalation management
    - Compliance monitoring
    """
    
    def __init__(self):
        self.max_verification_attempts = 3
        self.session_timeout_minutes = 30
        
    async def process_message(self, user_message: str, loan_id: int, session_id: str,
                            channel: ConversationChannel, metadata: Dict[str, Any] = None,
                            db: Session = None) -> ConversationResponse:
        """
        Process a user message and generate appropriate response
        
        Args:
            user_message: The user's input message
            loan_id: Associated loan ID
            session_id: Session identifier
            channel: Communication channel (chat, sms, email, voice)
            metadata: Additional metadata
            db: Database session
        
        Returns:
            ConversationResponse with assistant response and next steps
        """
        try:
            # Get or create conversation
            conversation = await self._get_or_create_conversation(
                loan_id, session_id, channel, db
            )
            
            # Check if conversation is in valid state
            if conversation.state == ConversationState.OPTED_OUT:
                return self._create_opted_out_response(conversation.conversation_id)
            
            # Get borrower and loan information
            loan = db.query(Loan).filter(Loan.id == loan_id).first()
            if not loan:
                raise ValueError(f"Loan {loan_id} not found")
            
            borrower = db.query(Borrower).filter(Borrower.id == loan.borrower_id).first()
            if not borrower:
                raise ValueError(f"Borrower for loan {loan_id} not found")
            
            # Check compliance before processing
            compliance_check = await compliance_service.check_contact_compliance(
                borrower, conversation, db
            )
            
            if not compliance_check["allowed"]:
                return self._create_compliance_blocked_response(
                    conversation.conversation_id, compliance_check["reason"]
                )
            
            # Save user message
            user_msg = Message(
                conversation_id=conversation.id,
                message_type=MessageType.USER,
                content=user_message,
                metadata=metadata or {}
            )
            db.add(user_msg)
            
            # Build conversation context
            conversation_context = await self._build_conversation_context(conversation, db)
            
            # Process with LLM
            assistant_response = llm_service.process_conversation(
                user_message=user_message,
                conversation_context=conversation_context,
                loan_id=loan_id,
                borrower_id=borrower.id
            )
            
            # Handle specific actions
            response = await self._handle_assistant_action(
                assistant_response, conversation, loan, borrower, db
            )
            
            # Save assistant message
            assistant_msg = Message(
                conversation_id=conversation.id,
                message_type=MessageType.ASSISTANT,
                content=assistant_response.message_to_user,
                metadata=assistant_response.dict(),
                confidence_score=assistant_response.confidence,
                compliance_flags=assistant_response.compliance_checks
            )
            db.add(assistant_msg)
            
            # Update conversation state
            await self._update_conversation_state(conversation, assistant_response, db)
            
            # Log compliance events
            await self._log_compliance_events(
                conversation, assistant_response, borrower.id, loan_id
            )
            
            db.commit()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            db.rollback()
            raise
    
    async def _get_or_create_conversation(self, loan_id: int, session_id: str,
                                        channel: ConversationChannel,
                                        db: Session) -> Conversation:
        """Get existing conversation or create new one"""
        
        # Try to find existing active conversation
        conversation = db.query(Conversation).filter(
            and_(
                Conversation.loan_id == loan_id,
                Conversation.conversation_id == session_id,
                Conversation.state.notin_([ConversationState.CLOSED, ConversationState.OPTED_OUT])
            )
        ).first()
        
        if conversation:
            # Update last activity
            conversation.last_activity = datetime.utcnow()
            return conversation
        
        # Create new conversation
        loan = db.query(Loan).filter(Loan.id == loan_id).first()
        if not loan:
            raise ValueError(f"Loan {loan_id} not found")
        
        conversation = Conversation(
            conversation_id=session_id or str(uuid.uuid4()),
            borrower_id=loan.borrower_id,
            loan_id=loan_id,
            state=ConversationState.INITIATED,
            channel=channel.value,
            session_data={},
            identity_verified=False,
            verification_attempts=0
        )
        
        db.add(conversation)
        db.flush()  # Get the ID
        
        logger.info(f"Created new conversation: {conversation.conversation_id}")
        return conversation
    
    async def _build_conversation_context(self, conversation: Conversation,
                                        db: Session) -> Dict[str, Any]:
        """Build conversation context for LLM processing"""
        
        # Get recent messages
        recent_messages = db.query(Message).filter(
            Message.conversation_id == conversation.id
        ).order_by(desc(Message.created_at)).limit(10).all()
        
        # Format message history
        message_history = []
        for msg in reversed(recent_messages):  # Reverse to get chronological order
            message_history.append({
                'role': msg.message_type.value,
                'content': msg.content,
                'timestamp': msg.created_at.isoformat(),
                'metadata': msg.metadata
            })
        
        return {
            'conversation_id': conversation.conversation_id,
            'state': conversation.state.value,
            'identity_verified': conversation.identity_verified,
            'verification_attempts': conversation.verification_attempts,
            'session_data': conversation.session_data or {},
            'recent_messages': message_history,
            'channel': conversation.channel,
            'last_activity': conversation.last_activity.isoformat()
        }
    
    async def _handle_assistant_action(self, assistant_response: AssistantResponse,
                                     conversation: Conversation, loan: Loan,
                                     borrower: Borrower, db: Session) -> ConversationResponse:
        """Handle specific assistant actions and generate appropriate response"""
        
        requires_action = False
        next_steps = []
        
        # Handle escalation
        if assistant_response.escalation:
            await self._escalate_conversation(
                conversation, assistant_response.metadata.get('escalation_reason', 'LLM requested escalation'), db
            )
            next_steps.append("Conversation escalated to human agent")
        
        # Handle payment plan proposal
        if assistant_response.structured_plan:
            plan_id = await self._create_payment_plan(
                loan, conversation, assistant_response.structured_plan, db
            )
            requires_action = True
            next_steps.append(f"Payment plan created with ID: {plan_id}")
        
        # Handle identity verification request
        if assistant_response.action == ActionType.VERIFY_IDENTITY:
            requires_action = True
            next_steps.append("Identity verification required")
        
        # Handle payment collection
        if assistant_response.action == ActionType.COLLECT_PAYMENT:
            requires_action = True
            next_steps.append("Payment processing required")
        
        return ConversationResponse(
            ok=True,
            conversation_id=conversation.conversation_id,
            assistant=assistant_response,
            session_data=conversation.session_data,
            requires_action=requires_action,
            next_steps=next_steps
        )
    
    async def _create_payment_plan(self, loan: Loan, conversation: Conversation,
                                 structured_plan, db: Session) -> int:
        """Create a payment plan from structured plan data"""
        
        plan_type_map = {
            'installment': PaymentPlanType.INSTALLMENT,
            'settlement': PaymentPlanType.SETTLEMENT,
            'one_time': PaymentPlanType.FULL_PAYMENT
        }
        
        payment_plan = PaymentPlan(
            loan_id=loan.id,
            conversation_id=conversation.id,
            plan_type=plan_type_map[structured_plan.type.value],
            total_amount=structured_plan.amount,
            installment_amount=structured_plan.amount if structured_plan.type.value == 'one_time' else structured_plan.amount,
            number_of_installments=structured_plan.installments or 1,
            first_payment_date=datetime.fromisoformat(structured_plan.first_due_date) if structured_plan.first_due_date else datetime.utcnow() + timedelta(days=7),
            payment_frequency=structured_plan.frequency or 'monthly',
            status='proposed'
        )
        
        db.add(payment_plan)
        db.flush()
        
        # Create scheduled payments for installment plans
        if structured_plan.type.value == 'installment' and structured_plan.installments:
            await self._create_scheduled_payments(payment_plan, db)
        
        logger.info(f"Created payment plan {payment_plan.id} for loan {loan.id}")
        return payment_plan.id
    
    async def _create_scheduled_payments(self, payment_plan: PaymentPlan, db: Session):
        """Create scheduled payment records for installment plans"""
        
        current_date = payment_plan.first_payment_date
        
        for i in range(payment_plan.number_of_installments):
            scheduled_payment = ScheduledPayment(
                payment_plan_id=payment_plan.id,
                installment_number=i + 1,
                due_date=current_date,
                amount=payment_plan.installment_amount,
                status='pending'
            )
            
            db.add(scheduled_payment)
            
            # Calculate next payment date based on frequency
            if payment_plan.payment_frequency == 'weekly':
                current_date += timedelta(weeks=1)
            elif payment_plan.payment_frequency == 'bi-weekly':
                current_date += timedelta(weeks=2)
            else:  # monthly
                # Add one month (approximate)
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
    
    async def _escalate_conversation(self, conversation: Conversation, reason: str, db: Session):
        """Escalate conversation to human agent"""
        
        conversation.state = ConversationState.ESCALATED
        conversation.escalation_reason = reason
        conversation.escalation_date = datetime.utcnow()
        
        # Log escalation event
        log_escalation_event(
            conversation_id=conversation.conversation_id,
            reason=reason,
            confidence_score=0.0
        )
        
        logger.info(f"Escalated conversation {conversation.conversation_id}: {reason}")
    
    async def _update_conversation_state(self, conversation: Conversation,
                                       assistant_response: AssistantResponse, db: Session):
        """Update conversation state based on assistant response"""
        
        if assistant_response.escalation:
            conversation.state = ConversationState.ESCALATED
        elif assistant_response.action == ActionType.VERIFY_IDENTITY:
            conversation.state = ConversationState.IDENTITY_VERIFICATION
        elif assistant_response.action in [ActionType.PROPOSE_PLAN, ActionType.COLLECT_PAYMENT]:
            conversation.state = ConversationState.ACTIVE_NEGOTIATION
        elif assistant_response.action == ActionType.CLOSE:
            conversation.state = ConversationState.CLOSED
        
        conversation.last_activity = datetime.utcnow()
    
    async def _log_compliance_events(self, conversation: Conversation,
                                   assistant_response: AssistantResponse,
                                   borrower_id: int, loan_id: int):
        """Log compliance-related events"""
        
        for check in assistant_response.compliance_checks:
            log_compliance_event(
                event_type=f"compliance_check_{check}",
                details={
                    'check_name': check,
                    'conversation_id': conversation.conversation_id,
                    'action': assistant_response.action.value,
                    'confidence': assistant_response.confidence
                },
                user_id=str(borrower_id),
                loan_id=str(loan_id)
            )
    
    def _create_opted_out_response(self, conversation_id: str) -> ConversationResponse:
        """Create response for opted-out conversations"""
        
        return ConversationResponse(
            ok=False,
            conversation_id=conversation_id,
            assistant=AssistantResponse(
                action=ActionType.CLOSE,
                message_to_user="This contact has opted out of communications. No further messages will be sent.",
                confidence=1.0,
                escalation=False,
                compliance_checks=["opt_out_respected"]
            ),
            requires_action=False
        )
    
    def _create_compliance_blocked_response(self, conversation_id: str, reason: str) -> ConversationResponse:
        """Create response when contact is blocked due to compliance"""
        
        return ConversationResponse(
            ok=False,
            conversation_id=conversation_id,
            assistant=AssistantResponse(
                action=ActionType.CLOSE,
                message_to_user=f"Contact not allowed at this time: {reason}",
                confidence=1.0,
                escalation=False,
                compliance_checks=["contact_time_restriction"]
            ),
            requires_action=False
        )
    
    async def verify_identity(self, conversation_id: str, verification_data: Dict[str, str],
                            db: Session) -> Dict[str, Any]:
        """Verify borrower identity using provided data"""
        
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()
        
        if not conversation:
            raise ValueError("Conversation not found")
        
        loan = db.query(Loan).filter(Loan.id == conversation.loan_id).first()
        borrower = db.query(Borrower).filter(Borrower.id == conversation.borrower_id).first()
        
        # Check verification attempts
        if conversation.verification_attempts >= self.max_verification_attempts:
            await self._escalate_conversation(
                conversation, "Maximum verification attempts exceeded", db
            )
            db.commit()
            return {
                "verified": False,
                "message": "Maximum verification attempts exceeded. Escalating to human agent.",
                "attempts_remaining": 0
            }
        
        # Verify identity
        verification_passed = True
        
        # Check last 4 digits of SSN
        if "last_four_ssn" in verification_data:
            if verification_data["last_four_ssn"] != borrower.ssn_last_four:
                verification_passed = False
        
        # Check last payment amount
        if "last_payment_amount" in verification_data:
            try:
                provided_amount = float(verification_data["last_payment_amount"])
                if abs(provided_amount - (loan.last_payment_amount or 0)) > 0.01:
                    verification_passed = False
            except ValueError:
                verification_passed = False
        
        conversation.verification_attempts += 1
        
        if verification_passed:
            conversation.identity_verified = True
            conversation.state = ConversationState.ACTIVE_NEGOTIATION
            
            log_compliance_event(
                event_type="identity_verified",
                details={"verification_method": list(verification_data.keys())},
                user_id=str(borrower.id),
                loan_id=str(loan.id)
            )
            
            db.commit()
            return {
                "verified": True,
                "message": "Identity verified successfully.",
                "attempts_remaining": self.max_verification_attempts - conversation.verification_attempts
            }
        else:
            attempts_remaining = self.max_verification_attempts - conversation.verification_attempts
            
            log_compliance_event(
                event_type="identity_verification_failed",
                details={"attempt_number": conversation.verification_attempts},
                user_id=str(borrower.id),
                loan_id=str(loan.id)
            )
            
            db.commit()
            return {
                "verified": False,
                "message": f"Identity verification failed. {attempts_remaining} attempts remaining.",
                "attempts_remaining": attempts_remaining
            }
    
    async def process_payment(self, loan_id: int, amount: float, payment_method: str,
                            conversation_id: str, metadata: Dict[str, Any] = None,
                            db: Session = None) -> Dict[str, Any]:
        """Process a payment for a loan"""
        
        loan = db.query(Loan).filter(Loan.id == loan_id).first()
        if not loan:
            raise ValueError(f"Loan {loan_id} not found")
        
        # Create transaction record
        transaction = Transaction(
            loan_id=loan_id,
            transaction_id=str(uuid.uuid4()),
            amount=amount,
            transaction_type=TransactionType.PAYMENT,
            description=f"Payment via {payment_method}",
            payment_method=payment_method,
            metadata=metadata
        )
        
        db.add(transaction)
        
        # Update loan balance
        loan.current_balance -= amount
        loan.last_payment_date = datetime.utcnow()
        loan.last_payment_amount = amount
        
        # Log payment event
        log_payment_event(
            loan_id=str(loan_id),
            amount=amount,
            payment_method=payment_method,
            status="completed",
            transaction_id=transaction.transaction_id
        )
        
        db.commit()
        
        return {
            "status": "completed",
            "transaction_id": transaction.transaction_id,
            "new_balance": loan.current_balance,
            "message": f"Payment of ${amount:.2f} processed successfully."
        }
    
    async def escalate_to_human(self, conversation_id: str, reason: str,
                              priority: str = "normal", notes: str = None,
                              db: Session = None) -> Dict[str, Any]:
        """Escalate conversation to human agent"""
        
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()
        
        if not conversation:
            raise ValueError("Conversation not found")
        
        await self._escalate_conversation(conversation, reason, db)
        
        # Create audit log entry
        audit_log = AuditLog(
            conversation_id=conversation_id,
            loan_id=conversation.loan_id,
            borrower_id=conversation.borrower_id,
            action="escalate_to_human",
            actor="system",
            details={
                "reason": reason,
                "priority": priority,
                "notes": notes
            }
        )
        
        db.add(audit_log)
        db.commit()
        
        return {
            "status": "escalated",
            "message": "Conversation has been escalated to a human agent.",
            "priority": priority,
            "estimated_response_time": "2-4 hours" if priority == "high" else "24-48 hours"
        }
    
    async def get_conversation(self, conversation_id: str, db: Session) -> Optional[Dict[str, Any]]:
        """Get conversation details and history"""
        
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()
        
        if not conversation:
            return None
        
        # Get messages
        messages = db.query(Message).filter(
            Message.conversation_id == conversation.id
        ).order_by(Message.created_at).all()
        
        # Get loan and borrower info
        loan = db.query(Loan).filter(Loan.id == conversation.loan_id).first()
        borrower = db.query(Borrower).filter(Borrower.id == conversation.borrower_id).first()
        
        return {
            "conversation": ConversationInfo(
                id=conversation.id,
                conversation_id=conversation.conversation_id,
                state=conversation.state.value,
                channel=conversation.channel,
                identity_verified=conversation.identity_verified,
                last_activity=conversation.last_activity,
                assigned_agent_id=conversation.assigned_agent_id
            ).dict(),
            "borrower": BorrowerInfo(
                id=borrower.id,
                name=borrower.name,
                email=borrower.email,
                phone=borrower.phone,
                consent_status=borrower.consent_status.value,
                preferred_contact_method=borrower.preferred_contact_method
            ).dict(),
            "loan": LoanInfo(
                id=loan.id,
                account_number=loan.account_number,
                principal_amount=loan.principal_amount,
                current_balance=loan.current_balance,
                due_date=loan.due_date,
                status=loan.status.value,
                days_overdue=loan.days_overdue,
                last_payment_date=loan.last_payment_date,
                last_payment_amount=loan.last_payment_amount
            ).dict(),
            "messages": [
                MessageInfo(
                    id=msg.id,
                    message_type=msg.message_type.value,
                    content=msg.content,
                    confidence_score=msg.confidence_score,
                    created_at=msg.created_at
                ).dict() for msg in messages
            ]
        }
    
    async def get_loan_conversations(self, loan_id: int, db: Session) -> List[Dict[str, Any]]:
        """Get all conversations for a specific loan"""
        
        conversations = db.query(Conversation).filter(
            Conversation.loan_id == loan_id
        ).order_by(desc(Conversation.created_at)).all()
        
        return [
            ConversationInfo(
                id=conv.id,
                conversation_id=conv.conversation_id,
                state=conv.state.value,
                channel=conv.channel,
                identity_verified=conv.identity_verified,
                last_activity=conv.last_activity,
                assigned_agent_id=conv.assigned_agent_id
            ).dict() for conv in conversations
        ]
    
    # Logging helper methods
    def log_conversation_event(self, conversation_id: str, message_type: str, 
                             content: str, metadata: Dict[str, Any] = None):
        """Log conversation event"""
        log_conversation_event(conversation_id, message_type, content, metadata)
    
    def log_payment_event(self, loan_id: str, amount: float, payment_method: str,
                         status: str, transaction_id: str = None, error: str = None):
        """Log payment event"""
        log_payment_event(loan_id, amount, payment_method, status, transaction_id, error)
    
    def log_escalation_event(self, conversation_id: str, reason: str, 
                           confidence_score: float, assigned_agent: str = None):
        """Log escalation event"""
        log_escalation_event(conversation_id, reason, confidence_score, assigned_agent)


# Global conversation service instance
conversation_service = ConversationService()
