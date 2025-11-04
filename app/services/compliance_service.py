import os
from typing import Dict, Any, List, Optional
from datetime import datetime, time, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from ..models.schemas import Borrower, Conversation, ComplianceEvent, ConversationState
from ..models.pydantic_models import ComplianceCheck, ComplianceReport
from ..utils.logging_config import get_logger, log_compliance_event

logger = get_logger(__name__)


class ComplianceService:
    """
    Compliance monitoring and enforcement service
    
    Handles:
    - FDCPA compliance checking
    - Contact time restrictions
    - Communication frequency limits
    - Opt-out management
    - Regulatory compliance monitoring
    """
    
    def __init__(self):
        # Load compliance configuration from environment
        self.max_settlement_percentage = float(os.getenv("MAX_SETTLEMENT_PERCENTAGE", "0.70"))
        self.max_installment_months = int(os.getenv("MAX_INSTALLMENT_MONTHS", "12"))
        self.contact_hours_start = os.getenv("CONTACT_HOURS_START", "08:00")
        self.contact_hours_end = os.getenv("CONTACT_HOURS_END", "21:00")
        self.max_daily_contact_attempts = 3
        self.max_weekly_contact_attempts = 7
        
        # Prohibited contact days (0=Monday, 6=Sunday)
        self.prohibited_days = [6]  # Sunday
        
        logger.info("Compliance Service initialized with FDCPA guidelines")
    
    async def check_contact_compliance(self, borrower: Borrower, conversation: Conversation,
                                     db: Session) -> Dict[str, Any]:
        """
        Comprehensive compliance check before allowing contact
        
        Returns:
            Dict with 'allowed' boolean and 'reason' if blocked
        """
        
        checks = []
        
        # Check opt-out status
        opt_out_check = self._check_opt_out_status(borrower)
        checks.append(opt_out_check)
        if not opt_out_check.passed:
            return {"allowed": False, "reason": "Borrower has opted out of communications"}
        
        # Check contact time restrictions
        time_check = self._check_contact_time(borrower)
        checks.append(time_check)
        if not time_check.passed:
            return {"allowed": False, "reason": time_check.details}
        
        # Check daily contact frequency
        daily_check = await self._check_daily_contact_frequency(borrower, db)
        checks.append(daily_check)
        if not daily_check.passed:
            return {"allowed": False, "reason": daily_check.details}
        
        # Check weekly contact frequency
        weekly_check = await self._check_weekly_contact_frequency(borrower, db)
        checks.append(weekly_check)
        if not weekly_check.passed:
            return {"allowed": False, "reason": weekly_check.details}
        
        # Log compliance check
        await self._log_compliance_check(borrower.id, checks)
        
        return {"allowed": True, "reason": None, "checks": checks}
    
    def _check_opt_out_status(self, borrower: Borrower) -> ComplianceCheck:
        """Check if borrower has opted out of communications"""
        
        if borrower.opt_out_date:
            return ComplianceCheck(
                check_name="opt_out_status",
                passed=False,
                details=f"Borrower opted out on {borrower.opt_out_date}",
                severity="critical"
            )
        
        return ComplianceCheck(
            check_name="opt_out_status",
            passed=True,
            details="Borrower has not opted out"
        )
    
    def _check_contact_time(self, borrower: Borrower) -> ComplianceCheck:
        """Check if current time is within allowed contact hours"""
        
        now = datetime.now()
        
        # Check if today is a prohibited day
        if now.weekday() in self.prohibited_days:
            return ComplianceCheck(
                check_name="contact_time",
                passed=False,
                details="Contact not allowed on Sundays",
                severity="warning"
            )
        
        # Parse contact hours
        try:
            start_time = time.fromisoformat(self.contact_hours_start)
            end_time = time.fromisoformat(self.contact_hours_end)
            current_time = now.time()
            
            # Check if current time is within allowed hours
            if not (start_time <= current_time <= end_time):
                return ComplianceCheck(
                    check_name="contact_time",
                    passed=False,
                    details=f"Contact only allowed between {self.contact_hours_start} and {self.contact_hours_end}",
                    severity="warning"
                )
        
        except ValueError as e:
            logger.error(f"Error parsing contact hours: {e}")
            # Default to allowing contact if configuration is invalid
        
        return ComplianceCheck(
            check_name="contact_time",
            passed=True,
            details="Contact time is within allowed hours"
        )
    
    async def _check_daily_contact_frequency(self, borrower: Borrower, db: Session) -> ComplianceCheck:
        """Check daily contact attempt limits"""
        
        today = datetime.now().date()
        today_start = datetime.combine(today, time.min)
        today_end = datetime.combine(today, time.max)
        
        # Count conversations started today
        daily_conversations = db.query(Conversation).filter(
            and_(
                Conversation.borrower_id == borrower.id,
                Conversation.created_at >= today_start,
                Conversation.created_at <= today_end
            )
        ).count()
        
        if daily_conversations >= self.max_daily_contact_attempts:
            return ComplianceCheck(
                check_name="daily_contact_frequency",
                passed=False,
                details=f"Maximum daily contact attempts ({self.max_daily_contact_attempts}) exceeded",
                severity="warning"
            )
        
        return ComplianceCheck(
            check_name="daily_contact_frequency",
            passed=True,
            details=f"Daily contact attempts: {daily_conversations}/{self.max_daily_contact_attempts}"
        )
    
    async def _check_weekly_contact_frequency(self, borrower: Borrower, db: Session) -> ComplianceCheck:
        """Check weekly contact attempt limits"""
        
        week_start = datetime.now() - timedelta(days=7)
        
        # Count conversations in the last 7 days
        weekly_conversations = db.query(Conversation).filter(
            and_(
                Conversation.borrower_id == borrower.id,
                Conversation.created_at >= week_start
            )
        ).count()
        
        if weekly_conversations >= self.max_weekly_contact_attempts:
            return ComplianceCheck(
                check_name="weekly_contact_frequency",
                passed=False,
                details=f"Maximum weekly contact attempts ({self.max_weekly_contact_attempts}) exceeded",
                severity="warning"
            )
        
        return ComplianceCheck(
            check_name="weekly_contact_frequency",
            passed=True,
            details=f"Weekly contact attempts: {weekly_conversations}/{self.max_weekly_contact_attempts}"
        )
    
    async def validate_payment_plan(self, plan_data: Dict[str, Any], loan_balance: float) -> List[ComplianceCheck]:
        """Validate payment plan against compliance rules"""
        
        checks = []
        
        # Check settlement percentage
        if plan_data.get("type") == "settlement":
            settlement_amount = plan_data.get("amount", 0)
            settlement_percentage = settlement_amount / loan_balance if loan_balance > 0 else 0
            
            if settlement_percentage > self.max_settlement_percentage:
                checks.append(ComplianceCheck(
                    check_name="settlement_percentage",
                    passed=False,
                    details=f"Settlement percentage {settlement_percentage:.1%} exceeds maximum {self.max_settlement_percentage:.1%}",
                    severity="error"
                ))
            else:
                checks.append(ComplianceCheck(
                    check_name="settlement_percentage",
                    passed=True,
                    details=f"Settlement percentage {settlement_percentage:.1%} is within limits"
                ))
        
        # Check installment plan duration
        if plan_data.get("type") == "installment":
            installments = plan_data.get("installments", 0)
            
            if installments > self.max_installment_months:
                checks.append(ComplianceCheck(
                    check_name="installment_duration",
                    passed=False,
                    details=f"Installment plan duration {installments} months exceeds maximum {self.max_installment_months} months",
                    severity="error"
                ))
            else:
                checks.append(ComplianceCheck(
                    check_name="installment_duration",
                    passed=True,
                    details=f"Installment plan duration {installments} months is within limits"
                ))
            
            # Check minimum payment amount
            payment_amount = plan_data.get("amount", 0)
            if payment_amount < 25:
                checks.append(ComplianceCheck(
                    check_name="minimum_payment",
                    passed=False,
                    details=f"Payment amount ${payment_amount} is below minimum $25",
                    severity="error"
                ))
            else:
                checks.append(ComplianceCheck(
                    check_name="minimum_payment",
                    passed=True,
                    details=f"Payment amount ${payment_amount} meets minimum requirements"
                ))
        
        return checks
    
    async def check_message_compliance(self, message: str, conversation_context: Dict[str, Any]) -> List[ComplianceCheck]:
        """Check message content for compliance violations"""
        
        checks = []
        
        # Check for prohibited language
        prohibited_phrases = [
            "threaten", "sue", "arrest", "jail", "garnish", "seize",
            "ruin credit", "legal action", "court", "lawsuit"
        ]
        
        message_lower = message.lower()
        found_violations = []
        
        for phrase in prohibited_phrases:
            if phrase in message_lower:
                found_violations.append(phrase)
        
        if found_violations:
            checks.append(ComplianceCheck(
                check_name="prohibited_language",
                passed=False,
                details=f"Prohibited language detected: {', '.join(found_violations)}",
                severity="critical"
            ))
        else:
            checks.append(ComplianceCheck(
                check_name="prohibited_language",
                passed=True,
                details="No prohibited language detected"
            ))
        
        # Check for required disclosures
        if not conversation_context.get("identity_verified", False):
            if any(keyword in message_lower for keyword in ["balance", "amount", "payment", "$"]):
                checks.append(ComplianceCheck(
                    check_name="identity_verification",
                    passed=False,
                    details="Account information shared without identity verification",
                    severity="error"
                ))
        
        # Check for debt collector identification
        if "debt collector" not in message_lower and "collection" not in message_lower:
            checks.append(ComplianceCheck(
                check_name="debt_collector_identification",
                passed=False,
                details="Message should identify as debt collection communication",
                severity="warning"
            ))
        
        return checks
    
    async def handle_opt_out_request(self, borrower_id: int, conversation_id: str, db: Session) -> Dict[str, Any]:
        """Handle borrower opt-out request"""
        
        borrower = db.query(Borrower).filter(Borrower.id == borrower_id).first()
        if not borrower:
            raise ValueError(f"Borrower {borrower_id} not found")
        
        # Update borrower opt-out status
        borrower.opt_out_date = datetime.utcnow()
        
        # Close all active conversations
        active_conversations = db.query(Conversation).filter(
            and_(
                Conversation.borrower_id == borrower_id,
                Conversation.state.notin_([ConversationState.CLOSED, ConversationState.OPTED_OUT])
            )
        ).all()
        
        for conv in active_conversations:
            conv.state = ConversationState.OPTED_OUT
        
        # Log compliance event
        log_compliance_event(
            event_type="borrower_opt_out",
            details={
                "conversation_id": conversation_id,
                "opt_out_date": borrower.opt_out_date.isoformat()
            },
            user_id=str(borrower_id)
        )
        
        db.commit()
        
        logger.info(f"Borrower {borrower_id} opted out of communications")
        
        return {
            "status": "opted_out",
            "message": "You have been successfully removed from our contact list.",
            "opt_out_date": borrower.opt_out_date.isoformat()
        }
    
    async def handle_debt_validation_request(self, borrower_id: int, loan_id: int, 
                                           conversation_id: str, db: Session) -> Dict[str, Any]:
        """Handle debt validation request"""
        
        # Log compliance event
        log_compliance_event(
            event_type="debt_validation_requested",
            details={
                "conversation_id": conversation_id,
                "request_date": datetime.utcnow().isoformat()
            },
            user_id=str(borrower_id),
            loan_id=str(loan_id)
        )
        
        # Pause collection activities
        active_conversations = db.query(Conversation).filter(
            and_(
                Conversation.borrower_id == borrower_id,
                Conversation.loan_id == loan_id,
                Conversation.state == ConversationState.ACTIVE_NEGOTIATION
            )
        ).all()
        
        for conv in active_conversations:
            conv.state = ConversationState.ESCALATED
            conv.escalation_reason = "Debt validation requested"
            conv.escalation_date = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Debt validation requested for loan {loan_id}, borrower {borrower_id}")
        
        return {
            "status": "validation_requested",
            "message": "Your debt validation request has been received. Collection activities have been paused pending validation.",
            "next_steps": "You will receive validation documentation within 30 days as required by law."
        }
    
    async def generate_compliance_report(self, conversation_id: str, db: Session) -> ComplianceReport:
        """Generate comprehensive compliance report for a conversation"""
        
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()
        
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Get all compliance events for this conversation
        compliance_events = db.query(ComplianceEvent).filter(
            ComplianceEvent.conversation_id == conversation_id
        ).order_by(ComplianceEvent.created_at).all()
        
        checks = []
        overall_status = "passed"
        requires_review = False
        
        for event in compliance_events:
            severity = event.severity or "info"
            passed = severity not in ["error", "critical"]
            
            if not passed:
                overall_status = "failed" if severity == "critical" else "warning"
                requires_review = True
            
            checks.append(ComplianceCheck(
                check_name=event.event_type,
                passed=passed,
                details=event.description or "No details available",
                severity=severity
            ))
        
        return ComplianceReport(
            conversation_id=conversation_id,
            checks=checks,
            overall_status=overall_status,
            requires_human_review=requires_review,
            timestamp=datetime.utcnow()
        )
    
    async def _log_compliance_check(self, borrower_id: int, checks: List[ComplianceCheck]):
        """Log compliance check results"""
        
        for check in checks:
            if not check.passed:
                log_compliance_event(
                    event_type=f"compliance_violation_{check.check_name}",
                    details={
                        "check_name": check.check_name,
                        "details": check.details,
                        "severity": check.severity
                    },
                    user_id=str(borrower_id)
                )
    
    def get_compliance_config(self) -> Dict[str, Any]:
        """Get current compliance configuration"""
        
        return {
            "max_settlement_percentage": self.max_settlement_percentage,
            "max_installment_months": self.max_installment_months,
            "contact_hours_start": self.contact_hours_start,
            "contact_hours_end": self.contact_hours_end,
            "max_daily_contact_attempts": self.max_daily_contact_attempts,
            "max_weekly_contact_attempts": self.max_weekly_contact_attempts,
            "prohibited_days": self.prohibited_days
        }


# Global compliance service instance
compliance_service = ComplianceService()
