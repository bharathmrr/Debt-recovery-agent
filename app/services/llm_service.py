import os
import json
import time
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

from ..models.pydantic_models import AssistantResponse, StructuredPlan, ActionType, PlanType
from ..services.rag_service import rag_service
from ..utils.logging_config import get_logger, log_llm_interaction, log_compliance_event

logger = get_logger(__name__)


class LLMService:
    """
    Large Language Model service for debt recovery conversations
    
    Handles:
    - Conversation management with ChatGroq
    - Prompt engineering and context assembly
    - Response parsing and validation
    - Compliance checking
    - Cost tracking and monitoring
    """
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.45,
            api_key=self.api_key,
            max_tokens=1000,
            timeout=30
        )
        
        # Load prompt templates
        self.system_prompt = self._load_system_prompt()
        self.few_shot_examples = self._load_few_shot_examples()
        
        logger.info("LLM Service initialized with ChatGroq")
    
    def _load_system_prompt(self) -> str:
        """Load the system prompt template"""
        return """You are "Collecta", an AI Debt Recovery Agent designed to help borrowers resolve their outstanding debts in a professional, empathetic, and compliant manner.

CORE PRINCIPLES:
- Be professional, respectful, empathetic, and concise
- Always prioritize compliance with debt collection regulations
- Focus on finding mutually beneficial solutions
- Maintain borrower dignity while protecting company interests

STRICT RULES:
1. ALWAYS verify borrower identity before sharing account information
2. Use ONLY the retrieved context for factual information (balances, dates, payments)
3. NEVER request sensitive data like full SSN - use secure channels for such information
4. Offer payment options within policy limits only
5. If borrower expresses distress, suicidal ideation, or requests to stop communication, IMMEDIATELY escalate to human agent
6. NEVER use threatening, abusive, or misleading language
7. Respect contact time restrictions and borrower preferences
8. Provide clear debt validation information when requested

AVAILABLE ACTIONS:
- inform: Provide account information or general guidance
- collect_payment: Request immediate payment
- propose_plan: Offer structured payment arrangements
- acknowledge: Confirm borrower statements or agreements
- request_info: Ask for additional information needed
- verify_identity: Request identity verification
- escalate: Transfer to human agent
- close: End conversation appropriately

PAYMENT OPTIONS (within policy):
- Full payment of outstanding balance
- Installment plans (up to 12 months, minimum $25/payment)
- Settlement offers (up to 70% of balance)

ESCALATION TRIGGERS:
- Identity verification failures (3+ attempts)
- Debt validation requests
- Bankruptcy or attorney mentions
- Threats of legal action against company
- Borrower distress or harassment claims
- Low confidence in response (<0.5)
- Complex negotiations beyond policy limits

COMPLIANCE REQUIREMENTS:
- Document all interactions for audit trail
- Mask PII in all logs and communications
- Verify contact permissions and time restrictions
- Provide required disclosures and notices
- Maintain professional tone at all times

OUTPUT FORMAT:
You MUST respond with a JSON object followed by a natural language message. The JSON must follow this exact schema:

{
  "action": "inform|collect_payment|propose_plan|acknowledge|request_info|verify_identity|escalate|close",
  "message_to_user": "<professional message text>",
  "structured_plan": {
    "type": "installment|settlement|one_time",
    "amount": <number>,
    "installments": <number or null>,
    "first_due_date": "YYYY-MM-DD"
  } | null,
  "confidence": <0.0-1.0>,
  "escalation": <true|false>,
  "compliance_checks": ["<check1>", "<check2>", ...]
}

IMPORTANT: Always include compliance_checks array with relevant verification steps taken."""

    def _load_few_shot_examples(self) -> str:
        """Load few-shot examples for better LLM performance"""
        return """
EXAMPLE 1 - Payment Plan Proposal:
Context: Balance $1200, last payment 2025-06-01, overdue 90 days, policy allows 6-month installments
User: "I can pay $200 now and then monthly payments"

Response JSON:
{
  "action": "propose_plan",
  "message_to_user": "Thank you for your willingness to resolve this account. I can set up a 6-month payment plan with $200 monthly payments starting November 10th, 2025. This would fully resolve your $1,200 balance. Would this arrangement work for you?",
  "structured_plan": {
    "type": "installment",
    "amount": 200,
    "installments": 6,
    "first_due_date": "2025-11-10"
  },
  "confidence": 0.92,
  "escalation": false,
  "compliance_checks": ["verified_contact_time", "payment_plan_within_policy", "borrower_consent_confirmed"]
}

EXAMPLE 2 - Escalation Required:
Context: User threatens legal action and demands full debt validation
User: "This debt is not mine! I'm calling my lawyer and filing complaints with the CFPB!"

Response JSON:
{
  "action": "escalate",
  "message_to_user": "I understand your concerns about this account. I'm immediately connecting you with a specialist who can provide the debt validation documentation you're requesting. They will contact you within 48 hours with all required information. In the meantime, we'll pause all collection activities on this account.",
  "structured_plan": null,
  "confidence": 0.98,
  "escalation": true,
  "compliance_checks": ["debt_validation_requested", "legal_threat_documented", "collection_pause_initiated"]
}

EXAMPLE 3 - Identity Verification:
Context: New conversation, borrower identity not yet verified
User: "What's my balance?"

Response JSON:
{
  "action": "verify_identity",
  "message_to_user": "I'd be happy to help you with your account information. For security purposes, I need to verify your identity first. Can you please provide the last 4 digits of your Social Security Number and the amount of your last payment?",
  "structured_plan": null,
  "confidence": 0.95,
  "escalation": false,
  "compliance_checks": ["identity_verification_required", "pii_protection_maintained"]
}

EXAMPLE 4 - Settlement Offer:
Context: Balance $2000, account 180+ days overdue, borrower offers lump sum
User: "I can pay $1000 right now to settle this completely"

Response JSON:
{
  "action": "propose_plan",
  "message_to_user": "Thank you for your settlement offer. I can accept $1,400 as a full settlement of your $2,000 balance (70% settlement per our policy). This would completely resolve the account. If you agree, I can send you a secure payment link to process this today. Would you like to proceed?",
  "structured_plan": {
    "type": "settlement",
    "amount": 1400,
    "installments": null,
    "first_due_date": "2025-11-05"
  },
  "confidence": 0.88,
  "escalation": false,
  "compliance_checks": ["settlement_within_policy", "written_agreement_required", "payment_processing_secure"]
}
"""

    def process_conversation(self, user_message: str, conversation_context: Dict[str, Any], 
                           loan_id: int, borrower_id: int) -> AssistantResponse:
        """
        Process a user message and generate an appropriate response
        
        Args:
            user_message: The user's input message
            conversation_context: Current conversation state and history
            loan_id: Associated loan ID
            borrower_id: Associated borrower ID
        
        Returns:
            AssistantResponse with action, message, and metadata
        """
        start_time = time.time()
        conversation_id = conversation_context.get('conversation_id', 'unknown')
        
        try:
            # Get RAG context
            rag_context = self._get_rag_context(user_message, loan_id, borrower_id)
            
            # Build the complete prompt
            full_prompt = self._build_prompt(user_message, conversation_context, rag_context)
            
            # Call LLM
            response = self._call_llm(full_prompt)
            
            # Parse and validate response
            parsed_response = self._parse_llm_response(response)
            
            # Validate against business rules
            validated_response = self._validate_response(parsed_response, conversation_context)
            
            # Log the interaction
            response_time = time.time() - start_time
            self._log_llm_interaction(conversation_id, full_prompt, response, response_time)
            
            return validated_response
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            
            # Return safe fallback response
            return self._get_fallback_response(conversation_id)
    
    def _get_rag_context(self, user_message: str, loan_id: int, borrower_id: int) -> Dict[str, Any]:
        """Retrieve relevant context using RAG"""
        try:
            # Get borrower and loan context
            borrower_context = rag_service.get_borrower_context(loan_id, borrower_id)
            
            # Get relevant policies based on user message
            policy_query = self._extract_policy_query(user_message)
            policy_context = rag_service.get_policy_context(policy_query, top_k=2)
            
            return {
                'borrower_context': borrower_context,
                'policy_context': policy_context
            }
            
        except Exception as e:
            logger.error(f"Error retrieving RAG context: {e}")
            return {'borrower_context': {}, 'policy_context': []}
    
    def _extract_policy_query(self, user_message: str) -> str:
        """Extract relevant policy query from user message"""
        # Simple keyword-based policy mapping
        policy_keywords = {
            'payment plan': 'payment plan installment policy',
            'settlement': 'settlement offer policy',
            'dispute': 'debt validation dispute policy',
            'lawyer': 'legal escalation policy',
            'bankruptcy': 'bankruptcy escalation policy',
            'harassment': 'harassment complaint policy',
            'stop': 'opt out communication policy'
        }
        
        user_lower = user_message.lower()
        for keyword, policy_query in policy_keywords.items():
            if keyword in user_lower:
                return policy_query
        
        return 'general debt collection policy'
    
    def _build_prompt(self, user_message: str, conversation_context: Dict[str, Any], 
                     rag_context: Dict[str, Any]) -> str:
        """Build the complete prompt for the LLM"""
        
        # Extract borrower information
        borrower_info = rag_context.get('borrower_context', {}).get('borrower', {})
        
        # Format RAG context
        rag_text = self._format_rag_context(rag_context)
        
        # Build conversation history
        conversation_history = self._format_conversation_history(conversation_context)
        
        # Current context
        current_context = f"""
CURRENT CONVERSATION:
Borrower: {borrower_info.get('name', 'Unknown')}
Account: {borrower_info.get('account_number', 'Unknown')}
Balance: ${borrower_info.get('current_balance', 0):.2f}
Days Overdue: {borrower_info.get('days_overdue', 0)}
Last Payment: {borrower_info.get('last_payment_date', 'None')} - ${borrower_info.get('last_payment_amount', 0):.2f}
Status: {borrower_info.get('status', 'Unknown')}
Identity Verified: {conversation_context.get('identity_verified', False)}
"""
        
        # Assemble full prompt
        full_prompt = f"""{self.system_prompt}

{self.few_shot_examples}

RETRIEVED CONTEXT:
{rag_text}

{current_context}

{conversation_history}

USER MESSAGE: "{user_message}"

Respond with the required JSON format followed by your message to the user."""
        
        return full_prompt
    
    def _format_rag_context(self, rag_context: Dict[str, Any]) -> str:
        """Format RAG context for inclusion in prompt"""
        context_parts = []
        
        # Add relevant documents
        for doc in rag_context.get('borrower_context', {}).get('relevant_documents', []):
            context_parts.append(f"- {doc['type']}: {doc['content'][:200]}...")
        
        # Add policy context
        for policy in rag_context.get('policy_context', []):
            context_parts.append(f"- Policy: {policy['content'][:200]}...")
        
        return '\n'.join(context_parts) if context_parts else "No additional context available."
    
    def _format_conversation_history(self, conversation_context: Dict[str, Any]) -> str:
        """Format recent conversation history"""
        messages = conversation_context.get('recent_messages', [])
        if not messages:
            return "CONVERSATION HISTORY: This is the start of the conversation."
        
        history_parts = ["CONVERSATION HISTORY:"]
        for msg in messages[-5:]:  # Last 5 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:100]  # Truncate long messages
            history_parts.append(f"{role.upper()}: {content}")
        
        return '\n'.join(history_parts)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the ChatGroq LLM with the assembled prompt"""
        try:
            messages = [SystemMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> AssistantResponse:
        """Parse the LLM response and extract JSON"""
        try:
            # Extract JSON block from response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if not json_match:
                # Try to find JSON without code blocks
                json_match = re.search(r'(\{[^}]*"action"[^}]*\})', response, re.DOTALL)
            
            if not json_match:
                raise ValueError("No valid JSON found in LLM response")
            
            json_str = json_match.group(1)
            parsed_json = json.loads(json_str)
            
            # Convert to AssistantResponse
            structured_plan = None
            if parsed_json.get('structured_plan'):
                plan_data = parsed_json['structured_plan']
                structured_plan = StructuredPlan(
                    type=PlanType(plan_data['type']),
                    amount=plan_data['amount'],
                    installments=plan_data.get('installments'),
                    first_due_date=plan_data.get('first_due_date'),
                    frequency=plan_data.get('frequency', 'monthly')
                )
            
            return AssistantResponse(
                action=ActionType(parsed_json['action']),
                message_to_user=parsed_json['message_to_user'],
                structured_plan=structured_plan,
                confidence=parsed_json.get('confidence', 0.5),
                escalation=parsed_json.get('escalation', False),
                compliance_checks=parsed_json.get('compliance_checks', []),
                metadata={'raw_response': response}
            )
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Raw response: {response}")
            raise
    
    def _validate_response(self, response: AssistantResponse, 
                          conversation_context: Dict[str, Any]) -> AssistantResponse:
        """Validate response against business rules and compliance"""
        
        validation_errors = []
        
        # Validate structured plan if present
        if response.structured_plan:
            plan_errors = self._validate_payment_plan(response.structured_plan)
            validation_errors.extend(plan_errors)
        
        # Check compliance requirements
        compliance_errors = self._check_compliance(response, conversation_context)
        validation_errors.extend(compliance_errors)
        
        # If validation fails and not already escalating, force escalation
        if validation_errors and not response.escalation:
            logger.warning(f"Validation errors found: {validation_errors}")
            
            response.action = ActionType.ESCALATE
            response.escalation = True
            response.message_to_user = ("I need to connect you with a specialist who can better "
                                     "assist with your request. They will contact you shortly.")
            response.structured_plan = None
            response.compliance_checks.append("validation_failed")
        
        return response
    
    def _validate_payment_plan(self, plan: StructuredPlan) -> List[str]:
        """Validate payment plan against business rules"""
        errors = []
        
        # Check settlement percentage
        if plan.type == PlanType.SETTLEMENT:
            max_settlement = float(os.getenv("MAX_SETTLEMENT_PERCENTAGE", "0.70"))
            # Note: We'd need the original balance to calculate percentage
            # This is a simplified check
            if plan.amount <= 0:
                errors.append("Settlement amount must be positive")
        
        # Check installment limits
        if plan.type == PlanType.INSTALLMENT:
            max_months = int(os.getenv("MAX_INSTALLMENT_MONTHS", "12"))
            if plan.installments and plan.installments > max_months:
                errors.append(f"Installment plan exceeds maximum {max_months} months")
            
            if plan.amount < 25:
                errors.append("Minimum payment amount is $25")
        
        return errors
    
    def _check_compliance(self, response: AssistantResponse, 
                         conversation_context: Dict[str, Any]) -> List[str]:
        """Check response for compliance violations"""
        errors = []
        
        # Check if identity verification is required
        if not conversation_context.get('identity_verified', False):
            if response.action in [ActionType.INFORM, ActionType.COLLECT_PAYMENT, ActionType.PROPOSE_PLAN]:
                if 'balance' in response.message_to_user.lower() or '$' in response.message_to_user:
                    errors.append("Cannot share account details without identity verification")
        
        # Check for prohibited language
        prohibited_words = ['threaten', 'sue', 'arrest', 'jail', 'garnish', 'seize']
        message_lower = response.message_to_user.lower()
        for word in prohibited_words:
            if word in message_lower:
                errors.append(f"Prohibited language detected: {word}")
        
        return errors
    
    def _log_llm_interaction(self, conversation_id: str, prompt: str, 
                           response: str, response_time: float):
        """Log LLM interaction for monitoring and cost tracking"""
        
        # Estimate token usage (rough approximation)
        prompt_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        completion_tokens = len(response.split()) * 1.3
        
        log_llm_interaction(
            conversation_id=conversation_id,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            model="llama-3.1-8b-instant",
            response_time=response_time
        )
    
    def _get_fallback_response(self, conversation_id: str) -> AssistantResponse:
        """Generate a safe fallback response when LLM fails"""
        
        log_compliance_event(
            event_type="llm_failure_fallback",
            details={"conversation_id": conversation_id},
            user_id=None,
            loan_id=None
        )
        
        return AssistantResponse(
            action=ActionType.ESCALATE,
            message_to_user=("I'm experiencing technical difficulties and want to ensure "
                           "you receive the best service. Let me connect you with a "
                           "specialist who can assist you immediately."),
            structured_plan=None,
            confidence=1.0,
            escalation=True,
            compliance_checks=["technical_failure_escalation"],
            metadata={"fallback_reason": "llm_processing_error"}
        )


# Global LLM service instance
llm_service = LLMService()
