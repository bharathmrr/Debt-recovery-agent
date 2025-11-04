import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import structlog
import colorlog
from structlog.stdlib import LoggerFactory


class ComplianceFilter(logging.Filter):
    """Filter to ensure compliance-sensitive logs are properly handled"""
    
    def filter(self, record):
        # Add compliance metadata to all records
        record.compliance_timestamp = datetime.utcnow().isoformat()
        record.service = "debt-recovery-agent"
        return True


class PIIMaskingProcessor:
    """Processor to mask PII in log messages"""
    
    @staticmethod
    def mask_pii(event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive information in log events"""
        sensitive_fields = [
            'ssn', 'social_security', 'credit_card', 'account_number',
            'phone', 'email', 'address', 'full_name'
        ]
        
        def mask_value(value: str) -> str:
            if len(value) <= 4:
                return '*' * len(value)
            return value[:2] + '*' * (len(value) - 4) + value[-2:]
        
        # Mask sensitive fields in the event dictionary
        for field in sensitive_fields:
            if field in event_dict:
                if isinstance(event_dict[field], str):
                    event_dict[field] = mask_value(event_dict[field])
        
        # Mask sensitive patterns in message text
        message = event_dict.get('event', '')
        if isinstance(message, str):
            # Simple regex patterns for common PII
            import re
            # Mask SSN patterns (XXX-XX-XXXX)
            message = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', message)
            # Mask phone patterns
            message = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', 'XXX-XXX-XXXX', message)
            # Mask email patterns
            message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'XXX@XXX.com', message)
            event_dict['event'] = message
        
        return event_dict


def setup_logging(log_level: str = "INFO", log_file_path: str = "logs/debt_recovery.log") -> None:
    """
    Set up comprehensive logging for the debt recovery system
    
    Features:
    - Console logging with colors
    - File logging with rotation
    - Structured logging with JSON format
    - PII masking for compliance
    - Audit trail capabilities
    """
    
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            PIIMaskingProcessor.mask_pii,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ComplianceFilter())
    
    # File handler with rotation (for general logs)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10,
        encoding='utf-8'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(compliance_timestamp)s - %(service)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(ComplianceFilter())
    
    # Audit log handler (separate file for compliance)
    audit_log_path = log_dir / "audit.log"
    audit_handler = logging.handlers.RotatingFileHandler(
        audit_log_path,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=50,  # Keep more audit logs
        encoding='utf-8'
    )
    audit_formatter = logging.Formatter(
        '%(asctime)s - AUDIT - %(compliance_timestamp)s - %(service)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    audit_handler.setFormatter(audit_formatter)
    audit_handler.addFilter(ComplianceFilter())
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Create audit logger
    audit_logger = logging.getLogger('audit')
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Don't propagate to root logger
    
    # Create conversation logger (for chat transcripts)
    conversation_log_path = log_dir / "conversations.log"
    conversation_handler = logging.handlers.RotatingFileHandler(
        conversation_log_path,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=20,
        encoding='utf-8'
    )
    conversation_formatter = logging.Formatter(
        '%(asctime)s - CONVERSATION - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    conversation_handler.setFormatter(conversation_formatter)
    
    conversation_logger = logging.getLogger('conversation')
    conversation_logger.addHandler(conversation_handler)
    conversation_logger.setLevel(logging.INFO)
    conversation_logger.propagate = False


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


def get_audit_logger() -> logging.Logger:
    """Get the audit logger for compliance tracking"""
    return logging.getLogger('audit')


def get_conversation_logger() -> logging.Logger:
    """Get the conversation logger for chat transcripts"""
    return logging.getLogger('conversation')


# Compliance logging helpers
def log_compliance_event(event_type: str, details: Dict[str, Any], user_id: str = None, loan_id: str = None):
    """Log compliance-related events"""
    audit_logger = get_audit_logger()
    
    log_data = {
        'event_type': event_type,
        'user_id': user_id,
        'loan_id': loan_id,
        'timestamp': datetime.utcnow().isoformat(),
        'details': details
    }
    
    audit_logger.info(f"COMPLIANCE_EVENT: {event_type}", extra=log_data)


def log_conversation_event(conversation_id: str, message_type: str, content: str, metadata: Dict[str, Any] = None):
    """Log conversation events with PII masking"""
    conversation_logger = get_conversation_logger()
    
    # Apply PII masking to content
    masked_content = PIIMaskingProcessor.mask_pii({'event': content})['event']
    
    log_data = {
        'conversation_id': conversation_id,
        'message_type': message_type,
        'content': masked_content,
        'metadata': metadata or {},
        'timestamp': datetime.utcnow().isoformat()
    }
    
    conversation_logger.info(f"CONV_{message_type}: {conversation_id}", extra=log_data)


def log_llm_interaction(conversation_id: str, prompt_tokens: int, completion_tokens: int, 
                       model: str, cost: float = None, response_time: float = None):
    """Log LLM API interactions for cost tracking and monitoring"""
    logger = get_logger("llm_interaction")
    
    logger.info(
        "LLM API call completed",
        conversation_id=conversation_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        model=model,
        cost=cost,
        response_time_seconds=response_time
    )


def log_payment_event(loan_id: str, amount: float, payment_method: str, status: str, 
                     transaction_id: str = None, error: str = None):
    """Log payment processing events"""
    logger = get_logger("payment")
    
    logger.info(
        "Payment event",
        loan_id=loan_id,
        amount=amount,
        payment_method=payment_method,
        status=status,
        transaction_id=transaction_id,
        error=error
    )


def log_escalation_event(conversation_id: str, reason: str, confidence_score: float, 
                        assigned_agent: str = None):
    """Log escalation events to human agents"""
    logger = get_logger("escalation")
    
    logger.warning(
        "Conversation escalated to human agent",
        conversation_id=conversation_id,
        reason=reason,
        confidence_score=confidence_score,
        assigned_agent=assigned_agent
    )
