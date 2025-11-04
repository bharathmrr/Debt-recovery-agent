from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import os
from datetime import datetime
from dotenv import load_dotenv

from .models.database import get_db, create_tables
from .models.pydantic_models import (
    UserMessage, ConversationResponse, IdentityVerificationRequest,
    PaymentRequest, EscalationRequest, HealthCheck, ErrorResponse
)
from .services.conversation_service import conversation_service
from .services.rag_service import rag_service
from .utils.logging_config import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file_path=os.getenv("LOG_FILE_PATH", "logs/debt_recovery.log")
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Debt Recovery Agent API...")
    
    # Create database tables
    create_tables()
    
    # Initialize RAG service with default documents
    try:
        rag_service.initialize_default_documents()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Debt Recovery Agent API...")


# Create FastAPI app
app = FastAPI(
    title="AI Debt Recovery Agent",
    description="An AI-first debt recovery system with compliance and safety features",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred. Please try again later.",
            details={"type": type(exc).__name__}
        ).dict()
    )


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "AI Debt Recovery Agent API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db = next(get_db())
        db.execute("SELECT 1")
        db_status = "healthy"
        db.close()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    # Check RAG service
    try:
        rag_service.search("test query", top_k=1)
        rag_status = "healthy"
    except Exception as e:
        logger.error(f"RAG service health check failed: {e}")
        rag_status = "unhealthy"
    
    return HealthCheck(
        status="healthy" if db_status == "healthy" and rag_status == "healthy" else "degraded",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        dependencies={
            "database": db_status,
            "rag_service": rag_status,
            "llm_service": "healthy"  # Assume healthy if no immediate errors
        }
    )


@app.post("/converse", response_model=ConversationResponse)
async def converse(
    message: UserMessage,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Main conversation endpoint for debt recovery interactions
    
    Handles:
    - User message processing
    - LLM response generation
    - Conversation state management
    - Compliance checking
    """
    try:
        logger.info(f"Processing conversation for loan_id: {message.loan_id}")
        
        # Process the conversation
        response = await conversation_service.process_message(
            user_message=message.user_text,
            loan_id=message.loan_id,
            session_id=message.session_id,
            channel=message.channel,
            metadata=message.metadata,
            db=db
        )
        
        # Log conversation event in background
        background_tasks.add_task(
            conversation_service.log_conversation_event,
            response.conversation_id,
            "user_message",
            message.user_text,
            message.metadata
        )
        
        background_tasks.add_task(
            conversation_service.log_conversation_event,
            response.conversation_id,
            "assistant_response",
            response.assistant.message_to_user,
            response.assistant.dict()
        )
        
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in conversation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to process conversation")


@app.post("/verify-identity")
async def verify_identity(
    request: IdentityVerificationRequest,
    db: Session = Depends(get_db)
):
    """
    Identity verification endpoint
    
    Verifies borrower identity using provided data points
    """
    try:
        logger.info(f"Processing identity verification for conversation: {request.conversation_id}")
        
        result = await conversation_service.verify_identity(
            conversation_id=request.conversation_id,
            verification_data=request.verification_data,
            db=db
        )
        
        return {
            "verified": result["verified"],
            "message": result["message"],
            "attempts_remaining": result.get("attempts_remaining", 0)
        }
        
    except ValueError as e:
        logger.warning(f"Identity verification error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in identity verification: {e}")
        raise HTTPException(status_code=500, detail="Identity verification failed")


@app.post("/process-payment")
async def process_payment(
    request: PaymentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Payment processing endpoint
    
    Handles payment collection and processing
    """
    try:
        logger.info(f"Processing payment for loan_id: {request.loan_id}")
        
        result = await conversation_service.process_payment(
            loan_id=request.loan_id,
            amount=request.amount,
            payment_method=request.payment_method,
            conversation_id=request.conversation_id,
            metadata=request.metadata,
            db=db
        )
        
        # Log payment event in background
        background_tasks.add_task(
            conversation_service.log_payment_event,
            request.loan_id,
            request.amount,
            request.payment_method,
            result["status"],
            result.get("transaction_id")
        )
        
        return result
        
    except ValueError as e:
        logger.warning(f"Payment processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing payment: {e}")
        raise HTTPException(status_code=500, detail="Payment processing failed")


@app.post("/escalate")
async def escalate_conversation(
    request: EscalationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Escalation endpoint for human agent handoff
    """
    try:
        logger.info(f"Escalating conversation: {request.conversation_id}")
        
        result = await conversation_service.escalate_to_human(
            conversation_id=request.conversation_id,
            reason=request.reason,
            priority=request.priority,
            notes=request.notes,
            db=db
        )
        
        # Log escalation event in background
        background_tasks.add_task(
            conversation_service.log_escalation_event,
            request.conversation_id,
            request.reason,
            0.0,  # confidence score not available here
            result.get("assigned_agent")
        )
        
        return result
        
    except ValueError as e:
        logger.warning(f"Escalation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error escalating conversation: {e}")
        raise HTTPException(status_code=500, detail="Escalation failed")


@app.get("/conversation/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get conversation details and history"""
    try:
        conversation = await conversation_service.get_conversation(conversation_id, db)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")


@app.get("/loan/{loan_id}/conversations")
async def get_loan_conversations(
    loan_id: int,
    db: Session = Depends(get_db)
):
    """Get all conversations for a specific loan"""
    try:
        conversations = await conversation_service.get_loan_conversations(loan_id, db)
        return {"conversations": conversations}
        
    except Exception as e:
        logger.error(f"Error retrieving loan conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")


@app.post("/webhook/twilio")
async def twilio_webhook(
    request: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Webhook endpoint for Twilio SMS integration"""
    try:
        logger.info("Processing Twilio webhook")
        
        # Extract message details
        from_number = request.get("From", "")
        message_body = request.get("Body", "")
        
        # Find associated loan/borrower by phone number
        # This would need to be implemented based on your phone number lookup logic
        
        # Process as SMS conversation
        # Implementation would go here
        
        return {"status": "received"}
        
    except Exception as e:
        logger.error(f"Error processing Twilio webhook: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")


@app.post("/webhook/stripe")
async def stripe_webhook(
    request: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Webhook endpoint for Stripe payment events"""
    try:
        logger.info("Processing Stripe webhook")
        
        event_type = request.get("type", "")
        
        if event_type == "payment_intent.succeeded":
            # Handle successful payment
            payment_data = request.get("data", {}).get("object", {})
            # Implementation would go here
            
        elif event_type == "payment_intent.payment_failed":
            # Handle failed payment
            payment_data = request.get("data", {}).get("object", {})
            # Implementation would go here
        
        return {"status": "received"}
        
    except Exception as e:
        logger.error(f"Error processing Stripe webhook: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")


@app.get("/analytics/conversations")
async def get_conversation_analytics(
    start_date: str = None,
    end_date: str = None,
    db: Session = Depends(get_db)
):
    """Get conversation analytics and metrics"""
    try:
        metrics = await conversation_service.get_conversation_metrics(
            start_date=start_date,
            end_date=end_date,
            db=db
        )
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


@app.get("/analytics/collections")
async def get_collection_analytics(
    start_date: str = None,
    end_date: str = None,
    db: Session = Depends(get_db)
):
    """Get collection performance analytics"""
    try:
        metrics = await conversation_service.get_collection_metrics(
            start_date=start_date,
            end_date=end_date,
            db=db
        )
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving collection analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


@app.post("/admin/rebuild-rag-index")
async def rebuild_rag_index(background_tasks: BackgroundTasks):
    """Admin endpoint to rebuild the RAG index"""
    try:
        background_tasks.add_task(rag_service.rebuild_index)
        return {"message": "RAG index rebuild started"}
        
    except Exception as e:
        logger.error(f"Error starting RAG index rebuild: {e}")
        raise HTTPException(status_code=500, detail="Failed to start index rebuild")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
