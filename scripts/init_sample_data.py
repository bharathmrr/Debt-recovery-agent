#!/usr/bin/env python3
"""
Sample Data Initialization Script

This script creates sample borrowers, loans, and other data for testing the debt recovery system.
"""

import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from sqlalchemy.orm import Session
from app.models.database import SessionLocal, create_tables
from app.models.schemas import (
    Borrower, Loan, Transaction, ConsentStatus, LoanStatus, TransactionType
)
from app.services.rag_service import rag_service


def create_sample_borrowers(db: Session):
    """Create sample borrowers for testing"""
    
    borrowers_data = [
        {
            "name": "John Smith",
            "email": "john.smith@email.com",
            "phone": "+1-555-0101",
            "address": "123 Main St, Anytown, ST 12345",
            "ssn_last_four": "1234",
            "consent_status": ConsentStatus.GRANTED,
            "preferred_contact_method": "email"
        },
        {
            "name": "Sarah Johnson",
            "email": "sarah.johnson@email.com", 
            "phone": "+1-555-0102",
            "address": "456 Oak Ave, Somewhere, ST 67890",
            "ssn_last_four": "5678",
            "consent_status": ConsentStatus.GRANTED,
            "preferred_contact_method": "sms"
        },
        {
            "name": "Michael Brown",
            "email": "michael.brown@email.com",
            "phone": "+1-555-0103", 
            "address": "789 Pine Rd, Elsewhere, ST 11111",
            "ssn_last_four": "9012",
            "consent_status": ConsentStatus.PENDING,
            "preferred_contact_method": "phone"
        },
        {
            "name": "Emily Davis",
            "email": "emily.davis@email.com",
            "phone": "+1-555-0104",
            "address": "321 Elm St, Nowhere, ST 22222", 
            "ssn_last_four": "3456",
            "consent_status": ConsentStatus.REVOKED,
            "preferred_contact_method": "email"
        }
    ]
    
    borrowers = []
    for data in borrowers_data:
        borrower = Borrower(**data)
        db.add(borrower)
        borrowers.append(borrower)
    
    db.flush()  # Get IDs
    print(f"Created {len(borrowers)} sample borrowers")
    return borrowers


def create_sample_loans(db: Session, borrowers):
    """Create sample loans for the borrowers"""
    
    loans_data = [
        {
            "borrower_id": borrowers[0].id,
            "account_number": "ACC001234",
            "principal_amount": 1500.00,
            "current_balance": 1200.00,
            "interest_rate": 0.18,
            "origination_date": datetime.now() - timedelta(days=365),
            "due_date": datetime.now() - timedelta(days=90),
            "last_payment_date": datetime.now() - timedelta(days=120),
            "last_payment_amount": 300.00,
            "status": LoanStatus.OVERDUE,
            "days_overdue": 90
        },
        {
            "borrower_id": borrowers[1].id,
            "account_number": "ACC005678",
            "principal_amount": 2500.00,
            "current_balance": 2800.00,
            "interest_rate": 0.22,
            "origination_date": datetime.now() - timedelta(days=200),
            "due_date": datetime.now() - timedelta(days=45),
            "last_payment_date": datetime.now() - timedelta(days=60),
            "last_payment_amount": 150.00,
            "status": LoanStatus.OVERDUE,
            "days_overdue": 45
        },
        {
            "borrower_id": borrowers[2].id,
            "account_number": "ACC009012",
            "principal_amount": 800.00,
            "current_balance": 950.00,
            "interest_rate": 0.15,
            "origination_date": datetime.now() - timedelta(days=180),
            "due_date": datetime.now() - timedelta(days=30),
            "last_payment_date": datetime.now() - timedelta(days=45),
            "last_payment_amount": 100.00,
            "status": LoanStatus.OVERDUE,
            "days_overdue": 30
        },
        {
            "borrower_id": borrowers[3].id,
            "account_number": "ACC003456",
            "principal_amount": 3000.00,
            "current_balance": 0.00,
            "interest_rate": 0.20,
            "origination_date": datetime.now() - timedelta(days=400),
            "due_date": datetime.now() - timedelta(days=30),
            "last_payment_date": datetime.now() - timedelta(days=10),
            "last_payment_amount": 3200.00,
            "status": LoanStatus.PAID,
            "days_overdue": 0
        }
    ]
    
    loans = []
    for data in loans_data:
        loan = Loan(**data)
        db.add(loan)
        loans.append(loan)
    
    db.flush()  # Get IDs
    print(f"Created {len(loans)} sample loans")
    return loans


def create_sample_transactions(db: Session, loans):
    """Create sample transaction history"""
    
    transactions = []
    
    # Create transactions for each loan
    for loan in loans:
        if loan.status == LoanStatus.PAID:
            # Paid loan - create payment history
            payment_dates = [
                loan.origination_date + timedelta(days=30),
                loan.origination_date + timedelta(days=60),
                loan.origination_date + timedelta(days=90),
                loan.last_payment_date
            ]
            
            payment_amounts = [500.00, 800.00, 700.00, 1200.00]
            
            for date, amount in zip(payment_dates, payment_amounts):
                transaction = Transaction(
                    loan_id=loan.id,
                    transaction_id=f"TXN_{loan.account_number}_{len(transactions)+1:03d}",
                    amount=amount,
                    transaction_type=TransactionType.PAYMENT,
                    description=f"Payment for account {loan.account_number}",
                    payment_method="credit_card",
                    processed_at=date
                )
                db.add(transaction)
                transactions.append(transaction)
        
        else:
            # Overdue loan - create some payment history
            if loan.last_payment_date:
                transaction = Transaction(
                    loan_id=loan.id,
                    transaction_id=f"TXN_{loan.account_number}_001",
                    amount=loan.last_payment_amount,
                    transaction_type=TransactionType.PAYMENT,
                    description=f"Last payment for account {loan.account_number}",
                    payment_method="bank_transfer",
                    processed_at=loan.last_payment_date
                )
                db.add(transaction)
                transactions.append(transaction)
            
            # Add some fees
            fee_transaction = Transaction(
                loan_id=loan.id,
                transaction_id=f"TXN_{loan.account_number}_FEE",
                amount=25.00,
                transaction_type=TransactionType.FEE,
                description="Late payment fee",
                processed_at=loan.due_date + timedelta(days=15)
            )
            db.add(fee_transaction)
            transactions.append(fee_transaction)
    
    print(f"Created {len(transactions)} sample transactions")
    return transactions


def create_sample_rag_documents(borrowers, loans):
    """Create sample documents for the RAG system"""
    
    # Add borrower profile documents
    for i, (borrower, loan) in enumerate(zip(borrowers, loans)):
        profile_content = f"""
        Borrower Profile: {borrower.name}
        Account: {loan.account_number}
        
        Contact Information:
        - Email: {borrower.email}
        - Phone: {borrower.phone}
        - Preferred Contact: {borrower.preferred_contact_method}
        
        Account Details:
        - Original Amount: ${loan.principal_amount:.2f}
        - Current Balance: ${loan.current_balance:.2f}
        - Status: {loan.status.value}
        - Days Overdue: {loan.days_overdue}
        
        Payment History:
        - Last Payment: ${loan.last_payment_amount:.2f} on {loan.last_payment_date.strftime('%Y-%m-%d') if loan.last_payment_date else 'N/A'}
        
        Notes:
        - Borrower has been responsive to previous communications
        - Prefers {borrower.preferred_contact_method} contact method
        - Account shows history of partial payments
        """
        
        rag_service.add_document(
            content=profile_content,
            document_type="borrower_profile",
            source=f"profile_{borrower.id}",
            title=f"Profile for {borrower.name}",
            loan_id=loan.id,
            borrower_id=borrower.id
        )
    
    # Add communication history documents
    communication_examples = [
        {
            "content": """
            Previous Communication Log - John Smith (ACC001234)
            Date: 2025-10-15
            Channel: Email
            
            Agent: Contacted borrower regarding overdue balance of $1,200
            Borrower Response: "I lost my job last month but should have work again soon. Can we work out a payment plan?"
            Resolution: Agreed to follow up in 2 weeks
            
            Follow-up needed: Yes
            Borrower showed willingness to pay and communicate
            """,
            "borrower_id": borrowers[0].id,
            "loan_id": loans[0].id
        },
        {
            "content": """
            Previous Communication Log - Sarah Johnson (ACC005678)
            Date: 2025-10-20
            Channel: SMS
            
            Agent: Sent payment reminder via SMS
            Borrower Response: "I can pay $200 this week and $200 next week"
            Resolution: Set up 2-payment plan
            
            Payment plan status: Active
            Next payment due: 2025-11-01
            """,
            "borrower_id": borrowers[1].id,
            "loan_id": loans[1].id
        }
    ]
    
    for comm in communication_examples:
        rag_service.add_document(
            content=comm["content"],
            document_type="communication",
            source="previous_communications",
            title="Communication History",
            loan_id=comm["loan_id"],
            borrower_id=comm["borrower_id"]
        )
    
    print("Created sample RAG documents")


def main():
    """Main function to initialize sample data"""
    
    print("Initializing sample data for AI Debt Recovery Agent...")
    
    # Create database tables
    create_tables()
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Check if data already exists
        existing_borrowers = db.query(Borrower).count()
        if existing_borrowers > 0:
            print(f"Sample data already exists ({existing_borrowers} borrowers found)")
            print("Skipping data creation. Use --force to recreate.")
            return
        
        # Create sample data
        borrowers = create_sample_borrowers(db)
        loans = create_sample_loans(db, borrowers)
        transactions = create_sample_transactions(db, loans)
        
        # Commit database changes
        db.commit()
        print("Database sample data created successfully")
        
        # Create RAG documents
        create_sample_rag_documents(borrowers, loans)
        print("RAG sample documents created successfully")
        
        print("\n" + "="*50)
        print("Sample Data Summary:")
        print(f"- Borrowers: {len(borrowers)}")
        print(f"- Loans: {len(loans)}")
        print(f"- Transactions: {len(transactions)}")
        print("="*50)
        
        print("\nSample API Test:")
        print("curl -X POST http://localhost:8000/converse \\")
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"loan_id": 1, "user_text": "What is my current balance?", "session_id": "test_session", "channel": "chat"}\'')
        
    except Exception as e:
        print(f"Error creating sample data: {e}")
        db.rollback()
        raise
    
    finally:
        db.close()


if __name__ == "__main__":
    main()
