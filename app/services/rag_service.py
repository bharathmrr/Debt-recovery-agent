import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import hashlib

from ..models.database import get_db
from ..models.schemas import VectorDocument, Loan, Borrower
from ..models.pydantic_models import RAGQuery, RAGResult, RAGResponse
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation service for debt recovery system
    
    Handles:
    - Document embedding and indexing
    - Similarity search for relevant context
    - Borrower-specific information retrieval
    - Policy and regulation lookup
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", index_path: str = "data/faiss_index"):
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index_path = index_path
        self.index = None
        self.document_metadata = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Load or create FAISS index
        self._load_or_create_index()
        
        logger.info(f"RAG Service initialized with model: {embedding_model}")
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one"""
        try:
            if os.path.exists(f"{self.index_path}.index"):
                self.index = faiss.read_index(f"{self.index_path}.index")
                
                # Load metadata
                if os.path.exists(f"{self.index_path}.metadata"):
                    with open(f"{self.index_path}.metadata", 'r') as f:
                        self.document_metadata = json.load(f)
                
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} documents")
            else:
                self._create_new_index()
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
        self.document_metadata = {}
        logger.info("Created new FAISS index")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            faiss.write_index(self.index, f"{self.index_path}.index")
            
            with open(f"{self.index_path}.metadata", 'w') as f:
                json.dump(self.document_metadata, f, indent=2, default=str)
            
            logger.info(f"Saved FAISS index with {self.index.ntotal} documents")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def _generate_document_id(self, content: str, document_type: str, source: str) -> str:
        """Generate unique document ID based on content hash"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{document_type}_{source}_{content_hash[:8]}"
    
    def add_document(self, content: str, document_type: str, source: str, 
                    title: str = None, metadata: Dict[str, Any] = None,
                    loan_id: int = None, borrower_id: int = None,
                    chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Add a document to the vector database with chunking
        
        Args:
            content: Document content
            document_type: Type of document (policy, borrower_profile, communication, regulation)
            source: Source of the document
            title: Document title
            metadata: Additional metadata
            loan_id: Associated loan ID
            borrower_id: Associated borrower ID
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
        
        Returns:
            List of document IDs created
        """
        try:
            # Split content into chunks
            chunks = self._chunk_text(content, chunk_size, overlap)
            document_ids = []
            
            db = next(get_db())
            
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_model.encode([chunk])
                embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)  # Normalize for cosine similarity
                
                # Generate document ID
                doc_id = self._generate_document_id(chunk, document_type, f"{source}_chunk_{i}")
                
                # Add to FAISS index
                self.index.add(embedding.astype('float32'))
                
                # Store metadata
                doc_metadata = {
                    'document_id': doc_id,
                    'document_type': document_type,
                    'source': source,
                    'title': title or f"{source} - Chunk {i+1}",
                    'content': chunk,
                    'chunk_index': i,
                    'loan_id': loan_id,
                    'borrower_id': borrower_id,
                    'metadata': metadata or {},
                    'created_at': datetime.utcnow().isoformat()
                }
                
                self.document_metadata[str(self.index.ntotal - 1)] = doc_metadata
                
                # Store in database
                vector_doc = VectorDocument(
                    document_id=doc_id,
                    document_type=document_type,
                    source=source,
                    title=doc_metadata['title'],
                    content=chunk,
                    chunk_index=i,
                    metadata=metadata,
                    loan_id=loan_id,
                    borrower_id=borrower_id,
                    embedding_model=self.embedding_model_name
                )
                
                db.add(vector_doc)
                document_ids.append(doc_id)
            
            db.commit()
            self._save_index()
            
            logger.info(f"Added document with {len(chunks)} chunks to vector database")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error adding document to vector database: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def search(self, query: str, top_k: int = 5, loan_id: int = None, 
              borrower_id: int = None, document_types: List[str] = None) -> RAGResponse:
        """
        Search for relevant documents using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of results to return
            loan_id: Filter by loan ID
            borrower_id: Filter by borrower ID
            document_types: Filter by document types
        
        Returns:
            RAGResponse with search results
        """
        start_time = datetime.utcnow()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k * 3, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                doc_metadata = self.document_metadata.get(str(idx))
                if not doc_metadata:
                    continue
                
                # Apply filters
                if loan_id and doc_metadata.get('loan_id') != loan_id:
                    continue
                
                if borrower_id and doc_metadata.get('borrower_id') != borrower_id:
                    continue
                
                if document_types and doc_metadata.get('document_type') not in document_types:
                    continue
                
                result = RAGResult(
                    document_id=doc_metadata['document_id'],
                    content=doc_metadata['content'],
                    score=float(score),
                    metadata=doc_metadata.get('metadata', {}),
                    document_type=doc_metadata['document_type']
                )
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            # Sort by score (descending)
            results.sort(key=lambda x: x.score, reverse=True)
            
            query_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"RAG search completed: {len(results)} results in {query_time:.3f}s")
            
            return RAGResponse(
                results=results,
                total_results=len(results),
                query_time=query_time
            )
            
        except Exception as e:
            logger.error(f"Error during RAG search: {e}")
            return RAGResponse(results=[], total_results=0, query_time=0.0)
    
    def get_borrower_context(self, loan_id: int, borrower_id: int) -> Dict[str, Any]:
        """
        Get comprehensive context for a specific borrower and loan
        
        Returns:
            Dictionary with borrower profile, loan details, and payment history
        """
        try:
            db = next(get_db())
            
            # Get loan and borrower information
            loan = db.query(Loan).filter(Loan.id == loan_id).first()
            borrower = db.query(Borrower).filter(Borrower.id == borrower_id).first()
            
            if not loan or not borrower:
                return {}
            
            # Search for borrower-specific documents
            borrower_docs = self.search(
                query=f"borrower {borrower.name} payment history account {loan.account_number}",
                top_k=3,
                loan_id=loan_id,
                borrower_id=borrower_id,
                document_types=['borrower_profile', 'communication']
            )
            
            context = {
                'borrower': {
                    'name': borrower.name,
                    'account_number': loan.account_number,
                    'current_balance': loan.current_balance,
                    'principal_amount': loan.principal_amount,
                    'due_date': loan.due_date.isoformat() if loan.due_date else None,
                    'days_overdue': loan.days_overdue,
                    'last_payment_date': loan.last_payment_date.isoformat() if loan.last_payment_date else None,
                    'last_payment_amount': loan.last_payment_amount,
                    'status': loan.status.value,
                    'preferred_contact_method': borrower.preferred_contact_method,
                    'consent_status': borrower.consent_status.value
                },
                'relevant_documents': [
                    {
                        'content': doc.content,
                        'type': doc.document_type,
                        'score': doc.score
                    }
                    for doc in borrower_docs.results
                ]
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting borrower context: {e}")
            return {}
        finally:
            db.close()
    
    def get_policy_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get relevant policy and regulation information
        
        Args:
            query: Policy-related query
            top_k: Number of policy documents to return
        
        Returns:
            List of relevant policy documents
        """
        try:
            policy_docs = self.search(
                query=query,
                top_k=top_k,
                document_types=['policy', 'regulation']
            )
            
            return [
                {
                    'content': doc.content,
                    'type': doc.document_type,
                    'score': doc.score,
                    'metadata': doc.metadata
                }
                for doc in policy_docs.results
            ]
            
        except Exception as e:
            logger.error(f"Error getting policy context: {e}")
            return []
    
    def initialize_default_documents(self):
        """Initialize the vector database with default policies and regulations"""
        
        # Default compliance policies
        compliance_policies = [
            {
                'content': """
                FDCPA Compliance Guidelines:
                - Contact hours: 8 AM to 9 PM in borrower's timezone
                - No contact on Sundays or federal holidays
                - Maximum 3 contact attempts per day
                - Must identify as debt collector
                - Cannot use threatening or abusive language
                - Must provide debt validation notice within 5 days
                - Cannot contact at work if prohibited by employer
                """,
                'title': "FDCPA Basic Compliance Rules",
                'document_type': "regulation",
                'source': "fdcpa_guidelines"
            },
            {
                'content': """
                Payment Plan Policies:
                - Maximum settlement percentage: 70% of outstanding balance
                - Maximum installment period: 12 months
                - Minimum payment amount: $25 per installment
                - Payment plans require written agreement
                - First payment due within 30 days of agreement
                - Missed payments may void the agreement
                """,
                'title': "Payment Plan Guidelines",
                'document_type': "policy",
                'source': "internal_policy"
            },
            {
                'content': """
                Escalation Triggers:
                - Borrower requests debt validation
                - Borrower disputes the debt
                - Borrower mentions bankruptcy or attorney
                - Borrower expresses financial hardship or distress
                - Borrower requests to speak with supervisor
                - System confidence score below 0.5
                - Complex settlement negotiations
                """,
                'title': "Human Escalation Guidelines",
                'document_type': "policy",
                'source': "escalation_policy"
            },
            {
                'content': """
                Identity Verification Requirements:
                - Last 4 digits of SSN
                - Last payment amount and date
                - Account number verification
                - Maximum 3 verification attempts
                - Lock account after failed attempts
                - Require human verification for locked accounts
                """,
                'title': "Identity Verification Policy",
                'document_type': "policy",
                'source': "verification_policy"
            }
        ]
        
        logger.info("Initializing vector database with default documents...")
        
        for policy in compliance_policies:
            self.add_document(
                content=policy['content'],
                document_type=policy['document_type'],
                source=policy['source'],
                title=policy['title']
            )
        
        logger.info("Default documents added to vector database")
    
    def rebuild_index(self):
        """Rebuild the entire FAISS index from database"""
        try:
            logger.info("Rebuilding FAISS index from database...")
            
            # Create new index
            self._create_new_index()
            
            db = next(get_db())
            documents = db.query(VectorDocument).all()
            
            embeddings = []
            metadata_list = []
            
            for doc in documents:
                # Generate embedding
                embedding = self.embedding_model.encode([doc.content])
                embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                embeddings.append(embedding[0])
                
                # Prepare metadata
                doc_metadata = {
                    'document_id': doc.document_id,
                    'document_type': doc.document_type,
                    'source': doc.source,
                    'title': doc.title,
                    'content': doc.content,
                    'chunk_index': doc.chunk_index,
                    'loan_id': doc.loan_id,
                    'borrower_id': doc.borrower_id,
                    'metadata': doc.metadata or {},
                    'created_at': doc.created_at.isoformat()
                }
                metadata_list.append(doc_metadata)
            
            if embeddings:
                # Add all embeddings to index
                embeddings_array = np.array(embeddings).astype('float32')
                self.index.add(embeddings_array)
                
                # Update metadata
                for i, metadata in enumerate(metadata_list):
                    self.document_metadata[str(i)] = metadata
                
                self._save_index()
            
            logger.info(f"Index rebuilt with {len(embeddings)} documents")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            raise
        finally:
            db.close()


# Global RAG service instance
rag_service = RAGService()
