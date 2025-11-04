# AI Debt Recovery Agent

An AI-first debt recovery system that converses with borrowers through multiple channels (chat, SMS, email, voice) using advanced LLM technology with compliance and safety guardrails.

## 🚀 Features

- **Multi-Channel Communication**: Chat, SMS, email, and voice support
- **AI-Powered Conversations**: Uses ChatGroq LLM for natural debt negotiation
- **RAG System**: Retrieval-Augmented Generation for contextual responses
- **Compliance First**: Built-in FDCPA compliance monitoring and enforcement
- **Payment Processing**: Integrated payment collection and plan management
- **Identity Verification**: Secure borrower identity verification
- **Escalation Management**: Automatic escalation to human agents when needed
- **Comprehensive Logging**: Audit trails and compliance logging with PII masking
- **Real-time Analytics**: Conversation and collection performance metrics

## 🏗️ Architecture

```
Frontend (Chat/SMS/Email) → FastAPI Backend → LLM Service (ChatGroq)
                                ↓
                         RAG Service (FAISS)
                                ↓
                         PostgreSQL Database
                                ↓
                         Compliance Service
```

### Core Components

- **FastAPI Backend**: REST API with conversation endpoints
- **LLM Service**: ChatGroq integration with prompt engineering
- **RAG Service**: Vector database for contextual information retrieval
- **Compliance Service**: FDCPA compliance monitoring and enforcement
- **Conversation Service**: Orchestrates the entire conversation flow
- **Database Models**: Comprehensive data models for loans, borrowers, conversations
- **Logging System**: Multi-level logging with PII masking and audit trails

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL database
- ChatGroq API key
- Optional: Twilio (SMS), Stripe (payments)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Debt-recovery-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Configure your environment**
```env
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/debt_recovery

# LLM API Keys
GROQ_API_KEY=your_groq_api_key_here

# External Services (Optional)
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
STRIPE_SECRET_KEY=your_stripe_secret_key

# Compliance Settings
MAX_SETTLEMENT_PERCENTAGE=0.70
MAX_INSTALLMENT_MONTHS=12
CONTACT_HOURS_START=08:00
CONTACT_HOURS_END=21:00
```

5. **Initialize the database**
```bash
# The application will create tables automatically on first run
python run.py
```

## 🚀 Usage

### Starting the Server

```bash
python run.py
```

The server will start on `http://localhost:8000`

### API Endpoints

- **Health Check**: `GET /health`
- **Main Conversation**: `POST /converse`
- **Identity Verification**: `POST /verify-identity`
- **Payment Processing**: `POST /process-payment`
- **Escalation**: `POST /escalate`
- **API Documentation**: `GET /docs`

### Example Conversation

```python
import requests

# Start a conversation
response = requests.post("http://localhost:8000/converse", json={
    "loan_id": 1,
    "user_text": "What's my current balance?",
    "session_id": "session_123",
    "channel": "chat"
})

print(response.json())
```

## 🔒 Compliance Features

### FDCPA Compliance
- Contact time restrictions (8 AM - 9 PM)
- Daily and weekly contact limits
- Opt-out request handling
- Debt validation request processing
- Prohibited language detection

### Data Protection
- PII masking in logs
- Secure identity verification
- Audit trail maintenance
- Compliance event logging

### Safety Guardrails
- Automatic escalation triggers
- Human oversight integration
- Confidence score monitoring
- Real-time compliance checking

## 📊 Monitoring & Analytics

### Logging
- **Console Logging**: Colored output for development
- **File Logging**: Rotating log files with retention
- **Audit Logging**: Compliance-focused audit trail
- **Conversation Logging**: Chat transcript storage

### Metrics
- Conversation success rates
- Payment collection metrics
- Escalation rates
- Compliance violation tracking
- LLM usage and cost tracking

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app tests/
```

## 🔧 Configuration

### LLM Configuration
- Model: `llama-3.1-8b-instant` (ChatGroq)
- Temperature: `0.45` (balanced creativity/consistency)
- Max Tokens: `1000`
- Timeout: `30 seconds`

### RAG Configuration
- Embedding Model: `all-MiniLM-L6-v2`
- Vector Database: FAISS
- Chunk Size: `500 tokens`
- Top-K Results: `5`

### Compliance Configuration
- Max Settlement: `70%`
- Max Installments: `12 months`
- Min Payment: `$25`
- Contact Hours: `8 AM - 9 PM`

## 📁 Project Structure

```
Debt-recovery-agent/
├── app/
│   ├── models/           # Database and Pydantic models
│   ├── services/         # Core business logic services
│   ├── utils/           # Utility functions and logging
│   └── main.py          # FastAPI application
├── prompts/             # LLM prompt templates
├── logs/               # Application logs (created at runtime)
├── data/               # Vector database files (created at runtime)
├── requirements.txt    # Python dependencies
├── .env.example       # Environment configuration template
├── run.py             # Application startup script
└── README.md          # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure compliance tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

### Compliance Disclaimer
This system is designed to assist with debt collection activities but does not replace the need for human oversight and legal compliance review. Always ensure your use complies with local, state, and federal regulations.

### Security Considerations
- Never hardcode API keys or sensitive information
- Use environment variables for all configuration
- Implement proper authentication in production
- Regular security audits recommended

### Production Deployment
- Use a production WSGI server (e.g., Gunicorn)
- Set up proper database connection pooling
- Implement rate limiting and request validation
- Configure proper logging and monitoring
- Set up backup and disaster recovery procedures

## 📞 Support

For questions, issues, or contributions, please open an issue in the GitHub repository.
real time Debt recovery ai agent for fin tech 
