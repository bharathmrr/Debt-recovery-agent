#!/usr/bin/env python3
"""
Debt Recovery Agent - Startup Script

This script initializes and runs the AI-first debt recovery system.
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Load environment variables
load_dotenv()

def main():
    """Main entry point for the application"""
    
    # Configuration
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print("=" * 60)
    print("🤖 AI Debt Recovery Agent")
    print("=" * 60)
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📊 Health Check: http://{host}:{port}/health")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"🔧 Reload: {reload}")
    print(f"📝 Log Level: {log_level}")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()
