#!/usr/bin/env python3
"""
System Test Script

Quick test to verify the AI Debt Recovery Agent is working correctly.
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_conversation():
    """Test a basic conversation"""
    print("\n💬 Testing conversation...")
    
    conversation_data = {
        "loan_id": 1,
        "user_text": "What is my current balance?",
        "session_id": "test_session_001",
        "channel": "chat"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/converse",
            json=conversation_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Conversation test passed")
            print(f"   Assistant: {result['assistant']['message_to_user']}")
            print(f"   Action: {result['assistant']['action']}")
            print(f"   Confidence: {result['assistant']['confidence']}")
            return True
        else:
            print(f"❌ Conversation test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Conversation test error: {e}")
        return False

def test_identity_verification():
    """Test identity verification"""
    print("\n🔐 Testing identity verification...")
    
    verification_data = {
        "conversation_id": "test_session_001",
        "verification_data": {
            "last_four_ssn": "1234",
            "last_payment_amount": "300.00"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/verify-identity",
            json=verification_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Identity verification test: {result['message']}")
            return result['verified']
        else:
            print(f"❌ Identity verification failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Identity verification error: {e}")
        return False

def test_verified_conversation():
    """Test conversation after identity verification"""
    print("\n💬 Testing verified conversation...")
    
    conversation_data = {
        "loan_id": 1,
        "user_text": "I can pay $200 per month for 6 months",
        "session_id": "test_session_001",
        "channel": "chat"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/converse",
            json=conversation_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Verified conversation test passed")
            print(f"   Assistant: {result['assistant']['message_to_user']}")
            print(f"   Action: {result['assistant']['action']}")
            
            if result['assistant']['structured_plan']:
                plan = result['assistant']['structured_plan']
                print(f"   Payment Plan: {plan['type']} - ${plan['amount']} x {plan.get('installments', 1)}")
            
            return True
        else:
            print(f"❌ Verified conversation test failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Verified conversation test error: {e}")
        return False

def test_api_docs():
    """Test API documentation endpoint"""
    print("\n📚 Testing API documentation...")
    
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ API documentation failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API documentation error: {e}")
        return False

def main():
    """Run all system tests"""
    print("🤖 AI Debt Recovery Agent - System Test")
    print("=" * 50)
    
    # Check if server is running
    print("Checking if server is running...")
    try:
        requests.get(BASE_URL, timeout=5)
    except requests.exceptions.RequestException:
        print("❌ Server is not running!")
        print("Please start the server with: python run.py")
        sys.exit(1)
    
    tests = [
        ("Health Check", test_health_check),
        ("API Documentation", test_api_docs),
        ("Basic Conversation", test_conversation),
        ("Identity Verification", test_identity_verification),
        ("Verified Conversation", test_verified_conversation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        print("\n🚀 Next steps:")
        print("1. Open http://localhost:8000/docs to explore the API")
        print("2. Try the conversation endpoint with different scenarios")
        print("3. Check the logs/ directory for detailed logging")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("💡 Common issues:")
        print("- Make sure PostgreSQL is running and configured")
        print("- Verify GROQ_API_KEY is set in your .env file")
        print("- Check that sample data was initialized")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
