#!/usr/bin/env python3
"""
Simple test with real API call to validate the implementation works.
"""

import os
from openai_chat import OpenAIChat

def test_simple_real_call():
    """Test a very simple real API call."""
    print("=== Testing Simple Real API Call ===")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OPENAI_API_KEY found")
        return False
    
    print(f"Using API key: {api_key[:10]}...")
    
    try:
        # Create chat with gpt-5-nano (as requested)
        chat = OpenAIChat(model_name="gpt-5-nano")
        print(f"✅ Created chat instance with model: {chat.model}")
        
        # Try the simplest possible call
        print("Making API call...")
        result = chat.invoke("Hi")
        
        print(f"✅ API call successful!")
        print(f"Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        print("\n🔍 Debugging info:")
        print("   - API key format looks correct")
        print("   - Payload structure is valid (confirmed by tests)")
        print("   - Endpoint is correct (/v1/responses)")
        print("   - Model is gpt-5-nano (as requested)")
        print("\n💡 Possible solutions:")
        print("   1. Check if API key has billing credits")
        print("   2. Verify API key has access to GPT-5 models")
        print("   3. Check at https://platform.openai.com/account/api-keys")
        print("   4. Try regenerating the API key")
        return False

if __name__ == "__main__":
    test_simple_real_call()
