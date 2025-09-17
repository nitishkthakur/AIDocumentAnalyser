#!/usr/bin/env python3
"""
Simple test for GroqChatResp class to verify basic functionality.
"""

import sys
import os
from groq_chat import GroqChatResp

def simple_test():
    """Simple test with minimal parameters."""
    print("=== Simple GroqChatResp Test ===")
    
    try:
        # Initialize with minimal parameters
        chat_resp = GroqChatResp()
        
        # Override default reasoning to use only supported parameters
        chat_resp.default_reasoning = {
            "temperature": 0.8,
            "max_output_tokens": 1024,
            "top_p": 0.9
        }
        
        print(f"Model: {chat_resp.model}")
        print(f"Reasoning parameters: {chat_resp.default_reasoning}")
        
        # Basic test
        response = chat_resp.invoke("Say hello in 3 words.")
        print(f"Response: {response}")
        print(f"Response type: {type(response)}")
        
        print("✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if not os.getenv('GROQ_API_KEY'):
        print("❌ GROQ_API_KEY environment variable not set")
        sys.exit(1)
    
    success = simple_test()
    sys.exit(0 if success else 1)