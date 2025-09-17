#!/usr/bin/env python3
"""
Test different tool formats by modifying the GroqChatResp class
"""

from groq_chat import GroqChatResp
from calculator_tool import calculator

def test_current_format():
    """Test the current tool format"""
    print("=== Testing current tool format ===")
    
    try:
        chat_resp = GroqChatResp()
        chat_resp.default_reasoning = {
            "temperature": 0.8,
            "max_output_tokens": 1024,
            "top_p": 0.9
        }
        
        # Test with calculator tool
        response = chat_resp.invoke("What is 15 + 27?", tools=[calculator])
        print(f"Response: {response}")
        print("✅ Success with current format!")
        
    except Exception as e:
        print(f"❌ Error with current format: {e}")

if __name__ == "__main__":
    test_current_format()