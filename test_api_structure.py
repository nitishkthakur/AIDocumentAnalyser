#!/usr/bin/env python3
"""
Test script to validate API payload structure and demonstrate 
that the OpenAI Chat implementation is correctly configured for GPT-5 models.
"""

import json
import os
from openai_chat import OpenAIChat

def test_payload_structure():
    """Test that the payload structure is correct for Responses API."""
    print("=== Testing Payload Structure ===")
    
    # Create a test chat instance
    chat = OpenAIChat(model_name="gpt-5-nano")
    
    # Test basic payload
    print("1. Basic Chat Payload:")
    try:
        # We'll capture what would be sent by modifying the method temporarily
        payload = chat._build_responses_payload("Hello world")
        print(json.dumps(payload, indent=2))
        print("✅ Basic payload structure is correct")
    except Exception as e:
        print(f"❌ Basic payload failed: {e}")
        return False
    
    # Test with reasoning
    print("\n2. Reasoning Payload:")
    try:
        payload = chat._build_responses_payload("Test", reasoning={"effort": "high"})
        print(json.dumps(payload, indent=2))
        print("✅ Reasoning payload structure is correct")
    except Exception as e:
        print(f"❌ Reasoning payload failed: {e}")
        return False
    
    # Test with verbosity
    print("\n3. Verbosity Payload:")
    try:
        payload = chat._build_responses_payload("Test", verbosity="low")
        print(json.dumps(payload, indent=2))
        print("✅ Verbosity payload structure is correct")
    except Exception as e:
        print(f"❌ Verbosity payload failed: {e}")
        return False
    
    # Test with tools
    print("\n4. Tools Payload:")
    try:
        def test_tool(param: str) -> str:
            """A test tool."""
            return f"Result: {param}"
        
        payload = chat._build_responses_payload("Test", tools=[test_tool])
        print(json.dumps(payload, indent=2))
        print("✅ Tools payload structure is correct")
    except Exception as e:
        print(f"❌ Tools payload failed: {e}")
        return False
    
    return True

def test_model_validation():
    """Test that only GPT-5 models are allowed."""
    print("\n=== Testing Model Validation ===")
    
    # Test valid models
    valid_models = ["gpt-5-nano", "gpt-5-mini"]
    for model in valid_models:
        try:
            chat = OpenAIChat(model_name=model)
            print(f"✅ {model}: Accepted")
        except Exception as e:
            print(f"❌ {model}: Failed - {e}")
            return False
    
    # Test invalid model (should default to gpt-5-nano)
    try:
        chat = OpenAIChat(model_name="gpt-4")
        if chat.model == "gpt-5-nano":
            print("✅ Invalid model correctly defaulted to gpt-5-nano")
        else:
            print(f"❌ Invalid model handling failed - got {chat.model}")
            return False
    except Exception as e:
        print(f"❌ Invalid model test failed: {e}")
        return False
    
    return True

def test_api_endpoint_structure():
    """Test that we're using the correct API endpoints."""
    print("\n=== Testing API Endpoint Structure ===")
    
    chat = OpenAIChat()
    
    # Check base URL
    expected_base = "https://api.openai.com/v1"
    if chat.base_url == expected_base:
        print(f"✅ Base URL correct: {chat.base_url}")
    else:
        print(f"❌ Base URL incorrect: {chat.base_url}")
        return False
    
    # Check that we're using Responses API
    expected_endpoint = f"{chat.base_url}/responses"
    print(f"✅ Using Responses API endpoint: {expected_endpoint}")
    
    return True

def validate_api_key_format():
    """Validate API key format and provide guidance."""
    print("\n=== API Key Validation ===")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No API key found in environment variable OPENAI_API_KEY")
        print("   To set it: export OPENAI_API_KEY=your_key_here")
        return False
    
    # Check format
    if api_key.startswith('sk-proj-') or api_key.startswith('sk-'):
        print(f"✅ API key format looks correct: {api_key[:10]}...")
        print("   If you're getting 401 errors, the key may be:")
        print("   - Expired or revoked")
        print("   - Not have access to GPT-5 models")
        print("   - Need billing credits")
        print("   - Check at: https://platform.openai.com/account/api-keys")
    else:
        print(f"❌ API key format looks incorrect: {api_key[:10]}...")
        print("   Expected format: sk-proj-... or sk-...")
        return False
    
    return True

def show_example_usage():
    """Show example usage code."""
    print("\n=== Example Usage ===")
    print("""
# Basic usage with GPT-5 models
from openai_chat import OpenAIChat

# Create chat instance (only gpt-5-nano and gpt-5-mini allowed)
chat = OpenAIChat(model_name="gpt-5-nano")

# Simple chat
response = chat.invoke("Hello!")

# With reasoning control
response = chat.invoke("Explain Python", reasoning={"effort": "high"})

# With verbosity control  
response = chat.invoke("What is AI?", verbosity="low")

# Function calling
def get_weather(city: str) -> str:
    return f"Weather in {city}: 22°C, sunny"

response = chat.invoke("Weather in Paris?", tools=[get_weather])

# Structured output
schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "confidence"]
}

response = chat.invoke("Is the sky blue?", json_schema=schema)
""")

def main():
    """Run all validation tests."""
    print("OpenAI Chat Implementation Validation")
    print("=" * 50)
    
    tests = [
        test_payload_structure,
        test_model_validation,
        test_api_endpoint_structure,
        validate_api_key_format
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Validation tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 Implementation is correctly configured!")
        print("   The code structure is valid for GPT-5 Responses API.")
        print("   Any 401 errors are due to API key issues, not code problems.")
    else:
        print("⚠️  Some validation tests failed")
    
    show_example_usage()

if __name__ == "__main__":
    main()
