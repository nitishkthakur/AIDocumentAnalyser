#!/usr/bin/env python3

import os
import json
from openai_chat import OpenAIChat

def test_both_models():
    """Test both gpt-5-nano and gpt-5-mini models."""
    print("\n=== Testing Both GPT-5 Models ===")
    
    models = ["gpt-5-nano", "gpt-5-mini"]
    results = {}
    
    for model in models:
        try:
            print(f"  Testing {model}...")
            chat = OpenAIChat(model_name=model)
            result = chat.invoke("Say hello in one word.")
            results[model] = f"✅ {result}"
            print(f"  {model}: {result}")
        except Exception as e:
            results[model] = f"❌ {str(e)}"
            print(f"  {model}: Error - {e}")
    
    success = all("✅" in result for result in results.values())
    if success:
        print("✅ Both models work successfully")
    else:
        print("❌ Some models failed")
    
    return success

def test_basic_api_call():
    """Test a basic API call to OpenAI."""
    print("=== Testing Basic API Call ===")
    
    chat = OpenAIChat(model_name="gpt-5-nano")  # Explicitly use gpt-5-nano
    
    try:
        result = chat.invoke("Hello! Just say 'Hi' back in one word.")
        print(f"✅ Basic API call successful: {result}")
        return True
    except Exception as e:
        print(f"❌ Basic API call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_reasoning():
    """Test API call with reasoning parameter."""
    print("\n=== Testing Reasoning Parameter ===")
    
    chat = OpenAIChat(model_name="gpt-5-nano")  # Explicitly use gpt-5-nano
    
    try:
        result = chat.invoke(
            "What is 1+1? Just give the number.", 
            reasoning={"effort": "minimal"}
        )
        print(f"✅ Reasoning parameter works: {result}")
        return True
    except Exception as e:
        print(f"❌ Reasoning parameter failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_verbosity_control():
    """Test verbosity parameter."""
    print("\n=== Testing Verbosity Control ===")
    
    chat = OpenAIChat(model_name="gpt-5-mini")  # Test with gpt-5-mini
    
    try:
        result = chat.invoke(
            "Explain Python. Be very brief.", 
            verbosity="low"
        )
        print(f"✅ Verbosity control works: {result}")
        return True
    except Exception as e:
        print(f"❌ Verbosity control failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_function_calling():
    """Test function calling with the Responses API."""
    print("\n=== Testing Function Calling ===")
    
    def get_temperature(city: str) -> str:
        """Get the current temperature for a city."""
        return f"The temperature in {city} is 22°C"
    
    chat = OpenAIChat(model_name="gpt-5-nano")  # Test function calling with gpt-5-nano
    
    try:
        result = chat.invoke(
            "What's the temperature in Paris?", 
            tools=[get_temperature]
        )
        print(f"✅ Function calling works: {result}")
        return True
    except Exception as e:
        print(f"❌ Function calling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_structured_output():
    """Test structured JSON output."""
    print("\n=== Testing Structured Output ===")
    
    schema = {
        "type": "object",
        "properties": {
            "greeting": {"type": "string"},
            "language": {"type": "string"}
        },
        "required": ["greeting", "language"],
        "additionalProperties": False
    }
    
    chat = OpenAIChat(model_name="gpt-5-mini")  # Test structured output with gpt-5-mini
    
    try:
        result = chat.invoke(
            "Say hello in French", 
            json_schema=schema
        )
        print(f"✅ Structured output works: {result}")
        print(f"   Type: {type(result)}")
        return True
    except Exception as e:
        print(f"❌ Structured output failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all API tests."""
    print("Testing OpenAI Chat with Real API Calls")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    tests = [
        test_both_models,
        test_basic_api_call,
        test_simple_reasoning,
        test_verbosity_control,
        test_function_calling,
        test_structured_output
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All API tests passed!")
    else:
        print("⚠️  Some tests failed - check the implementation")

if __name__ == "__main__":
    main()
