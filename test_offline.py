#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/nitish/Documents/github/AIDocumentAnalyser')

from openai_chat import OpenAIChat

def test_initialization():
    """Test that the class initializes correctly."""
    try:
        # Use a mock API key
        chat = OpenAIChat(api_key="test-key")
        print("✅ Initialization: OK")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def test_tool_schema_building():
    """Test tool schema generation."""
    try:
        chat = OpenAIChat(api_key="test-key")
        
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: 22°C, sunny"
        
        schemas = chat._build_tools([get_weather])
        
        expected_structure = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"],
                "additionalProperties": False
            }
        }
        
        if schemas and len(schemas) == 1:
            schema = schemas[0]
            if (schema.get("type") == "function" and 
                schema.get("name") == "get_weather" and
                schema.get("description") == "Get weather for a city." and
                "parameters" in schema):
                print("✅ Tool schema building: OK")
                print(f"   Generated schema: {schema}")
                return True
        
        print(f"❌ Tool schema building failed: {schemas}")
        return False
    except Exception as e:
        print(f"❌ Tool schema building failed: {e}")
        return False

def test_input_message_building():
    """Test input message building for Responses API."""
    try:
        chat = OpenAIChat(api_key="test-key", system_instructions="You are a helpful assistant.")
        
        messages = chat._build_input_messages("Hello, how are you?")
        
        expected = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        if messages == expected:
            print("✅ Input message building: OK")
            print(f"   Generated messages: {messages}")
            return True
        else:
            print(f"❌ Input message building failed:")
            print(f"   Expected: {expected}")
            print(f"   Got: {messages}")
            return False
    except Exception as e:
        print(f"❌ Input message building failed: {e}")
        return False

def test_json_schema_handling():
    """Test JSON schema extraction and processing."""
    try:
        chat = OpenAIChat(api_key="test-key")
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        
        processed = chat._extract_json_schema(schema)
        
        if processed and processed.get("additionalProperties") == False:
            print("✅ JSON schema handling: OK")
            print(f"   Processed schema: {processed}")
            return True
        else:
            print(f"❌ JSON schema handling failed: {processed}")
            return False
    except Exception as e:
        print(f"❌ JSON schema handling failed: {e}")
        return False

def test_payload_structure():
    """Test that payload structure is correct for Responses API."""
    try:
        chat = OpenAIChat(api_key="test-key")
        
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: 22°C, sunny"
        
        # Manually build what the payload should look like
        input_messages = chat._build_input_messages("What's the weather in London?")
        tool_schemas = chat._build_tools([get_weather])
        
        # Verify the structures
        if (isinstance(input_messages, list) and 
            len(input_messages) > 0 and 
            isinstance(tool_schemas, list) and 
            len(tool_schemas) > 0):
            print("✅ Payload structure: OK")
            print(f"   Input messages: {input_messages}")
            print(f"   Tool schemas: {tool_schemas}")
            return True
        else:
            print(f"❌ Payload structure failed")
            return False
    except Exception as e:
        print(f"❌ Payload structure failed: {e}")
        return False

def main():
    """Run all offline tests."""
    print("Testing OpenAI Chat Implementation (Offline Tests)")
    print("=" * 50)
    
    tests = [
        test_initialization,
        test_tool_schema_building,
        test_input_message_building,
        test_json_schema_handling,
        test_payload_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All offline tests passed!")
        print("📝 Note: The implementation is now using only the Responses API")
        print("📝 Note: Network connectivity is required for actual API calls")
    else:
        print("⚠️  Some tests failed - check the implementation")

if __name__ == "__main__":
    main()
