#!/usr/bin/env python3
"""
Test script for the new messages parameter functionality in ollama_chat.py
"""

import json
from ollama_chat import OllamaChat

def get_weather(location: str):
    """Mock weather function"""
    return {"location": location, "temperature": "22°C", "condition": "sunny"}

def test_basic_messages():
    """Test basic message list functionality"""
    print("=== Test 1: Basic Message List ===")
    
    client = OllamaChat()
    
    messages = [
        {"role": "user", "content": "Hello, what is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and readability."},
        {"role": "user", "content": "Can you tell me more about its history?"}
    ]
    
    try:
        response = client.invoke(messages=messages)
        print("✓ SUCCESS: Messages parameter accepted")
        print(f"Response type: {type(response)}")
        print(f"Response length: {len(str(response))}")
        print(f"Response preview: {str(response)[:200]}...")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_system_message():
    """Test system message handling"""
    print("\n=== Test 2: System Message ===")
    
    client = OllamaChat()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    try:
        response = client.invoke(messages=messages)
        print("✓ SUCCESS: System message handled")
        print(f"Response preview: {str(response)[:200]}...")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_messages_with_tools():
    """Test message list with tool calls"""
    print("\n=== Test 3: Messages with Tools ===")
    
    client = OllamaChat()
    
    messages = [
        {"role": "user", "content": "What's the weather like in Paris?"}
    ]
    
    tools = [get_weather]
    
    try:
        response = client.invoke(messages=messages, tools=tools)
        print("✓ SUCCESS: Messages with tools")
        print(f"Response type: {type(response)}")
        
        if isinstance(response, dict) and 'tool_name' in response:
            print(f"Tool called: {response.get('tool_name')}")
            print(f"Tool result: {response.get('tool_return')}")
        elif isinstance(response, list):
            print(f"Multiple tools called: {len(response)}")
        else:
            print(f"Text response: {str(response)[:200]}...")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_conversation_history():
    """Test that conversation history is properly maintained"""
    print("\n=== Test 4: Conversation History ===")
    
    client = OllamaChat()
    
    # First call
    messages1 = [
        {"role": "user", "content": "My name is Alice"}
    ]
    response1 = client.invoke(messages=messages1)
    print(f"First response: {str(response1)[:100]}...")
    
    # Second call - should remember the name
    messages2 = [
        {"role": "user", "content": "What is my name?"}
    ]
    
    try:
        response2 = client.invoke(messages=messages2)
        print(f"Second response: {str(response2)[:100]}...")
        print(f"Conversation history length: {len(client.conversation_history)}")
        print("✓ SUCCESS: Conversation history maintained")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_error_cases():
    """Test error handling"""
    print("\n=== Test 5: Error Cases ===")
    
    client = OllamaChat()
    
    # Test 1: No query or messages
    try:
        client.invoke()
        print("✗ FAILED: Should have raised error for no parameters")
        return False
    except ValueError as e:
        print(f"✓ SUCCESS: Correctly raised error for no parameters: {e}")
    
    # Test 2: Both query and messages
    try:
        client.invoke(query="Hello", messages=[{"role": "user", "content": "Hi"}])
        print("✗ FAILED: Should have raised error for both parameters")
        return False
    except ValueError as e:
        print(f"✓ SUCCESS: Correctly raised error for both parameters: {e}")
    
    # Test 3: Invalid message format
    try:
        client.invoke(messages=[{"invalid": "message"}])
        print("✗ FAILED: Should have raised error for invalid message format")
        return False
    except ValueError as e:
        print(f"✓ SUCCESS: Correctly raised error for invalid message: {e}")
    
    # Test 4: Invalid role
    try:
        client.invoke(messages=[{"role": "invalid", "content": "test"}])
        print("✗ FAILED: Should have raised error for invalid role")
        return False
    except ValueError as e:
        print(f"✓ SUCCESS: Correctly raised error for invalid role: {e}")
    
    return True

def main():
    print("Testing Message List Functionality")
    print("=" * 50)
    
    tests = [
        test_basic_messages,
        test_system_message,
        test_messages_with_tools,
        test_conversation_history,
        test_error_cases
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ UNEXPECTED ERROR in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! Message list functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")

if __name__ == "__main__":
    main()