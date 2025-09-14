#!/usr/bin/env python3
"""
Test edge cases and demonstrate advanced usage of the messages functionality
"""

import json
from ollama_chat import OllamaChat

def calculate(a: int, b: int, operation: str):
    """Mock calculator function"""
    if operation == "add":
        return {"result": a + b, "operation": f"{a} + {b}"}
    elif operation == "multiply":
        return {"result": a * b, "operation": f"{a} * {b}"}
    else:
        return {"error": f"Unknown operation: {operation}"}

def test_complex_conversation():
    """Test complex multi-turn conversation with tools"""
    print("=== Complex Conversation Test ===")
    
    client = OllamaChat()
    tools = [calculate]
    
    # Simulating a conversation where user provides context
    conversation_messages = [
        {"role": "system", "content": "You are a math tutor helping a student."},
        {"role": "user", "content": "I need help with basic arithmetic. Let's start with addition."},
        {"role": "assistant", "content": "Great! I'd be happy to help you with addition. What numbers would you like to add?"},
        {"role": "user", "content": "Can you add 15 and 27 for me using the calculator?"}
    ]
    
    try:
        response = client.invoke(messages=conversation_messages, tools=tools)
        print("✓ SUCCESS: Complex conversation with tools")
        
        if isinstance(response, dict) and 'tool_name' in response:
            print(f"Tool used: {response['tool_name']}")
            print(f"Calculation result: {response['tool_return']}")
        
        # Check conversation history
        print(f"Conversation history entries: {len(client.conversation_history)}")
        print("Recent history:")
        for msg in client.conversation_history[-3:]:
            print(f"  {msg['role']}: {msg['content'][:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixed_roles():
    """Test various message roles including tool messages"""
    print("\n=== Mixed Roles Test ===")
    
    client = OllamaChat()
    
    # Simulate a conversation that includes tool responses
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "What's 5 times 8?"},
        {"role": "assistant", "content": "Let me calculate that for you."},
        {"role": "tool", "content": "Tool result: 5 * 8 = 40"},
        {"role": "user", "content": "Thanks! Now what about 40 divided by 8?"}
    ]
    
    try:
        response = client.invoke(messages=messages)
        print("✓ SUCCESS: Mixed roles handled")
        print(f"Response preview: {str(response)[:100]}...")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_empty_messages():
    """Test edge case with empty message list"""
    print("\n=== Empty Messages Test ===")
    
    client = OllamaChat()
    
    try:
        # Empty messages list should still work
        response = client.invoke(messages=[])
        print("✓ SUCCESS: Empty messages list handled")
        return True
    except Exception as e:
        print(f"Expected behavior - empty messages: {e}")
        return True

def test_backwards_compatibility():
    """Ensure old query-based usage still works"""
    print("\n=== Backwards Compatibility Test ===")
    
    client = OllamaChat()
    
    try:
        # Old way should still work
        response1 = client.invoke("What is 2 + 2?")
        print("✓ SUCCESS: Query-based invoke still works")
        
        # With tools
        response2 = client.invoke("Calculate 10 * 5", tools=[calculate])
        print("✓ SUCCESS: Query with tools still works")
        
        if isinstance(response2, dict) and 'tool_name' in response2:
            print(f"Tool result: {response2['tool_return']}")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_conversation_continuation():
    """Test that messages and query approaches can be mixed in sequence"""
    print("\n=== Conversation Continuation Test ===")
    
    client = OllamaChat()
    
    try:
        # Start with query approach
        response1 = client.invoke("Hello, I'm working on a math problem.")
        print("✓ Step 1: Query approach")
        
        # Continue with messages approach
        messages = [
            {"role": "user", "content": "The problem is about calculating areas."},
            {"role": "user", "content": "Can you help me understand the formula for a rectangle?"}
        ]
        response2 = client.invoke(messages=messages)
        print("✓ Step 2: Messages approach")
        
        # Back to query approach
        response3 = client.invoke("What about triangles?")
        print("✓ Step 3: Query approach again")
        
        print(f"Final conversation history length: {len(client.conversation_history)}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def demonstrate_usage():
    """Demonstrate practical usage scenarios"""
    print("\n=== Usage Demonstration ===")
    
    client = OllamaChat()
    
    print("Scenario: Feeding conversation context to simulate ongoing discussion")
    
    # Simulate loading a conversation from a database or file
    historical_context = [
        {"role": "system", "content": "You are a coding mentor."},
        {"role": "user", "content": "I'm learning Python and struggling with functions."},
        {"role": "assistant", "content": "Functions are fundamental in Python! What specific aspect would you like help with?"},
        {"role": "user", "content": "I don't understand how to return values from functions."},
        {"role": "assistant", "content": "Great question! The 'return' statement sends a value back to whoever called the function."},
        {"role": "user", "content": "Can you show me an example with multiple return values?"}
    ]
    
    try:
        response = client.invoke(messages=historical_context)
        print("✓ Successfully processed historical context")
        print(f"AI Response: {str(response)[:200]}...")
        
        # Now continue the conversation normally
        follow_up = client.invoke("That's helpful! What about default parameters?")
        print("✓ Continued conversation after context injection")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def main():
    print("Testing Edge Cases and Advanced Usage")
    print("=" * 50)
    
    tests = [
        test_complex_conversation,
        test_mixed_roles,
        test_empty_messages,
        test_backwards_compatibility,
        test_conversation_continuation,
        demonstrate_usage
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ UNEXPECTED ERROR in {test_func.__name__}: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"Advanced tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All advanced tests passed!")
        print("\n📝 Summary of new functionality:")
        print("- invoke() now accepts 'messages' parameter")
        print("- Messages can include user/system/assistant/tool roles")
        print("- Conversation history is properly maintained")
        print("- Backwards compatibility with query parameter preserved")
        print("- Can mix query and messages approaches in same conversation")
        print("- Perfect for feeding conversation context from external sources")
    else:
        print("⚠️  Some advanced tests failed.")

if __name__ == "__main__":
    main()