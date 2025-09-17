#!/usr/bin/env python3
"""
Comprehensive test suite for GroqChatResp class.
Tests all functionality to ensure it matches GroqChat behavior.
"""

import json
import sys
import os
from groq_chat import GroqChatResp, GroqChat
from calculator_tool import calculator
from search_tool import search_web

def test_basic_functionality():
    """Test basic text generation."""
    print("=== Testing Basic Functionality ===")
    
    try:
        # Test GroqChatResp
        chat_resp = GroqChatResp()
        print(f"GroqChatResp initialized with model: {chat_resp.model}")
        print(f"Using Responses API endpoint: {chat_resp.base_url}/responses")
        
        # Basic text generation
        response = chat_resp.invoke("Say 'Hello from Responses API' in exactly 5 words.")
        print(f"Response: {response}")
        print(f"Response type: {type(response)}")
        
        # Check if response ID is tracked
        print(f"Current response ID: {chat_resp.current_response_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_comparison_with_groq_chat():
    """Compare GroqChatResp with GroqChat to ensure similar behavior."""
    print("\n=== Testing Comparison with GroqChat ===")
    
    try:
        # Initialize both classes
        chat_original = GroqChat()
        chat_resp = GroqChatResp()
        
        query = "What is 2 + 2? Answer with just the number."
        
        # Test original GroqChat
        print("Testing GroqChat (Chat Completions API)...")
        response_original = chat_original.invoke(query)
        print(f"GroqChat response: {response_original}")
        print(f"GroqChat response type: {type(response_original)}")
        
        # Test GroqChatResp
        print("\nTesting GroqChatResp (Responses API)...")
        response_resp = chat_resp.invoke(query)
        print(f"GroqChatResp response: {response_resp}")
        print(f"GroqChatResp response type: {type(response_resp)}")
        
        # Both should return strings for basic queries
        if isinstance(response_original, str) and isinstance(response_resp, str):
            print("✅ Both classes return string responses for basic queries")
        else:
            print("❌ Response types don't match")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def test_tool_calling():
    """Test tool calling functionality and 5-key return structure."""
    print("\n=== Testing Tool Calling ===")
    
    try:
        chat_resp = GroqChatResp()
        
        # Test single tool call
        print("Testing single tool call...")
        response = chat_resp.invoke("What is 15 + 27?", tools=[calculator])
        
        print(f"Response type: {type(response)}")
        
        if isinstance(response, dict):
            print("✅ Tool call returned dictionary")
            
            # Check for required keys
            required_keys = ['tool_name', 'tool_return', 'text', 'raw', 'tool_messages']
            for key in required_keys:
                if key in response:
                    print(f"✅ Key '{key}' found")
                else:
                    print(f"❌ Key '{key}' missing")
                    return False
            
            # Verify tool_messages structure
            tool_messages = response.get('tool_messages', [])
            if len(tool_messages) == 2:
                print("✅ tool_messages has correct length (2)")
                
                # Check assistant message
                assistant_msg = tool_messages[0]
                if assistant_msg.get('role') == 'assistant' and 'tool_calls' in assistant_msg:
                    print("✅ Assistant message with tool_calls found")
                else:
                    print("❌ Assistant message format incorrect")
                    return False
                
                # Check tool message
                tool_msg = tool_messages[1]
                if (tool_msg.get('role') == 'tool' and 
                    'content' in tool_msg and 
                    'tool_call_id' in tool_msg):
                    print("✅ Tool message with correct format found")
                else:
                    print("❌ Tool message format incorrect")
                    return False
            else:
                print(f"❌ tool_messages length is {len(tool_messages)}, expected 2")
                return False
            
            print(f"Tool name: {response['tool_name']}")
            print(f"Tool return: {response['tool_return']}")
            print(f"Response text: {response['text']}")
            
        else:
            print(f"❌ Expected dict, got {type(response)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Tool calling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation_continuity():
    """Test conversation continuity with response IDs."""
    print("\n=== Testing Conversation Continuity ===")
    
    try:
        chat_resp = GroqChatResp()
        
        # First message
        response1 = chat_resp.invoke("My name is Alice. Remember this.")
        print(f"First response: {response1}")
        print(f"Response ID after first message: {chat_resp.current_response_id}")
        
        # Second message (should remember context)
        response2 = chat_resp.invoke("What is my name?")
        print(f"Second response: {response2}")
        print(f"Response ID after second message: {chat_resp.current_response_id}")
        
        # Check if conversation history is maintained
        print(f"Conversation history length: {len(chat_resp.conversation_history)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversation continuity test failed: {e}")
        return False

def test_structured_output():
    """Test structured JSON output functionality."""
    print("\n=== Testing Structured Output ===")
    
    try:
        chat_resp = GroqChatResp()
        
        # Define a simple JSON schema
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["name", "age", "city"]
        }
        
        response = chat_resp.invoke(
            "Create a person with name John, age 25, from New York. Return as JSON.",
            json_schema=schema
        )
        
        print(f"Structured response: {response}")
        print(f"Response type: {type(response)}")
        
        if isinstance(response, dict):
            print("✅ Structured output returned dictionary")
            
            # Check required fields
            required_fields = ['name', 'age', 'city']
            for field in required_fields:
                if field in response:
                    print(f"✅ Field '{field}' found: {response[field]}")
                else:
                    print(f"❌ Field '{field}' missing")
                    return False
        else:
            print(f"❌ Expected dict, got {type(response)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Structured output test failed: {e}")
        return False

def test_error_handling():
    """Test error handling scenarios."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test with invalid API key
        print("Testing invalid API key...")
        try:
            chat_resp = GroqChatResp(api_key="invalid_key")
            response = chat_resp.invoke("Hello")
            print("❌ Should have failed with invalid API key")
            return False
        except Exception as e:
            print(f"✅ Correctly handled invalid API key: {type(e).__name__}")
        
        # Test with missing required parameters
        print("Testing missing parameters...")
        try:
            chat_resp = GroqChatResp()
            response = chat_resp.invoke()  # No query or messages
            print("❌ Should have failed with missing parameters")
            return False
        except ValueError as e:
            print(f"✅ Correctly handled missing parameters: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_conversation_storage_config():
    """Test conversation storage configuration."""
    print("\n=== Testing Conversation Storage Configuration ===")
    
    try:
        chat_resp = GroqChatResp()
        
        # Test default (should be True)
        print(f"Default conversation storage: {chat_resp.store_conversations}")
        
        # Test disabling storage
        chat_resp.configure_conversation_storage(False)
        print(f"After disabling: {chat_resp.store_conversations}")
        
        # Test enabling storage
        chat_resp.configure_conversation_storage(True)
        print(f"After enabling: {chat_resp.store_conversations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversation storage config test failed: {e}")
        return False

def test_extend_conversation_with_tool_messages():
    """Test extending conversation with tool messages."""
    print("\n=== Testing Extend Conversation with Tool Messages ===")
    
    try:
        chat_resp = GroqChatResp()
        
        # Get initial conversation history length
        initial_length = len(chat_resp.conversation_history)
        print(f"Initial conversation history length: {initial_length}")
        
        # Make a tool call
        response = chat_resp.invoke("What is 10 * 5?", tools=[calculator])
        
        if isinstance(response, dict) and 'tool_messages' in response:
            # Test the extend method
            chat_resp.extend_conversation_with_tool_messages(response)
            
            final_length = len(chat_resp.conversation_history)
            print(f"Final conversation history length: {final_length}")
            
            # Should have added the tool messages
            expected_increase = len(response['tool_messages'])
            actual_increase = final_length - initial_length - 2  # -2 for user and assistant messages already added
            
            if actual_increase == expected_increase:
                print(f"✅ Tool messages correctly added ({expected_increase} messages)")
            else:
                print(f"❌ Expected {expected_increase} additional messages, got {actual_increase}")
                return False
        else:
            print("❌ Tool call didn't return expected format")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Extend conversation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting GroqChatResp Test Suite")
    print("=" * 50)
    
    # Check if API key is available
    if not os.getenv('GROQ_API_KEY'):
        print("❌ GROQ_API_KEY environment variable not set")
        print("Please set your Groq API key: export GROQ_API_KEY='your_key_here'")
        return False
    
    tests = [
        test_basic_functionality,
        test_comparison_with_groq_chat,
        test_tool_calling,
        test_conversation_continuity,
        test_structured_output,
        test_error_handling,
        test_conversation_storage_config,
        test_extend_conversation_with_tool_messages
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED with exception: {e}")
        
        print("-" * 30)
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! GroqChatResp is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)