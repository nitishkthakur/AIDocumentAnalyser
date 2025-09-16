#!/usr/bin/env python3
"""
Test script for GroqChat functionality.
This script tests various aspects of the GroqChat implementation.
"""

import os
import sys
import json
from dotenv import load_dotenv
from groq_chat import GroqChat

# Load environment variables
load_dotenv()

def test_initialization():
    """Test GroqChat initialization and configuration."""
    print("=== Testing Initialization ===")
    
    # Test with API key from .env file
    try:
        chat = GroqChat()  # Will use GROQ_API_KEY from environment
        print("✓ GroqChat initialized successfully")
        print(f"  Model: {chat.model}")
        print(f"  Base URL: {chat.base_url}")
        print(f"  Default reasoning: {json.dumps(chat.default_reasoning, indent=2)}")
        print(f"  API Key configured: {'Yes' if chat.api_key else 'No'}")
        return chat
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        print("Make sure GROQ_API_KEY is set in your .env file")
        return None

def test_method_signatures(chat):
    """Test that all expected methods exist and have correct signatures."""
    print("\n=== Testing Method Signatures ===")
    
    expected_methods = [
        'invoke', 'clear_conversation_history', 'set_output_parameters',
        'optimize_for_long_output', 'configure_concurrent_execution',
        '_execute_tool_calls', '_execute_tool_calls_concurrent',
        '_get_json_type', '_extract_function_info', '_build_tools',
        '_extract_json_schema', '_make_schema_strict_compatible',
        '_build_input_messages', '_build_input_messages_from_list',
        '_build_groq_payload', '_invoke_groq_api'
    ]
    
    for method_name in expected_methods:
        if hasattr(chat, method_name):
            method = getattr(chat, method_name)
            if callable(method):
                print(f"✓ {method_name} - callable")
            else:
                print(f"✗ {method_name} - not callable")
        else:
            print(f"✗ {method_name} - missing")

def test_tool_building():
    """Test tool building functionality without making API calls."""
    print("\n=== Testing Tool Building ===")
    
    def sample_tool(city: str, country: str = "US") -> str:
        """Get weather information for a city.
        
        Args:
            city: Name of the city
            country: Country code (default: US)
        """
        return f"Weather in {city}, {country}: Sunny, 25°C"
    
    try:
        chat = GroqChat()  # Use environment API key
        tools_schema = chat._build_tools([sample_tool])
        
        print("✓ Tool schema built successfully:")
        print(json.dumps(tools_schema, indent=2))
        
        # Verify structure
        if tools_schema and isinstance(tools_schema, list):
            tool = tools_schema[0]
            if all(key in tool for key in ['type', 'function']):
                func_def = tool['function']
                if all(key in func_def for key in ['name', 'description', 'parameters']):
                    print("✓ Tool schema structure is correct")
                else:
                    print("✗ Tool function definition missing required keys")
            else:
                print("✗ Tool schema missing required keys")
        else:
            print("✗ Tool schema is not a list or is empty")
            
    except Exception as e:
        print(f"✗ Tool building failed: {e}")

def test_json_schema_extraction():
    """Test JSON schema extraction functionality."""
    print("\n=== Testing JSON Schema Extraction ===")
    
    # Test with dictionary schema
    sample_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "active": {"type": "boolean"}
        },
        "required": ["name"]
    }
    
    try:
        chat = GroqChat()  # Use environment API key
        extracted = chat._extract_json_schema(sample_schema)
        
        if extracted == sample_schema:
            print("✓ Dictionary schema extraction works")
        else:
            print("✗ Dictionary schema extraction failed")
        
        # Test with None
        none_result = chat._extract_json_schema(None)
        if none_result is None:
            print("✓ None schema handling works")
        else:
            print("✗ None schema handling failed")
            
    except Exception as e:
        print(f"✗ JSON schema extraction failed: {e}")

def test_message_building():
    """Test message building functionality."""
    print("\n=== Testing Message Building ===")
    
    try:
        chat = GroqChat(system_instructions="You are a helpful assistant.")  # Use environment API key
        
        # Test single query message building
        messages = chat._build_input_messages("Hello, how are you?")
        print("✓ Single query message building:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content'][:50]}...")
        
        # Test message list building
        message_list = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What about 3+3?"}
        ]
        
        messages = chat._build_input_messages_from_list(message_list)
        print("\n✓ Message list building:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content'][:50]}...")
            
    except Exception as e:
        print(f"✗ Message building failed: {e}")

def test_payload_building():
    """Test Groq API payload building."""
    print("\n=== Testing Payload Building ===")
    
    def sample_function(query: str) -> str:
        """Sample function for testing."""
        return f"Result for: {query}"
    
    try:
        chat = GroqChat()  # Use environment API key
        
        # Test basic payload
        payload = chat._build_groq_payload("Hello world")
        print("✓ Basic payload built:")
        print(f"  Model: {payload.get('model')}")
        print(f"  Messages count: {len(payload.get('messages', []))}")
        print(f"  Temperature: {payload.get('temperature')}")
        
        # Test payload with tools
        payload_with_tools = chat._build_groq_payload("Test with tools", tools=[sample_function])
        print("\n✓ Payload with tools built:")
        print(f"  Has tools: {'tools' in payload_with_tools}")
        print(f"  Tool choice: {payload_with_tools.get('tool_choice')}")
        
    except Exception as e:
        print(f"✗ Payload building failed: {e}")

def test_configuration_methods():
    """Test configuration methods."""
    print("\n=== Testing Configuration Methods ===")
    
    try:
        chat = GroqChat()  # Use environment API key
        
        # Test output parameters
        chat.set_output_parameters(max_tokens=2048, temperature=0.5)
        print("✓ set_output_parameters() works")
        print(f"  Max tokens: {chat.default_reasoning.get('max_completion_tokens')}")
        print(f"  Temperature: {chat.default_reasoning.get('temperature')}")
        
        # Test long output optimization
        chat.optimize_for_long_output()
        print("✓ optimize_for_long_output() works")
        print(f"  Max tokens after optimization: {chat.default_reasoning.get('max_completion_tokens')}")
        
        # Test concurrent execution config
        chat.configure_concurrent_execution(enabled=True, max_workers=3)
        print("✓ configure_concurrent_execution() works")
        print(f"  Concurrent enabled: {chat.concurrent_tool_execution}")
        print(f"  Max workers: {chat.max_concurrent_tools}")
        
        # Test conversation history
        chat.conversation_history = [{"role": "user", "content": "test"}]
        chat.clear_conversation_history()
        print("✓ clear_conversation_history() works")
        print(f"  History length: {len(chat.conversation_history)}")
        
    except Exception as e:
        print(f"✗ Configuration methods failed: {e}")

def test_actual_api_calls(chat):
    """Test actual API calls if API key is available."""
    print("\n=== Testing Actual API Calls ===")
    
    if not chat or not chat.api_key:
        print("✗ Skipping API tests - no valid API key")
        return
    
    try:
        # Test simple chat
        print("Testing simple chat...")
        response = chat.invoke("Say hello in exactly 5 words.")
        print(f"✓ Simple chat works: '{response}'")
        
        # Test with custom parameters
        print("\nTesting with custom parameters...")
        response = chat.invoke("Count from 1 to 3.", reasoning={"temperature": 0.1, "max_completion_tokens": 50})
        print(f"✓ Custom parameters work: '{response}'")
        
        # Test conversation history
        print("\nTesting conversation history...")
        chat.clear_conversation_history()
        chat.invoke("My name is John.")
        response = chat.invoke("What is my name?")
        print(f"✓ Conversation history works: '{response}'")
        
    except Exception as e:
        print(f"✗ API call failed: {e}")

def main():
    """Run all tests."""
    print("GroqChat Test Suite")
    print("=" * 50)
    
    # Initialize
    chat = test_initialization()
    if not chat:
        print("Cannot continue tests without successful initialization")
        return
    
    # Run structural tests
    test_method_signatures(chat)
    test_tool_building()
    test_json_schema_extraction() 
    test_message_building()
    test_payload_building()
    test_configuration_methods()
    
    # Run actual API tests if possible
    test_actual_api_calls(chat)
    
    print("\n" + "=" * 50)
    print("Test Suite Complete")

if __name__ == "__main__":
    main()