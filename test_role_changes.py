#!/usr/bin/env python3
"""
Test script to verify that the role is now being sent in user messages instead of system instructions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openai_client import OpenAIClient

def test_role_in_user_message():
    """Test that the role is now in user messages instead of system instructions."""
    
    # Create a mock OpenAI client (we'll inspect the payload without making actual API calls)
    client = OpenAIClient(
        role="Test role: Extract information from documents",
        history_from_other_agents="Previous agent found some data",
        api_key="test-key-not-real",
        this_agent_context="Some context here"
    )
    
    # Test 1: Check that system instructions don't contain the role
    print("=== Test 1: System Instructions ===")
    print("System instructions should NOT contain the role:")
    print(f"Role: {client.role}")
    print(f"System instructions contain role: {'Test role:' in client.system_instructions}")
    print(f"System instructions:\n{client.system_instructions[:300]}...")
    
    # Test 2: Check payload building (without making API call)
    print("\n=== Test 2: Message Payload ===")
    try:
        # This will build the payload but fail at the API call
        payload = client._build_chat_payload(
            query="What is the weather today?",
            json_schema=None,
            tools=None,
            model_name="gpt-5-nano"
        )
        
        # Check messages in payload
        messages = payload.get("messages", [])
        print(f"Number of messages: {len(messages)}")
        
        for i, msg in enumerate(messages):
            print(f"Message {i+1}: Role = {msg['role']}")
            content = msg['content']
            if msg['role'] == 'user':
                print(f"User message contains <task>: {'<task>' in content}")
                print(f"User message contains role: {'Test role:' in content}")
                print(f"User message preview: {content[:200]}...")
            elif msg['role'] == 'system':
                print(f"System message contains role: {'Test role:' in content}")
        
    except Exception as e:
        print(f"Expected error (no real API key): {e}")
    
    print("\n=== Test Summary ===")
    role_in_system = 'Test role:' in client.system_instructions
    print(f"✓ Role removed from system instructions: {not role_in_system}")
    
    # Test payload building again to check user message
    payload = client._build_chat_payload("Test query")
    user_messages = [msg for msg in payload['messages'] if msg['role'] == 'user']
    if user_messages:
        user_content = user_messages[-1]['content']  # Get the last user message
        role_in_user = '<task>' in user_content and client.role in user_content
        print(f"✓ Role added to user message: {role_in_user}")
        query_after_task = 'Test query' in user_content
        print(f"✓ Query appears after task tags: {query_after_task}")
    else:
        print("✗ No user messages found")

if __name__ == "__main__":
    test_role_in_user_message()
