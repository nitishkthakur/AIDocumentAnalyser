#!/usr/bin/env python3
"""
Quick verification of the tool_messages structure.
"""

from groq_chat import GroqChat
import json

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize and test
chat = GroqChat(model_name="meta-llama/llama-4-scout-17b-16e-instruct")
response = chat.invoke("What is 5 + 3?", tools=[calculator])

print("=== Response Structure ===")
print(f"Keys in response: {list(response.keys())}")
print()

print("=== Tool Messages Structure ===")
if 'tool_messages' in response:
    for i, msg in enumerate(response['tool_messages']):
        print(f"Message {i+1}:")
        print(f"  Role: {msg.get('role')}")
        print(f"  Content: {msg.get('content')}")
        if 'tool_call_id' in msg:
            print(f"  Tool Call ID: {msg.get('tool_call_id')}")
        if 'tool_calls' in msg:
            print(f"  Tool Calls: {len(msg.get('tool_calls', []))} calls")
        print()

print("=== Usage Example ===")
print("# To extend conversation history:")
print("chat.extend_conversation_with_tool_messages(response)")
print()
print("# Or manually:")
print("messages.extend(response['tool_messages'])")
print()
print("# Then continue conversation:")
print("next_response = chat.invoke('Follow up question', tools=[calculator])")