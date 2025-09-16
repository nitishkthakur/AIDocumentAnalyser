#!/usr/bin/env python3
"""
Test script to verify the new tool_messages functionality in GroqChat.
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

def test_tool_messages():
    """Test the new tool_messages functionality."""
    
    print("=== Testing GroqChat Tool Messages Functionality ===\n")
    
    # Initialize GroqChat
    try:
        chat = GroqChat(model_name="meta-llama/llama-4-scout-17b-16e-instruct")
        print(f"✅ GroqChat initialized successfully")
        print(f"Model: {chat.model}\n")
    except Exception as e:
        print(f"❌ Failed to initialize GroqChat: {e}")
        return
    
    # Test single tool call
    print("=== Test 1: Single Tool Call ===")
    try:
        response = chat.invoke("What is 15 * 7?", tools=[calculator])
        print(f"Response type: {type(response)}")
        
        if isinstance(response, dict):
            print("✅ Tool call response received")
            print(f"Tool name: {response.get('tool_name')}")
            print(f"Tool return: {response.get('tool_return')}")
            print(f"Text: {response.get('text')}")
            
            # Check if tool_messages key exists
            if 'tool_messages' in response:
                print("✅ tool_messages key found!")
                tool_messages = response['tool_messages']
                print(f"Number of tool messages: {len(tool_messages)}")
                
                for i, msg in enumerate(tool_messages):
                    print(f"  Message {i+1}: role={msg.get('role')}, content={msg.get('content')[:50]}...")
                    if 'tool_call_id' in msg:
                        print(f"    tool_call_id: {msg.get('tool_call_id')}")
                
                # Test extending conversation history
                print("\n--- Testing conversation history extension ---")
                initial_history_len = len(chat.conversation_history)
                chat.extend_conversation_with_tool_messages(response)
                final_history_len = len(chat.conversation_history)
                print(f"History length before: {initial_history_len}")
                print(f"History length after: {final_history_len}")
                print(f"Added {final_history_len - initial_history_len} messages to history")
                
                # Show the last few messages in history
                print("Last few messages in history:")
                for msg in chat.conversation_history[-3:]:
                    print(f"  {msg.get('role')}: {str(msg.get('content', ''))[:50]}...")
                
            else:
                print("❌ tool_messages key not found in response")
        else:
            print(f"❌ Unexpected response type: {type(response)}")
            
    except Exception as e:
        print(f"❌ Error in single tool call test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60 + "\n")
    
    # Test using tool messages in a follow-up conversation
    print("=== Test 2: Follow-up Conversation ===")
    try:
        # Clear history first
        chat.clear_conversation_history()
        
        # First tool call
        response1 = chat.invoke("Calculate 8 * 9", tools=[calculator])
        if isinstance(response1, dict) and 'tool_messages' in response1:
            print("✅ First tool call successful")
            
            # Extend conversation history
            chat.extend_conversation_with_tool_messages(response1)
            
            # Follow-up call using the extended history
            response2 = chat.invoke("Now divide that result by 4", tools=[calculator])
            print("✅ Follow-up conversation completed")
            
            if isinstance(response2, dict):
                print(f"Second response: {response2.get('tool_return')}")
            
    except Exception as e:
        print(f"❌ Error in follow-up conversation test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tool_messages()