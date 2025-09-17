#!/usr/bin/env python3
"""
Test script to verify that tool call history is properly maintained
to prevent repeated tool calls.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from groq_chat import GroqChat
from groq_chat_resp import GroqChatResp

def get_weather(city: str) -> str:
    """Get weather information for a city."""
    weather_data = {
        "paris": "Paris: 18°C, cloudy with light rain",
        "london": "London: 15°C, foggy",
        "tokyo": "Tokyo: 22°C, sunny",
        "new york": "New York: 20°C, partly cloudy"
    }
    return weather_data.get(city.lower(), f"{city}: Weather data not available")

def get_time(timezone: str = "UTC") -> str:
    """Get current time for a timezone."""
    import datetime
    time_data = {
        "utc": "2025-09-16 14:30:00 UTC",
        "est": "2025-09-16 10:30:00 EST", 
        "pst": "2025-09-16 07:30:00 PST",
        "jst": "2025-09-16 23:30:00 JST"
    }
    return time_data.get(timezone.lower(), f"Current time in {timezone}: 14:30:00")

def test_conversation_history_preservation(chat_class, class_name):
    """Test that tool call results are preserved in conversation history."""
    print(f"\n=== Testing {class_name} - Tool Call History Preservation ===")
    
    try:
        # Initialize chat client
        chat = chat_class()
        print(f"✓ {class_name} initialized successfully")
        
        # First interaction with tool call
        print("\n1. First tool call:")
        result1 = chat.invoke(
            "What's the weather in Paris?", 
            tools=[get_weather, get_time]
        )
        
        if isinstance(result1, dict) and 'tool_name' in result1:
            print(f"   Tool called: {result1['tool_name']}")
            print(f"   Result: {result1['tool_return']}")
            print(f"   Response: {result1['text']}")
        else:
            print(f"   Response: {result1}")
        
        # Check conversation history length
        history_len_after_first = len(chat.conversation_history)
        print(f"   Conversation history length: {history_len_after_first}")
        
        # Print conversation history for debugging
        print("   Conversation history:")
        for i, msg in enumerate(chat.conversation_history):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:50] + ('...' if len(msg.get('content', '')) > 50 else '')
            has_tool_calls = 'tool_calls' in msg
            print(f"     {i+1}. {role}: {content} {'[HAS_TOOL_CALLS]' if has_tool_calls else ''}")
        
        # Second interaction - should not repeat the same tool call
        print("\n2. Follow-up question:")
        result2 = chat.invoke(
            "Based on that weather information, should I bring an umbrella?",
            tools=[get_weather, get_time]  # Same tools available
        )
        
        if isinstance(result2, dict) and 'tool_name' in result2:
            print(f"   Tool called: {result2['tool_name']}")
            print(f"   Result: {result2['tool_return']}")
            print(f"   Response: {result2['text']}")
        else:
            print(f"   Response: {result2}")
        
        # Check conversation history length
        history_len_after_second = len(chat.conversation_history)
        print(f"   Conversation history length: {history_len_after_second}")
        
        # Print updated conversation history
        print("   Updated conversation history:")
        for i, msg in enumerate(chat.conversation_history):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:50] + ('...' if len(msg.get('content', '')) > 50 else '')
            has_tool_calls = 'tool_calls' in msg
            print(f"     {i+1}. {role}: {content} {'[HAS_TOOL_CALLS]' if has_tool_calls else ''}")
        
        # Verify tool call history is preserved
        tool_messages = [msg for msg in chat.conversation_history if msg.get('role') == 'tool']
        assistant_with_tools = [msg for msg in chat.conversation_history if msg.get('role') == 'assistant' and 'tool_calls' in msg]
        
        print(f"   Tool messages in history: {len(tool_messages)}")
        print(f"   Assistant messages with tool_calls: {len(assistant_with_tools)}")
        
        if len(tool_messages) > 0 and len(assistant_with_tools) > 0:
            print(f"   ✓ Tool call history is properly preserved")
            return True
        else:
            print(f"   ✗ Tool call history is NOT properly preserved")
            return False
            
    except Exception as e:
        print(f"✗ Error testing {class_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_sequential_calls(chat_class, class_name):
    """Test multiple sequential tool calls to ensure no repetition."""
    print(f"\n=== Testing {class_name} - Multiple Sequential Calls ===")
    
    try:
        chat = chat_class()
        
        # Clear any existing history
        chat.clear_conversation_history()
        
        queries = [
            "What's the weather in Tokyo?",
            "What time is it in JST timezone?", 
            "Based on the weather and time, what should I wear?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: {query}")
            result = chat.invoke(query, tools=[get_weather, get_time])
            
            if isinstance(result, dict) and 'tool_name' in result:
                print(f"   Tool: {result['tool_name']} -> {result['tool_return']}")
            else:
                print(f"   Response: {result}")
        
        # Check final conversation state
        print(f"\nFinal conversation history length: {len(chat.conversation_history)}")
        tool_calls_count = len([msg for msg in chat.conversation_history if msg.get('role') == 'assistant' and 'tool_calls' in msg])
        tool_responses_count = len([msg for msg in chat.conversation_history if msg.get('role') == 'tool'])
        
        print(f"Total tool calls made: {tool_calls_count}")
        print(f"Total tool responses recorded: {tool_responses_count}")
        
        return tool_calls_count == tool_responses_count
        
    except Exception as e:
        print(f"✗ Error in sequential test for {class_name}: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Tool Call History Preservation")
    print("=" * 50)
    
    # Test both implementations
    results = {}
    
    # Test GroqChat (Chat Completions API)
    try:
        results['GroqChat_history'] = test_conversation_history_preservation(GroqChat, "GroqChat")
        results['GroqChat_sequential'] = test_multiple_sequential_calls(GroqChat, "GroqChat")
    except Exception as e:
        print(f"Failed to test GroqChat: {e}")
        results['GroqChat_history'] = False
        results['GroqChat_sequential'] = False
    
    # Test GroqChatResp (Responses API)
    try:
        results['GroqChatResp_history'] = test_conversation_history_preservation(GroqChatResp, "GroqChatResp")
        results['GroqChatResp_sequential'] = test_multiple_sequential_calls(GroqChatResp, "GroqChatResp")
    except Exception as e:
        print(f"Failed to test GroqChatResp: {e}")
        results['GroqChatResp_history'] = False
        results['GroqChatResp_sequential'] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30} {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)