#!/usr/bin/env python3

import os
from openai_chat import OpenAIChat

def test_weather(city: str) -> str:
    """Get weather for a city."""
    return f"{city}: 24°C, sunny"

# Test with a simple tool call
if __name__ == "__main__":
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY not set - skipping test")
        exit(0)
    
    try:
        chat = OpenAIChat()
        print("Testing tool call functionality...")
        result = chat.invoke("What's the weather in Paris?", tools=[test_weather])
        print("Success! Tool call result:", result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
