#!/usr/bin/env python3
"""
Test the ollama_chat.py multiple tool call functionality with the exact format provided by user
"""

import json
from ollama_chat import OllamaChat

def get_current_weather(location: str, format: str = "celsius"):
    """Get current weather for a location"""
    # Mock weather data
    temp = "24°C" if format == "celsius" else "75°F"
    return {
        "location": location,
        "temperature": temp,
        "condition": "sunny",
        "format": format
    }

def get_air_quality(location: str):
    """Get air quality for a location"""
    # Mock air quality data
    return {
        "location": location,
        "aqi": 45,
        "quality": "good",
        "primary_pollutant": "pm2.5"
    }

def main():
    client = OllamaChat()
    
    # Test tools
    tools = [get_current_weather, get_air_quality]
    
    print("=== Testing Multiple Tool Call Format ===")
    print("Expected: List of dictionaries when multiple tools are called")
    print("Expected: Single dictionary when one tool is called")
    print()
    
    # Test 1: Multiple tool calls
    print("Test 1: Multiple Tool Calls")
    print("-" * 30)
    
    query1 = "Please get both the current weather and air quality for Toronto. Use celsius format for weather."
    
    try:
        result1 = client.invoke(query=query1, tools=tools)
        
        print(f"Result type: {type(result1)}")
        
        if isinstance(result1, list):
            print(f"✓ SUCCESS: Got list with {len(result1)} tool results")
            for i, tool_result in enumerate(result1, 1):
                print(f"\nTool Result {i}:")
                print(f"  tool_name: {tool_result.get('tool_name')}")
                print(f"  tool_return: {json.dumps(tool_result.get('tool_return'), indent=4)}")
                print(f"  has_text: {'text' in tool_result}")
                print(f"  has_raw: {'raw' in tool_result}")
        else:
            print(f"✗ UNEXPECTED: Got {type(result1)} instead of list")
            print(f"Result: {result1}")
            
    except Exception as e:
        print(f"✗ ERROR in multiple tool test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Single tool call
    print("Test 2: Single Tool Call")
    print("-" * 25)
    
    query2 = "Get the current weather for New York in fahrenheit."
    
    try:
        result2 = client.invoke(query=query2, tools=tools)
        
        print(f"Result type: {type(result2)}")
        
        if isinstance(result2, dict) and 'tool_name' in result2:
            print(f"✓ SUCCESS: Got single tool result dictionary")
            print(f"  tool_name: {result2.get('tool_name')}")
            print(f"  tool_return: {json.dumps(result2.get('tool_return'), indent=4)}")
            print(f"  has_text: {'text' in result2}")
            print(f"  has_raw: {'raw' in result2}")
        else:
            print(f"✗ UNEXPECTED: Got {type(result2)} format")
            print(f"Result: {result2}")
            
    except Exception as e:
        print(f"✗ ERROR in single tool test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()