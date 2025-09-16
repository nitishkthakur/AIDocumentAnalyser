#!/usr/bin/env python3
"""
Test tool calling functionality for GroqChat.
"""

import json
from dotenv import load_dotenv
from groq_chat import GroqChat

# Load environment variables
load_dotenv()

def get_weather(city: str, country: str = "US") -> str:
    """Get weather information for a city.
    
    Args:
        city: Name of the city
        country: Country code (default: US)
    """
    weather_data = {
        "Paris": "22°C, partly cloudy",
        "London": "15°C, rainy", 
        "New York": "18°C, sunny",
        "Tokyo": "25°C, humid"
    }
    return weather_data.get(city, f"{city}, {country}: 20°C, mild weather")

def calculate(operation: str, a: float, b: float) -> float:
    """Perform basic mathematical operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b != 0:
            return a / b
        else:
            return "Error: Division by zero"
    else:
        return "Error: Unknown operation"

def get_time(timezone: str = "UTC") -> str:
    """Get current time for a timezone.
    
    Args:
        timezone: Timezone code (default: UTC)
    """
    import datetime
    return f"Current time in {timezone}: {datetime.datetime.now().strftime('%H:%M:%S')}"

def test_single_tool_call():
    """Test single tool calling."""
    print("=== Testing Single Tool Call ===")
    
    try:
        chat = GroqChat()
        
        # Test weather function
        result = chat.invoke("What's the weather in Paris?", tools=[get_weather])
        
        print(f"Query: What's the weather in Paris?")
        if isinstance(result, dict) and 'tool_name' in result:
            print(f"✓ Tool called: {result['tool_name']}")
            print(f"✓ Tool result: {result['tool_return']}")
            print(f"✓ Assistant text: {result.get('text', 'No text')}")
        else:
            print(f"✗ Unexpected result format: {result}")
            
    except Exception as e:
        print(f"✗ Single tool call failed: {e}")

def test_multiple_tools():
    """Test with multiple available tools."""
    print("\n=== Testing Multiple Available Tools ===")
    
    try:
        chat = GroqChat()
        chat.clear_conversation_history()
        
        # Test with multiple tools available
        tools = [get_weather, calculate, get_time]
        result = chat.invoke("What is 15 multiplied by 4?", tools=tools)
        
        print(f"Query: What is 15 multiplied by 4?")
        if isinstance(result, dict) and 'tool_name' in result:
            print(f"✓ Tool called: {result['tool_name']}")
            print(f"✓ Tool result: {result['tool_return']}")
        else:
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"✗ Multiple tools test failed: {e}")

def test_tool_with_conversation():
    """Test tool calling within a conversation."""
    print("\n=== Testing Tool Call in Conversation ===")
    
    try:
        chat = GroqChat()
        chat.clear_conversation_history()
        
        # Start conversation
        chat.invoke("Hello, I'm planning a trip.")
        
        # Ask for weather with tool
        result = chat.invoke("Can you check the weather in London for me?", tools=[get_weather])
        
        print(f"Query: Can you check the weather in London for me?")
        if isinstance(result, dict) and 'tool_name' in result:
            print(f"✓ Tool called: {result['tool_name']}")
            print(f"✓ Tool result: {result['tool_return']}")
        else:
            print(f"Result: {result}")
            
        # Continue conversation
        follow_up = chat.invoke("Is that good weather for sightseeing?")
        print(f"Follow-up: {follow_up}")
        
    except Exception as e:
        print(f"✗ Conversation tool test failed: {e}")

def test_concurrent_tool_execution():
    """Test concurrent tool execution with multiple tool calls."""
    print("\n=== Testing Concurrent Tool Execution ===")
    
    def slow_function_1(name: str) -> str:
        """A function that simulates slow processing."""
        import time
        time.sleep(1)  # Simulate processing time
        return f"Processed {name} in function 1"
    
    def slow_function_2(name: str) -> str:
        """Another function that simulates slow processing."""
        import time
        time.sleep(1)  # Simulate processing time
        return f"Processed {name} in function 2"
    
    try:
        chat = GroqChat()
        chat.configure_concurrent_execution(enabled=True, max_workers=3)
        
        # This query might trigger multiple tool calls if the model decides to
        tools = [slow_function_1, slow_function_2, get_time]
        
        import time
        start_time = time.time()
        
        result = chat.invoke("Get the current time", tools=tools)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Result: {result}")
        
        # Test if concurrent execution is working
        if isinstance(result, list):
            print(f"✓ Multiple tool calls executed: {len(result)} calls")
        elif isinstance(result, dict) and 'tool_name' in result:
            print(f"✓ Single tool call executed: {result['tool_name']}")
        else:
            print(f"✓ No tool calls made, got response: {result}")
            
    except Exception as e:
        print(f"✗ Concurrent execution test failed: {e}")

def test_tool_error_handling():
    """Test error handling in tool execution."""
    print("\n=== Testing Tool Error Handling ===")
    
    def error_function(action: str) -> str:
        """A function that may raise errors."""
        if action == "error":
            raise ValueError("Intentional error for testing")
        return f"Action completed: {action}"
    
    try:
        chat = GroqChat()
        
        # Test with function that raises error
        result = chat.invoke("Please call the error function with action 'error'", tools=[error_function])
        
        print(f"Query: Please call the error function with action 'error'")
        if isinstance(result, dict) and 'tool_name' in result:
            print(f"✓ Tool called: {result['tool_name']}")
            print(f"✓ Tool result (should contain error): {result['tool_return']}")
        else:
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")

def main():
    """Run all tool calling tests."""
    print("GroqChat Tool Calling Test Suite")
    print("=" * 50)
    
    test_single_tool_call()
    test_multiple_tools()
    test_tool_with_conversation()
    test_concurrent_tool_execution()
    test_tool_error_handling()
    
    print("\n" + "=" * 50)
    print("Tool Calling Test Suite Complete")

if __name__ == "__main__":
    main()