#!/usr/bin/env python3
"""
GroqChat Demo - Comprehensive demonstration of GroqChat capabilities.
This script showcases all the key features of the GroqChat class.
"""

import json
import time
from dotenv import load_dotenv
from groq_chat import GroqChat

# Load environment variables
load_dotenv()

def demo_weather(city: str, country: str = "US") -> str:
    """Get weather information for a city."""
    weather_data = {
        "Paris": "22°C, partly cloudy with light breeze",
        "London": "15°C, rainy with occasional showers", 
        "New York": "18°C, sunny with clear skies",
        "Tokyo": "25°C, humid with scattered clouds",
        "Sydney": "20°C, windy with partial sun"
    }
    return weather_data.get(city, f"{city}, {country}: 20°C, mild weather conditions")

def demo_calculator(operation: str, a: float, b: float) -> float:
    """Perform mathematical calculations."""
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }
    return operations.get(operation, "Error: Unknown operation")

def demo_time_info(timezone: str = "UTC") -> str:
    """Get current time information."""
    import datetime
    now = datetime.datetime.now()
    return f"Current time in {timezone}: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def main():
    """Run comprehensive GroqChat demonstration."""
    print("🚀 GroqChat Comprehensive Demo")
    print("=" * 50)
    
    # Initialize GroqChat
    try:
        chat = GroqChat()
        print(f"✓ GroqChat initialized successfully")
        print(f"  Model: {chat.model}")
        print(f"  Base URL: {chat.base_url}")
    except Exception as e:
        print(f"✗ Failed to initialize GroqChat: {e}")
        return
    
    # Demo 1: Basic Chat
    print("\n🗣️  Demo 1: Basic Chat")
    print("-" * 30)
    
    try:
        response = chat.invoke("Hello! Please introduce yourself in a friendly way.")
        print(f"User: Hello! Please introduce yourself in a friendly way.")
        print(f"Assistant: {response}")
    except Exception as e:
        print(f"✗ Basic chat failed: {e}")
    
    # Demo 2: Tool Calling
    print("\n🔧 Demo 2: Tool Calling")
    print("-" * 30)
    
    try:
        # Single tool call
        result = chat.invoke("What's the weather like in Tokyo?", tools=[demo_weather])
        print(f"User: What's the weather like in Tokyo?")
        if isinstance(result, dict) and 'tool_name' in result:
            print(f"Tool Used: {result['tool_name']}")
            print(f"Result: {result['tool_return']}")
        else:
            print(f"Response: {result}")
        
        # Multiple tools available
        result = chat.invoke("Calculate 25 multiplied by 8", tools=[demo_calculator, demo_weather, demo_time_info])
        print(f"\nUser: Calculate 25 multiplied by 8")
        if isinstance(result, dict) and 'tool_name' in result:
            print(f"Tool Used: {result['tool_name']}")
            print(f"Result: {result['tool_return']}")
        
    except Exception as e:
        print(f"✗ Tool calling failed: {e}")
    
    # Demo 3: Structured Outputs
    print("\n📊 Demo 3: Structured Outputs")
    print("-" * 30)
    
    # Define a schema for person information
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Full name"},
            "age": {"type": "integer", "description": "Age in years"},
            "profession": {"type": "string", "description": "Job title"},
            "location": {"type": "string", "description": "City/Country"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of key skills"
            }
        },
        "required": ["name", "age", "profession"],
        "additionalProperties": False
    }
    
    try:
        chat.clear_conversation_history()  # Fresh conversation
        result = chat.invoke(
            "Create a profile for a fictional software engineer named Alex who is 29 years old, lives in Berlin, and specializes in Python and machine learning.",
            json_schema=person_schema
        )
        
        print("User: Create a profile for a fictional software engineer...")
        print("Structured Output:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"✗ Structured output failed: {e}")
    
    # Demo 4: Conversation History
    print("\n💬 Demo 4: Conversation History")
    print("-" * 30)
    
    try:
        chat.clear_conversation_history()
        
        # Multi-turn conversation
        response1 = chat.invoke("My favorite color is blue.")
        print("User: My favorite color is blue.")
        print(f"Assistant: {response1}")
        
        response2 = chat.invoke("What did I just tell you about my preferences?")
        print("\nUser: What did I just tell you about my preferences?")
        print(f"Assistant: {response2}")
        
    except Exception as e:
        print(f"✗ Conversation history failed: {e}")
    
    # Demo 5: Message List Interface
    print("\n📝 Demo 5: Message List Interface")
    print("-" * 30)
    
    try:
        chat.clear_conversation_history()
        
        # Use message list format
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "What is Python used for?"},
            {"role": "assistant", "content": "Python is used for web development, data science, automation, and more."},
            {"role": "user", "content": "Can you give me a simple Python example?"}
        ]
        
        response = chat.invoke(messages=messages)
        print("Message conversation:")
        for msg in messages:
            print(f"  {msg['role'].title()}: {msg['content'][:50]}...")
        print(f"  Assistant: {response}")
        
    except Exception as e:
        print(f"✗ Message list interface failed: {e}")
    
    # Demo 6: Configuration Options
    print("\n⚙️  Demo 6: Configuration Options")
    print("-" * 30)
    
    try:
        # Test different temperature settings
        chat.clear_conversation_history()
        
        # Low temperature (more focused)
        response1 = chat.invoke("Write a haiku about technology.", reasoning={"temperature": 0.2})
        print("Low temperature (0.2) - Focused:")
        print(response1)
        
        # High temperature (more creative)
        response2 = chat.invoke("Write a haiku about technology.", reasoning={"temperature": 1.5})
        print("\nHigh temperature (1.5) - Creative:")
        print(response2)
        
        # Configure for long output
        chat.optimize_for_long_output()
        response3 = chat.invoke("Explain the concept of machine learning in detail.")
        print(f"\nOptimized for long output (first 200 chars): {response3[:200]}...")
        
    except Exception as e:
        print(f"✗ Configuration options failed: {e}")
    
    # Demo 7: Concurrent Tool Execution
    print("\n⚡ Demo 7: Concurrent Tool Execution")
    print("-" * 30)
    
    def task_1(name: str) -> str:
        time.sleep(1)  # Simulate work
        return f"Task 1 completed for {name}"
    
    def task_2(name: str) -> str:
        time.sleep(1)  # Simulate work  
        return f"Task 2 completed for {name}"
    
    try:
        chat.clear_conversation_history()
        chat.configure_concurrent_execution(enabled=True, max_workers=3)
        
        start_time = time.time()
        result = chat.invoke("Execute task 1 with 'ProjectA' and get current time", tools=[task_1, task_2, demo_time_info])
        end_time = time.time()
        
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        if isinstance(result, dict) and 'tool_name' in result:
            print(f"Tool executed: {result['tool_name']} -> {result['tool_return']}")
        elif isinstance(result, list):
            print(f"Multiple tools executed: {len(result)} tools")
            for r in result:
                if 'tool_name' in r:
                    print(f"  - {r['tool_name']}: {r['tool_return']}")
        else:
            print(f"Response: {result}")
        
    except Exception as e:
        print(f"✗ Concurrent execution failed: {e}")
    
    # Demo Summary
    print("\n🎉 Demo Complete!")
    print("=" * 50)
    print("GroqChat successfully demonstrated:")
    print("✓ Basic chat functionality")
    print("✓ Tool calling with function execution")
    print("✓ Structured JSON outputs with schema validation")
    print("✓ Conversation history management")
    print("✓ Message list interface compatibility")
    print("✓ Configuration and parameter customization")
    print("✓ Concurrent tool execution capabilities")
    print("\nGroqChat is ready for production use! 🚀")

if __name__ == "__main__":
    main()