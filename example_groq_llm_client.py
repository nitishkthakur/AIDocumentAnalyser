#!/usr/bin/env python3
"""
Example usage of the GroqLLMClient class.
Demonstrates tool use, structured output, and conversation management.
"""

import math
from pydantic import BaseModel
from groq_chat_2 import GroqLLMClient


# Define some example tools
def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius.
    
    Args:
        radius: The radius of the circle
        
    Returns:
        The area of the circle
    """
    return math.pi * radius ** 2


def get_weather_info(city: str) -> dict:
    """Get weather information for a city.
    
    Args:
        city: The name of the city
        
    Returns:
        Weather information dictionary
    """
    # Mock weather data
    weather_db = {
        "New York": {"temp": "22°C", "condition": "Sunny", "humidity": "60%"},
        "London": {"temp": "16°C", "condition": "Cloudy", "humidity": "75%"},
        "Tokyo": {"temp": "25°C", "condition": "Rainy", "humidity": "80%"}
    }
    
    return weather_db.get(city, {"temp": "Unknown", "condition": "Unknown", "humidity": "Unknown"})


# Define Pydantic models for structured output
class CircleCalculation(BaseModel):
    """Result of circle area calculation."""
    radius: float
    area: float
    formula: str


class WeatherReport(BaseModel):
    """Weather report for a city."""
    city: str
    temperature: str
    condition: str
    humidity: str


def example_1_basic_usage():
    """Example 1: Basic text generation."""
    print("Example 1: Basic Usage")
    print("-" * 40)
    
    client = GroqLLMClient()
    raw_response, response = client.invoke("Tell me a fun fact about artificial intelligence.")
    print(f"Response: {response}")
    print(f"Raw response available: {raw_response is not None}")
    print()


def example_2_single_tool():
    """Example 2: Using a single tool."""
    print("Example 2: Single Tool Usage")
    print("-" * 40)
    
    client = GroqLLMClient()
    raw_response, response = client.invoke(
        "Calculate the area of a circle with radius 8",
        tools=[calculate_circle_area]
    )
    print(f"Response: {response}")
    print(f"Tool results: {client.tool_results}")
    print(f"Raw response available: {raw_response is not None}")
    print()


def example_3_multiple_tools():
    """Example 3: Using multiple tools."""
    print("Example 3: Multiple Tools Usage")
    print("-" * 40)
    
    client = GroqLLMClient()
    raw_response, response = client.invoke(
        "Calculate the area of a circle with radius 5, then get the weather for Tokyo",
        tools=[calculate_circle_area, get_weather_info]
    )
    print(f"Response: {response}")
    print(f"Tool results: {client.tool_results}")
    print(f"Raw response available: {raw_response is not None}")
    print()


def example_4_structured_output():
    """Example 4: Structured output with Pydantic."""
    print("Example 4: Structured Output")
    print("-" * 40)
    
    client = GroqLLMClient()
    raw_response, response = client.invoke(
        "Calculate the area of a circle with radius 6. Provide the result in structured format.",
        json_schema=CircleCalculation
    )
    print(f"Response type: {type(response)}")
    print(f"Response: {response}")
    print(f"Raw response available: {raw_response is not None}")
    if isinstance(response, CircleCalculation):
        print(f"  Radius: {response.radius}")
        print(f"  Area: {response.area}")
        print(f"  Formula: {response.formula}")
    print()


def example_5_conversation_state():
    """Example 5: Conversation state management."""
    print("Example 5: Conversation State")
    print("-" * 40)
    
    client = GroqLLMClient()
    
    # First interaction
    raw_response1, response1 = client.invoke("My favorite color is blue. Please remember this.")
    print(f"First response: {response1}")
    
    # Second interaction using context
    raw_response2, response2 = client.invoke("What did I tell you about my favorite color?")
    print(f"Second response: {response2}")
    
    # Show conversation history
    print(f"Conversation history length: {len(client.get_conversation_history())} messages")
    print()


def example_6_continuing_conversation():
    """Example 6: Continuing conversation without new message."""
    print("Example 6: Continuing Conversation")
    print("-" * 40)
    
    client = GroqLLMClient()
    
    # Set system instructions for context
    client.set_system_instructions("You are a helpful assistant that explains your calculations step by step.")
    
    # First, use a tool
    raw_response1, response1 = client.invoke(
        "Calculate the area of a circle with radius 7",
        tools=[calculate_circle_area]
    )
    print(f"Step 1 - Tool response: {response1}")
    print(f"Tool results: {client.tool_results}")
    
    # Now continue the conversation without providing a new message
    # The LLM will look at the tool results and system instructions to respond
    print("\nContinuing conversation based on tool results and system instructions...")
    raw_response2, response2 = client.invoke()  # No message provided!
    print(f"Step 2 - Continued response: {response2}")
    
    print(f"Total conversation messages: {len(client.get_conversation_history())}")
    print()


def example_7_system_instructions():
    """Example 7: Using system instructions."""
    print("Example 7: System Instructions")
    print("-" * 40)
    
    client = GroqLLMClient()
    
    # Set system instructions
    client.set_system_instructions("You are a helpful AI assistant that always responds in a very concise manner with exactly one sentence.")
    print(f"System instructions set: {client.system_instructions}")
    
    # Make a request - the system instructions will guide the response
    raw_response, response = client.invoke("Tell me about artificial intelligence.")
    print(f"Response with system instructions: {response}")
    
    # Check that system message was added to conversation
    history = client.get_conversation_history()
    if history and history[0].get("role") == "system":
        print("✅ System message properly added to conversation")
    
    print()


def example_8_complex_workflow():
    """Example 8: Complex workflow with tools and follow-up."""
    print("Example 8: Complex Workflow")
    print("-" * 40)
    
    client = GroqLLMClient()
    
    # Step 1: Use tools
    raw_response1, response1 = client.invoke(
        "Calculate the area of a circle with radius 10 and get weather for New York",
        tools=[calculate_circle_area, get_weather_info]
    )
    print(f"Step 1 response: {response1}")
    print(f"Tool results: {client.tool_results}")
    
    # Step 2: Follow up question using context
    raw_response2, response2 = client.invoke("Which is larger: the circle area or the temperature in celsius?")
    print(f"Step 2 response: {response2}")
    print()


def main():
    """Run all examples."""
    print("🚀 GroqLLMClient Examples")
    print("=" * 50)
    
    examples = [
        example_1_basic_usage,
        example_2_single_tool,
        example_3_multiple_tools,
        example_4_structured_output,
        example_5_conversation_state,
        example_6_continuing_conversation,
        example_7_system_instructions,
        example_8_complex_workflow
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            print()
    
    print("=" * 50)
    print("✅ Examples completed!")


if __name__ == "__main__":
    main()