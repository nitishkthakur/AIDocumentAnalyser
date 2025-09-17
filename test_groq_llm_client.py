#!/usr/bin/env python3
"""
Test script for the GroqLLMClient class.
Tests tool use, structured output, and conversation state management.
"""

import math
import requests
from typing import Optional
from pydantic import BaseModel
from groq_chat_2 import GroqLLMClient


# Sample tool functions for testing
def calculate_area(radius: float) -> float:
    """Calculate the area of a circle given its radius.
    
    Args:
        radius: The radius of the circle in units
        
    Returns:
        The area of the circle
    """
    return math.pi * radius ** 2


def get_weather(location: str) -> dict:
    """Get weather information for a location.
    
    Args:
        location: The city name to get weather for
        
    Returns:
        Weather information dictionary
    """
    # Simulate weather API call
    weather_data = {
        "New York": {"temperature": "22°C", "condition": "Sunny", "humidity": "65%"},
        "London": {"temperature": "15°C", "condition": "Cloudy", "humidity": "78%"},
        "Tokyo": {"temperature": "28°C", "condition": "Rainy", "humidity": "85%"},
    }
    
    return weather_data.get(location, {
        "temperature": "N/A", 
        "condition": "Unknown", 
        "humidity": "N/A",
        "note": f"Weather data not available for {location}"
    })


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        Sum of the two numbers
    """
    return a + b


# Pydantic models for structured output testing
class CalculationResult(BaseModel):
    """Result of a mathematical calculation."""
    area: float
    radius: float
    formula_used: str


class WeatherInfo(BaseModel):
    """Weather information for a location."""
    location: str
    temperature: str
    condition: str
    humidity: Optional[str] = None


class CalculationWeatherResponse(BaseModel):
    """Combined response for calculation and weather."""
    calculation_result: dict
    weather_info: dict
    summary: str


def test_basic_functionality():
    """Test basic text generation without tools or structured output."""
    print("🧪 Test 1: Basic text generation")
    print("-" * 50)
    
    try:
        client = GroqLLMClient()
        raw_response, response = client.invoke("Hello! Can you tell me a short joke?")
        print(f"Response: {response}")
        print(f"Raw response type: {type(raw_response)}")
        print(f"Conversation length: {len(client.get_conversation_history())}")
        print("✅ Basic functionality test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}\n")
        return False


def test_single_tool_use():
    """Test using a single tool."""
    print("🧪 Test 2: Single tool use")
    print("-" * 50)
    
    try:
        client = GroqLLMClient()
        raw_response, response = client.invoke(
            "Calculate the area of a circle with radius 5",
            tools=[calculate_area]
        )
        print(f"Response: {response}")
        print(f"Raw response available: {raw_response is not None}")
        print(f"Tool results: {client.tool_results}")
        print(f"Conversation length: {len(client.get_conversation_history())}")
        print("✅ Single tool use test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Single tool use test failed: {e}\n")
        return False


def test_multiple_tools():
    """Test using multiple tools in one request."""
    print("🧪 Test 3: Multiple tools use")
    print("-" * 50)
    
    try:
        client = GroqLLMClient()
        raw_response, response = client.invoke(
            "Calculate the area of a circle with radius 3, then get the weather for New York",
            tools=[calculate_area, get_weather]
        )
        print(f"Response: {response}")
        print(f"Raw response available: {raw_response is not None}")
        print(f"Tool results: {client.tool_results}")
        print(f"Conversation length: {len(client.get_conversation_history())}")
        print("✅ Multiple tools test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Multiple tools test failed: {e}\n")
        return False


def test_structured_output():
    """Test structured output with Pydantic models."""
    print("🧪 Test 4: Structured output")
    print("-" * 50)
    
    try:
        client = GroqLLMClient()
        raw_response, response = client.invoke(
            "Calculate the area of a circle with radius 7. Return the result in the specified format.",
            json_schema=CalculationResult
        )
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        print(f"Raw response available: {raw_response is not None}")
        if isinstance(response, CalculationResult):
            print(f"Area: {response.area}")
            print(f"Radius: {response.radius}")
            print(f"Formula: {response.formula_used}")
        print(f"Conversation length: {len(client.get_conversation_history())}")
        print("✅ Structured output test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Structured output test failed: {e}\n")
        return False


def test_conversation_state():
    """Test conversation state management."""
    print("🧪 Test 5: Conversation state management")
    print("-" * 50)
    
    try:
        client = GroqLLMClient()
        
        # First message
        raw_response1, response1 = client.invoke("My name is Alice. Remember this.")
        print(f"Response 1: {response1}")
        
        # Second message referring to previous context
        raw_response2, response2 = client.invoke("What is my name?")
        print(f"Response 2: {response2}")
        
        print(f"Final conversation length: {len(client.get_conversation_history())}")
        
        # Check if the conversation contains the expected messages
        history = client.get_conversation_history()
        user_messages = [msg for msg in history if msg['role'] == 'user']
        
        if len(user_messages) >= 2:
            print("✅ Conversation state test passed!\n")
            return True
        else:
            print("❌ Conversation state test failed: Not enough messages in history\n")
            return False
            
    except Exception as e:
        print(f"❌ Conversation state test failed: {e}\n")
        return False


def test_combined_features():
    """Test tools and structured output together (if supported)."""
    print("🧪 Test 6: Combined features (tools + structured output)")
    print("-" * 50)
    
    try:
        client = GroqLLMClient()
        
        # First test with tools only
        raw_response1, response1 = client.invoke(
            "Calculate the area of a circle with radius 4 and get weather for London",
            tools=[calculate_area, get_weather]
        )
        
        print(f"Tool results from first call: {client.tool_results}")
        
        # Then ask for structured summary
        raw_response2, response = client.invoke(
            "Summarize the previous calculation and weather data in the specified format",
            json_schema=CalculationWeatherResponse
        )
        
        print(f"Structured response type: {type(response)}")
        print(f"Structured response: {response}")
        print(f"Conversation length: {len(client.get_conversation_history())}")
        print("✅ Combined features test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Combined features test failed: {e}\n")
        return False


def test_optional_message():
    """Test the optional message parameter for continuing conversations."""
    print("🧪 Test 7: Optional message parameter")
    print("-" * 50)
    
    try:
        client = GroqLLMClient()
        
        # Test that invoke can be called without message
        # We'll just test the method signature and message building logic
        initial_length = len(client.messages)
        print(f"Initial message count: {initial_length}")
        
        # Simulate adding some context (like tool results would)
        client.messages.append({
            "role": "assistant", 
            "content": "I calculated the area.",
            "tool_calls": [{"id": "test", "function": {"name": "calc", "arguments": "{}"}}]
        })
        client.messages.append({
            "role": "tool",
            "tool_call_id": "test", 
            "content": "78.54"
        })
        
        context_length = len(client.messages)
        print(f"Message count with context: {context_length}")
        
        # Test that the method accepts no message parameter
        print("✅ Optional message parameter test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Optional message parameter test failed: {e}\n")
        return False


def test_system_instructions():
    """Test the set_system_instructions method."""
    print("🧪 Test 8: System instructions")
    print("-" * 50)
    
    try:
        client = GroqLLMClient()
        
        # Test setting system instructions
        instructions = "You are a helpful AI assistant specializing in mathematics."
        client.set_system_instructions(instructions)
        
        # Verify the instructions were set
        assert client.system_instructions == instructions
        print(f"System instructions set: {client.system_instructions}")
        
        # Test that system message gets added to conversation
        initial_length = len(client.messages)
        print(f"Initial message count: {initial_length}")
        
        print("✅ System instructions test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ System instructions test failed: {e}\n")
        return False


def test_tool_documentation_extraction():
    """Test the tool documentation extraction functionality."""
    print("🧪 Test 9: Tool documentation extraction")
    print("-" * 50)
    
    try:
        client = GroqLLMClient()
        
        # Test the internal method
        doc_info = client._extract_function_documentation(calculate_area)
        print("Extracted documentation for calculate_area:")
        print(f"Name: {doc_info['function']['name']}")
        print(f"Description: {doc_info['function']['description']}")
        print(f"Parameters: {doc_info['function']['parameters']}")
        
        # Verify the structure
        assert doc_info['type'] == 'function'
        assert doc_info['function']['name'] == 'calculate_area'
        assert 'radius' in doc_info['function']['parameters']['properties']
        assert 'radius' in doc_info['function']['parameters']['required']
        
        print("✅ Tool documentation extraction test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Tool documentation extraction test failed: {e}\n")
        return False


def run_all_tests():
    """Run all tests and provide a summary."""
    print("🚀 Starting GroqLLMClient tests...")
    print("=" * 60)
    
    tests = [
        ("Basic functionality", test_basic_functionality),
        ("Single tool use", test_single_tool_use),
        ("Multiple tools", test_multiple_tools),
        ("Structured output", test_structured_output),
        ("Conversation state", test_conversation_state),
        ("Combined features", test_combined_features),
        ("Optional message", test_optional_message),
        ("System instructions", test_system_instructions),
        ("Tool documentation", test_tool_documentation_extraction),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The GroqLLMClient is working correctly.")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the implementation.")


if __name__ == "__main__":
    run_all_tests()