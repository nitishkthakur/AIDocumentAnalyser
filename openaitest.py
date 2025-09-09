#!/usr/bin/env python3
"""
Simple OpenAI API test for function calling.
This script demonstrates how to make a POST request to OpenAI's API
with function calling capabilities and prints the response.
"""

import json
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Test OpenAI function calling with a simple POST request."""
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Define a simple function for the AI to call
    def get_weather(city: str, units: str = "celsius") -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is 24°C, sunny with light clouds."
    
    # Function schema for OpenAI
    function_schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["city"],
                "additionalProperties": False
            }
        }
    }
    
    # API endpoint and headers
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Request payload
    payload = {
        "model": "gpt-5-nano",
        "messages": [
            {
                "role": "user", 
                "content": "What's the weather like in Paris?"
            }
        ],
        "tools": [function_schema],
        "tool_choice": "auto",
        "reasoning": {"effort": "low"},
        "verbosity": "medium"
    }
    
    print("🚀 Sending request to OpenAI API...")
    print(f"📡 Endpoint: {url}")
    print(f"🤖 Model: {payload['model']}")
    print(f"💬 Query: {payload['messages'][0]['content']}")
    print(f"🔧 Tools: {len(payload['tools'])} function(s) available")
    print("-" * 50)
    
    try:
        # Make the POST request
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return
        
        # Parse response
        data = response.json()
        print("✅ Response received successfully!")
        print(f"📊 Response data keys: {list(data.keys())}")
        
        # Extract the message
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        
        print("\n📝 AI Response:")
        if content:
            print(f"Content: {content}")
        
        # Handle tool calls
        if tool_calls:
            print(f"\n🔧 Tool calls detected: {len(tool_calls)}")
            for i, tool_call in enumerate(tool_calls):
                function_name = tool_call.get("function", {}).get("name", "")
                function_args = tool_call.get("function", {}).get("arguments", "{}")
                
                print(f"\nTool Call {i+1}:")
                print(f"  Function: {function_name}")
                print(f"  Arguments: {function_args}")
                
                # Execute the function call (simulate)
                if function_name == "get_weather":
                    try:
                        args = json.loads(function_args)
                        result = get_weather(**args)
                        print(f"  ✅ Result: {result}")
                    except Exception as e:
                        print(f"  ❌ Execution error: {e}")
                else:
                    print(f"  ⚠️  Unknown function: {function_name}")
        else:
            print("\n🔧 No tool calls in response")
        
        # Print full response for debugging
        print("\n" + "="*50)
        print("📋 Full Response JSON:")
        print(json.dumps(data, indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print(f"Raw response: {response.text}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
