#!/usr/bin/env python3
"""
Simple debug test for ollama_chat.py tool execution
"""

import json
import time
from ollama_chat import OllamaChat

def simple_test_tool():
    """Simple test tool that returns current time."""
    return {"message": "Tool executed successfully", "time": time.time()}

def main():
    client = OllamaChat()
    
    # Simple test with one tool
    print("Testing single tool execution...")
    
    response = client.invoke(
        query="Use the simple test tool to get some information.",
        tools=[simple_test_tool]
    )
    
    print("Response type:", type(response))
    print("Response:", json.dumps(response, indent=2, default=str))

if __name__ == "__main__":
    main()