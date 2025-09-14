#!/usr/bin/env python3
"""
Test concurrent tool execution with better prompting
"""

import json
import time
from ollama_chat import OllamaChat

def get_time():
    """Get current time"""
    return {"current_time": time.strftime("%Y-%m-%d %H:%M:%S")}

def add_numbers(a: int, b: int):
    """Add two numbers"""
    return {"sum": a + b, "operands": [a, b]}

def get_greeting(name: str):
    """Get a greeting for someone"""
    return {"greeting": f"Hello, {name}!"}

def main():
    client = OllamaChat()
    
    tools = [get_time, add_numbers, get_greeting]
    
    # More explicit request for multiple tools
    query = """I need you to perform these three specific tasks using the available tools:
    1. Call get_time to get the current time
    2. Call add_numbers with arguments a=10 and b=25
    3. Call get_greeting with argument name="Alice"
    
    Please use all three tools to complete these tasks."""
    
    print("Testing concurrent tool execution with explicit instructions...")
    print("=" * 60)
    
    start_time = time.time()
    response = client.invoke(query=query, tools=tools)
    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print()
    
    if isinstance(response, list):
        print(f"✓ Concurrent execution: {len(response)} results")
        for i, result in enumerate(response, 1):
            print(f"\nResult {i}:")
            print(json.dumps(result, indent=2, default=str))
    else:
        print("Single result format:")
        print(json.dumps(response, indent=2, default=str))

if __name__ == "__main__":
    main()