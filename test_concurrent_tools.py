#!/usr/bin/env python3
"""
Test script for concurrent tool execution in ollama_chat.py
This tests the new multiple tool call functionality.
"""

import json
import time
from ollama_chat import OllamaChat

def test_tool_get_current_time():
    """Tool that returns current time."""
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "timestamp": int(time.time())
    }

def test_tool_calculate_math(expression: str):
    """Tool that evaluates a simple math expression."""
    try:
        # Only allow safe operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression"}
        
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "expression": expression}

def test_tool_generate_sequence(start: int, count: int):
    """Tool that generates a sequence of numbers."""
    return {
        "start": start,
        "count": count,
        "sequence": list(range(start, start + count))
    }

def main():
    # Initialize the client
    client = OllamaChat()
    
    # Configure for testing
    client.configure_concurrent_execution(enabled=True, max_workers=3)
    
    # Tool functions that can be called
    tool_functions = [
        test_tool_get_current_time,
        test_tool_calculate_math, 
        test_tool_generate_sequence
    ]
    
    # Test query that should trigger multiple tool calls
    query = "Please do the following three things: 1) Get the current time, 2) Calculate 15 * 4, and 3) Generate a sequence of 5 numbers starting from 10. Use the available tools to accomplish this."
    
    print("Testing concurrent tool execution...")
    print("=" * 50)
    
    try:
        start_time = time.time()
        
        # Call with tools
        response = client.invoke(
            query=query,
            tools=tool_functions
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Execution completed in {execution_time:.2f} seconds")
        print()
        
        # Check if we got multiple results (list format)
        if isinstance(response, list):
            print(f"✓ Received {len(response)} tool call results (concurrent execution)")
            print()
            
            for i, result in enumerate(response, 1):
                print(f"Tool Call {i}:")
                print(f"  Function: {result.get('function_name', 'Unknown')}")
                print(f"  Success: {result.get('success', False)}")
                if result.get('success'):
                    print(f"  Result: {json.dumps(result.get('result', {}), indent=2)}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
                print()
                
        else:
            print("✗ Received single result format (may not have triggered multiple tools)")
            print(f"Response: {json.dumps(response, indent=2)}")
        
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()