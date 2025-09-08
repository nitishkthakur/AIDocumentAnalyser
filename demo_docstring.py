#!/usr/bin/env python3
"""
Test script to demonstrate using the docstring extraction function 
with existing functions from the project.
"""

from utils import get_function_docstring, get_function_info
from openai_chat import OpenAIChat

def demo_docstring_extraction():
    """Demonstrate docstring extraction from various functions."""
    
    print("=== Docstring Extraction Demo ===")
    
    # Test with OpenAIChat methods
    chat = OpenAIChat()
    
    print("1. OpenAIChat.__init__ docstring:")
    print(f"   {get_function_docstring(OpenAIChat.__init__)}")
    
    print("\n2. OpenAIChat.invoke docstring:")
    print(f"   {get_function_docstring(OpenAIChat.invoke)}")
    
    print("\n3. OpenAIChat.clear_conversation_history docstring:")
    print(f"   {get_function_docstring(OpenAIChat.clear_conversation_history)}")
    
    # Test with built-in functions
    print("\n4. Built-in len() function:")
    print(f"   {get_function_docstring(len)}")
    
    print("\n5. Built-in print() function:")
    print(f"   {get_function_docstring(print)}")
    
    # Test with lambda (should be empty)
    lambda_func = lambda x: x * 2
    print("\n6. Lambda function:")
    print(f"   '{get_function_docstring(lambda_func)}'")
    
    # Test comprehensive info
    print("\n=== Comprehensive Function Info ===")
    info = get_function_info(OpenAIChat.invoke)
    print(f"Function: {info['name']}")
    print(f"Parameters: {info['parameters']}")
    print(f"Signature: {info['signature']}")
    print(f"File: {info['file']}")
    print(f"Line: {info['line_number']}")
    print(f"Docstring length: {len(info['docstring'])} characters")

if __name__ == "__main__":
    demo_docstring_extraction()
