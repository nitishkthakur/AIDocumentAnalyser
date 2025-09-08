#!/usr/bin/env python3
"""
Example usage of the parse_string_or_dict function.
"""

from utils import parse_string_or_dict

def demo_parse_function():
    """Demonstrate the parse_string_or_dict function with various examples."""
    
    print("=== parse_string_or_dict Function Demo ===\n")
    
    # Example 1: Dictionary input (returns same dictionary)
    print("1. Dictionary input:")
    original_dict = {'user': 'john', 'role': 'admin', 'permissions': ['read', 'write']}
    result = parse_string_or_dict(original_dict)
    print(f"   Input:  {original_dict}")
    print(f"   Output: {result}")
    print(f"   Same object: {result is original_dict}")
    
    # Example 2: JSON string
    print("\n2. JSON string:")
    json_str = '{"name": "ChatGPT", "model": "gpt-5-nano", "tokens": 1000}'
    result = parse_string_or_dict(json_str)
    print(f"   Input:  {json_str}")
    print(f"   Output: {result}")
    
    # Example 3: Python dictionary string
    print("\n3. Python dictionary string:")
    py_dict_str = "{'database': 'PostgreSQL', 'host': 'localhost', 'port': 5432}"
    result = parse_string_or_dict(py_dict_str)
    print(f"   Input:  {py_dict_str}")
    print(f"   Output: {result}")
    
    # Example 4: Dictionary embedded in text
    print("\n4. Dictionary embedded in text:")
    text_with_dict = "Server response: {'success': True, 'message': 'Data saved', 'id': 12345}"
    result = parse_string_or_dict(text_with_dict)
    print(f"   Input:  {text_with_dict}")
    print(f"   Output: {result}")
    
    # Example 5: Key-value pairs
    print("\n5. Key-value pairs:")
    kv_string = "status: active, count: 42, enabled: true"
    result = parse_string_or_dict(kv_string)
    print(f"   Input:  {kv_string}")
    print(f"   Output: {result}")
    
    print("\n=== Function Features ===")
    print("✅ Returns exact same dictionary if input is already a dict")
    print("✅ Parses JSON strings")
    print("✅ Parses Python dictionary strings")
    print("✅ Extracts dictionaries from longer text")
    print("✅ Handles key-value pairs")
    print("✅ Automatically converts data types (int, float, bool)")
    print("✅ Comprehensive error handling")

if __name__ == "__main__":
    demo_parse_function()
