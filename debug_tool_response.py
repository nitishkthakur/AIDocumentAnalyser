#!/usr/bin/env python3
"""
Debug tool calling response format
"""

from groq_chat import GroqChatResp
from calculator_tool import calculator
import json

def debug_tool_response():
    """Debug the raw response format from tool calls"""
    print("=== Debugging Tool Call Response ===")
    
    try:
        chat_resp = GroqChatResp()
        chat_resp.default_reasoning = {
            "temperature": 0.8,
            "max_output_tokens": 1024,
            "top_p": 0.9
        }
        
        # Get the raw result from _invoke_groq_responses_api
        result = chat_resp._invoke_groq_responses_api("What is 15 + 27?", tools=[calculator])
        
        print("Raw result keys:", result.keys())
        print("Tool calls:", result.get('tool_calls'))
        print("Tool results:", result.get('tool_results'))
        print("Text:", result.get('text'))
        print("Raw data structure:")
        print(json.dumps(result.get('raw'), indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tool_response()