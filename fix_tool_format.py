#!/usr/bin/env python3
"""
Quick fix to update the tool format in GroqChatResp class
"""

def fix_tool_format():
    # Read the file
    with open('groq_chat.py', 'r') as f:
        lines = f.readlines()
    
    # Find the GroqChatResp class and the _extract_function_info method
    in_groq_chat_resp = False
    in_extract_function_info = False
    groq_chat_resp_start = None
    
    for i, line in enumerate(lines):
        if line.strip().startswith('class GroqChatResp:'):
            in_groq_chat_resp = True
            groq_chat_resp_start = i
            print(f"Found GroqChatResp class at line {i+1}")
        elif in_groq_chat_resp and line.strip().startswith('class ') and not line.strip().startswith('class GroqChatResp:'):
            in_groq_chat_resp = False  # We've moved to another class
        elif in_groq_chat_resp and 'def _extract_function_info(self, func: t.Callable) -> dict:' in line:
            in_extract_function_info = True
            print(f"Found _extract_function_info in GroqChatResp at line {i+1}")
        elif in_extract_function_info and 'return {' in line:
            # Found the return statement in the GroqChatResp _extract_function_info method
            print(f"Found return statement at line {i+1}")
            # Look for the function format return
            if i+1 < len(lines) and '"type": "function",' in lines[i+1]:
                print(f"Found function format return, updating...")
                # Replace the return statement
                j = i
                while j < len(lines) and '}' not in lines[j]:
                    j += 1
                if j < len(lines):
                    # Replace the return block
                    new_return = '''        # Responses API format - name at top level
        return {
            "type": "function",
            "name": func.__name__,
            "description": docstring,
            "parameters": parameters
        }
'''
                    lines[i:j+1] = [new_return]
                    print("Updated the return statement")
                    break
            in_extract_function_info = False
    
    # Write the file back
    with open('groq_chat.py', 'w') as f:
        f.writelines(lines)
    
    print("Tool format fixed!")

if __name__ == "__main__":
    fix_tool_format()