#!/usr/bin/env python3
"""
Fix assistant message creation for Responses API format
"""

def fix_assistant_message():
    # Read the file
    with open('groq_chat.py', 'r') as f:
        lines = f.readlines()
    
    # Find and replace the problematic assistant_message creation
    for i, line in enumerate(lines):
        if "assistant_message = result['raw']['choices'][0]['message'].copy()" in line:
            print(f"Found assistant_message line at {i+1}")
            # Replace this line and the next few lines
            if i+1 < len(lines) and 'if \'content\' not in assistant_message:' in lines[i+1]:
                if i+2 < len(lines) and 'assistant_message[\'content\'] = ""' in lines[i+2]:
                    # Replace these 3 lines with the correct format
                    lines[i] = '                            # Create the assistant message with tool calls (Responses API format)\n'
                    lines[i+1] = '                            assistant_message = {\n'
                    lines[i+2] = '                                "role": "assistant",\n'
                    lines.insert(i+3, '                                "content": result.get(\'text\', \'\'),\n')
                    lines.insert(i+4, '                                "tool_calls": [tool_call] if \'tool_call\' in locals() else [first_tool_call]\n')
                    lines.insert(i+5, '                            }\n')
                    print(f"Replaced lines {i+1}-{i+3}")
                    break
    
    # Write the file back
    with open('groq_chat.py', 'w') as f:
        f.writelines(lines)
    
    print("Assistant message creation fixed!")

if __name__ == "__main__":
    fix_assistant_message()