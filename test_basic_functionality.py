#!/usr/bin/env python3
"""
Simple test to verify both classes can be imported and basic functionality works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that both classes can be imported."""
    print("Testing imports...")
    
    try:
        from groq_chat import GroqChat
        print("✓ GroqChat imported successfully")
    except Exception as e:
        print(f"✗ Failed to import GroqChat: {e}")
        return False
    
    try:
        from groq_chat_resp import GroqChatResp
        print("✓ GroqChatResp imported successfully")
    except Exception as e:
        print(f"✗ Failed to import GroqChatResp: {e}")
        return False
    
    return True

def test_initialization():
    """Test that both classes can be initialized."""
    print("\nTesting initialization...")
    
    from groq_chat import GroqChat
    from groq_chat_resp import GroqChatResp
    
    try:
        chat = GroqChat()
        print("✓ GroqChat initialized successfully")
        print(f"  Model: {chat.model}")
        print(f"  History length: {len(chat.conversation_history)}")
    except Exception as e:
        print(f"✗ Failed to initialize GroqChat: {e}")
        return False
    
    try:
        chat_resp = GroqChatResp()
        print("✓ GroqChatResp initialized successfully")
        print(f"  Model: {chat_resp.model}")
        print(f"  History length: {len(chat_resp.conversation_history)}")
        print(f"  Response ID: {chat_resp.current_response_id}")
    except Exception as e:
        print(f"✗ Failed to initialize GroqChatResp: {e}")
        return False
    
    return True

def test_tool_schema_generation():
    """Test that tool schemas are generated correctly."""
    print("\nTesting tool schema generation...")
    
    from groq_chat import GroqChat
    from groq_chat_resp import GroqChatResp
    
    def sample_tool(city: str, units: str = "metric") -> str:
        """Get weather for a city."""
        return f"Weather in {city} ({units})"
    
    try:
        chat = GroqChat()
        tools = chat._build_tools([sample_tool])
        print("✓ GroqChat tool schema generated")
        print(f"  Tool format: {tools[0]['type']}")
        print(f"  Function name: {tools[0]['function']['name']}")
        
        chat_resp = GroqChatResp()
        tools_resp = chat_resp._build_tools([sample_tool])
        print("✓ GroqChatResp tool schema generated")
        print(f"  Tool format: {tools_resp[0]['type']}")
        print(f"  Function name: {tools_resp[0]['name']}")  # Note: different format
        
        return True
    except Exception as e:
        print(f"✗ Tool schema generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run basic tests."""
    print("Running Basic Functionality Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_initialization,
        test_tool_schema_generation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("BASIC TEST RESULTS")
    print("=" * 40)
    
    test_names = ["Import Test", "Initialization Test", "Tool Schema Test"]
    for name, result in zip(test_names, results):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20} {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'✓ ALL BASIC TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)