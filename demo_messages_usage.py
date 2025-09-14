#!/usr/bin/env python3
"""
Practical demonstration of the new messages parameter functionality
"""

from ollama_chat import OllamaChat

def demo_conversation_context():
    """Demonstrate feeding conversation context to the LLM"""
    print("🎯 Demo: Feeding Conversation Context")
    print("=" * 40)
    
    client = OllamaChat()
    
    # Scenario: Loading a conversation from a database/file to continue where you left off
    previous_conversation = [
        {"role": "system", "content": "You are a helpful programming assistant."},
        {"role": "user", "content": "I'm building a web scraper in Python."},
        {"role": "assistant", "content": "That's great! Web scraping is very useful. Are you using any specific libraries like requests and BeautifulSoup?"},
        {"role": "user", "content": "Yes, I'm using requests but I'm getting blocked by some websites."},
        {"role": "assistant", "content": "That's common. Websites often block scrapers. You can try adding headers to make your requests look more like a real browser."},
        {"role": "user", "content": "How do I add headers? And what headers should I use?"}
    ]
    
    print("📋 Simulating conversation loaded from database...")
    print("Previous context:")
    for msg in previous_conversation[-3:]:
        print(f"  {msg['role']}: {msg['content'][:60]}...")
    
    print("\n🤖 Continuing conversation with context...")
    response = client.invoke(messages=previous_conversation)
    print(f"AI Response: {response[:300]}...")

def demo_role_based_context():
    """Demonstrate different message roles"""
    print("\n\n🎭 Demo: Different Message Roles")
    print("=" * 40)
    
    client = OllamaChat()
    
    # Scenario: Providing rich context with different roles
    context_messages = [
        {"role": "system", "content": "You are an expert data scientist reviewing code."},
        {"role": "user", "content": "I wrote this ML model but it's overfitting."},
        {"role": "assistant", "content": "Overfitting is a common issue. What techniques have you tried so far?"},
        {"role": "user", "content": "I added dropout and regularization."},
        {"role": "tool", "content": "Model performance: Training accuracy: 98%, Validation accuracy: 72%"},
        {"role": "user", "content": "What else can I try to reduce this gap?"}
    ]
    
    print("📊 Context includes tool results and system instructions...")
    response = client.invoke(messages=context_messages)
    print(f"Expert advice: {response[:300]}...")

def demo_mixed_approaches():
    """Demonstrate mixing query and messages approaches"""
    print("\n\n🔄 Demo: Mixing Query and Messages Approaches")
    print("=" * 40)
    
    client = OllamaChat()
    
    # Start with traditional query
    print("1️⃣ Starting with traditional query approach...")
    response1 = client.invoke("I need help with a Python project.")
    print(f"Response: {response1[:100]}...")
    
    # Continue with messages (injecting context)
    print("\n2️⃣ Injecting additional context via messages...")
    context = [
        {"role": "user", "content": "The project is a REST API using Flask."},
        {"role": "user", "content": "I'm having trouble with database connections."}
    ]
    response2 = client.invoke(messages=context)
    print(f"Response: {response2[:100]}...")
    
    # Back to query approach
    print("\n3️⃣ Continuing with query approach...")
    response3 = client.invoke("Should I use SQLAlchemy or raw SQL?")
    print(f"Response: {response3[:100]}...")
    
    print(f"\n📈 Total conversation history: {len(client.conversation_history)} messages")

def demo_practical_use_cases():
    """Show practical use cases for the messages feature"""
    print("\n\n💡 Practical Use Cases")
    print("=" * 40)
    
    print("✅ Use Case 1: Customer Support Bot")
    print("   - Load chat history from CRM")
    print("   - Continue conversation seamlessly")
    
    print("\n✅ Use Case 2: Code Review Assistant")
    print("   - Provide git history as context")
    print("   - Include previous review comments")
    
    print("\n✅ Use Case 3: Educational Tutor")
    print("   - Load student's learning progress")
    print("   - Personalize responses based on history")
    
    print("\n✅ Use Case 4: Multi-Agent Systems")
    print("   - Pass conversations between agents")
    print("   - Maintain context across agent handoffs")
    
    print("\n✅ Use Case 5: Conversation Analysis")
    print("   - Analyze existing chat logs")
    print("   - Generate insights from conversation patterns")

def main():
    print("🚀 OllamaChat Messages Parameter Demonstration")
    print("=" * 50)
    
    try:
        demo_conversation_context()
        demo_role_based_context()
        demo_mixed_approaches()
        demo_practical_use_cases()
        
        print("\n\n🎉 All demos completed successfully!")
        print("\n📚 Key Features Demonstrated:")
        print("  • Loading conversation context from external sources")
        print("  • Supporting all message roles (user/system/assistant/tool)")
        print("  • Maintaining conversation history automatically")
        print("  • Mixing query and messages approaches")
        print("  • Backwards compatibility with existing code")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()