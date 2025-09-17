from gem_react import GemReact, calculator, get_current_time, search

def main():
    agent = GemReact(model_name="mixtral-8x7b-32768")
    tools = [calculator, get_current_time, search]

    print("Starting multi-turn conversation with Groq ReAct Agent.")
    print("Try asking things like 'what is the time?', 'what is 2+2', or 'search for the capital of France'.")
    print("Type 'exit' or 'quit' to end the conversation.")

    # Turn 1
    query1 = "what is the current time?"
    print(f"\nUser: {query1}")
    response1 = agent.invoke(query1, tools=tools)
    print(f"Agent: {response1}")

    # Turn 2
    query2 = "Thanks. Now, what is 123 * 456?"
    print(f"\nUser: {query2}")
    response2 = agent.invoke(query2, tools=tools)
    print(f"Agent: {response2}")

    # Turn 3
    query3 = "search for the latest news on AI"
    print(f"\nUser: {query3}")
    response3 = agent.invoke(query3, tools=tools)
    print(f"Agent: {response3}")
    
    # Turn 4
    query4 = "what is 9 to the power of 3"
    print(f"\nUser: {query4}")
    response4 = agent.invoke(query4, tools=tools)
    print(f"Agent: {response4}")

if __name__ == "__main__":
    main()
