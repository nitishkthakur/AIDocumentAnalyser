#!/usr/bin/env python3
"""
Test concurrent tool execution for GroqChat.
This specifically tests multiple tool calls being executed in parallel.
"""

import time
import json
from dotenv import load_dotenv
from groq_chat import GroqChat

# Load environment variables
load_dotenv()

def fetch_user_data(user_id: int) -> dict:
    """Simulate fetching user data from database."""
    time.sleep(2)  # Simulate network/database delay
    return {
        "user_id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
        "status": "active"
    }

def calculate_metrics(data_type: str) -> dict:
    """Simulate calculating complex metrics."""
    time.sleep(1.5)  # Simulate processing time
    metrics = {
        "sales": {"total": 150000, "growth": 15.2},
        "users": {"total": 2500, "active": 1890},
        "revenue": {"total": 75000, "monthly": 6250}
    }
    return metrics.get(data_type, {"error": "Unknown data type"})

def generate_report(report_type: str) -> str:
    """Simulate generating a report."""
    time.sleep(1)  # Simulate processing
    reports = {
        "summary": "Q4 performance shows strong growth across all metrics",
        "detailed": "Comprehensive analysis reveals 20% increase in user engagement",
        "forecast": "Projected 25% growth for next quarter based on current trends"
    }
    return reports.get(report_type, "Report type not found")

def send_notification(recipient: str, message: str) -> str:
    """Simulate sending a notification."""
    time.sleep(0.5)  # Simulate network delay
    return f"Notification sent to {recipient}: {message}"

def test_sequential_execution():
    """Test sequential tool execution for comparison."""
    print("=== Testing Sequential Execution ===")
    
    try:
        chat = GroqChat()
        chat.configure_concurrent_execution(enabled=False)  # Disable concurrent execution
        
        start_time = time.time()
        
        # Make a request that should trigger multiple tool calls
        result = chat.invoke(
            "I need you to fetch user data for user ID 123, calculate sales metrics, and generate a summary report. Please do all three tasks.",
            tools=[fetch_user_data, calculate_metrics, generate_report]
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Sequential execution time: {execution_time:.2f} seconds")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, list):
            print(f"Multiple tool calls: {len(result)}")
            for i, call in enumerate(result, 1):
                if 'tool_name' in call:
                    print(f"  {i}. {call['tool_name']}: {str(call['tool_return'])[:100]}...")
        elif isinstance(result, dict) and 'tool_name' in result:
            print(f"Single tool call: {result['tool_name']}")
        else:
            print(f"Response: {str(result)[:200]}...")
            
        return execution_time
        
    except Exception as e:
        print(f"✗ Sequential execution test failed: {e}")
        return None

def test_concurrent_execution():
    """Test concurrent tool execution."""
    print("\n=== Testing Concurrent Execution ===")
    
    try:
        chat = GroqChat()
        chat.configure_concurrent_execution(enabled=True, max_workers=5)  # Enable concurrent execution
        
        start_time = time.time()
        
        # Make the same request with concurrent execution enabled
        result = chat.invoke(
            "I need you to fetch user data for user ID 456, calculate users metrics, and generate a detailed report. Please do all three tasks.",
            tools=[fetch_user_data, calculate_metrics, generate_report]
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Concurrent execution time: {execution_time:.2f} seconds")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, list):
            print(f"Multiple tool calls: {len(result)}")
            for i, call in enumerate(result, 1):
                if 'tool_name' in call:
                    print(f"  {i}. {call['tool_name']}: {str(call['tool_return'])[:100]}...")
        elif isinstance(result, dict) and 'tool_name' in result:
            print(f"Single tool call: {result['tool_name']}")
        else:
            print(f"Response: {str(result)[:200]}...")
            
        return execution_time
        
    except Exception as e:
        print(f"✗ Concurrent execution test failed: {e}")
        return None

def test_concurrent_with_many_tools():
    """Test concurrent execution with many available tools."""
    print("\n=== Testing Concurrent Execution with Many Tools ===")
    
    def task_a(param: str) -> str:
        time.sleep(1)
        return f"Task A completed with {param}"
    
    def task_b(param: str) -> str:
        time.sleep(1.2)
        return f"Task B completed with {param}"
    
    def task_c(param: str) -> str:
        time.sleep(0.8)
        return f"Task C completed with {param}"
    
    def task_d(param: str) -> str:
        time.sleep(1.5)
        return f"Task D completed with {param}"
    
    try:
        chat = GroqChat()
        chat.configure_concurrent_execution(enabled=True, max_workers=4)
        
        start_time = time.time()
        
        # Request that might trigger multiple tool calls
        result = chat.invoke(
            "Please execute task A with 'data1', task B with 'data2', and task C with 'data3'. Do all tasks.",
            tools=[task_a, task_b, task_c, task_d, send_notification]
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Multi-tool execution time: {execution_time:.2f} seconds")
        
        if isinstance(result, list):
            print(f"✓ Multiple tool calls executed concurrently: {len(result)}")
            for call in result:
                if 'tool_name' in call:
                    print(f"  - {call['tool_name']}: {call['tool_return']}")
        elif isinstance(result, dict) and 'tool_name' in result:
            print(f"✓ Single tool executed: {result['tool_name']} -> {result['tool_return']}")
        else:
            print(f"Response: {result}")
            
    except Exception as e:
        print(f"✗ Multi-tool concurrent test failed: {e}")

def test_error_handling_concurrent():
    """Test error handling in concurrent execution."""
    print("\n=== Testing Error Handling in Concurrent Execution ===")
    
    def working_function(data: str) -> str:
        time.sleep(1)
        return f"Successfully processed: {data}"
    
    def failing_function(data: str) -> str:
        time.sleep(0.5)
        raise ValueError(f"Intentional failure with {data}")
    
    def slow_function(data: str) -> str:
        time.sleep(2)
        return f"Slow processing completed: {data}"
    
    try:
        chat = GroqChat()
        chat.configure_concurrent_execution(enabled=True, max_workers=3)
        
        start_time = time.time()
        
        result = chat.invoke(
            "Please call the working function with 'test1', the failing function with 'test2', and the slow function with 'test3'.",
            tools=[working_function, failing_function, slow_function]
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Error handling execution time: {execution_time:.2f} seconds")
        
        if isinstance(result, list):
            print(f"✓ Concurrent execution with errors handled: {len(result)} results")
            for call in result:
                if 'tool_name' in call:
                    tool_result = call['tool_return']
                    if isinstance(tool_result, dict) and 'error' in tool_result:
                        print(f"  - {call['tool_name']}: ERROR - {tool_result['error']}")
                    else:
                        print(f"  - {call['tool_name']}: SUCCESS - {tool_result}")
        else:
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")

def test_performance_comparison():
    """Compare sequential vs concurrent execution performance."""
    print("\n=== Performance Comparison ===")
    
    seq_time = test_sequential_execution()
    conc_time = test_concurrent_execution()
    
    if seq_time and conc_time:
        improvement = ((seq_time - conc_time) / seq_time) * 100
        print(f"\nPerformance Analysis:")
        print(f"  Sequential: {seq_time:.2f}s")
        print(f"  Concurrent: {conc_time:.2f}s")
        print(f"  Improvement: {improvement:.1f}% faster")
        
        if improvement > 0:
            print("✓ Concurrent execution is faster!")
        else:
            print("⚠ Concurrent execution was not faster (possibly due to single tool call)")

def main():
    """Run all concurrent execution tests."""
    print("GroqChat Concurrent Tool Execution Test Suite")
    print("=" * 60)
    
    # Run performance comparison
    test_performance_comparison()
    
    # Run additional tests
    test_concurrent_with_many_tools()
    test_error_handling_concurrent()
    
    print("\n" + "=" * 60)
    print("Concurrent Tool Execution Test Suite Complete")
    print("\nNote: Performance improvements depend on whether the model")
    print("decides to make multiple tool calls simultaneously.")

if __name__ == "__main__":
    main()