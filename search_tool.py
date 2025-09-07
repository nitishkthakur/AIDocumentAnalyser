import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

def search_web(search_query: str) -> str:
    """
    Search the web using Tavily API and return formatted search results.
    
    Args:
        search_query (str): The search query to execute
        
    Returns:
        str: Formatted search results as a string with the format:
             <search result 1> --- </ search result 1> ... till search results n
    """
    # Get Tavily API key from environment
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if not tavily_api_key:
        return "Error: TAVILY_API_KEY not found in environment variables"
    
    # Tavily API endpoint
    url = "https://api.tavily.com/search"
    
    # Request payload with specified parameters
    payload = {
        "api_key": tavily_api_key,
        "query": search_query,
        "search_depth": "advanced",  # search mode is advanced
        "include_answer": True,      # return answer is True
        "max_results": 5,           # number of results to be returned = 7
        "include_raw_content": False,
        "include_images": False,
        "chunks_per_source": 3
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Make the API request
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        
        # Format the results
        formatted_results = []
        
        # Add the answer if available
        if "answer" in data and data["answer"]:
            formatted_results.append(f"<search result 1>{data['answer']}</ search result 1>")
            result_counter = 2
        else:
            result_counter = 1
        
        # Add individual search results
        if "results" in data:
            for result in data["results"]:
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "No URL")
                
                result_text = f"Title: {title}\nURL: {url}\nContent: {content}"
                formatted_results.append(f"<search result {result_counter}>{result_text}</ search result {result_counter}>")
                result_counter += 1
        
        # Join all results
        if formatted_results:
            return "\n\n".join(formatted_results)
        else:
            return "No search results found"
            
    except requests.exceptions.RequestException as e:
        return f"Error making request to Tavily API: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error parsing response from Tavily API: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


if __name__ == "__main__":
    # Test the function
    test_query = "latest developments in artificial intelligence"
    result = search_web(test_query)
    print(result)
    print(f"tokens: {len(result)/4}")
