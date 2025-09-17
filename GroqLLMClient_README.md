# GroqLLMClient

A simple and elegant Python client for Groq LLM API that supports tool use, structured output using Pydantic, and automatic conversation state management.

## Features

- **Tool Use**: Automatically extract function documentation and execute tools based on LLM responses
- **Structured Output**: Support for Pydantic BaseModel schemas for structured JSON responses
- **Conversation State**: Automatic conversation history management
- **Tool Results**: Automatic storage of tool execution results in the `tool_results` attribute
- **Simple API**: Clean and intuitive `invoke()` method interface

## Quick Start

### Basic Usage

```python
from groq_chat_2 import GroqLLMClient

# Initialize client (requires GROQ_API_KEY environment variable)
client = GroqLLMClient()

# Simple text generation
raw_response, response = client.invoke("Tell me a joke about programming")
print(response)  # Processed response
print(raw_response)  # Complete raw API response
```

### Tool Use

```python
import math

def calculate_area(radius: float) -> float:
    """Calculate the area of a circle given its radius.
    
    Args:
        radius: The radius of the circle
        
    Returns:
        The area of the circle
    """
    return math.pi * radius ** 2

# Use tools
client = GroqLLMClient()
raw_response, response = client.invoke(
    "Calculate the area of a circle with radius 5",
    tools=[calculate_area]
)

print(response)  # LLM's response after using the tool
print(client.tool_results)  # {'calculate_area': 78.53981633974483}
print(raw_response)  # Complete raw API response
```

### Structured Output

```python
from pydantic import BaseModel

class WeatherInfo(BaseModel):
    location: str
    temperature: str
    condition: str

client = GroqLLMClient()
raw_response, result = client.invoke(
    "Give me weather info for New York",
    json_schema=WeatherInfo
)

print(type(result))  # <class '__main__.WeatherInfo'>
print(result.location)  # Access structured data
print(raw_response)  # Complete raw API response
```

## API Reference

### GroqLLMClient

#### Constructor

```python
GroqLLMClient(
    api_key: Optional[str] = None,
    model: str = "llama-3.1-70b-versatile",
    base_url: str = "https://api.groq.com/openai/v1"
)
```

- `api_key`: Groq API key (uses `GROQ_API_KEY` env var if not provided)
- `model`: Model name to use
- `base_url`: Base URL for Groq API

#### Main Method

```python
invoke(
    message: str,
    tools: Optional[List[Callable]] = None,
    json_schema: Optional[BaseModel] = None
) -> tuple
```

- `message`: User message to send to the LLM
- `tools`: List of callable functions that can be used as tools
- `json_schema`: Pydantic BaseModel class for structured output

**Returns:**
- Tuple of `(raw_response, processed_response)` where:
  - `raw_response`: The complete raw API response from Groq
  - `processed_response`: The processed response (string, Pydantic model instance, or final text after tool execution)

#### Attributes

- `tool_results`: Dictionary containing the results of tool executions
- `messages`: List of conversation messages

#### Utility Methods

- `clear_conversation()`: Clear conversation history
- `get_conversation_history()`: Get current conversation messages

## Tool Documentation Format

Tools are automatically documented using Python's `inspect` module. The class extracts:

1. **Function Description**: Text before "Args:" in the docstring
2. **Parameter Information**: Parsed from the "Args:" section
3. **Type Information**: From function parameter annotations

Example tool function:

```python
def get_weather(location: str, units: str = "celsius") -> dict:
    """Get weather information for a specific location.
    
    Args:
        location: The city name to get weather for
        units: Temperature units (celsius or fahrenheit)
        
    Returns:
        Weather data dictionary
    """
    # Implementation here
    pass
```

This automatically generates the proper tool schema for the Groq API.

## Environment Setup

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

Or set the environment variable:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

## Dependencies

- `requests`: For HTTP API calls
- `pydantic`: For structured output (optional, has fallback)
- `python-dotenv`: For environment variable loading (optional, has fallback)

Install with:

```bash
pip install requests pydantic python-dotenv
```

## Examples

See `example_groq_llm_client.py` for comprehensive usage examples and `test_groq_llm_client.py` for test cases.

## Request Payload Format

The class generates requests in the format specified in your requirements:

```json
{
  "model": "llama-3.1-70b-versatile",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "function_name",
        "description": "Function description from docstring",
        "parameters": {
          "type": "object",
          "properties": {...},
          "required": [...]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "response_schema",
      "schema": {...}
    }
  },
  "temperature": 0.7,
  "max_completion_tokens": 2000,
  "stream": false,
  "top_p": 0.9,
  "frequency_penalty": 0,
  "presence_penalty": 0
}
```

## Error Handling

The class handles various error scenarios:

- Missing API key
- API request failures
- Tool execution errors
- JSON parsing errors for structured output
- Invalid function calls

All errors are propagated with descriptive messages.