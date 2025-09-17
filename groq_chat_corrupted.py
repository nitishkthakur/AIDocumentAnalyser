import inspect
import json
import typing as t
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

class GroqChat:
    """Groq chat client with tool calling, structured outputs, and concurrent execution.
    
    Features:
    - API key authentication for Groq Cloud
    - Tool calling and function execution with OpenAI-compatible format
    - Structured JSON outputs with schema validation
    - Conversation history management
    - Concurrent tool execution for multiple tool calls
    - Support for both json_object and json_schema response formats
    
    Basic usage:
        chat = GroqChat()
        response = chat.invoke("Hello")  # Returns string
        response = chat.invoke("Extract data", json_schema=schema)  # Returns parsed JSON
        response = chat.invoke("Get weather", tools=[weather_func])  # Returns {"tool_name": str, "tool_return": any, "text": str, "raw": dict, "tool_messages": list}
        
    Environment variable configuration:
        export GROQ_API_KEY="your_groq_api_key"
        chat = GroqChat()   # Will use API key from environment
    """
    
    TYPE_MAPPING = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}

    def __init__(self, api_key: str = None, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", 
                 reasoning: dict = None, system_instructions: str = "", base_url: str = "https://api.groq.com/openai/v1"):
        """Initialize the Groq chat client."""
        # Groq requires API key authentication
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        # Use meta-llama/llama-4-scout-17b-16e-instruct as default
        self.model = model_name or "meta-llama/llama-4-scout-17b-16e-instruct"
        self.base_url = base_url
        self.conversation_history: list[dict] = []
        self.system_instructions = system_instructions
        
        # Configuration for concurrent tool execution
        self.concurrent_tool_execution = True  # Enable concurrent execution for multiple tools
        self.max_concurrent_tools = 5  # Maximum number of concurrent tool executions

        # Reasoning defaults (adapted for Groq) - Configure for efficient usage
        self.default_reasoning = reasoning or {
            "temperature": 0.8,
            "max_completion_tokens": 1024,  # Groq uses max_completion_tokens instead of max_tokens
            "top_p": 0.9,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }
        self._validate_reasoning(self.default_reasoning)

    def _validate_reasoning(self, reasoning: dict) -> dict:
        """Validate reasoning parameters for Groq API compatibility."""
        valid_params = {
            'temperature', 'max_completion_tokens', 'top_p', 'presence_penalty', 
            'frequency_penalty', 'seed', 'stop', 'stream'
        }
        
        validated = {}
        for key, value in reasoning.items():
            if key in valid_params:
                # Validate parameter ranges
                if key == 'temperature' and not (0 <= value <= 2):
                    raise ValueError(f"Temperature must be between 0 and 2, got {value}")
                elif key in ['presence_penalty', 'frequency_penalty'] and not (-2 <= value <= 2):
                    raise ValueError(f"{key} must be between -2 and 2, got {value}")
                elif key == 'top_p' and not (0 <= value <= 1):
                    raise ValueError(f"top_p must be between 0 and 1, got {value}")
                
                validated[key] = value
            else:
                print(f"Warning: Parameter '{key}' is not supported by Groq API and will be ignored")
        
        return validated

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def extend_conversation_with_tool_messages(self, tool_response: dict) -> None:
        """Extend conversation history with tool messages from a tool call response.
        
        Args:
            tool_response: The tool response dictionary containing 'tool_messages' key
        """
        if isinstance(tool_response, dict) and 'tool_messages' in tool_response:
            self.conversation_history.extend(tool_response['tool_messages'])
        elif isinstance(tool_response, list):
            # Handle multiple tool calls
            for response in tool_response:
                if isinstance(response, dict) and 'tool_messages' in response:
                    self.conversation_history.extend(response['tool_messages'])

    def set_output_parameters(self, max_tokens: int = 1024, temperature: float = 0.8, 
                            top_p: float = 0.9, presence_penalty: float = 0.0, frequency_penalty: float = 0.0) -> None:
        """Set output generation parameters for Groq API.
        
        Args:
            max_tokens: Maximum tokens to generate (Groq uses max_completion_tokens)
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
        """
        self.default_reasoning.update({
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        })
        self._validate_reasoning(self.default_reasoning)

    def optimize_for_long_output(self) -> None:
        """Configure for longer outputs by setting higher token limits."""
        self.default_reasoning.update({
            "max_completion_tokens": 4096,  # Higher limit for longer outputs
            "temperature": 0.7,  # Slightly lower temperature for more focused output
        })

    def configure_concurrent_execution(self, enabled: bool = True, max_workers: int = 5) -> None:
        """Configure concurrent tool execution settings.
        
        Args:
            enabled (bool): Whether to enable concurrent execution for multiple tool calls
            max_workers (int): Maximum number of concurrent tool executions (default 5)
        """
        self.concurrent_tool_execution = enabled
        self.max_concurrent_tools = max_workers

    def _execute_tool_calls(self, tool_calls: list, available_tools: t.Optional[t.Iterable[t.Callable]]) -> dict:
        """Execute tool calls sequentially."""
        if not available_tools:
            return {}

        # Create a mapping from function names to actual functions
        tool_map = {func.__name__: func for func in available_tools}
        
        results = {}
        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            function_name = function_info.get("name", "")
            
            if function_name in tool_map:
                try:
                    # Parse arguments from JSON string
                    arguments = function_info.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    
                    # Call the function
                    result = tool_map[function_name](**arguments)
                    results[function_name] = result
                except Exception as e:
                    results[function_name] = {"error": str(e)}
        
        return results

    def _execute_single_tool_call(self, tool_call: dict, tool_map: dict) -> tuple[str, t.Any]:
        """Execute a single tool call and return (function_name, result)."""
        function_info = tool_call.get("function", {})
        function_name = function_info.get("name", "")
        
        if function_name not in tool_map:
            return function_name, {"error": f"Function '{function_name}' not found"}
        
        try:
            # Parse arguments from JSON string
            arguments = function_info.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            
            # Call the function
            result = tool_map[function_name](**arguments)
            return function_name, result
        except Exception as e:
            return function_name, {"error": str(e)}

    def _execute_tool_calls_concurrent(self, tool_calls: list, available_tools: t.Optional[t.Iterable[t.Callable]], max_workers: int = 5) -> dict:
        """Execute tool calls concurrently using ThreadPoolExecutor."""
        if not available_tools:
            return {}

        # Create a mapping from function names to actual functions
        tool_map = {func.__name__: func for func in available_tools}
        
        results = {}
        
        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tool calls
            future_to_tool = {
                executor.submit(self._execute_single_tool_call, tool_call, tool_map): tool_call 
                for tool_call in tool_calls
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_tool):
                function_name, result = future.result()
                results[function_name] = result
        
        return results

    def _get_json_type(self, python_type: t.Any) -> str:
        """Convert Python type to JSON Schema type."""
        if hasattr(python_type, '__origin__'):
            origin = python_type.__origin__
            if origin is list:
                return "array"
            elif origin is dict:
                return "object"
            elif origin is t.Union:
                args = python_type.__args__
                if len(args) == 2 and type(None) in args:
                    non_none_type = args[0] if args[1] is type(None) else args[1]
                    return self._get_json_type(non_none_type)
        
        return self.TYPE_MAPPING.get(python_type, "string")

    def _extract_function_info(self, func: t.Callable) -> dict:
        """Extract function information for Groq tool definition."""
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or f"Function {func.__name__}"
        
        parameters = {"type": "object", "properties": {}, "required": []}
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": self._get_json_type(param.annotation)}
            
            # Add description from docstring if available
            if docstring:
                lines = docstring.split('\n')
                for line in lines:
                    if param_name in line and ':' in line:
                        description = line.split(':', 1)[1].strip()
                        if description:
                            param_info["description"] = description
                        break
            
            parameters["properties"][param_name] = param_info
            
            # Required if no default value
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": docstring,
                "parameters": parameters
            }
        }

    def _build_tools(self, tools: t.Optional[t.Iterable[t.Callable]]) -> list[dict]:
        """Build Groq-compatible tools list from callable functions."""
        if not tools:
            return []
        
        tool_definitions = []
        for tool in tools:
            if callable(tool):
                tool_definitions.append(self._extract_function_info(tool))
        
        return tool_definitions

    def _extract_json_schema(self, schema_input: t.Any) -> dict | None:
        """Extract JSON schema from various input types."""
        if schema_input is None:
            return None
        
        if isinstance(schema_input, dict):
            return schema_input
        
        # Check if it's a Pydantic model
        if hasattr(schema_input, 'model_json_schema'):
            return schema_input.model_json_schema()
        
        # Check if it's a dataclass
        if hasattr(schema_input, '__dataclass_fields__'):
            # Simple dataclass to schema conversion
            properties = {}
            required = []
            for field_name, field in schema_input.__dataclass_fields__.items():
                properties[field_name] = {"type": self._get_json_type(field.type)}
                if field.default is inspect.Parameter.empty:
                    required.append(field_name)
            return {"type": "object", "properties": properties, "required": required}
        
        return None

    def _make_schema_strict_compatible(self, schema: dict) -> dict:
        """Make schema compatible with Groq's structured output requirements."""
        # Groq supports JSON Schema, ensure proper format
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        
        return schema

    def _build_input_messages(self, query: str) -> list[dict]:
        """Build input messages for Groq chat API from a single query."""
        messages = []
        
        # Add system instructions if provided
        if self.system_instructions:
            messages.append({
                "role": "system", 
                "content": self.system_instructions
            })
        
        # Add conversation history and current query
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def _build_input_messages_from_list(self, message_list: t.List[dict]) -> list[dict]:
        """Build input messages for Groq chat API from a message list.
        
        Args:
            message_list: List of messages with 'role' and 'content' keys
            
        Returns:
            List of messages ready for Groq API
        """
        messages = []
        
        # Add system instructions if provided and not already in message list
        has_system = any(msg.get('role') == 'system' for msg in message_list)
        if self.system_instructions and not has_system:
            messages.append({
                "role": "system", 
                "content": self.system_instructions
            })
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Validate and add provided messages
        valid_roles = {'user', 'assistant', 'system', 'tool'}
        for msg in message_list:
            if not isinstance(msg, dict):
                raise ValueError(f"Each message must be a dictionary, got {type(msg)}")
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content' keys")
            if msg['role'] not in valid_roles:
                raise ValueError(f"Message role must be one of {valid_roles}, got '{msg['role']}'")
            
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        return messages

    def _build_groq_payload(self, query: str, reasoning: dict = None, tools: t.Optional[t.Iterable[t.Callable]] = None) -> dict:
        """Build payload for testing purposes."""
        messages = self._build_input_messages(query)
        
        # Groq options based on reasoning
        reasoning_config = reasoning or self.default_reasoning
        
        # Build payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **reasoning_config
        }
        
        # Add tools if provided
        tool_schemas = self._build_tools(tools)
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"
        
        return payload

    def invoke(self, query: t.Optional[str] = None, json_schema: t.Optional[dict | t.Any] = None, 
               tools: t.Optional[t.Iterable[t.Callable]] = None, reasoning: t.Optional[dict] = None, 
               system_instructions: t.Optional[str] = None, messages: t.Optional[t.List[dict]] = None):
        """Send query or message list to Groq and return response.
        
        Args:
            query: Single query string (alternative to messages)
            messages: List of conversation messages with roles (user/system/assistant/tool)
            json_schema: Optional JSON schema for structured output
            tools: Optional tools for function calling
            reasoning: Optional reasoning parameters
            system_instructions: Optional system instructions
        
        Note: Either query OR messages must be provided, not both.
        
        Returns:
            - Tool calls: {"tool_name": str, "tool_return": any, "text": str, "raw": dict, "tool_messages": list} or list of such dicts
            - JSON schema: parsed JSON object
            - Otherwise: string response
        """
        # Validate input parameters
        if query is None and messages is None:
            raise ValueError("Either 'query' or 'messages' parameter must be provided")
        if query is not None and messages is not None:
            raise ValueError("Cannot provide both 'query' and 'messages' parameters. Use one or the other.")
        
        # Update system instructions if provided
        if system_instructions is not None:
            self.system_instructions = system_instructions
        
        # Use provided parameters or defaults
        effective_reasoning = reasoning or self.default_reasoning
        
        # Validate if provided
        if reasoning is not None:
            effective_reasoning = self._validate_reasoning(reasoning)
        
        # Use Groq chat API for all queries
        result = self._invoke_groq_api(query, json_schema, tools, effective_reasoning, messages)
        
        # Return tool result in simplified format
        if result.get('tool_calls') and result.get('tool_results'):
            tool_calls = result['tool_calls']
            tool_results = result['tool_results']
            if tool_calls and tool_results:
                # Handle multiple tool calls - return list of dictionaries
                if len(tool_calls) > 1:
                    tool_call_results = []
                    for tool_call in tool_calls:
                        function_name = tool_call.get("function", {}).get("name", "")
                        if function_name in tool_results:
                            # Create the assistant message with tool calls (ensuring content field exists)
                            assistant_message = result['raw']['choices'][0]['message'].copy()
                            if 'content' not in assistant_message:
                                assistant_message['content'] = ""
                            
                            # Create the tool response message
                            tool_message = {
                                "role": "tool",
                                "content": str(tool_results[function_name]),
                                "tool_call_id": tool_call.get("id")
                            }
                            
                            tool_call_results.append({
                                "tool_name": function_name,
                                "tool_return": tool_results[function_name],
                                "text": result['text'],
                                "raw": result['raw'],
                                "tool_messages": [assistant_message, tool_message]
                            })
                    return tool_call_results if tool_call_results else result['text']
                
                # Handle single tool call - return single dictionary (backward compatibility)
                else:
                    first_tool_call = tool_calls[0]
                    function_name = first_tool_call.get("function", {}).get("name", "")
                    if function_name in tool_results:
                        # Create the assistant message with tool calls (ensuring content field exists)
                        assistant_message = result['raw']['choices'][0]['message'].copy()
                        if 'content' not in assistant_message:
                            assistant_message['content'] = ""
                        
                        # Create the tool response message
                        tool_message = {
                            "role": "tool",
                            "content": str(tool_results[function_name]),
                            "tool_call_id": first_tool_call.get("id")
                        }
                        
                        return {
                            "tool_name": function_name, 
                            "tool_return": tool_results[function_name],
                            "text": result['text'],
                            "raw": result['raw'],
                            "tool_messages": [assistant_message, tool_message]
                        }
        
        # Parse JSON if schema was used
        if json_schema:
            try:
                return json.loads(result['text'])
            except json.JSONDecodeError:
                return result['text']
        
        return result['text']

    def _invoke_groq_api(self, query: t.Optional[str] = None, json_schema: t.Optional[dict | t.Any] = None, 
                        tools: t.Optional[t.Iterable[t.Callable]] = None, reasoning: dict = None, 
                        messages: t.Optional[t.List[dict]] = None) -> dict:
        """Handle all queries using Groq chat API."""
        url = f"{self.base_url}/chat/completions"
        
        # Build messages based on input type
        if messages is not None:
            # Use provided message list, appending to conversation history
            input_messages = self._build_input_messages_from_list(messages)
        else:
            # Use traditional query-based approach
            input_messages = self._build_input_messages(query)
        
        # Groq options based on reasoning
        reasoning_config = reasoning or self.default_reasoning
        
        # Build payload
        payload = {
            "model": self.model,
            "messages": input_messages,
            "stream": False,
            **reasoning_config
        }
        
        # Add tools if provided
        tool_schemas = self._build_tools(tools)
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True  # Enable parallel tool calls
        
        # Add structured output if provided
        schema = self._extract_json_schema(json_schema)
        if schema:
            # Use json_schema mode for structured outputs
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": self._make_schema_strict_compatible(schema)
                }
            }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Make request to Groq API
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', str(e))
                print(f"Groq API Error: {error_msg}")
                print(f"Request payload: {json.dumps(payload, indent=2)}")
                raise
            except json.JSONDecodeError:
                print(f"HTTP Error: {e}")
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
                raise
        
        data = response.json()
        
        # Extract text and tool calls from Groq response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        assistant_response = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        
        # Execute tool calls
        tool_execution_results = {}
        if tool_calls and tools:
            # Use concurrent execution for multiple tool calls, sequential for single
            if len(tool_calls) > 1 and self.concurrent_tool_execution:
                tool_execution_results = self._execute_tool_calls_concurrent(tool_calls, tools, self.max_concurrent_tools)
            else:
                tool_execution_results = self._execute_tool_calls(tool_calls, tools)
        
        # Update conversation history
        if messages is not None:
            # Add all provided messages to conversation history
            for msg in messages:
                self.conversation_history.append({
                    "role": msg['role'], 
                    "content": msg['content']
                })
        else:
            # Traditional query approach
            self.conversation_history.append({"role": "user", "content": query})
        
        # Always add the assistant response
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return {
            "text": assistant_response,
            "tool_calls": tool_calls,
            "tool_results": tool_execution_results,
            "raw": data
        }


if __name__ == "__main__":
    """Groq chat client using the Responses API with tool calling, structured outputs, and concurrent execution.
    
    This class is identical to GroqChat but uses the Groq Responses API (/v1/responses) instead 
    of the Chat Completions API (/v1/chat/completions). The Responses API provides stateful 
    conversations and enhanced conversation management features.
    
    Features:
    - API key authentication for Groq Cloud
    - Tool calling and function execution with OpenAI-compatible format
    - Structured JSON outputs with schema validation
    - Stateful conversation management with the Responses API
    - Concurrent tool execution for multiple tool calls
    - Support for both json_object and json_schema response formats
    - Response ID tracking for conversation continuity
    
    Basic usage:
        chat = GroqChatResp()
        response = chat.invoke("Hello")  # Returns string
        response = chat.invoke("Extract data", json_schema=schema)  # Returns parsed JSON
        response = chat.invoke("Get weather", tools=[weather_func])  # Returns {"tool_name": str, "tool_return": any, "text": str, "raw": dict, "tool_messages": list}
        
    Environment variable configuration:
        export GROQ_API_KEY="your_groq_api_key"
        chat = GroqChatResp()   # Will use API key from environment
    """
    
    TYPE_MAPPING = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}

    def __init__(self, api_key: str = None, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", 
                 reasoning: dict = None, system_instructions: str = "", base_url: str = "https://api.groq.com/openai/v1"):
        """Initialize the Groq chat client using Responses API."""
        # Groq requires API key authentication
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        # Use meta-llama/llama-4-scout-17b-16e-instruct as default
        self.model = model_name or "meta-llama/llama-4-scout-17b-16e-instruct"
        self.base_url = base_url
        self.conversation_history: list[dict] = []
        self.system_instructions = system_instructions
        
        # Responses API specific tracking
        self.current_response_id: str = None  # Track current response ID for conversation continuity
        self.store_conversations: bool = False  # Disable conversation storage by default (only supports false/null)
        
        # Configuration for concurrent tool execution
        self.concurrent_tool_execution = True  # Enable concurrent execution for multiple tools
        self.max_concurrent_tools = 5  # Maximum number of concurrent tool executions

        # Reasoning defaults (adapted for Groq Responses API) - Configure for efficient usage
        self.default_reasoning = reasoning or {
            "temperature": 0.8,
            "max_output_tokens": 1024,  # Responses API uses max_output_tokens
            "top_p": 0.9
        }
        self._validate_reasoning_resp(self.default_reasoning)

    def _validate_reasoning_resp(self, reasoning: dict) -> dict:
        """Validate reasoning parameters for Groq Responses API compatibility."""
        valid_params = {
            'temperature', 'max_output_tokens', 'top_p'
        }
        
        validated = {}
        for key, value in reasoning.items():
            if key in valid_params:
                # Validate parameter ranges
                if key == 'temperature' and not (0 <= value <= 2):
                    raise ValueError(f"Temperature must be between 0 and 2, got {value}")
                elif key == 'top_p' and not (0 <= value <= 1):
                    raise ValueError(f"top_p must be between 0 and 1, got {value}")
                
                validated[key] = value
            else:
                print(f"Warning: Parameter '{key}' is not supported by Groq Responses API and will be ignored")
        
        return validated

    def _validate_reasoning(self, reasoning: dict) -> dict:
        """Validate reasoning parameters for Groq API compatibility."""
        valid_params = {
            'temperature', 'max_completion_tokens', 'top_p', 'presence_penalty', 
            'frequency_penalty', 'seed', 'stop', 'stream'
        }
        
        validated = {}
        for key, value in reasoning.items():
            if key in valid_params:
                # Validate parameter ranges
                if key == 'temperature' and not (0 <= value <= 2):
                    raise ValueError(f"Temperature must be between 0 and 2, got {value}")
                elif key in ['presence_penalty', 'frequency_penalty'] and not (-2 <= value <= 2):
                    raise ValueError(f"{key} must be between -2 and 2, got {value}")
                elif key == 'top_p' and not (0 <= value <= 1):
                    raise ValueError(f"top_p must be between 0 and 1, got {value}")
                
                validated[key] = value
            else:
                print(f"Warning: Parameter '{key}' is not supported by Groq API and will be ignored")
        
        return validated

    def clear_conversation_history(self) -> None:
        """Clear the conversation history and reset response ID."""
        self.conversation_history = []
        self.current_response_id = None
    
    def extend_conversation_with_tool_messages(self, tool_response: dict) -> None:
        """Extend conversation history with tool messages from a tool call response.
        
        Args:
            tool_response: The tool response dictionary containing 'tool_messages' key
        """
        if isinstance(tool_response, dict) and 'tool_messages' in tool_response:
            self.conversation_history.extend(tool_response['tool_messages'])
        elif isinstance(tool_response, list):
            # Handle multiple tool calls
            for response in tool_response:
                if isinstance(response, dict) and 'tool_messages' in response:
                    self.conversation_history.extend(response['tool_messages'])

    def set_output_parameters(self, max_tokens: int = 1024, temperature: float = 0.8, 
                            top_p: float = 0.9, presence_penalty: float = 0.0, frequency_penalty: float = 0.0) -> None:
        """Set output generation parameters for Groq API.
        
        Args:
            max_tokens: Maximum tokens to generate (Groq uses max_completion_tokens)
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
        """
        self.default_reasoning.update({
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        })
        self._validate_reasoning(self.default_reasoning)

    def optimize_for_long_output(self) -> None:
        """Configure for longer outputs by setting higher token limits."""
        self.default_reasoning.update({
            "max_completion_tokens": 4096,  # Higher limit for longer outputs
            "temperature": 0.7,  # Slightly lower temperature for more focused output
        })

    def configure_concurrent_execution(self, enabled: bool = True, max_workers: int = 5) -> None:
        """Configure concurrent tool execution settings.
        
        Args:
            enabled (bool): Whether to enable concurrent execution for multiple tool calls
            max_workers (int): Maximum number of concurrent tool executions (default 5)
        """
        self.concurrent_tool_execution = enabled
        self.max_concurrent_tools = max_workers

    def configure_conversation_storage(self, store: bool = True) -> None:
        """Configure whether to store conversations on the server for continuity.
        
        Args:
            store (bool): Whether to enable server-side conversation storage
        """
        self.store_conversations = store

    def _execute_tool_calls(self, tool_calls: list, available_tools: t.Optional[t.Iterable[t.Callable]]) -> dict:
        """Execute tool calls sequentially."""
        if not available_tools:
            return {}

        # Create a mapping from function names to actual functions
        tool_map = {func.__name__: func for func in available_tools}
        
        results = {}
        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            function_name = function_info.get("name", "")
            
            if function_name in tool_map:
                try:
                    # Parse arguments from JSON string
                    arguments = function_info.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    
                    # Call the function
                    result = tool_map[function_name](**arguments)
                    results[function_name] = result
                except Exception as e:
                    results[function_name] = {"error": str(e)}
        
        return results

    def _execute_single_tool_call(self, tool_call: dict, tool_map: dict) -> tuple[str, t.Any]:
        """Execute a single tool call and return (function_name, result)."""
        function_info = tool_call.get("function", {})
        function_name = function_info.get("name", "")
        
        if function_name not in tool_map:
            return function_name, {"error": f"Function '{function_name}' not found"}
        
        try:
            # Parse arguments from JSON string
            arguments = function_info.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            
            # Call the function
            result = tool_map[function_name](**arguments)
            return function_name, result
        except Exception as e:
            return function_name, {"error": str(e)}

    def _execute_tool_calls_concurrent(self, tool_calls: list, available_tools: t.Optional[t.Iterable[t.Callable]], max_workers: int = 5) -> dict:
        """Execute tool calls concurrently using ThreadPoolExecutor."""
        if not available_tools:
            return {}

        # Create a mapping from function names to actual functions
        tool_map = {func.__name__: func for func in available_tools}
        
        results = {}
        
        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tool calls
            future_to_tool = {
                executor.submit(self._execute_single_tool_call, tool_call, tool_map): tool_call 
                for tool_call in tool_calls
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_tool):
                function_name, result = future.result()
                results[function_name] = result
        
        return results

    def _get_json_type(self, python_type: t.Any) -> str:
        """Convert Python type to JSON Schema type."""
        if hasattr(python_type, '__origin__'):
            origin = python_type.__origin__
            if origin is list:
                return "array"
            elif origin is dict:
                return "object"
            elif origin is t.Union:
                args = python_type.__args__
                if len(args) == 2 and type(None) in args:
                    non_none_type = args[0] if args[1] is type(None) else args[1]
                    return self._get_json_type(non_none_type)
        
        return self.TYPE_MAPPING.get(python_type, "string")

    def _extract_function_info(self, func: t.Callable) -> dict:
        """Extract function information for Groq tool definition."""
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or f"Function {func.__name__}"
        
        parameters = {"type": "object", "properties": {}, "required": []}
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": self._get_json_type(param.annotation)}
            
            # Add description from docstring if available
            if docstring:
                lines = docstring.split('\n')
                for line in lines:
                    if param_name in line and ':' in line:
                        description = line.split(':', 1)[1].strip()
                        if description:
                            param_info["description"] = description
                        break
            
            parameters["properties"][param_name] = param_info
            
            # Required if no default value
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        # Responses API format - name at top level
        return {
            "type": "function",
            "name": func.__name__,
            "description": docstring,
            "parameters": parameters
        }

    def _build_tools(self, tools: t.Optional[t.Iterable[t.Callable]]) -> list[dict]:
        """Build Groq-compatible tools list from callable functions."""
        if not tools:
            return []
        
        tool_definitions = []
        for tool in tools:
            if callable(tool):
                tool_definitions.append(self._extract_function_info(tool))
        
        return tool_definitions

    def _extract_json_schema(self, schema_input: t.Any) -> dict | None:
        """Extract JSON schema from various input types."""
        if schema_input is None:
            return None
        
        if isinstance(schema_input, dict):
            return schema_input
        
        # Check if it's a Pydantic model
        if hasattr(schema_input, 'model_json_schema'):
            return schema_input.model_json_schema()
        
        # Check if it's a dataclass
        if hasattr(schema_input, '__dataclass_fields__'):
            # Simple dataclass to schema conversion
            properties = {}
            required = []
            for field_name, field in schema_input.__dataclass_fields__.items():
                properties[field_name] = {"type": self._get_json_type(field.type)}
                if field.default is inspect.Parameter.empty:
                    required.append(field_name)
            return {"type": "object", "properties": properties, "required": required}
        
        return None

    def _make_schema_strict_compatible(self, schema: dict) -> dict:
        """Make schema compatible with Groq's structured output requirements."""
        # Groq supports JSON Schema, ensure proper format
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        
        return schema

    def _build_input_text(self, query: str) -> str:
        """Build input text for Groq Responses API from a single query."""
        # For Responses API, we need to construct a single input string
        # that includes conversation history and current query
        
        input_parts = []
        
        # Add conversation history as context
        for msg in self.conversation_history:
            if msg.get('role') == 'user':
                input_parts.append(f"User: {msg.get('content', '')}")
            elif msg.get('role') == 'assistant':
                input_parts.append(f"Assistant: {msg.get('content', '')}")
        
        # Add current query
        input_parts.append(f"User: {query}")
        
        return "\n".join(input_parts)
    
    def _build_input_text_from_list(self, message_list: t.List[dict]) -> str:
        """Build input text for Groq Responses API from a message list.
        
        Args:
            message_list: List of messages with 'role' and 'content' keys
            
        Returns:
            Input text ready for Groq Responses API
        """
        # Validate provided messages
        valid_roles = {'user', 'assistant', 'system', 'tool'}
        for msg in message_list:
            if not isinstance(msg, dict):
                raise ValueError(f"Each message must be a dictionary, got {type(msg)}")
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content' keys")
            if msg['role'] not in valid_roles:
                raise ValueError(f"Message role must be one of {valid_roles}, got '{msg['role']}'")
        
        input_parts = []
        
        # Add conversation history as context
        for msg in self.conversation_history:
            if msg.get('role') == 'user':
                input_parts.append(f"User: {msg.get('content', '')}")
            elif msg.get('role') == 'assistant':
                input_parts.append(f"Assistant: {msg.get('content', '')}")
        
        # Add provided messages
        for msg in message_list:
            if msg['role'] == 'user':
                input_parts.append(f"User: {msg['content']}")
            elif msg['role'] == 'assistant':
                input_parts.append(f"Assistant: {msg['content']}")
            elif msg['role'] == 'system':
                # For Responses API, system messages become part of instructions
                pass  # We'll handle this in the instructions field
        
        return "\n".join(input_parts)

    def _build_groq_payload(self, query: str, reasoning: dict = None, tools: t.Optional[t.Iterable[t.Callable]] = None) -> dict:
        """Build payload for testing purposes."""
        input_text = self._build_input_text(query)
        
        # Groq options based on reasoning
        reasoning_config = reasoning or self.default_reasoning
        
        # Build payload for Responses API
        payload = {
            "model": self.model,
            "input": input_text,
            "stream": False,
            **reasoning_config
        }
        
        # Add system instructions
        if self.system_instructions:
            payload["instructions"] = self.system_instructions
        
        # Note: store parameter disabled - Responses API only supports false/null
        
        # Add tools if provided
        tool_schemas = self._build_tools(tools)
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"
        
        return payload

    def invoke(self, query: t.Optional[str] = None, json_schema: t.Optional[dict | t.Any] = None, 
               tools: t.Optional[t.Iterable[t.Callable]] = None, reasoning: t.Optional[dict] = None, 
               system_instructions: t.Optional[str] = None, messages: t.Optional[t.List[dict]] = None):
        """Send query or message list to Groq Responses API and return response.
        
        Args:
            query: Single query string (alternative to messages)
            messages: List of conversation messages with roles (user/system/assistant/tool)
            json_schema: Optional JSON schema for structured output
            tools: Optional tools for function calling
            reasoning: Optional reasoning parameters
            system_instructions: Optional system instructions
        
        Note: Either query OR messages must be provided, not both.
        
        Returns:
            - Tool calls: {"tool_name": str, "tool_return": any, "text": str, "raw": dict, "tool_messages": list} or list of such dicts
            - JSON schema: parsed JSON object
            - Otherwise: string response
        """
        # Validate input parameters
        if query is None and messages is None:
            raise ValueError("Either 'query' or 'messages' parameter must be provided")
        if query is not None and messages is not None:
            raise ValueError("Cannot provide both 'query' and 'messages' parameters. Use one or the other.")
        
        # Update system instructions if provided
        if system_instructions is not None:
            self.system_instructions = system_instructions
        
        # Use provided parameters or defaults
        effective_reasoning = reasoning or self.default_reasoning
        
        # Validate if provided
        if reasoning is not None:
            effective_reasoning = self._validate_reasoning(reasoning)
        
        # Use Groq Responses API for all queries
        result = self._invoke_groq_responses_api(query, json_schema, tools, effective_reasoning, messages)
        
        # Return tool result in simplified format
        if result.get('tool_calls') and result.get('tool_results'):
            tool_calls = result['tool_calls']
            tool_results = result['tool_results']
            if tool_calls and tool_results:
                # Handle multiple tool calls - return list of dictionaries
                if len(tool_calls) > 1:
                    tool_call_results = []
                    for tool_call in tool_calls:
                        function_name = tool_call.get("function", {}).get("name", "")
                        if function_name in tool_results:
                            # Create the assistant message with tool calls (ensuring content field exists)
                            assistant_message = result['raw']['choices'][0]['message'].copy()
                            if 'content' not in assistant_message:
                                assistant_message['content'] = ""
                            
                            # Create the tool response message
                            tool_message = {
                                "role": "tool",
                                "content": str(tool_results[function_name]),
                                "tool_call_id": tool_call.get("id")
                            }
                            
                            tool_call_results.append({
                                "tool_name": function_name,
                                "tool_return": tool_results[function_name],
                                "text": result['text'],
                                "raw": result['raw'],
                                "tool_messages": [assistant_message, tool_message]
                            })
                    return tool_call_results if tool_call_results else result['text']
                
                # Handle single tool call - return single dictionary (backward compatibility)
                else:
                    first_tool_call = tool_calls[0]
                    function_name = first_tool_call.get("function", {}).get("name", "")
                    if function_name in tool_results:
                        # Create the assistant message with tool calls (ensuring content field exists)
                        assistant_message = result['raw']['choices'][0]['message'].copy()
                        if 'content' not in assistant_message:
                            assistant_message['content'] = ""
                        
                        # Create the tool response message
                        tool_message = {
                            "role": "tool",
                            "content": str(tool_results[function_name]),
                            "tool_call_id": first_tool_call.get("id")
                        }
                        
                        return {
                            "tool_name": function_name, 
                            "tool_return": tool_results[function_name],
                            "text": result['text'],
                            "raw": result['raw'],
                            "tool_messages": [assistant_message, tool_message]
                        }
        
        # Parse JSON if schema was used
        if json_schema:
            try:
                return json.loads(result['text'])
            except json.JSONDecodeError:
                return result['text']
        
        return result['text']

    def _invoke_groq_responses_api(self, query: t.Optional[str] = None, json_schema: t.Optional[dict | t.Any] = None, 
                                  tools: t.Optional[t.Iterable[t.Callable]] = None, reasoning: dict = None, 
                                  messages: t.Optional[t.List[dict]] = None) -> dict:
        """Handle all queries using Groq Responses API."""
        url = f"{self.base_url}/responses"  # Use responses endpoint instead of chat/completions
        
        # Build input text based on input type
        if messages is not None:
            # Use provided message list, appending to conversation history
            input_text = self._build_input_text_from_list(messages)
        else:
            # Use traditional query-based approach
            input_text = self._build_input_text(query)
        
        # Groq options based on reasoning
        reasoning_config = reasoning or self.default_reasoning
        
        # Build payload for Responses API
        payload = {
            "model": self.model,
            "input": input_text,
            "stream": False,
            **reasoning_config
        }
        
        # Add system instructions
        if self.system_instructions:
            payload["instructions"] = self.system_instructions
        
        # Note: store parameter disabled - Responses API only supports false/null
        
        # Add tools if provided
        tool_schemas = self._build_tools(tools)
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True  # Enable parallel tool calls
        
        # Add structured output if provided (adapted for Responses API)
        schema = self._extract_json_schema(json_schema)
        if schema:
            # For Responses API, structured output is handled differently
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "schema": self._make_schema_strict_compatible(schema)
                    }
                }
            }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Make request to Groq Responses API
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', str(e))
                print(f"Groq Responses API Error: {error_msg}")
                print(f"Request payload: {json.dumps(payload, indent=2)}")
                raise
            except json.JSONDecodeError:
                print(f"HTTP Error: {e}")
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
                raise
        
        data = response.json()
        
        # Update response ID for conversation continuity
        if 'id' in data:
            self.current_response_id = data['id']
        
        # Extract text and tool calls from Groq Responses API response
        # Responses API has a different structure: output array with various item types
        assistant_response = ""
        tool_calls = []
        
        output_items = data.get("output", [])
        for item in output_items:
            if item.get("type") == "message" and item.get("role") == "assistant":
                # Extract text content from message items
                content = item.get("content", [])
                for content_item in content:
                    if content_item.get("type") == "output_text":
                        assistant_response += content_item.get("text", "")
                
                # Extract tool calls if present in message
                if "tool_calls" in item:
                    tool_calls.extend(item.get("tool_calls", []))
            
            elif item.get("type") == "function_call":
                # Handle function calls as separate items in Responses API
                function_call = {
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}")
                    }
                }
                tool_calls.append(function_call)
        
        # Execute tool calls
        tool_execution_results = {}
        if tool_calls and tools:
            # Use concurrent execution for multiple tool calls, sequential for single
            if len(tool_calls) > 1 and self.concurrent_tool_execution:
                tool_execution_results = self._execute_tool_calls_concurrent(tool_calls, tools, self.max_concurrent_tools)
            else:
                tool_execution_results = self._execute_tool_calls(tool_calls, tools)
        
        # Update conversation history
        if messages is not None:
            # Add all provided messages to conversation history
            for msg in messages:
                self.conversation_history.append({
                    "role": msg['role'], 
                    "content": msg['content']
                })
        else:
            # Traditional query approach
            self.conversation_history.append({"role": "user", "content": query})
        
        # Always add the assistant response
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return {
            "text": assistant_response,
            "tool_calls": tool_calls,
            "tool_results": tool_execution_results,
            "raw": data
        }


if __name__ == "__main__":
    """Example usage with Groq API."""
    
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"{city}: 24°C, sunny"

    # Initialize with API key from environment
    try:
        chat = GroqChat()
        
        print("=== Groq Chat Configuration ===")
        print(f"Model: {chat.model}")
        print(f"Default parameters: {chat.default_reasoning}")
        
        # Basic chat
        print("\n=== Basic Chat ===")
        result = chat.invoke("Hello! Please respond in exactly 10 words.")
        print(f"Q: Hello! Please respond in exactly 10 words.")
        print(f"A: {result}")
        
        # Tool calling
        print("\n=== Tool Calling ===")
        result = chat.invoke("What's the weather in Paris?", tools=[get_weather])
        print(f"Q: What's the weather in Paris?")
        if isinstance(result, dict) and 'tool_name' in result:
            print(f"Tool: {result['tool_name']}")
            print(f"Result: {result['tool_return']}")
        else:
            print(f"Response: {result}")
            
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Make sure to set your GROQ_API_KEY environment variable")
    except Exception as e:
        print(f"Error: {e}")