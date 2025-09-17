import inspect
import json
import requests
import os
from typing import List, Dict, Any, Optional, Callable, Union

try:
    from pydantic import BaseModel
except ImportError:
    # Fallback for when pydantic is not installed
    class BaseModel:
        @classmethod
        def model_json_schema(cls):
            return {"type": "object", "properties": {}}

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback when python-dotenv is not installed
    def load_dotenv():
        pass
    load_dotenv()

load_dotenv()


class GroqLLMClient:
    """
    A simple Groq LLM client that supports tool use, structured output using Pydantic,
    and maintains conversation state automatically.
    
    Features:
    - Tool use with automatic function documentation extraction
    - Structured output using Pydantic BaseModel
    - Automatic conversation state management
    - Tool results storage in tool_results attribute
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "openai/gpt-oss-20b",
                 base_url: str = "https://api.groq.com/openai/v1"):
        """
        Initialize the Groq LLM client.
        
        Args:
            api_key: Groq API key (if not provided, will use GROQ_API_KEY environment variable)
            model: Model name to use
            base_url: Base URL for Groq API
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.base_url = base_url
        self.messages: List[Dict[str, Any]] = []
        self.tool_results: Dict[str, Any] = {}
        self.system_instructions: Optional[str] = None
        
        # Default headers for requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _extract_function_documentation(self, func: Callable) -> Dict[str, Any]:
        """
        Extract function documentation using Python's inspect module.
        
        Args:
            func: Function to extract documentation from
            
        Returns:
            Dictionary containing function name, description, and parameters
        """
        # Get function signature
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        
        # Extract description (part before "Args:")
        description = func.__name__
        if docstring:
            if "Args:" in docstring:
                description = docstring.split("Args:")[0].strip()
            else:
                description = docstring.strip()
        
        # Extract parameter information
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Parse docstring for parameter descriptions
        param_descriptions = {}
        if "Args:" in docstring:
            args_section = docstring.split("Args:")[1]
            if "Returns:" in args_section:
                args_section = args_section.split("Returns:")[0]
            
            for line in args_section.split('\n'):
                line = line.strip()
                if ':' in line:
                    param_name = line.split(':')[0].strip()
                    param_desc = line.split(':', 1)[1].strip()
                    param_descriptions[param_name] = param_desc
        
        # Build parameters schema
        for param_name, param in sig.parameters.items():
            param_type = self._python_type_to_json_type(param.annotation)
            param_info = {"type": param_type}
            
            # Add description if available
            if param_name in param_descriptions:
                param_info["description"] = param_descriptions[param_name]
            
            parameters["properties"][param_name] = param_info
            
            # Add to required if no default value
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters
            }
        }
    
    def _python_type_to_json_type(self, python_type: Any) -> str:
        """
        Convert Python type to JSON Schema type.
        
        Args:
            python_type: Python type annotation
            
        Returns:
            JSON Schema type string
        """
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        
        # Handle basic types
        if python_type in type_mapping:
            return type_mapping[python_type]
        
        # Handle typing module types
        if hasattr(python_type, '__origin__'):
            origin = python_type.__origin__
            if origin is list:
                return "array"
            elif origin is dict:
                return "object"
            elif origin is Union:
                # Handle Optional types (Union[X, None])
                args = python_type.__args__
                if len(args) == 2 and type(None) in args:
                    non_none_type = args[0] if args[1] is type(None) else args[1]
                    return self._python_type_to_json_type(non_none_type)
        
        # Default to string for unknown types
        return "string"
    
    def _build_tools_schema(self, tools: List[Callable]) -> List[Dict[str, Any]]:
        """
        Build tools schema from list of functions.
        
        Args:
            tools: List of callable functions
            
        Returns:
            List of tool schemas for Groq API
        """
        if not tools:
            return []
        
        tool_schemas = []
        for tool in tools:
            if callable(tool):
                tool_schemas.append(self._extract_function_documentation(tool))
        
        return tool_schemas
    
    def _execute_tools(self, tool_calls: List[Dict[str, Any]], available_tools: List[Callable]) -> Dict[str, Any]:
        """
        Execute tool calls and return results.
        
        Args:
            tool_calls: List of tool calls from LLM response
            available_tools: List of available tool functions
            
        Returns:
            Dictionary mapping tool names to their results
        """
        # Create tool mapping
        tool_map = {func.__name__: func for func in available_tools}
        
        results = {}
        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            function_name = function_info.get("name", "")
            
            if function_name in tool_map:
                try:
                    # Parse arguments
                    arguments = function_info.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    
                    # Execute function
                    result = tool_map[function_name](**arguments)
                    results[function_name] = result
                    
                except Exception as e:
                    results[function_name] = {"error": str(e)}
            else:
                results[function_name] = {"error": f"Function '{function_name}' not found"}
        
        return results
    
    def _add_assistant_message(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None):
        """Add assistant message to conversation history."""
        message = {
            "role": "assistant",
            "content": content or ""
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        self.messages.append(message)
    
    def _add_tool_messages(self, tool_calls: List[Dict[str, Any]], tool_results: Dict[str, Any]):
        """Add tool messages to conversation history."""
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name", "")
            tool_call_id = tool_call.get("id", "")
            
            result = tool_results.get(function_name, {"error": "No result"})
            
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result) if not isinstance(result, str) else result
            })
    
    def invoke(self, 
               message: Optional[str] = None, 
               tools: Optional[List[Callable]] = None, 
               json_schema: Optional[BaseModel] = None,
               reasoning: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Main invoke method that sends a message to Groq and handles tools/structured output.
        
        Args:
            message: Optional user message to send. If not provided, continues conversation with existing context
            tools: List of callable functions that can be used as tools
            json_schema: Pydantic BaseModel class for structured output
            
        Returns:
            Tuple of (raw_response, processed_response) where:
            - raw_response: The complete raw API response
            - processed_response: The processed response (string, Pydantic model, or final text after tool execution)
        """
        # Clear previous tool results
        self.tool_results = {}
        
        # Add system instructions if set and not already present
        if self.system_instructions and (not self.messages or self.messages[0].get("role") != "system"):
            self.messages.insert(0, {
                "role": "system",
                "content": self.system_instructions
            })
        
        # Add user message to conversation (only if provided)
        if message is not None:
            self.messages.append({
                "role": "user",
                "content": message
            })
        
        # Build request payload
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0.7,
            "max_completion_tokens": 2000,
            "stream": False,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            'reasoning_effort': reasoning
        }
        
        # Add tools if provided
        if tools:
            tool_schemas = self._build_tools_schema(tools)
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"
        
        # Add structured output if provided
        if json_schema:
            schema = json_schema.model_json_schema()
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.__name__.lower() + "_response",
                    "schema": schema
                }
            }
        
        # Make API call
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
        
        response_data = response.json()
        message_data = response_data["choices"][0]["message"]
        
        # Handle tool calls
        if "tool_calls" in message_data and message_data["tool_calls"]:
            tool_calls = message_data["tool_calls"]
            
            # Add assistant message with tool calls
            self._add_assistant_message(message_data.get("content"), tool_calls)
            
            # Execute tools
            if tools:
                self.tool_results = self._execute_tools(tool_calls, tools)
                
                # Add tool messages to conversation
                self._add_tool_messages(tool_calls, self.tool_results)
                
                # Make another API call to get the final response
                payload["messages"] = self.messages
                # Remove tools from the second call to get final response
                if "tools" in payload:
                    del payload["tools"]
                if "tool_choice" in payload:
                    del payload["tool_choice"]
                
                final_response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                if final_response.status_code == 200:
                    final_data = final_response.json()
                    final_content = final_data["choices"][0]["message"]["content"]
                    
                    # Add final assistant message
                    self._add_assistant_message(final_content)
                    
                    return response_data, final_content
                else:
                    return response_data, "Tool execution completed, but final response failed."
        
        # Handle structured output
        elif json_schema and message_data.get("content"):
            try:
                # Parse JSON response into Pydantic model
                content = message_data["content"]
                parsed_data = json.loads(content)
                result = json_schema(**parsed_data)
                
                # Add assistant message
                self._add_assistant_message(content)
                
                return response_data, result
                
            except (json.JSONDecodeError, ValueError) as e:
                # Add assistant message even if parsing fails
                self._add_assistant_message(message_data.get("content", ""))
                raise Exception(f"Failed to parse structured output: {e}")
        
        # Handle regular text response
        else:
            content = message_data.get("content", "")
            
            # Add assistant message
            self._add_assistant_message(content)
            
            return response_data, content
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.messages = []
        self.tool_results = {}
    
    def set_system_instructions(self, instructions: str):
        """Set system instructions for the conversation.
        
        Args:
            instructions: System instructions to guide the AI's behavior
        """
        self.system_instructions = instructions
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self.messages.copy()
