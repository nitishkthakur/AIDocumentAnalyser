import inspect
import json
import typing as t
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class OllamaChat:
    """Simplified Ollama chat client with tool calling and structured outputs.
    
    Basic usage:
        chat = OllamaChat()
        response = chat.invoke("Hello")  # Returns string
        response = chat.invoke("Extract data", json_schema=schema)  # Returns parsed JSON
        response = chat.invoke("Get weather", tools=[weather_func])  # Returns {"tool_name": str, "tool_return": any}
    """
    
    TYPE_MAPPING = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}

    def __init__(self, api_key: str = None, model_name: str = "qwen3:4b", 
                 reasoning: dict = None, verbosity: str = "medium",
                 system_instructions: str = "", base_url: str = "http://localhost:11434"):
        """Initialize the Ollama chat client."""
        # Ollama doesn't require an API key for local instances, but keep for compatibility
        self.api_key = api_key or os.getenv('OLLAMA_API_KEY', 'ollama')
        
        # Use llama3.2:3b as default, but allow other models
        self.model = model_name or "llama3.2:3b"
        self.base_url = base_url
        self.conversation_history: list[dict] = []
        self.system_instructions = system_instructions

        # Reasoning and verbosity defaults (adapted for Ollama)
        self.default_reasoning = reasoning or {"temperature": 0.8}
        self.default_verbosity = verbosity
        self._validate_reasoning(self.default_reasoning)
        self._validate_verbosity(self.default_verbosity)

    def _validate_reasoning(self, reasoning: dict) -> dict:
        """Validate reasoning parameter (Ollama uses temperature instead of effort)."""
        if not isinstance(reasoning, dict):
            raise ValueError("reasoning must be a dict")
        # Convert OpenAI effort levels to Ollama temperature
        if "effort" in reasoning:
            effort_to_temp = {
                "minimal": 0.2,
                "low": 0.4,
                "medium": 0.8,
                "high": 1.2
            }
            temp = effort_to_temp.get(reasoning["effort"], 0.8)
            reasoning["temperature"] = temp
            reasoning.pop("effort", None)
        if "temperature" not in reasoning:
            reasoning["temperature"] = 0.8
        return reasoning
    
    def _validate_verbosity(self, verbosity: str) -> str:
        """Validate verbosity parameter."""
        if verbosity not in ["low", "medium", "high"]:
            raise ValueError("verbosity must be: low, medium, or high")
        return verbosity

    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

    def _execute_tool_calls(self, tool_calls: list, available_tools: t.Optional[t.Iterable[t.Callable]]) -> dict:
        """Execute tool calls and return results."""
        if not tool_calls or not available_tools:
            return {}
        
        tool_map = {func.__name__: func for func in available_tools}
        results = {}
        
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args = tool_call.get("function", {}).get("arguments", "")
                
                # Handle both string and dict formats
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                elif not isinstance(tool_args, dict):
                    tool_args = {}
                
                if tool_name in tool_map:
                    results[tool_name] = tool_map[tool_name](**tool_args)
                else:
                    results[tool_name] = f"Error: Tool '{tool_name}' not found"
            except Exception as e:
                results[tool_name] = f"Error: {str(e)}"
        
        return results

    def _get_json_type(self, python_type: t.Any) -> str:
        """Convert Python type to JSON schema type."""
        origin = t.get_origin(python_type)
        if origin is t.Union:
            args = t.get_args(python_type)
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                python_type = non_none_types[0]
        
        if origin in (list, tuple):
            return "array"
        if origin is dict:
            return "object"
            
        return self.TYPE_MAPPING.get(python_type, "string")

    def _extract_function_info(self, func: t.Callable) -> dict:
        """Extract function information for tool schema."""
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        
        properties = {}
        required = []
        
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue
                
            param_type = self._get_json_type(param.annotation)
            param_info = {"type": param_type}
            
            if param.default is not inspect.Parameter.empty:
                param_info["default"] = param.default
            else:
                required.append(param_name)
                
            properties[param_name] = param_info
        
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": docstring,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def _build_tools(self, tools: t.Optional[t.Iterable[t.Callable]]) -> list[dict]:
        """Build tool schemas from callable functions."""
        if not tools:
            return []
        
        tool_schemas = []
        for func in tools:
            try:
                tool_schemas.append(self._extract_function_info(func))
            except Exception:
                continue
                
        return tool_schemas

    def _extract_json_schema(self, schema_input: t.Any) -> dict | None:
        """Extract JSON schema from various input types."""
        if schema_input is None:
            return None
            
        if isinstance(schema_input, dict):
            return self._make_schema_strict_compatible(schema_input)
        
        # Try Pydantic methods
        for method_name in ("model_json_schema", "schema"):
            method = getattr(schema_input, method_name, None)
            if callable(method):
                try:
                    return self._make_schema_strict_compatible(method())
                except Exception:
                    continue
                    
        return None

    def _make_schema_strict_compatible(self, schema: dict) -> dict:
        """Make schema compatible with structured output."""
        if not isinstance(schema, dict):
            return schema
        
        strict_schema = schema.copy()
        
        if strict_schema.get("type") == "object":
            if "additionalProperties" not in strict_schema:
                strict_schema["additionalProperties"] = False
        
        # Handle nested objects
        if "properties" in strict_schema:
            for prop_name, prop_schema in strict_schema["properties"].items():
                if isinstance(prop_schema, dict):
                    if prop_schema.get("type") == "object":
                        strict_schema["properties"][prop_name] = self._make_schema_strict_compatible(prop_schema)
                    elif prop_schema.get("type") == "array":
                        items = prop_schema.get("items", {})
                        if isinstance(items, dict) and items.get("type") == "object":
                            strict_schema["properties"][prop_name]["items"] = self._make_schema_strict_compatible(items)
        
        return strict_schema

    def _build_input_messages(self, query: str) -> list[dict]:
        """Build input messages for Ollama chat API."""
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

    def _build_ollama_payload(self, query: str, reasoning: dict = None, verbosity: str = "medium", tools: t.Optional[t.Iterable[t.Callable]] = None) -> dict:
        """Build payload for testing purposes."""
        messages = self._build_input_messages(query)
        
        # Ollama options based on verbosity
        verbosity_to_options = {
            "low": {"num_predict": 50, "temperature": 0.3},
            "medium": {"num_predict": 200, "temperature": 0.8},
            "high": {"num_predict": 500, "temperature": 1.2}
        }
        
        reasoning_config = reasoning or {"temperature": 0.8}
        base_options = verbosity_to_options.get(verbosity, verbosity_to_options["medium"])
        
        # Merge reasoning config with verbosity options
        options = {**base_options, **reasoning_config}
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "15m",  # 15 minutes as requested
            "options": options
        }
        
        # Add tools if provided
        if tools:
            tool_schemas = self._build_tools(tools)
            if tool_schemas:
                payload["tools"] = tool_schemas
        
        return payload

    def invoke(self, query: str, json_schema: t.Optional[dict | t.Any] = None, 
               tools: t.Optional[t.Iterable[t.Callable]] = None, reasoning: t.Optional[dict] = None, 
               verbosity: t.Optional[str] = None, system_instructions: t.Optional[str] = None):
        """Send query to Ollama and return response.
        
        Returns:
            - Tool calls: {"tool_name": str, "tool_return": any}
            - JSON schema: parsed JSON object
            - Otherwise: string response
        """
        # Update system instructions if provided
        if system_instructions is not None:
            self.system_instructions = system_instructions
        
        # Use provided parameters or defaults
        effective_reasoning = reasoning or self.default_reasoning
        effective_verbosity = verbosity or self.default_verbosity
        
        # Validate if provided
        if reasoning is not None:
            effective_reasoning = self._validate_reasoning(reasoning)
        if verbosity is not None:
            effective_verbosity = self._validate_verbosity(verbosity)
        
        # Use Ollama chat API for all queries
        result = self._invoke_ollama_api(query, json_schema, tools, effective_reasoning, effective_verbosity)
        
        # Return tool result in simplified format
        if result.get('tool_calls') and result.get('tool_results'):
            tool_calls = result['tool_calls']
            tool_results = result['tool_results']
            if tool_calls and tool_results:
                first_tool_call = tool_calls[0]
                tool_name = first_tool_call.get("function", {}).get("name", "")
                if tool_name in tool_results:
                    return {
                        "tool_name": tool_name, 
                        "tool_return": tool_results[tool_name],
                        "text": result['text']
                    }
        
        # Parse JSON if schema was used
        if json_schema:
            try:
                return json.loads(result['text'])
            except json.JSONDecodeError:
                return result['text']
        
        return result['text']

    def _invoke_ollama_api(self, query: str, json_schema: t.Optional[dict | t.Any] = None, 
                          tools: t.Optional[t.Iterable[t.Callable]] = None, reasoning: dict = None, 
                          verbosity: str = "medium") -> dict:
        """Handle all queries using Ollama chat API."""
        url = f"{self.base_url}/api/chat"
        
        messages = self._build_input_messages(query)
        
        # Ollama options based on verbosity and reasoning
        verbosity_to_options = {
            "low": {"num_predict": 50, "temperature": 0.3},
            "medium": {"num_predict": 200, "temperature": 0.8},
            "high": {"num_predict": 500, "temperature": 1.2}
        }
        
        reasoning_config = reasoning or {"temperature": 0.8}
        base_options = verbosity_to_options.get(verbosity, verbosity_to_options["medium"])
        
        # Merge reasoning config with verbosity options
        options = {**base_options, **reasoning_config}
        
        # Build payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "15m",  # 15 minutes as requested
            "options": options
        }
        
        # Add tools if provided
        tool_schemas = self._build_tools(tools)
        if tool_schemas:
            payload["tools"] = tool_schemas
        
        # Add structured output if provided
        schema = self._extract_json_schema(json_schema)
        if schema:
            payload["format"] = "json"
            # Add schema instruction to system message
            schema_instruction = f"Please respond with valid JSON that follows this schema: {json.dumps(schema)}"
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{schema_instruction}"
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})
            payload["messages"] = messages
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', str(e))
                print(f"Ollama API Error: {error_msg}")
                print(f"Request payload: {json.dumps(payload, indent=2)}")
                raise
            except json.JSONDecodeError:
                print(f"HTTP Error: {e}")
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
                raise
        
        data = response.json()
        
        # Extract text and tool calls from Ollama response
        message = data.get("message", {})
        assistant_response = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        
        # Execute tool calls
        tool_execution_results = {}
        if tool_calls and tools:
            tool_execution_results = self._execute_tool_calls(tool_calls, tools)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return {
            "text": assistant_response,
            "tool_calls": tool_calls,
            "tool_results": tool_execution_results
        }

if __name__ == "__main__":
    """Example usage."""
    
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"{city}: 24°C, sunny"

    chat = OllamaChat()

    # Basic chat
    print("=== Basic Chat ===")
    result = chat.invoke("What is Python?")
    print(f"Q: What is Python?")
    print(f"A: {result[:100]}...")
    
    # Tool calling
    print("\n=== Tool Call ===")
    tool_result = chat.invoke("Weather in Paris?", tools=[get_weather])
    print(f"Tool Result: {tool_result}")
    
    # Structured output
    print("\n=== Structured Output ===")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"],
        "additionalProperties": False
    }
    
    structured = chat.invoke("Extract: John is 30 years old", json_schema=schema)
    print(f"Structured: {structured}")
    
    print("\n✅ Examples complete")
