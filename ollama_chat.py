import inspect
import json
import typing as t
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class OllamaChat:
    """Simplified Ollama chat client with tool calling, structured outputs, and anti-truncation features.
    
    Features:
    - Prevents output truncation with num_predict=-1 (unlimited output)
    - Context window (8192 tokens by default, configurable via num_ctx env variable)
    - Environment variable support for num_ctx configuration
    - Configurable output parameters for different use cases
    - Tool calling and function execution
    - Structured JSON outputs with schema validation
    - Conversation history management
    
    Basic usage:
        chat = OllamaChat()
        response = chat.invoke("Hello")  # Returns string
        response = chat.invoke("Extract data", json_schema=schema)  # Returns parsed JSON
        response = chat.invoke("Get weather", tools=[weather_func])  # Returns {"tool_name": str, "tool_return": any}
        
    Environment variable configuration:
        export num_ctx=16384  # Set context window to 16k tokens
        chat = OllamaChat()   # Will use 16384 from environment
        
    Anti-truncation usage:
        chat.optimize_for_long_output()  # Configure for maximum output length
        chat.set_output_parameters(max_tokens=-1, context_size=65536)  # Custom parameters
    """
    
    TYPE_MAPPING = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}

    def __init__(self, api_key: str = None, model_name: str = "qwen3:4b", 
                 reasoning: dict = None, system_instructions: str = "", base_url: str = "http://localhost:11434"):
        """Initialize the Ollama chat client."""
        # Ollama doesn't require an API key for local instances, but keep for compatibility
        self.api_key = api_key or os.getenv('OLLAMA_API_KEY', 'ollama')
        
        # Use llama3.2:3b as default, but allow other models
        self.model = model_name or "llama3.2:3b"
        self.base_url = base_url
        self.conversation_history: list[dict] = []
        self.system_instructions = system_instructions

        # Get num_ctx from environment variable or use default
        env_num_ctx = os.getenv('num_ctx')
        default_num_ctx = 4*4096  # Default to 16384 tokens
        
        if env_num_ctx is not None:
            try:
                num_ctx_value = int(env_num_ctx)
                print(f"Using num_ctx from environment: {num_ctx_value}")
            except ValueError:
                print(f"Invalid num_ctx environment variable '{env_num_ctx}', using default: {default_num_ctx}")
                num_ctx_value = default_num_ctx
        else:
            num_ctx_value = default_num_ctx

        # Reasoning defaults (adapted for Ollama) - Set parameters to prevent truncation
        self.default_reasoning = reasoning or {
            "temperature": 0.8,
            "num_predict": -1,  # -1 means unlimited output until stop condition
            "num_ctx": num_ctx_value,   # Context window from env or default
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1
        }
        self._validate_reasoning(self.default_reasoning)

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
        
        # Set defaults to prevent truncation if not provided
        if "temperature" not in reasoning:
            reasoning["temperature"] = 0.8
        if "num_predict" not in reasoning:
            reasoning["num_predict"] = -1  # Unlimited output
        if "num_ctx" not in reasoning:
            # Get from environment or use default
            env_num_ctx = os.getenv('num_ctx')
            default_num_ctx = 2*4096
            if env_num_ctx is not None:
                try:
                    reasoning["num_ctx"] = int(env_num_ctx)
                except ValueError:
                    reasoning["num_ctx"] = default_num_ctx
            else:
                reasoning["num_ctx"] = default_num_ctx
        if "top_p" not in reasoning:
            reasoning["top_p"] = 0.9
        if "top_k" not in reasoning:
            reasoning["top_k"] = 40
        if "repeat_penalty" not in reasoning:
            reasoning["repeat_penalty"] = 1.1
            
        return reasoning
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

    def set_output_parameters(self, max_tokens: int = -1, context_size: int = 2*4096, 
                             temperature: float = 0.8, top_p: float = 0.9, top_k: int = 40) -> None:
        """Configure output parameters to prevent truncation.
        
        Args:
            max_tokens (int): Maximum output tokens (-1 for unlimited, recommended to prevent truncation)
            context_size (int): Context window size (8192 tokens by default)
            temperature (float): Randomness/creativity (0.0-2.0, default 0.8)
            top_p (float): Nucleus sampling threshold (0.0-1.0, default 0.9)
            top_k (int): Top-k sampling limit (default 40)
        """
        self.default_reasoning.update({
            "num_predict": max_tokens,
            "num_ctx": context_size,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        })

    def optimize_for_long_output(self) -> None:
        """Optimize settings specifically for generating long, untruncated outputs."""
        self.set_output_parameters(
            max_tokens=-1,      # Unlimited output
            context_size=65536, # Very large context (64k tokens)
            temperature=0.7,    # Slightly lower temperature for consistency
            top_p=0.95,         # Higher top_p for more diverse vocabulary
            top_k=50            # Higher top_k for more options
        )

    def get_effective_num_ctx(self) -> int:
        """Get the current effective num_ctx value (from env or default)."""
        env_num_ctx = os.getenv('num_ctx')
        if env_num_ctx is not None:
            try:
                return int(env_num_ctx)
            except ValueError:
                pass
        return self.default_reasoning.get('num_ctx', 2*4096)

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

    def _build_ollama_payload(self, query: str, reasoning: dict = None, tools: t.Optional[t.Iterable[t.Callable]] = None) -> dict:
        """Build payload for testing purposes."""
        messages = self._build_input_messages(query)
        
        # Ollama options based on reasoning - ensure no truncation
        reasoning_config = reasoning or self.default_reasoning
        
        # Ensure we have anti-truncation parameters
        if "num_predict" not in reasoning_config:
            reasoning_config["num_predict"] = -1  # Unlimited output
        if "num_ctx" not in reasoning_config:
            # Get from environment or use default
            env_num_ctx = os.getenv('num_ctx')
            default_num_ctx = 2*4096
            if env_num_ctx is not None:
                try:
                    reasoning_config["num_ctx"] = int(env_num_ctx)
                except ValueError:
                    reasoning_config["num_ctx"] = default_num_ctx
            else:
                reasoning_config["num_ctx"] = default_num_ctx
            
        options = reasoning_config
        
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
               system_instructions: t.Optional[str] = None):
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
        
        # Validate if provided
        if reasoning is not None:
            effective_reasoning = self._validate_reasoning(reasoning)
        
        # Use Ollama chat API for all queries
        result = self._invoke_ollama_api(query, json_schema, tools, effective_reasoning)
        
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
                        "text": result['text'],
                        "raw": result['raw']
                    }
        
        # Parse JSON if schema was used
        if json_schema:
            try:
                return json.loads(result['text'])
            except json.JSONDecodeError:
                return result['text']
        
        return result['text']

    def _invoke_ollama_api(self, query: str, json_schema: t.Optional[dict | t.Any] = None, 
                          tools: t.Optional[t.Iterable[t.Callable]] = None, reasoning: dict = None) -> dict:
        """Handle all queries using Ollama chat API."""
        url = f"{self.base_url}/api/chat"
        
        messages = self._build_input_messages(query)
        
        # Ollama options based on reasoning - ensure no truncation
        reasoning_config = reasoning or self.default_reasoning
        
        # Ensure we have anti-truncation parameters
        if "num_predict" not in reasoning_config:
            reasoning_config["num_predict"] = -1  # Unlimited output
        if "num_ctx" not in reasoning_config:
            # Get from environment or use default
            env_num_ctx = os.getenv('num_ctx')
            default_num_ctx = 2*4096
            if env_num_ctx is not None:
                try:
                    reasoning_config["num_ctx"] = int(env_num_ctx)
                except ValueError:
                    reasoning_config["num_ctx"] = default_num_ctx
            else:
                reasoning_config["num_ctx"] = default_num_ctx
            
        options = reasoning_config
        
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
        # Increase timeout for long responses to prevent truncation due to timeouts
        timeout = 600  # 10 minutes for very long outputs
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        
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
            "tool_results": tool_execution_results,
            "raw": data
        }

if __name__ == "__main__":
    """Example usage with anti-truncation features."""
    
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"{city}: 24°C, sunny"

    # Initialize with anti-truncation defaults
    chat = OllamaChat()
    
    print("=== Anti-Truncation Configuration ===")
    print(f"Default parameters: {chat.default_reasoning}")
    
    # Basic chat
    print("\n=== Basic Chat ===")
    result = chat.invoke("What is Python?")
    print(f"Q: What is Python?")
    print(f"A: {result[:200]}..." if len(result) > 200 else result)
    
    # Test long output generation
    print("\n=== Long Output Test ===")
    chat.optimize_for_long_output()  # Configure for maximum output length
    long_result = chat.invoke("Write a detailed explanation of machine learning with examples.")
    print(f"Generated {len(long_result)} characters")
    print(f"Sample: {long_result[:300]}...")
    
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
    
    # Custom output parameters
    print("\n=== Custom Parameters Test ===")
    chat.set_output_parameters(max_tokens=5000, context_size=16384)
    custom_result = chat.invoke("Explain quantum computing in detail")
    print(f"Custom output length: {len(custom_result)} characters")
    
    print("\n✅ Examples complete with anti-truncation features")
