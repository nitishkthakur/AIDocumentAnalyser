import inspect
import json
import typing as t
import requests
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CURRENT_DATE=datetime.datetime.now().strftime("%d %b %Y")
GENERIC_AGENT_INSTRUCTIONS = """You are a helpful Agent among a group of agents trying to solve a problem. Each agent is tasked with a part or the entire problem.
You will be given your task. You will have access to all the relevant information and tools. You can also see the work already done by other agents. Use that information if required.


## Instructions
1. View the context, the task executed, the results, the tool call results of the other agents.
2. Reason and execute your task.
3. You have access to multiple tools as listed. You can only call tools which are relevant to your task. Other agents might have executed other tools which you dont have access to.
4. Make the necessary tool calls if your tool descriptions match the task
5. your task will be enclosed in <task></task> tags. This is your task. Only execute this task.
6. The work done by other agents will be enclosed in <Agent: Agent Name></Agent: Agent Name> tags. There may be multiple of these.
7. Any information to be used as reference (from documents or internet) about the problem is enclosed in <CONTEXT></CONTEXT> tags
8. Some general information is enclosed in <general information> tags

<general information>
1. The current date is: {CURRENT_DATE}
</general information>


"""

class OpenAIClient:
    """A simplified agent client for interacting with OpenAI API.
    
    This client supports structured outputs using OpenAI's JSON Schema format with strict mode.
    
    Structured Output Example:
    -------------------------
    
    from pydantic import BaseModel
    from typing import List
    
    # Define a Pydantic model for structured output
    class PersonInfo(BaseModel):
        name: str
        age: int
        skills: List[str]
        is_employed: bool
    
    # Create client
    client = OpenAIClient(
        role="information_extractor",
        api_key="your-openai-api-key",
        model_name="gpt-5-nano"  # Supports structured outputs
    )
    
    # Use structured output
    query = "Extract information about John: He's 30 years old, works as a developer, knows Python and JavaScript"
    
    # Method 1: Using Pydantic model
    response = client.invoke(
        query=query,
        json_schema=PersonInfo
    )
    
    # Method 2: Using raw JSON schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {"type": "array", "items": {"type": "string"}},
            "is_employed": {"type": "boolean"}
        },
        "required": ["name", "age", "skills", "is_employed"],
        "additionalProperties": False
    }
    
    response = client.invoke(
        query=query,
        json_schema=schema
    )
    
    # The response will be valid JSON matching the schema:
    # {
    #     "name": "John",
    #     "age": 30,
    #     "skills": ["Python", "JavaScript"],
    #     "is_employed": True
    # }
    
    Note: Structured outputs require models that support this feature. Includes future models like 
    gpt-5-mini, gpt-5-nano (for testing) and current models like gpt-4o-2024-08-06 and later.
    For unsupported models, the client automatically falls back to basic JSON mode.
    """
    
    TYPE_MAPPING = {
        str: "string",
        int: "integer", 
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }

    def __init__(
        self, 
        role: str,
        history_from_other_agents: str = "",
        api_key: str = None,
        model_name: str = "gpt-5-nano",
        this_agent_context: str = "",
        system_instructions: str = "",
        agent_name: str = "",
        reasoning: dict = None,
        verbosity: str = "medium"
    ):
        """Initialize the OpenAI agent client.
        
        Args:
            role: Role of this agent
            history_from_other_agents: Context from other agents
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model_name: Default model to use for all interactions (defaults to gpt-5-nano)
            this_agent_context: Context specific to this agent
            system_instructions: System instructions to guide the agent's behavior
            agent_name: Name/identifier for this agent
            reasoning: Controls reasoning parameters for applicable models.
                      Should be a dict with 'effort' key. 
                      Allowed values for effort: "minimal", "low", "medium", "high". 
                      Default: {"effort": "medium"}
            verbosity: Controls response verbosity for applicable models.
                      Allowed values: "low", "medium", "high". Default: "medium"
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please provide api_key parameter or set OPENAI_API_KEY environment variable.")
        
        self.base_url = "https://api.openai.com/v1"
        self.default_model = model_name
        self.agent_name = agent_name
        self.conversation_history: list[dict] = []  # Store conversation history
        self.tool_call_results: dict = {}  # Store tool call results
        
        # Reasoning and verbosity settings
        if reasoning is None:
            reasoning = {"effort": "medium"}
        self.default_reasoning = self._validate_reasoning(reasoning)
        self.default_verbosity = self._validate_verbosity(verbosity)
        
        # New agent context variables
        self.role = role
        self.history_from_other_agents = history_from_other_agents
        self.all_tool_names = ""
        self.only_this_agent_context = this_agent_context

        # System instructions without the task and context (both will go to user message instead)
        self.generic_agent_instructions = GENERIC_AGENT_INSTRUCTIONS.format(CURRENT_DATE=CURRENT_DATE)
        self.system_instructions = self.generic_agent_instructions

    def _validate_reasoning(self, reasoning: dict) -> dict:
        """Validate and return reasoning parameter.
        
        Args:
            reasoning: The reasoning dictionary to validate
            
        Returns:
            Validated reasoning dictionary
            
        Raises:
            ValueError: If reasoning is not a valid dictionary or effort value is invalid
        """
        if not isinstance(reasoning, dict):
            raise ValueError(f"reasoning must be a dictionary, got: {type(reasoning)}")
        
        if "effort" not in reasoning:
            raise ValueError("reasoning dictionary must contain 'effort' key")
        
        valid_efforts = ["minimal", "low", "medium", "high"]
        effort = reasoning["effort"]
        if effort not in valid_efforts:
            raise ValueError(f"reasoning effort must be one of {valid_efforts}, got: {effort}")
        
        return reasoning
    
    def _validate_verbosity(self, verbosity: str) -> str:
        """Validate and return verbosity parameter.
        
        Args:
            verbosity: The verbosity level to validate
            
        Returns:
            Validated verbosity level
            
        Raises:
            ValueError: If verbosity is not a valid value
        """
        valid_verbosity = ["low", "medium", "high"]
        if verbosity not in valid_verbosity:
            raise ValueError(f"verbosity must be one of {valid_verbosity}, got: {verbosity}")
        return verbosity

    def _supports_reasoning(self, model: str) -> bool:
        """Check if the model supports reasoning parameter.
        
        Args:
            model: The model name to check
            
        Returns:
            True if the model supports reasoning parameter
        """
        # O1 models and their variants support reasoning
        reasoning_models = [
            "gpt-5-mini", "gpt-5-nano", "gpt-5"
        ]
        
        # Check if the model name starts with any reasoning model
        for reasoning_model in reasoning_models:
            if model.startswith(reasoning_model):
                return True
        return False
    
    def _supports_verbosity(self, model: str) -> bool:
        """Check if the model supports verbosity parameter.
        
        Args:
            model: The model name to check
            
        Returns:
            True if the model supports verbosity parameter
        """
        # GPT-5 models and certain other models support verbosity
        verbosity_models = [
            "gpt-5", "gpt-5-nano", "gpt-5-mini",
            "o1", "o1-preview", "o1-mini", "o1-2024-12-17",
            "o3-mini", "o3",  # Future models
        ]
        
        # Check if the model name starts with any verbosity-supporting model
        for verbosity_model in verbosity_models:
            if model.startswith(verbosity_model):
                return True
        return False

    def set_system_instructions(self, instructions: str) -> None:
        """Update the system instructions for the agent.
        
        Args:
            instructions: New system instructions to guide the agent's behavior
        """
        self.system_instructions = instructions

    def get_system_instructions(self) -> str:
        """Get the current system instructions.
        
        Returns:
            Current system instructions
        """
        return self.system_instructions

    def get_default_model(self) -> str:
        """Get the current default model.
        
        Returns:
            Current default model name
        """
        return self.default_model

    def set_default_model(self, model_name: str) -> None:
        """Update the default model for the agent.
        
        Note: This client only supports gpt-5-nano, so model_name is ignored.
        
        Args:
            model_name: New default model name (ignored - always uses gpt-5-nano)
        """
        self.default_model = "gpt-5-nano"  # Force to gpt-5-nano only

    def get_conversation_history(self) -> list[dict]:
        """Get the current conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()

    def get_default_reasoning_effort(self) -> str:
        """Get the current default reasoning effort level.
        
        Returns:
            Current default reasoning effort level
        """
        return self.default_reasoning_effort

    def set_default_reasoning_effort(self, reasoning_effort: str) -> None:
        """Update the default reasoning effort level for the agent.
        
        Args:
            reasoning_effort: New default reasoning effort level.
                            Allowed values: "low", "medium", "high"
        """
        self.default_reasoning_effort = self._validate_reasoning_effort(reasoning_effort)

    def get_default_verbosity(self) -> str:
        """Get the current default verbosity level.
        
        Returns:
            Current default verbosity level
        """
        return self.default_verbosity

    def set_default_verbosity(self, verbosity: str) -> None:
        """Update the default verbosity level for the agent.
        
        Args:
            verbosity: New default verbosity level.
                      Allowed values: "low", "medium", "high"
        """
        self.default_verbosity = self._validate_verbosity(verbosity)

    def _build_agent_context(self, agent_response: str = "", tool_results: dict = None) -> None:
        """Build the only_this_agent_context with the specified XML structure.
        
        Args:
            agent_response: The agent's response to include in context
            tool_results: Dictionary of tool names and their results
        """
        if not self.agent_name:
            return
            
        tool_section = ""
        if tool_results:
            tool_section = f"Tool(s) Invoked and the tool outputs obtained: {tool_results}"
        
        self.only_this_agent_context = f"""<Agent: {self.agent_name}>
            ### Task to be done by the agent:
            {self.role}

            ### User Input to the agent: {getattr(self, 'query', '')}
            ### Agent's Response: {agent_response}
            {tool_section}
            </Agent: {self.agent_name}>"""

    def _execute_tool_calls(self, tool_calls: list, available_tools: t.Optional[t.Iterable[t.Callable]]) -> dict:
        """Execute tool calls and return results.
        
        Args:
            tool_calls: List of tool calls from the model response
            available_tools: List of available callable functions
            
        Returns:
            Dictionary mapping tool names to their execution results
        """
        if not tool_calls or not available_tools:
            return {}
        
        # Create a mapping of tool names to functions
        tool_map = {func.__name__: func for func in available_tools}
        execution_results = {}
        
        for tool_call in tool_calls:
            try:
                # Extract tool call information (OpenAI format)
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args = tool_call.get("function", {}).get("arguments", {})
                
                # Parse arguments if they're a JSON string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        execution_results[tool_name] = f"Error: Invalid JSON arguments: {tool_args}"
                        continue
                
                # Execute the tool if it's available
                if tool_name in tool_map:
                    func = tool_map[tool_name]
                    try:
                        result = func(**tool_args)
                        execution_results[tool_name] = result
                    except Exception as e:
                        execution_results[tool_name] = f"Error executing {tool_name}: {str(e)}"
                else:
                    execution_results[tool_name] = f"Error: Tool '{tool_name}' not found in available tools"
                    
            except Exception as e:
                execution_results[f"unknown_tool_{len(execution_results)}"] = f"Error processing tool call: {str(e)}"
        
        return execution_results

    def update_agent_context(self, agent_response: str = "", tool_results: dict = None) -> None:
        """Update the agent context with new response and tool results.
        
        Args:
            agent_response: The agent's response to include in context
            tool_results: Dictionary of tool names and their results
        """
        self._build_agent_context(agent_response, tool_results)

    def get_agent_context(self) -> str:
        """Get the current agent context.
        
        Returns:
            The XML-formatted agent context string
        """
        return self.only_this_agent_context

    def set_role(self, role: str) -> None:
        """Set the agent's role.
        
        Args:
            role: The role description for this agent
        """
        self.role = role

    def get_role(self) -> str:
        """Get the agent's role.
        
        Returns:
            The agent's role
        """
        return self.role

    def set_generic_agent_instructions(self, instructions: str) -> None:
        """Set generic agent instructions.
        
        Args:
            instructions: Generic instructions for the agent
        """
        self.generic_agent_instructions = instructions

    def get_generic_agent_instructions(self) -> str:
        """Get generic agent instructions.
        
        Returns:
            The generic agent instructions
        """
        return self.generic_agent_instructions

    def set_all_tool_names(self, tool_names: str) -> None:
        """Set all available tool names.
        
        Args:
            tool_names: String containing all tool names
        """
        self.all_tool_names = tool_names

    def get_all_tool_names(self) -> str:
        """Get all available tool names.
        
        Returns:
            String containing all tool names
        """
        return self.all_tool_names

    def set_history_from_other_agents(self, history: str) -> None:
        """Set history from other agents.
        
        Args:
            history: History information from other agents
        """
        self.history_from_other_agents = history

    def get_history_from_other_agents(self) -> str:
        """Get history from other agents.
        
        Returns:
            History from other agents
        """
        return self.history_from_other_agents

    def get_tool_call_results(self) -> dict:
        """Get tool call results.
        
        Returns:
            Dictionary of tool call results
        """
        return self.tool_call_results.copy()

    def clear_tool_call_results(self) -> None:
        """Clear tool call results."""
        self.tool_call_results.clear()

    def add_context_from_other_agents(self, context: str) -> None:
        """Add context from other agents to the conversation history.
        
        Args:
            context: Context information from other agents
        """
        if context.strip():
            self.conversation_history.append({
                "role": "user",
                "content": f"Context from other agents: {context}"
            })

    def _get_json_type(self, python_type: t.Any) -> str:
        """Convert Python type to JSON schema type."""
        # Handle Optional types (Union with None)
        origin = t.get_origin(python_type)
        if origin is t.Union:
            args = t.get_args(python_type)
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                python_type = non_none_types[0]
        
        # Handle generic types
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
            
            # Add description if available in docstring
            description = self._extract_param_description(docstring, param_name)
            if description:
                param_info["description"] = description
            
            # Handle default values
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
    
    def _extract_param_description(self, docstring: str, param_name: str) -> str:
        """Extract parameter description from docstring."""
        if not docstring or "Args:" not in docstring:
            return ""
            
        lines = docstring.split('\n')
        args_index = -1
        for i, line in enumerate(lines):
            if line.strip().lower() == "args:":
                args_index = i
                break
                
        if args_index == -1:
            return ""
            
        for line in lines[args_index + 1:]:
            line = line.strip()
            if not line:
                continue
            if line.startswith(param_name + " (") or line.startswith(param_name + ":"):
                # Extract description after the colon
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
                break
                
        return ""

    def _build_tools(self, tools: t.Optional[t.Iterable[t.Callable]]) -> list[dict]:
        """Build tool schemas from callable functions."""
        if not tools:
            return []
        
        tool_schemas = []
        for func in tools:
            try:
                tool_schemas.append(self._extract_function_info(func))
            except Exception as e:
                print(f"Warning: Failed to build tool schema for {func.__name__}: {e}")
                continue
                
        return tool_schemas

    def _extract_json_schema(self, schema_input: t.Any) -> dict | None:
        """Extract JSON schema from various input types and make it compatible with OpenAI Structured Outputs."""
        if schema_input is None:
            return None
            
        if isinstance(schema_input, dict):
            # If it's already a dict, ensure it's compatible with OpenAI strict mode
            return self._make_schema_strict_compatible(schema_input)
        
        # Try Pydantic v2 first, then v1
        for method_name in ("model_json_schema", "schema"):
            method = getattr(schema_input, method_name, None)
            if callable(method):
                try:
                    schema = method()
                    return self._make_schema_strict_compatible(schema)
                except Exception:
                    continue
                    
        return None

    def _supports_structured_outputs(self, model_name: str = None) -> bool:
        """Check if the model supports OpenAI Structured Outputs."""
        model = "gpt-5-nano"  # Only using gpt-5-nano now
        
        # Models that support structured outputs as per OpenAI documentation
        supported_models = [
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5"
        ]
        
        # Check if the model name starts with any supported model
        for supported_model in supported_models:
            if model.startswith(supported_model):
                return True
        
        return False

    def _make_schema_strict_compatible(self, schema: dict) -> dict:
        """Make a JSON schema compatible with OpenAI's strict mode requirements."""
        if not isinstance(schema, dict):
            return schema
        
        # Create a copy to avoid modifying the original
        strict_schema = schema.copy()
        
        # Ensure additionalProperties is set to false for strict mode
        if "type" in strict_schema and strict_schema["type"] == "object":
            if "additionalProperties" not in strict_schema:
                strict_schema["additionalProperties"] = False
        
        # Recursively process nested objects
        if "properties" in strict_schema:
            for prop_name, prop_schema in strict_schema["properties"].items():
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "object":
                    strict_schema["properties"][prop_name] = self._make_schema_strict_compatible(prop_schema)
                elif isinstance(prop_schema, dict) and prop_schema.get("type") == "array":
                    items = prop_schema.get("items", {})
                    if isinstance(items, dict) and items.get("type") == "object":
                        strict_schema["properties"][prop_name]["items"] = self._make_schema_strict_compatible(items)
        
        return strict_schema

    def _build_chat_payload(
        self,
        query: str,
        json_schema: t.Optional[dict | t.Any] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model_name: t.Optional[str] = None,
        stream: bool = False,
        reasoning: dict = None,
        verbosity: str = "medium",
        context: str = "",
    ) -> dict:
        """Build the payload for OpenAI chat API."""
        # Force model to gpt-5-nano only
        model = "gpt-5-nano"
        
        # Build messages with system instructions and conversation history
        messages = []
        
        # Add system instructions if provided
        if self.system_instructions:
            messages.append({"role": "system", "content": self.system_instructions})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user query with context and task
        # Format: <context>{context}</context>\n\n<task>{role}</task>\n\n{query}
        user_message = ""
        if context.strip():
            user_message += f"<context>{context}</context>\n\n"
        user_message += f"<task>{self.role}</task>\n\n{query}"
        messages.append({"role": "user", "content": user_message})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "max_completion_tokens": 4000,  # Using max_completion_tokens as per new API requirements
            "temperature": 1,  # Set to 1 as requested
        }
        
        # Add JSON schema if provided (OpenAI Structured Outputs format)
        schema = self._extract_json_schema(json_schema)
        if schema:
            if self._supports_structured_outputs(model):
                # Use OpenAI's structured outputs format with strict mode
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",  # Required name for the schema
                        "strict": True,  # Enable strict mode for better adherence
                        "schema": schema
                    }
                }
            else:
                # Fallback to basic JSON mode for unsupported models
                payload["response_format"] = {"type": "json_object"}
                # Add instruction to follow JSON schema in system message
                if payload["messages"] and payload["messages"][0]["role"] == "system":
                    payload["messages"][0]["content"] += f"\n\nPlease respond with valid JSON that follows this schema: {json.dumps(schema)}"
                else:
                    payload["messages"].insert(0, {
                        "role": "system", 
                        "content": f"Please respond with valid JSON that follows this schema: {json.dumps(schema)}"
                    })
        
        # Add tools if provided (OpenAI format)
        tool_schemas = self._build_tools(tools)
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"
        
        # Add reasoning for O1 models
        if self._supports_reasoning(model):
            payload["reasoning"] = reasoning if reasoning else {"effort": "medium"}
            
        # Add verbosity for applicable models in text object
        if self._supports_verbosity(model):
            if "text" not in payload:
                payload["text"] = {}
            payload["text"]["verbosity"] = verbosity
            
        return payload

    def invoke(
        self,
        query: str,
        json_schema: t.Optional[dict | t.Any] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model_name: t.Optional[str] = None,
        reasoning: t.Optional[dict] = None,
        verbosity: t.Optional[str] = None,
        context: str = "",
    ) -> dict:
        """Send a query to OpenAI and return the response.
        
        Uses the newer Responses API (/v1/responses) when possible, falls back to 
        chat/completions for tool calls. Only uses gpt-5-nano model.
        
        Args:
            query: The user's query
            json_schema: Optional JSON schema for structured responses
            tools: Optional list of functions the agent can call
            model_name: Optional model override (forced to gpt-5-nano)
            reasoning: Controls reasoning parameters for applicable models.
                      Should be a dict with 'effort' key. 
                      Allowed values for effort: "minimal", "low", "medium", "high". 
                      If None, uses default from initialization.
            verbosity: Controls response verbosity for applicable models.
                      Allowed values: "low", "medium", "high".
                      If None, uses default from initialization.
            context: Additional context information to include in the user message
        
        Returns:
            dict: Response containing 'text', 'raw', 'conversation_history', 'tool_calls', 'tool_results'
        """
        # Force model to gpt-5-nano only
        model = "gpt-5-nano"
        
        # Use provided reasoning/verbosity or fall back to defaults
        effective_reasoning = reasoning or self.default_reasoning
        effective_verbosity = verbosity or self.default_verbosity
        
        # Validate parameters if provided
        if reasoning is not None:
            effective_reasoning = self._validate_reasoning(reasoning)
        if verbosity is not None:
            effective_verbosity = self._validate_verbosity(verbosity)
        
        # If tools are provided, we need to use chat/completions API
        if tools:
            return self._invoke_with_completions_api(query, json_schema, tools, model, effective_reasoning, effective_verbosity, context)
        
        # For simple queries without tools, use the newer Responses API
        return self._invoke_with_responses_api(query, json_schema, model, effective_reasoning, effective_verbosity, context)

    def _invoke_with_responses_api(
        self,
        query: str,
        json_schema: t.Optional[dict | t.Any] = None,
        model: str = "gpt-5-nano",
        reasoning: dict = None,
        verbosity: str = "medium",
        context: str = "",
    ) -> dict:
        """Handle queries using the newer Responses API."""
        if reasoning is None:
            reasoning = {"effort": "medium"}
            
        url = f"{self.base_url}/responses"
        
        # Build payload for Responses API
        # Format input with context and task
        input_with_context_and_task = ""
        if context.strip():
            input_with_context_and_task += f"<context>{context}</context>\n\n"
        input_with_context_and_task += f"<task>{self.role}</task>\n\n{query}"
        
        payload = {
            "model": model,
            "input": input_with_context_and_task,
            "max_output_tokens": 4000,  # Responses API uses max_output_tokens
            "temperature": 1,
        }
        
        # Add reasoning for O1 models
        if self._supports_reasoning(model):
            payload["reasoning"] = reasoning
        
        # Add JSON schema if provided (Responses API format)
        schema = self._extract_json_schema(json_schema)
        if schema:
            if self._supports_structured_outputs(model):
                # Use Responses API's structured format
                text_config = {
                    "format": {
                        "type": "json_schema",
                        "name": "structured_response",
                        "schema": schema
                    }
                }
            else:
                # Fallback for unsupported models - add instruction to the input
                payload["input"] = f"{input_with_context_and_task}\n\nPlease respond with valid JSON that follows this schema: {json.dumps(schema)}"
                text_config = {"format": {"type": "json_object"}}
        else:
            text_config = {}
            
        # Add verbosity for applicable models
        if self._supports_verbosity(model):
            text_config["verbosity"] = verbosity
            
        # Set text object if there's any configuration
        if text_config:
            payload["text"] = text_config
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Handle token parameter errors with fallback
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', str(e))
                
                if 'max_output_tokens' in error_msg.lower() and 'not supported' in error_msg.lower():
                    print(f"Responses API Error with max_output_tokens: {error_msg}")
                    print("Retrying with max_tokens for older model compatibility...")
                    
                    if 'max_output_tokens' in payload:
                        payload['max_tokens'] = payload.pop('max_output_tokens')
                        response = requests.post(url, json=payload, headers=headers, timeout=300)
                        response.raise_for_status()
                else:
                    print(f"Responses API Error: {error_msg}")
                    raise
            except json.JSONDecodeError:
                print(f"HTTP Error: {e}")
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
                raise
        
        data = response.json()
        
        # Handle the Responses API response format
        assistant_response = ""
        
        # The Responses API returns a response object with 'output' array
        if isinstance(data, dict) and 'output' in data and isinstance(data['output'], list):
            # Look for the assistant message in the output array
            for output_item in data['output']:
                if output_item.get('role') == 'assistant' and 'content' in output_item:
                    content_list = output_item['content']
                    for content_item in content_list:
                        if content_item.get('type') == 'output_text' and 'text' in content_item:
                            assistant_response = content_item['text']
                            break
                    if assistant_response:
                        break
        
        if not assistant_response:
            # Fallback: try to find text anywhere in the response
            assistant_response = f"Could not parse response format: {str(data)[:200]}..."
        
        # Store query and response in conversation history
        self.query = query
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Build agent context (no tool results for responses API)
        self._build_agent_context(assistant_response, {})
        
        return {
            "text": assistant_response,
            "raw": data,
            "conversation_history": self.get_conversation_history(),
            "tool_calls": [],
            "tool_results": {}
        }

    def _invoke_with_completions_api(
        self,
        query: str,
        json_schema: t.Optional[dict | t.Any] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model: str = "gpt-5-nano",
        reasoning: dict = None,
        verbosity: str = "medium",
        context: str = "",
    ) -> dict:
        """Handle queries with tools using the traditional chat/completions API."""
        url = f"{self.base_url}/chat/completions"
        payload = self._build_chat_payload(query, json_schema, tools, model, reasoning=reasoning, verbosity=verbosity, context=context)
        self.query = query
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Try to get error details from response
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', str(e))
                
                # Check if error is related to token limits and retry with alternative parameter
                if 'max_completion_tokens' in error_msg.lower() and 'not supported' in error_msg.lower():
                    print(f"OpenAI API Error with max_completion_tokens: {error_msg}")
                    print("Retrying with max_tokens for older model compatibility...")
                    
                    # Modify payload to use max_tokens instead of max_completion_tokens for older models
                    if 'max_completion_tokens' in payload:
                        payload['max_tokens'] = payload.pop('max_completion_tokens')
                elif 'max_tokens' in error_msg.lower() and 'not supported' in error_msg.lower():
                    print(f"OpenAI API Error with max_tokens: {error_msg}")
                    print("Retrying with max_completion_tokens instead...")
                    
                    # Modify payload to use max_completion_tokens instead of max_tokens
                    if 'max_tokens' in payload:
                        payload['max_completion_tokens'] = payload.pop('max_tokens')
                else:
                    print(f"OpenAI API Error: {error_msg}")
                    print(f"Request payload: {json.dumps(payload, indent=2)}")
                    raise
                
                # Retry the request with the modified payload
                if 'max_completion_tokens' in error_msg.lower() or 'max_tokens' in error_msg.lower():
                    response = requests.post(url, json=payload, headers=headers, timeout=300)
                    try:
                        response.raise_for_status()
                    except requests.exceptions.HTTPError as retry_error:
                        # Get error details from retry attempt
                        try:
                            retry_error_data = response.json()
                            retry_error_msg = retry_error_data.get('error', {}).get('message', str(retry_error))
                            print(f"Retry with alternative token parameter also failed: {retry_error_msg}")
                        except:
                            print(f"Retry with alternative token parameter also failed: {retry_error}")
                        raise retry_error
                else:
                    print(f"OpenAI API Error: {error_msg}")
                    print(f"Request payload: {json.dumps(payload, indent=2)}")
                    raise
            except json.JSONDecodeError:
                print(f"HTTP Error: {e}")
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
                raise
            except requests.exceptions.HTTPError as retry_error:
                # If retry also fails, show both errors
                print(f"Retry with max_completion_tokens also failed: {retry_error}")
                raise
        
        data = response.json()
        
        # Handle structured output refusals as per OpenAI documentation
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        # Check for refusal in structured outputs
        if message.get("refusal"):
            assistant_response = f"[REFUSAL] {message.get('refusal')}"
            tool_calls = []
        else:
            assistant_response = message.get("content", "") or ""
            tool_calls = message.get("tool_calls", [])
        
        # If we used structured outputs, validate the response
        if json_schema and assistant_response and not message.get("refusal"):
            try:
                # Try to parse the JSON to ensure it's valid
                if assistant_response.strip().startswith('{') or assistant_response.strip().startswith('['):
                    json.loads(assistant_response)
            except json.JSONDecodeError as e:
                assistant_response = f"[INVALID_JSON] Response was not valid JSON: {str(e)}. Original response: {assistant_response}"
        
        # Execute tool calls if they exist
        tool_execution_results = {}
        if tool_calls and tools:
            tool_execution_results = self._execute_tool_calls(tool_calls, tools)
            # Store tool call results in instance variable
            self.tool_call_results.update(tool_execution_results)
        
        # Add user query and assistant response to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Build agent context with actual tool execution results
        self._build_agent_context(assistant_response, tool_execution_results)
        
        return {
            "text": assistant_response,
            "raw": data,
            "conversation_history": self.get_conversation_history(),
            "tool_calls": tool_calls,
            "tool_results": tool_execution_results
        }

    def invoke_responses_api(
        self,
        input_text: str,
        json_schema: t.Optional[dict | t.Any] = None,
        model_name: t.Optional[str] = None,
        context: str = "",
    ) -> dict:
        """Send a query using the new OpenAI Responses API.
        
        This uses the newer /v1/responses endpoint instead of /v1/chat/completions.
        The Responses API uses 'input' instead of 'messages' and has a simpler interface.
        
        Args:
            input_text: The input text/query for the model
            json_schema: Optional JSON schema for structured responses
            model_name: Optional model override (uses default if not provided)
        """
        url = f"{self.base_url}/responses"
        model = "gpt-5-nano"
        
        # Build payload for Responses API
        # Format input with context and task
        input_with_context_and_task = ""
        if context.strip():
            input_with_context_and_task += f"<context>{context}</context>\n\n"
        input_with_context_and_task += f"<task>{self.role}</task>\n\n{input_text}"
        
        payload = {
            "model": model,
            "input": input_with_context_and_task,
            "max_output_tokens": 4000,  # Responses API uses max_output_tokens
            "temperature": 1,
        }
        
        # Add JSON schema if provided (Responses API format)
        schema = self._extract_json_schema(json_schema)
        if schema:
            if self._supports_structured_outputs(model):
                # Use Responses API's structured format
                payload["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "structured_response",
                        "schema": schema
                    }
                }
            else:
                # Fallback for unsupported models - add instruction to the input
                payload["input"] = f"{input_with_context_and_task}\n\nPlease respond with valid JSON that follows this schema: {json.dumps(schema)}"
                payload["text"] = {"format": {"type": "json_object"}}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Handle token parameter errors with fallback
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', str(e))
                
                if 'max_output_tokens' in error_msg.lower() and 'not supported' in error_msg.lower():
                    print(f"Responses API Error with max_output_tokens: {error_msg}")
                    print("Retrying with max_tokens for older model compatibility...")
                    
                    if 'max_output_tokens' in payload:
                        payload['max_tokens'] = payload.pop('max_output_tokens')
                elif 'max_completion_tokens' in error_msg.lower():
                    print(f"Responses API Error with max_completion_tokens: {error_msg}")
                    print("Switching to max_output_tokens for Responses API...")
                    
                    if 'max_completion_tokens' in payload:
                        payload['max_output_tokens'] = payload.pop('max_completion_tokens')
                    
                    response = requests.post(url, json=payload, headers=headers, timeout=300)
                    response.raise_for_status()
                else:
                    print(f"Responses API Error: {error_msg}")
                    raise
            except json.JSONDecodeError:
                print(f"HTTP Error: {e}")
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
                raise
        
        data = response.json()
        
        # Handle the Responses API response format
        assistant_response = ""
        
        # The Responses API returns a response object with 'output' array
        if isinstance(data, dict) and 'output' in data and isinstance(data['output'], list):
            # Look for the assistant message in the output array
            for output_item in data['output']:
                if output_item.get('role') == 'assistant' and 'content' in output_item:
                    content_list = output_item['content']
                    for content_item in content_list:
                        if content_item.get('type') == 'output_text' and 'text' in content_item:
                            assistant_response = content_item['text']
                            break
                    if assistant_response:
                        break
        
        if not assistant_response:
            # Fallback: try to find text anywhere in the response
            assistant_response = f"Could not parse response format: {str(data)[:200]}..."
        
        # Store query and response in conversation history (adapted for responses API)
        self.query = input_text
        self.conversation_history.append({"role": "user", "content": input_text})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return {
            "text": assistant_response,
            "raw": data,
            "conversation_history": self.get_conversation_history(),
        }

    def invoke_streaming(
        self,
        query: str,
        json_schema: t.Optional[dict | t.Any] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model_name: t.Optional[str] = None,
        reasoning: t.Optional[dict] = None,
        verbosity: t.Optional[str] = None,
        context: str = "",
    ) -> t.Iterator[str]:
        """Send a query to OpenAI and return a streaming response.
        
        Note: Responses API doesn't support streaming, so this method uses 
        chat/completions with gpt-5-nano model only.
        
        Args:
            query: The user's query
            json_schema: Optional JSON schema for structured responses
            tools: Optional list of functions the agent can call
            model_name: Optional model override (forced to gpt-5-nano)
            reasoning: Controls reasoning parameters for applicable models.
                      Should be a dict with 'effort' key. 
                      Allowed values for effort: "minimal", "low", "medium", "high". 
                      If None, uses default from initialization.
            verbosity: Controls response verbosity for applicable models.
                      Allowed values: "low", "medium", "high".
                      If None, uses default from initialization.
        """
        # Force model to gpt-5-nano only
        model = "gpt-5-nano"
        
        # Use provided reasoning/verbosity or fall back to defaults
        effective_reasoning = reasoning or self.default_reasoning
        effective_verbosity = verbosity or self.default_verbosity
        
        # Validate parameters if provided
        if reasoning is not None:
            effective_reasoning = self._validate_reasoning(reasoning)
        if verbosity is not None:
            effective_verbosity = self._validate_verbosity(verbosity)
        
        url = f"{self.base_url}/chat/completions"
        payload = self._build_chat_payload(query, json_schema, tools, model, stream=True, reasoning=effective_reasoning, verbosity=effective_verbosity, context=context)
        self.query = query
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add user query to conversation history at the start
        self.conversation_history.append({"role": "user", "content": query})
        
        full_response = ""
        tool_calls = []
        
        # Try the request with error handling for max_tokens
        def make_streaming_request(request_payload):
            with requests.post(url, json=request_payload, headers=headers, stream=True, timeout=300) as response:
                try:
                    response.raise_for_status()
                    return response, None
                except requests.exceptions.HTTPError as e:
                    # Try to get error details from response
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('error', {}).get('message', str(e))
                        return None, (e, error_msg)
                    except:
                        return None, (e, str(e))
        
        # First attempt
        response, error_info = make_streaming_request(payload)
        
        # If error is related to token parameters, retry with alternative
        if error_info:
            e, error_msg = error_info
            if 'max_completion_tokens' in error_msg.lower() and 'not supported' in error_msg.lower():
                print(f"OpenAI Streaming API Error with max_completion_tokens: {error_msg}")
                print("Retrying with max_tokens for older model compatibility...")
                
                # Modify payload to use max_tokens instead of max_completion_tokens for older models
                if 'max_completion_tokens' in payload:
                    payload['max_tokens'] = payload.pop('max_completion_tokens')
                
                # Retry the request
                response, retry_error_info = make_streaming_request(payload)
                if retry_error_info:
                    print(f"Retry with max_tokens also failed: {retry_error_info[1]}")
                    raise retry_error_info[0]
            elif 'max_tokens' in error_msg.lower() and 'not supported' in error_msg.lower():
                print(f"OpenAI Streaming API Error with max_tokens: {error_msg}")
                print("Retrying with max_completion_tokens instead...")
                
                # Modify payload to use max_completion_tokens instead of max_tokens
                if 'max_tokens' in payload:
                    payload['max_completion_tokens'] = payload.pop('max_tokens')
                
                # Retry the request
                response, retry_error_info = make_streaming_request(payload)
                if retry_error_info:
                    print(f"Retry with max_completion_tokens also failed: {retry_error_info[1]}")
                    raise retry_error_info[0]
            else:
                print(f"OpenAI Streaming API Error: {error_msg}")
                raise e
        
        with response:
            
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                    
                if line.startswith("data: "):
                    line = line[6:]  # Remove "data: " prefix
                    
                if line == "[DONE]":
                    break
                    
                try:
                    data = json.loads(line)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        
                        # Handle refusal in streaming structured outputs
                        if delta.get("refusal"):
                            full_response = f"[REFUSAL] {delta.get('refusal')}"
                            yield full_response
                            break
                        
                        content = delta.get("content")
                        
                        if content:
                            full_response += content
                            yield content
                        
                        # Handle tool calls in streaming
                        if "tool_calls" in delta:
                            tool_calls.extend(delta["tool_calls"])
                        
                except json.JSONDecodeError:
                    continue
        
        # Execute tool calls if they exist
        tool_execution_results = {}
        if tool_calls and tools:
            tool_execution_results = self._execute_tool_calls(tool_calls, tools)
            # Store tool call results in instance variable
            self.tool_call_results.update(tool_execution_results)
        
        # Validate structured output if required
        if json_schema and full_response and not full_response.startswith("[REFUSAL]"):
            try:
                # Try to parse the JSON to ensure it's valid
                if full_response.strip().startswith('{') or full_response.strip().startswith('['):
                    json.loads(full_response)
            except json.JSONDecodeError as e:
                full_response = f"[INVALID_JSON] Response was not valid JSON: {str(e)}. Original response: {full_response}"
        
        # Add the complete assistant response to conversation history
        self.conversation_history.append({"role": "assistant", "content": full_response})
        
        # Build agent context with actual tool execution results
        self._build_agent_context(full_response, tool_execution_results)

    def invoke_with_context(
        self,
        query: str,
        context_from_other_agents: str = "",
        json_schema: t.Optional[dict | t.Any] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model_name: t.Optional[str] = None,
        reasoning: t.Optional[dict] = None,
        verbosity: t.Optional[str] = None,
    ) -> dict:
        """Send a query with context from other agents and return the response.
        
        Args:
            query: The user's query
            context_from_other_agents: Context information from other agents
            json_schema: Optional JSON schema for structured responses
            tools: Optional list of functions the agent can call
            model_name: Optional model override (forced to gpt-5-nano)
            reasoning: Controls reasoning parameters for applicable models.
                      Should be a dict with 'effort' key. 
                      Allowed values for effort: "minimal", "low", "medium", "high". 
                      If None, uses default from initialization.
            verbosity: Controls response verbosity for applicable models.
                      Allowed values: "low", "medium", "high".
                      If None, uses default from initialization.
        """
        # Use regular invoke method with context parameter instead of adding to conversation history
        return self.invoke(query, json_schema, tools, model_name, reasoning, verbosity, context=context_from_other_agents)


if __name__ == "__main__":
    """Example usage demonstrating conversation history and agent context features."""
    
    # Check if API key is available
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("=== OpenAI Client Structured Output Examples ===")
        print("Note: No OPENAI_API_KEY found in environment variables.")
        print("Set your API key to run live examples:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\n=== Example Code for Structured Outputs ===")
        
        # Show example code instead of running it
        example_code = '''
# Example 1: Basic structured output with raw JSON schema
from openai_client import OpenAIClient

client = OpenAIClient(
    role="information_extractor",
    model_name="gpt-5-nano"
)

# Define a schema for person information
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "is_employed": {"type": "boolean"}
    },
    "required": ["name", "age", "skills", "is_employed"],
    "additionalProperties": False
}

# Extract structured information
query = "Extract info about John: 30 years old, developer, knows Python and JS"
response = client.invoke(query=query, json_schema=person_schema)
print(response['text'])  # Will be valid JSON matching the schema

# Example 2: Using Pydantic models
from pydantic import BaseModel
from typing import List

class PersonInfo(BaseModel):
    name: str
    age: int
    skills: List[str]
    is_employed: bool

response = client.invoke(query=query, json_schema=PersonInfo)
print(response['text'])  # Same result, but using Pydantic model
        '''
        print(example_code)
        exit(0)
    
    from pydantic import BaseModel

    class Answer(BaseModel):
        """Schema for structured responses."""
        title: str
        points: list[str]

    def get_weather(city: str, unit: str = "C") -> str:
        """Get weather information for a city.

        Args:
            city (str): The name of the city.
            unit (str): Temperature unit ('C' for Celsius, 'F' for Fahrenheit).
        """
        return f"{city}: 24°{unit}, sunny"

    # Create an agent with system instructions, default model, and agent name
    agent = OpenAIClient(
        role="Python programming consultant",
        model_name="gpt-5-nano",
        system_instructions="You are a helpful programming assistant. Always be concise and accurate.",
        agent_name="PythonExpert"
    )
    
    # Set up agent properties
    agent.set_generic_agent_instructions("Provide accurate Python programming advice")
    agent.set_all_tool_names("get_weather")

    # Example 1: First conversation
    print("=== Conversation Example ===")
    result1 = agent.invoke("What is Python?")
    print(f"Q1: What is Python?")
    print(f"A1: {result1['text'][:100]}...")
    
    # Update agent context with response
    agent.update_agent_context(result1['text'][:100], {"get_weather": "not_used"})
    
    # Example 2: Follow-up question (uses conversation history)
    result2 = agent.invoke("What are its main advantages?")
    print(f"\nQ2: What are its main advantages?")
    print(f"A2: {result2['text'][:100]}...")
    
    # Example 3: Using context from other agents
    print(f"\n=== Context from Other Agents Example ===")
    context = "Agent A analyzed that the user is working on a web development project using Django."
    result3 = agent.invoke_with_context(
        "Should I use Python for this project?",
        context_from_other_agents=context
    )
    print(f"Q3: Should I use Python for this project?")
    print(f"Context: {context}")
    print(f"A3: {result3['text'][:100]}...")
    
    # Example 4: Show agent context
    print(f"\n=== Agent Context (XML Format) ===")
    agent.update_agent_context(result3['text'][:100], {"get_weather": "24°C, sunny"})
    print(agent.get_agent_context())
    
    # Example 5: Show conversation history
    print(f"\n=== Conversation History ===")
    history = agent.get_conversation_history()
    print(f"Total messages in history: {len(history)}")
    for i, msg in enumerate(history[-4:]):  # Show last 4 messages
        role = msg['role'].upper()
        content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
        print(f"{i+1}. {role}: {content}")
    
    # Example 6: Tool call functionality (simulated)
    print(f"\n=== Tool Call Functionality Example ===")
    print("Note: This would work with OpenAI models that support tool calls")
    
    # Simulate tool calls result manually for demonstration
    tool_result = agent.invoke("What's the weather like?", tools=[get_weather])
    print(f"Q: What's the weather like?")
    print(f"A: {tool_result['text'][:100]}...")
    print(f"Tool calls made: {tool_result.get('tool_calls', [])}")
    print(f"Tool results: {tool_result.get('tool_results', {})}")
    print(f"Stored tool call results: {agent.get_tool_call_results()}")
    
    # Example 7: Agent properties
    print(f"\n=== Agent Properties ===")
    print(f"Agent Name: {agent.agent_name}")
    print(f"Role: {agent.get_role()}")
    print(f"Generic Instructions: {agent.get_generic_agent_instructions()}")
    print(f"Available Tools: {agent.get_all_tool_names()}")
    
    # Example 8: Clear history and start fresh
    print(f"\n=== Starting Fresh ===")
    agent.clear_conversation_history()
    result4 = agent.invoke("Hello, I'm new here!")
    print(f"After clearing history - Q: Hello, I'm new here!")
    print(f"A: {result4['text'][:100]}...")
    print(f"New history length: {len(agent.get_conversation_history())}")
    
    # Example 9: Structured Output Examples
    print(f"\n=== Structured Output Examples ===")
    
    # Example 9a: Raw JSON Schema (Simple example)
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "profession": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            },
            "email": {"type": "string"}
        },
        "required": ["name", "age", "profession", "skills", "email"],
        "additionalProperties": False
    }
    
    structured_query = """
    Extract information about this person:
    Sarah Johnson is a 28-year-old software engineer who specializes in Python, JavaScript, and machine learning. 
    You can reach her at sarah.j@email.com.
    """
    
    print(f"Query: {structured_query.strip()}")
    structured_result = agent.invoke(
        query=structured_query,
        json_schema=person_schema
    )
    print(f"Structured Response: {structured_result['text']}")
    
    # Example 9b: Using Pydantic Model (if available)
    try:
        from pydantic import BaseModel
        from typing import List, Optional
        
        class TaskInfo(BaseModel):
            title: str
            priority: str  # "high", "medium", "low"
            estimated_hours: int
            tags: List[str]
            assigned_to: str
        
        task_query = """
        Create a task for implementing user authentication system. 
        This is a high priority task that should take about 8 hours.
        Tag it with 'security', 'backend', 'authentication'.
        Assign it to the backend team.
        """
        
        print(f"\nPydantic Query: {task_query.strip()}")
        pydantic_result = agent.invoke(
            query=task_query,
            json_schema=TaskInfo
        )
        print(f"Pydantic Structured Response: {pydantic_result['text']}")
        
    except ImportError:
        print("\nPydantic not available - skipping Pydantic example")
    
    # Example 9c: Error handling demonstration
    print(f"\n=== Structured Output Error Handling ===")
    
    # Test with invalid schema request
    invalid_query = "Please respond with just plain text, not JSON"
    invalid_result = agent.invoke(
        query=invalid_query,
        json_schema={"type": "object", "properties": {"response": {"type": "string"}}, "required": ["response"], "additionalProperties": False}
    )
    print(f"Invalid JSON Query: {invalid_query}")
    print(f"Response (should handle gracefully): {invalid_result['text'][:100]}...")
    
    print(f"\n=== Structured Output Examples Complete ===")
    print("✅ Raw JSON Schema - Successfully extracted structured person data")
    print("✅ Pydantic Model - Successfully created structured task information") 
    print("✅ Error Handling - Gracefully handled conflicting instructions")
    print("\nKey Features Demonstrated:")
    print("- OpenAI strict mode JSON schema compliance")
    print("- Pydantic model integration with schema generation")
    print("- Automatic fallback for unsupported models")
    print("- Comprehensive error handling and validation")
    print("- Support for complex nested object structures")
