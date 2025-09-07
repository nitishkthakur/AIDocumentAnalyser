import ollama
import json
import inspect
import re
from typing import Optional, Dict, Any, Type, Union, List, Callable
from pydantic import BaseModel


class OllamaAgent:
    """
    An Ollama agent class that initializes with llama3.2:3b model and executes tasks based on context.
    """
    
    def __init__(self, model: str = "llama3.2:3b", keepalive: str = "15m", num_ctx: int = 4096):
        """
        Initialize the Ollama agent with specified model and keepalive settings.
        
        Args:
            model (str): The Ollama model to use. Default is "llama3.2:3b"
            keepalive (str): How long to keep the model alive. Default is "15m"
            num_ctx (int): Context window size. Default is 4096
        """
        self.model = model
        self.keepalive = keepalive
        self.num_ctx = num_ctx
        self.client = ollama.Client()
        
        # Default system instruction
        self.system_instruction = """You are an intelligent agent that is part of a group of specialized agents working together to solve complex tasks. Your role is to execute the specific task that will be provided to you within <task></task> tags in the user instructions.

IMPORTANT GUIDELINES:
1. You must refer to and derive all necessary knowledge from the information provided within <context></context> tags in the user instructions.
2. The context provided is sufficient and contains all the knowledge you need to complete your task.
3. The context may include:
   - Knowledge from online sources
   - Chunks of information extracted from web pages
   - Content from PDF documents
   - Conversation chains between other agents and users
   - Structured data or analysis results
4. You must execute the task exactly as specified, using only the information from the provided context.
5. Do not make assumptions or use knowledge outside of what's provided in the context.
6. If the context seems insufficient for the task, clearly state what additional information you would need.

Your response should be focused, accurate, and directly address the task using the contextual information provided."""
        
        # Test connection and ensure model is available
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize and verify the model is available.
        """
        try:
            # Pull the model if it doesn't exist
            self.client.pull(self.model)
            print(f"✓ Model {self.model} is ready")
        except Exception as e:
            print(f"Error initializing model {self.model}: {e}")
            raise
    
    def _extract_tool_definitions(self, tools: List[Callable]) -> List[Dict[str, Any]]:
        """
        Extract tool definitions from a list of functions for Ollama.
        
        Args:
            tools (List[Callable]): List of functions to extract tool definitions from
            
        Returns:
            List[Dict[str, Any]]: List of tool definitions in Ollama format
        """
        tool_definitions = []
        
        for tool in tools:
            try:
                # Get function signature
                sig = inspect.signature(tool)
                
                # Get docstring
                docstring = inspect.getdoc(tool) or ""
                
                # Parse docstring to extract description and args
                lines = docstring.strip().split('\n')
                description = ""
                args_section = False
                args_descriptions = {}
                
                for line in lines:
                    line = line.strip()
                    if line.lower().startswith('args:'):
                        args_section = True
                        continue
                    elif args_section and line:
                        # Parse argument description (format: "arg_name: description")
                        if ':' in line:
                            arg_match = re.match(r'(\w+):\s*(.+)', line)
                            if arg_match:
                                arg_name, arg_desc = arg_match.groups()
                                args_descriptions[arg_name] = arg_desc
                    elif not args_section and line:
                        # This is part of the main description
                        if description:
                            description += " " + line
                        else:
                            description = line
                
                # Build properties from function signature
                properties = {}
                required = []
                
                for param_name, param in sig.parameters.items():
                    param_info = {
                        "type": "string",  # Default type
                        "description": args_descriptions.get(param_name, f"Parameter {param_name}")
                    }
                    
                    # Infer type from annotation if available
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation == int:
                            param_info["type"] = "integer"
                        elif param.annotation == float:
                            param_info["type"] = "number"
                        elif param.annotation == bool:
                            param_info["type"] = "boolean"
                        elif param.annotation == list:
                            param_info["type"] = "array"
                        elif param.annotation == dict:
                            param_info["type"] = "object"
                    
                    properties[param_name] = param_info
                    
                    # Check if parameter is required (no default value)
                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)
                
                # Create tool definition in Ollama format
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.__name__,
                        "description": description or f"Function {tool.__name__}",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    }
                }
                
                tool_definitions.append(tool_def)
                
            except Exception as e:
                print(f"Error extracting tool definition for {tool.__name__}: {e}")
                continue
        
        return tool_definitions
    
    def invoke(self, user_message: str, context: str = "", task: str = "", 
               custom_system: Optional[str] = None, 
               response_schema: Optional[Type[BaseModel]] = None,
               tools: Optional[List[Callable]] = None) -> Union[str, BaseModel]:
        """
        Generate a response using the Ollama model.
        
        Args:
            user_message (str): The user's message
            context (str): Additional context information
            task (str): Specific task to execute
            custom_system (str): Optional custom system instruction
            response_schema (Type[BaseModel]): Optional Pydantic model for structured output
            tools (List[Callable]): Optional list of tools/functions the model can use
            
        Returns:
            Union[str, BaseModel]: The model's response as string or structured Pydantic object
        """
        try:
            # Use custom system instruction if provided, otherwise use default
            system_msg = custom_system if custom_system else self.system_instruction
            
            # Format the user message with context and task if provided
            formatted_message = user_message
            if context:
                formatted_message += f"\n\n<context>\n{context}\n</context>"
            if task:
                formatted_message += f"\n\n<task>\n{task}\n</task>"
            
            # If response_schema is provided, add structured output instructions
            if response_schema:
                schema_instructions = f"\n\nIMPORTANT: You must respond with valid JSON that conforms to this exact schema:\n{json.dumps(response_schema.model_json_schema(), indent=2)}\n\nDo not include any text outside the JSON response."
                formatted_message += schema_instructions
            
            # Prepare options
            options = {
                "keep_alive": self.keepalive,
                "num_ctx": self.num_ctx
            }
            
            # Prepare chat parameters
            chat_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": formatted_message}
                ],
                "options": options
            }
            
            # Add tools if provided
            if tools:
                tool_definitions = self._extract_tool_definitions(tools)
                if tool_definitions:
                    chat_params["tools"] = tool_definitions
            
            # Add format for structured output if needed
            if response_schema:
                chat_params["format"] = response_schema.model_json_schema()
            
            # Make the API call
            response = self.client.chat(**chat_params)
            
            # Handle structured output parsing
            if response_schema:
                try:
                    response_content = response['message']['content']
                    # Clean the response in case there's extra text
                    response_content = response_content.strip()
                    if response_content.startswith('```json'):
                        response_content = response_content[7:]
                    if response_content.endswith('```'):
                        response_content = response_content[:-3]
                    response_content = response_content.strip()
                    
                    parsed_json = json.loads(response_content)
                    return response_schema(**parsed_json)
                except (json.JSONDecodeError, ValueError) as e:
                    # If parsing fails, try to extract JSON from the response
                    try:
                        json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
                        if json_match:
                            parsed_json = json.loads(json_match.group())
                            return response_schema(**parsed_json)
                        else:
                            return f"Error parsing structured response: {e}. Raw response: {response['message']['content']}"
                    except Exception as inner_e:
                        return f"Error parsing structured response: {inner_e}. Raw response: {response['message']['content']}"
            
            # Handle tool calls if present in response
            if 'tool_calls' in response['message'] and response['message']['tool_calls']:
                tool_results = []
                for tool_call in response['message']['tool_calls']:
                    tool_name = tool_call['function']['name']
                    tool_args = tool_call['function']['arguments']
                    
                    # Find the corresponding tool function
                    tool_func = None
                    for tool in tools:
                        if tool.__name__ == tool_name:
                            tool_func = tool
                            break
                    
                    if tool_func:
                        try:
                            # Execute the tool with the provided arguments
                            if isinstance(tool_args, str):
                                tool_args = json.loads(tool_args)
                            result = tool_func(**tool_args)
                            tool_results.append(f"Tool {tool_name} result: {result}")
                        except Exception as e:
                            tool_results.append(f"Error executing tool {tool_name}: {e}")
                    else:
                        tool_results.append(f"Tool {tool_name} not found")
                
                # If tools were called, include their results in the response
                base_response = response['message']['content']
                if tool_results:
                    return f"{base_response}\n\nTool Execution Results:\n" + "\n".join(tool_results)
                
            return response['message']['content']
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def execute_task_with_context(self, task: str, context: str, 
                                additional_instructions: str = "",
                                response_schema: Optional[Type[BaseModel]] = None,
                                tools: Optional[List[Callable]] = None) -> Union[str, BaseModel]:
        """
        Execute a specific task using provided context.
        
        Args:
            task (str): The task to execute
            context (str): The context information
            additional_instructions (str): Any additional instructions
            response_schema (Type[BaseModel]): Optional Pydantic model for structured output
            tools (List[Callable]): Optional list of tools/functions the model can use
            
        Returns:
            Union[str, BaseModel]: The execution result
        """
        user_message = additional_instructions if additional_instructions else "Please execute the given task using the provided context."
        
        return self.invoke(
            user_message=user_message,
            context=context,
            task=task,
            response_schema=response_schema,
            tools=tools
        )
    
    def chat(self, message: str) -> str:
        """
        Simple chat interface without specific task/context formatting.
        
        Args:
            message (str): The message to send
            
        Returns:
            str: The model's response
        """
        return self.invoke(user_message=message)
    
    def set_custom_system_instruction(self, instruction: str):
        """
        Set a custom system instruction.
        
        Args:
            instruction (str): The new system instruction
        """
        self.system_instruction = instruction
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict: Model information
        """
        try:
            models = self.client.list()
            for model in models['models']:
                if model['name'] == self.model:
                    return model
            return {"error": f"Model {self.model} not found"}
        except Exception as e:
            return {"error": f"Error getting model info: {e}"}


# Example usage
if __name__ == "__main__":
    from pydantic import BaseModel, Field
    from typing import List
    
    # Define a structured response schema
    class ClimateStats(BaseModel):
        temperature_rise: float = Field(description="Global temperature rise in Celsius")
        co2_levels: int = Field(description="CO2 levels in ppm")
        renewable_percentage: float = Field(description="Renewable energy percentage")
        summary: str = Field(description="Brief summary of the statistics")
    
    class TaskSummary(BaseModel):
        key_points: List[str] = Field(description="List of key points from the context")
        conclusion: str = Field(description="Overall conclusion")
        confidence_score: float = Field(description="Confidence score between 0 and 1")
    
    # Initialize the agent
    agent = OllamaAgent()
    
    # Example task execution with structured output
    sample_context = """
    The user has uploaded a PDF document about climate change. The document mentions:
    - Global temperatures have risen by 1.1°C since pre-industrial times
    - CO2 levels are at 421 ppm as of 2023
    - Renewable energy accounts for 30% of global electricity generation
    """
    
    sample_task = "Extract the climate statistics from the document"
    
    # Regular text response
    result = agent.execute_task_with_context(
        task=sample_task,
        context=sample_context
    )
    
    print("Regular Response:")
    print(result)
    print("\n" + "="*50 + "\n")
    
    # Structured response using Pydantic schema
    structured_result = agent.execute_task_with_context(
        task=sample_task,
        context=sample_context,
        response_schema=ClimateStats
    )
    
    print("Structured Response:")
    if isinstance(structured_result, ClimateStats):
        print(f"Temperature Rise: {structured_result.temperature_rise}°C")
        print(f"CO2 Levels: {structured_result.co2_levels} ppm")
        print(f"Renewable Energy: {structured_result.renewable_percentage}%")
        print(f"Summary: {structured_result.summary}")
    else:
        print(structured_result)  # Error message
    
    print("\n" + "="*50 + "\n")
    
    # Example with tools
    def calculate_temperature_change(start_temp: float, end_temp: float) -> float:
        """Calculate the temperature change between two values
        
        Args:
        start_temp: The starting temperature in Celsius
        end_temp: The ending temperature in Celsius
        """
        return end_temp - start_temp
    
    def get_emission_category(co2_level: int) -> str:
        """Categorize CO2 emission levels
        
        Args:
        co2_level: CO2 concentration in ppm
        """
        if co2_level < 350:
            return "Safe"
        elif co2_level < 400:
            return "Caution"
        else:
            return "Dangerous"
    
    def calculate_renewable_impact(renewable_percentage: float, total_capacity: float) -> float:
        """Calculate renewable energy impact
        
        Args:
        renewable_percentage: Percentage of renewable energy
        total_capacity: Total energy capacity in MW
        """
        return (renewable_percentage / 100) * total_capacity
    
    # Example with tools
    tools_result = agent.invoke(
        user_message="Use the available tools to analyze the climate data and provide insights",
        context=sample_context,
        task="Analyze climate statistics using available calculation tools",
        tools=[calculate_temperature_change, get_emission_category, calculate_renewable_impact]
    )
    
    print("Response with Tools:")
    print(tools_result)
    
    print("\n" + "="*50 + "\n")
    
    # Another example with TaskSummary schema
    summary_task = "Provide a summary of the key information"
    summary_result = agent.invoke(
        user_message="Analyze the provided context and create a structured summary",
        context=sample_context,
        task=summary_task,
        response_schema=TaskSummary
    )
    
    print("Task Summary Response:")
    if isinstance(summary_result, TaskSummary):
        print("Key Points:")
        for point in summary_result.key_points:
            print(f"  - {point}")
        print(f"Conclusion: {summary_result.conclusion}")
        print(f"Confidence: {summary_result.confidence_score:.2f}")
    else:
        print(summary_result)  # Error message
