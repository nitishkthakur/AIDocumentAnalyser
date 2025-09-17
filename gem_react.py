import inspect
import json
import typing as t
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression.
    :param expression: The mathematical expression to evaluate.
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return str(e)

def get_current_time() -> str:
    """
    Returns the current time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search(query: str) -> str:
    """
    Performs a simple search.
    :param query: The search query.
    """
    return f"Search results for '{query}': This is a placeholder result."


class GemReact:
    """Groq chat client with ReAct agent logic for tool calling."""
    
    TYPE_MAPPING = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}

    def __init__(self, api_key: str = None, model_name: str = "mixtral-8x7b-32768", 
                 system_instructions: str = "", base_url: str = "https://api.groq.com/openai/v1"):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model_name
        self.base_url = base_url
        self.conversation_history: list[dict] = []
        self.system_instructions = system_instructions
        
        self.default_reasoning = {
            "temperature": 0.7,
            "max_tokens": 1024,
        }

    def _extract_function_info(self, func: t.Callable) -> dict:
        """Extract function information for Groq tool definition."""
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or f"Function {func.__name__}"
        
        parameters = {"type": "object", "properties": {}, "required": []}
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": self.TYPE_MAPPING.get(param.annotation, "string")}
            
            if docstring:
                lines = docstring.split('\n')
                for line in lines:
                    if param_name in line and ':' in line:
                        description = line.split(':', 1)[1].strip()
                        if description:
                            param_info["description"] = description
                        break
            
            parameters["properties"][param_name] = param_info
            
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
        
        return [self._extract_function_info(tool) for tool in tools if callable(tool)]

    def invoke(self, query: str, tools: t.Optional[t.Iterable[t.Callable]] = None):
        """Send query to Groq and handle tool calls in a ReAct loop."""
        
        messages = []
        if self.system_instructions:
            messages.append({"role": "system", "content": self.system_instructions})
        
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": query})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            **self.default_reasoning
        }

        tool_schemas = self._build_tools(tools)
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"

        response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        self.conversation_history.append({"role": "user", "content": query})
        
        response_message = data["choices"][0]["message"]
        
        if response_message.get("tool_calls"):
            self.conversation_history.append(response_message)
            
            tool_calls = response_message["tool_calls"]
            tool_map = {func.__name__: func for func in tools}

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_to_call = tool_map[function_name]
                function_args = json.loads(tool_call["function"]["arguments"])
                function_response = function_to_call(**function_args)

                self.conversation_history.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response),
                    }
                )
            
            second_payload = {
                "model": self.model,
                "messages": self.conversation_history,
                **self.default_reasoning
            }
            
            second_response = requests.post(f"{self.base_url}/chat/completions", json=second_payload, headers=headers)
            second_response.raise_for_status()
            second_data = second_response.json()
            
            final_response = second_data["choices"][0]["message"]["content"]
            self.conversation_history.append({"role": "assistant", "content": final_response})
            return final_response
        else:
            final_response = response_message["content"]
            self.conversation_history.append({"role": "assistant", "content": final_response})
            return final_response

if __name__ == '__main__':
    agent = GemReact(model_name="mixtral-8x7b-32768")
    tools = [calculator, get_current_time, search]

    print("Agent: Hello! How can I help you today?")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent.invoke(query, tools=tools)
        print(f"Agent: {response}")