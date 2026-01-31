from abc import ABC, abstractmethod
from typing import List, Any
import os
import json
from litellm import completion
from .tools import Tool
from .schemas import Message

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class LLMClient(ABC):
    @abstractmethod
    def start_chat(self, history: List[Message] = []) -> Any:
        pass
    
    @abstractmethod
    def send_message(self, chat: Any, message: str) -> str:
        pass


class LiteLLMClient(LLMClient):
    def __init__(self, model_name: str, tools: List[Tool] = [], api_base: str = None):
        self.model_name = model_name
        self.tools = tools
        self.api_base = None
        self.extra_headers = None
        
        if "ollama" in model_name.lower():
            self.api_base = api_base or OLLAMA_URL
            print(f"[LiteLLM] Using Ollama at: {self.api_base}")
        
        self.system_instruction = """You are an expert DevOps engineer and Python developer. 
Your goal is to autonomously solve infrastructure and coding tasks.
You can write files and execute commands. 
If a command fails, analyze the error output, fix the code or configuration, and try again.
Always verify your work by running the code you wrote.
When using Docker, ensure you write a Dockerfile and docker-compose.yml (if needed), then build and run it."""
        
        self.history = [{"role": "system", "content": self.system_instruction}]

    def _get_tools_schema(self):
        """Convert Tools to OpenAI function schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema.model_json_schema()
                }
            }
            for t in self.tools
        ]

    def start_chat(self, history: List[Message] = []) -> Any:
        self.history = [{"role": "system", "content": self.system_instruction}]
        if history:
            self.history.extend([{"role": msg.role, "content": msg.content} for msg in history])
        return self

    def send_message(self, chat: Any, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        
        tools_map = {t.name: t for t in self.tools}
        formatted_tools = self._get_tools_schema()
        
        MAX_TURNS = 10
        for turn in range(MAX_TURNS):
            print(f"\n[LiteLLM] Turn {turn+1}/{MAX_TURNS} - Model: {self.model_name}")
            
            try:
                completion_kwargs = {
                    "model": self.model_name,
                    "messages": self.history,
                    "tools": formatted_tools,
                    "tool_choice": "auto" if formatted_tools else None,
                    "temperature": 0.0
                }
                
                if self.api_base:
                    completion_kwargs["api_base"] = self.api_base
                if self.extra_headers:
                    completion_kwargs["extra_headers"] = self.extra_headers
                
                response = completion(**completion_kwargs)
            except Exception as e:
                return f"LiteLLM Error: {e}"

            response_message = response.choices[0].message
            
            if response_message.tool_calls:
                self.history.append(response_message)
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"  [Tool Call] {function_name} args={function_args}")
                    
                    if function_name in tools_map:
                        try:
                            tool_result = tools_map[function_name].run(**function_args)
                        except Exception as e:
                            tool_result = f"Error executing {function_name}: {e}"
                    else:
                        tool_result = f"Error: Tool {function_name} not found"
                    
                    print(f"  [Result] {str(tool_result)[:100]}...")

                    self.history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_result)
                    })
                
                continue
            
            self.history.append(response_message)
            return response_message.content

        return "Error: Max tool turns reached."
