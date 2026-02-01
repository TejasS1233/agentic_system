from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


class ToolResult(BaseModel):
    tool_name: str
    output: str
    error: Optional[str] = None


class Message(BaseModel):
    role: str  # "user", "model", "tool"
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_result: Optional[ToolResult] = None


class AgentState(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    workspace_path: str
    max_steps: int = 10
    current_step: int = 0
