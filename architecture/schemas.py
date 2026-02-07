"""Pydantic schemas for the orchestrator-executor system."""

from enum import Enum
from pydantic import BaseModel, Field


class Domain(str, Enum):
    """Tool domains matching Toolsmith categories."""

    MATH = "math"
    TEXT = "text"
    FILE = "file"
    WEB = "web"
    VISUALIZATION = "visualization"
    DATA = "data"
    SYSTEM = "system"
    CONVERSION = "conversion"
    SEARCH = "search"


class SubTask(BaseModel):
    """A granular subtask from the main query."""

    id: str = Field(..., description="Unique subtask ID like 'st_1', 'st_2'")
    description: str = Field(..., description="What this subtask needs to do")
    domain: Domain = Field(..., description="Primary domain for this subtask")
    depends_on: list[str] = Field(
        default_factory=list, description="IDs of subtasks this depends on"
    )
    input_from: str | None = Field(
        None, description="Which subtask's output to use as input"
    )


class DecompositionResult(BaseModel):
    """LLM output after decomposition."""

    original_query: str
    subtasks: list[SubTask]


class ToolMatch(BaseModel):
    """A tool matched to a subtask."""

    subtask_id: str
    tool_name: str
    tool_file: str
    matched: bool
    confidence: float


class ExecutionStep(BaseModel):
    """Single step in execution plan."""

    step_number: int
    subtask_id: str
    description: str
    tool_name: str
    tool_args_template: dict = Field(default_factory=dict)
    expected_output: str
    depends_on: list[int] = Field(default_factory=list)
    input_from: int | None = Field(
        None, description="Step number to get input data from"
    )
    result: str | None = None
    status: str = "pending"


class ExecutionPlan(BaseModel):
    """Detailed plan passed to executor agent."""

    original_query: str
    steps: list[ExecutionStep]

    def get_ready_steps(self) -> list[ExecutionStep]:
        """Return steps whose dependencies are all completed."""
        completed = {s.step_number for s in self.steps if s.status == "completed"}
        return [
            s
            for s in self.steps
            if s.status == "pending" and all(d in completed for d in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(s.status == "completed" for s in self.steps)

    def has_failed(self) -> bool:
        return any(s.status == "failed" for s in self.steps)
