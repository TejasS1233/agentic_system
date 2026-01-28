import os
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Any, Type
from pydantic import BaseModel, Field

class Tool(ABC):
    name: str
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    def run(self, **kwargs) -> str:
        pass

# --- Tool Arguments Schemas ---

class WriteFileArgs(BaseModel):
    filepath: str = Field(..., description="Path to the file to write (relative to workspace)")
    content: str = Field(..., description="Content to write to the file")

class RunCommandArgs(BaseModel):
    command: str = Field(..., description="Shell command to execute")

class ReadFileArgs(BaseModel):
    filepath: str = Field(..., description="Path to the file to read (relative to workspace)")

# --- Tool Implementations ---

class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file. Useful for creating scripts, Dockerfiles, etc."
    args_schema = WriteFileArgs

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path

    def run(self, filepath: str, content: str) -> str:
        full_path = os.path.join(self.workspace_path, filepath)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        try:
            with open(full_path, "w") as f:
                f.write(content)
            return f"Successfully wrote to {filepath}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

class RunCommandTool(Tool):
    name = "run_command"
    description = "Execute a shell command. Use this to run docker commands or python scripts."
    args_schema = RunCommandArgs
    
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path

    def run(self, command: str) -> str:
        try:
            # Using shell=True for flexibility, but be careful in prod
            result = subprocess.run(
                command, 
                cwd=self.workspace_path,
                shell=True, 
                capture_output=True, 
                text=True
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                 output += f"\nExit Code: {result.returncode}"
            return output
        except Exception as e:
            return f"Error executing command: {str(e)}"

class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file."
    args_schema = ReadFileArgs

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path

    def run(self, filepath: str) -> str:
        full_path = os.path.join(self.workspace_path, filepath)
        try:
            with open(full_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File {filepath} not found."
        except Exception as e:
            return f"Error reading file: {str(e)}"
