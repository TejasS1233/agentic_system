import os
import subprocess
from abc import ABC, abstractmethod
from typing import Type
from pydantic import BaseModel, Field
import sys

try:
    from architecture.toolsmith import Toolsmith
except ImportError:
    path = os.path.join(os.getcwd(), "architecture")
    if path not in sys.path:
        sys.path.append(path)
    try:
        from architecture.toolsmith import Toolsmith
    except Exception:
        Toolsmith = None


class Tool(ABC):
    name: str
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    def run(self, **kwargs) -> str:
        pass


# --- Tool Arguments Schemas ---


class WriteFileArgs(BaseModel):
    filepath: str = Field(
        ..., description="Path to the file to write (relative to workspace)"
    )
    content: str = Field(..., description="Content to write to the file")


class RunCommandArgs(BaseModel):
    command: str = Field(..., description="Shell command to execute")


class ReadFileArgs(BaseModel):
    filepath: str = Field(
        ..., description="Path to the file to read (relative to workspace)"
    )


class RequestNewToolArgs(BaseModel):
    description: str = Field(
        ...,
        description="Detailed description of the tool you need (e.g., 'A tool to calculate fibonacci').",
    )


# --- Tool Implementations ---


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write content to a file. Useful for creating scripts, Dockerfiles, etc."
    )
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
    description = (
        "Execute a shell command. Use this to run docker commands or python scripts."
    )
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
                text=True,
                timeout=120,  # 2 minute timeout
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\nExit Code: {result.returncode}"
            return output
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 120 seconds."
        except Exception as e:
            return f"Error executing command: {str(e)}"


class DockerRunCommandTool(Tool):
    """Run commands inside a Docker container with full Python environment."""

    name = "run_command"
    description = "Execute a shell command inside a Docker container. Has pip and python available."
    args_schema = RunCommandArgs

    def __init__(self, workspace_path: str, image: str = "python:3.11-slim"):
        self.workspace_path = workspace_path
        self.image = image
        self._container_id = None

    def _ensure_container(self) -> str:
        """Start a persistent container if not already running."""
        if self._container_id:
            # Check if container is still running
            check = subprocess.run(
                f"docker ps -q -f id={self._container_id}",
                shell=True,
                capture_output=True,
                text=True,
            )
            if check.stdout.strip():
                return self._container_id

        # Start a new container with workspace mounted
        result = subprocess.run(
            f"docker run -d -v {self.workspace_path}:/workspace -w /workspace {self.image} tail -f /dev/null",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start Docker container: {result.stderr}")

        self._container_id = result.stdout.strip()
        return self._container_id

    def run(self, command: str) -> str:
        try:
            container_id = self._ensure_container()

            # Execute command inside the container
            docker_cmd = f"docker exec {container_id} /bin/sh -c '{command}'"
            result = subprocess.run(
                docker_cmd, shell=True, capture_output=True, text=True, timeout=120
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\nExit Code: {result.returncode}"
            return output
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 120 seconds."
        except Exception as e:
            return f"Error executing command in Docker: {str(e)}"

    def cleanup(self):
        """Stop and remove the container."""
        if self._container_id:
            subprocess.run(
                f"docker stop {self._container_id}", shell=True, capture_output=True
            )
            subprocess.run(
                f"docker rm {self._container_id}", shell=True, capture_output=True
            )
            self._container_id = None


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


class RequestNewTool(Tool):
    name = "request_new_tool"
    description = "Use this when you CANNOT solve the problem with existing tools. The system will create a new Python tool for you."
    args_schema = RequestNewToolArgs

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self._toolsmith = None

    @property
    def toolsmith(self):
        if self._toolsmith is None and Toolsmith:
            self._toolsmith = Toolsmith()
        return self._toolsmith

    def run(self, description: str) -> str:
        if not self.toolsmith:
            return "Error: Toolsmith component not available in this environment."

        result = self.toolsmith.create_tool(description)
        return f"{result}\nIMPORTANT: The tool has been created. In the next turn, you can call it."
