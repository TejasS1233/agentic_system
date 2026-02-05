import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.getcwd(), "architecture"))

from architecture.orchestrator import Orchestrator  # noqa: E402
from architecture.dispatcher import Dispatcher  # noqa: E402


def main():
    print("=== IASCIS Architecture Demo ===")

    orchestrator = Orchestrator(model_name="gemini/gemini-3-flash-preview")
    dispatcher = Dispatcher()

    user_task = "Analyze the sensitive_payroll.csv and calculate the sum of bonuses."
    print(f"\n[User] Task: {user_task}")

    zone = dispatcher.route(
        task_description=user_task, file_context=["sensitive_payroll.csv"]
    )

    print("\n[Orchestrator] Planning task...")
    plan = orchestrator.run(user_task)

    print(f"\n[System] Initializing Execution Agent for Zone: {zone.upper()}")

    workspace_path = os.path.join(os.getcwd(), "workspace")
    os.makedirs(workspace_path, exist_ok=True)

    try:
        from execution.tools import (
            WriteFileTool,
            RunCommandTool,
            ReadFileTool,
            RequestNewTool,
        )
        from execution.llm import LiteLLMClient
        from execution.core import Agent
    except ImportError:
        sys.path.append(os.getcwd())
        from execution.tools import (
            WriteFileTool,
            RunCommandTool,
            ReadFileTool,
            RequestNewTool,
        )
        from execution.llm import LiteLLMClient
        from execution.core import Agent

    # Base Tools
    tools = [
        WriteFileTool(workspace_path),
        RunCommandTool(workspace_path),
        ReadFileTool(workspace_path),
        RequestNewTool(workspace_path),  # <--- The Meta Tool
    ]

    # Dynamic Tool Loading (Lazy Loader)
    tools_dir = os.path.join(workspace_path, "tools", "registry.json")
    if os.path.exists(tools_dir):
        import json
        import importlib.util

        try:
            with open(tools_dir, "r") as f:
                registry = json.load(f)

            for tool_name, meta in registry.items():
                tool_file = os.path.join(workspace_path, "tools", meta["file"])
                if os.path.exists(tool_file):
                    # Dynamic Import
                    spec = importlib.util.spec_from_file_location(tool_name, tool_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Instantiate the class (Assuming class name = tool_name)
                    tool_class = getattr(module, tool_name)
                    tools.append(tool_class(workspace_path))
                    print(f"[System] Loaded dynamic tool: {tool_name}")
        except Exception as e:
            print(f"[System] Failed to load dynamic tools: {e}")

    if zone == "private":
        model_name = "ollama/qwen2.5-coder:7b"
        print(f"[System] Using Local Model: {model_name}")
    else:
        model_name = "gemini/gemini-3-flash-preview"
        print(f"[System] Using Cloud Model: {model_name}")

    llm_client = LiteLLMClient(model_name=model_name, tools=tools)
    agent = Agent(workspace_path=workspace_path, tools=tools, llm_client=llm_client)

    print("\n[Agent] Executing Plan...")
    goal = f"Execute this plan:\n{plan}\n\nAnalyze the plan, write the necessary code, and run it. Do not just repeat the plan."
    result = agent.run(goal)

    print(f"\n[Agent] Final Result:\n{result}")
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
