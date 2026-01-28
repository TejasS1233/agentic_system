import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure architecture and execution modules are importable
sys.path.append(os.path.join(os.getcwd(), 'architecture'))

from architecture.orchestrator import Orchestrator
from architecture.dispatcher import Dispatcher
from architecture.toolsmith import Toolsmith

def main():
    print("=== IASCIS Architecture Demo ===")
    
    orchestrator = Orchestrator(model_name="gemini/gemini-3-flash-preview")
    dispatcher = Dispatcher()
    
    user_task = "Analyze the sensitive_payroll.csv and calculate the sum of bonuses."
    print(f"\n[User] Task: {user_task}")

    # 1. Routing
    zone = dispatcher.route(task_description=user_task, file_context=["sensitive_payroll.csv"])
    
    # 2. Planning
    print(f"\n[Orchestrator] Planning task...")
    plan = orchestrator.run(user_task)
    
    # 3. Execution
    print(f"\n[System] Initializing Execution Agent for Zone: {zone.upper()}")
    
    workspace_path = os.path.join(os.getcwd(), "workspace")
    os.makedirs(workspace_path, exist_ok=True)
    
    try:
        from execution.tools import WriteFileTool, RunCommandTool, ReadFileTool
        from execution.llm import LiteLLMClient
        from execution.core import Agent
    except ImportError:
        sys.path.append(os.getcwd())
        from execution.tools import WriteFileTool, RunCommandTool, ReadFileTool
        from execution.llm import LiteLLMClient
        from execution.core import Agent

    tools = [
        WriteFileTool(workspace_path),
        RunCommandTool(workspace_path),
        ReadFileTool(workspace_path)
    ]
    
    if zone == "private":
        model_name = "ollama/qwen2.5-coder:7b"
        print(f"[System] Using Local Model: {model_name}")
    else:
        model_name = "gemini/gemini-3-flash-preview"
        print(f"[System] Using Cloud Model: {model_name}")

    llm_client = LiteLLMClient(model_name=model_name, tools=tools)
    agent = Agent(workspace_path=workspace_path, tools=tools, llm_client=llm_client)
    
    print(f"\n[Agent] Executing Plan...")
    goal = f"Execute this plan:\n{plan}\n\nAnalyze the plan, write the necessary code, and run it. Do not just repeat the plan."
    result = agent.run(goal)
    
    print(f"\n[Agent] Final Result:\n{result}")
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
