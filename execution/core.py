from typing import List, Dict, Any, Type, Optional
from .schemas import AgentState
from .tools import Tool
from .llm import LLMClient

class Agent:
    def __init__(self, workspace_path: str, tools: List[Tool], llm_client: LLMClient):
        self.state = AgentState(workspace_path=workspace_path)
        self.tools = {t.name: t for t in tools}
        self.llm_client = llm_client
        self.chat = self.llm_client.start_chat()

    def run(self, goal: str):
        # Reset chat if needed, though typically LLMClient handles it
        self.chat = self.llm_client.start_chat()
        return self._run_with_retry(goal)

    def _run_with_retry(self, goal: str, max_retries: int = 5):
        import time
        
        delay = 10  # Start with 10s delay

        for attempt in range(max_retries):
            try:
                text_content = self.llm_client.send_message(self.chat, goal)
                return text_content
            except Exception as e:
                import traceback
                traceback.print_exc()
                if "429" in str(e) or "ResourceExhausted" in str(e) or "503" in str(e):
                    print(f"Service Unavailable or Quota exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                else:
                     return str(e)
        
        return "Failed after max retries."
