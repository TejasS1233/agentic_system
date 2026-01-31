import os
try:
    from litellm import completion
    print("[Orchestrator] LiteLLM successfully imported.")
except ImportError:
    print("[Orchestrator] CRITICAL: litellm not found. Run 'uv add litellm'")
    raise

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class Orchestrator:
    def __init__(self, model_name: str = "gemini/gemini-3-flash-preview"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_base = None
        self.extra_headers = None
        
        if "ollama" in model_name.lower():
            self.api_base = OLLAMA_URL

    def plan(self, goal: str):
        """Decomposes a high-level goal into actionable steps."""
        print(f"[Orchestrator] meaningful planning for: '{goal}' using {self.model_name}")
        
        messages = [{
            "role": "system", 
            "content": "You are an expert technical planner. Breakdown the user's request into steps."
        }, {
            "role": "user", 
            "content": goal
        }]

        try:
            kwargs = {"model": self.model_name, "messages": messages}
            
            if "gemini" in self.model_name.lower():
                kwargs["api_key"] = self.api_key
            
            if self.api_base:
                kwargs["api_base"] = self.api_base
            if self.extra_headers:
                kwargs["extra_headers"] = self.extra_headers
            
            response = completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            return f"Planning failed: {e}"

    def run(self, goal: str):
        """Main execution loop."""
        plan = self.plan(goal)
        print(f"[Orchestrator] Generated Plan:\n{plan}")
        return plan
