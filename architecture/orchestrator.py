import os
try:
    from litellm import completion
    print("[Orchestrator] LiteLLM successfully imported.")
except ImportError:
    print("[Orchestrator] CRITICAL: litellm not found. Run 'uv add litellm'")
    raise

class Orchestrator:
    def __init__(self, model_name: str = "gemini/gemini-3-flash-preview"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")

    def plan(self, goal: str):
        """
        Decomposes a high-level goal into a list of actionable steps.
        """
        print(f"[Orchestrator] meaningful planning for: '{goal}' using {self.model_name}")
        
        # In a real impl, this would be a complex prompt with history
        messages = [{
            "role": "system", 
            "content": "You are an expert technical planner. Breakdown the user's request into steps."
        }, {
            "role": "user", 
            "content": goal
        }]

        try:
            response = completion(model=self.model_name, messages=messages, api_key=self.api_key)
            plan_text = response.choices[0].message.content
            return plan_text
        except Exception as e:
            return f"Planning failed: {e}"

    def run(self, goal: str):
        """
        Main execution loop.
        """
        plan = self.plan(goal)
        print(f"[Orchestrator] Generated Plan:\n{plan}")
        return plan
