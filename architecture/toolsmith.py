import os
import re
import json
import time

from litellm import completion

from architecture.gatekeeper import Gatekeeper


class Toolsmith:
    def __init__(self, safe_mode=True, gatekeeper=None):
        self.safe_mode = safe_mode
        self.gatekeeper = gatekeeper or Gatekeeper(strict_mode=safe_mode)
        self.workspace_root = os.path.join(os.getcwd(), "workspace")
        self.tools_dir = os.path.join(self.workspace_root, "tools")
        self.registry_path = os.path.join(self.tools_dir, "registry.json")
        self.metrics_path = os.path.join(self.tools_dir, "metrics.json")
        self.tools_source_path = os.path.join(os.getcwd(), "execution", "tools.py")

        # Ensure directory exists
        os.makedirs(self.tools_dir, exist_ok=True)
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, "w") as f:
                json.dump({}, f)

        # Initialize metrics if not present
        if not os.path.exists(self.metrics_path):
            self._log_metrics("init", {"message": "Metrics initialized"})

        # ignore common words in tag matching
        self.stop_words = {
            "a",
            "an",
            "the",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "and",
            "or",
            "is",
            "it",
            "this",
            "that",
            "tool",
            "can",
            "please",
            "make",
            "create",
            "write",
            "i",
            "need",
            "want",
            "use",
            "help",
            "me",
            "from",
            "into",
        }

    def _log_metrics(self, event_type, details):
        """Logs usage metrics to a JSONL file for paper analysis."""
        entry = {"timestamp": time.time(), "event": event_type, "details": details}
        try:
            # We use append mode for a simpler log, or we could handle a JSON array.
            # For robustness, let's load, append, save.
            data = []
            if os.path.exists(self.metrics_path):
                try:
                    with open(self.metrics_path, "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    data = []

            data.append(entry)
            with open(self.metrics_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Toolsmith] Metrics logging failed: {e}")

    def _is_safe_code(self, code: str) -> bool:
        """
        Uses the Gatekeeper for static analysis to reject dangerous operations.
        Paper Claim: 'Autonomous Secure Tool Generation'.
        """
        if not self.safe_mode:
            return True

        result = self.gatekeeper.validate(code)

        if not result.is_safe:
            for violation in result.violations:
                print(f"[Toolsmith] Safety Violation: {violation}")

        return result.is_safe

    def _get_existing_tools_context(self):
        try:
            with open(self.tools_source_path, "r") as f:
                return f.read()
        except Exception as e:
            return f"# Could not read tools.py: {e}"

    def _tokenize(self, text):
        # Replace non-alphanumeric with space
        clean_text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        tokens = set(clean_text.split())
        return tokens - self.stop_words

    def create_tool(self, requirement: str):
        """
        Generates a new python tool based on a requirement using Tag-Based deduplication.
        """
        start_time = time.time()
        print(f"[Toolsmith] Received request: '{requirement}'")

        # 1. Tag-Based Deduplication Check
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)

            req_tokens = self._tokenize(requirement)

            best_match = None
            highest_score = 0.0

            for name, meta in registry.items():
                tool_tags = set(meta.get("tags", []))
                # Fallback to tokenizing description if tags missing
                if not tool_tags:
                    tool_tags = self._tokenize(meta.get("description", ""))

                # Jaccard-like Overlap Score: Intersection / Request Size
                # (How much of the request is covered by the tool?)
                if not req_tokens:
                    continue

                intersection = req_tokens & tool_tags
                score = len(intersection) / len(req_tokens)

                if score > highest_score:
                    highest_score = score
                    best_match = name

            # Threshold: 30% overlap implies relevance
            if highest_score >= 0.3:
                desc = registry[best_match].get("description", "No description")
                print(
                    f"[Toolsmith] Deduplication HIT. Match: '{best_match}' (Score: {highest_score:.2f})"
                )

                self._log_metrics(
                    "deduplication_hit",
                    {
                        "request": requirement,
                        "matched_tool": best_match,
                        "score": highest_score,
                        "latency": time.time() - start_time,
                    },
                )

                return f"EXISTING TOOL FOUND: '{best_match}' seems to match your request (Score: {highest_score:.2f}).\nDescription: {desc}\nPlease use this tool instead of creating a new one."

            self._log_metrics("generation_start", {"request": requirement})

        except Exception as e:
            print(f"[Toolsmith] Deduplication check failed: {e}")

        # 2. Context Injection
        existing_code = self._get_existing_tools_context()

        # 3. Generate Code (JSON Mode)
        print("[Toolsmith] Tool not found. Contacting Gemini 2.5 Flash...")
        try:
            response = completion(
                model="gemini/gemini-2.5-flash",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert Python Tool Generator.
You MUST generate a JSON object containing the tool code and metadata.

REFERENCE CODE STYLE:
{existing_code}

OUTPUT FORMAT (JSON ONLY):
{{
  "class_name": "NameOfTool",
  "filename": "name_of_tool.py",
  "tags": ["tag1", "tag2", "tag3"],
  "code": "import ... class NameOfTool(Tool): ..."
}}

RULES:
1. `code` must be a valid, escaped python string.
2. `tags` should be 3-5 keywords describing the tool.
3. STRUCTURE REQUIREMENTS:
   - IMPORTS: `from pydantic import BaseModel, Field`
   - ARGS CLASS: Define `class {{ClassName}}Args(BaseModel):` with fields.
   - TOOL CLASS: Define `class {{ClassName}}(Tool):`
   - ATTRIBUTE: Set `args_schema = {{ClassName}}Args` inside the tool class.
   - METHOD: `def run(self, arg1: Type, ...) -> str:` matching the args.
4. Use standard libraries (math, datetime) or allowed system modules (subprocess, sys).
5. DO NOT use `shutil` or key-value stores unless necessary.
6. The tools must be self-contained (include the `Tool` base class definition if distinct from `execution.tools`, or import it if the environment allows. Prefer defining a simple `Tool` abstract base class if unsure of environment).
""",
                    },
                    {"role": "user", "content": f"Create a tool for: {requirement}"},
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            tool_data = json.loads(content)

            class_name = tool_data["class_name"]
            file_name = tool_data["filename"]
            tags = tool_data.get("tags", [])
            tool_code = tool_data["code"]

            # --- SAFETY CHECK ---
            if not self._is_safe_code(tool_code):
                self._log_metrics(
                    "safety_violation",
                    {"request": requirement, "generated_code_snippet": tool_code[:100]},
                )
                return "Error: Generated tool code failed Safety Check (contains banned imports/calls). Request rejected for security."
            # --------------------

            file_path = os.path.join(self.tools_dir, file_name)

            # 4. Save to Disk
            with open(file_path, "w") as f:
                f.write(tool_code)

            print(f"[Toolsmith] Wrote new tool to {file_path}")

            # 5. Update Registry with Tags
            self._update_registry(class_name, file_name, requirement, tags)

            self._log_metrics(
                "tool_created",
                {
                    "request": requirement,
                    "tool_name": class_name,
                    "latency": time.time() - start_time,
                },
            )

            return f"Successfully created {class_name} with tags {tags}. It is ready to use."

        except Exception as e:
            import traceback

            traceback.print_exc()
            self._log_metrics(
                "generation_failed", {"request": requirement, "error": str(e)}
            )
            return f"Tool creation failed: {e}"

    def _update_registry(self, class_name, file_name, description, tags):
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

        data[class_name] = {"file": file_name, "description": description, "tags": tags}

        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)
