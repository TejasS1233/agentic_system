import json
from typing import Optional

from architecture.schemas import (
    Domain,
    SubTask,
    DecompositionResult,
    ToolMatch,
    ExecutionStep,
    ExecutionPlan,
)
from architecture.llm_manager import get_llm_manager
from architecture.toolsmith import Toolsmith
from architecture.tool_retriever import ToolRetriever
from architecture.prompts import DECOMPOSITION_PROMPT, ROUTING_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """decomposes queries into domain based subtasks and creates execution plans"""

    def __init__(
        self,
        toolsmith: Optional[Toolsmith] = None,
        executor=None,
        registry_path: Optional[str] = None,
    ):
        self.llm = get_llm_manager()
        self.toolsmith = toolsmith or Toolsmith()
        self.executor = executor
        self.registry_path = registry_path or self.toolsmith.registry_path
        self.retriever = ToolRetriever(self.registry_path)
        logger.info("Orchestrator initialized with ToolRetriever")

    def run(self, user_query: str) -> str:
        """Execute request, routing between direct generation and tool execution."""
        logger.info(f"Processing: {user_query[:80]}...")

        # 1. Classify request
        task_type = self._classify_request(user_query)
        logger.info(f"Task Classification: {task_type}")

        if task_type == "DIRECT_RESPONSE":
            return self._handle_direct_generation(user_query)

        # 2. Decompose and execute (Complex Task)
        decomposition = self._decompose(user_query)
        logger.info(f"Decomposed into {len(decomposition.subtasks)} subtasks")

        tool_matches = self._retrieve_tools(decomposition.subtasks)
        tool_matches = self._ensure_tools(tool_matches, decomposition.subtasks)
        plan = self._create_plan(decomposition, tool_matches)

        # if there is exec then run it otherwvise show the plan
        if self.executor:
            return self.executor.execute(plan)
        else:
            return plan.model_dump_json(indent=2)

    def _decompose(self, query: str) -> DecompositionResult:
        """LLM decomposes task into subtasks"""
        messages = [
            {"role": "system", "content": DECOMPOSITION_PROMPT},
            {"role": "user", "content": f"Decompose:\n{query}"},
        ]
        # for the error on the LLM / AI end
        response = self.llm.generate_json(messages=messages, max_tokens=2048)
        if response.get("error"):
            logger.error(f"Decomposition failed: {response['error']}")
            return DecompositionResult(
                original_query=query,
                subtasks=[SubTask(id="st_1", description=query, domain=Domain.SYSTEM)],
            )
        try:
            data = json.loads(response["content"])
            return DecompositionResult(**data)
        # for the logical error
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return DecompositionResult(
                original_query=query,
                subtasks=[SubTask(id="st_1", description=query, domain=Domain.SYSTEM)],
            )

    def _retrieve_tools(self, subtasks: list[SubTask]) -> list[ToolMatch]:
        """use tool retriver to find relevant tools for each task"""
        matches = []
        for st in subtasks:
            result = self.retriever.retrieve(
                query=st.description,
                top_k=3,
                expand=True,  # for the composable part
            )
            primary = result.get("primary_tools", [])
            if primary:
                # Best match
                best = primary[0]
                tool_name = best["tool"]
                distance = best["distance"]
                matched = distance < 1.0  # Stricter: trigger tool creation for poor matches
                confidence = max(0, 1 - distance)  # Linear confidence
                matches.append(
                    ToolMatch(
                        subtask_id=st.id,
                        tool_name=tool_name,
                        tool_file=self._get_tool_file(tool_name),
                        matched=matched,
                        confidence=confidence,
                    )
                )
                logger.info(
                    f"{st.id} → {tool_name} (dist: {distance:.3f}, matched: {matched})"
                )
            else:
                # No match found
                matches.append(
                    ToolMatch(
                        subtask_id=st.id,
                        tool_name="",
                        tool_file="",
                        matched=False,
                        confidence=0.0,
                    )
                )
                logger.info(f"{st.id} → NO MATCH")
        return matches

    def _get_tool_file(self, tool_name: str) -> str:
        """Get file path for a tool from registry."""
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            return registry.get(tool_name, {}).get("file", "")
        except:
            return ""

    def _ensure_tools(
        self, matches: list[ToolMatch], subtasks: list[SubTask]
    ) -> list[ToolMatch]:
        """Route to Toolsmith for any unmatched subtasks."""
        subtask_map = {st.id: st for st in subtasks}
        for match in matches:
            if not match.matched:
                st = subtask_map[match.subtask_id]
                logger.info(f"Toolsmith creating: {st.description}")
                # generate new tool
                result = self.toolsmith.create_tool(st.description)
                logger.info(f"Toolsmith: {result[:100]}...")

                # Parse tool name from Toolsmith response
                tool_name = None
                if "EXISTING TOOL FOUND:" in result:
                    # Extract from: "EXISTING TOOL FOUND: 'ToolName' seems to match..."
                    import re

                    m = re.search(r"EXISTING TOOL FOUND: '(\w+)'", result)
                    if m:
                        tool_name = m.group(1)
                elif "Successfully created" in result:
                    # Extract from: "Successfully created SaveFileAsJson..."
                    import re

                    m = re.search(r"Successfully created (\w+)", result)
                    if m:
                        tool_name = m.group(1)

                if tool_name:
                    match.tool_name = tool_name
                    match.tool_file = self._get_tool_file(tool_name)
                    match.matched = True
                    match.confidence = 1.0
                    logger.info(f"Tool assigned: {tool_name}")
                else:
                    # Fallback: Rebuild retriever and search
                    self.retriever = ToolRetriever(self.registry_path)
                    new_result = self.retriever.retrieve(st.description, top_k=1)
                    primary = new_result.get("primary_tools", [])
                    if primary:
                        match.tool_name = primary[0]["tool"]
                        match.tool_file = self._get_tool_file(match.tool_name)
                        match.matched = True
                        match.confidence = 1.0
        return matches

    def _create_plan(
        self, decomposition: DecompositionResult, tool_matches: list[ToolMatch]
    ) -> ExecutionPlan:
        """Generate execution plan with proper DAG data flow."""
        match_map = {m.subtask_id: m for m in tool_matches}
        step_num_map = {st.id: i + 1 for i, st in enumerate(decomposition.subtasks)}
        steps = []
        for i, st in enumerate(decomposition.subtasks):
            match = match_map.get(st.id)
            dep_steps = [step_num_map[d] for d in st.depends_on if d in step_num_map]
            # Convert input_from subtask ID to step number
            input_from_step = step_num_map.get(st.input_from) if st.input_from else None
            steps.append(
                ExecutionStep(
                    step_number=i + 1,
                    subtask_id=st.id,
                    description=st.description,
                    tool_name=match.tool_name if match else "",
                    expected_output=f"Output{st.description}",
                    depends_on=dep_steps,
                    input_from=input_from_step,
                )
            )
        
        # Log the complete plan
        logger.info("=" * 60)
        logger.info(f"EXECUTION PLAN: {len(steps)} steps")
        logger.info("=" * 60)
        for step in steps:
            deps_str = f"depends_on={step.depends_on}" if step.depends_on else "no dependencies"
            input_str = f"input_from=step_{step.input_from}" if step.input_from else "uses description"
            logger.info(f"  Step {step.step_number}: {step.description[:50]}...")
            logger.info(f"    Tool: {step.tool_name}")
            logger.info(f"    {deps_str}, {input_str}")
        logger.info("=" * 60)
        
        return ExecutionPlan(original_query=decomposition.original_query, steps=steps)

    def _classify_request(self, query: str) -> str:
        """Determine if request is direct response or complex task."""
        try:
            prompt = ROUTING_PROMPT.format(query=query)
            response = self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}], max_tokens=100
            )
            # Response content is a JSON string, need to parse it
            content_str = response.get("content", "{}")
            if not content_str:
                return "COMPLEX_TASK"

            data = json.loads(content_str)
            return data.get("category", "COMPLEX_TASK")

        except Exception as e:
            logger.warning(f"Classification failed: {e}, defaulting to COMPLEX_TASK")
            return "COMPLEX_TASK"

    def _handle_direct_generation(self, query: str) -> str:
        """Handle direct response requests (code, text, explanations)."""
        logger.info("Handling as direct response")

        prompt = f"""Fulfill the following request directly.
        
        User Request: {query}
        
        Rules:
        1. If writing code, put it in a Markdown code block (```python, ```bash, etc).
        2. If writing text/creative content, just write it naturally.
        3. Be concise and helpful.
        """

        # Use generate_text which expects a prompt string and returns a dict
        result = self.llm.generate_text(prompt)
        response = result.get("content", "")

        if not response:
            logger.error(f"Direct generation failed: {result.get('error')}")
            return f"Error generating response: {result.get('error')}"

        # Check for code blocks to save
        saved_files = []
        if "```" in response:
            import re

            # Extract all code blocks
            matches = re.findall(r"```(\w+)?\n(.*?)```", response, re.DOTALL)

            for i, (lang, code) in enumerate(matches):
                lang = lang.strip().lower() if lang else "txt"
                ext = (
                    "py"
                    if lang == "python"
                    else "sh"
                    if lang in ["bash", "shell"]
                    else "txt"
                )

                filename = f"generated_content_{i + 1}.{ext}"
                # If explicit name requested, logic could be added here, but simple for now

                filepath = self.executor.workspace / "outputs" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)

                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(code.strip())
                    saved_files.append(str(filepath))
                except Exception as e:
                    logger.warning(f"Failed to save generated content: {e}")

        if saved_files:
            logger.info(f"Saved generated content to {saved_files}")
            return f"{response}\n\n[System: Content saved to {', '.join(saved_files)}]"

        return response
