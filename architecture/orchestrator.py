"""Orchestrator - decomposes queries and creates execution plans."""

import json
import re
from typing import Optional

from architecture.llm_manager import get_llm_manager
from architecture.prompts import (
    DECOMPOSITION_PROMPT,
    ROUTING_PROMPT,
    get_direct_generation_prompt,
)
from architecture.schemas import (
    DecompositionResult,
    Domain,
    ExecutionPlan,
    ExecutionStep,
    SubTask,
    ToolMatch,
)
from architecture.tool_retriever import ToolRetriever
from architecture.toolsmith import Toolsmith
from utils.logger import get_logger

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.5


class Orchestrator:
    """Decomposes queries into domain-based subtasks and creates execution plans."""

    def __init__(
        self,
        toolsmith: Optional[Toolsmith] = None,
        executor=None,
        registry_path: Optional[str] = None,
    ):
        self.llm = get_llm_manager(model="llama-3.3-70b-versatile")
        self.toolsmith = toolsmith or Toolsmith()
        self.executor = executor
        self.registry_path = registry_path or self.toolsmith.registry_path
        self.retriever = ToolRetriever(self.registry_path)
        logger.info("Orchestrator initialized with ToolRetriever")

    def run(self, user_query: str) -> str:
        """Execute request, routing between direct generation and tool execution."""
        logger.info(f"Processing: {user_query[:80]}...")

        task_type = self._classify_request(user_query)
        logger.info(f"Task Classification: {task_type}")

        if task_type == "DIRECT_RESPONSE":
            return self._handle_direct_generation(user_query)

        decomposition = self._decompose(user_query)
        logger.info(f"Decomposed into {len(decomposition.subtasks)} subtasks")

        tool_matches = self._retrieve_tools(decomposition.subtasks)
        tool_matches = self._ensure_tools(tool_matches, decomposition.subtasks)
        plan = self._create_plan(decomposition, tool_matches)

        if self.executor:
            return self.executor.execute(plan)
        else:
            return plan.model_dump_json(indent=2)

    def _decompose(self, query: str) -> DecompositionResult:
        """LLM decomposes task into subtasks with awareness of available tools."""
        tool_context = self._get_tool_context_for_decomposition(query)

        enhanced_prompt = DECOMPOSITION_PROMPT
        if tool_context:
            enhanced_prompt += f"\n\nAVAILABLE TOOLS (use this to avoid redundant steps):\n{tool_context}\n\nIMPORTANT: If a tool already outputs the data you need (e.g., 'language' field in repos), do NOT create a separate extraction step."

        messages = [
            {"role": "system", "content": enhanced_prompt},
            {"role": "user", "content": f"Decompose:\n{query}"},
        ]

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
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return DecompositionResult(
                original_query=query,
                subtasks=[SubTask(id="st_1", description=query, domain=Domain.SYSTEM)],
            )

    def _get_tool_context_for_decomposition(self, query: str) -> str:
        """Retrieve relevant tools and format for decomposition prompt."""
        try:
            candidates = self.retriever.retrieve_with_scoring(query=query, top_k=5)
            if not candidates:
                return ""

            tool_lines = []
            for c in candidates[:3]:
                tool_name = c.get("tool", "")
                desc = c.get("description", "")[:100]

                registry_info = self._get_tool_registry_info(tool_name)
                output_schema = registry_info.get("output_schema", {})

                if output_schema:
                    output_keys = output_schema.get("keys", [])
                    item_keys = output_schema.get(
                        "repos_item_keys", output_schema.get("item_keys", [])
                    )
                    output_str = f"outputs: {output_keys}"
                    if item_keys:
                        output_str += f", each item has: {item_keys}"
                else:
                    output_str = ""

                tool_lines.append(
                    f"- {tool_name}: {desc}"
                    + (f" [{output_str}]" if output_str else "")
                )

            return "\n".join(tool_lines)
        except Exception as e:
            logger.warning(f"Could not get tool context: {e}")
            return ""

    def _get_tool_registry_info(self, tool_name: str) -> dict:
        """Get full registry entry for a tool."""
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            return registry.get(tool_name, {})
        except Exception:
            return {}

    def _retrieve_tools(self, subtasks: list[SubTask]) -> list[ToolMatch]:
        """Use hybrid retrieval for tool matching."""
        matches = []
        used_tools = set()

        for st in subtasks:
            domain = st.domain.value if hasattr(st.domain, "value") else str(st.domain)

            if domain == "search" or "search" in st.description.lower():
                serp_tool_name = "SerpSearchTool"
                try:
                    tool_file = self._get_tool_file(serp_tool_name)
                    if tool_file:
                        logger.info(
                            f"{st.id} -> {serp_tool_name} (Forced by Domain.SEARCH)"
                        )
                        matches.append(
                            ToolMatch(
                                subtask_id=st.id,
                                tool_name=serp_tool_name,
                                tool_file=tool_file,
                                matched=True,
                                confidence=1.0,
                            )
                        )
                        used_tools.add(serp_tool_name)
                        continue
                except Exception:
                    pass

            scored_results = self.retriever.retrieve_with_scoring(
                query=st.description,
                top_k=5,
            )

            available = [c for c in scored_results if c["tool"] not in used_tools]

            if not available:
                matches.append(
                    ToolMatch(
                        subtask_id=st.id,
                        tool_name="",
                        tool_file="",
                        matched=False,
                        confidence=0.0,
                    )
                )
                logger.info(f"{st.id} -> NO MATCH (all tools used)")
                continue

            selected_name = self.retriever.llm_select_tool(
                query=st.description,
                candidates=available,
                llm_manager=self.llm,
            )

            best_match = next(
                (c for c in available if c["tool"] == selected_name), available[0]
            )

            tool_name = best_match["tool"]
            similarity = best_match.get("similarity", 0)
            combined_score = best_match.get("combined_score", 0)
            tag_matches = best_match.get("tag_matches", 0)
            matched = combined_score >= SIMILARITY_THRESHOLD
            used_tools.add(tool_name)

            matches.append(
                ToolMatch(
                    subtask_id=st.id,
                    tool_name=tool_name,
                    tool_file=self._get_tool_file(tool_name),
                    matched=matched,
                    confidence=max(0, combined_score),
                )
            )

            domain_tools = set(self.retriever.get_tools_by_domain(domain))
            in_domain = "in-domain" if tool_name in domain_tools else "cross-domain"
            logger.info(
                f"{st.id} -> {tool_name} (LLM selected, semantic: {similarity:.3f}, tags: +{tag_matches}, {in_domain})"
            )

        return matches

    def _get_tool_file(self, tool_name: str) -> str:
        """Get file path for a tool from registry."""
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            return registry.get(tool_name, {}).get("file", "")
        except Exception:
            return ""

    def _ensure_tools(
        self, matches: list[ToolMatch], subtasks: list[SubTask]
    ) -> list[ToolMatch]:
        """Route to Toolsmith for any unmatched subtasks."""
        subtask_map = {st.id: st for st in subtasks}
        used_tools = set()

        for match in matches:
            if match.matched and match.tool_name:
                if match.tool_name in used_tools:
                    logger.info(
                        f"Tool {match.tool_name} already used. Invalidating match."
                    )
                    match.matched = False
                    match.tool_name = None
                    match.confidence = 0.0
                else:
                    used_tools.add(match.tool_name)
                    continue

            if not match.matched:
                st = subtask_map[match.subtask_id]
                description = st.description

                for attempt in range(2):
                    if attempt > 0:
                        logger.info(
                            f"Retrying tool creation (attempt {attempt + 1})..."
                        )
                        avoid_str = ", ".join(list(used_tools))
                        description = f"{st.description} (Create a NEW tool, do not use: {avoid_str})"

                    logger.info(f"Toolsmith creating: {description}")
                    result = self.toolsmith.create_tool(description)
                    logger.info(f"Toolsmith: {result[:100]}...")

                    tool_name = None
                    m = re.search(r"(?:Successfully created|Created) (\w+)", result)
                    if m:
                        tool_name = m.group(1)
                    elif "EXISTING TOOL FOUND:" in result:
                        m = re.search(r"EXISTING TOOL FOUND: '(\w+)'", result)
                        if m:
                            tool_name = m.group(1)

                    if tool_name and tool_name not in used_tools:
                        match.tool_name = tool_name
                        match.tool_file = self._get_tool_file(tool_name)
                        match.matched = True
                        match.confidence = 1.0
                        used_tools.add(tool_name)
                        logger.info(f"Tool assigned: {tool_name}")
                        break

                    if tool_name in used_tools:
                        logger.info(f"Tool {tool_name} is a duplicate.")

                if not match.matched:
                    self.retriever = ToolRetriever(self.registry_path)
                    new_result = self.retriever.retrieve(st.description, top_k=5)
                    primary = new_result.get("primary_tools", [])

                    for candidate in primary:
                        candidate_name = candidate["tool"]
                        if candidate_name not in used_tools:
                            similarity = candidate.get("similarity", 0)
                            if similarity >= SIMILARITY_THRESHOLD:
                                match.tool_name = candidate_name
                                match.tool_file = self._get_tool_file(match.tool_name)
                                match.matched = True
                                match.confidence = max(0, similarity)
                                used_tools.add(candidate_name)
                                logger.info(
                                    f"Fallback assigned: {candidate_name} (similarity: {similarity:.3f})"
                                )
                                break
                            else:
                                logger.info(
                                    f"Fallback candidate {candidate_name} rejected "
                                    f"(similarity {similarity:.3f} < {SIMILARITY_THRESHOLD})"
                                )

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
            input_from_step = step_num_map.get(st.input_from) if st.input_from else None
            step_type = getattr(st, "step_type", "tool") or "tool"
            steps.append(
                ExecutionStep(
                    step_number=i + 1,
                    subtask_id=st.id,
                    description=st.description,
                    tool_name=match.tool_name if match and step_type == "tool" else "",
                    step_type=step_type,
                    expected_output=f"Output{st.description}",
                    depends_on=dep_steps,
                    input_from=input_from_step,
                )
            )

        logger.info("=" * 60)
        logger.info(f"EXECUTION PLAN: {len(steps)} steps")
        logger.info("=" * 60)
        for step in steps:
            deps_str = (
                f"depends_on={step.depends_on}"
                if step.depends_on
                else "no dependencies"
            )
            input_str = (
                f"input_from=step_{step.input_from}"
                if step.input_from
                else "uses description"
            )
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

        prompt = get_direct_generation_prompt(query)

        result = self.llm.generate_text(prompt)
        response = result.get("content", "")

        if not response:
            logger.error(f"Direct generation failed: {result.get('error')}")
            return f"Error generating response: {result.get('error')}"

        saved_files = []
        if "```" in response:
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
