"""
Mermaid Diagram Tool - Generate Mermaid diagram code from natural language
descriptions and render it as a PNG image.

Supports: flowchart, sequence, class, state, ER, gantt, pie, mindmap, timeline, etc.
Uses mermaid.ink API (free, no auth) for rendering.
Calls Groq LLM API directly via requests for code generation inside Docker.
"""

import os
import json
import base64
import zlib
from typing import Optional
from pydantic import BaseModel, Field


class MermaidDiagramToolArgs(BaseModel):
    description: str = Field(
        ...,
        description=(
            "Natural language description of the diagram to generate. "
            "E.g. 'A flowchart showing user login flow with email verification' "
            "or 'Sequence diagram of API request between client, gateway and database'"
        ),
    )
    diagram_type: Optional[str] = Field(
        None,
        description=(
            "Diagram type hint: 'flowchart', 'sequence', 'class', 'state', "
            "'er', 'gantt', 'pie', 'mindmap', 'timeline', 'gitgraph'. "
            "Auto-detected from description if not provided."
        ),
    )
    output_path: Optional[str] = Field(
        None,
        description="Output file path for the rendered image (PNG). Defaults to /output/diagram_<n>.png",
    )
    output_format: Optional[str] = Field(
        "png",
        description="Output format: 'png' or 'svg'. Defaults to 'png'.",
    )
    theme: Optional[str] = Field(
        "default",
        description="Mermaid theme: 'default', 'dark', 'forest', 'neutral'.",
    )


class MermaidDiagramTool:
    """
    Generate Mermaid diagrams from natural language descriptions and render to image.

    Flow:
      1. Takes a plain-English description of a diagram
      2. Calls Groq LLM API directly to produce valid Mermaid markup
      3. Renders the markup to PNG/SVG via mermaid.ink (free, no auth)
      4. Saves the image and returns the Mermaid code + file path

    Supports all major Mermaid diagram types:
      flowchart, sequence, class, state, ER, gantt, pie, mindmap, timeline, gitgraph
    """

    name = "mermaid_diagram"
    description = (
        "Generate a Mermaid diagram from a natural language description and render it as a PNG/SVG image. "
        "Supports flowcharts, sequence diagrams, class diagrams, state diagrams, ER diagrams, "
        "gantt charts, pie charts, mindmaps, timelines, and git graphs. "
        "Give it a description like 'flowchart of user authentication with OAuth' and it produces "
        "the Mermaid code and a rendered image."
    )
    args_schema = MermaidDiagramToolArgs

    # Map of keywords to diagram types for auto-detection
    TYPE_KEYWORDS = {
        "flowchart": ["flowchart", "flow", "process", "workflow", "pipeline", "steps", "decision"],
        "sequence": ["sequence", "interaction", "message", "request", "response", "api call", "communication"],
        "classDiagram": ["class", "inheritance", "interface", "oop", "object", "uml class"],
        "stateDiagram-v2": ["state", "transition", "finite", "machine", "lifecycle", "status"],
        "erDiagram": ["er ", "entity", "relationship", "database schema", "tables", "foreign key"],
        "gantt": ["gantt", "timeline", "schedule", "project plan", "milestones", "sprint"],
        "pie": ["pie", "distribution", "percentage", "proportion", "share", "breakdown"],
        "mindmap": ["mindmap", "mind map", "brainstorm", "concept map", "idea map"],
        "timeline": ["timeline", "chronolog", "history", "events over time"],
        "gitGraph": ["git", "branch", "commit", "merge", "gitflow"],
    }

    def __init__(self, output_dir: str = "/output"):
        self.output_dir = output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            self.output_dir = "/tmp"

    def _detect_diagram_type(self, description: str) -> str:
        """Auto-detect the best Mermaid diagram type from the description."""
        desc_lower = description.lower()
        best_type = "flowchart"
        best_score = 0

        for diagram_type, keywords in self.TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            if score > best_score:
                best_score = score
                best_type = diagram_type

        return best_type

    def _call_llm_direct(self, prompt: str) -> str:
        """Call Groq API directly via requests (works inside Docker container)."""
        import requests as req

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return ""

        try:
            resp = req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.2,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return ""

    def _generate_mermaid_code(self, description: str, diagram_type: str, theme: str) -> str:
        """Use LLM to generate Mermaid markup from a natural language description."""

        prompt = f"""You are a Mermaid diagram expert. Generate ONLY valid Mermaid markup code.

RULES:
1. Output ONLY the raw Mermaid code. No markdown fences, no explanation, no extra text.
2. The first line must be the diagram type declaration (e.g. "flowchart TD", "sequenceDiagram", etc.).
3. Use the diagram type: {diagram_type}
4. Make the diagram detailed and accurate based on the description.
5. Use descriptive labels for nodes and edges.
6. For flowcharts, prefer TD (top-down) or LR (left-right) direction.
7. Ensure all syntax is valid Mermaid v10+.
8. Avoid special characters that could break rendering (use quotes for labels with special chars).
9. Keep node IDs simple alphanumeric (e.g., A, B, C or id1, id2).
10. Do NOT wrap in markdown code fences.

DESCRIPTION: {description}

Generate the Mermaid code now:"""

        content = self._call_llm_direct(prompt)

        if not content:
            return self._fallback_diagram(description, diagram_type)

        # Clean up: remove markdown fences if LLM wraps them
        code = content.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            lines = lines[1:]  # Remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove closing fence
            code = "\n".join(lines).strip()

        return code

    def _fallback_diagram(self, description: str, diagram_type: str) -> str:
        """Generate a simple fallback diagram when LLM is unavailable."""
        if diagram_type == "pie":
            return 'pie title Distribution\n    "Part A" : 40\n    "Part B" : 35\n    "Part C" : 25'
        elif diagram_type == "mindmap":
            return f"mindmap\n  root(({description[:30]}))\n    Branch A\n    Branch B\n    Branch C"
        elif diagram_type == "sequenceDiagram":
            return (
                "sequenceDiagram\n"
                "    participant A as Client\n"
                "    participant B as Server\n"
                "    participant C as Database\n"
                "    A->>B: Request\n"
                "    B->>C: Query\n"
                "    C-->>B: Result\n"
                "    B-->>A: Response"
            )
        else:
            return (
                f"flowchart TD\n"
                f'    A[Start] --> B["{description[:40]}"]\n'
                f"    B --> C[Process]\n"
                f"    C --> D[End]"
            )

    def _render_with_mermaid_ink(self, mermaid_code: str, output_format: str = "png") -> bytes:
        """Render Mermaid code to image using mermaid.ink (free, no auth, no Cloudflare)."""
        import urllib.request
        import urllib.error

        # mermaid.ink expects base64-encoded mermaid code in the URL
        encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("utf-8")
        
        if output_format == "svg":
            url = f"https://mermaid.ink/svg/{encoded}"
        else:
            url = f"https://mermaid.ink/img/{encoded}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MermaidDiagramTool/1.0",
                "Accept": "image/png,image/svg+xml,*/*",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"mermaid.ink error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach mermaid.ink: {e.reason}")

    def _render_with_kroki(self, mermaid_code: str, output_format: str = "png") -> bytes:
        """Fallback: Render via Kroki.io API with proper headers."""
        import urllib.request
        import urllib.error

        url = f"https://kroki.io/mermaid/{output_format}"
        data = mermaid_code.encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "text/plain",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MermaidDiagramTool/1.0",
                "Accept": f"image/{output_format}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Kroki API error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach Kroki API: {e.reason}")

    def _render(self, mermaid_code: str, output_format: str = "png") -> bytes:
        """Try mermaid.ink first, then fall back to Kroki."""
        try:
            return self._render_with_mermaid_ink(mermaid_code, output_format)
        except RuntimeError as e1:
            try:
                return self._render_with_kroki(mermaid_code, output_format)
            except RuntimeError as e2:
                raise RuntimeError(
                    f"All renderers failed.\n  mermaid.ink: {e1}\n  Kroki: {e2}"
                )

    def run(
        self,
        description: str,
        diagram_type: str = None,
        output_path: str = None,
        output_format: str = "png",
        theme: str = "default",
    ) -> str:
        """Generate a Mermaid diagram from description and render it as an image.

        Args:
            description: Natural language description of the diagram
            diagram_type: Optional type hint (flowchart, sequence, etc.)
            output_path: Where to save the rendered image
            output_format: 'png' or 'svg'
            theme: Mermaid theme (default, dark, forest, neutral)

        Returns:
            JSON string with mermaid_code, output_path, and status.
        """
        # 1. Detect or validate diagram type
        if not diagram_type or diagram_type.lower() == "auto":
            diagram_type = self._detect_diagram_type(description)

        # Normalize type names
        type_map = {
            "flowchart": "flowchart",
            "flow": "flowchart",
            "sequence": "sequenceDiagram",
            "class": "classDiagram",
            "state": "stateDiagram-v2",
            "er": "erDiagram",
            "gantt": "gantt",
            "pie": "pie",
            "mindmap": "mindmap",
            "timeline": "timeline",
            "gitgraph": "gitGraph",
            "git": "gitGraph",
        }
        diagram_type = type_map.get(diagram_type.lower(), diagram_type)

        # 2. Generate Mermaid code via LLM
        mermaid_code = self._generate_mermaid_code(description, diagram_type, theme)

        # 3. Add theme directive if not default
        if theme and theme != "default":
            theme_directive = f"%%{{init: {{'theme': '{theme}'}}}}%%\n"
            if not mermaid_code.startswith("%%{"):
                mermaid_code = theme_directive + mermaid_code

        # 4. Render to image
        output_format = output_format.lower() if output_format else "png"
        if output_format not in ("png", "svg"):
            output_format = "png"

        try:
            image_data = self._render(mermaid_code, output_format)
        except RuntimeError as e:
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                    "mermaid_code": mermaid_code,
                    "message": "Diagram code was generated but rendering failed. The Mermaid code is included above.",
                },
                indent=2,
            )

        # 5. Save the image — always under /output/ so it persists on host volume
        if output_path and not output_path.startswith(self.output_dir):
            output_path = os.path.join(self.output_dir, os.path.basename(output_path))
        if not output_path:
            existing = [
                f
                for f in os.listdir(self.output_dir)
                if f.startswith("diagram_") and f.endswith(f".{output_format}")
            ]
            idx = len(existing) + 1
            output_path = os.path.join(self.output_dir, f"diagram_{idx}.{output_format}")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(image_data)

        # 6. Also save the mermaid source code alongside the image
        mmd_path = output_path.rsplit(".", 1)[0] + ".mmd"
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(mermaid_code)

        return json.dumps(
            {
                "success": True,
                "mermaid_code": mermaid_code,
                "diagram_type": diagram_type,
                "output_path": output_path,
                "source_path": mmd_path,
                "format": output_format,
                "file_size_bytes": len(image_data),
                "message": f"Diagram rendered successfully and saved to {output_path}",
            },
            indent=2,
        )
