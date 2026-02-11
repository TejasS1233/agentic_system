"""
Slide Deck Generator Tool - Create HTML presentations from topics or paper content.

Uses LLM (Groq API direct) to distill content into slides with bullet points
and speaker notes. Outputs a self-contained HTML file using reveal.js CDN.
"""

import os
import json
from typing import Optional
from pydantic import BaseModel, Field


class SlideDeckGeneratorToolArgs(BaseModel):
    topic: str = Field(
        ...,
        description=(
            "Topic or content to create a presentation from. "
            "Can be a topic like 'Introduction to Transformers' or "
            "full paper text/abstract to distill into slides."
        ),
    )
    num_slides: Optional[int] = Field(
        8,
        description="Number of slides to generate (default 8, range 3-20).",
    )
    theme: Optional[str] = Field(
        "black",
        description="Reveal.js theme: 'black', 'white', 'league', 'beige', 'moon', 'night', 'serif', 'simple', 'solarized'.",
    )
    output_path: Optional[str] = Field(
        None,
        description="Output file path for the HTML presentation.",
    )


class SlideDeckGeneratorTool:
    """
    Generate HTML slide deck presentations from topics or paper content.

    Uses Groq LLM to distill content into structured slides with:
    - Title slide
    - Content slides with bullet points
    - Speaker notes for each slide
    - Conclusion slide

    Output is a self-contained HTML file using reveal.js CDN.
    """

    name = "slide_deck_generator"
    description = (
        "Generate a professional HTML slide deck presentation from a topic or paper content. "
        "Creates reveal.js slides with bullet points and speaker notes. "
        "Input a topic like 'Machine Learning in Healthcare' or paste paper text, "
        "and get a ready-to-present HTML file."
    )
    args_schema = SlideDeckGeneratorToolArgs

    def __init__(self, output_dir: str = "/output"):
        self.output_dir = output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            self.output_dir = "/tmp"

    def _call_llm(self, prompt: str) -> str:
        """Call Groq API directly."""
        import requests

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return ""

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                    "temperature": 0.3,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    def _generate_slides_json(self, topic: str, num_slides: int) -> list:
        """Use LLM to generate structured slide data."""
        prompt = f"""You are a presentation expert. Create a slide deck as a JSON array.

RULES:
1. Output ONLY a valid JSON array, no markdown fences, no explanation.
2. Each slide is an object with: "title" (string), "bullets" (array of strings), "notes" (string for speaker notes).
3. First slide should be a title slide with the topic as title and a subtitle in bullets.
4. Last slide should be a summary/conclusion or Q&A slide.
5. Each content slide should have 3-5 concise bullet points.
6. Speaker notes should be 2-3 sentences expanding on the bullets.
7. Generate exactly {num_slides} slides.

TOPIC/CONTENT:
{topic[:3000]}

Generate the JSON array now:"""

        content = self._call_llm(prompt)
        if not content:
            return self._fallback_slides(topic, num_slides)

        # Clean markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            slides = json.loads(content)
            if isinstance(slides, list) and len(slides) > 0:
                return slides
        except json.JSONDecodeError:
            pass

        return self._fallback_slides(topic, num_slides)

    def _fallback_slides(self, topic: str, num_slides: int) -> list:
        """Generate basic fallback slides."""
        slides = [
            {
                "title": topic[:60],
                "bullets": ["A comprehensive overview"],
                "notes": "Welcome to this presentation.",
            },
            {
                "title": "Background",
                "bullets": ["Context and motivation", "Prior work", "Key challenges"],
                "notes": "Let's start with the background.",
            },
            {
                "title": "Key Concepts",
                "bullets": ["Core idea", "Main components", "How it works"],
                "notes": "These are the fundamental concepts.",
            },
            {
                "title": "Results",
                "bullets": ["Key findings", "Performance metrics", "Comparison"],
                "notes": "Here are the main results.",
            },
            {
                "title": "Conclusion",
                "bullets": ["Summary of key points", "Future directions", "Questions?"],
                "notes": "Thank you for your attention.",
            },
        ]
        return slides[:num_slides]

    def _build_html(self, slides: list, theme: str, title: str) -> str:
        """Build a self-contained reveal.js HTML presentation."""
        slides_html = []
        for slide in slides:
            s_title = slide.get("title", "")
            bullets = slide.get("bullets", [])
            notes = slide.get("notes", "")

            bullet_html = ""
            if bullets:
                items = "".join(f"<li>{b}</li>" for b in bullets)
                bullet_html = f"<ul>{items}</ul>"

            notes_html = f"<aside class='notes'>{notes}</aside>" if notes else ""

            slides_html.append(
                f"""<section>
    <h2>{s_title}</h2>
    {bullet_html}
    {notes_html}
</section>"""
            )

        all_slides = "\n".join(slides_html)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4/dist/reset.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4/dist/reveal.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4/dist/theme/{theme}.css">
    <style>
        .reveal ul {{ text-align: left; }}
        .reveal li {{ margin-bottom: 0.5em; font-size: 0.85em; }}
        .reveal h2 {{ font-size: 1.5em; margin-bottom: 0.5em; }}
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
{all_slides}
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@4/dist/reveal.js"></script>
    <script>
        Reveal.initialize({{
            hash: true,
            slideNumber: true,
            showNotes: false,
            transition: 'slide'
        }});
    </script>
</body>
</html>"""

    def run(
        self,
        topic: str,
        num_slides: int = 8,
        theme: str = "black",
        output_path: str = None,
    ) -> str:
        """Generate an HTML slide deck presentation.

        Args:
            topic: Topic or content to create slides from
            num_slides: Number of slides (3-20)
            theme: Reveal.js theme name
            output_path: Output HTML file path

        Returns:
            JSON with success status, output path, and slide count.
        """
        num_slides = max(3, min(20, num_slides or 8))

        valid_themes = {
            "black",
            "white",
            "league",
            "beige",
            "moon",
            "night",
            "serif",
            "simple",
            "solarized",
        }
        if theme not in valid_themes:
            theme = "black"

        # Generate slides
        slides = self._generate_slides_json(topic, num_slides)

        # Build HTML
        title = slides[0].get("title", topic[:50]) if slides else topic[:50]
        html = self._build_html(slides, theme, title)

        # Save — always under /output/ so it persists on host volume
        if output_path and not output_path.startswith(self.output_dir):
            output_path = os.path.join(self.output_dir, os.path.basename(output_path))
        if not output_path:
            existing = [
                f
                for f in os.listdir(self.output_dir)
                if f.startswith("presentation_") and f.endswith(".html")
            ]
            idx = len(existing) + 1
            output_path = os.path.join(self.output_dir, f"presentation_{idx}.html")

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return json.dumps(
            {
                "success": True,
                "output_path": output_path,
                "num_slides": len(slides),
                "theme": theme,
                "title": title,
                "file_size_bytes": len(html.encode("utf-8")),
                "message": f"Presentation with {len(slides)} slides saved to {output_path}. Open in a browser to present.",
            },
            indent=2,
        )
