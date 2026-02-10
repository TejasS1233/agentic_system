"""
LaTeX Equation Renderer Tool - Render LaTeX math notation as PNG images.

Uses matplotlib's mathtext engine — no system TeX install needed.
Includes LLM-powered conversion from natural language / Unicode math
to proper LaTeX notation.
"""

import os
import json
import re
from typing import Optional
from pydantic import BaseModel, Field


class LatexEquationRendererToolArgs(BaseModel):
    equation: str = Field(
        ...,
        description=(
            "Math equation to render — can be natural language like 'quadratic formula' "
            "or LaTeX like '\\\\frac{-b \\\\pm \\\\sqrt{b^2-4ac}}{2a}'. "
            "The tool will auto-convert to proper LaTeX if needed."
        ),
    )
    font_size: Optional[int] = Field(
        20,
        description="Font size for the equation (default 20, range 10-60).",
    )
    dpi: Optional[int] = Field(
        200,
        description="Image resolution in DPI (default 200).",
    )
    color: Optional[str] = Field(
        "black",
        description="Text color: 'black', 'white', 'blue', 'red', etc.",
    )
    bg_color: Optional[str] = Field(
        "white",
        description="Background color: 'white', 'transparent', 'black', etc.",
    )
    output_path: Optional[str] = Field(
        None,
        description="Output file path for the PNG image.",
    )


class LatexEquationRendererTool:
    """
    Render LaTeX mathematical equations as clean PNG images.

    Accepts natural language descriptions (e.g. 'quadratic formula'),
    Unicode math, or raw LaTeX — automatically converts to proper LaTeX
    before rendering with matplotlib's mathtext engine.

    Features:
    - Auto-converts natural language / Unicode to LaTeX via LLM
    - Fractions, roots, integrals, sums, products
    - Greek letters, operators, accents
    - No system TeX installation required
    """

    name = "latex_equation_renderer"
    description = (
        "Render LaTeX math equations as PNG images. Input a LaTeX expression like "
        "'E = mc^2' or '\\frac{a}{b}', or natural language like 'quadratic formula'. "
        "Auto-converts to proper LaTeX. "
        "Supports fractions, integrals, Greek letters, sums, matrices, and more."
    )
    args_schema = LatexEquationRendererToolArgs

    def __init__(self, output_dir: str = "/output"):
        self.output_dir = output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            self.output_dir = "/tmp"

    def _call_llm(self, prompt: str) -> str:
        """Call Groq API directly."""
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
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    def _is_proper_latex(self, eq: str) -> bool:
        """Check if the equation already contains LaTeX commands."""
        latex_indicators = [
            "\\frac", "\\sqrt", "\\int", "\\sum", "\\prod",
            "\\alpha", "\\beta", "\\gamma", "\\theta", "\\pi",
            "\\pm", "\\cdot", "\\times", "\\infty", "\\partial",
            "\\lim", "\\log", "\\sin", "\\cos", "\\tan",
            "\\left", "\\right", "\\begin", "\\end",
            "^{", "_{",
        ]
        return any(cmd in eq for cmd in latex_indicators)

    def _ensure_latex(self, equation: str) -> str:
        """Convert natural language / Unicode math to proper LaTeX via LLM."""
        eq = equation.strip()

        # Already proper LaTeX — return as-is
        if self._is_proper_latex(eq):
            return eq

        # Use LLM to convert to LaTeX
        prompt = f"""Convert this math expression to proper LaTeX notation for matplotlib rendering.

RULES:
1. Output ONLY the raw LaTeX math expression, NO $, NO markdown fences, NO explanation.
2. Use proper LaTeX commands: \\frac{{}}{{}} for fractions, \\sqrt{{}} for roots, \\pm for ±, etc.
3. Use \\frac{{numerator}}{{denominator}} instead of inline division where appropriate.
4. The output must work with matplotlib's mathtext engine (subset of LaTeX).
5. Do NOT use \\text{{}}, \\mathrm{{}}, or \\displaystyle.
6. For the quadratic formula, output: x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}

INPUT: {eq}

LaTeX:"""

        result = self._call_llm(prompt)
        if result:
            # Clean up: remove any $ delimiters, markdown fences
            result = result.strip()
            result = result.strip("`")
            if result.startswith("```"):
                lines = result.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                result = "\n".join(lines).strip()
            result = result.strip("$").strip()

            if result:
                return result

        # Fallback: basic Unicode → LaTeX substitution
        return self._unicode_to_latex(eq)

    def _unicode_to_latex(self, eq: str) -> str:
        """Basic Unicode math → LaTeX substitution as fallback."""
        replacements = [
            ("±", "\\pm"), ("×", "\\times"), ("÷", "\\div"),
            ("√", "\\sqrt"), ("∞", "\\infty"), ("≠", "\\neq"),
            ("≤", "\\leq"), ("≥", "\\geq"), ("≈", "\\approx"),
            ("∑", "\\sum"), ("∏", "\\prod"), ("∫", "\\int"),
            ("α", "\\alpha"), ("β", "\\beta"), ("γ", "\\gamma"),
            ("δ", "\\delta"), ("θ", "\\theta"), ("λ", "\\lambda"),
            ("π", "\\pi"), ("σ", "\\sigma"), ("φ", "\\phi"),
            ("ω", "\\omega"), ("∂", "\\partial"), ("→", "\\rightarrow"),
            ("²", "^{2}"), ("³", "^{3}"), ("⁴", "^{4}"),
        ]
        for old, new in replacements:
            eq = eq.replace(old, new)

        # Try to convert sqrt(...) → \sqrt{...}
        eq = re.sub(r"sqrt\(([^)]+)\)", r"\\sqrt{\1}", eq)

        return eq

    def run(
        self,
        equation: str,
        font_size: int = 20,
        dpi: int = 200,
        color: str = "black",
        bg_color: str = "white",
        output_path: str = None,
    ) -> str:
        """Render a LaTeX equation to a PNG image.

        Args:
            equation: Math expression (natural language, Unicode, or LaTeX)
            font_size: Text size (10-60)
            dpi: Image resolution
            color: Text color
            bg_color: Background color ('transparent' for no background)
            output_path: Where to save the image

        Returns:
            JSON with success status and output path.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return json.dumps({
                "success": False,
                "error": "matplotlib not available. Install with: pip install matplotlib",
            }, indent=2)

        font_size = max(10, min(60, font_size or 20))
        dpi = max(72, min(600, dpi or 200))

        # Convert to proper LaTeX if needed
        latex_eq = self._ensure_latex(equation)

        # Wrap in $ for matplotlib mathtext
        eq = f"${latex_eq}$"

        try:
            # Create figure with transparent or colored background
            fig = plt.figure(figsize=(0.01, 0.01))

            transparent = bg_color.lower() in ("transparent", "none", "")
            if not transparent:
                fig.patch.set_facecolor(bg_color)

            # Render the equation
            text = fig.text(
                0.5, 0.5, eq,
                fontsize=font_size,
                color=color,
                ha="center",
                va="center",
                transform=fig.transFigure,
            )

            # Render once to get bounding box, then resize figure
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = text.get_window_extent(renderer=renderer)

            # Convert bbox from display to figure coords with padding
            pad = 20  # pixels
            width = (bbox.width + 2 * pad) / dpi
            height = (bbox.height + 2 * pad) / dpi
            fig.set_size_inches(max(width, 0.5), max(height, 0.3))

            # Re-render with correct size
            fig.canvas.draw()

            # Save — always under /output/ so it persists on host volume
            if output_path and not output_path.startswith(self.output_dir):
                output_path = os.path.join(self.output_dir, os.path.basename(output_path))
            if not output_path:
                existing = [f for f in os.listdir(self.output_dir) if f.startswith("equation_") and f.endswith(".png")]
                idx = len(existing) + 1
                output_path = os.path.join(self.output_dir, f"equation_{idx}.png")

            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

            fig.savefig(
                output_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.1,
                transparent=transparent,
                facecolor=fig.get_facecolor() if not transparent else "none",
            )
            plt.close(fig)

            file_size = os.path.getsize(output_path)

            return json.dumps({
                "success": True,
                "equation_input": equation,
                "equation_latex": latex_eq,
                "output_path": output_path,
                "dpi": dpi,
                "file_size_bytes": file_size,
                "message": f"Equation rendered and saved to {output_path}",
            }, indent=2)

        except Exception as e:
            plt.close("all")
            return json.dumps({
                "success": False,
                "error": str(e),
                "equation_input": equation,
                "equation_latex": latex_eq if 'latex_eq' in dir() else equation,
                "hint": "If the equation fails, try providing raw LaTeX like: \\frac{a}{b}",
            }, indent=2)
