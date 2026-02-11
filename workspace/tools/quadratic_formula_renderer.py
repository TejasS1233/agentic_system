import matplotlib.pyplot as plt
import matplotlib
from pydantic import BaseModel, Field


class QuadraticFormulaRendererArgs(BaseModel):
    equation_input: str = Field(..., description="Input equation string")
    equation_latex: str = Field(..., description="LaTeX equation string")
    output_path: str = Field(
        ..., description="Path to save the rendered equation image"
    )
    dpi: int = Field(..., description="DPI of the rendered image")


class QuadraticFormulaRenderer:
    name = "quadratic_formula_renderer"
    description = "Render a quadratic formula as an image"
    args_schema = QuadraticFormulaRendererArgs

    def run(
        self, equation_input: str, equation_latex: str, output_path: str, dpi: int
    ) -> str:
        matplotlib.rcParams["figure.dpi"] = dpi
        fig = plt.figure(figsize=(10, 2))
        plt.axis("off")
        plt.text(0.1, 0.6, f"${equation_latex}$", fontsize=20)
        plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return f"Equation rendered and saved to {output_path}"


def test_tool():
    tool = QuadraticFormulaRenderer()
    equation_input = "x = (-b \u00b1 \u221a(b\u00b2 - 4ac)) / 2a"
    equation_latex = "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"
    output_path = "/output/equation.png"
    dpi = 200
    print(tool.run(equation_input, equation_latex, output_path, dpi))


if __name__ == "__main__":
    test_tool()
