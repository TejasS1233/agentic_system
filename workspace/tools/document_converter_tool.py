"""
Document Converter Tool - Transform papers between PDF, LaTeX, and Markdown.

Reads input files from /inputs, converts between formats, and saves to /output.
Supported conversions:
  PDF  -> Markdown, LaTeX
  Markdown -> PDF, LaTeX
  LaTeX -> Markdown, PDF

Uses lightweight pure-Python libraries (pymupdf, fpdf2, markdown) — no system TeX needed.
"""

import os
import re
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class DocumentConverterToolArgs(BaseModel):
    input_path: str = Field(
        ...,
        description=(
            "Path to the input document file. "
            "E.g. '/inputs/paper.pdf', '/inputs/notes.md', '/inputs/report.tex'"
        ),
    )
    output_format: str = Field(
        ...,
        description=(
            "Desired output format: 'pdf', 'markdown' (or 'md'), 'latex' (or 'tex')"
        ),
    )
    output_path: Optional[str] = Field(
        None,
        description="Output file path. Auto-generated in /output if not provided.",
    )


class DocumentConverterTool:
    """
    Convert documents between PDF, LaTeX, and Markdown formats.

    Reads files from /inputs, converts to the desired format, saves to /output.
    All conversions are done with pure-Python libraries that work
    inside a Docker container with no system-level TeX installation.

    Supported conversions:
      - PDF  → Markdown  (via pymupdf text extraction with structure)
      - PDF  → LaTeX     (via pymupdf text extraction + LaTeX wrapping)
      - Markdown → PDF   (via markdown + fpdf2)
      - Markdown → LaTeX (via regex-based MD→LaTeX conversion)
      - LaTeX → Markdown (via regex-based LaTeX→MD conversion)
      - LaTeX → PDF      (via fpdf2 with LaTeX text extraction)
    """

    name = "document_converter"
    description = (
        "Convert documents between PDF, LaTeX, and Markdown formats. "
        "Place input files in the inputs folder. Supports: "
        "PDF to Markdown, PDF to LaTeX, Markdown to PDF, Markdown to LaTeX, "
        "LaTeX to PDF, LaTeX to Markdown. Outputs are saved to the output folder."
    )
    args_schema = DocumentConverterToolArgs

    def __init__(self, output_dir: str = "/output"):
        self.output_dir = output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            self.output_dir = "/tmp"

    # ── Format detection ─────────────────────────────────────────────

    def _detect_input_format(self, filepath: str) -> str:
        ext = Path(filepath).suffix.lower()
        fmt_map = {
            ".pdf": "pdf",
            ".md": "markdown",
            ".markdown": "markdown",
            ".tex": "latex",
            ".latex": "latex",
            ".txt": "markdown",  # treat plain text as markdown
        }
        return fmt_map.get(ext, "unknown")

    def _normalize_format(self, fmt: str) -> str:
        fmt = fmt.lower().strip()
        aliases = {
            "md": "markdown",
            "markdown": "markdown",
            "tex": "latex",
            "latex": "latex",
            "pdf": "pdf",
        }
        return aliases.get(fmt, fmt)

    # ── PDF → Markdown ───────────────────────────────────────────────

    def _pdf_to_markdown(self, input_path: str) -> str:
        import fitz  # pymupdf

        doc = fitz.open(input_path)
        md_parts = []

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_lines = []

            for block in blocks:
                if block["type"] == 0:  # text block
                    for line in block["lines"]:
                        text = ""
                        max_size = 0
                        is_bold = False
                        for span in line["spans"]:
                            text += span["text"]
                            max_size = max(max_size, span["size"])
                            if "bold" in span["font"].lower():
                                is_bold = True

                        text = text.strip()
                        if not text:
                            continue

                        # Heuristic heading detection by font size
                        if max_size >= 18:
                            text = f"# {text}"
                        elif max_size >= 15:
                            text = f"## {text}"
                        elif max_size >= 13:
                            text = f"### {text}"
                        elif is_bold:
                            text = f"**{text}**"

                        page_lines.append(text)

            if page_lines:
                md_parts.append("\n".join(page_lines))

            # Page separator
            if page_num < len(doc) - 1:
                md_parts.append("\n---\n")

        doc.close()
        return "\n\n".join(md_parts)

    # ── PDF → LaTeX ──────────────────────────────────────────────────

    def _pdf_to_latex(self, input_path: str) -> str:
        # First get markdown, then convert to LaTeX
        md_content = self._pdf_to_markdown(input_path)
        return self._markdown_to_latex(md_content)

    # ── Markdown → LaTeX ─────────────────────────────────────────────

    def _markdown_to_latex(self, md_content: str) -> str:
        """Convert Markdown text to LaTeX document."""
        lines = md_content.split("\n")
        latex_lines = []

        # Preamble
        latex_lines.append("\\documentclass{article}")
        latex_lines.append("\\usepackage[utf8]{inputenc}")
        latex_lines.append("\\usepackage{hyperref}")
        latex_lines.append("\\usepackage{graphicx}")
        latex_lines.append("\\usepackage{amsmath}")
        latex_lines.append("\\usepackage{listings}")
        latex_lines.append("")

        # Try to extract title from first heading
        title = "Converted Document"
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        latex_lines.append(f"\\title{{{self._escape_latex(title)}}}")
        latex_lines.append("\\date{}")
        latex_lines.append("")
        latex_lines.append("\\begin{document}")
        latex_lines.append("\\maketitle")
        latex_lines.append("")

        in_code_block = False
        in_list = False

        for line in lines:
            # Code blocks
            if line.strip().startswith("```"):
                if in_code_block:
                    latex_lines.append("\\end{lstlisting}")
                    in_code_block = False
                else:
                    lang = line.strip()[3:].strip()
                    if lang:
                        latex_lines.append(f"\\begin{{lstlisting}}[language={lang}]")
                    else:
                        latex_lines.append("\\begin{lstlisting}")
                    in_code_block = True
                continue

            if in_code_block:
                latex_lines.append(line)
                continue

            # Headings
            if line.startswith("### "):
                latex_lines.append(
                    f"\\subsubsection{{{self._escape_latex(line[4:].strip())}}}"
                )
                continue
            elif line.startswith("## "):
                latex_lines.append(
                    f"\\subsection{{{self._escape_latex(line[3:].strip())}}}"
                )
                continue
            elif line.startswith("# "):
                latex_lines.append(
                    f"\\section{{{self._escape_latex(line[2:].strip())}}}"
                )
                continue

            # Horizontal rule
            if line.strip() in ("---", "***", "___"):
                latex_lines.append("\\hrulefill")
                latex_lines.append("")
                continue

            # Unordered lists
            if re.match(r"^\s*[-*+]\s", line):
                if not in_list:
                    latex_lines.append("\\begin{itemize}")
                    in_list = True
                item_text = re.sub(r"^\s*[-*+]\s", "", line).strip()
                latex_lines.append(f"  \\item {self._convert_inline(item_text)}")
                continue

            # Ordered lists
            m = re.match(r"^\s*(\d+)\.\s", line)
            if m:
                if not in_list:
                    latex_lines.append("\\begin{enumerate}")
                    in_list = True
                item_text = re.sub(r"^\s*\d+\.\s", "", line).strip()
                latex_lines.append(f"  \\item {self._convert_inline(item_text)}")
                continue

            # End list if we're no longer in a list item
            if in_list and line.strip() == "":
                # Check if the list type is itemize or enumerate
                for prev in reversed(latex_lines):
                    if "\\begin{itemize}" in prev:
                        latex_lines.append("\\end{itemize}")
                        break
                    elif "\\begin{enumerate}" in prev:
                        latex_lines.append("\\end{enumerate}")
                        break
                in_list = False
                latex_lines.append("")
                continue

            # Empty lines
            if line.strip() == "":
                if in_list:
                    continue
                latex_lines.append("")
                continue

            # Regular paragraph
            latex_lines.append(self._convert_inline(line))

        # Close any open list
        if in_list:
            for prev in reversed(latex_lines):
                if "\\begin{itemize}" in prev:
                    latex_lines.append("\\end{itemize}")
                    break
                elif "\\begin{enumerate}" in prev:
                    latex_lines.append("\\end{enumerate}")
                    break

        latex_lines.append("")
        latex_lines.append("\\end{document}")

        return "\n".join(latex_lines)

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        # Remove markdown formatting first
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)

        specials = {
            "&": "\\&",
            "%": "\\%",
            "$": "\\$",
            "#": "\\#",
            "_": "\\_",
            "{": "\\{",
            "}": "\\}",
            "~": "\\textasciitilde{}",
            "^": "\\textasciicircum{}",
        }
        for char, replacement in specials.items():
            text = text.replace(char, replacement)
        return text

    def _convert_inline(self, text: str) -> str:
        """Convert inline Markdown formatting to LaTeX."""
        # Bold
        text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
        # Italic
        text = re.sub(r"\*(.+?)\*", r"\\textit{\1}", text)
        # Inline code
        text = re.sub(r"`(.+?)`", r"\\texttt{\1}", text)
        # Links
        text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\\href{\2}{\1}", text)
        # Escape remaining specials (but not already escaped ones)
        for char in ["&", "%", "$", "#", "_"]:
            text = text.replace(char, f"\\{char}")
        return text

    # ── LaTeX → Markdown ─────────────────────────────────────────────

    def _latex_to_markdown(self, latex_content: str) -> str:
        """Convert LaTeX source to Markdown."""
        text = latex_content

        # Remove preamble (everything before \begin{document})
        doc_match = re.search(r"\\begin\{document\}", text)
        if doc_match:
            text = text[doc_match.end() :]

        # Remove \end{document}
        text = re.sub(r"\\end\{document\}", "", text)

        # Remove \maketitle
        text = re.sub(r"\\maketitle", "", text)

        # Extract title
        title_match = re.search(r"\\title\{(.+?)\}", latex_content)
        title = title_match.group(1) if title_match else ""

        md_parts = []
        if title:
            md_parts.append(f"# {title}\n")

        # Sections
        text = re.sub(r"\\section\{(.+?)\}", r"\n## \1\n", text)
        text = re.sub(r"\\subsection\{(.+?)\}", r"\n### \1\n", text)
        text = re.sub(r"\\subsubsection\{(.+?)\}", r"\n#### \1\n", text)

        # Formatting
        text = re.sub(r"\\textbf\{(.+?)\}", r"**\1**", text)
        text = re.sub(r"\\textit\{(.+?)\}", r"*\1*", text)
        text = re.sub(r"\\emph\{(.+?)\}", r"*\1*", text)
        text = re.sub(r"\\texttt\{(.+?)\}", r"`\1`", text)

        # Links
        text = re.sub(r"\\href\{(.+?)\}\{(.+?)\}", r"[\2](\1)", text)
        text = re.sub(r"\\url\{(.+?)\}", r"[\1](\1)", text)

        # Lists
        text = re.sub(r"\\begin\{itemize\}", "", text)
        text = re.sub(r"\\end\{itemize\}", "", text)
        text = re.sub(r"\\begin\{enumerate\}", "", text)
        text = re.sub(r"\\end\{enumerate\}", "", text)
        text = re.sub(r"\\item\s*", "- ", text)

        # Code blocks
        text = re.sub(r"\\begin\{lstlisting\}(?:\[.*?\])?", "```", text)
        text = re.sub(r"\\end\{lstlisting\}", "```", text)
        text = re.sub(r"\\begin\{verbatim\}", "```", text)
        text = re.sub(r"\\end\{verbatim\}", "```", text)

        # Math
        text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
        text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text)

        # Horizontal rules
        text = re.sub(r"\\hrulefill", "---", text)

        # Remove remaining LaTeX commands
        text = re.sub(r"\\[a-zA-Z]+\{(.+?)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+", "", text)

        # Unescape LaTeX specials
        for escaped, char in [
            ("\\&", "&"),
            ("\\%", "%"),
            ("\\$", "$"),
            ("\\#", "#"),
            ("\\_", "_"),
            ("\\{", "{"),
            ("\\}", "}"),
        ]:
            text = text.replace(escaped, char)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        md_parts.append(text)
        return "\n".join(md_parts)

    # ── Markdown → PDF ───────────────────────────────────────────────

    def _markdown_to_pdf(self, md_content: str, output_path: str) -> None:
        """Convert Markdown to PDF using fpdf2."""
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        lines = md_content.split("\n")
        in_code = False

        for line in lines:
            # Code block toggles
            if line.strip().startswith("```"):
                in_code = not in_code
                if in_code:
                    pdf.set_font("Courier", size=9)
                else:
                    pdf.set_font("Helvetica", size=11)
                continue

            if in_code:
                pdf.set_font("Courier", size=9)
                pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
                continue

            # Headings
            if line.startswith("### "):
                pdf.set_font("Helvetica", "B", 13)
                pdf.cell(0, 8, line[4:].strip(), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
                continue
            elif line.startswith("## "):
                pdf.set_font("Helvetica", "B", 15)
                pdf.cell(0, 10, line[3:].strip(), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(3)
                continue
            elif line.startswith("# "):
                pdf.set_font("Helvetica", "B", 18)
                pdf.cell(0, 12, line[2:].strip(), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)
                continue

            # Horizontal rule
            if line.strip() in ("---", "***", "___"):
                pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
                pdf.ln(5)
                continue

            # Empty line
            if line.strip() == "":
                pdf.ln(4)
                continue

            # Regular text (strip basic markdown formatting)
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            clean = re.sub(r"\*(.+?)\*", r"\1", clean)
            clean = re.sub(r"`(.+?)`", r"\1", clean)
            clean = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", clean)

            # Handle list items
            if re.match(r"^\s*[-*+]\s", clean):
                clean = "  • " + re.sub(r"^\s*[-*+]\s", "", clean)
            elif re.match(r"^\s*\d+\.\s", clean):
                pass  # keep numbered lists as-is

            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, clean)

        pdf.output(output_path)

    # ── LaTeX → PDF ──────────────────────────────────────────────────

    def _latex_to_pdf(self, latex_content: str, output_path: str) -> None:
        """Convert LaTeX to PDF by first converting to Markdown, then to PDF."""
        md = self._latex_to_markdown(latex_content)
        self._markdown_to_pdf(md, output_path)

    # ── Main run method ──────────────────────────────────────────────

    def run(
        self,
        input_path: str,
        output_format: str,
        output_path: str = None,
    ) -> str:
        """Convert a document from one format to another.

        Args:
            input_path: Path to input file (e.g. /inputs/paper.pdf)
            output_format: Target format: 'pdf', 'markdown'/'md', 'latex'/'tex'
            output_path: Optional output file path

        Returns:
            JSON with success status, output path, and conversion details.
        """
        # Validate input
        if not os.path.exists(input_path):
            # Try common locations
            for prefix in ["/inputs", "/output", "/tools"]:
                candidate = os.path.join(prefix, os.path.basename(input_path))
                if os.path.exists(candidate):
                    input_path = candidate
                    break
            else:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Input file not found: {input_path}",
                    },
                    indent=2,
                )

        input_format = self._detect_input_format(input_path)
        output_format = self._normalize_format(output_format)

        if input_format == "unknown":
            return json.dumps(
                {
                    "success": False,
                    "error": f"Cannot detect format of: {input_path}. Supported: .pdf, .md, .tex",
                },
                indent=2,
            )

        if output_format not in ("pdf", "markdown", "latex"):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Unsupported output format: {output_format}. Use 'pdf', 'markdown', or 'latex'.",
                },
                indent=2,
            )

        if input_format == output_format:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Input and output formats are the same: {input_format}",
                },
                indent=2,
            )

        # Determine output path
        ext_map = {"pdf": ".pdf", "markdown": ".md", "latex": ".tex"}
        if not output_path:
            stem = Path(input_path).stem
            output_path = os.path.join(
                self.output_dir, f"{stem}_converted{ext_map[output_format]}"
            )

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        try:
            # Read input content (for text formats)
            if input_format in ("markdown", "latex"):
                with open(input_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

            # ── Route to correct converter ──
            conversion = f"{input_format}→{output_format}"

            if input_format == "pdf" and output_format == "markdown":
                result_text = self._pdf_to_markdown(input_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result_text)

            elif input_format == "pdf" and output_format == "latex":
                result_text = self._pdf_to_latex(input_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result_text)

            elif input_format == "markdown" and output_format == "latex":
                result_text = self._markdown_to_latex(content)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result_text)

            elif input_format == "markdown" and output_format == "pdf":
                self._markdown_to_pdf(content, output_path)

            elif input_format == "latex" and output_format == "markdown":
                result_text = self._latex_to_markdown(content)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result_text)

            elif input_format == "latex" and output_format == "pdf":
                self._latex_to_pdf(content, output_path)

            else:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Unsupported conversion: {conversion}",
                    },
                    indent=2,
                )

            # Get output file size
            file_size = os.path.getsize(output_path)

            return json.dumps(
                {
                    "success": True,
                    "conversion": conversion,
                    "input_path": input_path,
                    "input_format": input_format,
                    "output_format": output_format,
                    "output_path": output_path,
                    "file_size_bytes": file_size,
                    "message": f"Successfully converted {input_format} to {output_format}. Saved to {output_path}",
                },
                indent=2,
            )

        except ImportError as e:
            pkg = str(e).split("'")[-2] if "'" in str(e) else str(e)
            return json.dumps(
                {
                    "success": False,
                    "error": f"Missing dependency: {pkg}. Install with: pip install {pkg}",
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Conversion failed: {str(e)}",
                },
                indent=2,
            )
