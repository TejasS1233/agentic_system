import pandas as pd
import re
import io
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class TableParserToolArgs(BaseModel):
    content: str = Field(
        ..., description="Content containing tables (HTML or plain text from PDF)"
    )


class TableParserTool:
    name = "parse_table"
    description = "Parse and structure tables from HTML or plain text (PDF) content"
    args_schema = TableParserToolArgs

    def run(self, content: str = None, html_content: str = None) -> str:
        """Parse tables from either HTML or plain text content."""
        # Support both argument names for backwards compatibility
        text = content or html_content
        if not text:
            return "Error: No content provided"

        # Detect if content is HTML or plain text
        if self._is_html(text):
            return self._parse_html_tables(text)
        else:
            return self._parse_text_tables(text)

    def _is_html(self, text: str) -> bool:
        """Check if the content appears to be HTML."""
        html_patterns = ["<table", "<tr", "<td", "<th", "</table>"]
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in html_patterns)

    def _parse_html_tables(self, html_content: str) -> str:
        """Parse tables from HTML content."""
        soup = BeautifulSoup(html_content, "html.parser")
        tables = soup.find_all("table")
        parsed_tables = []
        for table in tables:
            try:
                # Use StringIO to prevent pandas from treating HTML as a file path
                df = pd.read_html(io.StringIO(str(table)))
                parsed_tables.append(df[0].to_dict(orient="records"))
            except Exception as e:
                parsed_tables.append({"error": str(e)})
        return str(parsed_tables) if parsed_tables else "No HTML tables found"

    def _parse_text_tables(self, text: str) -> str:
        """Parse tables from plain text (e.g., from PDF extraction)."""
        parsed_tables = []

        # Strategy 1: Look for delimiter-separated tables (tabs, pipes, multiple spaces)
        lines = text.strip().split("\n")

        # Try to find table-like structures
        table_blocks = self._find_table_blocks(lines)

        for block in table_blocks:
            table_data = self._parse_table_block(block)
            if table_data:
                parsed_tables.append(table_data)

        if parsed_tables:
            return str(parsed_tables)

        # Fallback: Try to extract any structured data
        structured = self._extract_key_value_pairs(text)
        if structured:
            return str(structured)

        return "No tabular data detected in plain text content"

    def _find_table_blocks(self, lines: List[str]) -> List[List[str]]:
        """Find contiguous blocks of lines that look like table rows."""
        blocks = []
        current_block = []

        for line in lines:
            line = line.strip()
            if not line:
                if len(current_block) >= 2:  # At least header + 1 row
                    blocks.append(current_block)
                current_block = []
            elif self._is_table_row(line):
                current_block.append(line)
            else:
                if len(current_block) >= 2:
                    blocks.append(current_block)
                current_block = []

        if len(current_block) >= 2:
            blocks.append(current_block)

        return blocks

    def _is_table_row(self, line: str) -> bool:
        """Check if a line looks like a table row."""
        # Has multiple tab-separated values
        if "\t" in line and len(line.split("\t")) >= 2:
            return True
        # Has pipe separators
        if "|" in line and len(line.split("|")) >= 3:
            return True
        # Has multiple space-separated columns (at least 2 gaps of 2+ spaces)
        if len(re.findall(r"\s{2,}", line)) >= 1:
            return True
        # Has comma-separated values
        if "," in line and len(line.split(",")) >= 2:
            return True
        return False

    def _parse_table_block(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse a block of lines into table records."""
        if not lines:
            return []

        # Determine delimiter
        first_line = lines[0]
        if "\t" in first_line:
            delimiter = "\t"
        elif "|" in first_line:
            delimiter = "|"
        elif "," in first_line:
            delimiter = ","
        else:
            # Multiple spaces
            delimiter = None

        rows = []
        for line in lines:
            if delimiter:
                cells = [c.strip() for c in line.split(delimiter) if c.strip()]
            else:
                cells = re.split(r"\s{2,}", line.strip())
            rows.append(cells)

        if len(rows) < 2:
            return []

        # First row is header
        headers = rows[0]
        records = []

        for row in rows[1:]:
            # Skip separator lines (like "---" or "===")
            if all(re.match(r"^[-=_]+$", cell) for cell in row if cell):
                continue

            record = {}
            for i, header in enumerate(headers):
                if i < len(row):
                    record[header] = row[i]
                else:
                    record[header] = ""
            if record:
                records.append(record)

        return records

    def _extract_key_value_pairs(self, text: str) -> List[Dict[str, str]]:
        """Extract key-value pairs from text (fallback method)."""
        patterns = [
            r"([A-Za-z][A-Za-z\s]+):\s*([^\n]+)",  # "Key: Value"
            r"([A-Za-z][A-Za-z\s]+)\s*=\s*([^\n]+)",  # "Key = Value"
        ]

        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                results.append({key.strip(): value.strip()})

        return results if results else None


def test_tool():
    tool = TableParserTool()

    # Test HTML parsing
    print("=== HTML Table Test ===")
    html_content = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>John</td><td>30</td></tr></table>"
    print(tool.run(html_content=html_content))

    # Test plain text table parsing (tab-separated)
    print("\n=== Tab-Separated Text Table Test ===")
    text_content = """Name\tAge\tCity
John\t30\tNew York
Jane\t25\tLos Angeles
Bob\t35\tChicago"""
    print(tool.run(content=text_content))

    # Test plain text table parsing (space-separated)
    print("\n=== Space-Separated Text Table Test ===")
    space_content = """Name       Age    City
John       30     New York
Jane       25     Los Angeles"""
    print(tool.run(content=space_content))

    # Test pipe-separated (markdown-style)
    print("\n=== Pipe-Separated Text Table Test ===")
    pipe_content = """| Name | Age | City |
| John | 30 | New York |
| Jane | 25 | Los Angeles |"""
    print(tool.run(content=pipe_content))


if __name__ == "__main__":
    test_tool()
