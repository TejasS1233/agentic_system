import re
from pydantic import BaseModel, Field


class TextToMarkdownArgs(BaseModel):
    text: str = Field(..., description="Text to convert to Markdown")


class TextToMarkdownTool:
    name = "text_to_markdown"
    description = "Convert extracted text to Markdown format"
    args_schema = TextToMarkdownArgs

    def run(self, text: str) -> str:
        # Convert headers
        text = re.sub(
            r"^(#+) (.*)$",
            lambda m: m.group(1) + " " + m.group(2),
            text,
            flags=re.MULTILINE,
        )

        # Convert bold text
        text = re.sub(r"\*(.*?)\*", r"**\1**", text)

        # Convert italic text
        text = re.sub(r"_(.*?)_", r"*\1*", text)

        # Convert links
        text = re.sub(r"\[(.*?)\]\((.*?)\)", r"[\1](\2)", text)

        return text


def test_tool():
    tool = TextToMarkdownTool()
    text = "# Header\n* Item 1\n* Item 2\nThis is a [link](https://www.example.com)."
    print(tool.run(text))


if __name__ == "__main__":
    test_tool()
