"""Universal Website Reader Tool - Converting URLs to clean Markdown for Agents."""

import requests
import json
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field


class UniversalWebsiteReaderArgs(BaseModel):
    url: str = Field(..., description="The URL to read and convert to markdown")
    method: Literal["auto", "ai", "browser"] = Field(
        "auto",
        description="Conversion method: 'auto' (fastest), 'ai' (workers AI fallback), 'browser' (for JS-heavy sites)",
    )
    retain_images: bool = Field(
        False, description="Whether to retain images in the markdown output"
    )


class UniversalWebsiteReaderTool:
    """
    Universal Website Reader that converts any URL to clean Markdown.
    Powered by markdown.new - optimized for AI agent consumption.
    """

    name = "universal_website_reader"
    description = "Read static websites or articles to get basic text content as Markdown. DO NOT use this for API endpoints, JSON data (.json), interactive/SPA sites (like Reddit/Twitter), or robust data extraction. For APIs or complex sites, returning NONE is preferred so a dedicated tool can be forged."
    args_schema = UniversalWebsiteReaderArgs

    def __init__(self, timeout: int = 45):
        self.timeout = timeout
        self.base_api_url = "https://markdown.new/"
        self.max_content_chars = 50_000

    def run(
        self, url: str, method: str = "auto", retain_images: bool = False
    ) -> Dict[str, Any]:
        """Convert a URL to markdown using markdown.new"""

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            payload = {"url": url, "method": method, "retain_images": retain_images}

            headers = {
                "Content-Type": "application/json",
                "Accept": "text/markdown",
                "User-Agent": "IASCISv2-Agent-Reader",
            }

            response = requests.post(
                self.base_api_url, json=payload, headers=headers, timeout=self.timeout
            )

            response.raise_for_status()

            # markdown.new may return JSON envelope or raw markdown
            raw = response.text
            markdown_content = raw
            estimated_tokens = None

            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    markdown_content = parsed.get(
                        "content", parsed.get("markdown", raw)
                    )
                    estimated_tokens = parsed.get("tokens")
                    if not estimated_tokens:
                        token_header = response.headers.get("x-markdown-tokens")
                        estimated_tokens = (
                            int(token_header)
                            if token_header and token_header.isdigit()
                            else None
                        )
            except (json.JSONDecodeError, TypeError):
                token_header = response.headers.get("x-markdown-tokens")
                estimated_tokens = (
                    int(token_header)
                    if token_header and token_header.isdigit()
                    else None
                )

            # Truncate extremely large content to prevent LLM context overflow
            if len(markdown_content) > self.max_content_chars:
                markdown_content = (
                    markdown_content[: self.max_content_chars]
                    + f"\n\n[...TRUNCATED at {self.max_content_chars} chars]"
                )

            return {
                "url": url,
                "success": True,
                "markdown": markdown_content,
                "estimated_tokens": estimated_tokens,
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Request timed out after {self.timeout}s reading {url}",
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                return {
                    "success": False,
                    "error": "Rate limit exceeded (500 requests per day per IP)",
                }
            return {
                "success": False,
                "error": f"HTTP Error {e.response.status_code}: {e.response.text}",
            }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Failed to fetch page: {str(e)}"}


def test_tool():
    tool = UniversalWebsiteReaderTool()
    print("Testing universal_website_reader...")
    result = tool.run("https://blog.cloudflare.com/markdown-for-agents/")
    print(
        json.dumps(
            {
                "success": result.get("success"),
                "url": result.get("url"),
                "estimated_tokens": result.get("estimated_tokens"),
                "content_preview": result.get("markdown", "")[:200] + "..."
                if result.get("markdown")
                else "",
                "error": result.get("error"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    test_tool()
