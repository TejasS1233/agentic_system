"""Universal Web Scraper Tool - Robust HTML parsing with BeautifulSoup."""

import requests
from typing import Literal, Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import json
import time
import random


class WebScraperArgs(BaseModel):
    url: str = Field(..., description="The URL to scrape")
    selectors: Optional[Dict[str, str]] = Field(
        None,
        description="CSS selectors to extract specific elements. Format: {'field_name': 'css_selector'}",
    )
    extract_type: Literal[
        "text", "html", "table", "links", "images", "structured", "auto"
    ] = Field(
        "auto",
        description="Type of extraction: 'text' (all text), 'html' (raw HTML), 'table' (tables as lists), "
        "'links' (all links), 'images' (all images), 'structured' (JSON-LD), 'auto' (smart extraction)",
    )
    include_metadata: bool = Field(
        True, description="Include page metadata (title, description, etc.)"
    )
    max_text_length: int = Field(
        5000, description="Maximum text length to return (0 for unlimited)"
    )


class WebScraperTool:
    """
    Universal Web Scraper using BeautifulSoup.

    Features:
    - CSS selector support for targeted extraction
    - Automatic content extraction modes
    - Table parsing to structured data
    - Link and image extraction
    - JSON-LD structured data extraction
    - Built-in rate limiting and retries
    - User-agent rotation
    """

    name = "web_scraper"
    description = """Scrape web pages and extract content. Supports CSS selectors for targeted extraction,
    automatic text/table/link extraction, and JSON-LD structured data parsing. Handles errors gracefully."""
    args_schema = WebScraperArgs

    # User agents for rotation
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    ]

    def __init__(self, timeout: int = 15, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries

    def run(
        self,
        url: str,
        selectors: Optional[Dict[str, str]] = None,
        extract_type: str = "auto",
        include_metadata: bool = True,
        max_text_length: int = 5000,
    ) -> Dict[str, Any]:
        """Scrape a web page and extract content."""

        # Validate URL
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Fetch the page
        html = self._fetch_page(url)
        if isinstance(html, dict) and "error" in html:
            return html

        # Parse with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style elements for text extraction
        for script in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            script.decompose()

        result = {"url": url, "success": True}

        # Include metadata if requested
        if include_metadata:
            result["metadata"] = self._extract_metadata(soup)

        # If selectors are provided, use them
        if selectors:
            result["extracted"] = self._extract_with_selectors(soup, selectors)

        # Extract based on type
        if extract_type == "text" or extract_type == "auto":
            text = self._extract_text(soup)
            if max_text_length > 0 and len(text) > max_text_length:
                text = text[:max_text_length] + "... (truncated)"
            result["text"] = text

        if extract_type == "html":
            result["html"] = str(soup)

        if extract_type == "table" or extract_type == "auto":
            tables = self._extract_tables(soup)
            if tables:
                result["tables"] = tables

        if extract_type == "links" or extract_type == "auto":
            result["links"] = self._extract_links(soup, url)

        if extract_type == "images":
            result["images"] = self._extract_images(soup, url)

        if extract_type == "structured" or extract_type == "auto":
            structured = self._extract_structured_data(soup)
            if structured:
                result["structured_data"] = structured

        return result

    def _fetch_page(self, url: str) -> Union[str, Dict]:
        """Fetch page with retries and user-agent rotation."""
        headers = {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.exceptions.Timeout:
                if attempt == self.max_retries:
                    return {"error": f"Request timed out after {self.timeout}s"}
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    return {"error": f"Failed to fetch page: {str(e)}"}

            # Wait before retry with exponential backoff
            time.sleep(1 * (attempt + 1))

        return {"error": "Max retries exceeded"}

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract page metadata."""
        metadata = {}

        # Title
        if soup.title:
            metadata["title"] = soup.title.string.strip() if soup.title.string else ""

        # Meta tags
        meta_mappings = {
            "description": ["description", "og:description", "twitter:description"],
            "keywords": ["keywords"],
            "author": ["author"],
            "image": ["og:image", "twitter:image"],
            "type": ["og:type"],
            "site_name": ["og:site_name"],
        }

        for key, names in meta_mappings.items():
            for name in names:
                tag = soup.find("meta", attrs={"name": name}) or soup.find(
                    "meta", attrs={"property": name}
                )
                if tag and tag.get("content"):
                    metadata[key] = tag["content"]
                    break

        # Canonical URL
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            metadata["canonical_url"] = canonical["href"]

        return metadata

    def _extract_with_selectors(
        self, soup: BeautifulSoup, selectors: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Extract content using CSS selectors."""
        extracted = {}

        for field_name, selector in selectors.items():
            try:
                elements = soup.select(selector)
                extracted[field_name] = [
                    el.get_text(strip=True)
                    for el in elements
                    if el.get_text(strip=True)
                ]
            except Exception as e:
                extracted[field_name] = [f"Error: {str(e)}"]

        return extracted

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from the page."""
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract tables as structured data."""
        tables = []

        for table in soup.find_all("table"):
            table_data = {"headers": [], "rows": []}

            # Extract headers
            headers = table.find_all("th")
            if headers:
                table_data["headers"] = [h.get_text(strip=True) for h in headers]

            # Extract rows
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if cells and not all(c.name == "th" for c in cells):
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    if any(row_data):
                        table_data["rows"].append(row_data)

            if table_data["rows"]:
                tables.append(table_data)

        return tables[:5]  # Limit to 5 tables

    def _extract_links(
        self, soup: BeautifulSoup, base_url: str
    ) -> List[Dict[str, str]]:
        """Extract all links from the page."""
        from urllib.parse import urljoin

        links = []
        seen = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]

            # Skip anchors and javascript
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            # Make absolute URL
            full_url = urljoin(base_url, href)

            if full_url not in seen:
                seen.add(full_url)
                links.append(
                    {
                        "text": a.get_text(strip=True)[:100] or "[No text]",
                        "url": full_url,
                    }
                )

        return links[:50]  # Limit to 50 links

    def _extract_images(
        self, soup: BeautifulSoup, base_url: str
    ) -> List[Dict[str, str]]:
        """Extract all images from the page."""
        from urllib.parse import urljoin

        images = []

        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if src:
                images.append(
                    {
                        "src": urljoin(base_url, src),
                        "alt": img.get("alt", ""),
                        "title": img.get("title", ""),
                    }
                )

        return images[:30]  # Limit to 30 images

    def _extract_structured_data(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract JSON-LD structured data."""
        structured = []

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                structured.append(data)
            except (json.JSONDecodeError, TypeError):
                continue

        return structured


# For direct testing
if __name__ == "__main__":
    tool = WebScraperTool()
    result = tool.run(
        "https://news.ycombinator.com/",
        selectors={"titles": ".titleline a", "scores": ".score"},
    )
    print(json.dumps(result, indent=2))
