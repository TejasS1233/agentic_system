"""SERP API Search Tool - Google search results via SerpAPI."""

import os
import requests
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class SerpSearchArgs(BaseModel):
    query: str = Field(..., description="The search query")
    search_type: Literal["search", "images", "news", "shopping"] = Field(
        "search",
        description="Type of Google search. MUST be one of: 'search' (for web results), 'images', 'news', or 'shopping'. Default is 'search'.",
    )
    num_results: int = Field(10, description="Number of results to return (max 100)")
    location: Optional[str] = Field(
        None, description="Location for localized results (e.g., 'New York, NY')"
    )
    language: str = Field("en", description="Language code (e.g., 'en', 'es', 'fr')")
    safe_search: bool = Field(True, description="Enable safe search filtering")


class SerpSearchTool:
    """
    Google Search Tool using SerpAPI.

    Provides access to Google search results including:
    - Organic web results
    - Image search
    - News articles
    - Shopping results
    - Knowledge panels and featured snippets
    """

    name = "serp_search"
    description = """Perform Google searches using SerpAPI. Supports web search, images, news, and shopping. 
    Returns structured results including titles, links, snippets, and rich data like knowledge panels."""
    args_schema = SerpSearchArgs

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERP_API_KEY")
        self.base_url = "https://serpapi.com/search"

    def run(
        self,
        query: str,
        search_type: str = "search",
        num_results: int = 10,
        location: Optional[str] = None,
        language: str = "en",
        safe_search: bool = True,
    ) -> Dict[str, Any]:
        """Execute a SERP search and return structured results."""

        if not self.api_key:
            return {
                "error": "SERP_API_KEY not found. Please set it in your environment variables."
            }

        # Normalize search_type - handle None and common LLM mistakes
        if search_type is None:
            search_type = "search"
        else:
            search_type = search_type.lower().strip()
            if search_type in ("google", "web", "search", ""):
                search_type = "search"
            elif search_type not in ("images", "news", "shopping"):
                search_type = "search"  # Default fallback

        # Build params based on search type
        params = {
            "api_key": self.api_key,
            "q": query,
            "hl": language,
            "num": min(num_results, 100),
            "safe": "active" if safe_search else "off",
        }

        # Set engine based on search type
        if search_type == "images":
            params["engine"] = "google_images"
            params["tbm"] = "isch"
        elif search_type == "news":
            params["engine"] = "google_news"
            params["tbm"] = "nws"
        elif search_type == "shopping":
            params["engine"] = "google_shopping"
            params["tbm"] = "shop"
        else:
            params["engine"] = "google"

        if location:
            params["location"] = location

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return self._parse_results(data, search_type)

        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Please try again."}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def _parse_results(self, data: Dict, search_type: str) -> Dict[str, Any]:
        """Parse SERP API response into a clean format."""
        result = {
            "query": data.get("search_parameters", {}).get("q", ""),
            "search_type": search_type,
            "total_results": data.get("search_information", {}).get("total_results", 0),
        }

        # Extract organic results for web search
        if search_type == "search":
            organic = data.get("organic_results", [])
            result["results"] = [
                {
                    "position": r.get("position"),
                    "title": r.get("title"),
                    "link": r.get("link"),
                    "snippet": r.get("snippet"),
                    "displayed_link": r.get("displayed_link"),
                }
                for r in organic
            ]

            # Include knowledge panel if available
            if "knowledge_graph" in data:
                kg = data["knowledge_graph"]
                result["knowledge_panel"] = {
                    "title": kg.get("title"),
                    "type": kg.get("type"),
                    "description": kg.get("description"),
                    "source": kg.get("source", {}).get("link"),
                }

            # Include featured snippet if available
            if "answer_box" in data:
                ab = data["answer_box"]
                result["featured_snippet"] = {
                    "type": ab.get("type"),
                    "answer": ab.get("answer") or ab.get("snippet"),
                    "title": ab.get("title"),
                    "link": ab.get("link"),
                }

            # Include related searches
            if "related_searches" in data:
                result["related_searches"] = [
                    r.get("query") for r in data["related_searches"]
                ]

        # Extract image results
        elif search_type == "images":
            images = data.get("images_results", [])
            result["results"] = [
                {
                    "position": img.get("position"),
                    "title": img.get("title"),
                    "link": img.get("link"),
                    "original": img.get("original"),
                    "thumbnail": img.get("thumbnail"),
                    "source": img.get("source"),
                }
                for img in images
            ]

        # Extract news results
        elif search_type == "news":
            news = data.get("news_results", [])
            result["results"] = [
                {
                    "position": n.get("position"),
                    "title": n.get("title"),
                    "link": n.get("link"),
                    "source": n.get("source"),
                    "date": n.get("date"),
                    "snippet": n.get("snippet"),
                    "thumbnail": n.get("thumbnail"),
                }
                for n in news
            ]

        # Extract shopping results
        elif search_type == "shopping":
            shopping = data.get("shopping_results", [])
            result["results"] = [
                {
                    "position": s.get("position"),
                    "title": s.get("title"),
                    "link": s.get("link"),
                    "price": s.get("price"),
                    "source": s.get("source"),
                    "rating": s.get("rating"),
                    "reviews": s.get("reviews"),
                    "thumbnail": s.get("thumbnail"),
                }
                for s in shopping
            ]

        result["count"] = len(result.get("results", []))
        return result


# For direct testing
if __name__ == "__main__":
    tool = SerpSearchTool()
    result = tool.run("Python programming tutorials", num_results=5)
    print(result)
