"""API Registry for Toolsmith.

Provides access to curated list of free, no-auth public APIs
and scrapable URLs that the toolsmith can use when generating tools.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


class APIRegistry:
    """Manages and searches the public API registry and scrapable URLs."""

    def __init__(self, registry_path: Optional[Path] = None, urls_path: Optional[Path] = None):
        base_path = Path(__file__).parent.parent / "public-apis"
        self.registry_path = registry_path or (base_path / "no_auth_apis.json")
        self.urls_path = urls_path or (base_path / "scrapable_urls.json")
        self._apis: List[Dict] = []
        self._urls: List[Dict] = []
        self._load_registry()
        self._load_urls()

    def _load_registry(self):
        """Load APIs from the JSON registry file."""
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._apis = data.get("apis", [])
        except Exception as e:
            print(f"Warning: Could not load API registry: {e}")
            self._apis = []

    def _load_urls(self):
        """Load scrapable URLs from the JSON file."""
        try:
            with open(self.urls_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._urls = data.get("urls", [])
        except Exception as e:
            print(f"Warning: Could not load scrapable URLs: {e}")
            self._urls = []

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for APIs matching a query.
        
        Args:
            query: Search term (matches against name and description)
            limit: Maximum number of results to return
            
        Returns:
            List of matching API entries
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_apis = []
        for api in self._apis:
            name = api.get("name", "").lower()
            endpoint = api.get("endpoint", "").lower()
            
            # Calculate relevance score
            score = 0
            
            # Exact name match
            if query_lower in name:
                score += 10
            
            # Word matches in name
            for word in query_words:
                if word in name:
                    score += 5
                if word in endpoint:
                    score += 2
            
            # Common keywords boost
            keywords = {
                "weather": ["weather", "meteo", "forecast", "temperature"],
                "crypto": ["coin", "crypto", "bitcoin", "currency", "exchange"],
                "animal": ["dog", "cat", "animal", "pet", "fox", "duck", "bear"],
                "joke": ["joke", "humor", "chuck", "dad", "quote"],
                "user": ["user", "random", "faker", "person"],
                "country": ["country", "countries", "geo", "location", "ip"],
                "movie": ["movie", "tv", "show", "film", "anime", "pokemon"],
                "food": ["food", "meal", "cocktail", "recipe", "drink"],
                "news": ["news", "hacker", "spaceflight"],
                "image": ["picsum", "placeholder", "avatar", "image", "photo"],
                "github": ["github", "repo", "repository", "commit", "contributor"],
            }
            
            for category, terms in keywords.items():
                if any(t in query_lower for t in terms):
                    if any(t in name or t in endpoint for t in terms):
                        score += 8
            
            if score > 0:
                scored_apis.append((score, api))
        
        # Sort by score descending
        scored_apis.sort(key=lambda x: x[0], reverse=True)
        
        return [api for _, api in scored_apis[:limit]]

    def search_urls(self, query: str, limit: int = 3) -> List[Dict]:
        """Search for scrapable URLs matching a query.
        
        Args:
            query: Search term
            limit: Maximum number of results to return
            
        Returns:
            List of matching URL entries
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_urls = []
        for url_entry in self._urls:
            name = url_entry.get("name", "").lower()
            description = url_entry.get("description", "").lower()
            url = url_entry.get("url", "").lower()
            
            score = 0
            
            if query_lower in name:
                score += 10
            if query_lower in description:
                score += 5
            
            for word in query_words:
                if word in name:
                    score += 5
                if word in description:
                    score += 3
                if word in url:
                    score += 2
            
            # Boost for common scraping terms
            scrape_keywords = {
                "trending": ["trending", "popular", "top", "hot"],
                "news": ["news", "articles", "posts", "stories"],
                "jobs": ["jobs", "hiring", "careers", "work"],
                "github": ["github", "repos", "developers"],
                "reddit": ["reddit", "subreddit"],
                "stackoverflow": ["stackoverflow", "questions", "programming"],
                "research": ["arxiv", "papers", "research", "academic"],
                "finance": ["crypto", "stocks", "finance", "market", "price"],
            }
            
            for category, terms in scrape_keywords.items():
                if any(t in query_lower for t in terms):
                    if any(t in name or t in description for t in terms):
                        score += 8
            
            if score > 0:
                scored_urls.append((score, url_entry))
        
        scored_urls.sort(key=lambda x: x[0], reverse=True)
        return [url for _, url in scored_urls[:limit]]

    def get_all(self) -> List[Dict]:
        """Get all APIs in the registry."""
        return self._apis.copy()

    def get_all_urls(self) -> List[Dict]:
        """Get all scrapable URLs."""
        return self._urls.copy()

    def format_for_prompt(self, apis: List[Dict]) -> str:
        """Format API entries for injection into LLM prompt."""
        if not apis:
            return ""
        
        lines = []
        for api in apis:
            name = api.get("name", "Unknown")
            endpoint = api.get("endpoint", "")
            method = api.get("method", "GET")
            returns = api.get("returns", "json")
            
            line = f"- **{name}**: `{method} {endpoint}` → returns {returns}"
            lines.append(line)
        
        return "\n".join(lines)

    def format_urls_for_prompt(self, urls: List[Dict]) -> str:
        """Format scrapable URLs for injection into LLM prompt."""
        if not urls:
            return ""
        
        lines = []
        for url_entry in urls:
            name = url_entry.get("name", "Unknown")
            url = url_entry.get("url", "")
            description = url_entry.get("description", "")
            selectors = url_entry.get("selectors", {})
            
            line = f"- **{name}**: `{url}`"
            if description:
                line += f" - {description}"
            if selectors:
                selector_str = ", ".join([f"{k}: '{v}'" for k, v in list(selectors.items())[:2]])
                line += f" (selectors: {selector_str})"
            lines.append(line)
        
        return "\n".join(lines)


# Singleton instance
_registry: Optional[APIRegistry] = None


def get_api_registry() -> APIRegistry:
    """Get the global API registry instance."""
    global _registry
    if _registry is None:
        _registry = APIRegistry()
    return _registry


def search_apis(query: str, limit: int = 5) -> List[Dict]:
    """Search for APIs matching a query."""
    return get_api_registry().search(query, limit)


def search_urls(query: str, limit: int = 3) -> List[Dict]:
    """Search for scrapable URLs matching a query."""
    return get_api_registry().search_urls(query, limit)


def format_apis_for_prompt(query: str, limit: int = 5) -> str:
    """Search and format APIs for prompt injection."""
    registry = get_api_registry()
    apis = registry.search(query, limit)
    return registry.format_for_prompt(apis)


def format_all_sources_for_prompt(query: str, api_limit: int = 4, url_limit: int = 2) -> str:
    """Search and format both APIs and scrapable URLs for prompt injection.
    
    Args:
        query: Search term
        api_limit: Max APIs to include
        url_limit: Max scrapable URLs to include
        
    Returns:
        Combined formatted string for prompt injection
    """
    registry = get_api_registry()
    
    # Get matching APIs
    apis = registry.search(query, api_limit)
    api_text = registry.format_for_prompt(apis)
    
    # Get matching scrapable URLs
    urls = registry.search_urls(query, url_limit)
    url_text = registry.format_urls_for_prompt(urls)
    
    sections = []
    if api_text:
        sections.append(f"**Direct APIs (no auth):**\n{api_text}")
    if url_text:
        sections.append(f"**Scrapable URLs (use BeautifulSoup/requests):**\n{url_text}")
    
    return "\n\n".join(sections) if sections else ""


if __name__ == "__main__":
    # Test the registry
    print("Testing API Registry...")
    
    test_queries = ["github", "trending", "news", "weather"]
    
    for query in test_queries:
        print(f"\n🔍 Searching: '{query}'")
        print(format_all_sources_for_prompt(query, api_limit=2, url_limit=2))

