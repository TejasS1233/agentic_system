"""Reddit Tool - Fetch posts from Reddit using public JSON API."""

import requests
import time
from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field


class RedditArgs(BaseModel):
    subreddit: Union[str, List[str]] = Field(
        ..., description="Subreddit name (without r/) or list of subreddit names"
    )
    query: Optional[str] = Field(
        None,
        description="Search query to find specific posts about a topic. If provided, searches within the subreddit(s).",
    )
    sort: str = Field(
        "hot",
        description="Sort order: 'hot', 'new', 'top', 'rising', or 'relevance' (for search)",
    )
    time_filter: str = Field(
        "all", description="Time filter: 'hour', 'day', 'week', 'month', 'year', 'all'"
    )
    limit: int = Field(10, description="Number of posts to fetch (max 25)")


class RedditTool:
    """
    Reddit Posts Fetcher.

    Uses Reddit's public JSON API (no auth needed) to fetch:
    - Hot posts
    - New posts
    - Top posts (with time filter)
    - Rising posts

    Works with any public subreddit.
    """

    name = "reddit"
    description = """Fetch or search posts from any public subreddit. Search for specific topics 
    with a query, or browse hot/new/top/rising posts. Returns titles, scores, text content, authors, 
    and comment counts. Supports searching across multiple subreddits. No authentication required."""
    args_schema = RedditArgs

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    VALID_LISTING_SORTS = {"hot", "new", "top", "rising"}
    VALID_SEARCH_SORTS = {"relevance", "hot", "top", "new", "comments"}

    def run(
        self,
        subreddit: Union[str, List[str]] = "all",
        query: Optional[str] = None,
        sort: str = "hot",
        time_filter: str = "all",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Fetch or search posts from a subreddit."""

        # Guard against None values
        sort = sort or "hot"
        time_filter = time_filter or "all"
        limit = limit or 10
        limit = min(int(limit), 25)

        # Handle list of subreddits
        if isinstance(subreddit, list):
            all_results = []
            for sub in subreddit:
                result = self._fetch_single(
                    subreddit=sub,
                    query=query,
                    sort=sort,
                    time_filter=time_filter,
                    limit=limit,
                )
                all_results.append(result)
                time.sleep(0.5)  # Be nice to Reddit API
            return {
                "subreddits": [r.get("subreddit", "") for r in all_results],
                "total_posts": sum(r.get("count", 0) for r in all_results),
                "results": all_results,
            }

        return self._fetch_single(subreddit, query, sort, time_filter, limit)

    def _fetch_single(
        self,
        subreddit: str,
        query: Optional[str] = None,
        sort: str = "hot",
        time_filter: str = "all",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Fetch posts from a single subreddit (listing or search)."""
        subreddit = str(subreddit).strip().replace("r/", "").replace("/", "")

        try:
            headers = {"User-Agent": self.USER_AGENT}

            if query:
                # Search endpoint
                if sort not in self.VALID_SEARCH_SORTS:
                    sort = "relevance"
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    "q": query,
                    "restrict_sr": 1,
                    "sort": sort,
                    "t": time_filter,
                    "limit": limit,
                    "raw_json": 1,
                }
            else:
                # Listing endpoint
                if sort not in self.VALID_LISTING_SORTS:
                    sort = "hot"
                url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
                params = {"limit": limit, "raw_json": 1}
                if sort == "top":
                    params["t"] = time_filter

            # Make request with retry
            response = None
            for attempt in range(3):
                try:
                    response = requests.get(
                        url, headers=headers, params=params, timeout=15
                    )
                    if response.status_code == 429:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break
                except requests.exceptions.Timeout:
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    raise

            if response is None:
                return {
                    "error": "Failed to reach Reddit after retries",
                    "subreddit": subreddit,
                    "count": 0,
                    "posts": [],
                }

            if response.status_code == 404:
                return {
                    "error": f"Subreddit r/{subreddit} not found",
                    "subreddit": subreddit,
                    "count": 0,
                    "posts": [],
                }
            elif response.status_code == 403:
                return {
                    "error": f"Subreddit r/{subreddit} is private or banned",
                    "subreddit": subreddit,
                    "count": 0,
                    "posts": [],
                }

            response.raise_for_status()
            data = response.json()

            # Parse posts
            posts = []
            children = data.get("data", {}).get("children", [])

            for item in children:
                post = item.get("data", {})

                post_data = {
                    "id": post.get("id"),
                    "title": post.get("title"),
                    "author": post.get("author"),
                    "score": post.get("score"),
                    "upvote_ratio": post.get("upvote_ratio"),
                    "num_comments": post.get("num_comments"),
                    "created_utc": post.get("created_utc"),
                    "url": post.get("url"),
                    "permalink": f"https://reddit.com{post.get('permalink')}",
                    "is_self": post.get("is_self"),
                    "flair": post.get("link_flair_text"),
                    "awards": post.get("total_awards_received", 0),
                    "subreddit": post.get("subreddit", subreddit),
                }

                # Include self-text for text posts (important for sentiment analysis)
                if post.get("is_self") and post.get("selftext"):
                    post_data["text"] = post.get("selftext")[:1000]
                else:
                    # Still include title as text for analysis
                    post_data["text"] = post.get("title", "")

                posts.append(post_data)

            return {
                "subreddit": subreddit,
                "query": query,
                "sort": sort,
                "time_filter": time_filter,
                "count": len(posts),
                "posts": posts,
            }

        except Exception as e:
            return {
                "error": f"Failed to fetch from Reddit: {str(e)}",
                "subreddit": subreddit,
                "count": 0,
                "posts": [],
            }


# For direct testing
if __name__ == "__main__":
    tool = RedditTool()
    result = tool.run(subreddit="programming", sort="top", time_filter="week", limit=5)
    print(result)
