"""Reddit Tool - Fetch posts from Reddit using public JSON API."""

import requests
from typing import Literal, Optional, Dict, List, Any
from pydantic import BaseModel, Field


class RedditArgs(BaseModel):
    subreddit: str = Field(..., description="Subreddit name (without r/)")
    sort: Literal["hot", "new", "top", "rising"] = Field(
        "hot", description="Sort order: 'hot', 'new', 'top', or 'rising'"
    )
    time_filter: Literal["hour", "day", "week", "month", "year", "all"] = Field(
        "day", description="Time filter for 'top' sort: 'hour', 'day', 'week', 'month', 'year', 'all'"
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
    description = """Fetch posts from any public subreddit. Get hot, new, top, or rising posts 
    with titles, scores, authors, and comments count. No authentication required."""
    args_schema = RedditArgs
    
    USER_AGENT = "AgenticSystem-RedditTool/1.0"
    
    def run(
        self,
        subreddit: str,
        sort: str = "hot",
        time_filter: str = "day",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Fetch posts from a subreddit."""
        
        limit = min(limit, 25)  # Cap at 25
        subreddit = subreddit.strip().replace("r/", "").replace("/", "")
        
        try:
            # Build URL
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            params = {"limit": limit, "raw_json": 1}
            
            if sort == "top":
                params["t"] = time_filter
            
            # Make request
            headers = {"User-Agent": self.USER_AGENT}
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 404:
                return {"error": f"Subreddit r/{subreddit} not found"}
            elif response.status_code == 403:
                return {"error": f"Subreddit r/{subreddit} is private or banned"}
            
            response.raise_for_status()
            data = response.json()
            
            # Parse posts
            posts = []
            for item in data.get("data", {}).get("children", []):
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
                }
                
                # Include self-text for text posts
                if post.get("is_self") and post.get("selftext"):
                    post_data["text"] = post.get("selftext")[:500]
                
                # Include thumbnail if available
                thumbnail = post.get("thumbnail")
                if thumbnail and thumbnail.startswith("http"):
                    post_data["thumbnail"] = thumbnail
                
                posts.append(post_data)
            
            return {
                "subreddit": subreddit,
                "sort": sort,
                "time_filter": time_filter if sort == "top" else None,
                "count": len(posts),
                "posts": posts
            }
            
        except Exception as e:
            return {"error": f"Failed to fetch from Reddit: {str(e)}"}


# For direct testing
if __name__ == "__main__":
    tool = RedditTool()
    result = tool.run(subreddit="programming", sort="top", time_filter="week", limit=5)
    print(result)
