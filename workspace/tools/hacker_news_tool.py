"""Hacker News Tool - Fetch stories and comments from HN API."""

import requests
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field


class HackerNewsArgs(BaseModel):
    story_type: Literal["top", "new", "best", "ask", "show", "job"] = Field(
        "top",
        description="Type of stories: 'top', 'new', 'best', 'ask', 'show', or 'job'",
    )
    limit: int = Field(10, description="Number of stories to fetch (max 30)")
    include_comments: bool = Field(
        False, description="Include top comments for each story"
    )


class HackerNewsTool:
    """
    Hacker News API Tool.

    Fetch stories from Hacker News using their official API:
    - Top stories
    - New stories
    - Best stories
    - Ask HN
    - Show HN
    - Jobs

    Optionally includes top comments for each story.
    """

    name = "hacker_news"
    description = """Fetch stories from Hacker News. Get top, new, best, ask, show, or job stories 
    with titles, scores, authors, and optionally comments. Uses the official HN API."""
    args_schema = HackerNewsArgs

    BASE_URL = "https://hacker-news.firebaseio.com/v0"

    def run(
        self, story_type: str = "top", limit: int = 10, include_comments: bool = False
    ) -> Dict[str, Any]:
        """Fetch stories from Hacker News."""

        limit = min(limit, 30)  # Cap at 30

        try:
            # Get story IDs
            endpoint_map = {
                "top": "topstories",
                "new": "newstories",
                "best": "beststories",
                "ask": "askstories",
                "show": "showstories",
                "job": "jobstories",
            }

            endpoint = endpoint_map.get(story_type, "topstories")
            story_ids = self._fetch(f"/{endpoint}.json")[:limit]

            # Fetch story details
            stories = []
            for story_id in story_ids:
                story = self._fetch(f"/item/{story_id}.json")
                if not story:
                    continue

                story_data = {
                    "id": story.get("id"),
                    "title": story.get("title"),
                    "url": story.get("url"),
                    "score": story.get("score"),
                    "author": story.get("by"),
                    "time": story.get("time"),
                    "comment_count": len(story.get("kids", [])),
                    "type": story.get("type"),
                    "hn_url": f"https://news.ycombinator.com/item?id={story.get('id')}",
                }

                # For text posts (Ask HN, etc.)
                if story.get("text"):
                    story_data["text"] = story.get("text")[:500]

                # Include top comments if requested
                if include_comments and story.get("kids"):
                    comments = []
                    for comment_id in story.get("kids", [])[:3]:
                        comment = self._fetch(f"/item/{comment_id}.json")
                        if comment and comment.get("text"):
                            comments.append(
                                {
                                    "author": comment.get("by"),
                                    "text": comment.get("text")[:300],
                                }
                            )
                    story_data["top_comments"] = comments

                stories.append(story_data)

            return {"type": story_type, "count": len(stories), "stories": stories}

        except Exception as e:
            return {"error": f"Failed to fetch from Hacker News: {str(e)}"}

    def _fetch(self, endpoint: str) -> Any:
        """Make a request to HN API."""
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()


# For direct testing
if __name__ == "__main__":
    tool = HackerNewsTool()
    result = tool.run(story_type="top", limit=5)
    print(result)
