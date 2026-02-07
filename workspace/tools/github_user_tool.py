"""GitHub User Profile Tool - Get user info via GitHub API."""

import requests
from typing import Literal, Optional, Dict, List, Any
from pydantic import BaseModel, Field


class GitHubUserArgs(BaseModel):
    username: str = Field(..., description="GitHub username")
    info_type: Literal["profile", "repos", "starred", "gists", "events"] = Field(
        "profile",
        description="Type of info: 'profile' (user details), 'repos' (public repos), "
                    "'starred' (starred repos), 'gists', or 'events' (recent activity)"
    )
    limit: int = Field(10, description="Number of items to fetch (for lists)")
    sort: Literal["updated", "created", "pushed", "stars"] = Field(
        "updated", description="Sort order for repos"
    )


class GitHubUserTool:
    """
    GitHub User Profile Tool.
    
    Fetch information about any GitHub user:
    - Profile info (bio, company, location, followers, etc.)
    - Public repositories
    - Starred repositories
    - Public gists
    - Recent activity/events
    """
    
    name = "github_user"
    description = """Fetch information about any GitHub user. Get profile details, public repos, 
    starred repos, gists, and recent activity. No authentication required for public profiles."""
    args_schema = GitHubUserArgs
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None):
        self.headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "AgenticSystem-GitHubTool/1.0"
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    def run(
        self,
        username: str,
        info_type: str = "profile",
        limit: int = 10,
        sort: str = "updated"
    ) -> Dict[str, Any]:
        """Fetch user information from GitHub."""
        
        try:
            if info_type == "profile":
                return self._get_profile(username)
            elif info_type == "repos":
                return self._get_repos(username, limit, sort)
            elif info_type == "starred":
                return self._get_starred(username, limit)
            elif info_type == "gists":
                return self._get_gists(username, limit)
            elif info_type == "events":
                return self._get_events(username, limit)
            else:
                return {"error": f"Unknown info_type: {info_type}"}
        except Exception as e:
            return {"error": f"Failed to fetch user info: {str(e)}"}
    
    def _request(self, endpoint: str) -> Any:
        """Make a request to GitHub API."""
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, headers=self.headers, timeout=15)
        
        if response.status_code == 404:
            raise ValueError("User not found")
        elif response.status_code == 403:
            raise ValueError("Rate limit exceeded. Try again later.")
        
        response.raise_for_status()
        return response.json()
    
    def _get_profile(self, username: str) -> Dict[str, Any]:
        """Get user profile."""
        data = self._request(f"/users/{username}")
        
        return {
            "username": data.get("login"),
            "name": data.get("name"),
            "bio": data.get("bio"),
            "company": data.get("company"),
            "location": data.get("location"),
            "email": data.get("email"),
            "blog": data.get("blog"),
            "twitter": data.get("twitter_username"),
            "avatar": data.get("avatar_url"),
            "public_repos": data.get("public_repos"),
            "public_gists": data.get("public_gists"),
            "followers": data.get("followers"),
            "following": data.get("following"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "hireable": data.get("hireable"),
            "type": data.get("type"),  # User or Organization
            "url": data.get("html_url"),
        }
    
    def _get_repos(self, username: str, limit: int, sort: str) -> Dict[str, Any]:
        """Get user's public repositories."""
        data = self._request(f"/users/{username}/repos?per_page={limit}&sort={sort}")
        
        repos = [
            {
                "name": r.get("name"),
                "full_name": r.get("full_name"),
                "description": r.get("description"),
                "language": r.get("language"),
                "stars": r.get("stargazers_count"),
                "forks": r.get("forks_count"),
                "is_fork": r.get("fork"),
                "updated_at": r.get("updated_at"),
                "url": r.get("html_url"),
            }
            for r in data
        ]
        
        return {
            "username": username,
            "count": len(repos),
            "repos": repos
        }
    
    def _get_starred(self, username: str, limit: int) -> Dict[str, Any]:
        """Get repos starred by user."""
        data = self._request(f"/users/{username}/starred?per_page={limit}")
        
        starred = [
            {
                "name": r.get("full_name"),
                "description": r.get("description"),
                "language": r.get("language"),
                "stars": r.get("stargazers_count"),
                "url": r.get("html_url"),
            }
            for r in data
        ]
        
        return {
            "username": username,
            "count": len(starred),
            "starred": starred
        }
    
    def _get_gists(self, username: str, limit: int) -> Dict[str, Any]:
        """Get user's public gists."""
        data = self._request(f"/users/{username}/gists?per_page={limit}")
        
        gists = [
            {
                "id": g.get("id"),
                "description": g.get("description"),
                "public": g.get("public"),
                "files": list(g.get("files", {}).keys()),
                "comments": g.get("comments"),
                "created_at": g.get("created_at"),
                "url": g.get("html_url"),
            }
            for g in data
        ]
        
        return {
            "username": username,
            "count": len(gists),
            "gists": gists
        }
    
    def _get_events(self, username: str, limit: int) -> Dict[str, Any]:
        """Get user's recent public events."""
        data = self._request(f"/users/{username}/events/public?per_page={limit}")
        
        events = []
        for e in data:
            event = {
                "type": e.get("type"),
                "repo": e.get("repo", {}).get("name"),
                "created_at": e.get("created_at"),
            }
            
            # Add type-specific details
            payload = e.get("payload", {})
            if e.get("type") == "PushEvent":
                event["commits"] = len(payload.get("commits", []))
                event["branch"] = payload.get("ref", "").replace("refs/heads/", "")
            elif e.get("type") == "PullRequestEvent":
                event["action"] = payload.get("action")
                event["pr_title"] = payload.get("pull_request", {}).get("title")
            elif e.get("type") == "IssuesEvent":
                event["action"] = payload.get("action")
                event["issue_title"] = payload.get("issue", {}).get("title")
            elif e.get("type") == "WatchEvent":
                event["action"] = payload.get("action")  # starred
            elif e.get("type") == "CreateEvent":
                event["ref_type"] = payload.get("ref_type")  # repository, branch, tag
                event["ref"] = payload.get("ref")
            
            events.append(event)
        
        return {
            "username": username,
            "count": len(events),
            "events": events
        }


# For direct testing
if __name__ == "__main__":
    tool = GitHubUserTool()
    result = tool.run("torvalds", info_type="profile")
    print(result)
