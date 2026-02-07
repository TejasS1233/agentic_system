"""GitHub Repository Tool - Get repo info via GitHub API."""

import requests
from typing import Literal, Optional, Dict, List, Any
from pydantic import BaseModel, Field


class GitHubRepoArgs(BaseModel):
    owner: str = Field(..., description="Repository owner (username or organization)")
    repo: str = Field(..., description="Repository name")
    info_type: Literal["overview", "commits", "issues", "prs", "contributors", "releases", "languages"] = Field(
        "overview", 
        description="Type of info to fetch: 'overview' (stats), 'commits' (recent), 'issues', 'prs', "
                    "'contributors', 'releases', or 'languages'"
    )
    limit: int = Field(10, description="Number of items to fetch (for lists)")


class GitHubRepoTool:
    """
    GitHub Repository Information Tool.
    
    Uses GitHub's public API (no auth required for public repos) to fetch:
    - Repository overview (description, stars, forks, language, etc.)
    - Recent commits
    - Open issues and PRs
    - Contributors
    - Releases
    - Language breakdown
    """
    
    name = "github_repo"
    description = """Fetch information about any public GitHub repository. Get stats, recent commits, 
    issues, PRs, contributors, releases, and languages. No authentication required for public repos."""
    args_schema = GitHubRepoArgs
    
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
        owner: str,
        repo: str,
        info_type: str = "overview",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Fetch repository information from GitHub."""
        
        try:
            if info_type == "overview":
                return self._get_overview(owner, repo)
            elif info_type == "commits":
                return self._get_commits(owner, repo, limit)
            elif info_type == "issues":
                return self._get_issues(owner, repo, limit)
            elif info_type == "prs":
                return self._get_prs(owner, repo, limit)
            elif info_type == "contributors":
                return self._get_contributors(owner, repo, limit)
            elif info_type == "releases":
                return self._get_releases(owner, repo, limit)
            elif info_type == "languages":
                return self._get_languages(owner, repo)
            else:
                return {"error": f"Unknown info_type: {info_type}"}
        except Exception as e:
            return {"error": f"Failed to fetch repo info: {str(e)}"}
    
    def _request(self, endpoint: str) -> Dict:
        """Make a request to GitHub API."""
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, headers=self.headers, timeout=15)
        
        if response.status_code == 404:
            raise ValueError("Repository not found")
        elif response.status_code == 403:
            raise ValueError("Rate limit exceeded. Try again later.")
        
        response.raise_for_status()
        return response.json()
    
    def _get_overview(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository overview."""
        data = self._request(f"/repos/{owner}/{repo}")
        
        return {
            "name": data.get("full_name"),
            "description": data.get("description"),
            "homepage": data.get("homepage"),
            "language": data.get("language"),
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "watchers": data.get("subscribers_count"),
            "open_issues": data.get("open_issues_count"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "pushed_at": data.get("pushed_at"),
            "size_kb": data.get("size"),
            "default_branch": data.get("default_branch"),
            "license": data.get("license", {}).get("name") if data.get("license") else None,
            "topics": data.get("topics", []),
            "is_fork": data.get("fork"),
            "is_archived": data.get("archived"),
            "visibility": data.get("visibility"),
            "url": data.get("html_url"),
        }
    
    def _get_commits(self, owner: str, repo: str, limit: int) -> Dict[str, Any]:
        """Get recent commits."""
        data = self._request(f"/repos/{owner}/{repo}/commits?per_page={limit}")
        
        commits = [
            {
                "sha": c.get("sha", "")[:7],
                "message": c.get("commit", {}).get("message", "").split("\n")[0],
                "author": c.get("commit", {}).get("author", {}).get("name"),
                "date": c.get("commit", {}).get("author", {}).get("date"),
                "url": c.get("html_url"),
            }
            for c in data
        ]
        
        return {
            "repo": f"{owner}/{repo}",
            "count": len(commits),
            "commits": commits
        }
    
    def _get_issues(self, owner: str, repo: str, limit: int) -> Dict[str, Any]:
        """Get open issues."""
        data = self._request(f"/repos/{owner}/{repo}/issues?state=open&per_page={limit}")
        
        # Filter out pull requests (they appear in issues endpoint)
        issues = [
            {
                "number": i.get("number"),
                "title": i.get("title"),
                "state": i.get("state"),
                "author": i.get("user", {}).get("login"),
                "labels": [l.get("name") for l in i.get("labels", [])],
                "comments": i.get("comments"),
                "created_at": i.get("created_at"),
                "url": i.get("html_url"),
            }
            for i in data if "pull_request" not in i
        ]
        
        return {
            "repo": f"{owner}/{repo}",
            "count": len(issues),
            "issues": issues
        }
    
    def _get_prs(self, owner: str, repo: str, limit: int) -> Dict[str, Any]:
        """Get open pull requests."""
        data = self._request(f"/repos/{owner}/{repo}/pulls?state=open&per_page={limit}")
        
        prs = [
            {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "state": pr.get("state"),
                "author": pr.get("user", {}).get("login"),
                "draft": pr.get("draft"),
                "created_at": pr.get("created_at"),
                "updated_at": pr.get("updated_at"),
                "url": pr.get("html_url"),
            }
            for pr in data
        ]
        
        return {
            "repo": f"{owner}/{repo}",
            "count": len(prs),
            "pull_requests": prs
        }
    
    def _get_contributors(self, owner: str, repo: str, limit: int) -> Dict[str, Any]:
        """Get top contributors."""
        data = self._request(f"/repos/{owner}/{repo}/contributors?per_page={limit}")
        
        contributors = [
            {
                "username": c.get("login"),
                "contributions": c.get("contributions"),
                "avatar": c.get("avatar_url"),
                "profile": c.get("html_url"),
            }
            for c in data
        ]
        
        return {
            "repo": f"{owner}/{repo}",
            "count": len(contributors),
            "contributors": contributors
        }
    
    def _get_releases(self, owner: str, repo: str, limit: int) -> Dict[str, Any]:
        """Get releases."""
        data = self._request(f"/repos/{owner}/{repo}/releases?per_page={limit}")
        
        releases = [
            {
                "tag": r.get("tag_name"),
                "name": r.get("name"),
                "prerelease": r.get("prerelease"),
                "draft": r.get("draft"),
                "published_at": r.get("published_at"),
                "author": r.get("author", {}).get("login"),
                "url": r.get("html_url"),
            }
            for r in data
        ]
        
        return {
            "repo": f"{owner}/{repo}",
            "count": len(releases),
            "releases": releases
        }
    
    def _get_languages(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get language breakdown."""
        data = self._request(f"/repos/{owner}/{repo}/languages")
        
        total = sum(data.values())
        languages = [
            {
                "language": lang,
                "bytes": bytes_count,
                "percentage": round(bytes_count / total * 100, 2) if total > 0 else 0
            }
            for lang, bytes_count in sorted(data.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            "repo": f"{owner}/{repo}",
            "total_bytes": total,
            "languages": languages
        }


# For direct testing
if __name__ == "__main__":
    tool = GitHubRepoTool()
    result = tool.run("torvalds", "linux", info_type="overview")
    print(result)
