"""GitHub Trending Tool - Scrape trending repos and developers."""

import requests
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import re


class GitHubTrendingArgs(BaseModel):
    trending_type: Literal["repos", "developers"] = Field(
        "repos", description="Type of trending: 'repos' or 'developers'"
    )
    language: Optional[str] = Field(
        None,
        description="Filter by programming language (e.g., 'python', 'javascript', 'go')",
    )
    since: Literal["daily", "weekly", "monthly"] = Field(
        "daily", description="Time range: 'daily', 'weekly', or 'monthly'"
    )
    spoken_language: Optional[str] = Field(
        None, description="Filter by spoken language code (e.g., 'en', 'zh', 'es')"
    )


class GitHubTrendingTool:
    """
    GitHub Trending Repos and Developers Scraper.

    Scrapes the GitHub trending page to get:
    - Trending repositories (with stars, forks, description)
    - Trending developers

    Supports filtering by:
    - Programming language
    - Time range (daily, weekly, monthly)
    - Spoken language
    """

    name = "github_trending"
    description = """Get trending repositories or developers from GitHub. Filter by programming language, 
    time range (daily/weekly/monthly), and spoken language. Returns stars, forks, and descriptions."""
    args_schema = GitHubTrendingArgs

    BASE_URL = "https://github.com/trending"

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    def run(
        self,
        trending_type: str = "repos",
        language: Optional[str] = None,
        since: str = "daily",
        spoken_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch trending repos or developers from GitHub."""

        # Normalize aliases
        trending_type = trending_type.lower().strip()
        if trending_type in ("repositories", "repository", "repo"):
            trending_type = "repos"
        elif trending_type in ("developer", "devs", "dev"):
            trending_type = "developers"

        try:
            if trending_type == "repos":
                return self._get_trending_repos(language, since, spoken_language)
            elif trending_type == "developers":
                return self._get_trending_developers(language, since)
            else:
                return {
                    "error": f"Unknown trending_type: {trending_type}. Use 'repos' or 'developers'"
                }
        except Exception as e:
            return {"error": f"Failed to fetch trending: {str(e)}"}

    def _fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page."""
        headers = {"User-Agent": self.USER_AGENT}
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to {url}. DNS/network issue? Error: {e}"
            )
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request to {url} timed out: {e}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP request failed for {url}: {e}")

    def _get_trending_repos(
        self, language: Optional[str], since: str, spoken_language: Optional[str]
    ) -> Dict[str, Any]:
        """Get trending repositories."""
        # Build URL
        url = self.BASE_URL

        # Only append language if it's a real language (not 'all', None, or empty)
        if language and language.lower() not in ("all", "any", "none", ""):
            url += f"/{language.lower()}"

        params = [f"since={since}"]
        if spoken_language and spoken_language.lower() not in (
            "all",
            "any",
            "none",
            "en",
            "",
        ):
            params.append(f"spoken_language_code={spoken_language}")

        if params:
            url += "?" + "&".join(params)

        soup = self._fetch_page(url)

        repos = []
        for article in soup.select("article.Box-row"):
            try:
                # Repository name and owner
                repo_link = article.select_one("h2 a")
                if not repo_link:
                    continue

                full_name = repo_link.get("href", "").strip("/")
                parts = full_name.split("/")
                if len(parts) != 2:
                    continue

                owner, name = parts

                # Description
                desc_elem = article.select_one("p")
                description = desc_elem.get_text(strip=True) if desc_elem else None

                # Language
                lang_elem = article.select_one("[itemprop='programmingLanguage']")
                repo_language = lang_elem.get_text(strip=True) if lang_elem else None

                # Stars and forks (from the inline stats)
                stats = article.select("a.Link--muted")
                stars = 0
                forks = 0

                for stat in stats:
                    href = stat.get("href", "")
                    text = stat.get_text(strip=True).replace(",", "")
                    if "/stargazers" in href:
                        stars = self._parse_number(text)
                    elif "/forks" in href:
                        forks = self._parse_number(text)

                # Today's stars
                today_stars_elem = article.select_one(
                    "span.d-inline-block.float-sm-right"
                )
                today_stars = 0
                if today_stars_elem:
                    today_text = today_stars_elem.get_text(strip=True)
                    match = re.search(r"([\d,]+)", today_text)
                    if match:
                        today_stars = self._parse_number(match.group(1))

                # Built by (contributors)
                built_by = []
                for avatar in article.select("span.d-inline-block a img"):
                    alt = avatar.get("alt", "")
                    if alt.startswith("@"):
                        built_by.append(alt[1:])

                repos.append(
                    {
                        "name": name,
                        "owner": owner,
                        "full_name": full_name,
                        "description": description,
                        "language": repo_language,
                        "stars": stars,
                        "forks": forks,
                        "today_stars": today_stars,
                        "built_by": built_by[:5],
                        "url": f"https://github.com/{full_name}",
                    }
                )

            except Exception:
                continue

        return {
            "type": "repos",
            "language": language,
            "since": since,
            "count": len(repos),
            "repos": repos,
        }

    def _get_trending_developers(
        self, language: Optional[str], since: str
    ) -> Dict[str, Any]:
        """Get trending developers."""
        url = f"{self.BASE_URL}/developers"
        if language:
            url += f"/{language.lower()}"
        url += f"?since={since}"

        soup = self._fetch_page(url)

        developers = []
        for article in soup.select("article.Box-row"):
            try:
                # Username
                username_elem = article.select_one("h1.h3 a")
                if not username_elem:
                    continue

                username = username_elem.get("href", "").strip("/")

                # Display name
                name_elem = article.select_one("p.f4 a")
                display_name = name_elem.get_text(strip=True) if name_elem else username

                # Avatar
                avatar_elem = article.select_one("img.avatar")
                avatar = avatar_elem.get("src") if avatar_elem else None

                # Popular repo
                repo_elem = article.select_one("article h1 a")
                popular_repo = None
                repo_description = None

                if repo_elem:
                    popular_repo = repo_elem.get_text(strip=True)
                    repo_desc_elem = article.select_one("article .f6")
                    if repo_desc_elem:
                        repo_description = repo_desc_elem.get_text(strip=True)

                developers.append(
                    {
                        "username": username,
                        "name": display_name,
                        "avatar": avatar,
                        "popular_repo": popular_repo,
                        "repo_description": repo_description,
                        "url": f"https://github.com/{username}",
                    }
                )

            except Exception:
                continue

        return {
            "type": "developers",
            "language": language,
            "since": since,
            "count": len(developers),
            "developers": developers,
        }

    def _parse_number(self, text: str) -> int:
        """Parse a number string that may have commas or k/m suffixes."""
        text = text.replace(",", "").strip().lower()

        if not text:
            return 0

        try:
            if text.endswith("k"):
                return int(float(text[:-1]) * 1000)
            elif text.endswith("m"):
                return int(float(text[:-1]) * 1000000)
            else:
                return int(text)
        except ValueError:
            return 0


# For direct testing
if __name__ == "__main__":
    tool = GitHubTrendingTool()
    result = tool.run(trending_type="repos", language="python", since="daily")
    print(result)
