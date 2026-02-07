import requests
from bs4 from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

class TrendingRepoLanguagesArgs(BaseModel):
    count: int = Field(..., description="Number of trending repositories to fetch")

class TrendingRepoLanguagesTool:
    name = "trending_repo_languages"
    description = "Extract language distribution from GitHub trending repositories"
    args_schema = TrendingRepoLanguagesArgs

    def run(self, count: int) -> str:
        try:
            url = "https://github.com/trending"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            repos = soup.select('article.Box-row h2 a')[:count]
            languages = {}
            for repo in repos:
                repo_url = f"https://github.com{repo['href']}"
                repo_response = requests.get(repo_url, headers=headers)
                repo_response.raise_for_status()
                repo_soup = BeautifulSoup(repo_response.text, 'html.parser')
                language = repo_soup.select_one('span[itemprop="programmingLanguage"]')
                if language:
                    lang_text = language.text.strip()
                    languages[lang_text] = languages.get(lang_text, 0) + 1
            return str(languages)
        except Exception as e:
            return f"Error: {str(e)}"

def test_tool():
    tool = TrendingRepoLanguagesTool()
    args = TrendingRepoLanguagesArgs(count=5)
    result = tool.run(**args.dict())
    print(result)

if __name__ == "__main__":
    test_tool()
