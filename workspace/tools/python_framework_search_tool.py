import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class PythonFrameworkSearchToolArgs(BaseModel):
    query: str = Field(..., description="Search query for Python frameworks")


class PythonFrameworkSearchTool:
    name = "python_framework_search"
    description = "Scrape and process search results to find the best Python frameworks"
    args_schema = PythonFrameworkSearchToolArgs

    def run(self, query: str) -> str:
        url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for result in soup.find_all("div", class_="g"):
            link = result.find("a")
            if link:
                title = link.text
                href = link["href"]
                results.append({"title": title, "href": href})
        return str(results)


def test_tool():
    tool = PythonFrameworkSearchTool()
    query = "best python framework"
    result = tool.run(query)
    print(result)


if __name__ == "__main__":
    test_tool()
