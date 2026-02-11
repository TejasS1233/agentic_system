import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class PythonFrameworksSearchToolArgs(BaseModel):
    query: str = Field(..., description="Query for searching Python frameworks")


class PythonFrameworksSearchTool:
    name = "search_python_frameworks"
    description = "Search for the best Python frameworks in 2024"
    args_schema = PythonFrameworksSearchToolArgs

    def run(self, query: str) -> str:
        url = f"https://duckduckgo.com/html?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all("a", class_="result__a")
        output = """
        Results for {query}:
        """
        for result in results:
            output += result.text + "\n"
        return output


def test_tool():
    tool = PythonFrameworksSearchTool()
    print(tool.run("best python frameworks 2024"))


if __name__ == "__main__":
    test_tool()
