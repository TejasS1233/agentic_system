import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class JoJoStandSearchArgs(BaseModel):
    query: str = Field(..., description="Search query for JoJo stands")


class JoJoStandSearchTool:
    name = "jojo_stand_search"
    description = "Search for top JoJo stands"
    args_schema = JoJoStandSearchArgs

    def run(self, query: str) -> str:
        if not query:
            return "Error: No query provided"
        url = f"https://jojo.fandom.com/wiki/{query.replace(' ', '_')}"
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, "html.parser")
            result = soup.find("p")
            if result:
                return str(result)
            else:
                return "No results found"
        except Exception as e:
            return f"Error: {str(e)}"


def test_tool():
    tool = JoJoStandSearchTool()
    print(tool.run("Gold_Experience"))


if __name__ == "__main__":
    test_tool()
