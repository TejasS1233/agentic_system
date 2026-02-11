import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class ExtractCitationsToolArgs(BaseModel):
    paper_url: str = Field(..., description="URL of the paper")


class ExtractCitationsTool:
    name = "extract_citations"
    description = "Extract citations from a research paper"
    args_schema = ExtractCitationsToolArgs

    def run(self, paper_url: str) -> dict:
        try:
            response = requests.get(
                paper_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            soup = BeautifulSoup(response.text, "html.parser")
            citations = []
            for ref in soup.find_all("div", class_="ref"):
                citations.append(ref.text.strip())
            return {"citations": citations}
        except Exception as e:
            return {"error": str(e)}


def test_tool():
    tool = ExtractCitationsTool()
    result = tool.run("https://arxiv.org/abs/1706.03762")
    print(result)


if __name__ == "__main__":
    test_tool()
