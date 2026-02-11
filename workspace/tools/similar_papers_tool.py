import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class SimilarPapersArgs(BaseModel):
    title: str = Field(..., description="Title of the paper to find similar papers for")


class SimilarPapersTool:
    name = "similar_papers"
    description = "Search for similar papers to the given title."
    args_schema = SimilarPapersArgs

    def run(self, title: str) -> str:
        url = f"https://paperswithcode.com/search?q={title}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        papers = []
        for paper in soup.find_all("div", class_="paper-card"):
            title = paper.find("h2", class_="paper-title").text.strip()
            papers.append(title)
        return str(papers)


def test_tool():
    tool = SimilarPapersTool()
    print(tool.run("Attention Is All You Need"))


if __name__ == "__main__":
    test_tool()
