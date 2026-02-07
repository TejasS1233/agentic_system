import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

class ThadomalShahaniPlacementStatsArgs(BaseModel):
    query: str = Field(..., description="Search query for placement statistics")

class ThadomalShahaniPlacementStats:
    name = "thadomal_shahani_placement_stats"
    description = "Search for placement statistics of Thadomal Shahani Engineering College"
    args_schema = ThadomalShahaniPlacementStatsArgs

    def run(self, query: str) -> str:
        url = "https://www.google.com/search"
        params = {
            "q": query
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, params=params, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        result = ""
        for item in soup.find_all('div', attrs={'class': 'BNeawe'}):
            result += item.text + "\n"
        return result

def test_tool():
    tool = ThadomalShahaniPlacementStats()
    print(tool.run("Thadomal Shahani Engineering College placement statistics"))
if __name__ == "__main__":
    test_tool()