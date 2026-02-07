import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

class BrawlStarsTopBrawlersArgs(BaseModel):
    region: str = Field(..., description="Region of the Brawl Stars server")

class BrawlStarsTopBrawlersTool:
    name = "brawl_stars_top_brawlers"
    description = "Get the top played brawlers in Brawl Stars"
    args_schema = BrawlStarsTopBrawlersArgs

    def run(self, region: str) -> str:
        url = f'https://brawlstars.fandom.com/wiki/{region}_Server/Brawlers'
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        brawlers = []
        for row in soup.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) > 0:
                brawler = cols[0].text.strip()
                brawlers.append(brawler)
        return {"brawlers": brawlers}

def test_tool():
    tool = BrawlStarsTopBrawlersTool()
    print(tool.run('Global'))
if __name__ == "__main__":
    test_tool()