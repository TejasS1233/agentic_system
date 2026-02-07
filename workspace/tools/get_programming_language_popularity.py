import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

class GetProgrammingLanguagePopularityArgs(BaseModel):
    language: str = Field(..., description="The programming language to check popularity for")

class GetProgrammingLanguagePopularityTool:
    name = "get_programming_language_popularity"
    description = "Get the popularity of a programming language from TIOBE Index"
    args_schema = GetProgrammingLanguagePopularityArgs

    def run(self, language: str) -> str:
        url = 'https://www.tiobe.com/tiobe-index/'
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'table table-striped table-top20'})
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 1 and cols[3].text.strip().lower() == language.lower():
                return f'{language} is currently ranked {cols[0].text.strip()} with a rating of {cols[4].text.strip()}'
        return f'{language} not found in TIOBE Index'

def test_tool():
    tool = GetProgrammingLanguagePopularityTool()
    print(tool.run('python'))
if __name__ == "__main__":
    test_tool()