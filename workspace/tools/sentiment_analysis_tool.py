import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pydantic import BaseModel, Field

class SentimentAnalysisArgs(BaseModel):
    text: str = Field(..., description="Text to analyze")

class SentimentAnalysisTool:
    name = "sentiment_analysis"
    description = "Compare community opinions based on sentiment analysis"
    args_schema = SentimentAnalysisArgs

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def run(self, text: str) -> str:
        sentiment_scores = self.sia.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            return "Positive sentiment"
        elif compound_score <= -0.05:
            return "Negative sentiment"
        else:
            return "Neutral sentiment"

def test_tool():
    tool = SentimentAnalysisTool()
    result = tool.run("I love this product!")
    print(result)

if __name__ == "__main__":
    nltk.download('vader_lexicon')
    test_tool()