"""Reddit Sentiment Analyzer - Analyze sentiment of Reddit posts using VADER."""

import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pydantic import BaseModel, Field
from typing import Union, Dict, List, Any

nltk.download("vader_lexicon", quiet=True)


class RedditSentimentAnalyzerArgs(BaseModel):
    data: Union[str, Dict, List] = Field(
        ...,
        description="Reddit post data - can be a single text string, a list of post titles/texts, "
        "or structured output from RedditTool (dict with 'posts' or 'results' keys)",
    )


class RedditSentimentAnalyzerTool:
    """
    Sentiment analyzer for Reddit posts using NLTK VADER.

    Accepts multiple input formats:
    - Single text string
    - List of text strings
    - RedditTool output (dict with posts containing titles/text)
    - Multi-subreddit RedditTool output (dict with 'results' list)

    Returns per-post sentiment scores and aggregate statistics.
    """

    name = "reddit_sentiment_analyzer"
    description = """Analyze sentiment of Reddit posts. Accepts raw text, list of texts, 
    or structured Reddit post data. Returns per-post and aggregate sentiment scores."""
    args_schema = RedditSentimentAnalyzerArgs

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def _extract_texts(self, data: Any) -> List[Dict[str, str]]:
        """Extract text items from various input formats."""
        texts = []

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                if data.strip():
                    return [{"source": "input", "text": data}]
                return []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and item.strip():
                    texts.append({"source": "list", "text": item})
                elif isinstance(item, dict):
                    text = (
                        item.get("text")
                        or item.get("title")
                        or item.get("selftext", "")
                    )
                    source = item.get("subreddit", item.get("author", "post"))
                    if text.strip():
                        texts.append({"source": str(source), "text": text})
            return texts

        if isinstance(data, dict):
            # Multi-subreddit: {"results": [{subreddit, posts}, ...]}
            if "results" in data and isinstance(data["results"], list):
                for sub_result in data["results"]:
                    if isinstance(sub_result, dict):
                        subreddit = sub_result.get("subreddit", "unknown")
                        for post in sub_result.get("posts", []):
                            if isinstance(post, dict):
                                text = post.get("text") or post.get("title", "")
                            elif isinstance(post, str):
                                text = post
                            else:
                                text = ""
                            if text.strip():
                                texts.append({"source": subreddit, "text": text})
                return texts

            # Single subreddit: {"subreddit": "...", "posts": [...]}
            if "posts" in data and isinstance(data["posts"], list):
                subreddit = data.get("subreddit", "unknown")
                for post in data["posts"]:
                    if isinstance(post, dict):
                        text = post.get("text") or post.get("title", "")
                    elif isinstance(post, str):
                        text = post
                    else:
                        text = ""
                    if text.strip():
                        texts.append({"source": subreddit, "text": text})
                return texts

            # Fallback: try each key as a source
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            texts.append({"source": key, "text": item})
                        elif isinstance(item, dict):
                            text = item.get("text") or item.get("title", "")
                            if text.strip():
                                texts.append({"source": key, "text": text})

        return texts

    def run(self, data: Union[str, Dict, List] = "") -> Dict[str, Any]:
        """Analyze sentiment of Reddit posts."""
        texts = self._extract_texts(data)

        if not texts:
            return {
                "error": "No text content found to analyze",
                "post_count": 0,
                "sentiments": [],
                "aggregate": {"neg": 0, "neu": 0, "pos": 0, "compound": 0},
            }

        sentiments = []
        by_source = {}

        for item in texts:
            scores = self.sia.polarity_scores(item["text"])
            entry = {
                "source": item["source"],
                "text": item["text"][:200],
                "sentiment": scores,
                "label": (
                    "positive"
                    if scores["compound"] >= 0.05
                    else "negative"
                    if scores["compound"] <= -0.05
                    else "neutral"
                ),
            }
            sentiments.append(entry)

            src = item["source"]
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(scores)

        all_scores = [s["sentiment"] for s in sentiments]
        n = len(all_scores)
        aggregate = {
            "neg": round(sum(s["neg"] for s in all_scores) / n, 4),
            "neu": round(sum(s["neu"] for s in all_scores) / n, 4),
            "pos": round(sum(s["pos"] for s in all_scores) / n, 4),
            "compound": round(sum(s["compound"] for s in all_scores) / n, 4),
        }

        source_aggregates = {}
        for src, scores_list in by_source.items():
            sn = len(scores_list)
            source_aggregates[src] = {
                "post_count": sn,
                "avg_compound": round(sum(s["compound"] for s in scores_list) / sn, 4),
                "positive_pct": round(
                    sum(1 for s in scores_list if s["compound"] >= 0.05) / sn * 100, 1
                ),
                "negative_pct": round(
                    sum(1 for s in scores_list if s["compound"] <= -0.05) / sn * 100, 1
                ),
                "neutral_pct": round(
                    sum(1 for s in scores_list if -0.05 < s["compound"] < 0.05)
                    / sn
                    * 100,
                    1,
                ),
            }

        pos_count = sum(1 for s in sentiments if s["label"] == "positive")
        neg_count = sum(1 for s in sentiments if s["label"] == "negative")
        neu_count = sum(1 for s in sentiments if s["label"] == "neutral")

        return {
            "post_count": n,
            "aggregate": aggregate,
            "distribution": {
                "positive": pos_count,
                "negative": neg_count,
                "neutral": neu_count,
                "positive_pct": round(pos_count / n * 100, 1),
                "negative_pct": round(neg_count / n * 100, 1),
                "neutral_pct": round(neu_count / n * 100, 1),
            },
            "by_subreddit": source_aggregates if len(source_aggregates) > 1 else None,
            "sentiments": sentiments[:50],
            "summary": (
                f"Analyzed {n} posts: {pos_count} positive ({round(pos_count / n * 100, 1)}%), "
                f"{neg_count} negative ({round(neg_count / n * 100, 1)}%), "
                f"{neu_count} neutral ({round(neu_count / n * 100, 1)}%). "
                f"Overall compound: {aggregate['compound']:.4f}"
            ),
        }


if __name__ == "__main__":
    tool = RedditSentimentAnalyzerTool()
    print(tool.run(data="AI is going to take all our jobs and it's terrifying"))
