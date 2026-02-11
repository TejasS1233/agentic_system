"""Reddit Subreddit Comparator - Compare community opinions across subreddits using JSON API."""

import json
from typing import Union, Dict, List, Any
from pydantic import BaseModel, Field


class RedditSubredditComparatorArgs(BaseModel):
    data: Union[str, Dict, List] = Field(
        ...,
        description="Sentiment analysis results from RedditSentimentAnalyzerTool, "
        "or structured Reddit post data with subreddit groupings",
    )


class RedditSubredditComparator:
    """
    Compare community opinions across multiple subreddits.

    Takes sentiment analysis results grouped by subreddit and produces
    a comparison report showing how different communities feel about a topic.

    Works with output from RedditSentimentAnalyzerTool (preferred) or
    raw Reddit post data.
    """

    name = "compare_subreddits"
    description = """Compare community opinions across subreddits. Takes sentiment data 
    grouped by subreddit and produces a comparison showing sentiment differences."""
    args_schema = RedditSubredditComparatorArgs

    def run(self, data: Union[str, Dict, List] = "") -> Dict[str, Any]:
        """Compare opinions across subreddits from sentiment data."""

        # Parse string input
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return {"error": "Could not parse input data", "comparison": {}}

        if not isinstance(data, dict):
            return {
                "error": "Expected dict input with sentiment data",
                "comparison": {},
            }

        # If we have by_subreddit data from sentiment analyzer, use it directly
        by_subreddit = data.get("by_subreddit")
        sentiments = data.get("sentiments", [])

        if by_subreddit and isinstance(by_subreddit, dict):
            return self._compare_from_aggregates(by_subreddit, sentiments, data)

        # If we have raw Reddit data with results (multi-subreddit)
        if "results" in data and isinstance(data["results"], list):
            return self._compare_from_raw_posts(data)

        # If we have sentiment data with source-tagged sentiments
        if sentiments:
            return self._compare_from_sentiments(sentiments, data)

        return {"error": "No subreddit-grouped data found to compare", "comparison": {}}

    def _compare_from_aggregates(
        self, by_subreddit: dict, sentiments: list, full_data: dict
    ) -> Dict[str, Any]:
        """Build comparison from pre-computed subreddit aggregates."""
        subreddits = list(by_subreddit.keys())

        comparison = {}
        for sub, stats in by_subreddit.items():
            # Get top posts for this subreddit
            sub_posts = [s for s in sentiments if s.get("source") == sub]
            most_positive = max(
                sub_posts, key=lambda x: x["sentiment"]["compound"], default=None
            )
            most_negative = min(
                sub_posts, key=lambda x: x["sentiment"]["compound"], default=None
            )

            comparison[sub] = {
                "post_count": stats.get("post_count", 0),
                "avg_compound": stats.get("avg_compound", 0),
                "positive_pct": stats.get("positive_pct", 0),
                "negative_pct": stats.get("negative_pct", 0),
                "neutral_pct": stats.get("neutral_pct", 0),
                "most_positive_post": most_positive.get("text", "")
                if most_positive
                else "",
                "most_negative_post": most_negative.get("text", "")
                if most_negative
                else "",
            }

        # Rank subreddits
        ranked = sorted(
            comparison.items(), key=lambda x: x[1]["avg_compound"], reverse=True
        )

        most_positive_sub = ranked[0][0] if ranked else "N/A"
        most_negative_sub = ranked[-1][0] if ranked else "N/A"

        # Build summary
        summary_parts = [f"Comparison across {len(subreddits)} subreddits:\n"]
        for sub, stats in ranked:
            sentiment_label = (
                "positive"
                if stats["avg_compound"] >= 0.05
                else "negative"
                if stats["avg_compound"] <= -0.05
                else "neutral"
            )
            summary_parts.append(
                f"  r/{sub}: {sentiment_label} (compound={stats['avg_compound']:.4f}, "
                f"{stats['positive_pct']}% pos, {stats['negative_pct']}% neg, "
                f"{stats['post_count']} posts)"
            )

        summary_parts.append(f"\nMost positive community: r/{most_positive_sub}")
        summary_parts.append(f"Most negative community: r/{most_negative_sub}")

        return {
            "subreddit_count": len(subreddits),
            "total_posts": full_data.get(
                "post_count", sum(s.get("post_count", 0) for s in by_subreddit.values())
            ),
            "comparison": comparison,
            "ranking": [{"subreddit": sub, **stats} for sub, stats in ranked],
            "most_positive_community": most_positive_sub,
            "most_negative_community": most_negative_sub,
            "summary": "\n".join(summary_parts),
        }

    def _compare_from_sentiments(
        self, sentiments: list, full_data: dict
    ) -> Dict[str, Any]:
        """Build comparison from individual sentiment entries with source tags."""
        by_source = {}
        for s in sentiments:
            src = s.get("source", "unknown")
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(s)

        aggregates = {}
        for src, posts in by_source.items():
            compounds = [p["sentiment"]["compound"] for p in posts if "sentiment" in p]
            if compounds:
                n = len(compounds)
                aggregates[src] = {
                    "post_count": n,
                    "avg_compound": round(sum(compounds) / n, 4),
                    "positive_pct": round(
                        sum(1 for c in compounds if c >= 0.05) / n * 100, 1
                    ),
                    "negative_pct": round(
                        sum(1 for c in compounds if c <= -0.05) / n * 100, 1
                    ),
                    "neutral_pct": round(
                        sum(1 for c in compounds if -0.05 < c < 0.05) / n * 100, 1
                    ),
                }

        return self._compare_from_aggregates(aggregates, sentiments, full_data)

    def _compare_from_raw_posts(self, data: dict) -> Dict[str, Any]:
        """Build a simple comparison from raw Reddit post data (no sentiment pre-computed)."""
        results = data.get("results", [])
        comparison = {}
        for sub_data in results:
            if isinstance(sub_data, dict):
                sub = sub_data.get("subreddit", "unknown")
                posts = sub_data.get("posts", [])
                comparison[sub] = {
                    "post_count": len(posts),
                    "top_posts": [
                        p.get("title", "")[:100]
                        for p in posts[:5]
                        if isinstance(p, dict)
                    ],
                }

        return {
            "subreddit_count": len(comparison),
            "comparison": comparison,
            "note": "Raw post comparison (no sentiment scores available)",
        }


if __name__ == "__main__":
    tool = RedditSubredditComparator()
    test_data = {
        "post_count": 6,
        "by_subreddit": {
            "technology": {
                "post_count": 3,
                "avg_compound": 0.2,
                "positive_pct": 66.7,
                "negative_pct": 0,
                "neutral_pct": 33.3,
            },
            "artificial": {
                "post_count": 3,
                "avg_compound": -0.1,
                "positive_pct": 33.3,
                "negative_pct": 33.3,
                "neutral_pct": 33.3,
            },
        },
        "sentiments": [
            {
                "source": "technology",
                "text": "AI is great!",
                "sentiment": {"compound": 0.5},
                "label": "positive",
            },
            {
                "source": "technology",
                "text": "Interesting development",
                "sentiment": {"compound": 0.1},
                "label": "positive",
            },
            {
                "source": "technology",
                "text": "meh",
                "sentiment": {"compound": 0.0},
                "label": "neutral",
            },
            {
                "source": "artificial",
                "text": "Scary times",
                "sentiment": {"compound": -0.3},
                "label": "negative",
            },
            {
                "source": "artificial",
                "text": "Cool stuff",
                "sentiment": {"compound": 0.3},
                "label": "positive",
            },
            {
                "source": "artificial",
                "text": "whatever",
                "sentiment": {"compound": 0.0},
                "label": "neutral",
            },
        ],
    }
    result = tool.run(data=test_data)
    import json as j

    print(j.dumps(result, indent=2))
