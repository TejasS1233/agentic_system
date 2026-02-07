import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from workspace.tools.web_search_tool import WebSearchTool


def test_search():
    tool = WebSearchTool()
    print("Testing Text Search...")
    results = tool.run(query="Python programming", max_results=3)
    print(f"Full Result: {results}")
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    results_list = results.get("results", [])
    print(f"Results: {len(results_list)}")
    for r in results_list:
        print(f"- {r.get('title')}: {r.get('href')}")

    print("\nTesting News Search...")
    news_results = tool.run(query="AI Technology", max_results=3, search_type="news")
    print(f"News Results: {len(news_results['results'])}")
    for r in news_results["results"]:
        print(f"- {r.get('title')}: {r.get('url') or r.get('link')}")

    print("\nTesting Fetch Content...")
    content_results = tool.run(
        query="Python official site", max_results=1, fetch_content=True
    )
    if content_results["results"]:
        res = content_results["results"][0]
        print(f"Content Preview for {res.get('href')}:")
        print(res.get("content", "")[:200])


if __name__ == "__main__":
    test_search()
