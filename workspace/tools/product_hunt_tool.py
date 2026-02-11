"""Product Hunt Tool - Scrape trending products from Product Hunt."""

import requests
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import re


class ProductHuntArgs(BaseModel):
    time_filter: Literal["today", "yesterday", "week", "month"] = Field(
        "today", description="Time filter: 'today', 'yesterday', 'week', or 'month'"
    )
    limit: int = Field(10, description="Number of products to fetch (max 20)")


class ProductHuntTool:
    """
    Product Hunt Trending Products Scraper.

    Scrapes Product Hunt to get trending tech products:
    - Today's products
    - Yesterday's products
    - This week's top products
    - This month's top products

    Returns product names, taglines, votes, and URLs.
    """

    name = "product_hunt"
    description = """Get trending products from Product Hunt. Fetch today's, yesterday's, weekly, 
    or monthly top products with names, taglines, votes, and links. Great for discovering new tech tools."""
    args_schema = ProductHuntArgs

    BASE_URL = "https://www.producthunt.com"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    def run(self, time_filter: str = "today", limit: int = 10) -> Dict[str, Any]:
        """Fetch trending products from Product Hunt."""

        limit = min(limit, 20)

        try:
            # Build URL based on time filter
            if time_filter == "today":
                url = self.BASE_URL
            elif time_filter == "yesterday":
                url = f"{self.BASE_URL}/time-travel"
            elif time_filter == "week":
                url = f"{self.BASE_URL}/leaderboard/weekly"
            elif time_filter == "month":
                url = f"{self.BASE_URL}/leaderboard/monthly"
            else:
                url = self.BASE_URL

            # Fetch page
            headers = {"User-Agent": self.USER_AGENT}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            products = []

            # Product Hunt uses dynamic loading, so we parse what's available
            # Look for product cards
            for card in soup.select(
                "[data-test='post-item'], .styles_item__Dk_nz, article"
            ):
                try:
                    # Try different selectors for name
                    name_elem = card.select_one(
                        "h3, [data-test='post-name'], .styles_name__J_qQr, a[href^='/posts/'] strong"
                    )
                    if not name_elem:
                        continue

                    name = name_elem.get_text(strip=True)
                    if not name:
                        continue

                    # Tagline
                    tagline_elem = card.select_one(
                        "[data-test='post-tagline'], .styles_tagline__kFXVh, p"
                    )
                    tagline = (
                        tagline_elem.get_text(strip=True) if tagline_elem else None
                    )

                    # Votes
                    votes_elem = card.select_one(
                        "[data-test='vote-button'], .styles_voteCount__zwuqk, button span"
                    )
                    votes = 0
                    if votes_elem:
                        votes_text = votes_elem.get_text(strip=True)
                        votes_match = re.search(r"(\d+)", votes_text)
                        if votes_match:
                            votes = int(votes_match.group(1))

                    # URL
                    link_elem = card.select_one("a[href^='/posts/']")
                    product_url = None
                    if link_elem:
                        href = link_elem.get("href", "")
                        product_url = (
                            f"{self.BASE_URL}{href}" if href.startswith("/") else href
                        )

                    # Topics/Tags
                    topics = []
                    for topic in card.select("a[href^='/topics/']"):
                        topics.append(topic.get_text(strip=True))

                    products.append(
                        {
                            "name": name,
                            "tagline": tagline,
                            "votes": votes,
                            "topics": topics[:3],
                            "url": product_url,
                        }
                    )

                    if len(products) >= limit:
                        break

                except Exception:
                    continue

            # If we didn't find products with structured selectors, try simpler approach
            if not products:
                # Fallback: look for any product links
                for link in soup.select("a[href^='/posts/']")[:limit]:
                    name = link.get_text(strip=True)
                    if name and len(name) > 2:
                        products.append(
                            {
                                "name": name,
                                "url": f"{self.BASE_URL}{link.get('href')}",
                            }
                        )

            return {
                "time_filter": time_filter,
                "count": len(products),
                "products": products,
            }

        except Exception as e:
            return {"error": f"Failed to fetch from Product Hunt: {str(e)}"}


# For direct testing
if __name__ == "__main__":
    tool = ProductHuntTool()
    result = tool.run(time_filter="today", limit=5)
    print(result)
