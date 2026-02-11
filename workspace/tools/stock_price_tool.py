"""Stock Price Tool - Fetch historical stock prices using free Yahoo Finance API."""

import requests
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class StockPriceArgs(BaseModel):
    symbols: List[str] = Field(
        ..., description="Stock ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])"
    )
    period: str = Field(
        "1mo",
        description="Time period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'",
    )
    interval: str = Field(
        "1d", description="Data interval: '1d' (daily), '1wk' (weekly), '1mo' (monthly)"
    )


class StockPriceTool:
    """
    Fetch historical stock prices from Yahoo Finance.

    Features:
    - Multiple stock symbols in one call
    - Configurable time periods (1 day to max history)
    - Daily, weekly, or monthly intervals
    - Returns date, open, high, low, close, volume
    - Structured output ready for charting

    No API key required. Uses Yahoo Finance public API.
    """

    name = "stock_price"
    description = """Fetch historical stock prices for one or more ticker symbols. 
    Returns date, open, high, low, close, volume. Supports configurable periods and intervals. 
    No API key needed."""
    args_schema = StockPriceArgs

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    HEADERS = {"User-Agent": "AgenticSystem-StockTool/1.0"}

    VALID_PERIODS = {
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    }
    VALID_INTERVALS = {
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    }

    def run(
        self, symbols: List[str], period: str = "1mo", interval: str = "1d"
    ) -> Dict[str, Any]:
        """Fetch stock prices for given symbols."""

        if not symbols:
            return {"success": False, "error": "No symbols provided."}

        if period not in self.VALID_PERIODS:
            period = "1mo"
        if interval not in self.VALID_INTERVALS:
            interval = "1d"

        symbols = [s.upper().strip() for s in symbols]
        all_data = {}
        errors = []

        for symbol in symbols:
            result = self._fetch_symbol(symbol, period, interval)
            if "error" in result:
                errors.append(f"{symbol}: {result['error']}")
            else:
                all_data[symbol] = result
            time.sleep(0.5)  # Rate limiting

        if not all_data:
            return {
                "success": False,
                "error": f"Failed to fetch all symbols. {'; '.join(errors)}",
            }

        # Build comparison data (aligned dates with close prices)
        comparison = self._build_comparison(all_data)

        # Build summary
        lines = ["=== Stock Price Data ==="]
        lines.append(f"Symbols: {', '.join(symbols)}")
        lines.append(f"Period: {period}, Interval: {interval}")
        lines.append("")

        for symbol, data in all_data.items():
            prices = data["close"]
            if not prices:
                continue
            start_price = prices[0]
            end_price = prices[-1]
            change = end_price - start_price
            change_pct = (change / start_price * 100) if start_price else 0
            high = max(prices)
            low = min(prices)

            lines.append(f"--- {symbol} ---")
            lines.append(f"  Start: ${start_price:.2f}  End: ${end_price:.2f}")
            lines.append(f"  Change: ${change:+.2f} ({change_pct:+.2f}%)")
            lines.append(f"  High: ${high:.2f}  Low: ${low:.2f}")
            lines.append(f"  Data points: {len(prices)}")
            lines.append("")

        if errors:
            lines.append(f"Errors: {'; '.join(errors)}")

        return {
            "success": True,
            "symbols": list(all_data.keys()),
            "period": period,
            "interval": interval,
            "data": all_data,
            "comparison": comparison,
            "errors": errors if errors else None,
            "summary": "\n".join(lines),
        }

    def _fetch_symbol(self, symbol: str, period: str, interval: str) -> Dict[str, Any]:
        """Fetch price data for a single symbol from Yahoo Finance."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/{symbol}",
                params={
                    "range": period,
                    "interval": interval,
                    "includePrePost": "false",
                },
                headers=self.HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            chart = data.get("chart", {})
            result = chart.get("result")
            if not result:
                error_msg = chart.get("error", {}).get(
                    "description", "No data returned"
                )
                return {"error": error_msg}

            r = result[0]
            timestamps = r.get("timestamp", [])
            indicators = r.get("indicators", {})
            quote = indicators.get("quote", [{}])[0]

            dates = []
            for ts in timestamps:
                t = time.gmtime(ts)
                dates.append(f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}")

            opens = [round(v, 2) if v else None for v in quote.get("open", [])]
            highs = [round(v, 2) if v else None for v in quote.get("high", [])]
            lows = [round(v, 2) if v else None for v in quote.get("low", [])]
            closes = [round(v, 2) if v else None for v in quote.get("close", [])]
            volumes = quote.get("volume", [])

            # Filter out None values
            clean_dates = []
            clean_close = []
            for i in range(len(dates)):
                if closes[i] is not None:
                    clean_dates.append(dates[i])
                    clean_close.append(closes[i])

            meta = r.get("meta", {})
            return {
                "symbol": symbol,
                "currency": meta.get("currency", "USD"),
                "exchange": meta.get("exchangeName", ""),
                "dates": clean_dates,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": clean_close,
                "volume": volumes,
            }
        except requests.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}

    def _build_comparison(self, all_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Build comparison-ready data with aligned dates."""
        if not all_data:
            return {}

        # Use dates from the first symbol as reference
        symbols = list(all_data.keys())
        ref_dates = all_data[symbols[0]].get("dates", [])

        comparison = {"dates": ref_dates}
        for symbol in symbols:
            data = all_data[symbol]
            # Build date->close lookup
            date_price = dict(zip(data.get("dates", []), data.get("close", [])))
            comparison[symbol] = [date_price.get(d) for d in ref_dates]

        return comparison


def test_tool():
    tool = StockPriceTool()
    result = tool.run(["AAPL", "GOOGL", "MSFT"], period="1mo", interval="1d")
    if result.get("summary"):
        print(result["summary"])
    else:
        print(result)


if __name__ == "__main__":
    test_tool()
