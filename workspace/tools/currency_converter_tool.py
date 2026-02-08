"""Currency Converter Tool - Convert between currencies using free exchangerate.host API."""

import requests
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class CurrencyConverterArgs(BaseModel):
    amount: float = Field(..., description="Amount to convert")
    from_currency: str = Field(..., description="Source currency code (e.g., 'USD', 'EUR', 'INR')")
    to_currency: str = Field(..., description="Target currency code (e.g., 'INR', 'GBP', 'JPY')")


class CurrencyConverterTool:
    """
    Currency converter using free exchange rate APIs.
    
    Converts between any two currencies using live exchange rates.
    Supports 150+ currencies including USD, EUR, GBP, INR, JPY, etc.
    No API key required.
    """
    
    name = "currency_converter"
    description = """Convert between any two currencies using live exchange rates. 
    Supports 150+ currencies. No API key needed."""
    args_schema = CurrencyConverterArgs

    # Primary: frankfurter.app (free, no key, reliable)
    PRIMARY_URL = "https://api.frankfurter.app/latest"
    # Fallback: open.er-api.com
    FALLBACK_URL = "https://open.er-api.com/v6/latest"

    def run(self, amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """Convert amount from one currency to another."""
        from_currency = from_currency.upper().strip()
        to_currency = to_currency.upper().strip()

        if from_currency == to_currency:
            return {
                "success": True,
                "amount": amount,
                "from": from_currency,
                "to": to_currency,
                "rate": 1.0,
                "converted": amount,
                "summary": f"{amount} {from_currency} = {amount} {to_currency} (same currency)",
            }

        # Try primary API
        result = self._fetch_frankfurter(amount, from_currency, to_currency)
        if result:
            return result

        # Fallback API
        result = self._fetch_open_er(amount, from_currency, to_currency)
        if result:
            return result

        return {
            "success": False,
            "error": f"Could not fetch exchange rate for {from_currency} -> {to_currency}. "
                     "Check that both currency codes are valid (e.g., USD, EUR, INR, GBP, JPY).",
        }

    def _fetch_frankfurter(self, amount: float, from_curr: str, to_curr: str) -> Optional[Dict]:
        """Fetch rate from frankfurter.app (free, no key)."""
        try:
            resp = requests.get(
                self.PRIMARY_URL,
                params={"amount": amount, "from": from_curr, "to": to_curr},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            rates = data.get("rates", {})
            if to_curr not in rates:
                return None

            converted = rates[to_curr]
            rate = converted / amount if amount != 0 else 0

            return {
                "success": True,
                "amount": amount,
                "from": from_curr,
                "to": to_curr,
                "rate": round(rate, 6),
                "converted": round(converted, 2),
                "source": "frankfurter.app",
                "date": data.get("date", ""),
                "summary": f"{amount} {from_curr} = {converted:,.2f} {to_curr} (rate: {rate:.6f}, date: {data.get('date', 'N/A')})",
            }
        except Exception:
            return None

    def _fetch_open_er(self, amount: float, from_curr: str, to_curr: str) -> Optional[Dict]:
        """Fallback: fetch rate from open.er-api.com (free, no key)."""
        try:
            resp = requests.get(
                f"{self.FALLBACK_URL}/{from_curr}",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("result") != "success":
                return None

            rates = data.get("rates", {})
            if to_curr not in rates:
                return None

            rate = rates[to_curr]
            converted = round(amount * rate, 2)

            return {
                "success": True,
                "amount": amount,
                "from": from_curr,
                "to": to_curr,
                "rate": round(rate, 6),
                "converted": converted,
                "source": "open.er-api.com",
                "date": data.get("time_last_update_utc", ""),
                "summary": f"{amount} {from_curr} = {converted:,.2f} {to_curr} (rate: {rate:.6f})",
            }
        except Exception:
            return None


def test_tool():
    tool = CurrencyConverterTool()
    result = tool.run(500, "USD", "INR")
    print(result.get("summary", result))


if __name__ == "__main__":
    test_tool()
