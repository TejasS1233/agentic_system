"""Weather Forecast Tool - Get weather forecasts for cities using Open-Meteo API (free, no API key)."""

import requests
from typing import Dict, Any
from pydantic import BaseModel, Field


class WeatherForecastArgs(BaseModel):
    city: str = Field(
        ...,
        description="City name to get weather forecast for (e.g., 'London', 'New York', 'Tokyo')",
    )
    days: int = Field(5, description="Number of forecast days (1-16)", ge=1, le=16)


class WeatherForecastTool:
    """
    Weather forecast tool using the free Open-Meteo API.

    Fetches multi-day weather forecasts for any city worldwide:
    - Daily temperature (min/max)
    - Precipitation amount and probability
    - Wind speed
    - Weather condition descriptions
    - Rainy day identification

    No API key required. Uses Open-Meteo geocoding + forecast APIs.
    """

    name = "weather_forecast"
    description = """Get weather forecasts for any city. Returns daily temperature, precipitation, 
    wind speed, and weather conditions. Identifies rainy days. Free API, no key needed."""
    args_schema = WeatherForecastArgs

    GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    # WMO Weather interpretation codes
    WMO_CODES = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}

    def run(self, city: str, days: int = 5) -> Dict[str, Any]:
        """Fetch weather forecast for a city."""

        # Step 1: Geocode the city name to coordinates
        geo = self._geocode(city)
        if "error" in geo:
            return geo

        # Step 2: Fetch forecast
        forecast = self._get_forecast(geo["latitude"], geo["longitude"], days)
        if "error" in forecast:
            return forecast

        # Step 3: Build structured result
        daily = forecast.get("daily", {})
        dates = daily.get("time", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        precip_sum = daily.get("precipitation_sum", [])
        precip_prob = daily.get("precipitation_probability_max", [])
        wind_max = daily.get("wind_speed_10m_max", [])
        weather_code = daily.get("weather_code", [])

        daily_forecasts = []
        rainy_days = []

        for i in range(len(dates)):
            code = weather_code[i] if i < len(weather_code) else 0
            condition = self.WMO_CODES.get(code, f"Unknown ({code})")
            is_rainy = code in self.RAIN_CODES or (
                i < len(precip_sum) and precip_sum[i] > 1.0
            )

            day_data = {
                "date": dates[i],
                "temp_max_c": temp_max[i] if i < len(temp_max) else None,
                "temp_min_c": temp_min[i] if i < len(temp_min) else None,
                "precipitation_mm": precip_sum[i] if i < len(precip_sum) else 0,
                "precip_probability_%": precip_prob[i] if i < len(precip_prob) else 0,
                "wind_max_kmh": wind_max[i] if i < len(wind_max) else None,
                "condition": condition,
                "is_rainy": is_rainy,
            }
            daily_forecasts.append(day_data)

            if is_rainy:
                rainy_days.append(dates[i])

        # Build readable output
        lines = [
            f"=== Weather Forecast: {geo['name']}, {geo.get('country', '')} ===",
            f"Coordinates: {geo['latitude']}°N, {geo['longitude']}°E",
            f"Forecast days: {len(daily_forecasts)}",
            "",
        ]

        for d in daily_forecasts:
            rain_marker = " [RAIN]" if d["is_rainy"] else ""
            lines.append(f"{d['date']}{rain_marker}")
            lines.append(f"   Temp: {d['temp_min_c']}°C — {d['temp_max_c']}°C")
            lines.append(f"   Condition: {d['condition']}")
            lines.append(
                f"   Precipitation: {d['precipitation_mm']}mm (prob: {d['precip_probability_%']}%)"
            )
            lines.append(f"   Wind: {d['wind_max_kmh']} km/h")
            lines.append("")

        if rainy_days:
            lines.append(f"Rainy days: {', '.join(rainy_days)}")
        else:
            lines.append("No rainy days in the forecast period.")

        return {
            "success": True,
            "city": geo["name"],
            "country": geo.get("country", ""),
            "coordinates": {"lat": geo["latitude"], "lon": geo["longitude"]},
            "forecast": daily_forecasts,
            "rainy_days": rainy_days,
            "summary": "\n".join(lines),
        }

    def _geocode(self, city: str) -> Dict[str, Any]:
        """Convert city name to coordinates using Open-Meteo Geocoding API."""
        try:
            resp = requests.get(
                self.GEOCODE_URL,
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                return {"error": f"City not found: '{city}'. Try a more specific name."}

            r = results[0]
            return {
                "name": r.get("name", city),
                "country": r.get("country", ""),
                "latitude": r["latitude"],
                "longitude": r["longitude"],
            }
        except requests.RequestException as e:
            return {"error": f"Geocoding failed: {str(e)}"}

    def _get_forecast(self, lat: float, lon: float, days: int) -> Dict[str, Any]:
        """Fetch weather forecast from Open-Meteo API."""
        try:
            resp = requests.get(
                self.FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": ",".join(
                        [
                            "temperature_2m_max",
                            "temperature_2m_min",
                            "precipitation_sum",
                            "precipitation_probability_max",
                            "wind_speed_10m_max",
                            "weather_code",
                        ]
                    ),
                    "forecast_days": min(days, 16),
                    "timezone": "auto",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            return {"error": f"Forecast fetch failed: {str(e)}"}


def test_tool():
    tool = WeatherForecastTool()
    result = tool.run("Tokyo", days=5)
    if isinstance(result, dict) and "summary" in result:
        print(result["summary"])
    else:
        print(result)


if __name__ == "__main__":
    test_tool()
