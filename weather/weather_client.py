# weather/weather_client.py
# WeatherAPI.com integration — current conditions + hourly match-day forecast.
# Cache TTL: 2.5 minutes (PSL overs take ~4 min; one fresh read per over in live mode).
#
# All functions return types from utils/situation.py.
# On any API failure the caller receives None; callers fall back to
# WeatherImpact.neutral() so the app never crashes due to weather.

from __future__ import annotations

import csv
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from utils.situation import WeatherForecast, WeatherReading

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BASE_URL       = "https://api.weatherapi.com/v1"
CACHE_TTL      = 150   # 2.5 minutes — covers one PSL over (~4 min) with margin
REQUEST_TIMEOUT = 8    # seconds before giving up on API call

# ---------------------------------------------------------------------------
# MODULE-LEVEL CACHE  {cache_key: (unix_timestamp, data)}
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

_WLOG = Path(__file__).parent.parent / "weather_debug.log"

def _wlog(msg: str) -> None:
    try:
        with open(_WLOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n"); f.flush()
    except Exception:
        pass


_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _api_key() -> str:
    """Load WEATHERAPI_KEY from .env or environment. Raises clearly if missing."""
    load_dotenv(_DOTENV_PATH)
    key = os.getenv("WEATHERAPI_KEY", "").strip()
    _wlog(f"_api_key(): key_found={bool(key)}, len={len(key)}, cwd={os.getcwd()!r}")
    if not key:
        raise EnvironmentError(
            "WEATHERAPI_KEY is not set. "
            "Add it to your .env file: WEATHERAPI_KEY=your_key_here"
        )
    return key


def _cache_get(key: str) -> Optional[object]:
    """Return cached value if still fresh, else None."""
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, data = entry
    if (time.monotonic() - ts) < CACHE_TTL:
        return data
    del _cache[key]
    return None


def _cache_set(key: str, data: object) -> None:
    _cache[key] = (time.monotonic(), data)


def _dewpoint(temp_c: float, humidity_pct: float) -> float:
    """
    Magnus formula fallback — used if API doesn't return dewpoint_c directly.
    Accurate to ±0.4°C for typical PSL match conditions.
    """
    a, b = 17.27, 237.7
    humidity_pct = max(1.0, min(100.0, humidity_pct))   # clamp to valid range
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(humidity_pct / 100.0)
    return round((b * alpha) / (a - alpha), 1)


def _parse_current_reading(data: dict) -> WeatherReading:
    """Parse a WeatherAPI current.json response into a WeatherReading."""
    c = data["current"]
    temp_c    = float(c.get("temp_c", 20))
    humidity  = float(c.get("humidity", 50))
    # dewpoint_c is present in WeatherAPI responses; fall back to calculation
    dewpoint  = float(c.get("dewpoint_c") or _dewpoint(temp_c, humidity))
    return WeatherReading(
        temp_c           = temp_c,
        humidity_pct     = humidity,
        wind_kph         = float(c.get("wind_kph", 0)),
        wind_dir         = str(c.get("wind_dir", "N")),
        dewpoint_c       = dewpoint,
        precip_mm        = float(c.get("precip_mm", 0)),
        rain_probability = 0,   # not available in current endpoint
        condition        = str(c.get("condition", {}).get("text", "Unknown")),
        fetched_at       = datetime.now(),
    )


def _parse_hourly_reading(hour: dict) -> WeatherReading:
    """Parse one slot from a WeatherAPI forecast hourly array."""
    temp_c   = float(hour.get("temp_c", 20))
    humidity = float(hour.get("humidity", 50))
    dewpoint = float(hour.get("dewpoint_c") or _dewpoint(temp_c, humidity))

    # WeatherAPI returns time as "2025-03-20 19:00"
    try:
        fetched_at = datetime.strptime(str(hour.get("time", "")), "%Y-%m-%d %H:%M")
    except ValueError:
        fetched_at = datetime.now()

    return WeatherReading(
        temp_c           = temp_c,
        humidity_pct     = humidity,
        wind_kph         = float(hour.get("wind_kph", 0)),
        wind_dir         = str(hour.get("wind_dir", "N")),
        dewpoint_c       = dewpoint,
        precip_mm        = float(hour.get("precip_mm", 0)),
        rain_probability = int(hour.get("chance_of_rain", 0)),
        condition        = str(hour.get("condition", {}).get("text", "Unknown")),
        fetched_at       = fetched_at,
    )


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def get_current_weather(lat: float, lon: float) -> Optional[WeatherReading]:
    """
    Fetch current conditions for (lat, lon).
    Result is cached for CACHE_TTL seconds — safe to call every over.
    Returns None on any network or API error.
    """
    cache_key = f"current:{lat:.4f},{lon:.4f}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached   # type: ignore[return-value]

    try:
        resp = requests.get(
            f"{BASE_URL}/current.json",
            params={"key": _api_key(), "q": f"{lat},{lon}", "aqi": "no"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        reading = _parse_current_reading(resp.json())
        _cache_set(cache_key, reading)
        _wlog(f"get_current_weather OK: lat={lat}, lon={lon}, temp={reading.temp_c}, condition={reading.condition!r}")
        return reading

    except EnvironmentError as e:
        _wlog(f"get_current_weather EnvironmentError: {e}")
        return None
    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        _wlog(f"get_current_weather HTTPError {e.response.status_code}: {body}")
        return None
    except requests.RequestException as e:
        _wlog(f"get_current_weather RequestException: {e}")
        return None
    except (KeyError, ValueError) as e:
        _wlog(f"get_current_weather parse error: {e}")
        return None


def get_match_forecast(
    lat: float,
    lon: float,
    match_datetime: datetime,
) -> Optional[WeatherForecast]:
    """
    Fetch hourly forecast for the match date.
    Selects the reading nearest to match start time as match_hour_reading.
    Result is cached for CACHE_TTL seconds.
    Returns None on any error.

    Uses WeatherAPI forecast endpoint (free tier: up to 3 days ahead).
    For past dates uses the history endpoint automatically.
    """
    date_str  = match_datetime.strftime("%Y-%m-%d")
    cache_key = f"forecast:{lat:.4f},{lon:.4f}:{date_str}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached   # type: ignore[return-value]

    # Decide endpoint: forecast for future, history for past
    today = datetime.now().date()
    days_ahead = (match_datetime.date() - today).days
    if days_ahead >= 0:
        endpoint = f"{BASE_URL}/forecast.json"
        params   = {"key": _api_key(), "q": f"{lat},{lon}", "days": min(days_ahead + 1, 3), "dt": date_str}
    else:
        endpoint = f"{BASE_URL}/history.json"
        params   = {"key": _api_key(), "q": f"{lat},{lon}", "dt": date_str}

    try:
        resp = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        # Navigate to the correct forecast day
        forecast_days = data.get("forecast", {}).get("forecastday", [])
        target_day = next(
            (d for d in forecast_days if d.get("date") == date_str),
            forecast_days[0] if forecast_days else None,
        )
        if not target_day:
            return None

        hourly = [_parse_hourly_reading(h) for h in target_day.get("hour", [])]
        if not hourly:
            return None

        # Find the reading closest to match start hour
        match_hour = match_datetime.hour
        match_reading = min(
            hourly,
            key=lambda r: abs(r.fetched_at.hour - match_hour),
            default=None,
        )

        forecast = WeatherForecast(hourly=hourly, match_hour_reading=match_reading)
        _cache_set(cache_key, forecast)
        return forecast

    except EnvironmentError as e:
        print(f"[weather_client] API key error: {e}")
        return None
    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        print(f"[weather_client] HTTP {e.response.status_code} fetching forecast "
              f"(lat={lat}, lon={lon}, date={date_str}): {body}")
        return None
    except requests.RequestException as e:
        print(f"[weather_client] Network error fetching forecast: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        print(f"[weather_client] Forecast parse error: {e}")
        return None


def get_venue_weather(venue_name: str) -> Optional[WeatherReading]:
    """
    Convenience wrapper: look up venue coordinates then fetch current weather.
    Used by the dugout screen for the live weather bar.
    """
    coords = _load_venue_coords()
    row = coords.get(venue_name)
    if row is None:
        print(f"[weather_client] Unknown venue: {venue_name!r}")
        return None
    return get_current_weather(float(row["lat"]), float(row["lon"]))


def get_venue_forecast(venue_name: str, match_datetime: datetime) -> Optional[WeatherForecast]:
    """
    Convenience wrapper: look up venue coordinates then fetch match forecast.
    Used by the prep room when generating the pre-match brief.
    """
    coords = _load_venue_coords()
    row = coords.get(venue_name)
    if row is None:
        print(f"[weather_client] Unknown venue: {venue_name!r}")
        return None
    return get_match_forecast(float(row["lat"]), float(row["lon"]), match_datetime)


# ---------------------------------------------------------------------------
# VENUE COORDINATES LOADER
# ---------------------------------------------------------------------------

_venue_coords_cache: Optional[dict] = None


def _load_venue_coords() -> dict[str, dict]:
    """
    Load weather/venue_coordinates.csv once and cache in memory.
    Returns dict keyed by venue name.
    """
    global _venue_coords_cache
    if _venue_coords_cache is not None:
        return _venue_coords_cache

    coords_path = Path(__file__).parent / "venue_coordinates.csv"
    result: dict[str, dict] = {}
    with open(coords_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["venue"]] = row
    _venue_coords_cache = result
    return result


def clear_cache() -> None:
    """Force-clear the weather cache. Useful for testing."""
    _cache.clear()


# ---------------------------------------------------------------------------
# DIAGNOSTIC  (python weather/weather_client.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    load_dotenv()
    key = os.getenv("WEATHERAPI_KEY", "").strip()
    print(f"[diag] WEATHERAPI_KEY loaded: {'YES (len=%d)' % len(key) if key else 'NO — missing from .env'}")
    if not key:
        sys.exit(1)

    # Raw request so we can see exactly what the API returns
    test_cases = [
        ("Lahore (Gaddafi)", 31.5147, 74.3429),
        ("Karachi (NSK)",    24.8943, 67.0631),
    ]
    for label, lat, lon in test_cases:
        url = f"{BASE_URL}/current.json"
        params = {"key": key, "q": f"{lat},{lon}", "aqi": "no"}
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.ok:
                d = r.json()
                loc  = d.get("location", {})
                curr = d.get("current", {})
                print(f"[diag] {label}: {loc.get('name')}, {loc.get('country')} — "
                      f"{curr.get('temp_c')}°C, humidity {curr.get('humidity')}%, "
                      f"condition: {curr.get('condition', {}).get('text')}")
            else:
                print(f"[diag] {label}: HTTP {r.status_code} — {r.text[:200]}")
        except Exception as exc:
            print(f"[diag] {label}: Exception — {exc}")
