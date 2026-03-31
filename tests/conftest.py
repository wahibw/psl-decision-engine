# tests/conftest.py
# Shared fixtures for the PSL Decision Engine test suite.

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

# Ensure project root is on sys.path regardless of how pytest is invoked
PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from utils.situation import WeatherImpact, LiveMatchState


# ---------------------------------------------------------------------------
# WEATHER FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture
def neutral_weather() -> WeatherImpact:
    return WeatherImpact(
        spinner_penalty    = 1.0,
        swing_bonus        = 1.0,
        pace_bounce_bonus  = 1.0,
        yorker_reliability = 1.0,
        dl_planning_needed = False,
        dew_onset_over     = 0,
        warnings           = [],
    )


@pytest.fixture
def dew_weather() -> WeatherImpact:
    """Heavy dew — onset over 13, full effect by over 17."""
    return WeatherImpact(
        spinner_penalty    = 0.60,
        swing_bonus        = 1.15,
        pace_bounce_bonus  = 1.05,
        yorker_reliability = 0.90,
        dl_planning_needed = False,
        dew_onset_over     = 13,
        warnings           = [],
    )


@pytest.fixture
def severe_dew_weather() -> WeatherImpact:
    return WeatherImpact(
        spinner_penalty    = 0.40,
        swing_bonus        = 1.10,
        pace_bounce_bonus  = 1.0,
        yorker_reliability = 0.85,
        dl_planning_needed = False,
        dew_onset_over     = 10,
        warnings           = [],
    )


# ---------------------------------------------------------------------------
# BOWLER / SQUAD FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture
def lahore_bowlers() -> list[str]:
    return [
        "Shaheen Shah Afridi",
        "Haris Rauf",
        "Usama Mir",
        "Sikandar Raza",
        "Mustafizur Rahman",
        "Fakhar Zaman",
    ]


@pytest.fixture
def lahore_squad() -> list[str]:
    return [
        "Fakhar Zaman", "Abdullah Shafique", "Sikandar Raza",
        "Shaheen Shah Afridi", "Hussain Talat", "Tayyab Tahir",
        "Usama Mir", "Haris Rauf", "Mustafizur Rahman",
        "Zaman Khan", "Agha Salman",
    ]


@pytest.fixture
def venue_lahore() -> str:
    return "Gaddafi Stadium, Lahore"


@pytest.fixture
def venue_rawalpindi() -> str:
    return "Rawalpindi Cricket Stadium"


@pytest.fixture
def venue_karachi() -> str:
    return "National Stadium, Karachi"


# ---------------------------------------------------------------------------
# LIVE MATCH STATE FIXTURE
# ---------------------------------------------------------------------------

@pytest.fixture
def live_state_over9(lahore_bowlers, dew_weather):
    """State at start of over 9: 68/2, Shaheen 3 overs bowled."""
    from engine.bowling_plan import generate_bowling_plan
    plan = generate_bowling_plan(lahore_bowlers, dew_weather, venue="Gaddafi Stadium, Lahore")
    return LiveMatchState(
        batting_team      = "Karachi Kings",
        bowling_team      = "Lahore Qalandars",
        venue             = "Gaddafi Stadium, Lahore",
        innings           = 1,
        target            = 0,
        bowling_plan      = plan,
        current_over      = 9,
        current_score     = 68,
        current_wickets   = 2,
        current_batter1   = "Babar Azam",
        current_batter2   = "Mohammad Rizwan",
        partnership_runs  = 24,
        partnership_balls = 18,
        overs_bowled_by   = {
            "Shaheen Shah Afridi": 3,
            "Mustafizur Rahman":   2,
            "Usama Mir":           2,
            "Haris Rauf":          1,
        },
        weather_impact = dew_weather,
    )
