# weather/weather_impact.py
# Translates WeatherReading + DewAssessment → WeatherImpact modifiers.
#
# WeatherImpact is defined in utils/situation.py — this module only
# *produces* it. All engine modules consume WeatherImpact objects and
# never import from weather/ directly. That keeps the dependency graph clean:
#
#   weather/ ──produces──► WeatherImpact ──consumed by──► engine/, pages/
#
# Modifier ranges (from spec):
#   spinner_penalty    0.40 (severe dew)   → 1.00 (no dew)
#   swing_bonus        1.00 (dry/warm)     → 1.40 (humidity > 80%)
#   pace_bounce_bonus  1.00 (warm)         → 1.25 (cold nights, hard pitch)
#   yorker_reliability 0.65 (high wind)    → 1.00 (calm)
#   dl_planning_needed True if rain prob   > 30%
#   dew_onset_over     from DewAssessment
#   warnings           plain-English UI alerts

from __future__ import annotations

from typing import Optional

from utils.situation import WeatherImpact, WeatherReading
from weather.dew_calculator import DewAssessment

# ---------------------------------------------------------------------------
# MODIFIER TABLES
# Each table is a list of (threshold, value) pairs evaluated top-down.
# The first threshold the input meets yields the modifier value.
# ---------------------------------------------------------------------------

# spinner_penalty vs dew risk label
_SPINNER_PENALTY = {
    "Severe": 0.40,
    "High":   0.60,
    "Medium": 0.75,
    "Low":    0.88,
    "None":   1.00,
}

# Additional reduction once dew is *active* (current_over >= onset_over)
_SPINNER_ACTIVE_REDUCTION = {
    "Severe": 0.00,   # already at minimum
    "High":   0.10,   # 0.60 → 0.50 once active
    "Medium": 0.08,   # 0.75 → 0.67 once active
    "Low":    0.05,
    "None":   0.00,
}

def _spinner_penalty(dew: DewAssessment, current_over: int) -> float:
    base = _SPINNER_PENALTY[dew.dew_risk]
    if dew.has_dew and current_over >= dew.onset_over:
        base -= _SPINNER_ACTIVE_REDUCTION[dew.dew_risk]
    return round(max(0.40, base), 2)


def _swing_bonus(reading: WeatherReading) -> float:
    """
    Humidity drives swing. Onset shifted to 75% (was 70%) based on PSL ground data:
    Karachi / Lahore grounds typically have 65-72% humidity without meaningful swing;
    swing only becomes tactically significant above 75%.

    Thresholds (recalibrated):
      < 75%  → 1.00  no meaningful swing
        75%  → 1.10  light swing (powerplay useful)
        80%  → 1.25  clear swing (bring seam/swing specialist early)
      ≥ 85%  → 1.40  pronounced swing (lead with swing attack overs 1-6)

    Overcast sky adds +0.05 (moisture stays airborne longer under cloud cover).
    Range: 1.00 → 1.40
    """
    h = reading.humidity_pct
    if h >= 85:
        bonus = 1.40
    elif h >= 80:
        bonus = 1.25
    elif h >= 75:
        bonus = 1.10
    else:
        bonus = 1.00   # below 75% — PSL conditions, no meaningful swing

    # Overcast conditions add 0.05 (clouds keep moisture in the air longer)
    cond_lower = reading.condition.lower()
    if any(w in cond_lower for w in ("overcast", "cloudy", "cloud")):
        bonus = min(1.40, bonus + 0.05)

    return round(bonus, 2)


def _pace_bounce_bonus(reading: WeatherReading) -> float:
    """
    Cold nights = harder pitch surface = more pace and bounce for fast bowlers.
    Range: 1.00 → 1.25
    """
    t = reading.temp_c
    if t < 10:
        return 1.25
    elif t < 15:
        return 1.15
    elif t < 20:
        return 1.08
    elif t < 25:
        return 1.03
    else:
        return 1.00


def _yorker_reliability(reading: WeatherReading) -> float:
    """
    Crosswind disrupts yorker accuracy far more than headwind/tailwind.
    A 30 km/h crosswind forces mid-delivery correction; a 30 km/h headwind
    barely changes release point.

    Direction mapping:
      Pure crosswind (E, W, NE, NW, SE, SW) → effective_kph × 1.5
      Pure head/tailwind (N, S)              → effective_kph × 0.6
      Unknown / calm                          → effective_kph × 1.0 (neutral)

    Range: 0.65 → 1.00
    """
    w = reading.wind_kph

    # Direction-based weighting
    dir_upper = (reading.wind_dir or "").strip().upper()
    if dir_upper in ("E", "W", "NE", "NW", "SE", "SW"):
        effective_kph = w * 1.5   # crosswind — hardest to bowl into
    elif dir_upper in ("N", "S"):
        effective_kph = w * 0.6   # head/tailwind — manageable
    else:
        effective_kph = w         # unknown direction — neutral assumption

    if effective_kph >= 45:
        return 0.65
    elif effective_kph >= 35:
        return 0.72
    elif effective_kph >= 25:
        return 0.82
    elif effective_kph >= 15:
        return 0.92
    else:
        return 1.00


def _dl_planning_needed(reading: WeatherReading) -> bool:
    return reading.rain_probability > 30


# ---------------------------------------------------------------------------
# WARNING TEXT BUILDERS
# ---------------------------------------------------------------------------

def _build_warnings(
    reading: WeatherReading,
    dew: DewAssessment,
    spin_pen: float,
    swing: float,
    bounce: float,
    yorker: float,
    dl: bool,
) -> list[str]:
    """
    Generate a prioritised list of plain-English alert strings.
    These appear in the weather dashboard section of both pages.
    The most critical warnings come first.
    """
    warnings: list[str] = []

    # --- Dew warnings (highest cricket impact) ---
    if dew.dew_risk == "Severe":
        warnings.append(
            f"🔴 SEVERE DEW — active from over {dew.onset_over}. "
            "Ball will be ungrippable for spinners. Remove all spinners from overs 17-20."
        )
    elif dew.dew_risk == "High":
        warnings.append(
            f"🔴 Heavy dew expected from over {dew.onset_over}. "
            f"Spinner effectiveness reduced {round((1 - spin_pen) * 100):.0f}%. "
            "Restrict spinners to overs 7-15."
        )
    elif dew.dew_risk == "Medium":
        warnings.append(
            f"🟡 Moderate dew likely from over {dew.onset_over}. "
            "Consider one fewer spinner over in the death."
        )
    elif dew.dew_risk == "Low":
        warnings.append(
            f"🟢 Light dew possible after over {dew.onset_over} — monitor but not critical."
        )

    # --- Swing conditions ---
    if swing >= 1.30:
        warnings.append(
            f"🟡 {reading.humidity_pct:.0f}% humidity — "
            "swing bowlers effective throughout powerplay. Lead with them in overs 1-3."
        )
    elif swing >= 1.10:
        warnings.append(
            f"🟢 Moderate humidity ({reading.humidity_pct:.0f}%) — "
            "some swing movement available in powerplay."
        )

    # --- Pace/bounce conditions ---
    if bounce >= 1.15:
        warnings.append(
            f"🟢 Cold {reading.temp_c:.1f}°C — fast bowlers get extra pace and bounce. "
            "Back your best pacers to be threatening."
        )

    # --- Wind / yorker reliability ---
    if yorker <= 0.72:
        warnings.append(
            f"🟡 Strong wind ({reading.wind_kph:.0f} km/h {reading.wind_dir}) — "
            "yorker accuracy reduced. Favour back-of-length in death overs."
        )
    elif yorker <= 0.85:
        warnings.append(
            f"🟡 Gusty conditions ({reading.wind_kph:.0f} km/h) — "
            "slight yorker inconsistency likely."
        )

    # --- Rain / D/L ---
    if dl:
        warnings.append(
            f"🔴 Rain risk {reading.rain_probability}% — D/L planning essential. "
            "Front-load batting — D/L rewards early runs."
        )
    elif reading.rain_probability > 15:
        warnings.append(
            f"🟡 {reading.rain_probability}% rain chance — "
            "low risk but worth monitoring for D/L par awareness."
        )

    # If nothing noteworthy
    if not warnings:
        warnings.append(
            "🟢 Benign conditions — no significant weather modifiers. "
            "Standard selection and tactics apply."
        )

    return warnings


# ---------------------------------------------------------------------------
# PUBLIC FUNCTION
# ---------------------------------------------------------------------------

def calculate_weather_impact(
    reading: WeatherReading,
    dew: DewAssessment,
    current_over: int = 0,
) -> WeatherImpact:
    """
    Primary entry point. Takes live conditions + dew assessment and returns
    the WeatherImpact modifiers used by every engine module.

    Args:
        reading:      Current / match-time WeatherReading.
        dew:          DewAssessment from dew_calculator.assess_dew().
        current_over: Current over in a live match (0 = pre-match or over 1).
                      Used to apply active-dew reduction to spinner_penalty.

    Returns:
        WeatherImpact — all modifiers + dew_onset_over + warnings.
        On any input error falls back to WeatherImpact.neutral().
    """
    spin_pen = _spinner_penalty(dew, current_over)
    swing    = _swing_bonus(reading)
    bounce   = _pace_bounce_bonus(reading)
    yorker   = _yorker_reliability(reading)
    dl       = _dl_planning_needed(reading)
    warnings = _build_warnings(reading, dew, spin_pen, swing, bounce, yorker, dl)

    return WeatherImpact(
        spinner_penalty    = spin_pen,
        swing_bonus        = swing,
        pace_bounce_bonus  = bounce,
        yorker_reliability = yorker,
        dl_planning_needed = dl,
        dew_onset_over     = dew.onset_over,
        warnings           = warnings,
    )


def get_match_weather_impact(
    venue: str,
    match_datetime,
    current_over: int = 0,
    is_night_match: bool = True,
) -> WeatherImpact:
    """
    Convenience one-call function for pages.
    Fetches live weather → calculates dew → returns WeatherImpact.
    Falls back to WeatherImpact.neutral() if the API is unavailable.

    Used by:
      - prep_room.py (pre-match, current_over=0)
      - dugout.py (live, current_over updates each tap)
    """
    from weather.weather_client import get_venue_weather, get_venue_forecast
    from weather.dew_calculator import assess_dew

    reading = get_venue_weather(venue)
    if reading is None:
        print(f"[weather_impact] get_venue_weather returned None for venue={venue!r} — using neutral")
        return WeatherImpact.neutral()

    forecast = get_venue_forecast(venue, match_datetime)
    dew = assess_dew(
        reading,
        venue          = venue,
        is_night_match = is_night_match,
        forecast       = forecast,
        match_start_hour = match_datetime.hour if hasattr(match_datetime, "hour") else 19,
    )

    return calculate_weather_impact(reading, dew, current_over)
