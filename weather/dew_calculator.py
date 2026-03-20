# weather/dew_calculator.py
# Converts a WeatherReading (+ optional hourly forecast) into a DewAssessment.
#
# Spec rules:
#   Dew likely when: humidity > 65% AND dewpoint within 3°C of air temp AND night match
#   Onset: typically over 12-14
#   Critical (ball slippery): by over 17-18
#
# Output feeds directly into WeatherImpact.dew_onset_over and
# LiveMatchState.dew_active.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.situation import WeatherForecast, WeatherReading

# ---------------------------------------------------------------------------
# DEW RISK THRESHOLDS
# dew_spread = air_temp - dewpoint_c
# The closer these are, the faster dew forms.
# ---------------------------------------------------------------------------

# (dew_spread_max, humidity_min) → risk label
# Evaluated in order — first match wins
_RISK_TABLE = [
    (1.0,  80, "Severe"),   # spread ≤1°C and humid  → severe
    (1.5,  70, "Severe"),   # spread ≤1.5°C and humid → severe
    (2.0,  80, "High"),
    (2.5,  70, "High"),
    (3.0,  65, "High"),
    (3.5,  65, "Medium"),
    (5.0,  55, "Medium"),
    (5.0,  40, "Low"),
]

# Venue-level baseline adjustments to onset over
# These reflect known ground characteristics (grass type, drainage, proximity to water)
_VENUE_ONSET_ADJUST = {
    "Gaddafi Stadium, Lahore":              -1,   # famously heavy dew
    "National Stadium, Karachi":             +1,   # sea breeze keeps it drier
    "Rawalpindi Cricket Stadium":           -1,   # cold nights, fast onset
    "Multan Cricket Stadium":               -1,   # plains, cold after dark
    "Arbab Niaz Stadium, Peshawar":          0,
    "Dubai International Cricket Stadium":  -2,   # desert humidity trap, very early
    "Sharjah Cricket Stadium":              -2,   # coastal, earliest onset
    "Sheikh Zayed Stadium, Abu Dhabi":      -2,   # coastal
}

# ---------------------------------------------------------------------------
# OUTPUT DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class DewAssessment:
    dew_risk:        str         # "None" | "Low" | "Medium" | "High" | "Severe"
    onset_over:      int         # over when dew becomes active (0 = no dew)
    critical_over:   int         # over when ball is noticeably slippery (onset + 3)
    spinner_warning: bool        # True if spinners need managing in death overs
    toss_advice:     str         # plain English impact on toss decision
    details:         list[str] = field(default_factory=list)  # 1-3 UI alerts

    @property
    def has_dew(self) -> bool:
        return self.dew_risk != "None"

    @property
    def is_active_at(self) -> int:
        """Alias for onset_over — used by WeatherImpact."""
        return self.onset_over

    @classmethod
    def no_dew(cls) -> DewAssessment:
        """Return when conditions are clearly dry."""
        return cls(
            dew_risk="None",
            onset_over=0,
            critical_over=0,
            spinner_warning=False,
            toss_advice="No dew expected — toss decision not weather-influenced.",
            details=[],
        )


# ---------------------------------------------------------------------------
# CORE CALCULATION
# ---------------------------------------------------------------------------

def _base_risk(dew_spread: float, humidity_pct: float) -> str:
    """
    Map (dew_spread, humidity) to a risk label using the threshold table.
    dew_spread = air_temp_c - dewpoint_c
    """
    for spread_max, humidity_min, label in _RISK_TABLE:
        if dew_spread <= spread_max and humidity_pct >= humidity_min:
            return label
    return "Low" if dew_spread <= 5.0 and humidity_pct >= 30 else "None"


def _onset_over(dew_spread: float, risk: str, venue: str, temp_c: float) -> int:
    """
    Estimate the over when dew first affects play.

    Base logic:
      risk=Severe → base onset 10
      risk=High   → base onset 12
      risk=Medium → base onset 14
      risk=Low    → base onset 17

    Adjustments:
      Colder air (< 15°C) accelerates onset by up to -2 overs
      Venue-specific offset from _VENUE_ONSET_ADJUST
    """
    if risk == "None":
        return 0

    base = {"Severe": 10, "High": 12, "Medium": 14, "Low": 17}[risk]

    # Temperature adjustment: cold = earlier condensation
    if temp_c < 12:
        base -= 2
    elif temp_c < 17:
        base -= 1
    elif temp_c > 28:
        base += 1   # hot nights slow condensation slightly

    # Venue-specific offset
    base += _VENUE_ONSET_ADJUST.get(venue, 0)

    # Never before over 6 (powerplay conditions differ) or after over 19
    return max(6, min(19, base))


def _forecast_adjusted_onset(
    forecast: WeatherForecast,
    base_onset: int,
    match_start_hour: int,
) -> int:
    """
    Use hourly forecast to refine onset estimate.
    Finds the first hour where dew_spread ≤ 2.5°C during the match window
    and translates it to an over number.
    Overs-per-hour assumed: 12 overs = 1 hour (realistic PSL pace including breaks).
    """
    if not forecast.hourly:
        return base_onset

    for reading in forecast.hourly:
        hour = reading.fetched_at.hour
        if hour < match_start_hour:
            continue
        hours_into_match = hour - match_start_hour
        over_approx = hours_into_match * 12   # ~12 overs/hr
        if over_approx > 20:
            break
        if reading.dew_spread <= 2.5 and reading.humidity_pct >= 65:
            # First forecast hour that crosses the threshold
            return max(6, min(19, over_approx))

    return base_onset


def _spinner_warning(onset_over: int, risk: str) -> bool:
    """Spinners need managing if dew onset is before over 16."""
    return onset_over > 0 and onset_over <= 16


def _toss_advice(risk: str, onset_over: int) -> str:
    if risk == "None":
        return "No dew expected — toss decision not weather-influenced."
    if risk in ("Severe", "High"):
        return (
            f"BOWL FIRST — heavy dew expected from over {onset_over}. "
            "Second innings batting is significantly easier on a wet ball."
        )
    if risk == "Medium":
        return (
            f"Lean towards bowling first — moderate dew likely from over {onset_over}. "
            "Second innings advantage probable but not decisive."
        )
    return (
        "Slight dew possible in the final overs — "
        "not a major toss factor but monitor conditions."
    )


def _build_details(
    reading: WeatherReading,
    risk: str,
    onset_over: int,
    critical_over: int,
) -> list[str]:
    """Generate 1-3 plain-English alert strings for the UI."""
    details = []

    if risk == "None":
        details.append(
            f"Dry conditions (spread {reading.dew_spread:.1f}°C, "
            f"humidity {reading.humidity_pct:.0f}%) — no dew impact."
        )
        return details

    # Dew reading summary
    details.append(
        f"Temp {reading.temp_c:.1f}°C  |  Dewpoint {reading.dewpoint_c:.1f}°C  |  "
        f"Spread {reading.dew_spread:.1f}°C  |  Humidity {reading.humidity_pct:.0f}%"
    )

    # Risk-specific warning
    if risk == "Severe":
        details.append(
            f"SEVERE DEW — onset expected over {onset_over}, "
            f"ball critically slippery by over {critical_over}. "
            "Avoid all spinners in overs 17-20."
        )
    elif risk == "High":
        details.append(
            f"Heavy dew from over {onset_over} — "
            f"spinner effectiveness drops significantly after over {onset_over}. "
            "Plan spinner allocation in overs 7-15 only."
        )
    elif risk == "Medium":
        details.append(
            f"Moderate dew expected from over {onset_over}. "
            "Monitor spinner grip — consider 1-over reduction in death overs."
        )
    else:
        details.append(
            f"Light dew possible after over {onset_over} — "
            "monitor but not a major tactical factor."
        )

    return details


# ---------------------------------------------------------------------------
# PUBLIC FUNCTION
# ---------------------------------------------------------------------------

def assess_dew(
    reading: WeatherReading,
    venue: str = "",
    is_night_match: bool = True,
    forecast: Optional[WeatherForecast] = None,
    match_start_hour: int = 19,
) -> DewAssessment:
    """
    Primary entry point. Takes a WeatherReading and returns a full DewAssessment.

    Args:
        reading:          Current or match-time weather conditions.
        venue:            Venue name (matches venue_coordinates.csv) for local adjustment.
        is_night_match:   Dew only forms at night — day matches get DewAssessment.no_dew().
        forecast:         Optional hourly forecast to refine the onset estimate.
        match_start_hour: Local hour of first ball (24h clock). Default 19 = 7 PM.

    Returns:
        DewAssessment with risk level, onset over, spinner warning, and toss advice.
    """
    # Day matches: no dew at all
    if not is_night_match:
        return DewAssessment.no_dew()

    # Evaluate base risk from current conditions
    risk = _base_risk(reading.dew_spread, reading.humidity_pct)

    if risk == "None":
        return DewAssessment.no_dew()

    # Calculate onset over
    onset = _onset_over(reading.dew_spread, risk, venue, reading.temp_c)

    # Refine with hourly forecast if available
    if forecast:
        onset = _forecast_adjusted_onset(forecast, onset, match_start_hour)

    critical = min(onset + 3, 19)

    return DewAssessment(
        dew_risk        = risk,
        onset_over      = onset,
        critical_over   = critical,
        spinner_warning = _spinner_warning(onset, risk),
        toss_advice     = _toss_advice(risk, onset),
        details         = _build_details(reading, risk, onset, critical),
    )
