# utils/situation.py
# Core data contract types for the entire app.
# Every engine module and page imports its context objects from here.
#
# Three layers of types are defined here:
#   1. Weather data types  (WeatherReading, WeatherForecast, WeatherImpact)
#      — used by weather/ modules and referenced in MatchContext / LiveMatchState
#   2. MatchContext        — pre-match planning context (static for a given match)
#   3. LiveMatchState      — mutable in-match state updated each over
#
# BowlingPlan is defined in engine/bowling_plan.py and referenced here with
# Optional[Any] to avoid a circular import. At runtime LiveMatchState carries
# the actual BowlingPlan object.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

# ---------------------------------------------------------------------------
# WEATHER DATA TYPES
# Defined here (not in weather/) so every module can import them without
# pulling in WeatherAPI client code.
# ---------------------------------------------------------------------------

@dataclass
class WeatherReading:
    """A single point-in-time weather observation or forecast slot."""
    temp_c:           float
    humidity_pct:     float
    wind_kph:         float
    wind_dir:         str           # "NE", "SW", etc.
    dewpoint_c:       float
    precip_mm:        float
    rain_probability: int           # 0-100
    condition:        str           # "Clear", "Partly cloudy", "Rain", etc.
    fetched_at:       datetime = field(default_factory=datetime.now)

    @property
    def dew_spread(self) -> float:
        """Difference between air temp and dewpoint. ≤3°C = dew likely."""
        return self.temp_c - self.dewpoint_c

    @property
    def is_dew_likely(self) -> bool:
        return self.dew_spread <= 3.0 and self.humidity_pct > 65


@dataclass
class WeatherForecast:
    """Hourly forecast for the match day."""
    hourly:              list[WeatherReading]
    match_hour_reading:  Optional[WeatherReading] = None   # reading closest to first ball

    def reading_at_over(self, over: int, first_ball_hour: int = 19) -> Optional[WeatherReading]:
        """
        Return the forecast reading that best represents conditions at a given over.
        Assumes 3-4 overs per hour. over is 0-indexed.
        """
        hours_elapsed = over // 3
        target_hour = first_ball_hour + hours_elapsed
        for r in self.hourly:
            if r.fetched_at.hour == target_hour:
                return r
        return self.match_hour_reading


@dataclass
class WeatherImpact:
    """
    Weather → cricket impact modifiers.
    Produced by weather/weather_impact.py from WeatherReading + DewAssessment.
    All multipliers are applied to base player scores in xi_selector and bowling_plan.
    """
    spinner_penalty:    float       # 0.4 (severe dew) → 1.0 (no dew)
    swing_bonus:        float       # 1.0 → 1.4 (humidity > 80%)
    pace_bounce_bonus:  float       # 1.0 → 1.25 (cold, hard pitch)
    yorker_reliability: float       # 0.65 → 1.0 (high wind reduces accuracy)
    dl_planning_needed: bool        # True if rain probability > 30%
    dew_onset_over:     int         # over when dew expected to be active (0 = no dew)
    warnings:           list[str]   # plain-English alerts for UI display
    raw_humidity:       float = 0.0 # UI hook
    raw_temp:           float = 0.0 # UI hook
    raw_wind_kph:       float = 0.0 # UI hook
    raw_wind_dir:       str   = ""  # UI hook — e.g. "NE", "SW"

    @classmethod
    def load_engine_constants(cls) -> dict:
        import json
        try:
            from pathlib import Path
            proj_root = Path(__file__).resolve().parent.parent
            with open(proj_root / 'data' / 'processed' / 'quetta_registry.json', 'r', encoding='utf-8') as f:
                return json.load(f).get("engine_constants", {})
        except Exception:
            return {}

    @classmethod
    def neutral(cls) -> WeatherImpact:
        """Safe default — no weather modifier applied. Used when API unavailable."""
        constants = cls.load_engine_constants()
        # Default to 1.0 if not found, meaning no penalty
        return cls(
            spinner_penalty=1.0,
            swing_bonus=1.0,
            pace_bounce_bonus=1.0,
            yorker_reliability=1.0,
            dl_planning_needed=False,
            dew_onset_over=0,
            warnings=["Weather data unavailable — no modifiers applied"],
        )

    # Number of overs over which dew builds from first appearance to full effect.
    # e.g. onset=13 → 0% at over 13, 25% at 14, 50% at 15, 75% at 16, 100% at 17+
    DEW_GRADIENT_OVERS: int = 4

    @property
    def dew_active_at(self) -> bool:
        """True if dew is expected at some point during the match."""
        return self.dew_onset_over > 0

    @property
    def severe_dew(self) -> bool:
        return self.spinner_penalty < 0.6

    def dew_probability_at(self, over: int) -> float:
        """
        Dew intensity at a given over as a gradient (0.0 = none → 1.0 = full).

        Ramps linearly from 0.0 at onset_over to 1.0 at onset_over + DEW_GRADIENT_OVERS.
        Returns 0.0 for day matches (dew_onset_over == 0) or overs before onset.

        Example (onset=13, gradient=4):
            over 12 → 0.00  (pre-onset)
            over 13 → 0.00  (onset — first over, still minimal)
            over 14 → 0.25
            over 15 → 0.50
            over 16 → 0.75
            over 17 → 1.00  (full effect)
        """
        onset = self.dew_onset_over
        if onset <= 0 or over <= onset:
            return 0.0
        overs_past_onset = over - onset
        return min(1.0, overs_past_onset / self.DEW_GRADIENT_OVERS)

    def spinner_penalty_at(self, over: int) -> float:
        """
        Spinner penalty at a specific over, accounting for gradual dew build-up.

        Returns 1.0 (no penalty) before onset, linearly blending to self.spinner_penalty
        (the full-dew value, e.g. 0.6) as dew intensity reaches 1.0.

        Example (spinner_penalty=0.6, onset=13):
            over 14 → 1.0 - 0.25 * (1.0 - 0.6) = 0.90  (10% impaired)
            over 15 → 1.0 - 0.50 * 0.4           = 0.80  (20% impaired)
            over 17 → 1.0 - 1.00 * 0.4           = 0.60  (full impairment)
        """
        intensity = self.dew_probability_at(over)
        if intensity <= 0.0:
            return 1.0
        return round(1.0 - intensity * (1.0 - self.spinner_penalty), 3)

    def seam_swing_bonus(self, bowling_style: str) -> float:
        """
        Per-bowler swing bonus based on bowling style.
        Only genuine swing/seam specialists get the full atmospheric humidity bonus.
        Pure fast bowlers (express pace, short-pitch) get ~35% — humid air helps
        somewhat but they're not generating outswing as their stock delivery.

        Usage:
            effective_bonus = weather.seam_swing_bonus(bowler_style)
        """
        style = bowling_style.lower()
        is_swing_seam = any(w in style for w in ("swing", "seam", "medium-fast", "left-arm medium"))
        if is_swing_seam:
            return self.swing_bonus
        return 1.0 + (self.swing_bonus - 1.0) * 0.35


# ---------------------------------------------------------------------------
# MATCH CONTEXT  —  pre-match, set once, never mutated during the match
# ---------------------------------------------------------------------------

@dataclass
class MatchContext:
    """
    Everything known before the first ball is bowled.
    Built in the Prep Room and passed to every pre-match engine function.
    """
    our_team:           str
    opposition:         str
    venue:              str
    match_datetime:     datetime
    match_type:         str             # "league" | "qualifier" | "eliminator" | "final"
    full_squad:         list[str]       # 16-18 players available for selection
    weather_forecast:   WeatherForecast
    weather_impact:     WeatherImpact
    is_home_venue:      bool = False

    @property
    def is_knockout(self) -> bool:
        return self.match_type in ("qualifier", "eliminator", "final")

    @property
    def squad_size(self) -> int:
        return len(self.full_squad)

    def venue_short(self) -> str:
        """City-level venue label for display."""
        mapping = {
            "Gaddafi Stadium, Lahore":          "Lahore",
            "National Stadium, Karachi":         "Karachi",
            "Rawalpindi Cricket Stadium":        "Rawalpindi",
            "Multan Cricket Stadium":            "Multan",
            "Dubai International Cricket Stadium": "Dubai",
            "Sharjah Cricket Stadium":           "Sharjah",
            "Sheikh Zayed Stadium, Abu Dhabi":   "Abu Dhabi",
        }
        return mapping.get(self.venue, self.venue.split(",")[0])


# ---------------------------------------------------------------------------
# LIVE MATCH STATE  —  mutable, updated once per over from the dugout screen
# ---------------------------------------------------------------------------

@dataclass
class LiveMatchState:
    """
    The complete in-match state used by all dugout-mode engine functions.
    Updated by the analyst with one tap at the start of each over.

    BowlingPlan is typed as Optional[Any] to avoid circular import with
    engine/bowling_plan.py. At runtime it will always be a BowlingPlan instance.
    """

    # --- Set once at match start ---
    batting_team:       str
    bowling_team:       str
    venue:              str
    innings:            int             # 1 or 2
    target:             int             # 0 if innings 1
    bowling_plan:       Optional[Any]   # BowlingPlan from engine/bowling_plan.py

    # --- Updated each over ---
    current_over:       int             # 1-indexed (over 1 = first over currently being bowled)
    current_score:      int
    current_wickets:    int
    current_batter1:    str
    current_batter2:    str
    partnership_runs:   int
    partnership_balls:  int
    overs_bowled_by:    dict = field(default_factory=dict)   # {"Shaheen": 3}
    wickets_this_over:  int = 0         # wickets that fell in the current/last over

    # --- Live weather (fetched every 10 min) ---
    current_weather:    Optional[WeatherReading] = None
    weather_impact:     Optional[WeatherImpact]  = None

    # --- Derived (computed via compute_derived()) ---
    balls_remaining:    int   = field(init=False)
    required_runs:      int   = field(init=False)
    rrr:                float = field(init=False)
    crr:                float = field(init=False)
    phase:              str   = field(init=False)
    dew_active:         bool  = field(init=False)
    dew_intensity:      float = field(init=False)   # 0.0 (none) → 1.0 (full), gradient

    def __post_init__(self):
        self.compute_derived()

    # ------------------------------------------------------------------
    # DERIVED FIELD CALCULATION
    # Call this once after updating any over-level field.
    # ------------------------------------------------------------------

    def compute_derived(self):
        total_balls = 20 * 6
        # current_over is 1-indexed: over 1 = first over being bowled, 0 balls completed yet.
        # Overs COMPLETED = current_over - 1.
        overs_completed = max(0, self.current_over - 1)
        balls_bowled = overs_completed * 6
        self.balls_remaining = max(0, total_balls - balls_bowled)

        # CRR = cumulative run rate since ball 1 (conventional cricket definition:
        # total runs / overs completed). Compare against RRR to gauge innings trajectory.
        # This is NOT an instantaneous or recent-overs rate.
        self.crr = (self.current_score / overs_completed) if overs_completed > 0 else 0.0

        # Innings 2 chase metrics
        if self.innings == 2 and self.target > 0:
            self.required_runs = max(0, self.target - self.current_score)
            self.rrr = (self.required_runs / (self.balls_remaining / 6)) if self.balls_remaining > 0 else 99.9
        else:
            self.required_runs = 0
            self.rrr = 0.0

        # Phase label (PSL: powerplay = overs 1-6, middle = 7-15, death = 16-20)
        if self.current_over <= 6:
            self.phase = "powerplay"
        elif self.current_over <= 15:
            self.phase = "middle"
        elif self.current_over <= 20:
            self.phase = "death"
        else:
            self.phase = "super_over"

        # Dew — binary flag for backward compatibility; gradient intensity for UI
        onset = self.weather_impact.dew_onset_over if self.weather_impact else 0
        self.dew_active    = (onset > 0) and (self.current_over >= onset)
        self.dew_intensity = (
            self.weather_impact.dew_probability_at(self.current_over)
            if self.weather_impact else 0.0
        )

    # ------------------------------------------------------------------
    # CONVENIENCE UPDATER
    # Used by the dugout screen after analyst taps "next over".
    # ------------------------------------------------------------------

    def advance_over(
        self,
        score: int,
        wickets: int,
        batter1: str,
        batter2: str,
        partnership_runs: int,
        partnership_balls: int,
        bowler_this_over: Optional[str] = None,
        weather_reading: Optional[WeatherReading] = None,
        weather_impact: Optional[WeatherImpact] = None,
    ):
        """Advance one over and recompute all derived fields."""
        self.current_over     += 1
        # Compute wickets_this_over BEFORE updating current_wickets
        self.wickets_this_over = max(0, wickets - self.current_wickets)
        self.current_score    = score
        self.current_wickets  = wickets
        self.current_batter1  = batter1
        self.current_batter2  = batter2
        self.partnership_runs  = partnership_runs
        self.partnership_balls = partnership_balls

        if bowler_this_over:
            self.overs_bowled_by[bowler_this_over] = (
                self.overs_bowled_by.get(bowler_this_over, 0) + 1
            )

        if weather_reading is not None:
            self.current_weather = weather_reading
        if weather_impact is not None:
            self.weather_impact = weather_impact

        self.compute_derived()

    def overs_remaining_for(self, bowler: str) -> int:
        """How many overs a bowler has left (max 4 per PSL rules)."""
        used = self.overs_bowled_by.get(bowler, 0)
        return max(0, 4 - used)

    @property
    def is_powerplay(self) -> bool:
        return self.phase == "powerplay"

    @property
    def is_death(self) -> bool:
        return self.phase == "death"

    @property
    def partnership_sr(self) -> float:
        if self.partnership_balls == 0:
            return 0.0
        return round((self.partnership_runs / self.partnership_balls) * 100, 1)

    @property
    def display_score(self) -> str:
        return f"{self.current_score}/{self.current_wickets}"

    @property
    def display_over(self) -> str:
        return f"Over {self.current_over}"   # 1-indexed — no offset needed
