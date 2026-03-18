# engine/partnership_engine.py
# Tracks current batting partnership and assesses danger level in real time.
# Called by the dugout screen after each over update.
#
# Public API:
#   assess_partnership(batter1, batter2, runs, balls, context) -> PartnershipAssessment

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJ_ROOT    = Path(__file__).resolve().parent.parent
HISTORY_PATH = PROJ_ROOT / "data" / "processed" / "partnership_history.parquet"

# ---------------------------------------------------------------------------
# DANGER THRESHOLDS  (balls together)
# ---------------------------------------------------------------------------

GROWING_THRESHOLD    = 20   # partnership past this -> "Growing"
DANGEROUS_THRESHOLD  = 30   # partnership past this -> "Dangerous"
CRITICAL_THRESHOLD   = 40   # partnership past this -> "Critical"

# Minimum historical occurrences before we cite H2H data with confidence.
# 3 was too low — one unusual match can skew pace/spin breakdown percentages.
# 8 ensures at least a small but meaningful sample before making style claims.
MIN_OCCURRENCES = 8

# League-average fallbacks used when no PSL partnership history exists for a pair.
# Source: PSL 2019-2025 all-partnership averages.
_LEAGUE_AVG_PARTNERSHIP_RUNS  = 28.0
_LEAGUE_AVG_PARTNERSHIP_BALLS = 22.0
_LEAGUE_AVG_OVER_BROKEN       = 10.0
_LEAGUE_AVG_PACE_PCT          = 50.0
_LEAGUE_AVG_SPIN_PCT          = 35.0
_LEAGUE_AVG_CHANGE_PCT        = 40.0


# ---------------------------------------------------------------------------
# DATA CLASS
# ---------------------------------------------------------------------------

@dataclass
class PartnershipAssessment:
    batter1:                str
    batter2:                str
    current_runs:           int
    current_balls:          int
    current_sr:             float

    # Historical data for this pair (career)
    historical_avg_runs:    float
    historical_avg_balls:   float
    historical_occurrences: int

    # Danger
    danger_level:           str     # "Monitoring" | "Growing" | "Dangerous" | "Critical"
    danger_trigger:         str     # plain English reason

    # How this pair is typically broken
    break_with_pace_pct:    float
    break_with_spin_pct:    float
    break_with_change_pct:  float
    avg_over_when_broken:   float

    # Recommendation
    recommended_action:     str
    confidence:             str     # "High" | "Medium" | "Low"

    # For dugout display
    danger_score:           int     # 0-100, drives progress bar
    alert_message:          str     # one-line for situation read

    # Data quality flag: False = generic T20 league averages used (no PSL history for this pair)
    is_historical:          bool = False


# ---------------------------------------------------------------------------
# HISTORY LOADER
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_history(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _reload_history() -> None:
    _load_history.cache_clear()


def _get_pair_history(
    batter1: str,
    batter2: str,
    history: pd.DataFrame,
) -> Optional[pd.Series]:
    """
    Look up career (season=0) history for a batter pair.
    Returns None if pair not in data.
    """
    b1, b2 = tuple(sorted([batter1, batter2]))
    row = history[
        (history["batter1"] == b1)
        & (history["batter2"] == b2)
        & (history["season"] == 0)
    ]
    return row.iloc[0] if not row.empty else None


# ---------------------------------------------------------------------------
# DANGER SCORE (0-100)
# ---------------------------------------------------------------------------

def _compute_danger_score(
    current_balls:  int,
    current_sr:     float,
    hist_avg_balls: float,
    hist_avg_runs:  float,
) -> int:
    """
    0-100 score that fills a progress bar green -> amber -> red.

    Factors:
      - Balls together (primary): past 40 = exponentially more dangerous
      - SR above historical average: accelerating = extra danger
    """
    # Balls component: 0-70 points
    if current_balls >= CRITICAL_THRESHOLD:
        balls_score = 70 + min(30, (current_balls - CRITICAL_THRESHOLD) * 2)
    elif current_balls >= DANGEROUS_THRESHOLD:
        balls_score = 50 + int((current_balls - DANGEROUS_THRESHOLD) / (CRITICAL_THRESHOLD - DANGEROUS_THRESHOLD) * 20)
    elif current_balls >= GROWING_THRESHOLD:
        balls_score = 25 + int((current_balls - GROWING_THRESHOLD) / (DANGEROUS_THRESHOLD - GROWING_THRESHOLD) * 25)
    else:
        balls_score = int(current_balls / GROWING_THRESHOLD * 25)

    # SR component: 0-30 points — are they accelerating vs their norm?
    # Divisor /2 (was /3): a 60-point SR surge now scores 30/30 (max danger),
    # not 20/30. A pair batting at 180 vs a 120 norm is match-winning pace — it
    # should register at the top of the danger scale, not two-thirds of it.
    if hist_avg_balls > 0:
        expected_sr = (hist_avg_runs / hist_avg_balls * 100) if hist_avg_balls > 0 else 120.0
        sr_delta    = current_sr - expected_sr
        sr_score    = int(min(30, max(0, sr_delta / 2)))
    else:
        sr_score = 10 if current_sr >= 140 else 0

    return min(100, balls_score + sr_score)


# ---------------------------------------------------------------------------
# ACTION RECOMMENDATION
# ---------------------------------------------------------------------------

def _recommend_action(
    batter1:         str,
    batter2:         str,
    current_balls:   int,
    pace_pct:        float,
    spin_pct:        float,
    change_pct:      float,
    occurrences:     int,
) -> tuple[str, str]:
    """
    Returns (recommended_action, confidence).
    """
    if occurrences < MIN_OCCURRENCES:
        confidence = "Low"
        if current_balls >= CRITICAL_THRESHOLD:
            action = "Partnership critical — consider any bowling change to disrupt rhythm."
        else:
            action = "Insufficient H2H data — use general tactics."
        return action, confidence

    confidence = "High" if occurrences >= 12 else "Medium"   # High only with strong sample
    b1_last = batter1.split()[-1]
    b2_last = batter2.split()[-1]

    if pace_pct >= 50 and pace_pct >= spin_pct:
        action = (
            f"Bring pace — breaks {b1_last}/{b2_last} "
            f"{pace_pct:.0f}% of the time historically."
        )
    elif spin_pct >= 50 and spin_pct > pace_pct:
        action = (
            f"Bring spin — breaks {b1_last}/{b2_last} "
            f"{spin_pct:.0f}% of the time historically."
        )
    elif change_pct >= 50:
        action = (
            f"Bowling change works — {change_pct:.0f}% of breaks came within 2 balls "
            f"of a change."
        )
    else:
        action = (
            f"No strong pattern — no clear preference pace vs spin for "
            f"{b1_last}/{b2_last}."
        )

    return action, confidence


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def assess_partnership(
    batter1:      str,
    batter2:      str,
    current_runs: int,
    current_balls:int,
    history_path: Optional[Path] = None,
) -> PartnershipAssessment:
    """
    Assess the danger level and tactical recommendation for the current partnership.

    Args:
        batter1:       Name of first batter (either striker or non-striker)
        batter2:       Name of second batter
        current_runs:  Runs in the current partnership so far
        current_balls: Balls in the current partnership so far
        history_path:  Override for partnership_history.parquet path

    Returns:
        PartnershipAssessment with danger level, historical context, and recommendation.
    """
    hp    = str(history_path or HISTORY_PATH)
    history = _load_history(hp)

    raw_sr     = round(current_runs / current_balls * 100, 1) if current_balls > 0 else 0.0

    # Look up historical pair data
    hist = _get_pair_history(batter1, batter2, history)

    T20_GENERIC_SR = 120.0   # fallback expected SR for unknown pairs

    if hist is not None:
        hist_avg_runs   = float(hist.get("avg_runs",   0) or 0)
        hist_avg_balls  = float(hist.get("avg_balls",  0) or 0)
        occurrences     = int(hist.get("occurrences",  0) or 0)
        pace_pct        = float(hist.get("broken_by_pace_pct",           0) or 0)
        spin_pct        = float(hist.get("broken_by_spin_pct",           0) or 0)
        change_pct      = float(hist.get("broken_by_bowling_change_pct", 0) or 0)
        avg_over_broken = float(hist.get("avg_over_when_broken",         0) or 0)
    else:
        hist_avg_runs   = _LEAGUE_AVG_PARTNERSHIP_RUNS
        hist_avg_balls  = _LEAGUE_AVG_PARTNERSHIP_BALLS
        occurrences     = 0
        pace_pct        = _LEAGUE_AVG_PACE_PCT
        spin_pct        = _LEAGUE_AVG_SPIN_PCT
        change_pct      = _LEAGUE_AVG_CHANGE_PCT
        avg_over_broken = _LEAGUE_AVG_OVER_BROKEN

    # Small-sample SR blending: pairs with <5 occurrences have unreliable histories.
    # Blend observed SR toward T20 generic (120) weighted by data confidence.
    # At 0 occurrences: use T20 generic entirely.
    # At 5+ occurrences: use raw_sr (full confidence in historical pattern).
    # Formula: current_sr = raw_sr × (occ/5) + T20_GENERIC_SR × (1 − occ/5)
    blend_weight = min(1.0, occurrences / 8.0)   # full confidence at MIN_OCCURRENCES (8)
    hist_sr      = (hist_avg_runs / hist_avg_balls * 100) if hist_avg_balls > 0 else T20_GENERIC_SR
    current_sr   = round(raw_sr * blend_weight + hist_sr * (1.0 - blend_weight), 1)

    # Danger level
    if current_balls >= CRITICAL_THRESHOLD:
        danger_level = "Critical"
        danger_trigger = (
            f"Partnership past {current_balls} balls — "
            f"historically partnerships accelerate sharply after {DANGEROUS_THRESHOLD} balls. "
            f"Intervention needed NOW."
        )
    elif current_balls >= DANGEROUS_THRESHOLD:
        danger_level = "Dangerous"
        danger_trigger = (
            f"{current_balls} balls together — dangerous territory. "
            f"Historical PSL avg for this stage: pairs accelerate 40-60% above early-partnership SR."
        )
    elif current_balls >= GROWING_THRESHOLD:
        danger_level = "Growing"
        danger_trigger = (
            f"{current_balls} balls — partnership building. "
            f"Monitor closely; both batters have read the conditions."
        )
    else:
        danger_level = "Monitoring"
        danger_trigger = f"New partnership — {current_balls} balls so far. Assess in next few overs."

    # Danger score — use blended SR (small-sample pairs converge toward expected)
    danger_score = _compute_danger_score(
        current_balls, current_sr, hist_avg_balls, hist_avg_runs
    )

    # Recommendation
    action, confidence = _recommend_action(
        batter1, batter2, current_balls,
        pace_pct, spin_pct, change_pct, occurrences
    )

    # One-line alert for situation read
    b1_last = batter1.split()[-1]
    b2_last = batter2.split()[-1]
    if danger_level == "Critical":
        alert = (
            f"CRITICAL: {b1_last}/{b2_last} {current_runs} runs / {current_balls} balls "
            f"({current_sr:.0f} SR) — {action}"
        )
    elif danger_level == "Dangerous":
        alert = (
            f"{b1_last}/{b2_last} {current_runs} runs / {current_balls} balls — DANGEROUS. "
            f"{action}"
        )
    elif danger_level == "Growing":
        alert = (
            f"{b1_last}/{b2_last} growing: {current_runs} off {current_balls} balls. "
            f"Consider change this over."
        )
    else:
        alert = f"Partnership: {b1_last}/{b2_last} — {current_runs} runs / {current_balls} balls."

    return PartnershipAssessment(
        batter1                = batter1,
        batter2                = batter2,
        current_runs           = current_runs,
        current_balls          = current_balls,
        current_sr             = current_sr,
        historical_avg_runs    = round(hist_avg_runs,  1),
        historical_avg_balls   = round(hist_avg_balls, 1),
        historical_occurrences = occurrences,
        danger_level           = danger_level,
        danger_trigger         = danger_trigger,
        break_with_pace_pct    = pace_pct,
        break_with_spin_pct    = spin_pct,
        break_with_change_pct  = change_pct,
        avg_over_when_broken   = avg_over_broken,
        recommended_action     = action,
        confidence             = confidence,
        danger_score           = danger_score,
        alert_message          = alert,
        is_historical          = hist is not None,
    )


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ("Babar Azam",    "Mohammad Rizwan", 12,  8),
        ("Babar Azam",    "Mohammad Rizwan", 47, 26),
        ("Babar Azam",    "Mohammad Rizwan", 89, 45),
        ("Fakhar Zaman",  "Abdullah Shafique", 62, 38),
        ("Shaheen Shah Afridi", "Haris Rauf", 8, 12),  # bowlers tail
    ]

    print(f"\n{'='*65}")
    print(f"  partnership_engine.py -- self test")
    print(f"{'='*65}")

    for b1, b2, runs, balls in test_cases:
        pa = assess_partnership(b1, b2, runs, balls)
        print(f"\n  {b1} / {b2}")
        print(f"  Current: {runs} runs / {balls} balls  (SR {pa.current_sr:.0f})")
        print(f"  History: avg {pa.historical_avg_runs} runs / {pa.historical_avg_balls} balls "
              f"({pa.historical_occurrences} occurrences)")
        print(f"  Danger:  [{pa.danger_level}]  Score: {pa.danger_score}/100")
        print(f"  Pace brk: {pa.break_with_pace_pct:.0f}%  "
              f"Spin brk: {pa.break_with_spin_pct:.0f}%  "
              f"Change: {pa.break_with_change_pct:.0f}%")
        print(f"  Action:  {pa.recommended_action}")
        print(f"  Alert:   {pa.alert_message}")

    print(f"\n{'='*65}\n")
