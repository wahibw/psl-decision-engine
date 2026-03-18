# engine/toss_advisor.py
# Toss decision advisor for PSL matches.
#
# Inputs:  venue name, WeatherImpact, match_type, opposition team name
# Output:  TossAdvice dataclass — recommendation (BAT / BOWL / TOSS-UP),
#          confidence, primary reason, and full rationale list.
#
# Decision logic (priority order):
#   1. Severe/High dew → BOWL (dew gives second-innings batting a large advantage)
#   2. Venue chase-win-pct > 60% → BOWL (venue heavily favours chasing)
#   3. Venue defend-win-pct > 60% → BAT (venue heavily favours first innings total)
#   4. Rain risk (DL) → BAT (front-load runs; first-innings score protected by DL)
#   5. Venue avg_first > avg_second by 15+ → BAT (strong first-innings advantage)
#   6. Rawalpindi (high-scoring) → BAT first (setting a big total limits chase success)
#   7. Match type: knockout → weight bat (controllable innings > chase pressure)
#   8. Medium dew → lean BOWL with lower confidence
#   9. No strong signal → TOSS-UP

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.situation import WeatherImpact

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

_VENUE_STATS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "venue_stats.csv"

_venue_stats_cache: Optional[dict] = None


def _load_venue_stats() -> dict[str, dict]:
    global _venue_stats_cache
    if _venue_stats_cache is not None:
        return _venue_stats_cache
    result: dict[str, dict] = {}
    if not _VENUE_STATS_PATH.exists():
        return result
    try:
        with open(_VENUE_STATS_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                venue = row.get("venue", "").strip()
                if venue:
                    result[venue] = row
    except Exception:
        pass
    _venue_stats_cache = result
    return result


_ZERO_NULL_KEYS = ("economy", "sr", "pct", "score", "rate")

def _fv(stats: dict, col: str, default: float) -> float:
    """Safe float read from venue stats dict.

    0.0 in economy/SR/pct/score/rate columns = missing data, not a genuine zero.
    """
    try:
        v = stats.get(col, "")
        result = float(v) if str(v).strip() not in ("", "nan") else default
        if result == 0.0 and any(k in col for k in _ZERO_NULL_KEYS):
            return default
        return result
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# OUTPUT TYPE
# ---------------------------------------------------------------------------

@dataclass
class TossAdvice:
    """
    Toss recommendation for the coaching staff.

    recommendation: "BAT" | "BOWL" | "TOSS-UP"
    confidence:     "Strong" | "Moderate" | "Marginal"
    primary_reason: One sentence — the single dominant factor.
    rationale:      Full list of factors considered (for analyst briefing).
    dew_note:       Standalone dew alert (always shown if dew expected).
    """
    recommendation: str               # "BAT" | "BOWL" | "TOSS-UP"
    confidence:     str               # "Strong" | "Moderate" | "Marginal"
    primary_reason: str
    rationale:      list[str] = field(default_factory=list)
    dew_note:       str = ""


# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def advise_toss(
    venue:        str,
    weather:      WeatherImpact,
    match_type:   str = "league",          # "league" | "qualifier" | "eliminator" | "final"
    opposition:   str = "",
) -> TossAdvice:
    """
    Return a toss recommendation based on venue history, dew, weather, and match context.

    Confidence levels:
      Strong   — one factor clearly dominant (dew severe, venue strongly one-sided)
      Moderate — two or more factors agree, no strong contradiction
      Marginal — mixed signals or no dominant factor
    """
    stats      = _load_venue_stats()
    venue_data = stats.get(venue)
    rationale: list[str] = []
    bat_score  = 0     # positive = favour bat; negative = favour bowl
    is_knockout = match_type in ("qualifier", "eliminator", "final")

    # ------------------------------------------------------------------
    # FACTOR 1 — Dew (highest weight: +/-4)
    # ------------------------------------------------------------------
    dew_note = ""
    if weather.dew_onset_over > 0:
        onset = weather.dew_onset_over
        if weather.severe_dew:
            bat_score  -= 4   # strongly favour bowl
            dew_note    = (
                f"SEVERE DEW — expected from over {onset}. "
                f"Second innings batting significantly easier on a wet ball. BOWL FIRST."
            )
            rationale.append(dew_note)
        elif weather.spinner_penalty < 0.75:   # "High" dew
            bat_score  -= 3
            dew_note    = (
                f"Heavy dew from over {onset} — bowl first to force opposition to chase "
                f"with a wet ball. Spinner effectiveness reduced {round((1-weather.spinner_penalty)*100):.0f}%."
            )
            rationale.append(dew_note)
        elif weather.spinner_penalty < 0.90:   # "Medium" dew
            bat_score  -= 1
            dew_note    = f"Moderate dew possible from over {onset} — slight edge to bowling first."
            rationale.append(dew_note)
        else:
            dew_note = f"Light dew possible (over {onset}) — not a dominant toss factor."
            rationale.append(dew_note)
    else:
        rationale.append("No dew expected — toss decision driven by venue and conditions.")

    # ------------------------------------------------------------------
    # FACTOR 2 — Rain / D/L planning (bat first to protect total)
    # ------------------------------------------------------------------
    if weather.dl_planning_needed:
        bat_score += 2
        rationale.append(
            "Rain risk: D/L method rewards teams with a completed first innings total. "
            "BAT — set a score that DLS can protect."
        )

    # ------------------------------------------------------------------
    # FACTOR 3 — Venue historical win rates
    # ------------------------------------------------------------------
    if venue_data:
        chase_win  = _fv(venue_data, "chase_win_pct",   50.0)
        defend_win = _fv(venue_data, "defend_win_pct",  50.0)
        avg_first  = _fv(venue_data, "avg_first_score", 160.0)
        avg_second = _fv(venue_data, "avg_second_score",155.0)
        score_gap  = avg_first - avg_second

        if chase_win >= 60:
            bat_score -= 2
            rationale.append(
                f"{venue}: teams batting second win {chase_win:.0f}% of matches here. "
                f"BOWL — second innings is clearly easier at this ground."
            )
        elif defend_win >= 60:
            bat_score += 2
            rationale.append(
                f"{venue}: teams batting first win {defend_win:.0f}% of matches here. "
                f"BAT — setting a total is the dominant strategy."
            )
        else:
            rationale.append(
                f"{venue}: chase {chase_win:.0f}% vs defend {defend_win:.0f}% — "
                f"roughly even historical split."
            )

        # Score gap: large first-innings advantage → prefer batting
        if score_gap >= 15:
            bat_score += 1
            rationale.append(
                f"Average first score ({avg_first:.0f}) is {score_gap:.0f} runs higher than "
                f"average second score ({avg_second:.0f}) — pitch typically plays better first up."
            )
        elif score_gap <= -10:
            bat_score -= 1
            rationale.append(
                f"Average second score ({avg_second:.0f}) is higher — pitch improves as game progresses."
            )
    else:
        rationale.append(f"No historical data for {venue!r} — venue factor not applied.")

    # ------------------------------------------------------------------
    # FACTOR 4 — Match type context (knockout = slight bat preference)
    # ------------------------------------------------------------------
    if is_knockout:
        bat_score += 1
        rationale.append(
            f"Knockout match ({match_type}): batting first gives full control of your innings. "
            f"Chase pressure in eliminators historically hurts lower-ranked teams."
        )

    # ------------------------------------------------------------------
    # FACTOR 5 — Pace/bounce conditions (bat = early wickets for pace; bowl = use conditions)
    # ------------------------------------------------------------------
    if weather.pace_bounce_bonus >= 1.15:
        bat_score -= 1
        rationale.append(
            f"Hard/cold pitch (bounce bonus {weather.pace_bounce_bonus:.2f}x) — "
            f"bowl first to exploit early pace movement."
        )

    # ------------------------------------------------------------------
    # BUILD RECOMMENDATION
    # ------------------------------------------------------------------
    if bat_score >= 3:
        recommendation = "BAT"
        confidence = "Strong" if bat_score >= 4 else "Moderate"
    elif bat_score <= -3:
        recommendation = "BOWL"
        confidence = "Strong" if bat_score <= -4 else "Moderate"
    elif bat_score == 2:
        recommendation = "BAT"
        confidence = "Moderate"
    elif bat_score == -2:
        recommendation = "BOWL"
        confidence = "Moderate"
    elif bat_score in (1, -1):
        recommendation = "BAT" if bat_score > 0 else "BOWL"
        confidence = "Marginal"
    else:
        recommendation = "TOSS-UP"
        confidence = "Marginal"

    # Primary reason = first rationale entry (highest-priority factor)
    primary = rationale[0] if rationale else "No dominant factor — call it in the air."

    return TossAdvice(
        recommendation = recommendation,
        confidence     = confidence,
        primary_reason = primary,
        rationale      = rationale,
        dew_note       = dew_note,
    )


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils.situation import WeatherImpact

    test_cases = [
        ("National Stadium, Karachi",  WeatherImpact(0.45, 1.10, 1.05, 0.90, False, 12, ["Heavy dew"]), "league"),
        ("Rawalpindi Cricket Stadium",  WeatherImpact(1.00, 1.00, 1.20, 1.00, False, 0,  []),            "league"),
        ("Dubai International Cricket Stadium", WeatherImpact(1.00, 1.05, 1.00, 0.95, True,  0,  ["Rain risk"]), "final"),
        ("Gaddafi Stadium, Lahore",    WeatherImpact(0.70, 1.15, 1.00, 0.88, False, 14, ["Moderate dew"]), "qualifier"),
    ]

    print(f"\n{'='*70}")
    print(f"  toss_advisor.py — self-test")
    print(f"{'='*70}")

    for venue, wx, mtype in test_cases:
        advice = advise_toss(venue, wx, mtype)
        print(f"\n  Venue:  {venue}")
        print(f"  Type:   {mtype}")
        print(f"  >> {advice.recommendation}  [{advice.confidence}]")
        print(f"    {advice.primary_reason[:80]}")
        for r in advice.rationale[1:]:
            print(f"    • {r[:80]}")

    print(f"\n{'='*70}\n")
