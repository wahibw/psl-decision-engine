# engine/bowling_recommender.py
# Simplified live over recommendation.
# Takes the bowling plan + current match state -> recommends who bowls THIS over.
#
# Public API:
#   recommend_bowler_this_over(bowling_plan, state, weather) -> BowlerRecommendation

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.bowling_plan import BowlingPlan
from utils.situation import LiveMatchState, WeatherImpact


# ---------------------------------------------------------------------------
# DATA CLASS
# ---------------------------------------------------------------------------

@dataclass
class BowlerRecommendation:
    recommended_bowler: str
    plan_bowler:        str         # who the plan assigned
    is_plan_bowler:     bool        # True if recommendation matches plan
    reason:             str
    warning:            str         # "" or dew/weather warning
    overs_remaining:    int         # how many overs this bowler has left (PSL cap)
    alternatives:       list[str]   # other valid options this over
    tier_note:          str         # "" | "Medium confidence — limited PSL data" | "No PSL history — T20 career estimate"


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def recommend_bowler_this_over(
    bowling_plan: BowlingPlan,
    state:        LiveMatchState,
    weather:      WeatherImpact,
) -> BowlerRecommendation:
    """
    Recommend who should bowl the current over.

    Logic (from spec):
      1. Check bowling plan — who is assigned this over?
      2. Validate they are still available (overs remaining under PSL cap)
      3. Apply weather check — if assigned spinner and dew_active -> flag warning
      4. If plan assignment invalid -> suggest backup from plan
      5. Return recommendation with reason
    """
    current_over = state.current_over   # 1-based (over 1 = first over)

    # Find plan assignment for this over
    plan_assignment = None
    backup_assignment = None
    for oa in bowling_plan.overs:
        if oa.over == current_over:
            plan_assignment   = oa.primary_bowler
            backup_assignment = oa.backup_bowler
            plan_phase        = oa.phase
            break

    if plan_assignment is None:
        # Over not in plan (e.g. super over or beyond plan)
        import warnings
        max_planned = max((oa.over for oa in bowling_plan.overs), default=0)
        warnings.warn(
            f"Over {current_over} not found in bowling plan (plan covers overs 1-{max_planned}). "
            f"Falling back to TBD. Check for state sync issues.",
            UserWarning,
            stacklevel=2,
        )
        plan_assignment   = "TBD"
        backup_assignment = "TBD"
        plan_phase        = "Death"

    # Check overs remaining for plan bowler
    overs_bowled  = state.overs_bowled_by.get(plan_assignment, 0)
    overs_left    = state.overs_remaining_for(plan_assignment)

    warning = ""

    # Step 2: Is the plan bowler available?
    if overs_left <= 0:
        # Plan bowler exhausted — escalate to backup
        recommended = backup_assignment
        reason = (
            f"{plan_assignment} has bowled {overs_bowled} overs (cap reached). "
            f"Switching to backup: {backup_assignment}."
        )
        backup_overs_left = state.overs_remaining_for(backup_assignment)
        overs_remaining_final = backup_overs_left
    else:
        recommended = plan_assignment
        reason = f"Plan assignment: {plan_assignment} for over {current_over} ({plan_phase})."
        overs_remaining_final = overs_left

    # Step 3: Weather check — dew active and spinner assigned?
    if recommended != "TBD":
        from engine.bowling_plan import _load_player_meta
        from pathlib import Path
        PLAYER_INDEX = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_index_2026_enriched.csv"
        PLAYER_INDEX_FALLBACK = Path(__file__).resolve().parent.parent.parent / "player_index_2026_enriched.csv"
        pi_path = PLAYER_INDEX if PLAYER_INDEX.exists() else PLAYER_INDEX_FALLBACK
        try:
            meta = _load_player_meta(pi_path)
            from engine.bowling_plan import _bowl_type
            bt = _bowl_type(recommended, meta)
        except Exception:
            bt = "unknown"

        if bt == "spin" and weather.dew_active_at and weather.dew_onset_over and current_over >= weather.dew_onset_over:
            warning = (
                f"DEW ACTIVE (onset over {weather.dew_onset_over}) — "
                f"{recommended} is a spinner. Consider switching to pace this over."
            )
            # Suggest a pace alternative from the plan if available
            pace_alternatives = [
                oa.primary_bowler for oa in bowling_plan.overs
                if oa.over > current_over
                and state.overs_remaining_for(oa.primary_bowler) > 0
                and _bowl_type(oa.primary_bowler, meta) == "pace"
            ]
            if pace_alternatives:
                warning += f" Suggested: {pace_alternatives[0]}."

    # Step 4: Gather alternatives
    available = [
        b for b in state.overs_bowled_by.keys()
        if state.overs_remaining_for(b) > 0 and b != recommended
    ]
    # Add bowlers from plan who haven't bowled yet
    all_plan_bowlers = list(bowling_plan.bowler_summary.keys())
    for b in all_plan_bowlers:
        if b not in state.overs_bowled_by and b != recommended:
            available.append(b)
    available = list(dict.fromkeys(available))[:3]   # deduplicate, max 3

    # Data tier note for recommended bowler
    tier_note = ""
    try:
        import csv
        from pathlib import Path
        _pi = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_index_2026_enriched.csv"
        if _pi.exists():
            with open(_pi, newline="", encoding="utf-8") as _f:
                for _row in csv.DictReader(_f):
                    if _row.get("player_name", "").strip() == recommended:
                        _tier = int(_row.get("data_tier", 1) or 1)
                        if _tier == 2:
                            tier_note = "Medium confidence — limited PSL data"
                        elif _tier == 3:
                            _sr  = _row.get("t20_career_sr", "")
                            _eco = _row.get("t20_career_economy", "")
                            _parts = []
                            if _sr and _sr.lower() not in ("nan",""):
                                _parts.append(f"T20 SR {float(_sr):.0f}")
                            if _eco and _eco.lower() not in ("nan",""):
                                _parts.append(f"economy {float(_eco):.2f}")
                            _stat = f" ({', '.join(_parts)})" if _parts else ""
                            tier_note = f"No PSL history — T20 career estimate{_stat}"
                        break
    except Exception:
        pass

    return BowlerRecommendation(
        recommended_bowler = recommended,
        plan_bowler        = plan_assignment,
        is_plan_bowler     = (recommended == plan_assignment),
        reason             = reason,
        warning            = warning,
        overs_remaining    = overs_remaining_final,
        alternatives       = available,
        tier_note          = tier_note,
    )


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from engine.bowling_plan import generate_bowling_plan
    from utils.situation import LiveMatchState, WeatherImpact
    from datetime import datetime

    lahore_bowlers = [
        "Shaheen Shah Afridi", "Haris Rauf", "Zaman Khan",
        "Rashid Khan", "Liam Dawson",
    ]

    weather = WeatherImpact(
        spinner_penalty    = 0.60,
        swing_bonus        = 1.10,
        pace_bounce_bonus  = 1.05,
        yorker_reliability = 0.92,
        dl_planning_needed = False,
        dew_onset_over     = 13,
        warnings           = [],
    )

    plan = generate_bowling_plan(lahore_bowlers, weather)

    # Simulate over 14 — Rashid Khan (spinner) has been assigned, but dew is active
    state = LiveMatchState(
        batting_team     = "Karachi Kings",
        bowling_team     = "Lahore Qalandars",
        venue            = "Gaddafi Stadium, Lahore",
        innings          = 1,
        current_over     = 14,
        current_score    = 112,
        current_wickets  = 3,
        target           = 0,
        bowling_plan     = None,
        current_batter1  = "Babar Azam",
        current_batter2  = "Mohammad Rizwan",
        partnership_runs = 34,
        partnership_balls= 22,
        overs_bowled_by  = {
            "Shaheen Shah Afridi": 2,
            "Haris Rauf":          2,
            "Zaman Khan":          2,
            "Rashid Khan":         2,
            "Liam Dawson":         2,
        },
    )

    rec = recommend_bowler_this_over(plan, state, weather)
    print(f"\nOver {state.current_over} recommendation:")
    print(f"  Recommended : {rec.recommended_bowler}")
    print(f"  Plan said   : {rec.plan_bowler}")
    print(f"  On plan     : {rec.is_plan_bowler}")
    print(f"  Reason      : {rec.reason}")
    print(f"  Warning     : {rec.warning or 'None'}")
    print(f"  Overs left  : {rec.overs_remaining}")
    print(f"  Alternatives: {rec.alternatives}")
