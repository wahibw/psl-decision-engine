# engine/match_intelligence.py
# Generates the single most important situation read after each over update.
# Called by the dugout screen. Returns ONE sentence for the analyst to brief the coach.
#
# Public API:
#   generate_situation_read(state, bowling_plan, partnership, opposition_prediction, weather)
#   -> SituationRead

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from engine.bowling_plan import BowlingPlan
from engine.bowling_recommender import BowlerRecommendation
from engine.opposition_predictor import OppositionBattingPrediction, PredictedBatter
from engine.partnership_engine import PartnershipAssessment
from utils.situation import LiveMatchState, WeatherImpact

_PROJ_ROOT              = Path(__file__).resolve().parent.parent
_PLAYER_INDEX           = _PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv"
_PLAYER_INDEX_FALLBACK  = _PROJ_ROOT.parent / "player_index_2026_enriched.csv"
_RECENT_FORM_PATH       = _PROJ_ROOT / "data" / "processed" / "recent_form.parquet"

_CLASSIFIER_PATH = _PROJ_ROOT / "models" / "saved" / "situation_classifier"
_MIN_CONF        = 0.6   # minimum confidence to use ML vs rule-based

_classifier_pipeline = None   # module-level lazy cache


@lru_cache(maxsize=1)
def _load_recent_form_map(path: str) -> dict[str, dict]:
    """Load {player_name: {bat_form_score, bat_avg, bat_sr, bat_trend}} from recent_form.parquet."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import pandas as _pd
        df      = _pd.read_parquet(p)
        overall = df[df["venue"] == ""]
        result  = {}
        for _, row in overall.iterrows():
            name = row.get("player_name", "")
            if name:
                result[name] = {
                    "bat_form_score": float(row.get("bat_form_score", 50.0)),
                    "bat_avg":        float(row.get("bat_avg",        0.0)),
                    "bat_sr":         float(row.get("bat_sr",         0.0)),
                    "bat_trend":      str(row.get("bat_trend",        "stable")),
                    "bat_innings":    int(row.get("bat_innings",      0)),
                }
        return result
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_player_roles(path: str) -> dict[str, str]:
    """Load {player_name: primary_role} from player_index.csv."""
    roles: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return roles
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("player_name", "").strip()
            role = row.get("primary_role", "").strip()
            if name:
                roles[name] = role
    return roles


def _player_index_path() -> str:
    return str(_PLAYER_INDEX if _PLAYER_INDEX.exists() else _PLAYER_INDEX_FALLBACK)


# ---------------------------------------------------------------------------
# ML SITUATION CLASSIFIER (Upgrade 4 — DistilBERT)
# ---------------------------------------------------------------------------

def _load_classifier():
    """Lazy-load DistilBERT situation classifier from models/saved/situation_classifier/."""
    global _classifier_pipeline
    if _classifier_pipeline is not None:
        return _classifier_pipeline
    if not _CLASSIFIER_PATH.exists():
        return None
    try:
        from transformers import pipeline as hf_pipeline
        _classifier_pipeline = hf_pipeline(
            "text-classification",
            model=str(_CLASSIFIER_PATH),
            tokenizer=str(_CLASSIFIER_PATH),
            top_k=None,          # return scores for all 3 classes
        )
        return _classifier_pipeline
    except Exception:
        return None


def _classify_situation_ml(state) -> tuple | None:
    """
    Run DistilBERT on the current match state.
    Returns (label, confidence) or None if model not available or confidence below threshold.
    Labels: CRITICAL | WARNING | INFO
    """
    clf = _load_classifier()
    if clf is None:
        return None

    over    = state.current_over
    score   = state.current_score
    wickets = state.current_wickets
    phase   = getattr(state, "phase", "middle")
    dew     = "yes" if getattr(state, "dew_active", False) else "no"

    # Bowler type proxy from phase (matches training logic)
    if phase == "powerplay":
        btype = "pace"
    elif phase == "death":
        btype = "pace/mixed"
    else:
        btype = "mixed"

    p_runs  = getattr(state, "partnership_runs",  0)
    p_balls = min(getattr(state, "partnership_balls", 0), 24)

    text = (
        f"Over {over}. "
        f"Score {score}/{wickets}. "
        f"Partnership {p_runs} off {p_balls}. "
        f"Phase {phase}. "
        f"Dew active: {dew}. "
        f"Bowler: {btype}."
    )

    try:
        results = clf(text)[0]   # list of {label: ..., score: ...}
        best = max(results, key=lambda x: x["score"])
        if best["score"] >= _MIN_CONF:
            return best["label"], best["score"]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# DATA CLASS
# ---------------------------------------------------------------------------

@dataclass
class SituationRead:
    priority:       str     # "CRITICAL" | "WARNING" | "INFO"
    message:        str     # ONE sentence. Plain English. For the coach.
    action_needed:  bool    # True = brief coach this over
    detail:         str     # 2-3 sentences for analyst context only


# ---------------------------------------------------------------------------
# PRIORITY HIERARCHY
# ---------------------------------------------------------------------------

# Each check returns (priority_rank, message, action_needed, detail) or None.
# Lower rank number = higher priority. Rank 0 = most critical.

def _check_bowling_plan_critical(
    state:        LiveMatchState,
    bowling_plan: BowlingPlan,
) -> tuple | None:
    """
    Bowler has ACTUALLY bowled 4 overs but the pre-match template still has
    them assigned to upcoming overs.

    This can legitimately happen mid-match when:
      - A bowler was brought on earlier than planned due to a wicket/partnership break
      - A bowling change was forced by injury or a captain's decision
    The template is a starting plan, not a live tracker. When this fires, the
    analyst must immediately identify a backup bowler from the contingencies list.
    """
    for bowler, allocated_overs in bowling_plan.bowler_summary.items():
        bowled = state.overs_bowled_by.get(bowler, 0)
        if bowled >= 4:
            remaining_plan = [o for o in allocated_overs if o > state.current_over]
            if remaining_plan:
                return (
                    0,
                    f"PLAN DEVIATION: {bowler.split()[-1]} has bowled 4 overs "
                    f"but template has {len(remaining_plan)} more assigned — reassign NOW.",
                    True,
                    f"{bowler} was allocated overs {allocated_overs} in the template. "
                    f"They've now hit the PSL 4-over cap (actual overs bowled: {bowled}). "
                    f"Overs {remaining_plan} need a new bowler. "
                    f"Check backup column in bowling plan and contingencies list.",
                )
    return None


def _check_partnership_critical(
    partnership: PartnershipAssessment,
) -> tuple | None:
    if partnership.danger_level == "Critical":
        return (
            1,
            partnership.alert_message,
            True,
            f"Partnership: {partnership.batter1} / {partnership.batter2} — "
            f"{partnership.current_runs} runs off {partnership.current_balls} balls "
            f"(SR {partnership.current_sr:.0f}). Historical avg for this pair: "
            f"{partnership.historical_avg_runs:.0f} runs / {partnership.historical_avg_balls:.0f} balls. "
            f"Recommendation: {partnership.recommended_action}",
        )
    return None


def _check_dew_onset(
    state:   LiveMatchState,
    weather: WeatherImpact,
) -> tuple | None:
    """Flag the over when dew first becomes active."""
    # current_over and dew_onset_over are both 1-indexed — compare directly
    onset = weather.dew_onset_over
    if onset is not None and onset > 0 and state.current_over == onset:
        return (
            2,
            f"DEW NOW ACTIVE (over {state.current_over}) — "
            f"move spinner out of overs {onset + 1}-20 immediately.",
            True,
            f"Dew onset reached at over {onset}. Spinner effectiveness reduced significantly. "
            f"Bowler swap needed if spinner was planned for overs {onset+1}-20. "
            f"Check bowling plan backup column.",
        )
    return None


def _check_death_chase_compound(
    state:      LiveMatchState,
    opposition: OppositionBattingPrediction,
) -> tuple | None:
    """
    Compound alert: second innings, required rate > 15, <=3 wickets remaining,
    AND both batters at the crease are specialist bowlers (primary_role == "Bowler").
    This is the most actionable situation read in a PSL chase — match effectively over
    through conventional batting.

    Fail-safe: only fires when both batters are confirmed as "Bowler" in player_index.
    Players not found in the index are NOT flagged.
    """
    # Condition 1: second innings
    if state.innings != 2:
        return None

    # Condition 2: required rate > 15
    balls_left   = max(1, (20 - state.current_over) * 6)
    runs_needed  = max(0, state.target - state.current_score)
    rr           = (runs_needed / balls_left) * 6
    if rr <= 15:
        return None

    # Condition 3: 3 or fewer wickets remaining
    wkts_remaining = 10 - state.current_wickets
    if wkts_remaining > 3:
        return None

    # Condition 4: both batters are confirmed specialist Bowlers
    roles = _load_player_roles(_player_index_path())
    role1 = roles.get(state.current_batter1)
    role2 = roles.get(state.current_batter2)
    # Only flag if both are found AND both are "Bowler" — fail safe for unknowns
    if role1 != "Bowler" or role2 != "Bowler":
        return None

    overs_left = max(1, 20 - state.current_over)
    return (
        2,
        f"CHASE EFFECTIVELY OVER: RR {rr:.1f} needed, {wkts_remaining} wickets left, "
        f"both batters are tail — captain must decide: attack or protect NRR.",
        True,
        f"Required {runs_needed} off {balls_left} balls at RR {rr:.1f}. "
        f"{state.current_batter1} and {state.current_batter2} are both specialist bowlers. "
        f"No realistic path to victory through conventional batting. "
        f"Options: (1) aggressive swing to improve NRR, (2) rotate tail "
        f"to preserve remaining wicket for a batting cameo.",
    )


def _check_dangerous_batter_incoming(
    state:      LiveMatchState,
    opposition: OppositionBattingPrediction,
) -> tuple | None:
    """Most dangerous opposition batter about to come in (within 1 wicket)."""
    wickets = state.current_wickets
    # Batting positions are 1-indexed. Two batters are always at the crease.
    # wickets=0 → positions 1 & 2 in, position 3 is next
    # wickets=1 → positions 1 (or 2) & 2 in, position 3 is next
    # wickets=2 → position 4 is next
    # General: next = wickets + 3 when wickets==0, else wickets + 2
    next_position = wickets + 3 if wickets == 0 else wickets + 2
    for pb in opposition.predicted_order:
        if pb.position == next_position and pb.danger_rating == "High":
            return (
                3,
                f"DANGER: {pb.player_name.split()[-1]} ({pb.phase_strength}) "
                f"incoming next wicket — set field NOW. {pb.key_note}",
                True,
                f"{pb.player_name} at position {pb.position}: {pb.key_note} "
                f"Career SR {pb.career_sr:.0f}, death SR {pb.death_sr:.0f}. "
                f"Recommended action: {next(iter([b for b in opposition.bowling_implications[:1]]), 'See bowling implications.')}",
            )
    return None


def _check_rain_spike(
    weather: WeatherImpact,
) -> tuple | None:
    if weather.dl_planning_needed:
        return (
            4,
            "Rain risk — D/L planning needed. Front-load batting, early runs are worth more.",
            True,
            "Rain probability above 30%. D/L method rewards teams that score heavily in overs 1-10. "
            "If batting, prioritise early boundaries. If bowling, front-load strike bowlers.",
        )
    return None


def _check_partnership_dangerous(
    partnership: PartnershipAssessment,
) -> tuple | None:
    if partnership.danger_level == "Dangerous":
        return (
            5,
            partnership.alert_message,
            True,
            f"Partnership at {partnership.current_balls} balls. "
            f"Historical: pair averages {partnership.historical_avg_runs:.0f} runs together. "
            f"{partnership.recommended_action}",
        )
    return None


def _check_dew_active_ongoing(
    state:   LiveMatchState,
    weather: WeatherImpact,
) -> tuple | None:
    """Dew already active but spinner still in plan for coming overs."""
    # current_over and dew_onset_over are both 1-indexed — compare directly
    dew_active_now = weather.dew_active_at and weather.dew_onset_over <= state.current_over
    if not dew_active_now:
        return None
    # Check if next planned over has a spinner
    if state.bowling_plan is None:
        return None
    for oa in state.bowling_plan.overs:
        if oa.over == state.current_over + 1 and oa.weather_note:
            return (
                6,
                f"Dew active — {oa.primary_bowler.split()[-1]} flagged for over {oa.over}: {oa.weather_note}",
                True,
                f"Spinner scheduled for next over with dew active. Check backup bowler in plan.",
            )
    return None


def _check_death_bowler_shortage(
    state:        LiveMatchState,
    bowling_plan: BowlingPlan,
) -> tuple | None:
    """Death specialist running out of overs with death overs remaining."""
    current = state.current_over
    if current < 14:
        return None

    death_overs_remaining = max(0, 20 - current)
    if death_overs_remaining < 3:
        return None

    # Find death bowlers in plan
    death_bowlers = [
        b for b, overs in bowling_plan.bowler_summary.items()
        if any(o >= 16 for o in overs)
    ]
    for b in death_bowlers:
        remaining = state.overs_remaining_for(b)
        if remaining == 1 and death_overs_remaining >= 3:
            return (
                7,
                f"Warning: {b.split()[-1]} has only 1 over left but {death_overs_remaining} death overs to go — plan backup.",
                True,
                f"Death specialist {b} nearly exhausted. Identify which bowler covers the gap. "
                f"Check contingencies in bowling plan.",
            )
    return None


def _check_partnership_growing(
    partnership: PartnershipAssessment,
) -> tuple | None:
    if partnership.danger_level == "Growing":
        return (
            8,
            partnership.alert_message,
            False,
            f"Partnership at {partnership.current_balls} balls — approaching danger territory. "
            f"Historical avg: {partnership.historical_avg_runs:.0f} runs together. "
            f"Pre-empt: {partnership.recommended_action}",
        )
    return None


def _check_dl_par(
    state:   LiveMatchState,
) -> tuple | None:
    """
    Second innings: current run rate has fallen significantly behind required rate.

    NOTE: This is a run-rate approximation ONLY. Proper D/L par (Duckworth-Lewis-Stern)
    requires the official DLS resource table and cannot be computed from RRR/CRR alone.
    Use the official DLS calculator for any rain-interrupted scenario. This check
    is intended solely for tracking normal run-rate health in uninterrupted chases.
    """
    if state.innings != 2:
        return None
    state.compute_derived()
    if state.crr <= 0:
        return None

    # Threshold tightens in death overs — a 15% RR gap at over 16 is a crisis,
    # the same gap at over 5 is normal T20 variance.
    over = state.current_over
    if over <= 10:
        # Powerplay + early middle: only flag genuinely severe early pressure
        threshold = 1.5
        phase = "powerplay/early-middle"
    elif over <= 15:
        # Mid-overs: moderate tightening
        threshold = 1.35
        phase = "mid-overs"
    else:
        # Death: flag even moderate gaps — every over matters
        threshold = 1.15
        phase = "death overs"

    # Still require a meaningful absolute RRR to avoid noise (e.g. 3 vs 2 ratio early)
    if state.rrr >= 8 and state.rrr > state.crr * threshold:
        deficit = state.rrr - state.crr
        return (
            9,
            f"Run rate warning ({phase}): need {state.rrr:.1f}, current {state.crr:.1f} "
            f"(+{deficit:.1f} behind). Batting must accelerate. "
            f"[Use official DLS calculator if rain threatens.]",
            True,
            f"Required rate {state.rrr:.1f}, current {state.crr:.1f} — gap {deficit:.1f}/over. "
            f"Phase: {phase} (threshold {threshold:.2f}x). "
            f"IMPORTANT: This is not an official D/L par figure. "
            f"For rain scenarios, use the official DLS resource table. "
            f"Batting side needs to accelerate under normal conditions.",
        )
    return None


def _check_batting_collapse(
    state: LiveMatchState,
) -> tuple | None:
    """
    Detect a batting collapse: 2+ wickets in the current over (or last over).
    Multiple wickets in quick succession shifts the entire match dynamic —
    the batting order prediction becomes unreliable and bowling must adjust.
    Only fires in the first innings (bowling side). Second innings collapses
    are handled by the D/L and run-rate checks.
    """
    if state.innings != 1:
        return None
    # state.wickets_this_over is set externally; fall back to 0 if not present
    wickets_this_over = getattr(state, "wickets_this_over", 0)
    if wickets_this_over >= 2:
        return (
            1.5,   # between partnership critical (1) and dew onset (2)
            f"COLLAPSE: {wickets_this_over} wickets in over {state.current_over} — "
            f"batting order disrupted, lower-order threat imminent.",
            True,
            f"Multiple wickets this over means opposition batting order is now unreliable. "
            f"Tail batters incoming are unpredictable (promote pinch-hitters is common). "
            f"Keep bowling pressure high — don't loosen the field.",
        )
    return None


def _check_exceptional_form_batter(
    state:      LiveMatchState,
    opposition: OppositionBattingPrediction,
) -> tuple | None:
    """
    Warn when an opposition batter currently at the crease (or incoming next wicket)
    is in exceptional recent form (bat_form_score ≥ 70).
    Only fires once per match to avoid noise — ranks between dangerous-batter-incoming
    (3) and rain spike (4) at rank 3.5.
    """
    rf = _load_recent_form_map(str(_RECENT_FORM_PATH))
    if not rf:
        return None

    # Check batters currently at the crease
    for batter in (state.current_batter1, state.current_batter2):
        if not batter:
            continue
        form = rf.get(batter)
        if form and form["bat_form_score"] >= 70 and form["bat_innings"] >= 5:
            score = form["bat_form_score"]
            sr    = form["bat_sr"]
            trend = form["bat_trend"]
            trend_note = " (rising)" if trend == "rising" else ""
            return (
                3.5,
                f"FORM ALERT: {batter.split()[-1]} is in exceptional form "
                f"({score:.0f}/100{trend_note}, {sr:.0f} SR last 10) - "
                f"assign your best matchup bowler immediately.",
                True,
                f"{batter} has a bat_form_score of {score:.0f}/100 "
                f"({form['bat_avg']:.0f} avg, {sr:.0f} SR across {form['bat_innings']} innings). "
                f"Trend: {trend}. This is a high-priority in-match threat. "
                f"Check matchup matrix for their bowling style weakness.",
            )
    return None


def _check_plan_on_track(
    state:        LiveMatchState,
    bowling_plan: BowlingPlan,
    partnership:  PartnershipAssessment,
) -> tuple:
    """Default INFO message — plan on track, partnership not a threat."""
    current = state.current_over
    # Find what the plan says for the next over
    next_over_assignment = None
    for oa in bowling_plan.overs:
        if oa.over == current + 1:
            next_over_assignment = oa
            break

    if next_over_assignment:
        msg = (
            f"Plan on track — {next_over_assignment.primary_bowler.split()[-1]} "
            f"for over {next_over_assignment.over} ({next_over_assignment.phase}) as planned."
        )
        detail = (
            f"Bowling plan is being followed. "
            f"Current partnership: {partnership.current_runs} runs / {partnership.current_balls} balls "
            f"({partnership.danger_level}). No immediate action needed."
        )
    else:
        msg    = f"Plan on track — over {current} complete. No immediate action needed."
        detail = "All within expected parameters. Continue per plan."

    return (10, msg, False, detail)


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def generate_situation_read(
    state:        LiveMatchState,
    bowling_plan: BowlingPlan,
    partnership:  PartnershipAssessment,
    opposition:   OppositionBattingPrediction,
    weather:      WeatherImpact,
) -> SituationRead:
    """
    Determine the single most important situation happening right now.

    Priority (highest first):
      CRITICAL: bowling plan broken (cap exceeded), critical partnership,
                dew just activated, dangerous batter incoming, rain risk
      WARNING:  dangerous partnership, dew active next over, death bowler
                shortage, growing partnership, D/L par deficit
      INFO:     plan on track

    Returns ONE SituationRead — the highest-priority alert only.
    """
    checks = [
        _check_bowling_plan_critical(state, bowling_plan),
        _check_partnership_critical(partnership),
        _check_death_chase_compound(state, opposition),
        _check_batting_collapse(state),
        _check_dew_onset(state, weather),
        _check_dangerous_batter_incoming(state, opposition),
        _check_exceptional_form_batter(state, opposition),
        _check_rain_spike(weather),
        _check_partnership_dangerous(partnership),
        _check_dew_active_ongoing(state, weather),
        _check_death_bowler_shortage(state, bowling_plan),
        _check_partnership_growing(partnership),
        _check_dl_par(state),
        _check_plan_on_track(state, bowling_plan, partnership),
    ]

    # Filter None, sort by rank
    valid = sorted([c for c in checks if c is not None], key=lambda x: x[0])
    rank, message, action_needed, detail = valid[0]

    # Map rank to priority label
    if rank <= 4:
        priority = "CRITICAL"
    elif rank <= 7:
        priority = "WARNING"
    else:
        priority = "INFO"

    # --- Upgrade 4: ML situation classifier augmentation ---
    # Only let ML upgrade an INFO read; rule-based CRITICAL/WARNING takes precedence.
    if priority == "INFO":
        ml = _classify_situation_ml(state)
        if ml is not None:
            ml_label, ml_conf = ml
            if ml_label in ("CRITICAL", "WARNING"):
                priority      = ml_label
                action_needed = ml_label == "CRITICAL"
                message       = f"[ML {ml_conf:.0%}] " + message
                detail        = detail + f" | ML situation classifier: {ml_label} ({ml_conf:.0%} confidence)."

    return SituationRead(
        priority      = priority,
        message       = message,
        action_needed = action_needed,
        detail        = detail,
    )


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from datetime import datetime
    from utils.situation import LiveMatchState, WeatherImpact
    from engine.bowling_plan import generate_bowling_plan, BowlingPlan
    from engine.partnership_engine import assess_partnership
    from engine.opposition_predictor import predict_batting_order

    bowlers = ["Shaheen Shah Afridi", "Haris Rauf", "Zaman Khan", "Rashid Khan", "Liam Dawson"]

    weather_dew = WeatherImpact(
        spinner_penalty    = 0.60,
        swing_bonus        = 1.10,
        pace_bounce_bonus  = 1.05,
        yorker_reliability = 0.92,
        dl_planning_needed = False,
        dew_onset_over     = 13,
        warnings           = [],
    )

    plan = generate_bowling_plan(bowlers, weather_dew)

    opposition = predict_batting_order(
        "Karachi Kings", "Gaddafi Stadium, Lahore", bowlers
    )

    test_cases = [
        # (over, score, wickets, p_runs, p_balls, description)
        (14, 110, 2, 48, 40, "Dew just activated (over 14), dangerous partnership"),
        (16, 130, 3, 18, 14, "Normal — plan on track"),
        (17, 145, 3, 62, 45, "Critical partnership"),
    ]

    clf_status = "loaded" if _load_classifier() is not None else "not found (rule-based fallback active)"
    print(f"\n{'='*65}")
    print(f"  match_intelligence.py -- situation read test")
    print(f"  ML classifier: {clf_status}")
    print(f"{'='*65}")

    for over, score, wickets, p_runs, p_balls, desc in test_cases:
        state = LiveMatchState(
            batting_team     = "Karachi Kings",
            bowling_team     = "Lahore Qalandars",
            venue            = "Gaddafi Stadium, Lahore",
            innings          = 1,
            current_over     = over,
            current_score    = score,
            current_wickets  = wickets,
            target           = 0,
            bowling_plan     = plan,
            current_batter1  = "Babar Azam",
            current_batter2  = "Mohammad Rizwan",
            partnership_runs = p_runs,
            partnership_balls= p_balls,
            overs_bowled_by  = {b: (over // len(bowlers)) for b in bowlers},
        )

        pa = assess_partnership("Babar Azam", "Mohammad Rizwan", p_runs, p_balls)
        sr = generate_situation_read(state, plan, pa, opposition, weather_dew)

        print(f"\n  Scenario: {desc}")
        print(f"  [{sr.priority}] {sr.message}")
        print(f"  Action needed: {sr.action_needed}")
        print(f"  Detail: {sr.detail[:90]}...")

    print(f"\n{'='*65}\n")
