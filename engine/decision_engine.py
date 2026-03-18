# engine/decision_engine.py
# Central interface — one call generates the complete pre-match brief
# or live dugout intelligence update.
#
# Public API:
#   generate_prematch_brief(context) -> PreMatchBrief
#   update_live_intelligence(state, brief) -> LiveUpdate

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.situation import MatchContext, LiveMatchState, WeatherImpact

# Engine imports
from engine.matchup_engine import get_key_matchups_for_brief, MatchupNote
from engine.opposition_predictor import predict_batting_order, OppositionBattingPrediction
from engine.partnership_engine import assess_partnership, PartnershipAssessment
from engine.bowling_plan import generate_bowling_plan, BowlingPlan
from engine.bowling_recommender import recommend_bowler_this_over, BowlerRecommendation
from engine.batting_scenarios import generate_batting_scenarios, BattingScenario
from engine.match_intelligence import generate_situation_read, SituationRead
from engine.xi_selector import select_xi, XiOption

# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class TossRecommendation:
    recommendation: str     # "BAT FIRST" | "BOWL FIRST"
    reasoning:      list[str]
    dl_note:        str     # "" or D/L warning if rain risk


@dataclass
class PreMatchBrief:
    # Input context
    our_team:           str
    opposition:         str
    venue:              str
    match_datetime:     datetime

    # Outputs
    xi_options:         list[XiOption]          # 3 alternatives
    toss:               TossRecommendation
    opposition_order:   OppositionBattingPrediction
    bowling_plan:       BowlingPlan
    batting_scenarios:  list[BattingScenario]   # 4 cards
    matchup_notes:      list[MatchupNote]        # 3-4 H2H notes
    weather_impact:     WeatherImpact
    weather_warnings:   list[str]
    data_tier_notes:    list[str]           # warnings for Tier 2/3 squad players

    # Meta
    generated_at:       datetime = field(default_factory=datetime.now)
    captain:            Optional[str] = None      # locked captain name (displayed with "C" badge)


@dataclass
class LiveUpdate:
    current_over:       int
    situation_read:     SituationRead
    bowler_recommendation: BowlerRecommendation
    partnership:        PartnershipAssessment
    weather_warnings:   list[str]


# ---------------------------------------------------------------------------
# TOSS RECOMMENDATION
# ---------------------------------------------------------------------------

def _load_venue_toss_stats(venue: str) -> dict:
    """Load chase_win_pct and avg_first_score from venue_stats.csv for this venue."""
    import csv as _csv
    v_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "venue_stats.csv"
    try:
        with open(v_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if row.get("venue", "").strip().lower() == venue.strip().lower():
                    def _f(k, d=50.0):
                        try: return float(row.get(k, d) or d)
                        except: return d
                    return {
                        "chase_win_pct":   _f("chase_win_pct",   50.0),
                        "defend_win_pct":  _f("defend_win_pct",  50.0),
                        "avg_first_score": _f("avg_first_score", 165.0),
                        "matches":         int(_f("matches", 10)),
                    }
    except Exception:
        pass
    return {"chase_win_pct": 50.0, "defend_win_pct": 50.0, "avg_first_score": 165.0, "matches": 0}


def _recommend_toss(
    venue:           str,
    weather:         WeatherImpact,
    opposition_team: str,
    match_datetime:  Optional[datetime] = None,
) -> TossRecommendation:
    """
    Data-driven toss recommendation using venue_stats.csv + weather context.

    Priority order:
      1. Dew / rain — overrides all venue stats (highest impact)
      2. Venue chase/defend win rates from actual PSL historical data
      3. Month awareness — March/April = spring dew risk slightly higher in Pakistan

    Thresholds for venue stats:
      chase_win_pct > 56%  → BOWL FIRST (opposition chases well here — so do we)
      defend_win_pct > 56% → BAT FIRST  (setting totals works here)
      44-56%               → NEUTRAL / weather breaks tie
    """
    reasons: list[str] = []
    rec = "BAT FIRST"   # default

    vstats = _load_venue_toss_stats(venue)
    chase_pct  = vstats["chase_win_pct"]
    defend_pct = vstats["defend_win_pct"]
    avg_score  = vstats["avg_first_score"]
    n_matches  = vstats["matches"]

    # Month context (spring = dew risk elevated in Pakistan)
    month_dew_risk = False
    if match_datetime is not None:
        month = match_datetime.month if hasattr(match_datetime, "month") else 0
        month_dew_risk = month in (2, 3, 4)  # Feb-Apr: pre-monsoon dew season

    # --- Priority 1: Rain / D-L (checked first — sets a total early regardless of dew) ---
    if weather.dl_planning_needed:
        rec = "BAT FIRST"
        reasons.append(
            f"Rain risk — D/L favours the team that scores early. BAT FIRST to control D/L par."
        )

    # --- Priority 2: Dew (only when no rain risk) ---
    elif weather.severe_dew:
        rec = "BOWL FIRST"
        reasons.append(
            f"Severe dew from over {weather.dew_onset_over} — batting second is significantly "
            f"easier on a wet ball. Bowl first, exploit dew conditions when chasing."
        )
    elif weather.spinner_penalty <= 0.75:
        rec = "BOWL FIRST"
        reasons.append(
            f"Heavy dew likely (spinner penalty {weather.spinner_penalty:.2f}) — "
            f"batting second benefits from wet ball. Bowl first."
        )

    # --- Priority 3: Venue historical data ---
    else:
        sample_note = f" ({n_matches} PSL matches)" if n_matches >= 10 else " (limited data)"
        if chase_pct >= 57:
            rec = "BOWL FIRST"
            reasons.append(
                f"Chase-friendly venue: {chase_pct:.0f}% of teams chasing have won here{sample_note}. "
                f"Bowl first and let your batting attack the second-innings score."
            )
            if avg_score < 160:
                reasons.append(
                    f"Average first-innings score only {avg_score:.0f} — low first-innings totals "
                    f"are routinely chased at this ground."
                )
        elif defend_pct >= 57:
            rec = "BAT FIRST"
            reasons.append(
                f"Defend-friendly venue: {defend_pct:.0f}% of first-innings teams have won here{sample_note}. "
                f"Post a total and trust your bowling attack."
            )
            if avg_score >= 170:
                reasons.append(
                    f"Average first-innings score {avg_score:.0f} — big totals put pressure on the chase."
                )
        else:
            # Near 50-50 — weather and form decide
            rec = "BAT FIRST"
            reasons.append(
                f"Venue near-even split (chase {chase_pct:.0f}% / defend {defend_pct:.0f}%){sample_note}. "
                f"Bat first to control the tempo."
            )

    # --- Priority 3: Month context ---
    if month_dew_risk and rec == "BOWL FIRST":
        reasons.append(
            "March/April dew season in Pakistan — dew typically arrives mid-innings and advantages "
            "the team batting second. Bowl first recommendation is reinforced."
        )
    elif month_dew_risk and rec == "BAT FIRST":
        reasons.append(
            "March/April dew season in Pakistan — monitor pre-match conditions; "
            "if significant dew is expected, consider bowling first instead."
        )

    # Swing conditions note
    if weather.swing_bonus >= 1.20 and rec == "BOWL FIRST":
        reasons.append(
            f"Swing conditions ({weather.swing_bonus:.2f}x) also favour your pacers with the new ball "
            f"— good time to bowl first."
        )
    elif weather.swing_bonus >= 1.20 and rec == "BAT FIRST":
        reasons.append(
            f"Note: swing conditions ({weather.swing_bonus:.2f}x) might tempt bowling first, "
            f"but dew/venue data overrides that."
        )

    # D/L note
    dl_note = ""
    if weather.dl_planning_needed:
        dl_note = (
            "Rain probability above 30% — D/L planning essential. "
            "Front-load batting: D/L rewards runs scored early."
        )

    return TossRecommendation(
        recommendation = rec,
        reasoning      = reasons,
        dl_note        = dl_note,
    )


# ---------------------------------------------------------------------------
# DATA TIER HELPERS
# ---------------------------------------------------------------------------

def _load_player_tiers(squad: list[str]) -> dict[str, int]:
    """Load data_tier for each squad player from player_index.csv."""
    pi_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_index_2026_enriched.csv"
    tiers: dict[str, int] = {}
    try:
        with open(pi_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("player_name", "").strip()
                if name in squad:
                    try:
                        tiers[name] = int(row.get("data_tier", 1) or 1)
                    except (ValueError, TypeError):
                        tiers[name] = 1
    except Exception:
        pass
    return tiers


def _load_t20_proxies(squad: list[str]) -> dict[str, dict]:
    """Load T20 career proxy stats for Tier 3 players from player_index.csv."""
    pi_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_index_2026_enriched.csv"
    proxies: dict[str, dict] = {}
    try:
        with open(pi_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("player_name", "").strip()
                if name in squad:
                    tier = int(row.get("data_tier", 1) or 1)
                    if tier == 3:
                        def _f(v):
                            try: return float(v) if v and v.lower() not in ("nan","") else None
                            except: return None
                        proxies[name] = {
                            "t20_career_sr":      _f(row.get("t20_career_sr")),
                            "t20_career_avg":     _f(row.get("t20_career_avg")),
                            "t20_career_economy": _f(row.get("t20_career_economy")),
                        }
    except Exception:
        pass
    return proxies


# ---------------------------------------------------------------------------
# PRE-MATCH BRIEF GENERATOR
# ---------------------------------------------------------------------------

def generate_prematch_brief(
    our_team:          str,
    opposition:        str,
    venue:             str,
    match_datetime:    datetime,
    our_squad:         list[str],
    weather_impact:    Optional[WeatherImpact] = None,
    season:            int = 0,
    innings:           int = 1,
    forced_players:    Optional[list[str]] = None,
    captain:           Optional[str] = None,
) -> PreMatchBrief:
    """
    Generate a complete pre-match brief.

    Args:
        our_team:       Our team name
        opposition:     Opposition team name
        venue:          Match venue
        match_datetime: Match date and time
        our_squad:      List of 16-18 squad players
        weather_impact: Pre-computed WeatherImpact (or None for neutral)
        season:         PSL season for opposition profile lookup
        innings:        Expected innings (1 = bat first, 2 = chase)

    Returns:
        PreMatchBrief with all sections populated.
    """
    if len(our_squad) < 11:
        raise ValueError(
            f"generate_prematch_brief() requires at least 11 players in the squad, "
            f"got {len(our_squad)}."
        )

    if weather_impact is None:
        weather_impact = WeatherImpact.neutral()

    # Load data tiers + T20 proxies for squad players
    player_tiers   = _load_player_tiers(our_squad)
    t20_proxies    = _load_t20_proxies(our_squad)

    # Build data tier warning notes
    data_tier_notes: list[str] = []
    tier2 = [p for p in our_squad if player_tiers.get(p, 1) == 2]
    tier3 = [p for p in our_squad if player_tiers.get(p, 1) == 3]
    if tier2:
        data_tier_notes.append(
            f"Limited PSL data (Medium confidence): {', '.join(tier2)}."
        )
    for p in tier3:
        proxy = t20_proxies.get(p, {})
        sr  = proxy.get("t20_career_sr")
        avg = proxy.get("t20_career_avg")
        eco = proxy.get("t20_career_economy")
        parts = []
        if sr:  parts.append(f"T20 SR {sr:.0f}")
        if avg: parts.append(f"avg {avg:.1f}")
        if eco: parts.append(f"economy {eco:.2f}")
        stat_str = f" ({', '.join(parts)})" if parts else ""
        data_tier_notes.append(
            f"No PSL history — T20 career estimate: {p}{stat_str}."
        )

    # 1a. Opposition profile (needed before XI selection for matchup bonuses)
    from engine.opposition_predictor import predict_batting_order as _pred_opp
    from engine.bowling_plan import _load_opposition_profile as _load_opp_profile
    _opp_profile_check = _load_opp_profile(opposition)
    if _opp_profile_check.get("is_estimated", False):
        data_tier_notes.append(
            f"Opposition profile for {opposition} is estimated (no PSL history). "
            f"XI selection matchup bonuses and bowling plan opposition intelligence "
            f"have no empirical basis."
        )
    _opp_preview = _pred_opp(
        team    = opposition,
        venue   = venue,
        our_bowlers = [],   # no bowlers known yet — used only for LH% here
        season  = season,
    )
    opp_lh_pct    = _opp_preview.left_hand_count / max(1, len(_opp_preview.predicted_order)) * 100
    opp_batter_names = [b.player_name for b in _opp_preview.predicted_order[:6]]

    # Load spin economies by subtype for per-style matchup bonuses in XI selection
    _opp_profile = _load_opp_profile(opposition)
    opp_spin_economies = {
        "spin":        _opp_profile.get("vs_spin_economy",          8.0),
        "legspin":     _opp_profile.get("vs_legspin_economy",       _opp_profile.get("vs_spin_economy", 8.0)),
        "offspin":     _opp_profile.get("vs_offspin_economy",       _opp_profile.get("vs_spin_economy", 8.0)),
        "leftarm_spin":_opp_profile.get("vs_leftarm_spin_economy",  _opp_profile.get("vs_spin_economy", 8.0)),
    }

    # 1. Playing XI options (with opposition-aware matchup bonuses including spin subtypes)
    xi_options = select_xi(
        squad                     = our_squad,
        venue                     = venue,
        weather                   = weather_impact,
        innings                   = innings,
        forced_players            = forced_players,
        opposition_lh_pct         = opp_lh_pct,
        opposition_batters        = opp_batter_names,
        opposition_spin_economies = opp_spin_economies,
    )

    # 2. Toss recommendation
    toss = _recommend_toss(venue, weather_impact, opposition, match_datetime)

    # 3. Opposition batting order prediction
    # Identify our bowlers from primary XI
    primary_xi = xi_options[0] if xi_options else None
    our_bowlers = [
        p.player_name for p in (primary_xi.players if primary_xi else [])
        if "Bowler" in p.role or "All-rounder" in p.role
    ]

    opposition_order = predict_batting_order(
        team        = opposition,
        venue       = venue,
        our_bowlers = our_bowlers,
        season      = season,
    )

    # 4. Bowling plan
    bowling_plan = generate_bowling_plan(
        our_bowlers     = our_bowlers,
        weather         = weather_impact,
        venue           = venue,
        opposition_team = opposition,
    )

    # 5. Batting scenarios
    opp_profiles_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "opposition_profiles.csv"
    opp_bowling: dict = {}
    try:
        import pandas as pd
        op = pd.read_csv(opp_profiles_path)
        row = op[(op["team"] == opposition) & (op["season"] == 0)]
        if not row.empty:
            r = row.iloc[0]
            _ZERO_NULL_KEYS = ("economy", "sr", "pct", "score", "rate")
            def _zf(col: str, default: float) -> float:
                # 0.0 in economy/SR/pct/score/rate columns = missing data, not a genuine zero
                try:
                    v = r.get(col)
                    result = float(v) if pd.notna(v) else default
                    if result == 0.0 and any(k in col for k in _ZERO_NULL_KEYS):
                        return default
                    return result
                except (ValueError, TypeError):
                    return default
            opp_bowling = {
                "vs_spin_economy": _zf("vs_spin_economy", 7.5),
                "vs_pace_economy": _zf("vs_pace_economy", 8.0),
                "pace_overs_pct":  _zf("pace_overs_pct",  50.0),
                "spin_overs_pct":  _zf("spin_overs_pct",  20.0),
            }
    except Exception:
        pass

    our_batters = [p.player_name for p in (primary_xi.players if primary_xi else [])]
    batting_scenarios = generate_batting_scenarios(
        our_batters        = our_batters,
        opposition_bowling = opp_bowling,
        venue              = venue,
        weather            = weather_impact,
    )

    # 6. Matchup notes
    opposition_batters = [
        pb.player_name for pb in opposition_order.predicted_order[:6]
    ]
    matchup_notes = get_key_matchups_for_brief(
        our_bowlers        = our_bowlers,
        opposition_batters = opposition_batters,
        max_notes          = 4,
    )

    return PreMatchBrief(
        our_team          = our_team,
        opposition        = opposition,
        venue             = venue,
        match_datetime    = match_datetime,
        xi_options        = xi_options,
        toss              = toss,
        opposition_order  = opposition_order,
        bowling_plan      = bowling_plan,
        batting_scenarios = batting_scenarios,
        matchup_notes     = matchup_notes,
        weather_impact    = weather_impact,
        weather_warnings  = weather_impact.warnings,
        data_tier_notes   = data_tier_notes,
        captain           = captain,
    )


# ---------------------------------------------------------------------------
# LIVE DUGOUT UPDATE
# ---------------------------------------------------------------------------

def update_live_intelligence(
    state:         LiveMatchState,
    brief:         PreMatchBrief,
) -> LiveUpdate:
    """
    Generate a live intelligence update for the current over.
    Called by the dugout screen after each over tap.
    """
    weather = brief.weather_impact

    # Partnership assessment
    partnership = assess_partnership(
        batter1       = state.current_batter1,
        batter2       = state.current_batter2,
        current_runs  = state.partnership_runs,
        current_balls = state.partnership_balls,
    )

    # Bowler recommendation
    bowler_rec = recommend_bowler_this_over(
        bowling_plan = brief.bowling_plan,
        state        = state,
        weather      = weather,
    )

    # Situation read
    sit_read = generate_situation_read(
        state       = state,
        bowling_plan= brief.bowling_plan,
        partnership = partnership,
        opposition  = brief.opposition_order,
        weather     = weather,
    )

    return LiveUpdate(
        current_over          = state.current_over,
        situation_read        = sit_read,
        bowler_recommendation = bowler_rec,
        partnership           = partnership,
        weather_warnings      = weather.warnings,
    )


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.situation import WeatherImpact

    squad = [
        "Fakhar Zaman", "Abdullah Shafique", "Sikandar Raza",
        "Shaheen Shah Afridi", "Liam Dawson", "Mohammad Hafeez",
        "Rashid Khan", "Haris Rauf", "Zaman Khan",
        "Agha Salman", "Mohammad Nawaz", "Sahibzada Farhan",
    ]

    weather = WeatherImpact(
        spinner_penalty    = 0.60,
        swing_bonus        = 1.15,
        pace_bounce_bonus  = 1.05,
        yorker_reliability = 0.90,
        dl_planning_needed = False,
        dew_onset_over     = 13,
        warnings           = ["Heavy dew expected from over 13. Spinner use restricted."],
    )

    print("Generating pre-match brief...")
    brief = generate_prematch_brief(
        our_team       = "Lahore Qalandars",
        opposition     = "Karachi Kings",
        venue          = "Gaddafi Stadium, Lahore",
        match_datetime = datetime(2025, 3, 20, 19, 0),
        our_squad      = squad,
        weather_impact = weather,
    )

    print(f"\n{'='*65}")
    print(f"  PRE-MATCH BRIEF: {brief.our_team} vs {brief.opposition}")
    print(f"  Venue: {brief.venue}  |  {brief.match_datetime.strftime('%d %b %Y %H:%M')}")
    print(f"{'='*65}")
    print(f"\n  TOSS: {brief.toss.recommendation}")
    for r in brief.toss.reasoning:
        print(f"    - {r}")

    print(f"\n  PLAYING XI (Option A):")
    for p in brief.xi_options[0].players:
        print(f"    {p.batting_position}. {p.player_name:<25} [{p.role}]")
    print(f"  {brief.xi_options[0].constraint_note}")

    print(f"\n  OPPOSITION ORDER (top 5):")
    for pb in brief.opposition_order.predicted_order[:5]:
        print(f"    {pb.position}. {pb.player_name:<25} [{pb.danger_rating} danger]  {pb.key_note[:50]}")

    print(f"\n  BOWLING PLAN (overs 1-6):")
    for oa in brief.bowling_plan.overs[:6]:
        print(f"    Over {oa.over}: {oa.primary_bowler}")

    print(f"\n  MATCHUP NOTES: {len(brief.matchup_notes)} notes")
    for mn in brief.matchup_notes:
        print(f"    [{mn.confidence}] {mn.note[:70]}")

    print(f"\n  WEATHER: {brief.weather_warnings[0] if brief.weather_warnings else 'None'}")
    print(f"\n{'='*65}\n")
