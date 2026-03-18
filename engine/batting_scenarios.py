# engine/batting_scenarios.py
# Generates 4 scenario cards for the pre-match brief.
#
# rank_players_fallback: transparent rule-based ranking.
# Replace with trained model when models/saved/batting_scenario_model.pkl is available.
#
# Public API:
#   generate_batting_scenarios(our_batters, opposition_bowling, venue, weather) -> list[BattingScenario]

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from utils.situation import WeatherImpact

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJ_ROOT    = Path(__file__).resolve().parent.parent
PLAYER_INDEX = PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv"
PLAYER_INDEX_FALLBACK = PROJ_ROOT.parent / "player_index_2026_enriched.csv"


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class BatterInstruction:
    player_name:  str
    position:     int
    role_in_card: str     # "Anchor" | "Aggressor" | "Finisher" | "Stabiliser"
    instruction:  str     # one-line coaching note


@dataclass
class BattingScenario:
    scenario_id:    str            # "A" | "B" | "C" | "D"
    name:           str
    description:    str
    trigger:        str            # when does this scenario apply?
    batting_order:  list[BatterInstruction]
    key_message:    str            # headline for the card (one sentence)
    weather_note:   str            # "" or weather-adjusted note


# ---------------------------------------------------------------------------
# PLAYER META
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_meta(path: str) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not Path(path).exists():
        return meta
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("player_name", "").strip()
            if name:
                def _f(key: str, default: float, _row: dict = row) -> float:
                    try:
                        v = _row.get(key, "")
                        return float(v) if v and str(v).strip() not in ("", "nan") else default
                    except (ValueError, TypeError):
                        return default
                meta[name] = {
                    "primary_role":   row.get("primary_role",  "Batsman").strip(),
                    "batting_style":  row.get("batting_style", "Right-hand bat").strip(),
                    "t20_career_sr":  _f("t20_career_sr",  120.0),
                    "t20_career_avg": _f("t20_career_avg",   15.0),
                    # New columns from player_index_2026_enriched.csv
                    "bat_sr_set":       _f("bat_sr_set",       0.0),
                    "bat_sr_chase":     _f("bat_sr_chase",     0.0),
                    "innings_sr_delta": _f("innings_sr_delta", 0.0),
                    "bowl_dot_pct":     _f("bowl_dot_pct",     0.0),
                }
    return meta


# ---------------------------------------------------------------------------
# FALLBACK RANKER
# ---------------------------------------------------------------------------

def _get_sr(player: str, scenario_id: str, meta: dict) -> float:
    """
    Return the most appropriate strike rate for this player in this scenario.

    For chase scenarios (C), prefer bat_sr_chase if available.
    For setting scenarios (A, D), prefer bat_sr_set if available.
    Scenario B (collapse) uses career average SR — not context-specific.
    Falls back to t20_career_sr (0.0 means no data, not actual zero SR).
    """
    m = meta.get(player, {})
    if scenario_id == "C":
        chase_sr = m.get("bat_sr_chase", 0.0)
        if chase_sr > 0:
            return chase_sr
    elif scenario_id in ("A", "D"):
        set_sr = m.get("bat_sr_set", 0.0)
        if set_sr > 0:
            return set_sr
    # Scenario B or no innings-split data available
    return float(m.get("t20_career_sr", 120.0) or 120.0)


def _rank_players_fallback(
    players:     list[str],
    scenario_id: str,
    meta:        dict[str, dict],
) -> list[tuple[str, float]]:
    """
    Rule-based player ranking for each scenario.
    Returns list of (player_name, score) sorted best-first.
    """
    results: list[tuple[str, float, int]] = []  # (name, score, role_priority)

    for player in players:
        info   = meta.get(player, {})
        role   = info.get("primary_role", "Batsman")
        sr     = _get_sr(player, scenario_id, meta)
        avg    = info.get("t20_career_avg",  15.0)
        is_lhb = "left" in info.get("batting_style", "").lower()

        if scenario_id == "A":
            # Ideal Start: rank by SR (setting). Batsmen/WK first, All-rounders next, Bowlers last.
            score = sr
            if role in ("Batsman", "Wicketkeeper", "WK-Batsman"):
                prio = 0
            elif role == "All-rounder":
                prio = 1
            else:
                prio = 2

        elif scenario_id == "B":
            # Collapse: rank by avg (anchor ability). Batsmen/All-rounders first.
            score = avg
            prio  = 0 if role in ("Batsman", "WK-Batsman", "Wicketkeeper", "All-rounder") else 1

        elif scenario_id == "C":
            # Death Chase: rank by chase SR. All-rounders/Bowlers with sr>=130 promoted.
            score = sr
            prio  = 0 if (role in ("All-rounder", "Bowler") and sr >= 130) else 1

        else:  # D — Conservative Build
            # Left-hand batters get +5 avg bonus for variety-of-attack consideration.
            score = avg + (5.0 if is_lhb else 0.0)
            prio  = 0  # no role-based priority for scenario D

        results.append((player, score, prio))

    results.sort(key=lambda x: (x[2], -x[1]))
    return [(name, score) for name, score, _ in results]


# ---------------------------------------------------------------------------
# ROLE CLASSIFIER
# ---------------------------------------------------------------------------

def _classify_role(player: str, scenario: str, meta: dict[str, dict]) -> str:
    """
    Assign a role label for a player in this scenario.
    """
    role = meta.get(player, {}).get("primary_role", "Batsman")

    if scenario == "A":   # Ideal — build from platform
        if role in ("Batsman", "Wicketkeeper", "WK-Batsman"):
            return "Aggressor"
        return "Support"

    elif scenario == "B":  # Collapse — reset innings
        if role == "All-rounder":
            return "Stabiliser"
        if role == "Batsman":
            return "Anchor"
        return "Support"

    elif scenario == "C":  # Death chase — power hitting
        if role in ("All-rounder", "Bowler"):
            return "Finisher"
        return "Aggressor"

    else:                  # D — Conservative build
        if role in ("Batsman", "Wicketkeeper", "WK-Batsman"):
            return "Anchor"
        return "Support"


def _build_instruction(
    player:       str,
    position:     int,
    role:         str,
    scenario:     str,
    score:        float,
    meta:         dict[str, dict],
    weather:      WeatherImpact,
    opp_bowling:  dict,
) -> str:
    """Generate a coaching instruction for this batter in this scenario."""
    name_last = player.split()[-1]
    bat_style = meta.get(player, {}).get("batting_style", "Right-hand bat")
    is_left   = "left" in bat_style.lower()
    vs_spin   = float(opp_bowling.get("vs_spin_economy",  7.5))
    vs_pace   = float(opp_bowling.get("vs_pace_economy",  8.0))

    if scenario == "A":
        if position <= 2:
            return f"Keep going — platform set. Target {name_last} scoring at 150+ SR."
        if score >= 60:
            return f"Positive intent from ball 1. {name_last} to accelerate immediately."
        return f"Consolidate for 8-10 balls then accelerate — over 14 onwards target 180+ SR."

    elif scenario == "B":
        if position <= 3:
            return (
                f"{name_last} anchor role — minimum 15 balls before taking risks. "
                f"Get us to over 10 with 4+ wickets in hand."
            )
        if role == "Finisher":
            return f"{name_last} batting at {position} — hold back, explode in last 5 overs."
        return f"Play to your strengths. Rotate strike, rebuild. No unnecessary risks before over 12."

    elif scenario == "C":
        if vs_spin >= 8.5:
            return f"Opposition struggles vs spinners — {name_last} look to slog-sweep if spinner bowled."
        if position <= 3 and score >= 55:
            return f"Already in — accelerate NOW. {name_last} must clear the boundary, no half-measures."
        return f"Come in swinging — {name_last} needs to score at 200+ SR. Every dot is too costly."

    else:  # D
        if position <= 2:
            return (
                f"Patience. Pitch is doing something — {name_last} to play straight, "
                f"rotate strike, no loose shots before over 12."
            )
        if score >= 50:
            return f"{name_last} to push — once in for 15+ balls, play your natural game."
        return f"Solid contribution expected from {name_last}. Avoid the big shot until over 14."


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def generate_batting_scenarios(
    our_batters:       list[str],
    opposition_bowling: dict,
    venue:             str,
    weather:           WeatherImpact,
    player_index_path: Optional[Path] = None,
) -> list[BattingScenario]:
    """
    Generate 4 scenario cards for the pre-match brief.

    Args:
        our_batters:        List of batters in our squad (XI order preferred)
        opposition_bowling: Dict from opposition_profiles (vs_spin_economy, etc.)
        venue:              Match venue
        weather:            WeatherImpact (applied to death-chase scenarios)
        model_path:         Override for batting_scenario_model.pkl
        player_index_path:  Override for player_index.csv

    Returns:
        List of 4 BattingScenario objects (A, B, C, D).
    """
    pi   = str(player_index_path or (PLAYER_INDEX if PLAYER_INDEX.exists() else PLAYER_INDEX_FALLBACK))
    meta = _load_meta(pi)

    scenarios: list[BattingScenario] = []

    scenario_configs = {
        "A": {
            "name":        "Ideal Start",
            "description": "Platform set — 60+ at over 10, <=2 wickets",
            "trigger":     "60+ on board after 10 overs with 2 or fewer wickets down",
            "key_message": "Platform set — batters 3-6 to accelerate immediately and target 190+.",
        },
        "B": {
            "name":        "Tough Start (Collapse)",
            "description": "Under pressure — 3+ wickets by over 6, score <=35",
            "trigger":     "3 or more wickets lost by the end of the powerplay with under 35 on the board",
            "key_message": "Rebuild first, accelerate later — get to over 15 with 4 wickets in hand.",
        },
        "C": {
            "name":        "Death Chase",
            "description": "60+ needed off last 5 overs in second innings",
            "trigger":     "Innings 2, over 15+, required rate above 12 per over",
            "key_message": "Clear the boundary from ball 1 — this is all or nothing.",
        },
        "D": {
            "name":        "Conservative Build",
            "description": "Low-scoring match — pitch doing something, target likely under 150",
            "trigger":     "Low pitch score (CRR under 7.5 at over 12) — wickets in hand matter more than SR",
            "key_message": "Wickets in hand are the asset — build to over 15 then release.",
        },
    }

    for sc_id, cfg in scenario_configs.items():
        ranked = _rank_players_fallback(our_batters, sc_id, meta)

        # Build batting order for this scenario
        batting_order: list[BatterInstruction] = []
        for pos, (player, score) in enumerate(ranked[:8], start=1):
            role        = _classify_role(player, sc_id, meta)
            instruction = _build_instruction(
                player, pos, role, sc_id, score, meta, weather, opposition_bowling
            )
            batting_order.append(BatterInstruction(
                player_name  = player,
                position     = pos,
                role_in_card = role,
                instruction  = instruction,
            ))

        # Weather note for the card
        w_note = ""
        if sc_id == "C" and weather.yorker_reliability <= 0.82:
            w_note = (
                f"Wind {weather.yorker_reliability:.0%} yorker reliability — "
                f"target wide yorker areas, expect more full-tosses."
            )
        elif sc_id in ("A", "D") and weather.swing_bonus >= 1.20:
            w_note = (
                f"Swing conditions early — cautious start for openers in PP. "
                f"Don't gift your wicket to swing in overs 1-3."
            )
        elif sc_id == "B" and weather.dl_planning_needed:
            w_note = (
                "Rain risk — D/L rewards early runs even in a collapse. "
                "Don't be overly conservative in overs 6-10."
            )

        scenarios.append(BattingScenario(
            scenario_id  = sc_id,
            name         = cfg["name"],
            description  = cfg["description"],
            trigger      = cfg["trigger"],
            batting_order= batting_order,
            key_message  = cfg["key_message"],
            weather_note = w_note,
        ))

    return scenarios


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils.situation import WeatherImpact

    lahore_batters = [
        "Fakhar Zaman", "Abdullah Shafique", "Sikandar Raza",
        "Shaheen Shah Afridi", "Liam Dawson", "Mohammad Hafeez",
        "Rashid Khan", "Haris Rauf",
    ]

    opp_bowling = {
        "vs_spin_economy":     8.8,
        "vs_pace_economy":     7.2,
        "pace_overs_pct":      50.0,
        "spin_overs_pct":      20.0,
    }

    weather = WeatherImpact(
        spinner_penalty    = 0.75,
        swing_bonus        = 1.20,
        pace_bounce_bonus  = 1.05,
        yorker_reliability = 0.92,
        dl_planning_needed = False,
        dew_onset_over     = 15,
        warnings           = [],
    )

    cards = generate_batting_scenarios(
        lahore_batters, opp_bowling,
        "Gaddafi Stadium, Lahore", weather
    )

    print(f"\n{'='*65}")
    print(f"  Batting Scenario Cards -- Lahore Qalandars")
    print(f"{'='*65}")

    for card in cards:
        print(f"\n  [{card.scenario_id}] {card.name}")
        print(f"  Trigger : {card.trigger}")
        print(f"  Message : {card.key_message}")
        if card.weather_note:
            print(f"  Weather : {card.weather_note}")
        print(f"  {'Pos':<4}  {'Player':<25}  {'Role':<12}  Instruction")
        print(f"  {'-'*80}")
        for bi in card.batting_order:
            print(
                f"  {bi.position:<4}  {bi.player_name:<25}  "
                f"{bi.role_in_card:<12}  {bi.instruction}"
            )

    print(f"\n{'='*65}\n")
