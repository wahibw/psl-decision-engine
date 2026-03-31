# engine/bowling_plan.py
# Generates a 20-over bowling plan template.
# Used in Match Prep Room to produce the over-by-over bowling card.
#
# Public API:
#   generate_bowling_plan(bowlers, opposition, weather, venue, context) -> BowlingPlan

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.situation import LiveMatchState, WeatherImpact

# ---------------------------------------------------------------------------
# EXCEPTIONS
# ---------------------------------------------------------------------------

class PlanValidationError(Exception):
    """Raised when a generated bowling plan violates the PSL 4-over cap rules."""


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJ_ROOT    = Path(__file__).resolve().parent.parent
STATS_PATH   = PROJ_ROOT / "data" / "processed" / "player_stats.parquet"
PLAYER_INDEX = PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv"
PLAYER_INDEX_FALLBACK = PROJ_ROOT.parent / "player_index_2026_enriched.csv"
OPP_PROFILES       = PROJ_ROOT / "data" / "processed" / "opposition_profiles.csv"

# Current PSL season — used to blend career aggregate with recent-season data.
# Update each season when build_opposition_profiles re-runs with new data.
CURRENT_PSL_SEASON = 2025

# Blend weight for current-season data vs career aggregate.
# 0.60 = current season contributes 60%, career 40%.
OPP_CURRENT_SEASON_WEIGHT = 0.60
MATCHUP_PATH       = PROJ_ROOT / "data" / "processed" / "matchup_matrix.parquet"
RECENT_FORM_PATH   = PROJ_ROOT / "data" / "processed" / "recent_form.parquet"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MAX_OVERS_PER_BOWLER = 4
TOTAL_OVERS          = 20

# League-average fallbacks. Source: PSL 2019-2025 all-bowler career averages.
LEAGUE_AVG_DOT_PCT   = 35.0   # % of balls that are dots across all PSL bowlers

PHASE_LABEL = {
    0:  "PP",        1:  "PP",        2:  "PP",        3:  "PP",        4:  "PP",        5:  "PP",
    6:  "Early-Mid", 7:  "Early-Mid", 8:  "Early-Mid", 9:  "Early-Mid",
    10: "Late-Mid",  11: "Late-Mid",  12: "Late-Mid",  13: "Late-Mid",
    14: "Pre-Death", 15: "Pre-Death",
    16: "Death",     17: "Death",     18: "Death",     19: "Death",
}

PHASE_NAME = {
    "PP":         "powerplay",
    "Early-Mid":  "middle",       # _phase_fitness alias
    "Late-Mid":   "middle",       # _phase_fitness alias (differentiated internally)
    "Mid":        "middle",       # backward-compat alias
    "Pre-Death":  "pre-death",
    "Death":      "death",
}


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class OverAssignment:
    over:           int         # 1-based (over 1 = first over)
    primary_bowler: str
    backup_bowler:  str
    phase:          str         # "PP" | "Mid" | "Death"
    reason:         str
    weather_note:   str         # "" or warning text


@dataclass
class BowlingPlan:
    overs:          list[OverAssignment]   # 20 items, over 1-20
    bowler_summary: dict[str, list[int]]   # {"BowlerName": [1,3,7,8]} (allocated overs)
    key_decisions:  list[str]
    contingencies:  list[str]
    plan_warnings:  list[str] = field(default_factory=list)  # cap violations surfaced to UI


# ---------------------------------------------------------------------------
# PLAYER META
# ---------------------------------------------------------------------------

def _load_player_meta(pi_path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not pi_path.exists():
        return meta
    with open(pi_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name  = row.get("player_name", "").strip()
            style = (row.get("bowling_style") or "").lower()
            if not name:
                continue
            def _f(key: str, default: float, _row: dict = row) -> float:
                try:
                    v = _row.get(key, "")
                    return float(v) if v and str(v).strip() not in ("", "nan") else default
                except (ValueError, TypeError):
                    return default
            meta[name] = {
                "primary_role":  row.get("primary_role", "Batsman").strip(),
                "bowling_style": row.get("bowling_style", "").strip(),
                "is_pace":  any(w in style for w in ("fast","medium","seam","swing","pace")),
                "is_spin":  any(w in style for w in ("spin","off","leg","googly","chinaman","slow")),
                # New columns from player_index_2026_enriched.csv
                "bat_sr_set":       _f("bat_sr_set",       0.0),
                "bat_sr_chase":     _f("bat_sr_chase",     0.0),
                "innings_sr_delta": _f("innings_sr_delta", 0.0),
                "bowl_dot_pct":     _f("bowl_dot_pct",     0.0),
            }
    return meta


def _is_genuine_bowler(player: str, meta: dict[str, dict]) -> bool:
    m = meta.get(player, {})
    return m.get("primary_role", "Batsman") in ("Bowler", "All-rounder")


def _bowl_type(player: str, meta: dict[str, dict]) -> str:
    m = meta.get(player, {})
    if m.get("is_pace"):
        return "pace"
    if m.get("is_spin"):
        return "spin"
    return "unknown"


# ---------------------------------------------------------------------------
# BOWLER PHASE STATS
# ---------------------------------------------------------------------------

def _load_t20_proxy(pi_path: Path) -> dict[str, dict]:
    """
    Load T20 career stats from player_index.csv for bowlers with no PSL data.
    Returns {player_name: {t20_eco, data_tier}}
    """
    proxy: dict[str, dict] = {}
    if not pi_path or not pi_path.exists():
        return proxy
    with open(pi_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("player_name", "").strip()
            if not name:
                continue
            def _safe(key: str, _row: dict = row) -> float:
                try:
                    return float(_row.get(key) or 0) or 0.0
                except (ValueError, TypeError):
                    return 0.0
            t20_eco     = _safe("t20_career_economy")
            ipl_eco     = _safe("ipl_career_economy")
            ipl_matches = int(_safe("ipl_matches"))
            try:
                tier = int(float(row.get("data_tier") or 3))
            except (ValueError, TypeError):
                tier = 3
            # Prefer IPL economy (subcontinent conditions ≈ PSL) when available
            use_ipl = ipl_matches >= 5 and ipl_eco > 0
            proxy[name] = {
                "t20_eco":      ipl_eco  if use_ipl else t20_eco,
                "data_tier":    2        if use_ipl else tier,
                "bowl_dot_pct": _safe("bowl_dot_pct"),   # T20I dot ball % — real data for Tier 2/3
            }
    return proxy


def _load_opposition_profile(
    opposition_team: str,
    opp_path: Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """
    Load opposition profile from opposition_profiles.csv.

    Freshness strategy:
      1. Always load season=0 (career aggregate) as the base.
      2. If a CURRENT_PSL_SEASON row exists, blend:
            60% current-season + 40% career for all numeric stats.
         Blended profile carries profile_freshness="current-season".
      3. If no current-season row, profile_freshness="career-only" and a
         staleness note is stored so callers can surface a plan warning.
      4. If analyst-supplied `overrides` dict is provided, it is applied on
         top of the blended profile.  Supported keys:
            vs_spin_economy, vs_pace_economy, powerplay_sr, death_sr,
            left_hand_top6_pct, vs_leftarm_economy — numeric stat overrides.
            injury_notes (str)  — free-text e.g. "Babar out with hamstring"
            injured_out (list)  — names of unavailable batters
            form_note (str)     — free-text form context
         After applying overrides, profile_freshness="override-applied".

    Returns a dict with keys:
        left_hand_top6_pct, vs_spin_economy, vs_pace_economy,
        vs_leftarm_economy, powerplay_sr, death_sr, is_estimated,
        profile_freshness, staleness_note, injury_notes,
        injured_out, form_note
    """
    NEUTRAL = {
        "left_hand_top6_pct":  33.0,
        "vs_spin_economy":      8.0,
        "vs_legspin_economy":   8.0,   # fallback == vs_spin_economy if column absent
        "vs_offspin_economy":   8.0,
        "vs_leftarm_spin_economy": 8.0,
        "vs_pace_economy":      8.0,
        "vs_leftarm_economy":   8.0,
        "powerplay_sr":       130.0,
        "death_sr":           155.0,
        "is_estimated":        True,
    }
    path = opp_path or OPP_PROFILES
    if not path.exists() or not opposition_team:
        return NEUTRAL
    try:
        df = pd.read_csv(path)
        # Career aggregate row (season == 0) for the team
        mask = (df["team"] == opposition_team) & (df["season"] == 0)
        row  = df[mask]
        if row.empty:
            # Try partial match (case-insensitive)
            mask = df["team"].str.lower().str.contains(opposition_team.lower(), na=False) \
                   & (df["season"] == 0)
            row = df[mask]
        if row.empty:
            # No match found — return neutral defaults with is_estimated=True so
            # callers can show a clear "TEAM NOT FOUND — estimated profile" warning.
            import warnings
            warnings.warn(
                f"Opposition profile not found for '{opposition_team}'. "
                f"Using neutral league-average defaults (is_estimated=True). "
                f"Check team name spelling against opposition_profiles.csv.",
                UserWarning,
                stacklevel=3,
            )
            return NEUTRAL
        r = row.iloc[0]
        _ZERO_NULL_KEYS = ("economy", "sr", "pct", "score", "rate")
        def _f(col: str, default: float) -> float:
            try:
                v = r.get(col)
                result = float(v) if pd.notna(v) else default
                # 0.0 in economy/SR/pct/score/rate columns = missing data, not a genuine zero
                if result == 0.0 and any(k in col for k in _ZERO_NULL_KEYS):
                    return default
                return result
            except (ValueError, TypeError):
                return default
        # Generic vs_spin_economy is the fallback for any subtype column that
        # is absent from the CSV (pre-2026 data files won't have the subtype split).
        generic_spin_eco = _f("vs_spin_economy", 8.0)
        profile: dict = {
            "left_hand_top6_pct":      _f("left_hand_top6_pct",         33.0),
            "vs_spin_economy":         generic_spin_eco,
            # Spin subtype columns — fall back to generic if not in CSV yet
            "vs_legspin_economy":      _f("vs_legspin_economy",          generic_spin_eco),
            "vs_offspin_economy":      _f("vs_offspin_economy",          generic_spin_eco),
            "vs_leftarm_spin_economy": _f("vs_leftarm_spin_economy",     generic_spin_eco),
            "vs_pace_economy":         _f("vs_pace_economy",              8.0),
            "vs_leftarm_economy":      _f("vs_leftarm_economy",           8.0),
            "powerplay_sr":            _f("powerplay_sr",                130.0),
            "death_sr":                _f("death_sr",                   155.0),
            "is_estimated":            bool(r.get("is_estimated", False)),
            # Freshness metadata
            "profile_freshness": "career-only",
            "staleness_note":    "",
            "injury_notes":      "",
            "injured_out":       [],
            "form_note":         "",
        }

        # ----------------------------------------------------------------
        # FRESHNESS BLEND: try current-season row
        # ----------------------------------------------------------------
        _BLEND_KEYS = (
            "vs_spin_economy", "vs_legspin_economy", "vs_offspin_economy",
            "vs_leftarm_spin_economy", "vs_pace_economy", "vs_leftarm_economy",
            "powerplay_sr", "death_sr", "left_hand_top6_pct",
        )
        season_mask = (df["team"].str.lower() == opposition_team.lower()) \
                      & (df["season"] == CURRENT_PSL_SEASON)
        season_row  = df[season_mask]

        if not season_row.empty:
            sr = season_row.iloc[0]
            def _fs(col: str, default: float, _sr=sr) -> float:
                try:
                    v = _sr.get(col)
                    result = float(v) if pd.notna(v) else default
                    return default if result == 0.0 and any(k in col for k in _ZERO_NULL_KEYS) else result
                except (ValueError, TypeError):
                    return default
            W = OPP_CURRENT_SEASON_WEIGHT
            for key in _BLEND_KEYS:
                career_val  = profile.get(key, 0.0)
                current_val = _fs(key, float(career_val))
                profile[key] = round(W * current_val + (1.0 - W) * career_val, 2)
            profile["profile_freshness"] = "current-season"
        else:
            profile["staleness_note"] = (
                f"Opposition profile for {opposition_team} uses career aggregate only "
                f"(no PSL {CURRENT_PSL_SEASON} data found). "
                f"Consider manual overrides for any known in-season form/injury changes."
            )

        # ----------------------------------------------------------------
        # ANALYST OVERRIDES (injuries, in-season form, manual corrections)
        # ----------------------------------------------------------------
        if overrides:
            _NUMERIC_OVERRIDE_KEYS = {
                "vs_spin_economy", "vs_pace_economy", "vs_leftarm_economy",
                "vs_legspin_economy", "vs_offspin_economy", "vs_leftarm_spin_economy",
                "powerplay_sr", "death_sr", "left_hand_top6_pct",
            }
            for key, val in overrides.items():
                if key in _NUMERIC_OVERRIDE_KEYS:
                    try:
                        profile[key] = float(val)
                    except (ValueError, TypeError):
                        pass
            profile["injury_notes"] = str(overrides.get("injury_notes", ""))
            profile["injured_out"]  = list(overrides.get("injured_out",  []))
            profile["form_note"]    = str(overrides.get("form_note",     ""))
            profile["profile_freshness"] = "override-applied"

        return profile
    except Exception:
        return NEUTRAL


def _load_bowler_phase_stats(
    bowlers:    list[str],
    stats_path: Path,
    pi_path:    Path | None = None,
) -> dict[str, dict]:
    """
    Returns {player: {pp_economy, mid_economy, death_economy,
                       pp_wkts_po, mid_wkts_po, death_wkts_po,
                       pp_overs, mid_overs, death_overs}}

    For players with no PSL parquet entry, falls back to t20_career_economy
    from player_index.csv with phase multipliers:
      PP    = t20_eco × 1.05  (opening overs slightly harder)
      Mid   = t20_eco × 1.00  (baseline)
      Death = t20_eco × 1.12  (death overs tougher)
    and sets proxy_overs=12 so _phase_fitness() confidence ≈ 0.60
    (reflects real uncertainty vs a player with 20+ PSL phase overs).
    """
    t20_proxy = _load_t20_proxy(pi_path) if pi_path else {}

    try:
        df = pd.read_parquet(stats_path)
        career = df[df["season"] == 0]
    except Exception:
        career = pd.DataFrame()

    result: dict[str, dict] = {}

    for b in bowlers:
        def _get(phase: str, col: str, default: float, _b: str = b) -> float:
            if career.empty:
                return default
            row = career[(career["player_name"] == _b) & (career["phase"] == phase)]
            if row.empty:
                return default
            v = row.iloc[0].get(col)
            return float(v) if pd.notna(v) else default

        def _wkts_po(phase: str, _b: str = b) -> float:
            if career.empty:
                return 0.2
            row = career[(career["player_name"] == _b) & (career["phase"] == phase)]
            if row.empty:
                return 0.2
            overs = float(row.iloc[0].get("bowl_overs") or 0)
            wkts  = float(row.iloc[0].get("bowl_wickets") or 0)
            return round(wkts / overs, 3) if overs > 0 else 0.2

        def _overs(phase: str, _b: str = b) -> float:
            if career.empty:
                return 0.0
            row = career[(career["player_name"] == _b) & (career["phase"] == phase)]
            if row.empty:
                return 0.0
            v = row.iloc[0].get("bowl_overs")
            return float(v) if pd.notna(v) else 0.0

        pp_ov    = _overs("powerplay")
        mid_ov   = _overs("middle")
        death_ov = _overs("death")
        total_ov = pp_ov + mid_ov + death_ov

        # Player has meaningful PSL data — use it directly
        if total_ov >= 2.0:
            # bowl_dot_pct priority: PSL parquet > player index T20I > league avg (35.0)
            px_dot      = t20_proxy.get(b, {})
            pi_dot_pct  = px_dot.get("bowl_dot_pct", 0.0)
            parquet_dot = _get("overall", "bowl_dot_pct", 0.0)
            if parquet_dot > 0:
                bowl_dot = parquet_dot
            elif pi_dot_pct > 0:
                bowl_dot = pi_dot_pct   # use T20I value when PSL parquet has no data
            else:
                bowl_dot = LEAGUE_AVG_DOT_PCT  # final fallback to league average
            result[b] = {
                "pp_economy":    _get("powerplay", "bowl_economy",  8.5),
                "mid_economy":   _get("middle",    "bowl_economy",  7.5),
                "death_economy": _get("death",     "bowl_economy",  9.5),
                "pp_wkts_po":    _wkts_po("powerplay"),
                "mid_wkts_po":   _wkts_po("middle"),
                "death_wkts_po": _wkts_po("death"),
                "pp_overs":      pp_ov,
                "mid_overs":     mid_ov,
                "death_overs":   death_ov,
                "bowl_dot_pct":  bowl_dot,
            }
            continue

        # No / negligible PSL data — fall back to T20 career proxy
        px  = t20_proxy.get(b, {})
        eco = px.get("t20_eco", 0.0)
        if eco <= 0.0:
            # No T20 career data either — use safe generic defaults
            eco = 8.5
        # Phase-specific multipliers mirroring real T20 patterns
        pp_eco    = round(eco * 1.05, 2)
        mid_eco   = round(eco * 1.00, 2)
        death_eco = round(eco * 1.12, 2)
        # proxy_overs=12 → confidence = 12/20 = 0.60 in _phase_fitness()
        PROXY_OVERS  = 12.0
        pi_dot_pct   = px.get("bowl_dot_pct", 0.0)
        bowl_dot     = pi_dot_pct if pi_dot_pct > 0 else LEAGUE_AVG_DOT_PCT
        result[b] = {
            "pp_economy":    pp_eco,
            "mid_economy":   mid_eco,
            "death_economy": death_eco,
            "pp_wkts_po":    0.24,   # generic T20 wicket rate
            "mid_wkts_po":   0.22,
            "death_wkts_po": 0.26,
            "pp_overs":      PROXY_OVERS,
            "mid_overs":     PROXY_OVERS,
            "death_overs":   PROXY_OVERS,
            "bowl_dot_pct":  bowl_dot,
        }

    return result


# ---------------------------------------------------------------------------
# RECENT FORM LOADER
# ---------------------------------------------------------------------------

# Minimum recent overs before we trust recent economy over career average.
# Below this, recent_eco is noise (2-3 overs in last 10 matches tells us nothing).
MIN_RECENT_OVERS_FOR_ECO_BLEND = 5.0

# How much recent form overrides career economy when we have enough recent data.
# 0.60 = recent form is 60% of the blended economy, career 40%.
# Rationale: last ~5 matches are 3-5× more predictive of next-match performance
# than career average for in-form / out-of-form diagnosis.
RECENT_ECO_WEIGHT = 0.60

# Threshold beyond which a bowler is flagged as "form concern" in plan_warnings.
# 15% worse than career = meaningful enough to mention to the captain.
FORM_CONCERN_ECO_DELTA_PCT = 0.15


def _load_recent_form(bowlers: list[str], venue: str = "") -> dict[str, dict]:
    """
    Load recent form data from recent_form.parquet for a list of bowlers.

    Returns {player_name: {
        "form_score":    float,   # 0-100 composite (50 = neutral)
        "recent_eco":    float,   # economy in last ~10 matches (0.0 = no data)
        "recent_overs":  float,   # overs bowled in that window
        "trend":         str,     # "improving" | "stable" | "declining"
        "venue_eco":     float,   # economy at THIS venue (0.0 = no data)
    }}

    Defaults to neutral values on any error or missing data.
    """
    neutral = {"form_score": 50.0, "recent_eco": 0.0,
               "recent_overs": 0.0, "trend": "stable", "venue_eco": 0.0}
    result: dict[str, dict] = {b: dict(neutral) for b in bowlers}

    if not RECENT_FORM_PATH.exists():
        return result
    try:
        df      = pd.read_parquet(RECENT_FORM_PATH)
        overall = df[df["venue"] == ""]
        venue_df = df[df["venue"] == venue] if venue else pd.DataFrame()

        for b in bowlers:
            row = overall[overall["player_name"] == b]
            if row.empty:
                continue
            r = row.iloc[0]

            def _f(col: str, default: float, _r=r) -> float:
                v = _r.get(col, default)
                return float(v) if pd.notna(v) else default

            recent_overs = _f("bowl_overs", 0.0)
            recent_eco   = _f("bowl_economy", 0.0)
            form_score   = _f("bowl_form_score", 50.0)
            trend        = str(r.get("bowl_trend", "stable") or "stable")

            # Venue-specific economy
            venue_eco = 0.0
            if not venue_df.empty:
                vrow = venue_df[venue_df["player_name"] == b]
                if not vrow.empty:
                    ve = vrow.iloc[0].get("venue_bowl_economy", 0.0)
                    venue_eco = float(ve) if pd.notna(ve) and float(ve) > 0 else 0.0

            result[b] = {
                "form_score":   form_score,
                "recent_eco":   recent_eco   if recent_overs >= MIN_RECENT_OVERS_FOR_ECO_BLEND else 0.0,
                "recent_overs": recent_overs,
                "trend":        trend,
                "venue_eco":    venue_eco,
            }
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# PHASE FITNESS SCORING
# ---------------------------------------------------------------------------

def _phase_fitness(
    b:               str,
    phase:           str,
    stats:           dict[str, dict],
    meta:            dict[str, dict],
    weather:         WeatherImpact,
    opp:             dict | None = None,
    bowl_form_score: float = 50.0,
    recent_eco:      float = 0.0,    # overall economy in last ~10 matches (0 = unavailable)
) -> float:
    """
    Return a 0-100 fitness score for a bowler in a specific phase.
    Higher = better suited to this phase for THIS match.

    Weights by phase (reflecting real T20 priorities):
      PP    — economy 40% | wickets 40% | dots 20%
              Pace swing bonus applied if conditions favour movement.
      Mid   — economy 50% | wickets 30% | dots 20%
              Spinners get bonus on dry/turning pitches (low swing_bonus).
      Death — economy 25% | wickets 55% | dots 20%
              Wicket-taking is the #1 death priority.
              Spinners heavily penalised if dew is present.

    Opposition adjustments (when opp dict is provided):
      - vs_spin_economy < 7.0 → spinners punished hard → -8 spinner fitness
      - vs_spin_economy > 9.5 → spinners thrive → +8 spinner fitness
      - vs_pace_economy < 7.5 → pace punished → -6 pace fitness
      - vs_pace_economy > 9.5 → pace thrive → +6 pace fitness
      - left_hand_top6_pct >= 50 → left-arm bowlers +6 (LH vs left-arm angle)
      - powerplay_sr > 155 → PP wicket-takers boosted +5 (aggressive openers)
      - death_sr > 175    → death wicket-takers boosted +5 (aggressive finishers)

    Sample reliability: a bowler with <10 phase overs has their score
    blended toward the generic average (50) to avoid over-relying on
    small samples (e.g. Mustafizur with only 5 PP overs).
    """
    s   = stats.get(b, {})
    bt  = _bowl_type(b, meta)
    opp = opp or {}

    # Opposition batting tendencies
    opp_vs_pace_eco  = opp.get("vs_pace_economy",    8.0)
    opp_vs_larm_eco  = opp.get("vs_leftarm_economy", 8.0)
    opp_lh_pct       = opp.get("left_hand_top6_pct", 33.0)
    opp_pp_sr        = opp.get("powerplay_sr",       130.0)
    opp_death_sr     = opp.get("death_sr",           155.0)

    # Spin subtype economy: use per-style column if available, else generic
    bowling_style = meta.get(b, {}).get("bowling_style", "").lower()
    is_left_arm   = "left" in bowling_style
    if "leg" in bowling_style or "googly" in bowling_style or "chinaman" in bowling_style:
        opp_vs_spin_eco = opp.get("vs_legspin_economy", opp.get("vs_spin_economy", 8.0))
    elif "left" in bowling_style and any(w in bowling_style for w in ("orthodox", "spin", "slow")):
        opp_vs_spin_eco = opp.get("vs_leftarm_spin_economy", opp.get("vs_spin_economy", 8.0))
    else:
        # off-spin, right-arm spin, other slow variants
        opp_vs_spin_eco = opp.get("vs_offspin_economy", opp.get("vs_spin_economy", 8.0))

    def _norm(v: float, lo: float, hi: float) -> float:
        return max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100.0

    # Recent-form economy blend: when we have enough recent data, shift each
    # phase economy toward recent form.  Career economy alone misses a bowler
    # who was econ 7.5 career but 9.8 in their last 6 matches.
    # recent_eco=0.0 means unavailable — blend is skipped.
    def _blend_eco(career_eco: float) -> float:
        if recent_eco > 0.0:
            return round(career_eco * (1.0 - RECENT_ECO_WEIGHT) + recent_eco * RECENT_ECO_WEIGHT, 2)
        return career_eco

    if phase == "PP":
        eco    = _blend_eco(s.get("pp_economy",  8.5))
        wpo    = s.get("pp_wkts_po",  0.2)
        # bowl_dot_pct priority: PSL parquet > player index T20I > league avg (35.0)
        # Resolved upstream in _load_bowler_phase_stats(); 35.0 is the last-resort default.
        dots   = s.get("bowl_dot_pct", LEAGUE_AVG_DOT_PCT)
        sample = s.get("pp_overs",    0.0)

        # Swing conditions boost pace bowlers in PP.
        # seam_swing_bonus() gives full bonus to swing/seam specialists,
        # 35% of the bonus to pure express-pace bowlers.
        if bt == "pace":
            style_lower = meta.get(b, {}).get("bowling_style", "").lower()
            effective_bonus = weather.seam_swing_bonus(style_lower)
            eco = eco / max(0.5, effective_bonus)

        eco_s  = _norm(12.0 - eco,  0.0, 6.0)   # economy: lower raw = higher score
        wpo_s  = _norm(wpo,         0.0, 0.55)
        dot_s  = _norm(dots,        25.0, 60.0)
        # PP: economy outweighs wickets — one conceded boundary is not recoverable.
        # Research (IPL/BBL) shows PP economy is a stronger match outcome predictor
        # than PP wicket count. Weights adjusted accordingly.
        raw    = eco_s * 0.50 + wpo_s * 0.30 + dot_s * 0.20

        # Opposition PP SR: aggressive openers → reward wicket-takers
        if opp_pp_sr > 155:
            raw += 5.0  # phase-level bonus for ALL bowlers (we need early wickets)
        elif opp_pp_sr < 115:
            raw -= 3.0  # cautious openers — economy matters more

        # Opposition vs pace / spin
        if bt == "pace" and opp_vs_pace_eco < 7.5:
            raw -= 6.0  # they handle pace well — pace bowlers less effective
        elif bt == "pace" and opp_vs_pace_eco > 9.5:
            raw += 6.0
        if bt == "spin" and opp_vs_spin_eco < 7.0:
            raw -= 8.0  # they punish spinners in PP
        elif bt == "spin" and opp_vs_spin_eco > 9.5:
            raw += 8.0

        # Hard structural penalty: spinners in PP are a tactical risk in T20.
        # Modern aggressive openers target spinners in the powerplay ring field.
        # Apply a base penalty unless conditions strongly favour spin (no-swing
        # + high opposition vs-pace economy) OR spinner has elite PP sample data.
        if bt == "spin":
            _spin_pp_justified = (
                weather.swing_bonus < 1.05         # no swing — pace less lethal
                and opp_vs_pace_eco < 7.5          # openers handle pace well
                and opp_vs_spin_eco > 9.0          # openers genuinely struggle vs spin
            )
            if not _spin_pp_justified:
                raw -= 12.0   # strong default: spinner in PP is the exception, not rule
            elif sample < 6:
                raw -= 4.0    # justified but small sample — moderate caution

        # Pace structural bonus in PP — new ball + ring field favour pace
        if bt == "pace":
            raw += 5.0   # pace has natural PP advantage regardless of swing

        # Left-hand heavy batting lineup: left-arm bowlers get extra angle advantage
        if is_left_arm and opp_lh_pct >= 50:
            raw += 6.0

    elif phase == "Death":
        eco    = _blend_eco(s.get("death_economy",  9.5))
        wpo    = s.get("death_wkts_po",  0.2)
        dots   = s.get("bowl_dot_pct",  LEAGUE_AVG_DOT_PCT)
        sample = s.get("death_overs",    0.0)

        # Dew degrades spinners at death — use over 18 as representative death over
        # so the gradient (not binary) penalty correctly reflects late-innings dew build-up.
        if bt == "spin":
            eco = eco / max(0.3, weather.spinner_penalty_at(18))

        eco_s  = _norm(14.0 - eco,  0.0, 8.0)   # wider range at death (higher ECO acceptable)
        wpo_s  = _norm(wpo,         0.0, 0.65)   # wickets are the priority at death
        dot_s  = _norm(dots,        25.0, 55.0)
        raw    = eco_s * 0.25 + wpo_s * 0.55 + dot_s * 0.20

        # Aggressive death batting → wicket-taking bowlers become more valuable
        if opp_death_sr > 175:
            raw += 5.0
        elif opp_death_sr < 140:
            raw -= 3.0  # conservative finishers → economy more important

        # Opposition vs pace / spin at death
        if bt == "pace" and opp_vs_pace_eco < 7.5:
            raw -= 5.0
        elif bt == "pace" and opp_vs_pace_eco > 9.5:
            raw += 5.0
        if bt == "spin" and opp_vs_spin_eco < 7.0:
            raw -= 7.0
        elif bt == "spin" and opp_vs_spin_eco > 9.5:
            raw += 5.0

        # Left-arm angle at death
        if is_left_arm and opp_lh_pct >= 50:
            raw += 5.0

    elif phase in ("Early-Mid", "middle", "Mid"):
        # Early-Mid: overs 7-10. Spinners dominate, containment + dot-ball pressure.
        # Batters may still be cautious/new. Economy + dots outweigh wickets.
        eco    = _blend_eco(s.get("mid_economy",  8.0))
        wpo    = s.get("mid_wkts_po",  0.2)
        dots   = s.get("bowl_dot_pct", LEAGUE_AVG_DOT_PCT)
        sample = s.get("mid_overs",    0.0)

        # Dry/slow conditions: spinners get a stronger boost in early-mid
        if bt == "spin" and weather.swing_bonus <= 1.05:
            eco = eco * 0.88    # larger bonus than generic mid (was 0.92)

        # No meaningful dew at overs 7-10 — use over 8 representative (pre-onset)
        # (spinner_penalty_at(8) returns 1.0 unless onset is unusually early)
        if bt == "spin":
            early_dew = weather.spinner_penalty_at(8)
            if early_dew < 1.0:
                eco = eco / max(0.5, early_dew)

        eco_s  = _norm(12.0 - eco,  0.0, 6.0)
        wpo_s  = _norm(wpo,         0.0, 0.50)
        dot_s  = _norm(dots,        25.0, 58.0)
        # Economy 45% | Wickets 30% | Dots 25% — containment priority in early-mid
        raw    = eco_s * 0.45 + wpo_s * 0.30 + dot_s * 0.25

        # Opposition vs spin in early-mid: the prime spinner window
        if bt == "spin" and opp_vs_spin_eco < 7.0:
            raw -= 8.0  # they attack spinners hard
        elif bt == "spin" and opp_vs_spin_eco > 9.5:
            raw += 9.0  # they struggle vs spin — prioritise spinners here

        if bt == "pace" and opp_vs_pace_eco < 7.5:
            raw -= 4.0
        elif bt == "pace" and opp_vs_pace_eco > 9.5:
            raw += 4.0

        if is_left_arm and opp_lh_pct >= 50:
            raw += 4.0

    else:  # Late-Mid: overs 11-14
        # Batters are set and looking to accelerate. Wickets become critical.
        # Partial dew may be building (onset ~13). Pace change-ups more valuable.
        eco    = _blend_eco(s.get("mid_economy",  8.0))
        wpo    = s.get("mid_wkts_po",  0.2)
        dots   = s.get("bowl_dot_pct", LEAGUE_AVG_DOT_PCT)
        sample = s.get("mid_overs",    0.0)

        # Dry conditions still help spinners slightly in late-mid
        if bt == "spin" and weather.swing_bonus <= 1.05:
            eco = eco * 0.94

        # Partial dew in late-mid (e.g. onset=13, over 14 = 25% intensity)
        if bt == "spin":
            late_dew_penalty = weather.spinner_penalty_at(14)
            if late_dew_penalty < 1.0:
                eco = eco / max(0.5, late_dew_penalty)

        eco_s  = _norm(12.0 - eco,  0.0, 6.0)
        wpo_s  = _norm(wpo,         0.0, 0.50)
        dot_s  = _norm(dots,        25.0, 58.0)
        # Economy 40% | Wickets 40% | Dots 20% — shift toward wickets as batters attack
        raw    = eco_s * 0.40 + wpo_s * 0.40 + dot_s * 0.20

        # Opposition vs spin: spinners slightly less dominant in late-mid (set batters)
        if bt == "spin" and opp_vs_spin_eco < 7.0:
            raw -= 9.0  # they punish spin when set
        elif bt == "spin" and opp_vs_spin_eco > 9.5:
            raw += 7.0

        if bt == "pace" and opp_vs_pace_eco < 7.5:
            raw -= 5.0
        elif bt == "pace" and opp_vs_pace_eco > 9.5:
            raw += 6.0  # pace change-ups effective against set batters

        if is_left_arm and opp_lh_pct >= 50:
            raw += 4.0

    # Sample reliability blend: small-sample bowlers converge toward 45
    RELIABLE_OVERS = 20.0
    confidence = min(1.0, sample / RELIABLE_OVERS)
    career_fit = raw * confidence + 45.0 * (1.0 - confidence)

    # Blend in recent form (30% weight). bowl_form_score is 0-100 where 50=neutral.
    # Normalise to 0-100 scale (career_fit is already in 0-100 range from _norm()).
    return round(career_fit * 0.70 + bowl_form_score * 0.30, 2)


# ---------------------------------------------------------------------------
# PHASE SPECIALIST CLASSIFICATION
# ---------------------------------------------------------------------------

def _classify_bowlers(
    bowlers:     list[str],
    stats:       dict[str, dict],
    meta:        dict[str, dict],
    weather:     WeatherImpact,
    opp:         dict | None = None,
    venue:       str = "",
) -> dict[str, list[str]]:
    """
    Classify each bowler into phases, ranked by _phase_fitness (descending).
    All genuine bowlers appear in all phase pools; part-timers only in Mid.
    Spinners are excluded from Death pool if severe dew.
    Pre-Death pool (overs 15-16): medium-pace / secondary options — top death
    specialists are reserved for overs 17-20.
    Opposition profile (opp) adjusts fitness scores when provided.
    Recent bowl_form_score (0-100) blended at 30% weight; recent economy
    (last ~10 matches) blended at 60% into phase economies when available.
    """
    recent_form_data = _load_recent_form(bowlers, venue)

    pp_bowlers:         list[tuple[str, float]] = []
    early_mid_bowlers:  list[tuple[str, float]] = []
    late_mid_bowlers:   list[tuple[str, float]] = []
    predeath_bowlers:   list[tuple[str, float]] = []
    death_bowlers:      list[tuple[str, float]] = []

    for b in bowlers:
        bt         = _bowl_type(b, meta)
        is_genuine = _is_genuine_bowler(b, meta)
        fd         = recent_form_data.get(b, {})
        bfs        = fd.get("form_score",  50.0)
        r_eco      = fd.get("recent_eco",   0.0)

        pp_fit         = _phase_fitness(b, "PP",        stats, meta, weather, opp, bfs, r_eco)
        early_mid_fit  = _phase_fitness(b, "Early-Mid", stats, meta, weather, opp, bfs, r_eco)
        late_mid_fit   = _phase_fitness(b, "Late-Mid",  stats, meta, weather, opp, bfs, r_eco)
        death_fit      = _phase_fitness(b, "Death",     stats, meta, weather, opp, bfs, r_eco)

        if is_genuine:
            pp_bowlers.append((b, pp_fit))
            early_mid_bowlers.append((b, early_mid_fit))
            late_mid_bowlers.append((b, late_mid_fit))
            if not (weather.severe_dew and bt == "spin"):
                death_bowlers.append((b, death_fit))
            # Pre-Death pool: ranked by late-mid fitness (similar demands)
            predeath_bowlers.append((b, late_mid_fit))
        else:
            # Part-timers only in mid phases (safest window)
            early_mid_bowlers.append((b, early_mid_fit * 0.6))
            late_mid_bowlers.append((b, late_mid_fit * 0.6))

    pp_bowlers.sort(        key=lambda x: x[1], reverse=True)
    early_mid_bowlers.sort( key=lambda x: x[1], reverse=True)
    late_mid_bowlers.sort(  key=lambda x: x[1], reverse=True)
    predeath_bowlers.sort(  key=lambda x: x[1], reverse=True)
    death_bowlers.sort(     key=lambda x: x[1], reverse=True)

    return {
        "PP":         [b for b, _ in pp_bowlers],
        "Early-Mid":  [b for b, _ in early_mid_bowlers],
        "Late-Mid":   [b for b, _ in late_mid_bowlers],
        "Mid":        [b for b, _ in early_mid_bowlers],   # backward-compat alias
        "Pre-Death":  [b for b, _ in predeath_bowlers],
        "Death":      [b for b, _ in death_bowlers],
    }


# ---------------------------------------------------------------------------
# CAP VALIDATION
# ---------------------------------------------------------------------------

def _validate_four_over_cap(
    over_assignments: list[OverAssignment],
    meta: dict[str, dict],
    overs_already_used: Optional[dict] = None,
) -> None:
    """
    Validate that the bowling plan respects PSL rules:
      1. No bowler appears in more than 4 over slots.
      2. Genuine Bowlers (primary_role == 'Bowler') cover at least 16 of 20 overs.

    Raises PlanValidationError describing the first violation found.
    """
    counts: dict[str, int] = {}
    for oa in over_assignments:
        if oa.primary_bowler and oa.primary_bowler not in ("TBD", "COMPLETED"):
            counts[oa.primary_bowler] = counts.get(oa.primary_bowler, 0) + 1

    # Rule 1 — 4-over cap
    for bowler, n in counts.items():
        if n > MAX_OVERS_PER_BOWLER:
            raise PlanValidationError(
                f"{bowler} assigned {n} overs — exceeds PSL cap of {MAX_OVERS_PER_BOWLER}."
            )

    # Rule 2 — genuine Bowlers must cover ≥ 80% of total match overs
    # Include already-bowled overs (from re-optimisation context) so partial plans
    # aren't penalised for genuines who already bowled in completed overs.
    prior = overs_already_used or {}
    prior_total   = sum(prior.values())
    prior_genuine = sum(
        v for b, v in prior.items()
        if meta.get(b, {}).get("primary_role", "") == "Bowler"
    )
    total_match   = prior_total + sum(counts.values())
    genuine_overs = prior_genuine + sum(
        n for b, n in counts.items()
        if meta.get(b, {}).get("primary_role", "") == "Bowler"
    )
    min_genuine = max(1, round(total_match * 0.80))
    if genuine_overs < min_genuine:
        raise PlanValidationError(
            f"Genuine Bowlers cover only {genuine_overs}/{total_match} overs — minimum is {min_genuine}. "
            f"Squad may not have enough specialist bowlers."
        )


# ---------------------------------------------------------------------------
# PLAN BUILDER
# ---------------------------------------------------------------------------

def generate_bowling_plan(
    our_bowlers:              list[str],
    weather:                  WeatherImpact,
    venue:                    str = "",
    opposition_team:          str = "",
    stats_path:               Optional[Path] = None,
    player_index_path:        Optional[Path] = None,
    opposition_batting_order: Optional[list] = None,
    overs_already_used:       Optional[dict] = None,
    _start_from_over:         int = 1,
    opposition_overrides:     Optional[dict] = None,
) -> BowlingPlan:
    """
    Generate a 20-over bowling plan template.

    Args:
        our_bowlers:     List of bowlers in our squad (only these will be assigned)
        weather:         WeatherImpact object (modifies spin/pace allocation)
        venue:           Match venue (informational, used in notes)
        opposition_team: Opposition name (informational)
        stats_path:      Override for player_stats.parquet
        player_index_path: Override for player_index.csv

    Returns:
        BowlingPlan with 20 OverAssignments + summary + key decisions + contingencies.

    PSL rules enforced:
        - Max 4 overs per bowler
        - Need at least 5 bowlers to cover 20 overs
    """
    sp = Path(stats_path) if stats_path else STATS_PATH
    pi = Path(player_index_path) if player_index_path else (
        PLAYER_INDEX if PLAYER_INDEX.exists() else PLAYER_INDEX_FALLBACK
    )

    meta  = _load_player_meta(pi)
    stats = _load_bowler_phase_stats(our_bowlers, sp, pi_path=pi)
    opp   = _load_opposition_profile(opposition_team, overrides=opposition_overrides)

    # Load matchup matrix for danger-batter intelligence
    _matchup_df: Optional[pd.DataFrame] = None
    if MATCHUP_PATH.exists():
        try:
            _matchup_df = pd.read_parquet(MATCHUP_PATH)
        except Exception as _e:
            import warnings as _w
            _w.warn(f"Could not load matchup_matrix.parquet: {_e}", UserWarning, stacklevel=2)
    else:
        import warnings as _w
        _w.warn(
            f"matchup_matrix.parquet not found at {MATCHUP_PATH}. "
            "Matchup key decisions will be skipped.",
            UserWarning,
            stacklevel=2,
        )

    # Identify genuine bowlers vs part-timers
    genuine = [b for b in our_bowlers if _is_genuine_bowler(b, meta)]
    parttimers = [b for b in our_bowlers if not _is_genuine_bowler(b, meta)]

    if len(genuine) < 5:
        # Fall back: take highest-overs part-timers to reach 5
        parttimers_sorted = sorted(
            parttimers,
            key=lambda b: stats.get(b, {}).get("mid_overs", 0),
            reverse=True,
        )
        needed = 5 - len(genuine)
        genuine = genuine + parttimers_sorted[:needed]
        parttimers = [b for b in parttimers if b not in genuine]

    # Phase-specialist classification (opposition + venue + recent-form aware)
    phase_pools = _classify_bowlers(genuine + parttimers, stats, meta, weather, opp, venue)

    pp_pool           = phase_pools["PP"]
    early_mid_pool    = phase_pools["Early-Mid"]
    late_mid_pool     = phase_pools["Late-Mid"]
    predeath_pool     = phase_pools["Pre-Death"]
    death_pool        = phase_pools["Death"]

    # ------------------------------------------------------------------
    # PARTNERSHIP-BASED ALLOCATION
    # ------------------------------------------------------------------
    # Real T20 structure: two bowlers form a pair (one from each end),
    # alternating overs. A partnership continues until:
    #   - a bowler exhausts their 4-over quota, OR
    #   - a phase boundary / planned change-up is reached
    # Only then is a new bowler brought in to replace one end.
    #
    # Phase blocks:
    #   PP        overs 0-5   (6 overs): opening pair, 3 overs each
    #   Mid-1     overs 6-10  (5 overs): first middle pair (e.g. spinners)
    #   Mid-2     overs 11-13 (3 overs): second middle pair (change-up / returnee)
    #   Pre-Death overs 14-15 (2 overs): secondary bowlers — save death specialists
    #   Death     overs 16-19 (4 overs): death pair, 2+2 overs
    # ------------------------------------------------------------------

    overs_used:  dict[str, int] = {
        b: (overs_already_used or {}).get(b, 0) for b in our_bowlers
    }
    # death_reserved[b] = number of overs to hold back from non-death phases
    death_reserved: dict[str, int] = {}
    assignments: dict[int, str] = {}
    backup:      dict[int, str] = {}

    # ------------------------------------------------------------------
    # PRE-PLAN: reserve death overs for top-2 death specialists
    # Top death bowlers reserve 2 overs each (4 of the 4 death overs 17-20).
    # Pre-Death overs (15-16) are covered by secondary bowlers so the best
    # death specialist only enters at over 17 at the earliest.
    # If only one death specialist is available they bowl overs 19-20, not 15-16.
    # ------------------------------------------------------------------
    top_death = death_pool[:2]
    for b in top_death:
        death_reserved[b] = 2

    def _available_now(b: str, phase: str) -> bool:
        """Check if b can bowl in this phase without breaking death reservation."""
        used  = overs_used.get(b, 0)
        quota = MAX_OVERS_PER_BOWLER
        # Reserve death overs for both Death and Pre-Death phases:
        # top death specialists should not be consumed before over 17.
        if phase not in ("Death",):
            quota = quota - death_reserved.get(b, 0)
        return used < quota

    def _best_for_phase(pool: list[str], phase: str, exclude: str = "") -> Optional[str]:
        """Pick the best available bowler for this phase, ranked by fitness score."""
        cands = [b for b in pool if _available_now(b, phase) and b != exclude]
        if not cands:
            # Relax death reservation if no one else available
            cands = [b for b in pool
                     if overs_used.get(b, 0) < MAX_OVERS_PER_BOWLER and b != exclude]
        if not cands:
            cands = [b for b in genuine
                     if overs_used.get(b, 0) < MAX_OVERS_PER_BOWLER and b != exclude]
        if not cands:
            return None
        # Genuine bowlers preferred; within each group rank by phase fitness (descending)
        cands.sort(key=lambda b: (
            0 if _is_genuine_bowler(b, meta) else 1,
            -_phase_fitness(b, phase, stats, meta, weather, opp),
        ))
        return cands[0]

    def _assign_pair(over_indices: list[int], phase: str, pool: list[str]) -> None:
        """
        Assign a bowling partnership to a block of overs.
        Two bowlers (end_a, end_b) alternate overs within the block.
        If one is exhausted (or hits their non-death quota), swap in the next best.
        """
        end_a = _best_for_phase(pool, phase)
        if end_a is None:
            end_a = our_bowlers[0] if our_bowlers else "TBD"
        end_b = _best_for_phase(pool, phase, exclude=end_a)
        if end_b is None:
            # Expand beyond phase pool — try any bowler with overs remaining
            _all_remaining = [
                b for b in our_bowlers
                if b != end_a and overs_used.get(b, 0) < MAX_OVERS_PER_BOWLER
            ]
            end_b = _all_remaining[0] if _all_remaining else end_a

        for idx, over in enumerate(over_indices):
            bowling_a = (idx % 2 == 0)
            primary   = end_a if bowling_a else end_b

            # Swap if this bowler has hit their phase quota
            if not _available_now(primary, phase):
                other       = end_b if bowling_a else end_a
                replacement = _best_for_phase(pool, phase, exclude=other)
                if replacement is None:
                    replacement = other
                if bowling_a:
                    end_a = replacement
                else:
                    end_b = replacement
                primary = replacement

            overs_used[primary] = overs_used.get(primary, 0) + 1
            assignments[over]   = primary

            partner      = end_b if bowling_a else end_a
            backup[over] = partner if partner != primary else (
                _best_for_phase(genuine, phase, exclude=primary) or primary
            )

    # When re-optimising mid-match, skip overs already bowled so their quota
    # is not double-counted (pre-populated via overs_already_used).
    # _start_from_over is 1-indexed; convert to 0-indexed for range comparisons.
    _si = max(0, _start_from_over - 1)

    def _remaining(indices: list[int]) -> list[int]:
        return [i for i in indices if i >= _si]

    # PP: one opening pair (best 2 PP fitness), 3 overs each
    if _remaining(list(range(0, 6))):
        _assign_pair(_remaining(list(range(0, 6))), "PP", pp_pool)

    # Early-Mid (overs 7-10): spinners dominate, economy + dots priority
    if _remaining(list(range(6, 10))):
        _assign_pair(_remaining(list(range(6, 10))), "Early-Mid", early_mid_pool)

    # Late-Mid (overs 11-14): set batters accelerating, wickets priority
    # Death specialists' reserved overs prevent them appearing here.
    # Prefer bowlers not yet used heavily — fresh change-up options.
    remaining_late_mid = [b for b in late_mid_pool + pp_pool if _available_now(b, "Late-Mid")]
    if _remaining(list(range(10, 14))):
        _assign_pair(
            _remaining(list(range(10, 14))),
            "Late-Mid", remaining_late_mid if remaining_late_mid else late_mid_pool,
        )

    # Pre-Death (overs 15-16): secondary bowlers — reserve top death specialists for 17-20
    # Top-2 death specialists are excluded via death_reserved; use predeath_pool which
    # ranks by mid fitness (medium-pace / secondary spinners preferred here).
    predeath_available = [b for b in predeath_pool if _available_now(b, "Pre-Death")
                          and b not in top_death]
    if not predeath_available:
        # Secondary pool exhausted — widen to all bowlers except top death specialists.
        # Death specialists are excluded even in this fallback to preserve their full
        # 4-over quota for overs 17-20.
        predeath_available = [
            b for b in our_bowlers
            if _available_now(b, "Pre-Death") and b not in top_death
        ]
    if not predeath_available:
        # Truly nothing left — last resort: use predeath_pool including death specialists.
        # This only fires if the squad has fewer than 5 genuine bowlers.
        predeath_available = predeath_pool
    if _remaining(list(range(14, 16))):
        _assign_pair(_remaining(list(range(14, 16))), "Pre-Death", predeath_available)

    # Death (overs 17-20): top death specialists, 2+2 overs
    if _remaining(list(range(16, 20))):
        _assign_pair(_remaining(list(range(16, 20))), "Death", death_pool)

    # ------------------------------------------------------------------
    # Post-processing fix 1: eliminate back-to-back overs by the same bowler
    # (happens at phase boundaries when the same bowler tops two pools)
    # ------------------------------------------------------------------
    _over_seq = sorted(assignments.keys())
    for _i in range(len(_over_seq) - 1):
        _o1, _o2 = _over_seq[_i], _over_seq[_i + 1]
        if _o2 != _o1 + 1:          # non-adjacent — skip
            continue
        _dup = assignments.get(_o1)
        if not _dup or _dup != assignments.get(_o2):
            continue
        # Same bowler in consecutive overs — find the nearest later non-duplicate to swap
        for _j in range(_i + 2, len(_over_seq)):
            _o3 = _over_seq[_j]
            _other = assignments.get(_o3)
            if not _other or _other in ("TBD", "COMPLETED") or _other == _dup:
                continue
            # Ensure the swap doesn't violate 4-over cap for either bowler
            _dup_overs   = sum(1 for b in assignments.values() if b == _dup)
            _other_overs = sum(1 for b in assignments.values() if b == _other)
            if _dup_overs > MAX_OVERS_PER_BOWLER or _other_overs > MAX_OVERS_PER_BOWLER:
                continue
            # Ensure _o3-1 and _o3+1 won't become consecutive with _dup after swap
            _o3_prev = _over_seq[_j - 1] if _j > 0 else -99
            _o3_next = _over_seq[_j + 1] if _j + 1 < len(_over_seq) else -99
            if assignments.get(_o3_prev) == _dup or assignments.get(_o3_next) == _dup:
                continue
            # Safe to swap _o2 and _o3
            assignments[_o2], assignments[_o3] = assignments[_o3], assignments[_o2]
            backup[_o2], backup[_o3] = backup.get(_o3, _dup), backup.get(_o2, _other)
            break

    # ------------------------------------------------------------------
    # Post-processing fix 2: guarantee genuine bowlers ≥ 2 overs
    # A 1-over allocation wastes a bowling slot and signals poor planning
    # ------------------------------------------------------------------
    def _alloc_count(b: str) -> int:
        return sum(1 for v in assignments.values() if v == b)

    _genuine_in_plan = [b for b in genuine if _alloc_count(b) > 0]
    _one_over = [b for b in _genuine_in_plan if _alloc_count(b) == 1]
    _four_over = [b for b in _genuine_in_plan if _alloc_count(b) == 4]

    for _needy in _one_over:
        if not _four_over:
            break
        _donor = _four_over.pop(0)
        # Find the donor's latest non-PP, non-Death over to reassign
        _donor_overs = sorted(
            [o for o, b in assignments.items()
             if b == _donor and PHASE_LABEL.get(o, "") not in ("Death", "PP")],
            reverse=True,
        )
        if not _donor_overs:
            continue
        _swap_over = _donor_overs[0]
        # Verify the swap doesn't create new consecutive pair
        _prev = _swap_over - 1
        _next = _swap_over + 1
        if assignments.get(_prev) == _needy or assignments.get(_next) == _needy:
            # Try second candidate
            if len(_donor_overs) > 1:
                _swap_over = _donor_overs[1]
                _prev, _next = _swap_over - 1, _swap_over + 1
                if assignments.get(_prev) == _needy or assignments.get(_next) == _needy:
                    continue
            else:
                continue
        assignments[_swap_over] = _needy
        overs_used[_needy]  = overs_used.get(_needy, 0) + 1
        overs_used[_donor]  = overs_used.get(_donor, 0) - 1
        if overs_used[_donor] == 3:
            _four_over_remaining = [b for b in _genuine_in_plan if _alloc_count(b) == 4]
            _four_over = _four_over_remaining

    # ------------------------------------------------------------------
    # Build OverAssignment list
    # ------------------------------------------------------------------
    over_assignments: list[OverAssignment] = []
    prev_primary: Optional[str] = None

    for over in range(TOTAL_OVERS):
        # Overs before _start_from_over were not allocated — mark as COMPLETED so
        # cap validation and bowler_summary don't count them against any bowler.
        _fallback = "COMPLETED" if over < _si else (our_bowlers[0] if our_bowlers else "TBD")
        primary = assignments.get(over, _fallback)
        bkp     = backup.get(over, primary)
        phase   = PHASE_LABEL[over]
        bt      = _bowl_type(primary, meta)
        s       = stats.get(primary, {})
        partner = bkp

        # Reason — distinguish new spell vs continuing spell; include role justification
        new_spell = (primary != prev_primary)
        _dot_pct  = s.get("dot_pct", LEAGUE_AVG_DOT_PCT)
        _wkt_rate = s.get("bowl_avg", None)   # lower = better wicket-taker
        _role_tag = ""
        if _dot_pct >= 40:
            _role_tag = f" Dot-ball threat ({_dot_pct:.0f}%)."
        elif _wkt_rate is not None and _wkt_rate < 22:
            _role_tag = f" Wicket-taker (avg {_wkt_rate:.1f})."

        if phase == "PP":
            econ   = s.get("pp_economy", 8.5)
            reason = (
                f"New ball — opens with {partner} from other end. PP economy {econ:.1f}.{_role_tag}"
                if new_spell
                else f"Continuing PP spell — economy {econ:.1f}.{_role_tag}"
            )
        elif phase == "Death":
            econ   = s.get("death_economy", 9.5)
            reason = (
                f"Death partnership with {partner} — economy {econ:.1f}.{_role_tag} Selected for death execution."
                if new_spell
                else f"Continuing death spell — economy {econ:.1f}."
            )
        elif phase == "Pre-Death":
            econ   = s.get("mid_economy", 7.5)
            reason = (
                f"Pre-death spell (overs 15-16) — bridges middle to death. "
                f"Preserves top specialists for 17-20. Economy {econ:.1f}.{_role_tag}"
                if new_spell
                else f"Continuing pre-death spell — economy {econ:.1f}."
            )
        elif phase == "Early-Mid":
            econ   = s.get("mid_economy", 7.5)
            reason = (
                f"Early-middle spell (overs 7-10) — pair with {partner}. Economy {econ:.1f}.{_role_tag}"
                if new_spell
                else f"Continuing early-middle spell — economy {econ:.1f}."
            )
        elif phase == "Late-Mid":
            econ   = s.get("mid_economy", 7.5)
            reason = (
                f"Late-middle spell (overs 11-14) — pair with {partner}. Economy {econ:.1f}.{_role_tag}"
                if new_spell
                else f"Continuing late-middle spell — economy {econ:.1f}."
            )
        else:
            econ   = s.get("mid_economy", 7.5)
            reason = (
                f"Middle spell — pair with {partner}. Economy {econ:.1f}.{_role_tag}"
                if new_spell
                else f"Continuing middle spell — economy {econ:.1f}."
            )

        # Weather note — use gradient intensity for nuanced dew warnings
        w_note = ""
        if phase in ("Death", "Pre-Death", "Late-Mid") and bt == "spin":
            _dew_intensity = weather.dew_probability_at(over + 1)   # over is 0-indexed
            if _dew_intensity >= 0.75:
                w_note = (
                    f"Dew warning — spinner risky here "
                    f"(onset over {weather.dew_onset_over}, {round(_dew_intensity*100)}% intensity). Consider pace."
                )
            elif _dew_intensity >= 0.25:
                w_note = (
                    f"Dew building ({round(_dew_intensity*100)}% intensity from over {weather.dew_onset_over}) "
                    f"— monitor grip, 1-2 more overs before switching to pace."
                )
        elif phase == "PP" and bt == "pace" and weather.swing_bonus >= 1.20:
            w_note = f"Swing conditions ({weather.swing_bonus:.2f}x) — ideal for pace."

        over_assignments.append(OverAssignment(
            over           = over + 1,
            primary_bowler = primary,
            backup_bowler  = bkp,
            phase          = phase,
            reason         = reason,
            weather_note   = w_note,
        ))
        prev_primary = primary

    # Bowler summary (exclude COMPLETED/TBD placeholders from skipped overs)
    bowler_summary: dict[str, list[int]] = {}
    for over, b in assignments.items():
        if b and b not in ("TBD", "COMPLETED"):
            bowler_summary.setdefault(b, []).append(over + 1)

    # Key decisions — weather + fitness aware
    key_decisions: list[str] = []

    # New ball decision — based on PP fitness + weather
    if pp_pool:
        top_pp = pp_pool[0]
        bt     = _bowl_type(top_pp, meta)
        pp_fit = _phase_fitness(top_pp, "PP", stats, meta, weather)
        if bt == "pace" and weather.swing_bonus >= 1.20:
            key_decisions.append(
                f"Lead with {top_pp} — swing conditions ({weather.swing_bonus:.2f}x bonus) "
                f"suit pace. PP fitness score: {pp_fit:.0f}/100."
            )
        elif bt == "pace" and weather.pace_bounce_bonus >= 1.10:
            key_decisions.append(
                f"Lead with {top_pp} — hard/bouncy conditions favour pace new ball. "
                f"PP fitness: {pp_fit:.0f}/100."
            )
        else:
            s = stats.get(top_pp, {})
            key_decisions.append(
                f"Give {top_pp} the new ball — highest PP fitness ({pp_fit:.0f}/100). "
                f"PP: {s.get('pp_economy', 0):.1f} eco, "
                f"{s.get('pp_wkts_po', 0):.2f} wkts/over."
            )

    # Death pair — explain selection and pre-death separation
    if death_pool:
        d1 = death_pool[0]
        d2 = death_pool[1] if len(death_pool) > 1 else None
        s1 = stats.get(d1, {})
        fit1 = _phase_fitness(d1, "Death", stats, meta, weather)
        # Identify who covers pre-death overs 15-16
        predeath_covers = [assignments.get(o) for o in (14, 15) if assignments.get(o)]
        predeath_names  = list(dict.fromkeys(b for b in predeath_covers if b not in top_death))
        if predeath_names:
            key_decisions.append(
                f"Saving {d1} for overs 17-20 — pre-death overs 15-16 covered by "
                f"{predeath_names[0]}."
            )
        msg = (
            f"Death pair: {d1} (death fitness {fit1:.0f}/100 — "
            f"{s1.get('death_wkts_po', 0):.2f} wkts/over, "
            f"{s1.get('death_economy', 0):.1f} eco)"
        )
        if d2:
            s2   = stats.get(d2, {})
            fit2 = _phase_fitness(d2, "Death", stats, meta, weather)
            msg += (
                f" + {d2} ({fit2:.0f}/100 — "
                f"{s2.get('death_wkts_po', 0):.2f} wkts/over)."
            )
        key_decisions.append(msg)

    # Spinner allocation — weather driven (gradient-aware)
    spin_bowlers = [b for b in our_bowlers if _bowl_type(b, meta) == "spin"]
    if spin_bowlers:
        if weather.severe_dew:
            # Full-intensity dew at over 18 — hard constraint
            full_over = (weather.dew_onset_over or 0) + weather.DEW_GRADIENT_OVERS
            key_decisions.append(
                f"SEVERE DEW — {', '.join(spin_bowlers)} restricted to overs 7-14 only. "
                f"Dew reaches full intensity around over {full_over} — no spinners after over {weather.dew_onset_over}."
            )
        elif weather.spinner_penalty <= 0.75:
            full_over = (weather.dew_onset_over or 0) + weather.DEW_GRADIENT_OVERS
            key_decisions.append(
                f"Moderate dew (penalty {weather.spinner_penalty:.2f}) — keep "
                f"{', '.join(spin_bowlers)} in middle overs (7-14), avoid death. "
                f"Dew builds gradually from over {weather.dew_onset_over}, full effect ~over {full_over}."
            )
        elif weather.swing_bonus <= 1.05:
            key_decisions.append(
                f"Dry/slow conditions — spinners ({', '.join(spin_bowlers)}) "
                f"get a boost. Front-load them in mid-overs."
            )
        else:
            key_decisions.append(
                f"Normal conditions — {', '.join(spin_bowlers)} available throughout."
            )

    # Pace conditions note
    if weather.swing_bonus >= 1.15 or weather.pace_bounce_bonus >= 1.10:
        pace_bowlers = [b for b in our_bowlers if _bowl_type(b, meta) == "pace"
                        and _is_genuine_bowler(b, meta)]
        if pace_bowlers:
            key_decisions.append(
                f"Pace-friendly conditions (swing {weather.swing_bonus:.2f}x, "
                f"bounce {weather.pace_bounce_bonus:.2f}x) — maximise "
                f"{', '.join(pace_bowlers[:2])} overs early."
            )

    # Opposition-specific key decisions (inserted before the cap)
    if opposition_team and opp:
        opp_vs_spin = opp.get("vs_spin_economy",    8.0)
        opp_vs_pace = opp.get("vs_pace_economy",    8.0)
        opp_lh_pct  = opp.get("left_hand_top6_pct", 33.0)
        opp_pp_sr   = opp.get("powerplay_sr",       130.0)
        opp_death_sr= opp.get("death_sr",           155.0)
        estimated   = opp.get("is_estimated",        False)
        tag         = " [NO DATA — default assumption]" if estimated else ""

        if estimated:
            key_decisions.insert(0,
                f"[WARNING] ESTIMATED PROFILE — {opposition_team} has no PSL career data. "
                f"All opposition intelligence below is seeded from league defaults. "
                f"Treat as orientation only."
            )

        spin_bowlers_avail = [b for b in our_bowlers if _bowl_type(b, meta) == "spin"]
        pace_bowlers_avail = [b for b in our_bowlers if _bowl_type(b, meta) == "pace"
                              and _is_genuine_bowler(b, meta)]
        left_arm_avail     = [b for b in our_bowlers
                              if "left" in meta.get(b, {}).get("bowling_style", "").lower()]

        # Spinner matchup intelligence — per spin subtype where data available
        for spinner in spin_bowlers_avail[:2]:
            sp_style = meta.get(spinner, {}).get("bowling_style", "").lower()
            if "leg" in sp_style or "googly" in sp_style or "chinaman" in sp_style:
                sub_eco = opp.get("vs_legspin_economy", opp_vs_spin)
                sub_label = "leg-spin"
            elif "left" in sp_style and any(w in sp_style for w in ("orthodox", "spin", "slow")):
                sub_eco = opp.get("vs_leftarm_spin_economy", opp_vs_spin)
                sub_label = "left-arm spin"
            else:
                sub_eco = opp.get("vs_offspin_economy", opp_vs_spin)
                sub_label = "off-spin"

            if sub_eco < 7.0:
                key_decisions.append(
                    f"vs {opposition_team}{tag}: They score freely vs {sub_label} "
                    f"(eco {sub_eco:.1f}). {spinner.split()[-1]}: tight lines only, no freebies."
                )
            elif sub_eco > 9.5:
                key_decisions.append(
                    f"vs {opposition_team}{tag}: They struggle vs {sub_label} "
                    f"(eco {sub_eco:.1f}). Front-load {spinner.split()[-1]} in mid-overs."
                )

        # Pace matchup intelligence
        if opp_vs_pace < 7.5 and pace_bowlers_avail:
            key_decisions.append(
                f"vs {opposition_team}{tag}: They play pace well (eco {opp_vs_pace:.1f}). "
                f"Mix up lengths and vary pace — avoid predictable lines for "
                f"{', '.join(pace_bowlers_avail[:2])}."
            )
        elif opp_vs_pace > 9.5 and pace_bowlers_avail:
            key_decisions.append(
                f"vs {opposition_team}{tag}: They struggle vs pace (eco {opp_vs_pace:.1f}). "
                f"Load up {', '.join(pace_bowlers_avail[:2])} — aggressive pace plans pay off."
            )

        # Left-arm angle
        if opp_lh_pct >= 50 and left_arm_avail:
            key_decisions.append(
                f"vs {opposition_team}{tag}: {opp_lh_pct:.0f}% left-handed top-6 — "
                f"left-arm angle is a weapon. Maximise {', '.join(left_arm_avail[:2])} overs."
            )

        # PP aggressiveness
        if opp_pp_sr > 155:
            key_decisions.append(
                f"vs {opposition_team}{tag}: Explosive PP openers (SR {opp_pp_sr:.0f}). "
                f"Must take early wickets — attack plan in overs 1-3."
            )
        elif opp_pp_sr < 115:
            key_decisions.append(
                f"vs {opposition_team}{tag}: Cautious PP batting (SR {opp_pp_sr:.0f}). "
                f"Apply dot-ball pressure — build asking-rate squeeze from the start."
            )

        # Death aggressiveness
        if opp_death_sr > 175:
            key_decisions.append(
                f"vs {opposition_team}{tag}: Dangerous finishers (death SR {opp_death_sr:.0f}). "
                f"Use your best wicket-takers at death — can't afford to concede boundaries."
            )

    # Reserve the last 3 slots for [MATCHUP] notes — trim prior decisions now
    # so high-value matchup intelligence is never crowded out.
    key_decisions = key_decisions[:7]

    # [MATCHUP] key decisions — danger batters vs our bowlers
    # bowler_adv in the parquet uses formula: (dismissal_pct/100) - (sr/150), range -4 to +1.
    # Thresholds of 15/-15 are calibrated for the live formula:
    #   (PRIOR_SR - sr)*0.6 + (dismissal_pct - PRIOR_DIS)*3.0
    # We compute that equivalent here so the thresholds fire correctly.
    _PRIOR_SR  = 130.0
    _PRIOR_DIS =   7.5
    _MIN_BALLS =   8    # minimum meaningful sample

    if opposition_batting_order and _matchup_df is not None:
        _matchup_notes: list[tuple[float, str]] = []  # (abs_adv, note)
        try:
            # Build lookup: (batter, bowler) -> (balls, sr, dismissal_pct)
            _needed = ["batter", "bowler", "balls", "sr", "dismissal_pct"]
            _mx_raw = (
                _matchup_df[_needed]
                .set_index(["batter", "bowler"])
                .to_dict(orient="index")
            )
            for _pb in opposition_batting_order[:7]:  # top-7 batting positions, High danger only
                _danger = getattr(_pb, "danger_rating", None) or (
                    _pb.get("danger_rating") if isinstance(_pb, dict) else None
                )
                if _danger != "High":
                    continue
                _bname = getattr(_pb, "player_name", None) or (
                    _pb.get("player_name") if isinstance(_pb, dict) else None
                )
                if not _bname:
                    continue
                _pos = getattr(_pb, "position", None) or (
                    _pb.get("position") if isinstance(_pb, dict) else "?"
                )
                for _bowler in our_bowlers:
                    _row = _mx_raw.get((_bname, _bowler))
                    if _row is None or int(_row["balls"]) < _MIN_BALLS:
                        continue
                    # Compute live-formula equivalent for correct scale
                    _adv = (
                        (_PRIOR_SR  - float(_row["sr"])) * 0.6
                        + (float(_row["dismissal_pct"]) - _PRIOR_DIS) * 3.0
                    )
                    if _adv >= 15.0:
                        _matchup_notes.append((
                            abs(_adv),
                            f"[MATCHUP] {_bowler.split()[-1]} vs {_bname} (pos {_pos}) — "
                            f"bowler edge (adv +{_adv:.0f}). Bowl him at {_bname} early."
                        ))
                    elif _adv <= -15.0:
                        _matchup_notes.append((
                            abs(_adv),
                            f"[MATCHUP] DANGER — {_bname} (pos {_pos}) dominates "
                            f"{_bowler.split()[-1]} (adv {_adv:.0f}). Avoid this matchup or "
                            f"set boundary protection."
                        ))
        except Exception as _me:
            import warnings as _w2
            _w2.warn(f"Matchup key-decision build failed: {_me}", UserWarning, stacklevel=2)

        _matchup_notes.sort(key=lambda x: x[0], reverse=True)
        for _, _note in _matchup_notes[:3]:
            key_decisions.append(_note)

    # Contingencies
    contingencies: list[str] = []
    if pp_pool and len(pp_pool) >= 2:
        contingencies.append(
            f"If {pp_pool[0]} goes for 12+ in over 1, bring {pp_pool[1]} for over 3 instead."
        )
    if death_pool and len(death_pool) >= 2:
        contingencies.append(
            f"If {death_pool[0]} concedes 14+ in any over before over 17, "
            f"switch to {death_pool[1]} for the remaining death overs. "
            f"(14+ in death = economy 14.0+; unacceptable even in slog overs.)"
        )
    if weather.dl_planning_needed:
        contingencies.append(
            "Rain risk — front-load your strike bowlers in overs 1-10. D/L favours the side with early wickets."
        )

    plan_warnings: list[str] = []

    # Opposition profile freshness warnings — surface to captain before match
    if opp:
        _freshness = opp.get("profile_freshness", "career-only")
        _staleness = opp.get("staleness_note", "")
        if _staleness:
            plan_warnings.append(_staleness)
        if _freshness == "current-season":
            key_decisions.insert(0,
                f"[Opposition] {opposition_team} profile blended: "
                f"{round(OPP_CURRENT_SEASON_WEIGHT*100):.0f}% PSL {CURRENT_PSL_SEASON} "
                f"+ {round((1-OPP_CURRENT_SEASON_WEIGHT)*100):.0f}% career aggregate."
            )
        _injury_notes = opp.get("injury_notes", "")
        _injured_out  = opp.get("injured_out",  [])
        _form_note    = opp.get("form_note",    "")
        if _injured_out:
            key_decisions.insert(0,
                f"[INJURY] {', '.join(_injured_out)} unavailable for {opposition_team}. "
                + (_injury_notes if _injury_notes else "Batting order may shift.")
            )
        elif _injury_notes:
            key_decisions.insert(0, f"[Squad note] {_injury_notes}")
        if _form_note:
            key_decisions.insert(0 if not _injured_out else 1,
                f"[Form note] {_form_note}"
            )

    def _rebuild_assignments() -> None:
        """Sync bowler_summary from the current assignments dict and rebuild over_assignments."""
        nonlocal over_assignments
        # Rebuild summary from assignments
        bowler_summary.clear()
        for idx, bname in assignments.items():
            if bname and bname not in ("TBD", "COMPLETED"):
                bowler_summary.setdefault(bname, []).append(idx + 1)
        for v in bowler_summary.values():
            v.sort()
        over_assignments = [
            OverAssignment(
                over           = oa.over,
                primary_bowler = assignments.get(oa.over - 1, oa.primary_bowler),
                backup_bowler  = oa.backup_bowler,
                phase          = oa.phase,
                reason         = oa.reason,
                weather_note   = oa.weather_note,
            )
            for oa in over_assignments
        ]

    def _rebalance_cap() -> None:
        """
        One-pass rebalancing: for every bowler exceeding 4 overs, move their
        excess over slots to the next-best eligible bowler (by _phase_fitness)
        who is still under cap. Modifies assignments and bowler_summary in place.
        """
        import warnings as _rw
        _rb_form = _load_recent_form(our_bowlers, venue)
        for bowler, alloc in list(bowler_summary.items()):
            excess = alloc[MAX_OVERS_PER_BOWLER:]
            if not excess:
                continue
            _rw.warn(
                f"[bowling_plan] {bowler} allocated {len(alloc)} overs — "
                f"exceeds PSL cap. Attempting rebalance.",
                UserWarning,
                stacklevel=4,
            )
            for _ov in excess:
                phase_label = PHASE_LABEL[_ov - 1]
                # Find best eligible replacement under cap
                pool = phase_pools.get(phase_label, phase_pools.get("Mid", []))
                best_sub: str | None = None
                best_fit: float = -1.0
                for candidate in pool:
                    if candidate == bowler:
                        continue
                    current_count = sum(1 for a in assignments.values() if a == candidate)
                    if current_count >= MAX_OVERS_PER_BOWLER:
                        continue
                    cfd   = _rb_form.get(candidate, {})
                    bfs   = cfd.get("form_score", 50.0)
                    r_eco = cfd.get("recent_eco",  0.0)
                    fit = _phase_fitness(
                        candidate, PHASE_NAME.get(phase_label, "middle"),
                        stats, meta, weather, opp, bfs, r_eco,
                    )
                    if fit > best_fit:
                        best_fit = fit
                        best_sub = candidate
                assignments[_ov - 1] = best_sub if best_sub else "TBD"
            # Cap original bowler's summary immediately
            bowler_summary[bowler] = alloc[:MAX_OVERS_PER_BOWLER]
        _rebuild_assignments()

    # --- First validation attempt ---
    try:
        _validate_four_over_cap(over_assignments, meta, overs_already_used)
    except PlanValidationError as _e:
        # Attempt one rebalancing pass
        _rebalance_cap()
        # Re-validate after rebalancing
        try:
            _validate_four_over_cap(over_assignments, meta, overs_already_used)
        except PlanValidationError as _e2:
            import warnings as _vw
            msg = f"[bowling_plan] Cap violation after rebalance: {_e2}"
            _vw.warn(msg, UserWarning, stacklevel=2)
            plan_warnings.append(str(_e2))

    # --- Form warnings: flag bowlers whose recent economy is significantly worse
    #     than their career average, and highlight improving bowlers. ---
    _form_data = _load_recent_form(our_bowlers, venue)
    for _b in our_bowlers:
        _fd = _form_data.get(_b, {})
        _r_eco   = _fd.get("recent_eco",   0.0)
        _r_overs = _fd.get("recent_overs", 0.0)
        _trend   = _fd.get("trend",        "stable")
        if _r_overs < MIN_RECENT_OVERS_FOR_ECO_BLEND or _r_eco <= 0.0:
            continue
        # Compute career blended economy (PP×4 + Mid×10 + Death×6 overs weighting)
        _s = stats.get(_b, {})
        _career_eco = (
            _s.get("pp_economy",    8.5) * 4
            + _s.get("mid_economy", 7.5) * 10
            + _s.get("death_economy", 9.5) * 6
        ) / 20.0
        if _r_eco > _career_eco * (1.0 + FORM_CONCERN_ECO_DELTA_PCT):
            _delta_pct = round((_r_eco / _career_eco - 1.0) * 100)
            plan_warnings.append(
                f"{_b}: recent economy {_r_eco:.2f} is {_delta_pct}% above career average "
                f"({_career_eco:.2f}) — monitor form."
            )
        elif _trend == "improving" and _r_eco < _career_eco * (1.0 - FORM_CONCERN_ECO_DELTA_PCT):
            key_decisions.append(
                f"{_b} is in improving form (recent eco {_r_eco:.2f} vs career {_career_eco:.2f}) "
                f"— consider using full 4-over allocation."
            )

    return BowlingPlan(
        overs          = over_assignments,
        bowler_summary = bowler_summary,
        key_decisions  = key_decisions,
        contingencies  = contingencies,
        plan_warnings  = plan_warnings,
    )


# ---------------------------------------------------------------------------
# LIVE RE-OPTIMISER
# ---------------------------------------------------------------------------

def reoptimise_bowling_plan(
    state:                    LiveMatchState,
    weather:                  WeatherImpact,
    original_plan:            BowlingPlan,
    our_bowlers:              list[str],
    opposition_team:          str = "",
    stats_path:               Optional[Path] = None,
    player_index_path:        Optional[Path] = None,
    opposition_batting_order: Optional[list] = None,
    opposition_overrides:     Optional[dict] = None,
) -> BowlingPlan:
    """
    Re-optimise the bowling plan for all remaining overs based on current match state.

    Called from the Dugout screen at the start of each over after the analyst
    updates the score, wickets, and bowler who just bowled.

    Completed overs (< state.current_over) are carried over unchanged from
    original_plan.  Remaining overs are freshly allocated, respecting each
    bowler's remaining quota (4 - already bowled this innings).

    Args:
        state:          Current LiveMatchState (overs_bowled_by is the key input).
        weather:        Updated WeatherImpact (may reflect new dew / wind data).
        original_plan:  The BowlingPlan that was generated pre-match.
        our_bowlers:    Full list of our bowlers (same as generate_bowling_plan call).
        opposition_team, stats_path, player_index_path, opposition_batting_order:
                        Passed through to generate_bowling_plan unchanged.

    Returns:
        A new BowlingPlan where completed overs are locked and remaining overs
        reflect the updated quota-constrained allocation.
    """
    current_over = state.current_over  # 1-indexed; this over is still to be bowled

    # Nothing to re-optimise before the first ball
    if current_over <= 1:
        return original_plan

    # Pre-populate overs used from live state so the allocator respects remaining quota
    overs_already_used: dict[str, int] = dict(state.overs_bowled_by)

    # Generate a fresh plan for remaining overs only.
    # _start_from_over ensures completed overs are not allocated (quota not double-counted).
    fresh = generate_bowling_plan(
        our_bowlers              = our_bowlers,
        weather                  = weather,
        venue                    = state.venue,
        opposition_team          = opposition_team,
        stats_path               = stats_path,
        player_index_path        = player_index_path,
        opposition_batting_order = opposition_batting_order,
        overs_already_used       = overs_already_used,
        _start_from_over         = current_over,
        opposition_overrides     = opposition_overrides,
    )

    # Lock completed overs: replace fresh plan's placeholder assignments with
    # the original plan's actual (historical) assignments.
    completed: dict[int, OverAssignment] = {
        oa.over: oa
        for oa in original_plan.overs
        if oa.over < current_over
    }
    # Fresh plan only has assignments for overs >= current_over (due to _start_from_over).
    # Build merged list: historical from original_plan + remaining from fresh plan.
    completed_overs  = [completed[i] for i in sorted(completed)]
    remaining_overs  = [oa for oa in fresh.overs if oa.over >= current_over]
    merged_overs: list[OverAssignment] = completed_overs + remaining_overs

    # bowler_summary shows only REMAINING (future) planned overs — use fresh plan's
    # summary directly since it was built only from the remaining allocation.
    bowler_summary = fresh.bowler_summary

    # Surface remaining quotas as the first key decision
    quota_parts = [
        f"{b} ({4 - overs_already_used.get(b, 0)} left)"
        for b in our_bowlers
        if overs_already_used.get(b, 0) > 0
    ]
    reopt_note = (
        f"[Re-optimised at over {current_over}]"
        + (f" Quotas: {', '.join(quota_parts)}" if quota_parts else "")
    )

    return BowlingPlan(
        overs          = merged_overs,
        bowler_summary = bowler_summary,
        key_decisions  = [reopt_note] + fresh.key_decisions,
        contingencies  = fresh.contingencies,
        plan_warnings  = fresh.plan_warnings,
    )


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.situation import WeatherImpact

    lahore_bowlers = [
        "Shaheen Shah Afridi",
        "Haris Rauf",
        "Usama Mir",
        "Mustafizur Rahman",
        "Sikandar Raza",
        "Fakhar Zaman",   # part-timer
    ]

    # Dew scenario
    weather_dew = WeatherImpact(
        spinner_penalty    = 0.60,
        swing_bonus        = 1.15,
        pace_bounce_bonus  = 1.05,
        yorker_reliability = 0.92,
        dl_planning_needed = False,
        dew_onset_over     = 13,
        warnings           = ["Heavy dew from over 13"],
    )

    # Minimal opposition order stubs for [MATCHUP] self-test
    _test_order = [
        {"player_name": "Azam Khan",    "position": 2, "danger_rating": "High"},
        {"player_name": "David Warner", "position": 1, "danger_rating": "Medium"},
        {"player_name": "Khushdil Shah","position": 6, "danger_rating": "High"},
    ]

    plan = generate_bowling_plan(
        our_bowlers              = lahore_bowlers,
        weather                  = weather_dew,
        venue                    = "Gaddafi Stadium, Lahore",
        opposition_team          = "Karachi Kings",
        opposition_batting_order = _test_order,
    )

    print(f"\nBowling Plan — Lahore vs Karachi (with dew)")
    print(f"{'='*65}")
    print(f"{'Over':<6}  {'Phase':<6}  {'Primary Bowler':<25}  {'Backup':<22}  Note")
    print(f"{'-'*95}")
    for oa in plan.overs:
        note = f"  [{oa.weather_note}]" if oa.weather_note else ""
        print(
            f"  {oa.over:<4}  {oa.phase:<6}  {oa.primary_bowler:<25}  "
            f"{oa.backup_bowler:<22}  {oa.reason[:40]}{note}"
        )

    print(f"\nBowler Allocations:")
    for bowler, overs_list in sorted(plan.bowler_summary.items()):
        print(f"  {bowler:<25}  overs: {overs_list}  ({len(overs_list)} overs)")

    print(f"\nKey Decisions:")
    for kd in plan.key_decisions:
        print(f"  - {kd}")

    print(f"\nContingencies:")
    for c in plan.contingencies:
        print(f"  - {c}")

    if plan.plan_warnings:
        print(f"\nPlan Warnings:")
        for w in plan.plan_warnings:
            print(f"  ! {w}")
    print()
